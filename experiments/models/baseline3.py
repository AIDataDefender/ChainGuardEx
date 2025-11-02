import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch as dglnn
import torch.nn.utils.rnn as rnn_utils

class GraphBranch(nn.Module):
    """
    This module processes the heterogeneous graph.
    It projects all node types to a common 'hidden_dim'.
    Supports AST, CFG (block), CG (function), and DFG (dfg_node) nodes.
    """
    def __init__(self, in_dims, hidden_dim, num_layers=2):
        super(GraphBranch, self).__init__()
        self.in_dims = in_dims
        self.hidden_dim = hidden_dim
        
        # Input projection layers
        # We project all node types (ast, block, function, dfg_node) to hidden_dim
        # so they can be processed by the GNN layers.
        self.project_ast = nn.Linear(in_dims['ast_node'], hidden_dim)
        self.project_block = nn.Linear(in_dims['block'], hidden_dim)
        self.project_func = nn.Linear(in_dims['function'], hidden_dim)
        self.project_dfg = nn.Linear(in_dims['dfg_node'], hidden_dim)  # DFG nodes have features
        
        self.activation = nn.ReLU()
        
        # GNN Layers - include DFG edge types
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            # Define one GNN layer (e.g., GraphConv) for each edge type
            # Original graph edges
            conv_dict = {
                'child_of': dglnn.GraphConv(hidden_dim, hidden_dim),       # AST edges
                'flows': dglnn.GraphConv(hidden_dim, hidden_dim),          # CFG edges
                'calls': dglnn.GraphConv(hidden_dim, hidden_dim),          # CG edges
                'data_flow': dglnn.GraphConv(hidden_dim, hidden_dim),      # DFG edges
            }
            # HeteroGraphConv wraps them all up
            self.layers.append(dglnn.HeteroGraphConv(conv_dict, aggregate='sum'))

    def forward(self, g):
        # 1. Get initial features for all node types
        h = {}
        h['ast_node'] = g.nodes['ast_node'].data['feat']
        h['block'] = g.nodes['block'].data['feat']
        h['function'] = g.nodes['function'].data['feat']
        
        # 2. Project features to hidden_dim
        h['ast_node'] = self.activation(self.project_ast(h['ast_node']))
        h['block'] = self.activation(self.project_block(h['block']))
        h['function'] = self.activation(self.project_func(h['function']))
        
        # 3. Handle 'dfg_node' - DFG nodes now have features
        num_dfg_nodes = g.num_nodes('dfg_node')
        if num_dfg_nodes > 0:
            # Get DFG node features and project them
            h['dfg_node'] = g.nodes['dfg_node'].data['feat']
            h['dfg_node'] = self.activation(self.project_dfg(h['dfg_node']))
            
        # 4. Run through GNN layers
        for layer in self.layers:
            h_new = layer(g, h)
            # Apply activation to all node types
            h = {k: self.activation(v) for k, v in h_new.items()}
            
        # Return the dictionary of final node features
        return h

class CodeBranch(nn.Module):
    """
    This module processes the padded code sequences using an LSTM.
    It correctly handles padding using pack_padded_sequence.
    """
    def __init__(self, input_dim, hidden_dim, num_layers=1, bidirectional=True):
        super(CodeBranch, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True  # Our data is (B, L, F)
        )
        
        # Adjust output dim if bidirectional
        self.out_dim = hidden_dim * 2 if bidirectional else hidden_dim

    def forward(self, padded_code, lengths):
        # 'lengths' must be on CPU for pack_padded_sequence
        lengths_cpu = lengths.cpu()
        
        # 1. Pack the sequence
        packed_input = rnn_utils.pack_padded_sequence(
            padded_code, 
            lengths_cpu, 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # 2. Run through LSTM
        # packed_output shape is (total_unpadded_length, hidden_dim * 2)
        packed_output, (hidden, cell) = self.lstm(packed_input)
        
        # 3. Unpack the sequence
        # output shape is (B, max_len, hidden_dim * 2)
        output, _ = rnn_utils.pad_packed_sequence(
            packed_output, 
            batch_first=True
        )
        
        return output

class CombinedModel(nn.Module):
    """
    The main model. It combines the Graph and Code branches and produces
    three separate outputs for your three label sets.
    It can be configured to run in code-only, graph-only, or combined mode.
    """
    def __init__(self, graph_in_dims, code_in_dim, hidden_dim, out_dim=9, use_graph=True, use_code=True):
        super(CombinedModel, self).__init__()
        
        self.use_graph = use_graph
        self.use_code = use_code
        
        if not self.use_graph and not self.use_code:
            raise ValueError("At least one of use_graph or use_code must be True.")
        
        # 1. Init Branches
        if self.use_graph:
            self.graph_branch = GraphBranch(graph_in_dims, hidden_dim)
        
        if self.use_code:
            self.code_branch = CodeBranch(
                input_dim=code_in_dim, 
                hidden_dim=hidden_dim,
                bidirectional=True
            )
            lstm_out_dim = self.code_branch.out_dim
            self.code_classifier = nn.Linear(lstm_out_dim, out_dim)
        
        # 2. Init Classifier Heads
        if self.use_graph:
            self.cg_classifier = nn.Linear(hidden_dim, out_dim)
            self.cfg_classifier = nn.Linear(hidden_dim, out_dim)
            self.dfg_classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, batch):
        outputs = {}
        
        # --- Branch 1: Graph Model ---
        if self.use_graph:
            g = batch['graph']
            h_graph = self.graph_branch(g)
            
            # Align and classify 'function' nodes for 'cg' output
            h_func = h_graph['function']
            func_nodes_per_graph = g.batch_num_nodes('function').tolist()
            h_func_split = torch.split(h_func, func_nodes_per_graph)
            padded_h_func = rnn_utils.pad_sequence(h_func_split, batch_first=True)
            outputs['cg'] = self.cg_classifier(padded_h_func)
            
            # Align and classify 'block' nodes for 'cfg' output
            h_block = h_graph['block']
            block_nodes_per_graph = g.batch_num_nodes('block').tolist()
            h_block_split = torch.split(h_block, block_nodes_per_graph)
            padded_h_block = rnn_utils.pad_sequence(h_block_split, batch_first=True)
            outputs['cfg'] = self.cfg_classifier(padded_h_block)
            
            # Align and classify 'dfg_node' nodes for 'dfg' output
            h_dfg = h_graph['dfg_node']
            dfg_nodes_per_graph = g.batch_num_nodes('dfg_node').tolist()
            h_dfg_split = torch.split(h_dfg, dfg_nodes_per_graph)
            padded_h_dfg = rnn_utils.pad_sequence(h_dfg_split, batch_first=True)
            outputs['dfg'] = self.dfg_classifier(padded_h_dfg)
        else:
            # If not using graph, create empty tensors for graph tasks
            # This ensures consistency in output structure
            device = batch.get('code', torch.tensor([0])).device if 'code' in batch else torch.device('cpu')
            outputs['cg'] = torch.empty(0, 0, 9, device=device)
            outputs['cfg'] = torch.empty(0, 0, 9, device=device)
            outputs['dfg'] = torch.empty(0, 0, 9, device=device)
        
        # --- Branch 2: Code Model ---
        if self.use_code:
            padded_code = batch['code']
            code_lengths = batch['code_lengths']
            h_code = self.code_branch(padded_code, code_lengths)
            outputs['code'] = self.code_classifier(h_code)
        else:
            # If not using code, create empty tensor for code task
            device = batch.get('graph').device if 'graph' in batch else torch.device('cpu')
            outputs['code'] = torch.empty(0, 0, 9, device=device)

        return outputs
