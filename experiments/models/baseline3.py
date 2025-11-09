import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch as dglnn
import torch.nn.utils.rnn as rnn_utils
import dgl.function as fn

class GraphBranch(nn.Module):
    """
    This module processes the heterogeneous graph using GINE (Graph Isomorphism Network with Edge features).
    It projects all node types to a common 'hidden_dim'.
    Supports AST, CFG (block), CG (function), and DFG (dfg_node) nodes.
    """
    def __init__(self, node_in_dims, edge_in_dims, hidden_dim, num_layers=2):
        super(GraphBranch, self).__init__()
        self.in_dims = node_in_dims # Renamed for clarity
        self.hidden_dim = hidden_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Input projection layers
        self.project_ast = nn.Linear(node_in_dims['ast_node'], hidden_dim)
        self.project_block = nn.Linear(node_in_dims['block'], hidden_dim)
        self.project_func = nn.Linear(node_in_dims['function'], hidden_dim)
        self.project_dfg = nn.Linear(node_in_dims['dfg_node'], hidden_dim) 
        
        # 2. ADD THIS BLOCK - Create projection layers for edge features
        self.project_edges = nn.ModuleDict({
            'child_of': nn.Linear(edge_in_dims['ast_edge'], hidden_dim),
            'flows': nn.Linear(edge_in_dims['cfg_edge'], hidden_dim),
            'calls': nn.Linear(edge_in_dims['cg_edge'], hidden_dim),
            'data_flow': nn.Linear(edge_in_dims['dfg_edge'], hidden_dim),
        })

        self.activation = nn.ReLU()
        
        # GNN Layers - using GINEConv for edge-aware message passing
        self.layers = nn.ModuleList()
        # Define MLP for GINE aggregation
        apply_func = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
            
        for _ in range(num_layers):
            # Define one GINE layer for each edge type
            # Including both intra-graph edges and inter-graph link edges
            conv_dict = {
                # Intra-graph edges (within each graph component)
                'child_of': dglnn.GINEConv(apply_func),           # AST edges
                'flows': dglnn.GINEConv(apply_func),              # CFG edges
                'calls': dglnn.GINEConv(apply_func),              # CG edges
                'data_flow': dglnn.GINEConv(apply_func),          # DFG edges
                
                # Inter-graph link edges (connecting different graph components)
                'has_block': dglnn.GINEConv(apply_func),          # CG -> CFG
                'part_of_func': dglnn.GINEConv(apply_func),       # CFG -> CG
                'defines_ast': dglnn.GINEConv(apply_func),        # CG -> AST
                'defined_by_func': dglnn.GINEConv(apply_func),    # AST -> CG
                'related_dfg': dglnn.GINEConv(apply_func),        # CFG -> DFG
                'related_block': dglnn.GINEConv(apply_func),      # DFG -> CFG
            }
            # HeteroGraphConv wraps them all up
            self.layers.append(dglnn.HeteroGraphConv(conv_dict, aggregate='sum'))

    def forward(self, g):
        # 1. Project Node Features
        h = {}
        if g.num_nodes('ast_node') > 0:
            h['ast_node'] = self.activation(self.project_ast(g.nodes['ast_node'].data['feat']))
        if g.num_nodes('block') > 0:
            h['block'] = self.activation(self.project_block(g.nodes['block'].data['feat']))
        if g.num_nodes('function') > 0:
            h['function'] = self.activation(self.project_func(g.nodes['function'].data['feat']))
        if g.num_nodes('dfg_node') > 0:
            h['dfg_node'] = self.activation(self.project_dfg(g.nodes['dfg_node'].data['feat']))
        
        # 2. Prepare edge features for GINEConv
        # We project features for edges that have them (e.g., 'flows')
        # and create correct-dimension zero-tensors for linking edges (e.g., 'has_block')
        
        mod_kwargs = {}
        
        for stype, etype, dtype in g.canonical_etypes:
            relation_key = etype
            num_edges = g.num_edges((stype, etype, dtype))

            if num_edges == 0:
                # DGL's HeteroGraphConv CAN handle None if num_edges is 0
                mod_kwargs[relation_key] = {
                    'edge_feat': torch.empty(0, self.hidden_dim, device=self.device)
                }
                continue

            # Case 1: Edge has features (e.g., 'flows', 'calls')
            if relation_key in self.project_edges:
                raw_efea = g.edges[stype, etype, dtype].data['feat']
                mod_kwargs[relation_key] = {
                    'edge_feat': self.activation(self.project_edges[relation_key](raw_efea))
                }
            
            # Case 2: Edge is a linking edge (e.g., 'has_block')
            else:
                # We must provide a zero-tensor of the *hidden_dim*
                # This fixes the '256 vs 1' mismatch
                device = h[stype].device # Get device from source node features
                zero_efea = torch.zeros(num_edges, self.hidden_dim, device=device)
                mod_kwargs[relation_key] = {'edge_feat': zero_efea}

        # 3. Run through GNN layers
        for layer in self.layers:
            h_new = layer(g, h, mod_kwargs=mod_kwargs)
            h = {k: self.activation(v) for k, v in h_new.items()}
            
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
    def __init__(self, node_in_dims, edge_in_dims, code_in_dim, graph_hidden_dim, code_hidden_dim, out_dim=8, use_graph=True, use_code=True):
        super(CombinedModel, self).__init__()
        
        self.use_graph = use_graph
        self.use_code = use_code

        if not self.use_graph and not self.use_code:
            raise ValueError("At least one of use_graph or use_code must be True.")
        
        # 1. Init Branches
        if self.use_graph:
            self.graph_branch = GraphBranch(node_in_dims, edge_in_dims, graph_hidden_dim)
        
        if self.use_code:
            self.code_branch = CodeBranch(
                input_dim=code_in_dim, 
                hidden_dim=code_hidden_dim,
                bidirectional=True
            )
            lstm_out_dim = self.code_branch.out_dim
            self.code_classifier = nn.Linear(lstm_out_dim, out_dim)
        
        # 2. Init Classifier Heads
        if self.use_graph:
            self.cg_classifier = nn.Linear(graph_hidden_dim, out_dim)
            self.cfg_classifier = nn.Linear(graph_hidden_dim, out_dim)
            self.dfg_classifier = nn.Linear(graph_hidden_dim, out_dim)

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
            outputs['cg'] = torch.empty(0, 0, 8, device=device)
            outputs['cfg'] = torch.empty(0, 0, 8, device=device)
            outputs['dfg'] = torch.empty(0, 0, 8, device=device)
        
        # --- Branch 2: Code Model ---
        if self.use_code:
            padded_code = batch['code']
            code_lengths = batch['code_lengths']
            h_code = self.code_branch(padded_code, code_lengths)
            outputs['code'] = self.code_classifier(h_code)
        else:
            # If not using code, create empty tensor for code task
            device = batch.get('graph').device if 'graph' in batch else torch.device('cpu')
            outputs['code'] = torch.empty(0, 0, 8, device=device)

        return outputs
