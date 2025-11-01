import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch as dglnn
import torch.nn.utils.rnn as rnn_utils

class GraphBranch(nn.Module):
    """
    This module processes the heterogeneous graph.
    It projects all node types to a common 'hidden_dim'.
    It creates an embedding for 'dfg_node' since it has no features.
    """
    def __init__(self, in_dims, hidden_dim, num_layers=2):
        super(GraphBranch, self).__init__()
        self.in_dims = in_dims
        self.hidden_dim = hidden_dim
        
        # 1. Create a learnable embedding for 'dfg_node' (which has 0 features)
        # We'll project it up to the hidden_dim
        self.dfg_embedding = nn.Embedding(1, hidden_dim) 
        
        # 2. Input projection layers
        # We project all node types (ast, block, function) to hidden_dim
        # so they can be processed by the GNN layers.
        self.project_ast = nn.Linear(in_dims['ast_node'], hidden_dim)
        self.project_block = nn.Linear(in_dims['block'], hidden_dim)
        self.project_func = nn.Linear(in_dims['function'], hidden_dim)
        
        self.activation = nn.ReLU()
        
        # 3. GNN Layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            # We define one GNN layer (e.g., GraphConv) for each edge type
            conv_dict = {
                'child_of': dglnn.GraphConv(hidden_dim, hidden_dim),
                'flows': dglnn.GraphConv(hidden_dim, hidden_dim),
                'calls': dglnn.GraphConv(hidden_dim, hidden_dim)
            }
            # HeteroGraphConv wraps them all up
            self.layers.append(dglnn.HeteroGraphConv(conv_dict, aggregate='sum'))

    def forward(self, g):
        # 1. Get initial features
        h = {}
        h['ast_node'] = g.nodes['ast_node'].data['feat']
        h['block'] = g.nodes['block'].data['feat']
        h['function'] = g.nodes['function'].data['feat']
        
        # 2. Project features to hidden_dim
        h['ast_node'] = self.activation(self.project_ast(h['ast_node']))
        h['block'] = self.activation(self.project_block(h['block']))
        h['function'] = self.activation(self.project_func(h['function']))
        
        # 3. Handle 'dfg_node'
        # Get the number of dfg_nodes in this batch
        num_dfg_nodes = g.num_nodes('dfg_node')
        if num_dfg_nodes > 0:
            # Use the learned embedding for all dfg_nodes
            dfg_indices = torch.zeros(num_dfg_nodes, dtype=torch.long, device=g.device)
            h['dfg_node'] = self.dfg_embedding(dfg_indices)
            
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
    """
    def __init__(self, graph_in_dims, code_in_dim, hidden_dim, out_dim=9):
        super(CombinedModel, self).__init__()
        
        # 1. Init Branches
        self.graph_branch = GraphBranch(graph_in_dims, hidden_dim)
        
        self.code_branch = CodeBranch(
            input_dim=code_in_dim, 
            hidden_dim=hidden_dim,
            bidirectional=True
        )
        
        # Get the LSTM's output dim
        lstm_out_dim = self.code_branch.out_dim
        
        # 2. Init Classifier Heads
        # We need 3 separate classifiers
        
        # For 'labels' (from code branch)
        self.code_classifier = nn.Linear(lstm_out_dim, out_dim)
        
        # For 'cg_labels' (from graph branch 'function' nodes)
        self.cg_classifier = nn.Linear(hidden_dim, out_dim)
        
        # For 'cfg_labels' (from graph branch 'block' nodes)
        self.cfg_classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, batch):
        g = batch['graph']
        padded_code = batch['code']
        code_lengths = batch['code_lengths']
        
        # --- Branch 1: Graph Model ---
        # h_graph is a dict: {'ast_node': tensor, 'block': tensor, ...}
        # Tensors are CONCATENATED, e.g., shape (total_block_nodes, hidden_dim)
        h_graph = self.graph_branch(g)
        
        # --- Branch 2: Code Model ---
        # h_code is a PADDED tensor, shape (B, max_code_len, lstm_out_dim)
        h_code = self.code_branch(padded_code, code_lengths)
        
        # --- Alignment and Classification ---
        
        # 1. Code -> 'labels'
        # This is easy, both are padded sequences
        # code_preds shape: (B, max_code_len, 9)
        code_preds = self.code_classifier(h_code)
        
        # 2. Graph 'function' nodes -> 'cg_labels'
        # This is the tricky part. We must align the GNN's concatenated
        # output with the padded labels.
        h_func = h_graph['function'] # Shape (total_func_nodes, hidden_dim)
        
        # Get a list of how many 'function' nodes are in each graph
        func_nodes_per_graph = g.batch_num_nodes('function').tolist()
        
        # Split the concatenated tensor into a list of tensors
        h_func_split = torch.split(h_func, func_nodes_per_graph)
        
        # Pad the list of tensors to match the batch's max_len
        # padded_h_func shape: (B, max_func_nodes, hidden_dim)
        padded_h_func = rnn_utils.pad_sequence(h_func_split, batch_first=True)
        
        # Now we can classify it
        # cg_preds shape: (B, max_func_nodes, 9)
        cg_preds = self.cg_classifier(padded_h_func)
        
        # 3. Graph 'block' nodes -> 'cfg_labels'
        # Repeat the same alignment process for 'block' nodes
        h_block = h_graph['block'] # Shape (total_block_nodes, hidden_dim)
        block_nodes_per_graph = g.batch_num_nodes('block').tolist()
        h_block_split = torch.split(h_block, block_nodes_per_graph)
        
        # padded_h_block shape: (B, max_block_nodes, hidden_dim)
        padded_h_block = rnn_utils.pad_sequence(h_block_split, batch_first=True)
        
        # cfg_preds shape: (B, max_block_nodes, 9)
        cfg_preds = self.cfg_classifier(padded_h_block)

        # Return all predictions
        return {
            'code': code_preds,  # (B, max_code_len, 9)
            'cg': cg_preds,    # (B, max_func_nodes, 9)
            'cfg': cfg_preds   # (B, max_block_nodes, 9)
        }
