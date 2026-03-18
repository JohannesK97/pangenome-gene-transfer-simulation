import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import numpy as np
from collections import defaultdict

import Data_Loader

class RecipientFinder(nn.Module):
    """
    A model that stacks several ParentChildFusionLayers followed by
    fully connected layers for prediction.

    Parameters
    ----------
    in_channels : int
        Dimensionality of the input node features.
    fc_hidden_channels : int
        Dimensionality after the ParentChildFusionLayers.
    num_fusion_layers : int
        Number of ParentChildFusionLayers to apply sequentially.
    dropout : float
        Dropout probability for the fully connected classifier.
    """

    def __init__(self, internal_node_data_dim, 
                 conv_out_channels = 4, conv_kernel_size = 5, fc_hidden_channels = 16, 
                 num_fc_layers = 3, dropout=0.1, max_number_of_snps = 300, len_alphabet = 5):

        super().__init__()

        self.max_number_of_snps = max_number_of_snps
        self.len_alphabet = len_alphabet
        self.internal_node_data_dim = internal_node_data_dim
        
        # ----- CNN Layer -----

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=conv_out_channels,
            kernel_size=(len_alphabet, conv_kernel_size),
            stride=(1, conv_kernel_size),
            padding=0
        )

        dummy_x = torch.zeros(1, 1, len_alphabet, max_number_of_snps)
        with torch.no_grad():
            dummy_out = self.conv(dummy_x)
        dim_after_conv = dummy_out.shape
        current_dim = dim_after_conv[1] * dim_after_conv[3]
        
        # ----- LSTM Cell: Bottom-up traversal -----

        self.conv_lstm = nn.Conv2d(
            in_channels=1,
            out_channels=conv_out_channels,
            kernel_size=(len_alphabet, conv_kernel_size),
            stride=(1, conv_kernel_size),
            padding=0
        )
        
        self.Tree_LSTM = BinaryTreeLSTMCell(hidden_dim = current_dim, internal_node_data_dim = internal_node_data_dim)
        current_dim = 2* current_dim + internal_node_data_dim
        #current_dim = current_dim + internal_node_data_dim

        # ----- GCN Layer -----

        self.fusion_layer = Data_Loader.ParentChildFusionLayer_RecipientFinder(in_dim=current_dim)
        current_dim = 3 * current_dim

        # ----- Fully Connected Layers -----
        self.fc_layers = nn.ModuleList()
        self.dropout = dropout

        # First FC layer: (current_dim → fc_hidden_channels) OR directly → 1
        if num_fc_layers == 1:
            self.fc_layers.append(nn.Linear(current_dim, 1))
        else:
            # First hidden layer
            self.fc_layers.append(nn.Linear(current_dim, fc_hidden_channels))
            # Middle hidden FC layers
            for _ in range(num_fc_layers - 2):
                self.fc_layers.append(nn.Linear(fc_hidden_channels, fc_hidden_channels))
            # Final output layer
            self.fc_layers.append(nn.Linear(fc_hidden_channels, 1))

    def forward(self, x, internal_node_data, level, edge_index):
        """
        Apply stacked fusion layers followed by linear classifiers.
        """      
        N = x.size(0)
    
        # --- Restore one-hot structure ---
        x = x.view(N, self.max_number_of_snps, self.len_alphabet)
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)   # [N, 1, 5, max_snps]
    
        # ----- CNN -----
        x_internal = self.conv(x)
        x_internal = F.relu(x_internal)
    
        x_internal = x_internal.squeeze(2)     # [N, C, W]
        x_internal = x_internal.flatten(start_dim=1)  # [N, C * W]
        
        # ----- LSTM Cell: Bottom-up traversal -----


        x_lstm = self.conv_lstm(x)
        x_lstm = F.relu(x_lstm)
    
        x_lstm = x_lstm.squeeze(2)     # [N, C, W]
        x_lstm = x_lstm.flatten(start_dim=1)  # [N, C * W]

        children = self.build_children(edge_index, N)
        
        recurrent_data = self.tree_lstm_bottom_up(
            x = x_lstm,
            children = children,
            internal_node_data = internal_node_data,
            level = level,
            cell = self.Tree_LSTM
        )
        
        x = torch.cat((x_internal, recurrent_data, internal_node_data), dim=1)

        #x = torch.cat((x_internal, internal_node_data), dim=1)

        # ----- GCN Layer -----

        x = self.fusion_layer(x, edge_index)
        
        # ----- Fully Connected Layers -----
        for i, layer in enumerate(self.fc_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x)
            if i < len(self.fc_layers) - 1:
                x = F.relu(x)

        return x.view(-1)

    def build_children(self, edge_index, num_nodes):
        """
        edge_index: [2, E] tensor, child -> parent
        """
        children = [[] for _ in range(num_nodes)]
    
        src, dst = edge_index
        for c, p in zip(src.tolist(), dst.tolist()):
            children[p].append(c)
    
        return children
    
    def build_levels(self, level):
        levels = defaultdict(list)
        for v, d in enumerate(level):
            levels[int(d)].append(v)
        return levels
        

    def tree_lstm_bottom_up(self, x, children, internal_node_data, level, cell):
        """
        x: [N, D] leaf embeddings (CNN output), internal nodes arbitrary
        children: list of length N, [] or [l, r]
        level: [N] integer level values
        """
        device = x.device
        N, D = x.shape
    
        h = torch.zeros(N, D, device=device)
        c = torch.zeros(N, D, device=device)
    
        levels = self.build_levels(level)
    
        max_level = max(levels.keys())
    
        # level = 0 → leaves
        leaf_nodes = levels[0]
        h[leaf_nodes] = x[leaf_nodes]
    
        # process internal levels
        for d in range(1, max_level + 1):
            nodes = levels[d]
    
            # gather children indices
            left = torch.tensor(
                [children[v][0] for v in nodes],
                device=device
            )
            right = torch.tensor(
                [children[v][1] for v in nodes],
                device=device
            )
            
            node_data = internal_node_data[nodes]
    
            hl = h[left]
            cl = c[left]
            hr = h[right]
            cr = c[right]
            
    
            h_new, c_new = cell(node_data, hl, cl, hr, cr)
    
            h[nodes] = h_new
            c[nodes] = c_new
   
        return h


class BinaryTreeLSTMCell(nn.Module):
    """
    Binary Tree-LSTM cell (Tai et al., 2015).
    """

    def __init__(self, hidden_dim, internal_node_data_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.internal_node_data_dim = internal_node_data_dim
        
        def lin():
            return nn.Linear(hidden_dim + internal_node_data_dim, hidden_dim)

        self.i = lin()
        self.f = lin()
        self.u = lin()
        self.o = lin()

    def forward(self, x, hl, cl, hr, cr):
        """
        hl, cl: left child hidden & cell
        hr, cr: right child hidden & cell

        The left and right order does not matter (invariant).
        """

        hsum = hl + hr

        i = torch.sigmoid(self.i(torch.cat((x, hsum), dim=1)))
        fl = torch.sigmoid(self.f(torch.cat((x, hl), dim=1)))
        fr = torch.sigmoid(self.f(torch.cat((x, hr), dim=1)))
        o = torch.sigmoid(self.o(torch.cat((x, hsum), dim=1)))
        u = torch.tanh(self.u(torch.cat((x, hsum), dim=1)))

        c = i * u + fl * cl + fr * cr

        h = o * torch.tanh(c)

        return h, c
