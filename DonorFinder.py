import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
import numpy as np
from collections import defaultdict

import Data_Loader

class DonorFinder(nn.Module):

    def __init__(self, internal_node_data_dim, graph_information_dim,
                 conv_out_channels = 4, conv_kernel_size = 10, fc_hidden_channels = 512,
                 recurrent_dim = 128,
                 num_fc_layers = 4, dropout=0.2, max_number_of_snps = 300, len_alphabet = 5, recurrent_nn = True, subtree_information = False):

        super().__init__()

        #self.num_nodes = num_nodes
        self.max_number_of_snps = max_number_of_snps
        self.len_alphabet = len_alphabet
        self.internal_node_data_dim = internal_node_data_dim
        self.graph_information_dim = graph_information_dim
        self.recurrent_nn = recurrent_nn
        self.recurrent_dim = recurrent_dim
        self.subtree_information = subtree_information

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

        current_dim = internal_node_data_dim + graph_information_dim
        self.Tree_LSTM_top_bottom = BinaryTreeLSTMCell_top_bottom(hidden_dim = recurrent_dim, 
                                                                  internal_node_data_dim = internal_node_data_dim + graph_information_dim)
        self.Tree_LSTM_bottom_top = BinaryTreeLSTMCell_bottom_top(hidden_dim = recurrent_dim, 
                                                                  internal_node_data_dim = internal_node_data_dim + graph_information_dim)
        
        if recurrent_nn:
            #current_dim = 2 * current_dim + internal_node_data_dim
            current_dim = internal_node_data_dim + 2 * recurrent_dim
        else:
            current_dim = internal_node_data_dim

        if self.subtree_information:
            current_dim = 3 * current_dim

        # ----- GCN Layer -----

        self.fusion_layer = Data_Loader.ParentChildFusionLayer(in_dim=current_dim)
        #current_dim = 6 * current_dim

        current_dim = current_dim + graph_information_dim

        # ----- Shared Fully Connected Backbone -----
        self.shared_layers = nn.ModuleList()
        self.dropout = dropout

        current_fc_hidden_channels = fc_hidden_channels
        if num_fc_layers <= 1:
            self.shared_layers.append(nn.Linear(current_dim, fc_hidden_channels))
        else:
            self.shared_layers.append(nn.Linear(current_dim, fc_hidden_channels))
            for _ in range(num_fc_layers - 1):
                self.shared_layers.append(nn.Linear(current_fc_hidden_channels, current_fc_hidden_channels // 2))
                current_fc_hidden_channels = current_fc_hidden_channels // 2
        
        # ----- Task Heads -----
        
        self.head_donor_score = nn.Linear(current_fc_hidden_channels, 1)


    def forward(self, x, internal_node_data, graph_information, level, edge_index, valid_node, batch):
        """
        Apply stacked fusion layers followed by linear classifiers.
        """      
        N = x.size(0)
        #N = self.num_nodes
        device = x.device

        num_graphs = batch.max().item() + 1
        graph_information = graph_information.view(num_graphs, -1)
        graph_information = graph_information[batch]

        #internal_node_data = (internal_node_data - mean) / std
        #internal_node_data = torch.clamp(internal_node_data, -10, 10)

        # --- Restore one-hot structure ---
        x = x.view(N, self.max_number_of_snps, self.len_alphabet)
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)   # [N, 1, 5, max_snps]

        """
        # ----- CNN -----
        x_internal = self.conv(x)
        x_internal = F.relu(x_internal)
    
        x_internal = x_internal.squeeze(2)     # [N, C, W]
        x_internal = x_internal.flatten(start_dim=1)  # [N, C * W]
        """

        # ----- LSTM Cell: Bottom-up traversal -----

        """
        x_lstm = self.conv_lstm(x)
        x_lstm = F.relu(x_lstm)
    
        x_lstm = x_lstm.squeeze(2)     # [N, C, W]
        x_lstm = x_lstm.flatten(start_dim=1)  # [N, C * W]
        """

        x_lstm = internal_node_data
        x_lstm = torch.zeros(N, self.recurrent_dim, device=internal_node_data.device)
        #x_lstm = torch.cat((internal_node_data, graph_information), dim=1)
        
        children = self.build_children(edge_index, N)
        parent = self.build_parent(edge_index, N)

        if self.recurrent_nn:
            recurrent_data_top_down = self.tree_lstm_top_down(
                x = x_lstm,
                parent = parent,
                internal_node_data = torch.cat((internal_node_data, graph_information), dim=1),
                level = level,
                cell = self.Tree_LSTM_top_bottom
            )        
            recurrent_data_bottom_top = self.tree_lstm_bottom_up(
                x = x_lstm,
                children = children,
                internal_node_data = torch.cat((internal_node_data, graph_information), dim=1),
                level = level,
                cell = self.Tree_LSTM_bottom_top
            )
            
            #x = torch.cat((x_internal, recurrent_data, internal_node_data), dim=1)
            x = torch.cat((recurrent_data_top_down, recurrent_data_bottom_top, internal_node_data), dim=1)
        else:
            x = internal_node_data

        # ----- Subtree Layer -----

        if self.subtree_information:
            x_max_list = self.compute_subtree_representation(
                x = x,
                valid_node = valid_node,
                children = children,
                level = level,
                aggregation = "max"
            )
            x_min_list = self.compute_subtree_representation(
                x = x,
                valid_node = valid_node,
                children = children,
                level = level,
                aggregation = "min"
            )

            x_max = torch.cat([
                t if t is not None else torch.zeros(1, x.size(1), device=device)
                for t in x_max_list
            ], dim=0)
            
            x_min = torch.cat([
                t if t is not None else torch.zeros(1, x.size(1), device=device)
                for t in x_min_list
            ], dim=0)
            
            x = torch.cat((x, x_max, x_min), dim=1)
        
        # ----- GCN Layer -----

        #x = self.fusion_layer(x, edge_index)

        x = torch.cat((x, graph_information), dim=1)

        # --- GLOBAL GRAPH CONTEXT ---
        #g = global_mean_pool(x, batch)      # [num_graphs, dim]
        #x = torch.cat((x, g[batch]), dim=1) # expand global vector to nodes
                        
        # ----- Shared Backbone -----

        for layer in self.shared_layers:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(layer(x))

        # ----- Heads -----

        """
        real_event_logits = self.head_real_event(mean_x).view(-1)
        
        recipient_child_logits = self.head_recipient_child(mean_x).view(-1)
        
        donor_child_logits = self.head_donor_child(x).view(-1)
        
        return {
            "real_event_logits": real_event_logits,
            "recipient_child_logits": recipient_child_logits,
            "donor_child_logits": donor_child_logits
        }

        node_logits = self.head_event(x)  # [num_nodes_total, 3]
        
        return node_logits
        """
        
        node_scores = self.head_donor_score(x).view(-1)  # [num_nodes_total]

        """
        max_per_graph = torch.zeros(batch.max() + 1, device=x.device)
        max_per_graph.scatter_reduce_(
            0,
            batch,
            node_scores,
            reduce="amax",
            include_self=False
        )
        
        node_scores = node_scores - max_per_graph[batch]
        """
        
        return node_scores


    def build_children(self, edge_index, num_nodes):
        """
        edge_index: [2, E] tensor, child -> parent
        """
        children = [[] for _ in range(num_nodes)]
    
        src, dst = edge_index
        for c, p in zip(src.tolist(), dst.tolist()):
            children[p].append(c)
    
        return children

    def build_parent(self, edge_index, num_nodes):
        """
        edge_index: [2, E] child -> parent
        returns parent list of length N (root gets -1)
        """
        parent = [-1] * num_nodes
    
        src, dst = edge_index
        for c, p in zip(src.tolist(), dst.tolist()):
            parent[c] = p
    
        return parent
        
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

    def tree_lstm_top_down(self, x, parent, internal_node_data, level, cell):
        """
        x: [N, D] embeddings
        parent: list length N
        level: tensor [N]
        """
    
        device = x.device
        N, D = x.shape
    
        h = torch.zeros(N, D, device=device)
        c = torch.zeros(N, D, device=device)
    
        levels = self.build_levels(level)
        max_level = max(levels.keys())
    
        # root = highest level
        root_nodes = levels[max_level]
    
        # initialize root(s)
        h[root_nodes] = x[root_nodes]
        c[root_nodes] = torch.zeros_like(h[root_nodes])
    
        # traverse downward
        for d in reversed(range(max_level)):
            nodes = levels[d]
    
            parents = torch.tensor(
                [parent[v] for v in nodes],
                device=device
            )
    
            h_parent = h[parents]
            c_parent = c[parents]
    
            node_data = internal_node_data[nodes]
    
            h_new, c_new = cell(node_data, h_parent, c_parent)
    
            h[nodes] = h_new
            c[nodes] = c_new
    
        return h

    def compute_subtree_representation(self, x, valid_node, children, level, aggregation="max"):
        """
        Computes subtree representations for every node,
        using ONLY valid nodes (valid_node == 1).
    
        x: [N, D]
        children: list of lists
        level: tensor [N]
        """
    
        device = x.device
        N, D = x.shape

        valid_mask = valid_node.view(-1).bool()
    
        levels = self.build_levels(level)
        max_level = max(levels.keys())
    
        subtree_repr = [None] * N
    
        # --- Leaves ---
        for leaf in levels[0]:
            if valid_mask[leaf]:
                subtree_repr[leaf] = x[leaf].unsqueeze(0)
            else:
                subtree_repr[leaf] = None
    
        # --- Bottom-up traversal ---
        for d in range(1, max_level + 1):
            nodes = levels[d]
    
            for v in nodes:
    
                child_list = children[v]
    
                collected = []
    
                # collect VALID child subtree tensors
                for c in child_list:
                    if subtree_repr[c] is not None:
                        collected.append(subtree_repr[c])
    
                # add current node if valid
                if valid_mask[v]:
                    collected.append(x[v].unsqueeze(0))
    
                # if no valid node in subtree
                if len(collected) == 0:
                    subtree_repr[v] = None
                    continue
    
                combined = torch.cat(collected, dim=0)
    
                if aggregation == "sum":
                    subtree_repr[v] = combined.sum(dim=0, keepdim=True)
    
                elif aggregation == "mean":
                    subtree_repr[v] = combined.mean(dim=0, keepdim=True)
    
                elif aggregation == "min":
                    subtree_repr[v] = combined.min(dim=0, keepdim=True).values
    
                elif aggregation == "max":
                    subtree_repr[v] = combined.max(dim=0, keepdim=True).values
    
                else:
                    raise ValueError("Unknown aggregation type")
    
        return subtree_repr


class BinaryTreeLSTMCell_bottom_top(nn.Module):
    """
    Binary Tree-LSTM cell (Tai et al., 2015) from leafs to root.
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

class BinaryTreeLSTMCell_top_bottom(nn.Module):
    """
    Binary Tree-LSTM cell (Tai et al., 2015) from root to leafs.
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

    def forward(self, x, h_parent, c_parent):

        i = torch.sigmoid(self.i(torch.cat((x, h_parent), dim=1)))
        f = torch.sigmoid(self.f(torch.cat((x, h_parent), dim=1)))
        o = torch.sigmoid(self.o(torch.cat((x, h_parent), dim=1)))
        u = torch.tanh(self.u(torch.cat((x, h_parent), dim=1)))

        c = i * u + f * c_parent
        h = o * torch.tanh(c)

        return h, c

