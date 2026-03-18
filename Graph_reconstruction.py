import torch
import networkx as nx

from pyvis.network import Network
import subprocess
import webbrowser
from pathlib import Path
import networkx as nx
import numpy as np

import Data_Loader


def most_likely_recipient_node(G: nx.DiGraph, scores: torch.Tensor, threshold = 0.5):

    best_node = None
    best_score = -float("inf")

    for node in G.nodes():

        score = scores[node].item()

        # descendants = alle Nodes unterhalb
        descendants = nx.descendants(G, node)

        if len(descendants) > 0:
            max_desc_score = scores[list(descendants)].max().item()
        else:
            max_desc_score = -float("inf")

        # wenn unterhalb kein hoher Score
        if max_desc_score <= threshold:

            if score > best_score:
                best_score = score
                best_node = node

    if best_score < threshold:
        best_node = None
    return best_node, best_score


def most_likely_donor_nodes(
    probs: torch.Tensor,
    possible_donor_mask: torch.Tensor,
    threshold: float = 0.1,
    top_k: int = 3
):
    """
    Returns the most likely donor nodes based on probability scores.

    Parameters
    ----------
    probs : torch.Tensor
        Probability for each node (shape: [N]).
    possible_donor_mask : torch.Tensor
        Mask indicating which nodes are valid donors (shape: [N] or [N,1]).
    threshold : float, optional
        If set, only nodes with probability >= threshold are returned.
    top_k : int
        Maximum number of nodes to return.

    Returns
    -------
    list of tuples
        [(node_id, probability), ...] sorted by probability descending.
    """

    probs = probs.squeeze()
    mask = possible_donor_mask.squeeze().bool()

    # Only consider valid donors
    valid_indices = torch.where(mask)[0]
    valid_probs = probs[valid_indices]

    # Sort by probability
    sorted_probs, order = torch.sort(valid_probs, descending=True)
    sorted_nodes = valid_indices[order]

    results = []

    for node, prob in zip(sorted_nodes, sorted_probs):

        if threshold is not None and prob < threshold:
            break

        results.append((int(node.item()), float(prob.item())))

        if len(results) >= top_k:
            break

    return results

def visualize_graph_reconstruction(graph, graph_tensor, probs, suspected_donor):

    # ----------------------------------------------------------
    # True recipient / donor aus Tensor bestimmen
    # ----------------------------------------------------------

    recipient = (
        (graph_tensor.time_dist_from_recipient[:,0] +
         graph_tensor.time_dist_from_recipient[:,1] == 0)
        .nonzero(as_tuple=True)[0].item()
    )

    donor = (
        (graph_tensor.time_dist_from_donor[:,0] +
         graph_tensor.time_dist_from_donor[:,1] == 0)
        .nonzero(as_tuple=True)[0].item()
    )

    print(max(list(graph.nodes())))
    if donor < max(list(graph.nodes())):
        donor_parent = list(graph.predecessors(donor))[0]
    else: 
        donor_parent = donor

    if suspected_donor < max(list(graph.nodes())):
        suspected_donor_parent = list(graph.predecessors(suspected_donor))[0]
    else: 
        suspected_donor_parent = suspected_donor

    # ----------------------------------------------------------
    # Recipient child bestimmen
    # ----------------------------------------------------------

    event_sum = int(graph_tensor.event_label.sum())

    if event_sum == 1:
        recipient_child = sorted(list(graph.successors(recipient)))[0]
    elif event_sum == 2:
        recipient_child = sorted(list(graph.successors(recipient)))[1]
    else:
        recipient_child = None

    # ----------------------------------------------------------
    # Daten vorbereiten
    # ----------------------------------------------------------

    gene_absence_presence_matrix = [
        graph.nodes[node]["gene_present_below_node"] > 0
        for node in graph.nodes() if node < 100
    ]

    child_sim = graph_tensor.child_similarity.cpu().numpy()

    pred_probs = {node: float(probs[i]) for i, node in enumerate(graph.nodes())}
    child_sim_map = {node: child_sim[i] for i, node in enumerate(graph.nodes())}

    true_recipient_parent_nodes = [
        node for node in graph.nodes()
        if graph.nodes[node]["recipient"].get("is_parent_node", False)
    ]

    # ----------------------------------------------------------
    # Top-5 Donor Nodes
    # ----------------------------------------------------------

    sorted_probs = sorted(pred_probs.items(), key=lambda x: x[1], reverse=True)
    top5_donor_nodes = [node for node, prob in sorted_probs[:5] if prob > 0]

    # ----------------------------------------------------------
    # Layout berechnen
    # ----------------------------------------------------------

    x_spacing = 100
    y_spacing = 100

    node_x = {}
    node_y = {}

    max_level = max(graph.nodes[n].get("level", 0) for n in graph.nodes)

    def get_descendant_leaves(G, node):
        stack = list(G.successors(node))
        reachable_leaves = []

        while stack:
            temp = stack.pop()
            children = list(G.successors(temp))

            if children:
                stack.extend(children)
            else:
                reachable_leaves.append(temp)

        return reachable_leaves

    leaves = [n for n in graph.nodes if graph.nodes[n].get("level",0) == 0]

    for i,node in enumerate(sorted(leaves)):
        node_x[node] = i * x_spacing
        node_y[node] = (max_level) * y_spacing

    levels = sorted(set(nx.get_node_attributes(graph,"level").values()))

    for level in levels[1:]:

        nodes_in_level = [n for n in graph.nodes if graph.nodes[n].get("level",0) == level]

        for node in nodes_in_level:

            leaves = get_descendant_leaves(graph,node)

            if leaves:
                node_x[node] = np.mean([node_x[l] for l in leaves if l in node_x])
            else:
                node_x[node] = 0

            node_y[node] = (max_level-level)*y_spacing

    # ----------------------------------------------------------
    # Network initialisieren
    # ----------------------------------------------------------

    net = Network(height="900px", width="100%", directed=True)

    net.set_options("""
    {
      "nodes": { "shape": "dot", "size": 12, "font": { "size": 24 }},
      "edges": { "arrows": { "to": { "enabled": true, "scaleFactor": 0.5 }}},
      "physics": { "enabled": false }
    }
    """)

    # ----------------------------------------------------------
    # Nodes hinzufügen
    # ----------------------------------------------------------

    for node in graph.nodes():

        pred_value = pred_probs[node]
        level = graph.nodes[node].get("level",0)

        if level > 0:

            sim_vals = child_sim_map[node]

            label = (
                f"{node}\n"
                f"Pred:{pred_value:.3f}"
                f"  Sim/Dif:{int(sim_vals[0])} {int(sim_vals[1])} {int(sim_vals[2])} "
                f"{int(sim_vals[3])} {int(sim_vals[4])} {int(sim_vals[5])}"
            )

        else:
            label = f"{node}"

        if node in top5_donor_nodes:
            color = "red"

        elif node in true_recipient_parent_nodes:
            color = "blue"

        elif node < 100 and gene_absence_presence_matrix[node]:
            color = "orange"

        elif node < 100:
            color = "black"

        else:
            color = "lightblue"

        net.add_node(
            node,
            label=label,
            color=color,
            x=node_x[node],
            y=node_y[node]
        )

    # ----------------------------------------------------------
    # Tree edges
    # ----------------------------------------------------------

    for u,v in graph.edges():
        net.add_edge(u,v)

    # ----------------------------------------------------------
    # HGT edges
    # ----------------------------------------------------------

    mid_donor_x = (1.1 * node_x[donor] + 0.9 * node_x[donor_parent]) / 2
    mid_donor_y = (1.1 * node_y[donor] + 0.9 * node_y[donor_parent]) / 2

    mid_recipient_x = (node_x[recipient] + node_x[recipient_child]) / 2
    mid_recipient_y = (node_y[recipient] + node_y[recipient_child]) / 2

    mid_suspected_donor_x = (0.9 * node_x[suspected_donor] + 1.1 * node_x[suspected_donor_parent]) / 2
    mid_suspected_donor_y = (0.9 * node_y[suspected_donor] + 1.1 * node_y[suspected_donor_parent]) / 2

    net.add_node("Donor", x=mid_donor_x, y=mid_donor_y, size=1, color="rgba(0,0,0,0)")
    net.add_node("Recipient", x=mid_recipient_x, y=mid_recipient_y, size=1, color="rgba(0,0,0,0)")
    net.add_node("Suspected Donor", x=mid_suspected_donor_x, y=mid_suspected_donor_y, size=1, color="rgba(0,0,0,0)")

    net.add_edge("Donor","Recipient", color="red", arrows="to", width=7)
    net.add_edge("Suspected Donor","Recipient", color="green", arrows="to", width=7)

    # ----------------------------------------------------------
    # HTML speichern
    # ----------------------------------------------------------

    html_file = Path("/mnt/c/ProjectHGT/graph.html")
    html_file.parent.mkdir(parents=True, exist_ok=True)

    net.show(str(html_file), notebook=False)

    win_path = subprocess.run(
        ["wslpath","-w",str(html_file)],
        capture_output=True,
        text=True
    ).stdout.strip()

    subprocess.run(["cmd.exe","/C","start","chrome",win_path])

