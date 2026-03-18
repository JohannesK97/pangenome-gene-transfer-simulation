import torch
from pathlib import Path
from pyvis.network import Network
import subprocess
import webbrowser
import networkx as nx

import Data_Loader


folder = "/mnt/c/Users/uhewm/Desktop/ProjectHGT"
Path(folder).mkdir(parents=True, exist_ok=True)  # sicherstellen, dass der Ordner existiert

def save_model_old(model, filename="donorfinder_model.pt", print_save_path = True):
    """
    Speichert das Modell (nur Parameter) im angegebenen Ordner.
    """
    path = Path(folder) / filename
    torch.save(model.state_dict(), path)
    if print_save_path:
        print(f"Model saved to {path}")

def save_DonorFinder(model, global_max = None, global_min = None,  eps = None, filename="donorfinder_model.pt", print_save_path=True):
    path = Path(folder) / filename

    checkpoint = {
        "model_class": model.__class__.__name__,
        "config": {
            #"num_nodes": model.num_nodes,
            "internal_node_data_dim": model.internal_node_data_dim,
            "graph_information_dim": model.graph_information_dim,
        },
        "state_dict": model.state_dict(),
        "global_max": global_max,
        "global_min": global_min,
        "eps": eps,    
    }

    torch.save(checkpoint, path)

    if print_save_path:
        print(f"Model saved to {path}")


def load_model_old(model_class, num_nodes, internal_node_data_dim, graph_information_dim, filename="donorfinder_model.pt"):
    """
    Lädt ein Modell mit den gespeicherten Parametern.
    model_class: Klasse des Modells (z.B. DonorFinder)
    num_nodes, internal_node_data_dim, graph_information_dim: müssen beim Erstellen des Modells wieder exakt gleich sein
    """
    path = Path(folder) / filename
    model = model_class(
        num_nodes=num_nodes,
        internal_node_data_dim=internal_node_data_dim,
        graph_information_dim=graph_information_dim
    )
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model

def load_DonorFinder(model_class, filename="donorfinder_model.pt"):
    path = Path(folder) / filename

    checkpoint = torch.load(path, weights_only=False)

    model = model_class(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    global_max = checkpoint.get("global_max")
    global_min = checkpoint.get("global_min")
    eps = checkpoint.get("eps")
    
    print(f"Model loaded from {path}")

    return model, global_max, global_min, eps

def save_RecipientFinder(model, best_threshold=None, filename="recipient_finder_model.pt", print_save_path=True):
    path = Path(folder) / filename

    checkpoint = {
        "model_class": model.__class__.__name__,
        "config": {
            "internal_node_data_dim": model.internal_node_data_dim
        },
        "state_dict": model.state_dict(),
        "best_threshold": best_threshold
    }

    torch.save(checkpoint, path)

    if print_save_path:
        print(f"Model saved to {path}")


def load_RecipientFinder(model_class, filename="recipient_finder_model.pt"):
    path = Path(folder) / filename

    checkpoint = torch.load(path, weights_only=False)

    model = model_class(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    best_threshold = checkpoint.get("best_threshold", 0.5)

    print(f"Model loaded from {path}")
    print(f"Best threshold: {best_threshold:.3f}")

    return model, best_threshold


def visualize_RecipientFinder(file):
    random_file = random.choice(files)
    
    d = Data_Loader.load_file(random_file)
    single_graph = Data_Loader.aggregate_sequences(d)
    
    single_data = Data_Loader.graph_to_dataset_RecipientFinder(single_graph)
    
    """
    for node, attrs in single_graph.nodes(data=True):
        print(node, attrs)
    
    print(single_data.x)
    """
    
    hgt_nodes = [node for node in single_graph.nodes() if single_graph.nodes[node]["recipient"]["is_parent_node"] ]
    gene_absence_presence_matrix = [single_graph.nodes[node]["gene_present_below_node"] > 0 for node in single_graph.nodes() if node < 100]
    
    # === 1. Modellvorhersagen berechnen ===
    model.eval()
    with torch.no_grad():
        logits = model(single_data.x, single_data.internal_node_data, single_data.level, single_data.edge_index)  # Shape: [num_nodes]
        probs = torch.sigmoid(logits).cpu().numpy()  # Werte zwischen 0 und 1
    
    # Map von Node-ID zu Wahrscheinlichkeit
    pred_probs = {i: p > best_threshold for i, p in enumerate(probs)}
    pred_nodes = sorted([i for i, flag in pred_probs.items() if flag])
    
    node_to_id = {node: i for i, node in enumerate(single_graph.nodes)}
    print("Predicted Nodes: ", sorted([node_to_id[i] for i, flag in pred_probs.items() if flag]))
    print("True Nodes: ", sorted(set(hgt_nodes)))
    
    pred_probs = {node_to_id[i]: p for i, p in enumerate(probs)}
    
    
    # --- x/y Koordinaten für Blätter und innere Knoten berechnen ---
    x_spacing = 100
    y_spacing = 100
    
    node_x = {}
    node_y = {}
    
    # Maximaler Level aus single_graph
    max_level = max(single_graph.nodes[n].get("level", 0) for n in single_graph.nodes)
    
    # Hilfsfunktion: finde alle Blätter unterhalb eines Knotens
    def get_descendant_leaves(G, node):
        """Alle Blätter, die von `node` erreichbar sind (rekursiv)."""
        stack = list(G.successors(node))
        reachable_leaves = []
        while stack:
            temp_node = stack.pop()
            children = list(G.successors(temp_node))
            if len(children) > 0:
                stack.extend(children)
            else:
                reachable_leaves.append(temp_node)
        return reachable_leaves
    
    # === Blätter (Level 0) oben ===
    leaves = [n for n in single_graph.nodes if single_graph.nodes[n].get("level", 0) == 0]
    for i, node in enumerate(sorted(leaves)):  
        node_x[node] = i * x_spacing
        node_y[node] = (max_level - 0) * y_spacing  # Blätter oben
    
    # === Innere Knoten: levelweise platzieren ===
    levels_in_graph = sorted(set(nx.get_node_attributes(single_graph, "level").values()))
    for level in levels_in_graph[1:]:  # 0 schon behandelt
        nodes_in_level = [n for n in single_graph.nodes if single_graph.nodes[n].get("level", 0) == level]
        for node in nodes_in_level:
            # Finde alle Blätter unterhalb
            reachable_leaves = get_descendant_leaves(single_graph, node)
            if reachable_leaves:
                leaf_x = [node_x[l] for l in reachable_leaves if l in node_x]
                node_x[node] = np.mean(leaf_x)
            else:
                node_x[node] = 0
            node_y[node] = (max_level - level) * y_spacing
       
    # === Netzwerk initialisieren (Hierarchical Layout deaktiviert!) ===
    net = Network(height="900px", width="100%", directed=True)
    
    net.set_options("""
    {
      "nodes": {
        "shape": "dot",
        "size": 12,
        "font": { "size": 30 }
      },
      "edges": {
        "arrows": {
          "to": { "enabled": true, "scaleFactor": 0.5 }
        }
      },
      "physics": {
        "enabled": false
      }
    }
    """)
    # === Knoten hinzufügen mit festen x/y ===
    
    ATTRS = [
        "sum_seq",
        "tree_length",
        "time",
        "pred"
    ]
    
    for node in single_graph.nodes():
        values = {}
    
        for key in ATTRS:
            if key == "pred":
                values[key] = float(pred_probs[node])
            else:
                values[key] = float(single_graph.nodes[node].get(key, 0))
    
        # Title-String
        title = ", ".join([f"{k}: {values[k]:.2f}" for k in ATTRS])
    
        # Label-String (mit Zeilenumbrüchen)
        label_values = ", ".join([f"{values[k]:.2f}" for k in ATTRS])
        label = f"{node}\n({label_values})"
    
        # Farbe
        if node in hgt_nodes and pred_probs[node] > best_threshold:
            color = "green"
        elif node not in hgt_nodes and pred_probs[node] > best_threshold:
            color = "violet"
        elif node in hgt_nodes and pred_probs[node] <= best_threshold:
            color = "red"
        elif node < 100 and gene_absence_presence_matrix[node] == 1:
            color = "orange"
        elif node < 100 and gene_absence_presence_matrix[node] == 0:
            color = "black"
        else:
            color = "lightblue"
    
    
        net.add_node(node, label=label, title=title, color=color,
                     x=node_x[node], y=node_y[node])
    
    # === Kanten hinzufügen ===
    for u, v in single_graph.edges():
        net.add_edge(u, v)
    
    # === HTML-Datei speichern und direkt in Chrome öffnen ===
    html_file = Path("/mnt/c/Users/uhewm/OneDrive/PhD/Project No.2/pangenome/graph.html")
    html_file = Path("/mnt/c/ProjectHGT/graph.html")
    html_file.parent.mkdir(parents=True, exist_ok=True)
    net.show(str(html_file), notebook=False)
    
    # WSL-Pfad in Windows-Pfad umwandeln
    win_path = subprocess.run(["wslpath", "-w", str(html_file)], capture_output=True, text=True).stdout.strip()
    
    # Direkt in Chrome öffnen
    subprocess.run(["cmd.exe", "/C", "start", "chrome", win_path])
