import networkx as nx

def root(G: nx.DiGraph):
    """Return the unique root of the tree (node with indegree 0)."""
    roots = [n for n, d in G.in_degree() if d == 0]
    if len(roots) != 1:
        raise ValueError("Graph has no unique root")
    return roots[0]

def leaves_under_node(G: nx.DiGraph, node):
    """Return all leaves below a given node."""
    descendants = nx.descendants(G, node)
    return [n for n in descendants if G.out_degree(n) == 0]

def mrca(G: nx.DiGraph, node1, node2):
    """Return the most recent common ancestor (lowest common ancestor)."""
    root = root(G)
    return nx.lowest_common_ancestor(G, node1, node2, root)

