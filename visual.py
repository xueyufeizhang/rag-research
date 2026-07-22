import json
import networkx as nx
from pyvis.network import Network

with open("./dickens/graph.json") as f:
    G = nx.node_link_graph(json.load(f))

# # Remove the isolated nodes from visualization
# largest = max(nx.connected_components(G), key=len)
# G = G.subgraph(largest).copy()


TYPE_COLORS = {
    "Artifact": "#d73027",
    "Person": "#fc8d59",
    "Concept": "#fee090",
    "Location": "#4575b4",
    "NaturalObject": "#91bfdb",
    "Content": "#ac667e",
    "Creature": "#8e44ad",
    "Event": "#f9ae78",
    "Organization": "#12685d",
    "Method": "#00a391",
    "Data": "#63bbb0",
    "None": "#999999"
}


for node, attrs in G.nodes(data=True):
    attrs["label"] = attrs.get("name", "")
    attrs["color"] = TYPE_COLORS.get(attrs.get("type") or "None", "#999999")
    attrs["size"] = 10 + (G.degree(node) ** 0.5) * 4
    attrs["title"] = (f"Type: {attrs.get('type') or 'Unknown'}\n"
                      f"{attrs.get('description') or '(no description)'}\n"
                      f"Appears in {len(attrs.get('source_id', []))} chunk(s)")

for u, v, attrs in G.edges(data=True):
    attrs["title"] = attrs.get("description", "")
    attrs["width"] = 1 + len(attrs.get("keywords", []))


net = Network(height="900px", width="100%", notebook=False)
net.from_nx(G)
net.show("./knowledge_graph.html", notebook=False)