import pandas as pd
import pickle
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite

# Read train dataset (change path)
dataset_path = "./data/amazon/dataset.tsv"
dataset = pd.read_csv(dataset_path, sep="\t", header=None, names=["user", "item"])

# Comment until line 22 if normal dataset (not augmented)
# Read dictionary of new interactions and add them to the old dataset (change path)
user_interactions = pickle.load(open("./data/amazon_35_MMSSL/augmented_sample_dict_gpt35-16k_mmssl","rb"))
new_interactions = []
for user_id, items in sorted(user_interactions.items()):  # Sorting by user
    if 0 in items and not (np.isnan(items[0])):  # Check if liked item exists
        new_interactions.append([user_id, items[0]])
    if 1 in items and not (np.isnan(items[1])):  # Check if not-liked item exists
        new_interactions.append([user_id, items[1]])

new_interactions_df = pd.DataFrame(new_interactions, columns=["user", "item"])

dataset = pd.concat([dataset, new_interactions_df]).reset_index(drop=True)
dataset = dataset.sort_values(by=["user"]).reset_index(drop=True)

# Define items as "i_itemID" to make it bipartite
dataset['item'] = dataset['item'].astype(int)
dataset['item'] = 'i_' + dataset['item'].astype(str)

# Create graph
G = nx.Graph()
G.add_nodes_from(dataset["user"].unique(), bipartite=0)  # Users
G.add_nodes_from(dataset["item"].unique(), bipartite=1)  # Items
edges = list(zip(dataset["user"], dataset["item"])) # Edges
G.add_edges_from(edges)
print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

original_nodes = set(G.nodes())
original_edges = set(G.edges())

# Connected components
print(f"Is the graph connected? {nx.is_connected(G)}")
print(f"Number of connected components: {nx.number_connected_components(G)}")
print(f"Is the graph bipartite? {bipartite.is_bipartite(G)}")
print(f"Number of isolated nodes: {len(list(nx.isolates(G)))}")

# Keep only the largest component to use networkx
largest_cc = max(nx.connected_components(G), key=len)
G = G.subgraph(largest_cc).copy()
print(f"New graph size: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

remaining_nodes = set(G.nodes())
remaining_edges = set(G.edges())

removed_nodes = original_nodes - remaining_nodes
removed_edges = original_edges - remaining_edges

print(f"Removed {len(removed_nodes)} nodes: {removed_nodes}")
print(f"Removed {len(removed_edges)} edges: {removed_edges}")

with open("./data/amazon_35_MMSSL/output.txt", "w") as file:
    file.write(f"Removed {len(removed_nodes)} nodes: {removed_nodes}")
    file.write(f"Removed {len(removed_edges)} edges: {removed_edges}")

# Remove rows where either user or item is in removed_nodes
df_cleaned = dataset[~dataset["user"].isin(removed_nodes) & ~dataset["item"].isin(removed_nodes)]
df_cleaned.to_csv("./data/amazon_35_MMSSL/new_dataset.tsv", index=False, header=False, sep='\t')
print(f"Is now the graph connected? {nx.is_connected(G)}")
print(f"Number of new connected components: {nx.number_connected_components(G)}")


