import networkx as nx
import matplotlib.pyplot as plt
import random
from scipy.sparse import lil_matrix
import numpy as np


# Read the dataset
def read_dataset(file_path):
    connections = []
    with open(file_path, 'r') as file:
        for line in file:
            point1, point2 = line.strip().split()  # Assuming points are space-separated
            connections.append((point1, point2))
    return connections

# Create a graph
def create_graph(connections):
    G = nx.Graph()

    # Add nodes
    for connection in connections:
        G.add_node(connection[0])
        G.add_node(connection[1])

    # Add edges
    G.add_edges_from(connections)

    return G

# Visualize the graph
def visualize_graph(G):
    pos = nx.spring_layout(G)  # Positions for all nodes
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='k', linewidths=1, font_size=10)
    plt.title("Graph of Connections")
    plt.show()

# Function to calculate normalized PageRank values based on neighbors
def normalized_pagerank(graph, node, pagerank):
    neighbors = list(graph.neighbors(node))
    sum_pagerank = sum(pagerank[neighbor] for neighbor in neighbors)
    if sum_pagerank == 0:
        return {neighbor: 1/len(neighbors) for neighbor in neighbors}
    return {neighbor: pagerank[neighbor] / sum_pagerank for neighbor in neighbors}

# Function to traverse the graph from a given node
def traverse_graph(graph, start_node, num_friends, pagerank):
    friends = []
    current_node = start_node
    while len(friends) < num_friends:
        neighbors = list(graph.neighbors(current_node))
        if not neighbors:
            break
        neighbor_pagerank = normalized_pagerank(graph, current_node, pagerank)
        next_node = random.choices(neighbors, weights=[neighbor_pagerank[n] for n in neighbors], k=1)[0]
        friends.append(next_node)
        current_node = next_node
    return friends


def graph_to_sparse_matrix(G):
    num_nodes = len(G.nodes())
    adjacency_matrix = lil_matrix((num_nodes, num_nodes), dtype=int)

    for edge in G.edges():
        # Convert node indices to integers
        node1 = int(edge[0])
        node2 = int(edge[1])

        adjacency_matrix[node1, node2] = 1
        adjacency_matrix[node2, node1] = 1  # For undirected graph, add symmetrically

    return adjacency_matrix

def power_of_sparse_matrix(A, n):
    if n == 0:
        return lil_matrix(np.eye(A.shape[0]))
    elif n % 2 == 0:
        half_power = power_of_sparse_matrix(A, n // 2)
        return half_power.dot(half_power)
    else:
        return A.dot(power_of_sparse_matrix(A, n - 1))


def weighted_sum(G, num_steps):
    sparse_matrix = graph_to_sparse_matrix(G)
    weights = [1 / 10 ** i for i in range(1, num_steps + 1)]
    total_weighted_sum = [0] * 4001

    for i in range(1, num_steps + 1):
        powered_matrix = power_of_sparse_matrix(sparse_matrix, i)
        print(powered_matrix.toarray())
        for target_node in range(1, 4001):
            num_paths = powered_matrix[0, target_node]
            total_weighted_sum[target_node] += (num_paths * weights[i - 1])

    return total_weighted_sum


def exponential_mechanism(scores, epsilon):
    max_score = max(scores)

    probabilities = [np.exp(epsilon * (score) / (2*max_score)) for score in scores]  # Exponential mechanism probabilities
    total_prob = sum(probabilities)
    normalized_probs = [prob / total_prob for prob in probabilities]  # Normalize probabilities
    sampled_index = np.random.choice(len(scores), p=normalized_probs)  # Sample index based on probabilities
    return sampled_index

if __name__ == "__main__":
    file_path = "facebook_combined.txt/facebook_combined.txt"  # Provide the path to your dataset file
    connections = read_dataset(file_path)
    G = create_graph(connections)

    # # Define a function to recursively find neighbors
    # def get_neighbors_recursive(graph, node, level, visited=None):
    #     if visited is None:
    #         visited = set()
    #     if level == 0:
    #         return [node]
    #     if node not in visited:
    #         visited.add(node)
    #         neighbors = [node]
    #         for neighbor in graph.neighbors(node):
    #             neighbors += get_neighbors_recursive(graph, neighbor, level - 1, visited)
    #         return neighbors
    #     return []
    #
    #
    # # Define a function to generate the subgraph
    # def generate_subgraph(graph, node, levels):
    #     subgraph_nodes = set()
    #     for level in range(levels + 1):
    #         subgraph_nodes.update(get_neighbors_recursive(graph, node, level))
    #     return graph.subgraph(subgraph_nodes)
    #
    #
    # # Generate the subgraph centered around node 0 with 2 levels of neighbors
    # subgraph = generate_subgraph(G, '0', 1)
    #
    # # Draw the subgraph
    # pos = nx.spring_layout(subgraph)  # Layout for visualization
    # nx.draw(subgraph, pos, with_labels=True, node_color='skyblue', node_size=1000, font_size=12)
    # plt.title('Subgraph centered around node 0 with 2 levels of neighbors')
    # plt.show()

    # # visualize_graph(G)
    #
    sparse_matrix = graph_to_sparse_matrix(G)
    print(sparse_matrix.toarray())
    #
    # # for i in range(11):
    # #     result = power_of_sparse_matrix(sparse_matrix, i)
    # #     print(f"Power {i}:\n{result.toarray()}\n")
    #
    num_steps = 5
    epsilon = 15
    scores = weighted_sum(G, num_steps)
    print(scores)
    print(sorted(scores,reverse=True))
    #
    for _ in range(10):
        sampled_node_index = exponential_mechanism(scores, epsilon)
        print("Sampled node index:", sampled_node_index)
        print(scores[sampled_node_index])

        # Remove the sampled node from the scores list
        scores[sampled_node_index] = -float('inf')

#     # Calculate PageRank
#     pagerank = nx.pagerank(G)
#
#     # Print PageRank values
#     print("PageRank values:")
#     for node, pr in pagerank.items():
#         print(f"{node}: {pr}")
#     #
#     # # Visualize the graph
#     # pos = nx.spring_layout(G)
#     # nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='k', linewidths=1, font_size=10)
#     # plt.title("Graph with PageRank values")
#     # plt.show()
#
#     # Select top 5 users randomly
#     top_users = random.sample(sorted(pagerank.keys(), key=lambda x: pagerank[x], reverse=True), 5)
#
#     print("Accurate top_users PageRank values:")
#     for user in top_users:
#         print(f"{user}")
#
#     # Add exponential noise to PageRank scores
#     epsilon = 0.1  # Privacy parameter
#     noisy_pagerank = {user: pagerank[user] + np.random.exponential(1 / epsilon) for user in top_users}
#
#     # Print noisy PageRank values
#     print("Noisy PageRank values:")
#     for user, pr in noisy_pagerank.items():
#         print(f"{user}: {pr}")
#
# #start with person 0 and find friends based on normalized PageRank values
#     start_node = '0'
#     chosen_friends = []
#
#     for _ in range(10):
#         # Traverse the graph to find friends
#         friends_list = traverse_graph(G, start_node, pagerank)
#
#         # Assign weights based on position in the list
#         weights = [2 ** (len(friends_list) - i - 1) for i in range(len(friends_list))]
#
#         # Differential privacy parameters
#         epsilon = 0.1
#         sensitivity = max(weights)
#
#         # Apply exponential mechanism
#         chosen_index = exponential_mechanism(weights, sensitivity, epsilon)
#         chosen_friend = friends_list[chosen_index]
#         chosen_friends.append(chosen_friend[0])  # Append the friend only
#
#         # Remove chosen friend and its subtree
#         removal_node = chosen_friend[0]
#         parent_node = #parent of removal node
#
#         # Update neighbors of parent node
#         parentnew_neighbors = list(G.neighbors(removal_node))
#         for neighbor in parentnew_neighbors:
#             if neighbor != parent_node:
#                 G.add_edge(neighbor, parent_node)
#
#         G.remove_node(removal_node)
#
#
#     # Print selected friends
#     print("Selected friends:")
#     for friend in chosen_friends:
#         print(friend)
