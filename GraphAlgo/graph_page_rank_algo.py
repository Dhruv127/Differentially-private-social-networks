import networkx as nx
import math
import matplotlib.pyplot as plt
import random
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

def recursive_dfs_with_points(graph, start_node, iterations):
    points_dict = {node: 0 for node in graph.nodes()}

    for _ in range(iterations):
        visited = set()
        stack = [(start_node, 1)]
        while stack:
            node, depth = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            neighbors = list(graph.neighbors(node))
            np.random.shuffle(neighbors)  # Shuffle neighbors randomly
            for neighbor in neighbors:
                if neighbor not in visited:  # Explore only unvisited neighbors
                    points_dict[neighbor] += 1 / (2 ** depth)
                    stack.append((neighbor, depth + 1))
                    break

    return points_dict

def exponential_mechanism(points_dict, sensitivity, epsilon, num_samples=10):
    probabilities = {node: np.exp(epsilon * points / (2 * sensitivity)) for node, points in points_dict.items()}
    total = sum(probabilities.values())
    probabilities = {node: prob / total for node, prob in probabilities.items()}
    node_population = list(probabilities.keys())
    probabilities_list = list(probabilities.values())

    if len(node_population) < num_samples:
        return node_population  # Return all available nodes if there are fewer than num_samples
    else:
        sampled_nodes = np.random.choice(node_population, size=num_samples, p=probabilities_list, replace=False)
        return sampled_nodes.tolist()

def calculate_distances(graph, start_node, recommended_nodes):
    distances = {}
    for node in recommended_nodes:
        distance = nx.shortest_path_length(graph, source=start_node, target=node)
        distances[node] = distance
    return distances

def cluster_nodes(points_dict, num_clusters):
    # Sort nodes based on their scores
    sorted_nodes = sorted(points_dict.items(), key=lambda x: x[1], reverse=True)
    # print(sorted_nodes)

    # Determine the number of nodes in each cluster
    nodes_per_cluster = len(sorted_nodes) // num_clusters

    # Initialize clusters
    clusters = [[] for _ in range(num_clusters)]


    for idx, (node, score) in enumerate(sorted_nodes):
        cluster_idx = min(idx // nodes_per_cluster, num_clusters - 1)  # Ensure not to exceed num_clusters
        clusters[cluster_idx].append((node, score))

    # Calculate total score of all nodes
    total_score = sum(points_dict.values())

    # Assign score to each cluster as the sum of scores of nodes it contains
    cluster_scores = [sum(score for _, score in cluster) for cluster in clusters]

    # Store which node has which score
    node_score_mapping = {}
    for cluster_idx, cluster in enumerate(clusters):
        for node, score in cluster:
            node_score_mapping[node] = score

    return clusters, node_score_mapping, cluster_scores

def exponential_mechanism_cluster(cluster_scores, epsilon, num_samples):
    probabilities = np.exp(epsilon * np.array(cluster_scores))
    probabilities /= np.sum(probabilities)
    sampled_cluster_idx = np.random.choice(len(cluster_scores), size=num_samples, p=probabilities, replace=False)
    return sampled_cluster_idx.tolist()

if __name__ == "__main__":
    file_path = "facebook_combined.txt/facebook_combined.txt"  # Provide the path to your dataset file
    connections = read_dataset(file_path)
    G = create_graph(connections)

    start_node = '0'  # Choose any start node
    iterations = 1000  # Only one random DFS path
    points_dict = recursive_dfs_with_points(G, start_node, iterations)

    print(sorted(points_dict.values(), reverse=True))

    # Exponential mechanism parameters
    sensitivity = iterations/2 # Sensitivity of the points function
    epsilon = 50000  # Privacy parameter

    # # Sample 10 nodes using exponential mechanism
    # recommended_points = exponential_mechanism(points_dict, sensitivity, epsilon)

    # distances = calculate_distances(G, '0', recommended_points)
    # for node, distance in distances.items():
    #     print(f"Distance from node {'0'} to node {node}: {distance}")
    #
    # for node in recommended_points:
    #     print(f"Score for {node} is: {points_dict[node]}")
    #
    # print("Recommended points:", recommended_points)

    # # Sample 10 nodes using exponential mechanism
    # num_samples = 10
    # recommended_points = exponential_mechanism(points_dict, sensitivity, epsilon, num_samples)

    # Cluster nodes into equal-sized clusters with equal total scores
    num_clusters = 10
    clusters, node_score_mapping, cluster_scores = cluster_nodes(points_dict, 20)

    print(sorted(cluster_scores, reverse=True))
    # print(len(clusters))

    # Sample clusters using exponential mechanism
    sampled_cluster_idx = exponential_mechanism_cluster(cluster_scores, 1, 10)

    # Sample nodes from sampled clusters
    sampled_nodes = []
    for idx in sampled_cluster_idx:
        cluster = clusters[idx]
        sampled_node = random.choice(cluster)[0]
        sampled_nodes.append(sampled_node)

    # Output sampled nodes
    print("Sampled nodes:", sampled_nodes)

    for node in sampled_nodes:
        print(f"Score for {node} is: {points_dict[node]}")
