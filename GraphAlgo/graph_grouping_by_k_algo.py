
import networkx as nx
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

def get_clusters_by_exact_distance(graph, start_node):
    """
    Divide a NetworkX graph into clusters based on exact distances from a starting node.

    Parameters:
        graph (nx.Graph): The input graph.
        start_node: The starting node for clustering.

    Returns:
        clusters (list): A list of clusters, where each cluster is represented as a dictionary
                        containing nodes grouped by their distance from the starting node.
    """
    clusters = []
    visited = set()
    queue = [(start_node, 0)]

    while queue:
        node, distance = queue.pop(0)

        if node not in visited:
            visited.add(node)

            # Add the node to the appropriate cluster
            while len(clusters) < distance + 1:
                clusters.append({})

            # Check if the current cluster exceeds 100 nodes
            while len(clusters[distance]) >= 100:
                distance += 1
                if len(clusters) < distance + 1:
                    clusters.append({})

            # Add node to the current cluster
            clusters[distance][node] = True

            # Explore neighbors
            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    queue.append((neighbor, distance + 1))

    return clusters


def visualize_distance(graph, start_node, distance):
    """
    Visualize nodes in the graph that are at a specific distance from the starting node.

    Parameters:
        graph (nx.Graph): The input graph.
        start_node: The starting node for visualization.
        distance: The specific distance from the starting node to visualize.
    """
    nodes_at_distance = set(nx.single_source_shortest_path_length(graph, start_node, cutoff=distance).keys())

    pos = nx.spring_layout(graph)  # Layout for visualization
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=500)

    # Highlight nodes at the specified distance
    nx.draw_networkx_nodes(graph, pos, nodelist=list(nodes_at_distance), node_color='salmon', node_size=500)

    plt.title(f"Nodes at distance {distance} from node {start_node}")
    plt.show()

def calculate_distances(graph, start_node, recommended_nodes):
    distances = {}
    for node in recommended_nodes:
        distance = nx.shortest_path_length(graph, source=start_node, target=node)
        distances[node] = distance
    return distances

def exponential_mechanism_probabilities(cluster_sizes, epsilon):
    """
    Compute probabilities using exponential mechanism.

    Parameters:
        cluster_sizes (list): List of cluster sizes.
        epsilon (float): Sensitivity parameter.

    Returns:
        sampled_index (int): Index sampled using exponential mechanism.
    """
    probabilities = [0.5 ** (i-1) for i in range(len(cluster_sizes))]
    scaled_probabilities = [np.exp(epsilon * p / 2) for p in probabilities]
    total = sum(scaled_probabilities)
    normalized_probabilities = [p / total for p in scaled_probabilities]
    return normalized_probabilities

def sample_node_from_cluster(cluster):
    """
    Sample a node uniformly from the given cluster.

    Parameters:
        cluster (dict): Cluster dictionary containing nodes.

    Returns:
        node: Sampled node.
    """
    return random.choice(list(cluster.keys()))

def sample_nodes_from_clusters_non_private(clusters, total_samples=10):
    """
    Sample nodes uniformly from all clusters in the non-private approach.

    Parameters:
        clusters (list): List of cluster dictionaries containing nodes.
        total_samples (int): Total number of samples to be returned.

    Returns:
        sampled_nodes (list): List of sampled nodes from all clusters.
    """
    sampled_nodes = []
    remaining_samples = total_samples

    for cluster in clusters:
        if remaining_samples == 0:
            break

        num_samples_current_cluster = min(remaining_samples, len(cluster))
        sampled_nodes.extend(random.sample(list(cluster.keys()), num_samples_current_cluster))
        remaining_samples -= num_samples_current_cluster

    return sampled_nodes

def calculate_accuracy(distances_private):
    """
    Calculate the Root Mean Squared Error (RMSE) between distances obtained from private and non-private approaches.

    Parameters:
        distances_private (dict): Distances obtained from the private approach.
        distances_non_private (dict): Distances obtained from the non-private approach.

    Returns:
        rmse (float): Root Mean Squared Error.
    """
    squared_errors = [(distances_private[node] -1)**2 for node in distances_private]
    mean_squared_error = sum(squared_errors) / len(squared_errors)
    rmse = np.sqrt(mean_squared_error)
    return rmse

 # Function to visualize clusters based on level
def visualize_clusters_by_level(graph, clusters):
        pos = nx.spring_layout(graph)  # Layout for visualization

        # Define a color map for different levels
        color_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        # Iterate over each cluster
        for i, cluster in enumerate(clusters):
            level = i + 1  # Level starts from 1
            nodes = list(cluster.keys())
            nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_color=color_map[i % len(color_map)],
                                   label=f'Level {level}')
            nx.draw_networkx_edges(graph, pos, alpha=0.5)

        plt.title("Clusters formed based on level")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    file_path = "facebook_combined.txt/facebook_combined.txt"  # Provide the path to your dataset file
    connections = read_dataset(file_path)
    G = create_graph(connections)

    # Specify the starting node and maximum distance
    start_node = '0'
    max_distance = 1

    visualize_distance(G, start_node, 10)

    # Get clusters based on exact distances from the starting node
    clusters = get_clusters_by_exact_distance(G, start_node)
    clusters = clusters[1:]

    # Print the clusters
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1}: {list(cluster.keys())}")

    # Compute exponential mechanism probabilities
    cluster_sizes = [len(cluster) for cluster in clusters]

    visualize_clusters_by_level(G, clusters)

    epsilon= 5
    probabilities = exponential_mechanism_probabilities(cluster_sizes,epsilon)

    # Sample nodes according to probabilities
    sampled_nodes = []
    sampled_nodes_non_private=[]
    while len(sampled_nodes) < 10:
        index = random.choices(range(len(probabilities)), probabilities)[0]
        node = sample_node_from_cluster(clusters[index])
        sampled_nodes.append(node)

    sampled_nodes_non_private=sample_nodes_from_clusters_non_private(clusters)


    distances = calculate_distances(G, '0', sampled_nodes)
    for node,distance in distances.items():
        print(f"Distance from node {'0'} to node {node}: {distance}")

    print("Sampled nodes:", sampled_nodes)

    # Distances in the private approach
    distances_private = calculate_distances(G, '0', sampled_nodes)

    # Distances in the non-private approach (baseline)
    sampled_nodes_non_private = sample_nodes_from_clusters_non_private(clusters)

    print(distances_private)

    # Calculate RMSE
    rmse = calculate_accuracy(distances_private)
    print("RMSE between private and non-private approaches:", rmse)

    epsilon_values = []
    rmse_values = []

    # First loop from 0 to 2 with step 0.01
    for epsilon in np.arange(0, 2, 0.1):
        cluster_sizes = [len(cluster) for cluster in clusters]
        probabilities = exponential_mechanism_probabilities(cluster_sizes, epsilon)

        sampled_nodes = []
        while len(sampled_nodes) < 10:
            index = random.choices(range(len(probabilities)), probabilities)[0]
            node = sample_node_from_cluster(clusters[index])
            sampled_nodes.append(node)

        distances_private = calculate_distances(G, start_node, sampled_nodes)
        rmse = calculate_accuracy(distances_private)

        epsilon_values.append(epsilon)
        rmse_values.append(rmse)

    # Second loop from 2 to 3 with step 0.1
    for epsilon in np.arange(2, 3, 0.1):
        cluster_sizes = [len(cluster) for cluster in clusters]
        probabilities = exponential_mechanism_probabilities(cluster_sizes, epsilon)

        sampled_nodes = []
        while len(sampled_nodes) < 10:
            index = random.choices(range(len(probabilities)), probabilities)[0]
            node = sample_node_from_cluster(clusters[index])
            sampled_nodes.append(node)

        distances_private = calculate_distances(G, start_node, sampled_nodes)
        rmse = calculate_accuracy(distances_private)

        if(rmse<0):
            print(epsilon)

        epsilon_values.append(epsilon)
        rmse_values.append(rmse)

    # Third loop from 3 to 6 with step 0.2
    for epsilon in np.arange(3, 20, 0.2):
        cluster_sizes = [len(cluster) for cluster in clusters]
        probabilities = exponential_mechanism_probabilities(cluster_sizes, epsilon)

        sampled_nodes = []
        while len(sampled_nodes) < 10:
            index = random.choices(range(len(probabilities)), probabilities)[0]
            node = sample_node_from_cluster(clusters[index])
            sampled_nodes.append(node)

        distances_private = calculate_distances(G, start_node, sampled_nodes)
        rmse = calculate_accuracy(distances_private)

        epsilon_values.append(epsilon)
        rmse_values.append(rmse)

    # Fit a polynomial curve
    degree = 2  # You can adjust the degree of the polynomial
    coefficients = np.polyfit(epsilon_values, rmse_values, degree)

    # Generate polynomial function using the coefficients
    polynomial = np.poly1d(coefficients)

    # Generate values for epsilon to plot the polynomial curve
    epsilon_values_curve = np.linspace(min(epsilon_values), max(epsilon_values), 100)
    rmse_values_curve = polynomial(epsilon_values_curve)

    # Plotting scatter plot and polynomial curve
    plt.scatter(epsilon_values, rmse_values, marker='o', label='Data')
    plt.plot(epsilon_values_curve, rmse_values_curve, label='Polynomial Fit', color='red')
    plt.title('RMSE vs Epsilon')
    plt.xlabel('Epsilon')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.legend()
    plt.show()


