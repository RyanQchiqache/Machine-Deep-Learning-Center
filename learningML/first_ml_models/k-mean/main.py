from tfidf import TFIDFVectorizer
from kmean import KMean
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull
import seaborn as sns


def get_documents():
    """Returns a dataset of documents grouped into various topics."""
    return [
        # NLP & AI
        "Natural language processing enables machines to understand human speech.",
        "Text classification is a fundamental NLP task with applications in sentiment analysis.",
        "Machine learning algorithms improve natural language understanding models.",
        "Deep learning techniques like transformers revolutionized NLP.",
        "Neural networks are the backbone of deep learning and AI systems.",
        "Artificial intelligence is driving innovation in many industries.",
        "Self-supervised learning helps train deep neural networks with less labeled data.",
        "Reinforcement learning is used in robotics and autonomous systems.",
        "Machine translation allows people to communicate across languages using AI.",
        "Speech recognition systems convert spoken words into text using deep learning.",
        "AI-driven chatbots improve customer service and automate responses.",
        "Autonomous vehicles rely on AI and sensor data for navigation.",

        # Quantum Computing
        "Quantum computing is an emerging technology with applications in cryptography.",
        "Quantum mechanics influences the development of quantum algorithms.",
        "Quantum computers can potentially break classical encryption methods.",
        "Superconducting qubits are used in advanced quantum circuits.",
        "Quantum entanglement enables instant communication between particles.",
        "Quantum computing could solve problems beyond classical computation.",

        # Sports & Basketball
        "I love sports and I wish to be in a basketball game.",
        "One day I will be a basketball player, and I will play in the NBA.",
        "Since I was young, my dream was to play basketball.",
        "LeBron James is one of the greatest basketball players of all time.",
        "Football is a globally popular sport played in many countries.",
        "The FIFA World Cup is the most-watched sporting event worldwide.",
        "Tennis requires agility, precision, and strategic play to win matches.",
        "The Olympics bring together athletes from around the world in various sports.",

        # Space & Astronomy
        "NASA has launched several missions to explore Mars and other planets.",
        "Black holes are regions in space where gravity is so strong that nothing can escape.",
        "The James Webb Space Telescope is expected to revolutionize astronomy.",
        "Scientists have discovered thousands of exoplanets in our galaxy.",
        "The moon landing in 1969 was a significant milestone in space exploration.",
        "Astrophysics studies the fundamental laws governing the universe.",
        "Neutron stars are the remnants of supernova explosions.",

        # Medicine & Biology
        "CRISPR technology is used for gene editing and genetic research.",
        "Vaccines help prevent the spread of infectious diseases globally.",
        "The human genome project mapped all the genes in human DNA.",
        "Advancements in biotechnology are leading to breakthroughs in medicine.",
        "Stem cell research has the potential to treat various medical conditions.",
        "The study of microbiology helps us understand bacteria and viruses.",
        "Cancer research aims to develop new treatments for the disease."
    ]


def preprocess_documents(documents):
    """Converts text into a TF-IDF matrix and applies PCA for dimensionality reduction."""
    print(" Converting text into TF-IDF matrix...")
    vectorizer = TFIDFVectorizer()
    tf_idf_matrix = np.nan_to_num(vectorizer.fit_transform(documents))  # Handle NaN values

    print(" Applying PCA for dimensionality reduction...")
    n_components = min(10, tf_idf_matrix.shape[1])
    pca = PCA(n_components=n_components)
    reduced_tf_idf = pca.fit_transform(tf_idf_matrix)

    return reduced_tf_idf


def apply_clustering(X, k=6):
    """Performs K-Means clustering on the reduced TF-IDF matrix."""
    print(f" Running K-Means clustering with k={k}...")
    kmeans = KMean(k=k)
    return kmeans.fit(X)





def visualize_clusters_2d(X, clusters):
    """Uses t-SNE to visualize clusters in 2D space with convex hulls."""
    print(" Applying t-SNE for 2D visualization...")
    tsne = TSNE(n_components=2, perplexity=min(10, len(X) - 1), random_state=42)
    tsne_results = tsne.fit_transform(X)

    # Convert to NumPy array
    tsne_results = np.array(tsne_results)
    clusters = np.array(clusters)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Unique cluster labels
    unique_clusters = np.unique(clusters)

    # Assign colors from seaborn palette
    palette = sns.color_palette("husl", len(unique_clusters))

    for i, cluster in enumerate(unique_clusters):
        # Get points belonging to the cluster
        points = tsne_results[clusters == cluster]

        # Scatter plot for points
        ax.scatter(points[:, 0], points[:, 1], color=palette[i], alpha=0.7, edgecolors='k', label=f"Cluster {cluster}")

        # Convex Hull for 2D
        if len(points) > 2:  # Convex hull requires at least 3 points
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            ax.fill(hull_points[:, 0], hull_points[:, 1], color=palette[i], alpha=0.2)

    # Labels and aesthetics
    ax.set_title("2D t-SNE Visualization of Clusters")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend()

    plt.show()


def visualize_clusters_3d(X, clusters):
    """Uses t-SNE to visualize clusters in 3D space using Plotly."""
    print("Applying t-SNE for 3D visualization...")
    tsne = TSNE(n_components=3, perplexity=min(10, len(X) - 1), random_state=42)
    tsne_results = tsne.fit_transform(X)

    # Convert to NumPy array
    tsne_results = np.array(tsne_results)
    clusters = np.array(clusters)

    # Create a Plotly 3D scatter plot
    fig = px.scatter_3d(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        z=tsne_results[:, 2],
        color=clusters.astype(str),  # Convert cluster numbers to strings for better visualization
        title="3D t-SNE Visualization of Document Clusters",
        labels={'x': 't-SNE Dim 1', 'y': 't-SNE Dim 2', 'z': 't-SNE Dim 3'},
        opacity=0.8
    )

    fig.update_traces(marker=dict(size=6, line=dict(width=1)))
    fig.show()


def print_cluster_assignments(documents, clusters):
    """Displays document cluster assignments."""
    print("\n📌 Cluster Assignments:")
    for i, doc in enumerate(documents):
        print(f" Document {i} (Cluster {clusters[i]}): {doc}")


def main():
    """Main function to execute the full NLP clustering pipeline."""
    print(" Starting NLP Document Clustering Pipeline...\n")

    documents = get_documents()
    reduced_tf_idf = preprocess_documents(documents)
    clusters = apply_clustering(reduced_tf_idf)

    print_cluster_assignments(documents, clusters)

    # Visualizing 2D and 3D separately
    visualize_clusters_2d(reduced_tf_idf, clusters)
    visualize_clusters_3d(reduced_tf_idf, clusters)

    print("\n Clustering pipeline completed!")


if __name__ == '__main__':
    main()

