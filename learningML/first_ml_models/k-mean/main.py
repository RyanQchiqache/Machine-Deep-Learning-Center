from tfidf import TFIDFVectorizer
from kmean import KMean
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


def main():
    """Main function to perform TF-IDF transformation, PCA, K-Means clustering, and visualization."""

    # 📝 Expanded Corpus with Diverse Topics
    documents = [
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

    #  **Step 1: Convert Text to TF-IDF Matrix**
    vectorizer = TFIDFVectorizer()
    tf_idf_matrix = vectorizer.fit_transform(documents)

    # **Step 2: Fix potential NaN values (important for PCA)**
    if np.isnan(tf_idf_matrix).any():
        print("⚠️ Warning: NaN values detected in TF-IDF matrix. Replacing with 0.")
        tf_idf_matrix = np.nan_to_num(tf_idf_matrix)

    # Step 3: Reduce Dimensionality with PCA**
    n_components = min(10, tf_idf_matrix.shape[1])  # Ensure PCA components ≤ features
    pca = PCA(n_components=n_components)
    reduced_tf_idf = pca.fit_transform(tf_idf_matrix)

    # **Step 4: Apply K-Means Clustering**
    kmeans = KMean(k=6)  # Adjust based on number of topics
    clusters = kmeans.fit(reduced_tf_idf)

    # **Step 5: Apply t-SNE for Visualization**
    perplexity = min(10, len(documents) - 1)  # Ensure valid perplexity
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_results = tsne.fit_transform(reduced_tf_idf)

    # **Step 6: Plot Clusters**
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=clusters, cmap='rainbow', alpha=0.7)
    plt.title("t-SNE Visualization of Document Clusters")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.colorbar(scatter, label="Cluster ID")
    plt.show()

    # **Step 7: Print Cluster Assignments**
    for i, doc in enumerate(documents):
        print(f" Document {i} (Cluster {clusters[i]}): {doc}")


# 🔥 Run the main function
if __name__ == '__main__':
    main()
