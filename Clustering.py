
"""
Clustering

 Reduces dimensionality with PCA
 Clusters with K-means
 Plots a 2D visualization so you can look at the clusters

"""

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class Cluster:

    def __init__(self, features):
        """features: (N, 2048) array from FeatureExtraction.py"""
        self.features = features

    # ---------------------------------------------------------------------------
    # Reduce dimensionality with PCA, then cluster with K-means
    # ---------------------------------------------------------------------------
    def _cluster_features(self, n_clusters=8, n_pca_components=50):
        """
            n_clusters: how many groups K-means should split the data into
            n_pca_components: how many dimensions to reduce down to before clustering
                               (2048 is too high-dimensional to cluster directly)

            Returns:
                cluster_labels: which cluster (0 to n_clusters-1) each row belongs to
                reduced: the PCA-reduced features, useful for further analysis
        """

        pca = PCA(n_components=n_pca_components)
        reduced = pca.fit_transform(self.features)

        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
        cluster_labels = kmeans.fit_predict(reduced)

        return cluster_labels, reduced

    # ---------------------------------------------------------------------------
    # Visualize the clusters in 2D
    # ---------------------------------------------------------------------------
    def _plot_clusters(self,reduced, cluster_labels, save_path="clusters.png"):
        embedding = TSNE(n_components=2, random_state=0).fit_transform(reduced)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            embedding[:, 0], embedding[:, 1],
            c=cluster_labels, cmap="tab10", s=5
        )
        plt.legend(*scatter.legend_elements(), title="Cluster")
        plt.title("MRNet slice feature clusters (RadImageNet ResNet-50 features)")
        plt.savefig(save_path, dpi=150)
        plt.show()
        print(f"Saved plot to {save_path}")

    # ---------------------------------------------------------------------------
    # Orchestration
    # ---------------------------------------------------------------------------
    def run(self):

        print("Clustering...")
        cluster_labels, reduced = self._cluster_features()

        print("Plotting...")
        self._plot_clusters(reduced, cluster_labels)


