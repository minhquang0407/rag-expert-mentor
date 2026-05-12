# Chapter 1: Foundations of Graph Theory

## Graph Representations
A graph $G = (V, E)$ consists of a set of vertices $V$ and edges $E$. Graphs can be represented in several ways:
- **Adjacency Matrix** $A$: A square matrix where $A_{ij} = 1$ if there is an edge between vertex $i$ and $j$, otherwise $A_{ij} = 0$. For weighted graphs, $A_{ij}$ stores the edge weight.
- **Adjacency List**: Each vertex maintains a list of its neighbors. More memory-efficient for sparse graphs with $O(V + E)$ space complexity.
- **Edge List**: A flat list of $(u, v)$ pairs. Simple but slow for neighbor lookups.

The **degree matrix** $D$ is a diagonal matrix where $D_{ii} = \sum_j A_{ij}$ represents the number of edges connected to vertex $i$. The **Laplacian matrix** is defined as $L = D - A$, which plays a central role in spectral graph theory and graph signal processing.

## Spectral Graph Theory
The eigenvalues and eigenvectors of the graph Laplacian $L$ reveal fundamental structural properties of the graph. The Laplacian can be decomposed as $L = U \Lambda U^T$ where $U$ is the matrix of eigenvectors and $\Lambda$ is the diagonal matrix of eigenvalues.

The smallest eigenvalue is always 0 (corresponding to the constant eigenvector), and the number of zero eigenvalues equals the number of connected components. The second-smallest eigenvalue $\lambda_2$ (Fiedler value) measures graph connectivity — larger values indicate more connected graphs.

**Graph Fourier Transform**: Given a signal $x$ on the graph vertices, its graph Fourier transform is $\hat{x} = U^T x$, and the inverse transform is $x = U \hat{x}$. This provides the theoretical foundation for designing graph convolution operations in the spectral domain.

# Chapter 2: Graph Neural Networks (GNN)

## Message Passing Framework
The core idea behind most GNN architectures is the **message passing** paradigm. At each layer $l$, every node $v$ updates its feature vector by:

1. **Aggregate**: Collect messages from neighboring nodes $\mathcal{N}(v)$:
$$m_v^{(l)} = \text{AGGREGATE}(\{h_u^{(l-1)} : u \in \mathcal{N}(v)\})$$

2. **Update**: Combine the aggregated message with the node's own features:
$$h_v^{(l)} = \text{UPDATE}(h_v^{(l-1)}, m_v^{(l)})$$

Common aggregation functions include SUM, MEAN, and MAX. The choice of aggregation directly affects the model's expressiveness — Xu et al. (2019) proved that SUM aggregation is the most expressive, as it can distinguish multisets of neighbor features.

**Over-smoothing Problem**: As GNN depth increases, node representations converge to indistinguishable values. This limits practical GNN models to 2-4 layers. Solutions include residual connections (like ResGCN), jumping knowledge networks, and DropEdge regularization.

## Graph Convolutional Network (GCN)
Kipf & Welling (2017) introduced GCN as a simplified spectral approach. The layer-wise propagation rule is:

$$H^{(l+1)} = \sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W^{(l)})$$

Where:
- $\tilde{A} = A + I_N$ is the adjacency matrix with added self-loops
- $\tilde{D}$ is the corresponding degree matrix
- $W^{(l)}$ is a trainable weight matrix
- $\sigma$ is a nonlinear activation function (typically ReLU)

The symmetric normalization $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ prevents feature explosion for high-degree nodes and ensures numerical stability during training.

## Graph Attention Network (GAT)
Veličković et al. (2018) introduced attention mechanisms to GNNs. Instead of treating all neighbors equally, GAT computes attention coefficients:

$$\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(a^T[Wh_i \| Wh_j]))}{\sum_{k \in \mathcal{N}(i)} \exp(\text{LeakyReLU}(a^T[Wh_i \| Wh_k]))}$$

The final node representation uses multi-head attention for stability:
$$h_i' = \|_{k=1}^{K} \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^k W^k h_j\right)$$

**Advantages over GCN**: GAT does not require knowing the full graph structure upfront (transductive), can handle dynamic graphs, and learns edge importance implicitly.

# Chapter 3: Applications of GNN

## Node Classification
The most common GNN task: predict labels for unlabeled nodes using labeled nodes and graph structure. Applications include:
- **Citation Networks**: Classify academic papers into research topics (Cora, CiteSeer, PubMed datasets)
- **Social Networks**: Detect bot accounts, predict user interests
- **Fraud Detection**: Identify fraudulent transactions in financial networks

Training uses semi-supervised learning: only a small fraction of nodes have labels, but the model leverages the entire graph structure for propagation.

## Link Prediction
Predict missing or future edges in a graph. Critical for:
- **Recommendation Systems**: Predict user-item interactions (collaborative filtering enhanced with graph structure)
- **Knowledge Graph Completion**: Predict missing relations between entities (TransE, DistMult, RotatE)
- **Drug Discovery**: Predict drug-target interactions using molecular graphs

Common approach: encode nodes with GNN, then score candidate edges using dot product, bilinear form, or learned decoders.

## Graph Classification
Classify entire graphs rather than individual nodes. Requires a **readout** (pooling) operation to produce a fixed-size graph-level representation:
- **Global Mean/Sum Pooling**: Simple average or sum of all node features
- **Hierarchical Pooling** (DiffPool, SAGPool): Learn to coarsen the graph progressively
- **Set2Set, SortPool**: More sophisticated permutation-invariant pooling

Applications: molecular property prediction (chemistry), program analysis (code graphs), document classification (syntax trees).
