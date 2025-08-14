

# Transformer Basic

This repository contains a fundamental, from-scratch implementation of the Transformer decoder block in PyTorch. It is designed to be a clear, modular, and educational resource for understanding the core components of the Transformer architecture along with various other optimisations.

---

## Features

-   **Modular Components**: The architecture is broken down into logical modules: `attention`, `MHA` (Multi-Head Attention), and `normalize`, making it easy to understand each part individually.
-   **Standard Transformer decoder**: Implements a complete Transformer decoder block, including Multi-Head Self-Attention, residual connections, layer normalization, and a position-wise feed-forward network.
-   **Pure PyTorch**: Built entirely using standard PyTorch libraries, ensuring seamless integration into any PyTorch-based project.
-   **Customizable Dimensions**: Flexible aPI allows for easy configuration of model dimensions, number of heads, and dropout rates.

---

## How It Works

The `transformer_basic` module decapsulates the logic of a single decoder block from the original "Attention Is All You Need" paper. This block processes an input sequence of embeddings and produces an output sequence of the same length.

### Core Components

1.  **Multi-Head Attention (`MHA`)**: This is the heart of the Transformer. Instead of performing a single attention function, it projects the queries, keys, and values into multiple "heads," or subspaces.
    -   **Parallel Attention**: Attention is computed independently in each head, allowing the model to jointly attend to information from different representational subspaces at different positions.
    -   **Scaled Dot-Product Attention**: The underlying `attention` module computes scores by taking the dot product of the query and key, scaling the result, applying an optional mask, and converting the scores to probabilities using a softmax function.

2.  **Add & Norm (Residuals and Normalization)**: Each sub-layer (Multi-Head Attention and the Feed-Forward Network) is wrapped with a residual connection followed by layer normalization.
    -   **Residual Connections**: The input to a sub-layer is added to its output (`x + sublayer(x)`). This helps prevent the vanishing gradient problem and allows for the construction of much deeper models.
    -   **Layer Normalization**: Normalizes the features across the embedding dimension to stabilize the network and speed up training.

3.  **Position-wise Feed-Forward Network (`FFNN`)**: A simple, fully connected feed-forward network applied to each position separately and identically. It consists of two linear transformations with a ReLU activation in between, providing additional non-linearity to the model.

---

## Installation

The only dependency is PyTorch. You can install it via pip:

```bash
pip install torch
```

---

## Usage

### Using the Transformer Block

You can easily import and use the `transformers` class in your own projects. Simply instantiate it with your desired dimensions and pass your input embeddings through it.

### Example

```python
import torch
from transformer_basic import transformers # Assuming your file is named transformer_basic.py

# 1. Define model hyperparameters
batch_size = 16
seq_length = 50
d_model = 512       # The dimension of the embeddings
heads = 8           # The number of attention heads
q_dim = 512         # The dimension for Q, K, V projections (must be divisible by heads)
dropout = 0.1       # Dropout rate

# 2. Create a dummy input tensor
# (batch_size, sequence_length, embedding_dimension)
input_embeddings = torch.randn(batch_size, seq_length, d_model)

# 3. Instantiate the Transformer decoder block
transformer_block = transformers(
    d_model=d_model,
    heads=heads,
    Q_dim=q_dim,
    dropout=dropout
)

# 4. Pass the input through the model
output = transformer_block(input_embeddings)

# 5. Print the output shape
print("Input Shape:", input_embeddings.shape)
print("Output Shape:", output.shape)
# Expected Output:
# Input Shape: torch.Size([16, 50, 512])
# Output Shape: torch.Size([16, 50, 512])
```

### API Parameters

The main `transformers` class takes the following arguments:

-   `d_model`: The dimensionality of the input and output embeddings for the model (integer).
-   `heads`: The number of parallel attention heads to use in the Multi-Head Attention module (integer).
-   `Q_dim`: The dimensionality of the query, key, and value projection vectors. **This value must be divisible by `heads`** (integer).
-   `dropout`: The dropout probability to be used in the attention mechanism and after each sub-layer (float, optional).

---

## License

This project is licensed under the MIT License.
