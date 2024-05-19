# Core of the GPT model
![Example Image](./asset/structure.png)

### Transformer Block
- A Masked Multi-head Attention layer
- A Feed forward layer
- Both with Shortcut Connection and LayerNorm and Dropout

### LayerNorm 
- Training deep neural networks with many layers can sometimes prove challenging due to issues like vanishing or exploding gradients.
- implement layer normalization to improve the stability and efficiency of neural network training.
- The main idea behind layer normalization is to adjust the activations (outputs) of a neural network layer to have a mean of 0 and a variance of 1, also known as unit variance. This adjustment speeds up the convergence to effective weights and ensures consistent, reliable training.
![Example Image](./asset/layernorm.png)

### Feed Forward Block
- A linear layer passed to a GELU activation layer to a linear layer
- In the 124 million parameter GPT model, it receives the input batches with tokens that have an embedding size of 768 each via the GPT_CONFIG_124M dictionary where GPT_CONFIG_124M["emb_dim"] = 768.
![Example Image](./asset/feedforward.png)
- The FeedForward module we implemented in this section plays a crucial role in enhancing the model's ability to learn from and generalize the data. Although the input and output dimensions of this module are the same, it internally expands the embedding dimension into a higher-dimensional space through the first linear layer as illustrated in Figure 4.10. This expansion is followed by a non-linear GELU activation, and then a contraction back to the original dimension with the second linear transformation. Such a design allows for the exploration of a richer representation space.
![Example Image](./asset/feedforward1.png)

### Shortcut Connections
The concept behind shortcut connections, also known as skip or residual connections. Shortcut connections were proposed to mitigate the challenge of vanishing gradients.
A shortcut connection creates an alternative, shorter path for the gradient to flow through the network by skipping one or more layers, which is achieved by adding the 
output of one layer to the output of a later layer. This is why these connections are also known as skip connections. They play a crucial role in preserving the flow of 
the gradients during the backward pass in training.
![Example Image](./asset/ShortCut.png)

# A Masked Multi-head Attention Layer
### Self-Attention
- In self-attention, the "self" refers to the mechanism's ability to compute attention weights by relating different positions within a single input sequence. It assesses and learns the relationships and dependencies between various parts of the input itself.
- This is in contrast to traditional attention mechanisms, where the focus is on the relationships between elements of two different sequences.

### Computing the attention weights step by step
We will implement the self-attention mechanism step by step by introducing the three trainable weight matrices W_query_matrix, W_key_matrix, and W_value_matrix. These three matrices are used to project the
embedded input tokens, x(i), into query, key, and value vectors.

![Example Image](./asset/atten_w_step.png)

- keys_matrix = inputs_matrix @ W_key_matrix
- values_matrix = inputs_matrix @ W_value_matrix
- queries_matrix = inputs_matrix @ W_query_matrix
- attention_scores_matrix = queries_matrix @ keys_matrix.T
- attention_weights_matrix = torch.softmax (attention_scores_matrix / (keys_matrix.shape[-1])**0.5, dim=-1)
- context_matrix = attention_weights_matrix @ values_matrix
- context_vector_X = context_matrix[X]

### The Rationale Behind Scaled-Dot Product Attention
The reason for the normalization by the embedding dimension size is to improve the training performance by avoiding small gradients. 
For instance, when scaling up the embedding dimension, which is typically greater than thousand for GPT-like LLMs, large dot products can result 
in very small gradients during backpropagation due to the softmax function applied to them. As dot products increase, the softmax function 
behaves more like a step function, resulting in gradients nearing zero. These small gradients can drastically slow down learning or cause training to stagnate.
The scaling by the square root of the embedding dimension is the reason why this self-attention mechanism is also called scaled-dot product attention.

### Masked Attention
Causal attention, also known as masked attention, is a specialized form of self-attention. It restricts a model to only consider previous and current inputs in a sequence when processing any given token. 
![Example Image](./asset/MaskedAttention.png)
As illustrated in Figure 3.19, we mask out the attention weights above the diagonal, and we normalize the non-masked attention weights, such that the attention weights sum to 1 in each row.

### Masking Additional Attention Weights With Dropout
![Example Image](./asset/MaskingDropout.png)
When applying dropout to an attention weight matrix with a rate of 50%, half of the elements in the matrix are randomly set to zero. 
To compensate for the reduction in active elements, the values of the remaining elements in the matrix are scaled up by a factor of 1/0.5 =2. 
This scaling is crucial to maintain the overall balance of the attention weights, ensuring that the average influence of the attention mechanism remains consistent during both the training and inference phases.

### Extending Single-Head Attention To Multi-Head Attention
In practical terms, implementing multi-head attention involves creating multiple instances of the self-attention mechanism, each with its own weights, and then combining their outputs. 
It's crucial for the kind of complex pattern recognition that models like transformer-based LLMs are known for. 

To implement, splits the input into multiple heads by reshaping the projected query, key, and value tensors and then combines the results from these heads after computing attention.
![Example Image](./asset/Multi-Head.png)
