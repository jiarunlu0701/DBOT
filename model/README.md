# Core of the GPT model
![Example Image](/Users/lujiarun/PycharmProjects/DBOT/asset/structure.png)
Jiarun
### Transformer Block
- A Masked Multi-head Attention layer
- A Feed forward layer
- Both with Shortcut Connection and LayerNorm and Dropout

### Self-Attention

### Masked Multi-Head-Attention Block

### LayerNorm 
- Training deep neural networks with many layers can sometimes prove challenging due to issues like vanishing or exploding gradients.
- implement layer normalization to improve the stability and efficiency of neural network training.
- The main idea behind layer normalization is to adjust the activations (outputs) of a neural network layer to have a mean of 0 and a variance of 1, also known as unit variance. This adjustment speeds up the convergence to effective weights and ensures consistent, reliable training.
![Example Image](/Users/lujiarun/PycharmProjects/DBOT/asset/layernorm.png)


### Feed Forward Block
- A linear layer passed to a GELU activation layer to a linear layer
- In the 124 million parameter GPT model, it receives the input batches with tokens that have an embedding size of 768 each via the GPT_CONFIG_124M dictionary where GPT_CONFIG_124M["emb_dim"] = 768.
![Example Image](/Users/lujiarun/PycharmProjects/DBOT/asset/feedforward.png)
- The FeedForward module we implemented in this section plays a crucial role in enhancing the model's ability to learn from and generalize the data. Although the input and output dimensions of this module are the same, it internally expands the embedding dimension into a higher-dimensional space through the first linear layer as illustrated in Figure 4.10. This expansion is followed by a non-linear GELU activation, and then a contraction back to the original dimension with the second linear transformation. Such a design allows for the exploration of a richer representation space.
![Example Image](/Users/lujiarun/PycharmProjects/DBOT/asset/feedforward1.png)
