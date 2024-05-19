import torch
from Model import GPT_MODEL
from Utils import text_to_token, token_to_text, generate_text

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 256,  # Context length
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False  # Query-Key-Value bias
}
checkpoint = torch.load("model_and_optimizer.pth")
model = GPT_MODEL(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.to('cpu')
print("Model name:", model.__class__.__name__)

torch.manual_seed(123)
token_ids = generate_text(
    model=model,
    input_ids=text_to_token("Hello world, my name is Jiarun Lu"),
    max_new_tokens=50,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)
print("Output text:\n", token_to_text(token_ids))