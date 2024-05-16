import torch
import tiktoken
from model import GPT_MODEL

def generate_text(model, input_ids, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        input_ids_current = input_ids[:, -context_size:]
        with torch.no_grad():
            logits = model(input_ids_current)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        input_ids_next = torch.argmax(probas, dim=-1, keepdim=True)
        input_ids = torch.cat((input_ids, input_ids_next), dim=1)
    return input_ids


start_context = "Hello, World"
tokenizer = tiktoken.get_encoding("gpt2")
encoded = tokenizer.encode(start_context)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False  # Query-Key-Value bias
}
torch.manual_seed(123)
model = GPT_MODEL(GPT_CONFIG_124M)
model.eval() # disable dropout

out = generate_text(
    model=model,
    input_ids=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output:", out)
print("Output length:", len(out[0]))

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)