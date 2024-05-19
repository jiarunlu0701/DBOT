import tiktoken
import torch

tokenizer = tiktoken.get_encoding("gpt2")


def text_to_token(text):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_to_text(input_ids):
    input_ids_flatten = input_ids.squeeze(0)
    return tokenizer.decode(input_ids_flatten.tolist())
