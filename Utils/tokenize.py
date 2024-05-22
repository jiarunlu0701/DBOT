import tiktoken
import torch
from transformers import BertTokenizer

tokenizer = tiktoken.get_encoding("gpt2")
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")


def text_to_token(text):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_to_text(input_ids):
    input_ids_flatten = input_ids.squeeze(0)
    return tokenizer.decode(input_ids_flatten.tolist())
