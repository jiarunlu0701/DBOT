import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

tokenizer = tiktoken.get_encoding("gpt2")
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")


class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ids = []
        self.target_ids = []

        # Use a sliding window to chunk the text into overlapping sequences of max_length
        for i in range(0, len(text) - max_length, stride):
            input_chunk = text[i: i + max_length]
            target_chunk = text[i + 1: i + max_length + 1]

            # Tokenize the input and target chunks
            input_tokens = self.tokenizer.encode(input_chunk, add_special_tokens=False)
            target_tokens = self.tokenizer.encode(target_chunk, add_special_tokens=False)

            input_tokens = input_tokens[:self.max_length] + [self.tokenizer.pad_token_id] * (self.max_length - len(input_tokens))
            target_tokens = target_tokens[:self.max_length] + [self.tokenizer.pad_token_id] * (self.max_length - len(target_tokens))

            self.input_ids.append(torch.tensor(input_tokens))
            self.target_ids.append(torch.tensor(target_tokens))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(text, batch_size, max_length, stride, shuffle=True, drop_last=True):
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader
