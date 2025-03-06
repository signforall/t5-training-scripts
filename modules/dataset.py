import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import torch.nn.functional as F
import json

class VideoDataset(Dataset):
    def __init__(self, h5_file_path, label_file_path, tokenizer, max_seq_length=600, test_set=False, with_labels=True):
        self.data = []
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.test_set = test_set
        self.with_labels = with_labels

        data_file = self.load_h5_file(h5_file_path)
        
        if self.with_labels:
            self.sentences = self.load_text_file(label_file_path)

        self.data = self.process_data(data_file)
        
        self.filter_data()

    def h5_to_dict(self, group):
        return {key: self.h5_to_dict(item) if isinstance(item, h5py.Group) else item[()] for key, item in group.items()}

    def load_h5_file(self, file_path):
        with h5py.File(file_path, 'r') as f:
            return list(self.h5_to_dict(f).values())

    def load_text_file(self, file_path):
        return json.load(open(file_path, encoding='utf-8'))

    def process_data(self, data_file):
        processed_data = []
        for item in data_file:
            key = list(item.keys())[0]
            features = torch.tensor(list(item.values())[0], dtype=torch.float32)
            if self.with_labels:
                sentence = self.sentences[key][key]['translation']
                processed_data.append([key, features, sentence])
            else:
                processed_data.append([key, features, None])
        return processed_data
    
    def filter_data(self):
        self.data = [i for i in self.data if i[1].shape[0] <= self.max_seq_length]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        key, features, sentence = self.data[idx]
        original_length = len(features)
        features = F.pad(features, (0, 0, self.max_seq_length - original_length, 0), value=-100)
        attention_mask = torch.ones_like(features)
        attention_mask[:self.max_seq_length - original_length, :] = 0.0
        if not self.test_set:
            input_ids = self.tokenizer(sentence, truncation=True, padding="max_length", max_length=512, return_tensors="pt")["input_ids"].squeeze()
            input_ids[input_ids == self.tokenizer.pad_token_id] = -100
        else:
            input_ids = sentence
        return {"features": features, "attention_mask": attention_mask, "labels": input_ids, "key": key}
