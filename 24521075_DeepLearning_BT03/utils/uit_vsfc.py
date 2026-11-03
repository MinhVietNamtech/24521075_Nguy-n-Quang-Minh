import torch
from torch import nn
import json
import pandas as pd
import string
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class Vocab:
    def __init__(self, train_path, label_key="topic"):
        self.pad = "<pad>"
        self.unk = "<unk>"
        self.label_key = label_key

        data = json.load(open(train_path, encoding='utf-8'))
        self.vocabs = set()
        self.labels = set()

        for item in data:
            sentence = self.preprocess_sentence(item["sentence"])
            self.vocabs.update(sentence.split())
            self.labels.add(item[self.label_key])

        self.w2i = {
            self.pad: 0,
            self.unk: 1
        }

        for idx, word in enumerate(sorted(self.vocabs), start=2):
            self.w2i[word] = idx
        self.i2w = {idx: word for word, idx in self.w2i.items()}
        self.l2i = {label: idx for idx, label in enumerate(sorted(self.labels))}
        self.i2l = {idx: label for label, idx in self.l2i.items()}

    @property
    def n_labels(self) -> int:
        return len(self.l2i)
    
    @property
    def len(self) -> int:
        return len(self.w2i)        # kích thước vocab -> số dòng của Embedding

    @property
    def pad_idx(self) -> int:
        return self.w2i[self.pad]

    def preprocess_sentence(self, sentence: str) -> str:
        translator = str.maketrans("", "", string.punctuation)  # xóa dấu câu
        sentence = sentence.lower()                             # chuyển chữ thường 
        sentence = sentence.translate(translator)
        return sentence

    def encode_sentence(self, sentence):
        sentence = self.preprocess_sentence(sentence)
        splits = sentence.split()

        if len(splits)==0:
            splits = [self.unk]

        unk_id = self.w2i[self.unk]
        word_ids = [self.w2i.get(word, unk_id) for word in splits]
        return torch.tensor(word_ids, dtype=torch.long)

    def encode_label(self, label: str) -> torch.Tensor:
        label_idx = self.l2i[label]

        return torch.tensor(label_idx, dtype=torch.long)
    
    # chuyển idx -> label 
    def decode_label(self, label_ids: torch.Tensor) -> list:
        label_ids = label_ids.tolist()
        labels = [self.i2l[idx] for idx in label_ids]

        return labels

def collate_fn(items: list[dict]):
    input_ids = [item['input_ids'] for item in items]
    lengths = torch.tensor([ids.shape[0] for ids in input_ids], dtype=torch.long)
    max_len = lengths.max()

    input_ids = torch.stack(
        [F.pad(
            input,
            pad=(0, max_len - input.shape[0]),
            value= 0,
            mode= "constant"
        ) for input in input_ids]
    )
    labels = torch.stack([item["label"] for item in items])

    return {
        "input_ids": input_ids,     # (bs, max_len)
        "lengths": lengths,         # (bs, )
        "label": labels             # (bs, )
    }

class UIT_VSFC(Dataset):
    def __init__(self, path: str, vocab: Vocab):
        super(UIT_VSFC, self).__init__()
        self._data = json.load(open(path,encoding="utf-8"))
        self.vocab = vocab

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        sentence = self._data[index]["sentence"]
        label = self._data[index][self.vocab.label_key]
        enc_sentence = self.vocab.encode_sentence(sentence)
        enc_label = self.vocab.encode_label(label)
        return {
            "input_ids": enc_sentence,
            "label": enc_label
        }
