import json
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch
import torch.nn.functional as F

class Vocab():
    def __init__(self, train_path="./data/PhoNER_COVID19/data/word/train_word.json"):
        self.pad = "<pad>"
        self.unk = "<unk>"
        self.vocabs = set()
        self.labels = set()
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                words = item["words"]
                tags = item["tags"]
                self.vocabs.update(self.preprocess_sentence(words))
                self.labels.update(tags)

        self.w2i = {
            self.pad: 0,
            self.unk: 1
        }
        for idx, vocab in enumerate(self.vocabs, start=2):
            self.w2i[vocab] = idx
        self.i2w = {idx: vocab for vocab, idx in self.w2i.items()}
        self.l2i = {label: idx for idx, label in enumerate(sorted(self.labels))}
        self.i2l = {idx: label for label, idx in self.l2i.items()}

    def preprocess_sentence(self, sentence):
        return [word.strip().lower() for word in sentence]

    @property
    def n_labels(self) -> int:
        return len(self.l2i)
    
    @property
    def len(self) -> int:
        return len(self.w2i)        # kích thước vocab -> số dòng của Embedding

    @property
    def pad_idx(self) -> int:
        return self.w2i[self.pad]

    def encode_sentence(self, sentence: list[str]):
        if len(sentence)==0:
            sentence = [self.unk]
        sentence = self.preprocess_sentence(sentence)
        unk_id = self.w2i[self.unk]
        word_ids = [self.w2i.get(word, unk_id) for word in sentence]
        return torch.tensor(word_ids, dtype=torch.long)

    def encode_tags(self, tags: list[str]):
        tag_ids = [self.l2i[tag] for tag in tags]
        return torch.tensor(tag_ids, dtype=torch.long)

    def decode_tags(self, tag_ids: torch.Tensor):
        tag_ids = tag_ids.tolist()
        return [self.i2l[tag_id] for tag_id in tag_ids]

    
def collate_fn(items: list[dict]):
    word_ids = [item['word_ids'] for item in items]
    tags = [item['tags'] for item in items]
    lengths = torch.tensor([len(item['word_ids']) for item in items], dtype=torch.long)
    max_length = max(lengths)

    padded_word_ids = torch.stack(
        [F.pad(
            word_id, 
            pad=(0, max_length-len(word_id)),
            value=0,
            mode='constant') 
            for word_id in word_ids]
    )
    padded_tag_ids = torch.stack(
        [F.pad(
            tag, 
            pad=(0, max_length-len(tag)),
            value=-1, # padding value
            mode='constant')
            for tag in tags]            
    )
    return {
        "word_ids": padded_word_ids,     # (bs, max_len)
        "lengths": lengths,         # (bs, )
        "tags": padded_tag_ids       # (bs, )
    }


class PhoNER(Dataset):
    def __init__(self, data_path="./data/PhoNER_COVID19/data/word/dev_word.json", vocab: Vocab=None):
        self.vocab = vocab
        data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        words = item["words"]
        tags = item["tags"]
        return {
            "word_ids": self.vocab.encode_sentence(words),
            "tags": self.vocab.encode_tags(tags)
        }

