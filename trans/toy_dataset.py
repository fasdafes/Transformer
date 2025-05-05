
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# 英法对照句子
eng_sentences = [
    "i am a student",
    "he is a teacher",
    "she is a nurse",
    "you are a boy",
    "they are girls",
    "hello",
    "good morning",
    "thank you",
    "how are you",
    "i am fine"
]

fra_sentences = [
    "je suis un étudiant",
    "il est un professeur",
    "elle est une infirmière",
    "tu es un garçon",
    "elles sont des filles",
    "bonjour",
    "bonjour",
    "merci",
    "comment ça va",
    "je vais bien"
]

def tokenize(sentence):
    return sentence.lower().split()

def build_vocab(sentences):
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
    idx = 3
    for sentence in sentences:
        for token in tokenize(sentence):
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab

def encode(sentence, vocab, add_bos=False, add_eos=False):
    tokens = tokenize(sentence)
    ids = [vocab[token] for token in tokens]
    if add_bos:
        ids = [vocab["<bos>"]] + ids
    if add_eos:
        ids = ids + [vocab["<eos>"]]
    return torch.tensor(ids, dtype=torch.long)

class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab):
        self.src = src_sentences
        self.tgt = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src_ids = encode(self.src[idx], self.src_vocab)
        tgt_ids = encode(self.tgt[idx], self.tgt_vocab, add_bos=True, add_eos=True)
        return src_ids, tgt_ids

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)

    tgt_input = tgt_batch[:, :-1]
    tgt_output = tgt_batch[:, 1:]
    return {
        "src": src_batch,
        "tgt_input": tgt_input,
        "tgt_output": tgt_output
    }

def build_toy_dataloader(batch_size=2):
    eng_vocab = build_vocab(eng_sentences)
    fra_vocab = build_vocab(fra_sentences)
    dataset = TranslationDataset(eng_sentences, fra_sentences, eng_vocab, fra_vocab)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader, eng_vocab, fra_vocab
