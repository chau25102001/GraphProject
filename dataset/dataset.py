import os

import numpy as np
import torch
import tqdm
from transformers import AutoModel, AutoTokenizer
from utils.utils import process_string
from dataset.utils import load_sparse


def load_adj(path, device=torch.device('cpu')):
    filename = os.path.join(path, 'code_adj.npz')
    adj = torch.from_numpy(load_sparse(filename)).to(device=device, dtype=torch.float32)
    return adj


class EHRDataset:
    def __init__(self, data_path, label='m', batch_size=32, shuffle=True, device=torch.device('cpu')):
        super().__init__()
        self.path = data_path
        self.code_x, self.visit_lens, self.y, self.divided, self.neighbors = self._load(label)
        self.idx = np.arange(self._size)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

    @property
    def _size(self):
        return self.code_x.shape[0]

    def _load(self, label):
        code_x = load_sparse(os.path.join(self.path, 'code_x.npz'))
        visit_lens = np.load(os.path.join(self.path, 'visit_lens.npz'))['lens']
        if label == 'm':  # load multi-labels
            y = load_sparse(os.path.join(self.path, 'code_y.npz'))
        elif label == 'h':  # load heart failure labels
            y = np.load(os.path.join(self.path, 'hf_y.npz'))['hf_y']
        else:
            raise KeyError('Unsupported label type')
        divided = load_sparse(os.path.join(self.path, 'divided.npz'))
        neighbors = load_sparse(os.path.join(self.path, 'neighbors.npz'))
        return code_x, visit_lens, y, divided, neighbors

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)

    def size(self):
        return self._size

    def label(self):
        return self.y

    def __len__(self):
        len_ = self._size // self.batch_size
        return len_ if self._size % self.batch_size == 0 else len_ + 1

    def __getitem__(self, index):
        device = self.device
        start = index * self.batch_size
        end = start + self.batch_size
        slices = self.idx[start:end]
        code_x = torch.from_numpy(self.code_x[slices]).to(device)
        visit_lens = torch.from_numpy(self.visit_lens[slices]).to(device=device, dtype=torch.long)
        y = torch.from_numpy(self.y[slices]).to(device=device, dtype=torch.float32)
        divided = torch.from_numpy(self.divided[slices]).to(device)
        neighbors = torch.from_numpy(self.neighbors[slices]).to(device)
        return code_x, visit_lens, divided, y, neighbors


class EHRDatasetWithNotes:
    def __init__(self, data_path, label='m', batch_size=32, shuffle=True, device=torch.device('cpu'),
                 text_model_name='yikuan8/Clinical-Longformer'):
        super().__init__()
        self.path = data_path
        self.code_x, self.visit_lens, self.y, self.divided, self.neighbors, self.notes = self._load(label)
        self.idx = np.arange(self._size)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.llm, self.tokenizer = self._load_llm(text_model_name)
        self.patient_admission_note_embeddings, self.attention_masks = self.get_embeddings()  # cache embedding

    @property
    def _size(self):
        return self.code_x.shape[0]

    def _load(self, label):
        code_x = load_sparse(os.path.join(self.path, 'code_x.npz'))
        visit_lens = np.load(os.path.join(self.path, 'visit_lens.npz'))['lens']
        if label == 'm':  # load multi-labels
            y = load_sparse(os.path.join(self.path, 'code_y.npz'))
        elif label == 'h':  # load heart failure labels
            y = np.load(os.path.join(self.path, 'hf_y.npz'))['hf_y']
        else:
            raise KeyError('Unsupported label type')
        divided = load_sparse(os.path.join(self.path, 'divided.npz'))
        neighbors = load_sparse(os.path.join(self.path, 'neighbors.npz'))
        notes = open(os.path.join(self.path, 'notes.txt')).readlines()
        notes = [eval(note.strip()) for note in notes]  # convert string to list
        return code_x, visit_lens, y, divided, neighbors, notes

    def _load_llm(self, model_name):
        llm = AutoModel.from_pretrained(model_name).to(self.device)
        llm.requires_grad_(False)  # turn off gradient
        llm.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return llm, tokenizer

    def get_embeddings(self):
        cache_path = os.path.join(self.path, 'note_embeddings.pt')
        if os.path.exists(cache_path):
            save_obj = torch.load(cache_path)
            embeddings = save_obj['embeddings']
            attention_masks = save_obj['attention_masks']
            embeddings = [e.to(self.device) for e in embeddings]
            attention_masks = [a.to(self.device) for a in attention_masks]
            return embeddings, attention_masks
        with torch.no_grad():
            embeddings = []
            attention_masks = []
            for i, admission_notes in tqdm.tqdm(enumerate(self.notes), total=len(self.notes),
                                                desc="Caching note embedding"):
                admission_notes_embeddings = []
                attention_mask = []
                assert len(admission_notes) == self.visit_lens[
                    i], "Number of notes does not match number of visits, found {} notes and {} visits".format(
                    len(admission_notes), self.visit_lens[i])
                for note in admission_notes:
                    note = process_string(note)
                    if len(note) == 0:
                        attention_mask.append(0)
                    else:
                        attention_mask.append(1)
                    encoded_inputs = self.tokenizer(note, return_tensors='pt', padding='max_length', truncation=True,
                                                    max_length=self.tokenizer.model_max_length - 2, # add 2 special tokens
                                                    # stride=self.tokenizer.model_max_length // 2,
                                                    # return_overflowing_tokens=True
                                                    )
                    input_ids = encoded_inputs['input_ids'].to(self.device)
                    attn_mask = encoded_inputs['attention_mask'].to(self.device)
                    # print(input_ids.shape)
                    outputs = self.llm(input_ids, attention_mask=attn_mask).last_hidden_state
                    embedding = outputs[:, 0, :]  # CLS embedding
                    embedding = torch.mean(embedding, dim=0)  # mean if overflow, 1 x embed_dim
                    # print(embedding.shape)
                    admission_notes_embeddings.append(embedding)
                attention_masks.append(torch.tensor(attention_mask, dtype=torch.long)) # admission_len x embed_dim
                admission_notes_embeddings = torch.vstack(admission_notes_embeddings)  # admission_len x embed_dim
                assert admission_notes_embeddings.shape[0] == self.visit_lens[
                    i], "Number of embeddings does not match number of visits, found {} embeddings and {} visits for patient: {}".format(
                    admission_notes_embeddings.shape[0], self.visit_lens[i], i)
                embeddings.append(admission_notes_embeddings)
            save_obj = {"embeddings": embeddings, "attention_masks": attention_masks}
            torch.save(save_obj, cache_path) # save cache for later use

        del self.tokenizer
        del self.llm
        torch.cuda.empty_cache()
        return embeddings, attention_masks

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)

    def size(self):
        return self._size

    def label(self):
        return self.y

    def __len__(self):
        len_ = self._size // self.batch_size
        return len_ if self._size % self.batch_size == 0 else len_ + 1

    def __getitem__(self, index):
        device = self.device
        start = index * self.batch_size
        end = start + self.batch_size
        slices = self.idx[start:end]
        code_x = torch.from_numpy(self.code_x[slices]).to(device)
        visit_lens = torch.from_numpy(self.visit_lens[slices]).to(device=device, dtype=torch.long)
        y = torch.from_numpy(self.y[slices]).to(device=device, dtype=torch.float32)
        divided = torch.from_numpy(self.divided[slices]).to(device)
        neighbors = torch.from_numpy(self.neighbors[slices]).to(device)
        notes_embeddings = [self.patient_admission_note_embeddings[i].to(device) for i in slices]  # list of tensors, each shape visit_len x note_size
        attention_masks = [self.attention_masks[i].to(device) for i in slices] # list of tensors # each shape visit_len
        for ne in notes_embeddings:
            ne.requires_grad_(False)
        return code_x, visit_lens, divided, y, neighbors, notes_embeddings, attention_masks
