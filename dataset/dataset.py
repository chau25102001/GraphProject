import os
import torch
import numpy as np

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
        self._size = self.code_x.shape[0] # number of patients
        self.idx = np.arange(self._size)
        self.batch_size = batch_size
        self.shuffle= shuffle
        self.device = device

    def _load(self, label):
        code_x = load_sparse(os.path.join(self.path, 'code_x.npz'))
        visit_lens = np.load(os.path.join(self.path, 'visit_lens.npz'))['lens']
        if label == 'm': # load multi-labels
            y = load_sparse(os.path.join(self.path, 'code_y.npz'))
        elif label == 'h': # load heart failure labels
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
