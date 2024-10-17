import torch
import pickle
from tqdm import tqdm
import numpy as np
with open('./data/icd_embeddings.pickle', 'rb') as handle:
    embedding_mapping = pickle.load(handle)

with open('./data/mimic3/encoded/code_map.pkl', 'rb') as f:
    code_map = pickle.load(f)
    # print(code_map)

mimic3_embeddings = []
for code in tqdm(code_map):
    code = str(code)
    # try:
    embedding = embedding_mapping[code]['embedding']
    embedding = embedding[0]
    mimic3_embeddings.append(embedding)
mimic3_embeddings_final = np.stack(mimic3_embeddings,axis=0)
print(mimic3_embeddings_final.shape)
torch.save(torch.from_numpy(mimic3_embeddings_final), 'pretraining/bge_embeddings.pt')    # .npy extension is added if not given
model = torch.load('pretraining/bge_embeddings.pt')
print('loaded', model.shape)

