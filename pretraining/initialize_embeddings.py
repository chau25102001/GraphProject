import transformers
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle
import torch
import numpy as np

import umap
from sklearn.preprocessing import StandardScaler, MinMaxScaler


MODEL_NAME= "ncbi/MedCPT-Query-Encoder"
# model = SentenceTransformer(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

OUTPUT_SIZE = 32

icd_names = pd.read_csv('../data/mimic3/raw/D_ICD_DIAGNOSES.csv.gz')
icd_names = icd_names[['ICD9_CODE','LONG_TITLE']]
add_icd_names = pd.read_csv('../data/add_icd.csv')
icd_names = pd.concat([icd_names,add_icd_names],ignore_index=True)
# print(icd_names)


code_name_mapping = {}
for _,row in icd_names.iterrows():
    code_name_mapping[row['ICD9_CODE']] =  row['LONG_TITLE']

    
embedding_mapping = {}
print('Calculating embeddings')
for code in tqdm(code_name_mapping):
    code_title = code_name_mapping[code]
    code_title = code_title.lower()
    code_title = code_title.replace('(','')
    code_title = code_title.replace(')','')
    code_title = code_title.replace('[','')
    code_title = code_title.replace(']','')
    code_title = code_title.replace(',','')
    code_title = code_title.replace('-','')
    code_title = code_title.replace('/','')
    # embedding = model.encode([code_title])
    
    with torch.no_grad():
        # tokenize the queries
        encoded = tokenizer(
            [code_title], 
            truncation=True, 
            padding=True, 
            return_tensors='pt', 
            max_length=256,
        )

        # encode the queries (use the [CLS] last hidden states as the representations)
        embedding = model(**encoded).last_hidden_state[:, 0, :]
        print(embedding.shape)
    
    code = str(code)
    split_pos = 4 if code.startswith('E') else 3
    code = code[:split_pos] + '.' + code[split_pos:] if len(code) > split_pos else code
    # print(code,code_title)
    # print(embedding.shape)

    embedding_mapping[code] = {'title':code_title, 'embedding':embedding}

with open('../data/icd_embeddings.pickle', 'wb') as handle:
    pickle.dump(embedding_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../data/icd_embeddings.pickle', 'rb') as handle:
    embedding_mapping = pickle.load(handle)

with open('../data/mimic3/encoded/code_map.pkl', 'rb') as f:
    code_map = pickle.load(f)
    print(code_map)

mimic3_embeddings = []
for code in tqdm(code_map):
    code = str(code)
    # try:
    embedding = embedding_mapping[code]['embedding']
    embedding = embedding[0]
    mimic3_embeddings.append(embedding)
    # except:
    #     # print(code)
    #     raw_code = code.replace('.','')
    #     print(raw_code)
mimic3_embeddings_final = np.stack(mimic3_embeddings,axis=0)

reducer = umap.UMAP(n_neighbors=15,min_dist=0.25,n_components=OUTPUT_SIZE)
transformed_mimic3 = reducer.fit_transform(mimic3_embeddings_final)

scaler = MinMaxScaler((-1,1))
mimic3_embeddings = scaler.fit_transform(transformed_mimic3)

torch.save(torch.from_numpy(mimic3_embeddings), 'umap_embeddings.pt')    # .npy extension is added if not given
