import transformers
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pickle
import torch
import numpy as np

# import umap
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def normalize_adj(adj):
    """Normalized an adjacent matrix by dividing each row by the sum of the row"""
    s = adj.sum(axis=-1, keepdims=True)
    s[s == 0] = 1
    result = adj / s
    return result


MODEL_NAME= "ncbi/MedCPT-Query-Encoder"
MODEL_NAME = "BAAI/bge-large-en-v1.5"
# model = SentenceTransformer(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

OUTPUT_SIZE = 48

icd_names = pd.read_csv('./data/mimic3/raw/D_ICD_DIAGNOSES.csv.gz')
icd_names = icd_names[['ICD9_CODE','LONG_TITLE']]
add_icd_names = pd.read_csv('./data/add_icd.csv')
icd_names = pd.concat([icd_names,add_icd_names],ignore_index=True)
# print(icd_names)


load_from_pretrained = False

if not load_from_pretrained:
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

        embedding_mapping[code] = {'title':code_title, 'embedding':embedding}

    with open('./data/icd_embeddings.pickle', 'wb') as handle:
        pickle.dump(embedding_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./data/icd_embeddings.pickle', 'rb') as handle:
    embedding_mapping = pickle.load(handle)

with open('./data/mimic3/encoded/code_map.pkl', 'rb') as f:
    code_map = pickle.load(f)

mimic3_embeddings = []
for code in tqdm(code_map):
    code = str(code)
    embedding = embedding_mapping[code]['embedding']
    embedding = embedding[0]
    mimic3_embeddings.append(embedding)
mimic3_embeddings_final = np.stack(mimic3_embeddings,axis=0)
print(mimic3_embeddings_final.shape)
reducer = umap.UMAP(n_neighbors=15,min_dist=0.25,n_components=OUTPUT_SIZE)
transformed_mimic3 = reducer.fit_transform(mimic3_embeddings_final)

scaler = MinMaxScaler((-1,1))
mimic3_embeddings = scaler.fit_transform(transformed_mimic3)

# calculate similarity matrix
norms = np.linalg.norm(mimic3_embeddings, axis=1, keepdims=True)
normalized_embeddings = mimic3_embeddings/ norms

# Step 2: Compute the cosine similarity matrix
similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

torch.save(torch.from_numpy(mimic3_embeddings), 'pretraining/umap_embeddings.pt')    # .npy extension is added if not given
torch.save(torch.from_numpy(similarity_matrix), 'pretraining/similarity_matrix.pt')    # .npy extension is added if not given
