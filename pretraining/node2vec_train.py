import os.path
import numpy as np
import networkx as nx
import sys
sys.path.append(".")
from dataset.dataset import load_adj
from node2vec import Node2Vec
from argparse import ArgumentParser
import yaml

def main():
    parser = ArgumentParser("Training Node2Vec")
    parser.add_argument("--config", type=str, default="configs/chet_h.yaml")
    args = parser.parse_args()

    save_root = "./pretrained_weights"
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    EMBEDDING_FILENAME = 'embeddings.emb'
    EMBEDDING_MODEL_FILENAME = 'embeddings.model'

    config = yaml.safe_load(open(args.config, "r"))
    data_path = os.path.join('data', 'standard')
    code_adj = load_adj(data_path, device='cpu')
    graph = nx.from_numpy_array(code_adj.numpy().astype(np.float64))

    node2vec = Node2Vec(graph, dimensions=config['code_size'], walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    model.wv.save_word2vec_format(EMBEDDING_FILENAME)
    print("Node2Vec model saved to", EMBEDDING_FILENAME)
    model.save(EMBEDDING_MODEL_FILENAME)

if __name__ == "__main__":
    main()