import os
from argparse import ArgumentParser
from pprint import pprint
import torch

from dataset.dataset import load_adj, EHRDataset
from evaluations.metrics import evaluate_hf, evaluate_codes
from models.model import Model
import yaml

from train import historical_hot, seed_everything


def main():
    parser = ArgumentParser("Training EHR")
    parser.add_argument("--config", type=str, default="configs/chet_h.yaml")
    parser.add_argument("--model", type=str, default="best", help="Model to use")
    parser.add_argument("--split", type=str, default="test", help="Split to use", choices=['train', 'test', 'valid'])
    args = parser.parse_args()

    """Load the configurations from .yaml file"""
    config = yaml.safe_load(open(args.config, "r"))
    pprint(config)

    seed = config['seed'] # random seed
    dataset = config['dataset'] # dataset name
    task = config['task'] # task, h or m
    use_cuda = config['use_cuda']
    graph_layer_type = config.get('graph_layer_type', 'gat')  # graph layer type
    use_text_embedding = config.get('use_text_embeddings', False)  # use text embeddings or not
    text_emb_size = config.get('text_emb_size', 300)  # text embedding size
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu') # device

    code_size = config['code_size']  # size of disease codes embeddings
    graph_size = config['graph_size']  # size of graph embeddings
    hidden_size = config['hidden_size']  # size of GRU hidden state
    t_attention_size = config['t_attention_size']  # size of temporal attention
    t_output_size = hidden_size  # size of temporal output

    batch_size = config['batch_size']  # batch size for train, val, and test

    seed_everything(seed) # seed everything for reproducibility
    dataset_path = os.path.join('data', 'standard')
    split_path = os.path.join(dataset_path, args.split)

    code_adj = load_adj(dataset_path, device=device)
    code_num = len(code_adj)

    # Load the dataset
    test_data = EHRDataset(split_path, label=task, batch_size=batch_size, shuffle=False, device=device)
    test_historical = historical_hot(test_data.code_x, code_num, test_data.visit_lens)

    output_size = code_num if task == 'm' else 1
    evaluate_fn = evaluate_codes if task == 'm' else evaluate_hf  # select the proper evaluation function
    activation = torch.nn.Sigmoid()  # final activation function
    loss_fn = torch.nn.BCELoss()  # loss function
    dropout_rate = config.get('dropout', 0)  # dropout rate for classifier layer

    param_path = os.path.join(config['output_dir'], 'params', dataset, task)

    """Define the model"""
    model = Model(code_num=code_num, code_size=code_size,
                  adj=code_adj, graph_size=graph_size, hidden_size=hidden_size, t_attention_size=t_attention_size,
                  t_output_size=t_output_size,
                  output_size=output_size, dropout_rate=dropout_rate, activation=activation,
                  graph_layer_type=graph_layer_type,
                  use_text_embeddings=use_text_embedding,
                  text_emb_size=text_emb_size)

    """Load the trained model"""
    model_path = os.path.join(param_path, f"{args.model}.pt")
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    """Evaluate the model"""
    with torch.no_grad():
        test_loss, f1_score, _ = evaluate_fn(model, test_data, loss_fn, output_size,
                                              historical_hot(test_data.code_x, code_num, test_data.visit_lens)
                                             )
if __name__ == "__main__":
    main()
