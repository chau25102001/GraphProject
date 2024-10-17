import os
import random
import time
from argparse import ArgumentParser
from pprint import pprint

import numpy as np
import termcolor
import torch

from dataset.dataset import load_adj, EHRDataset
from evaluations.metrics import evaluate_hf, evaluate_codes
from models.model import Model
from utils.utils import format_time, MultiStepLRScheduler
import yaml

def historical_hot(code_x, code_num, lens):
    result = np.zeros((len(code_x), code_num), dtype=int)
    for i, (x, l) in enumerate(zip(code_x, lens)):
        result[i] = x[l - 1]
    return result


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    parser = ArgumentParser("Training EHR")
    parser.add_argument("--config", type=str, default="configs/chet_h.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, "r"))
    pprint(config)

    seed = config['seed']
    dataset = config['dataset']
    task = config['task']
    use_cuda = config['use_cuda']
    pretrained_embeddings_path = config.get('pretrained_embeddings_path', None)
    freeze_embedding = config.get('freeze_embeddings', False)
    graph_layer_type = config.get('graph_layer_type', 'gat')
    use_text_embedding = config.get('use_text_embedding', False)
    text_emb_size = config.get('text_emb_size', 300)
    load_modules = config.get('load_modules', [])
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')

    code_size = config['code_size']
    graph_size = config['graph_size']
    hidden_size = config['hidden_size']
    t_attention_size = config['t_attention_size']
    t_output_size = hidden_size

    batch_size = config['batch_size']
    epochs = config['epochs']
    init_lr = config['lr']
    milestones = config.get('milestones', None)
    lrs = config.get('lrs', None)
    lr_scheduler_type = config.get('lr_scheduler', 'multi_step')

    seed_everything(seed)

    dataset_path = os.path.join('data', 'standard')
    train_path = os.path.join(dataset_path, 'train')
    valid_path = os.path.join(dataset_path, 'valid')
    test_path = os.path.join(dataset_path, 'test')

    code_adj = load_adj(dataset_path, device=device)
    code_num = len(code_adj)
    print(termcolor.colored(f"loading train data ...", "green"))
    train_data = EHRDataset(train_path, label=task, batch_size=batch_size, shuffle=True, device=device)
    print(termcolor.colored(f"loading valid data ...", "green"))
    valid_data = EHRDataset(valid_path, label=task, batch_size=batch_size, shuffle=False, device=device)
    print(termcolor.colored(f"loading test data ...", "green"))
    test_data = EHRDataset(test_path, label=task, batch_size=batch_size, shuffle=False, device=device)

    test_historical = historical_hot(valid_data.code_x, code_num, valid_data.visit_lens)


    output_size = code_num if task == 'm' else 1
    evaluate_fn = evaluate_codes if task == 'm' else evaluate_hf
    activation = torch.nn.Sigmoid()
    loss_fn = torch.nn.BCELoss()
    dropout_rate = config.get('dropout', 0)

    param_path = os.path.join(config['output_dir'], 'params', dataset, task)
    if not os.path.exists(param_path):
        os.makedirs(param_path)

    model = Model(code_num=code_num, code_size=code_size,
                  adj=code_adj, graph_size=graph_size, hidden_size=hidden_size, t_attention_size=t_attention_size,
                  t_output_size=t_output_size,
                  output_size=output_size, dropout_rate=dropout_rate, activation=activation,
                  graph_layer_type=graph_layer_type,
                  use_text_embeddings=use_text_embedding,
                  text_emb_size=text_emb_size)

    if pretrained_embeddings_path is not None:
        model.embedding_layer.init_weights(pretrained_embeddings_path, freeze=freeze_embedding, modules=load_modules)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    if epochs > 0 and lr_scheduler_type is not None:
        if lr_scheduler_type == 'multi_step':
            assert milestones is not None and lrs is not None
            assert len(milestones) == len(lrs)
            scheduler = MultiStepLRScheduler(optimizer, epochs, init_lr,
                                             milestones, lrs)

        elif lr_scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)
        else:
            raise ValueError("Invalid learning rate scheduler")
    else:
        scheduler = None
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    best_loss = 1e9
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_num = 0
        steps = len(train_data)
        st = time.time()
        if scheduler:
            scheduler.step()
        for step in range(len(train_data)):
            optimizer.zero_grad()
            code_x, visit_lens, divided, y, neighbors = train_data[step]
            output = model(code_x, divided, neighbors, visit_lens).squeeze()
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * output_size * len(code_x)
            total_num += len(code_x)

            end_time = time.time()
            remaining_time = format_time((end_time - st) / (step + 1) * (steps - step - 1))
            print('\r Epoch %d / %d,  Step %d / %d, remaining time: %s, loss: %.4f'
                  % (epoch + 1, epochs, step + 1, steps, remaining_time, total_loss / total_num), end='')
        train_data.on_epoch_end()
        et = time.time()
        time_cost = format_time(et - st)
        print('\r Epoch %d / %d,  Step %d / %d, time cost: %s, loss: %.4f' % (
            epoch + 1, epochs, steps, steps, time_cost, total_loss / total_num))
        with torch.no_grad():
            print("evaluating on valid data ...")
            valid_loss, f1_score = evaluate_fn(model, valid_data, loss_fn, output_size, test_historical)
            print("evaluating on test data ...")
            test_loss, f1_score = evaluate_fn(model, test_data, loss_fn, output_size,
                                              historical_hot(test_data.code_x, code_num, test_data.visit_lens))

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(param_path, 'best.pt'))
        torch.save(model.state_dict(), os.path.join(param_path, 'last.pt'))

    model.load_state_dict(torch.load(os.path.join(param_path, 'best.pt')))  # load best checkpoint
    with torch.no_grad():
        test_loss, f1_score = evaluate_fn(model, test_data, loss_fn, output_size,
                                          historical_hot(test_data.code_x, code_num, test_data.visit_lens))


if __name__ == "__main__":
    main()
