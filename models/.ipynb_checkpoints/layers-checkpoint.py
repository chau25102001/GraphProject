import torch
import torch.nn as nn

from models.utils import SingleHeadAttentionLayer


class EmbeddingLayer(nn.Module):
    def __init__(self, code_num, code_size, graph_size, use_text_embeddings = True, text_emb_size = 1024, ckpt_path = 'pretraining/bge_embeddings.pt', freeze=True):
        """
        :param code_num: number of diseases
        :param code_size: size of disease code embeddings
        :param graph_size: size of graph embeddings
        """
        super().__init__()
        self.code_num = code_num
        
        self.code_text = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, text_emb_size)))

        self.u_embeddings = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, graph_size)))

        self.c_embeddings = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, code_size)))
        self.n_embeddings = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, code_size)))
        
        self.use_text_embeddings = use_text_embeddings
        if use_text_embeddings:
            self.init_weights(ckpt_path = ckpt_path, freeze = freeze)
            self.c_fc = nn.Sequential(nn.Linear(text_emb_size, code_size), nn.LeakyReLU(0.1), nn.Linear(code_size, code_size))
            self.n_fc = nn.Sequential(nn.Linear(text_emb_size, code_size), nn.LeakyReLU(0.1), nn.Linear(code_size, code_size))
        

    def init_weights(self, ckpt_path, modules=['code_text'], freeze=False):
        # mapping = {'c_embeddings': self.c_embeddings, 'n_embeddings': self.n_embeddings, 'u_embeddings': self.u_embeddings}
        
        # mapping = {'c_embeddings': self.c_embeddings, 'n_embeddings': self.n_embeddings}
        mapping = {'code_text': self.code_text}
        ckpt = torch.load(ckpt_path)
        print('ckpt', ckpt.shape)
        for module in modules:
            print(f"Loading {module} from {ckpt_path}")
            mapping[module].data = ckpt

        if freeze: # freeze the embeddings
            print('Freezed')
            # self.c_embeddings.requires_grad_(False) 
            # self.n_embeddings.requires_grad_(False)
            # self.u_embeddings.requires_grad = False
            self.code_text.requires_grad_(False) 
            # self.n_text.requires_grad_(False)
        # print(code_text.data.shape)

    def forward(self):
        if self.use_text_embeddings:
            self.c_embeddings = self.c_fc(self.code_text)
            self.n_embeddings = self.n_fc(self.code_text)

        return self.c_embeddings, self.n_embeddings, self.u_embeddings


class GraphLayer(nn.Module):
    def __init__(self, adj, code_size, graph_size):
        """
        :param adj: adjacency matrix, shape code_num x code_num
        :param code_size: size of code embeddings
        :param graph_size: size of graph embeddings
        """
        super().__init__()
        self.adj = adj
        self.dense = nn.Linear(code_size, graph_size)
        self.activation = nn.LeakyReLU()

    def forward(self, code_x, neighbor, c_embeddings, n_embeddings):
        """
        :param code_x: center code, shape code_num
        :param neighbor: neighbor code, shape code_num
        :param c_embeddings: center embeddings, shape code_num x code_size
        :param n_embeddings: neighbor embeddings, shape code_num x code_size
        """
        center_codes = torch.unsqueeze(code_x, dim=-1)  # code_num x 1
        neighbor_codes = torch.unsqueeze(neighbor, dim=-1)  # code_num x 1

        center_embeddings = center_codes * c_embeddings  # select embeddings, code_num x code_size
        neighbor_embeddings = neighbor_codes * n_embeddings  # select embeddings, code_num x code_size
        cc_embeddings = center_codes * torch.matmul(self.adj,
                                                    center_embeddings)  # sum of adjacent nodes' embeddings code_num x code_size
        cn_embeddings = center_codes * torch.matmul(self.adj,
                                                    neighbor_embeddings)  # sum of undiagnosed neighbor nodes' embeddings code_num x code_size
        nn_embeddings = neighbor_codes * torch.matmul(self.adj, neighbor_embeddings)
        nc_embeddings = neighbor_codes * torch.matmul(self.adj, center_embeddings)
        # print(center_embeddings.shape, cc_embeddings.shape, cn_embeddings.shape)
        co_embeddings = self.activation(self.dense(center_embeddings + cc_embeddings + cn_embeddings))
        no_embeddings = self.activation(self.dense(neighbor_embeddings + nn_embeddings + nc_embeddings))
        return co_embeddings, no_embeddings  # code_num x graph_size, code_num x graph_size


class TransitionLayer(nn.Module):
    def __init__(self, code_num, graph_size, hidden_size, t_attention_size, t_output_size):
        super().__init__()
        self.gru = nn.GRUCell(input_size=graph_size, hidden_size=hidden_size)
        self.single_head_attention = SingleHeadAttentionLayer(graph_size, graph_size, t_output_size, t_attention_size)
        self.activation = nn.Tanh()

        self.code_num = code_num
        self.hidden_size = hidden_size

    def forward(self, t, co_embeddings, divided, no_embeddings, unrelated_embeddings, hidden_state=None):
        """
        :param t: visit t
        :param co_embeddings: diagnosed disease embeddings, shape code_num x graph_size
        :param divided: divided, shape code_num x 3
        :param no_embeddings: previous undiagnosed neighbor embeddings, shape code_num x graph_size
        :param unrelated_embeddings: unrelated disease embeddings, shape code_num x graph_size
        :param hidden_state: hidden state, shape code_num x hidden_size
        """
        m1, m2, m3 = divided[:, 0], divided[:, 1], divided[:, 2]  # code_num, code_num, code_num
        m1_index = torch.where(m1 > 0)[0]  # disease id that appear in visit t and t-1
        m2_index = torch.where(m2 > 0)[
            0]  # disease id that appear in visit t and is an undiagnosed neighbor in visit t-1
        m3_index = torch.where(m3 > 0)[0]  # disease id that appear in visit t and is an unrelated disease in visit t-1
        h_new = torch.zeros((self.code_num, self.hidden_size), dtype=co_embeddings.dtype).to(co_embeddings.device)
        output_m1 = 0
        output_m23 = 0
        if len(m1_index) > 0:  # embedding of persistent diseases
            m1_embedding = co_embeddings[m1_index]  # select m1 embeddings, len(m1_index) x graph_size
            h = hidden_state[
                m1_index] if hidden_state is not None else None  # select corresponding hidden states, len(m1_index) x hidden_size
            h_m1 = self.gru(m1_embedding, h)  # len(m1_index) x hidden_size
            h_new[m1_index] = h_m1  # fill in the result
            output_m1, _ = torch.max(h_m1, dim=-2)  # max pool
        if t > 0 and len(m2_index) + len(m3_index) > 0:  # embedding of emerging disease
            q = torch.vstack([no_embeddings[m2_index], unrelated_embeddings[
                m3_index]])  # (len(m2_index) + len(m3_index)) x graph_size, neighbor embeddings of last visit + unrelated embeddings of last visit
            v = torch.vstack([co_embeddings[m2_index], co_embeddings[
                m3_index]])  # (len(m2_index) + len(m3_index)) x graph_size, diagnosed emerging disease embeddings of this visit
            h_m23 = self.activation(self.single_head_attention(q, q, v))
            h_new[m2_index] = h_m23[:len(m2_index)]
            h_new[m3_index] = h_m23[len(m2_index):]
            output_m23, _ = torch.max(h_m23, dim=-2)  # max pool
        if len(m1_index) == 0:  # if there is no persistent disease
            output = output_m23
        elif len(m2_index) + len(m3_index) == 0:  # if there is not emerging disease
            output = output_m1
        else:  # if there is both
            output, _ = torch.max(torch.vstack([output_m1, output_m23]), dim=-2)
        return output, h_new
