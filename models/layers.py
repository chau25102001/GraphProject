import torch
import torch.nn as nn

from models.utils import SingleHeadAttentionLayer, MultiHeadAttentionLayer


class EmbeddingLayer(nn.Module):
    def __init__(self, code_num, code_size, graph_size, use_text_embeddings = True, text_emb_size = 1024):
        """
        Embedding layer, including diagnosed disease embeddings, undiagnosed neighbor embeddings and unrelated disease embeddings
        :param code_num: number of diseases
        :param code_size: size of disease code embeddings
        :param graph_size: size of graph embeddings
        """
        super().__init__()
        self.code_num = code_num
        self.u_embeddings = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, graph_size)))

        self.c_embeddings = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, code_size)))
        self.n_embeddings = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, code_size)))

        self.use_text_embeddings = use_text_embeddings
        self.code_text = None
        if use_text_embeddings:
            self.code_text = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, text_emb_size)))
            self.c_fc = nn.Sequential(nn.Linear(text_emb_size, code_size), nn.LeakyReLU(0.1), nn.Linear(code_size, code_size))
            self.n_fc = nn.Sequential(nn.Linear(text_emb_size, code_size), nn.LeakyReLU(0.1), nn.Linear(code_size, code_size))


    def init_weights(self, ckpt_path, modules=['code_text'], freeze=False):
        """
        Function to load pretrained embeddings
        :param ckpt_path: path to the pretrained embeddings
        :param modules: modules to load, including 'c_embeddings', 'n_embeddings', 'code_text'
        :param freeze: freeze the embeddings or not
        """
        mapping = {'c_embeddings': self.c_embeddings, 'n_embeddings': self.n_embeddings, 'code_text': self.code_text}
        ckpt = torch.load(ckpt_path)
        for module in modules:
            if hasattr(mapping[module], 'data'):
                print(f"Loading {module} from {ckpt_path}")
                mapping[module].data = ckpt

            if freeze: # freeze the embeddings
                if hasattr(mapping[module], 'requires_grad_'):
                    print('Freezed')
                    mapping[module].requires_grad_(False)

    def forward(self):
        if self.use_text_embeddings:
            c_embeddings = self.c_fc(self.code_text)
            n_embeddings = self.n_fc(self.code_text)
            return c_embeddings, n_embeddings, self.u_embeddings
        return self.c_embeddings, self.n_embeddings, self.u_embeddings


class GraphLayer(nn.Module):
    def __init__(self, adj, code_size, graph_size):
        """
        Basic Graph aggregation layer, a node embedding is aggregated with its adjacent nodes' embeddings
        :param adj: adjacency matrix, shape code_num x code_num
        :param code_size: size of code embeddings
        :param graph_size: size of graph embeddings
        """
        super().__init__()
        self.adj = adj
        self.dense = nn.Linear(code_size, graph_size)
        self.activation = nn.LeakyReLU()

    def get_embedding(self, code_x, neighbor, c_embeddings, n_embeddings):
        """
        Aggregate the embeddings of center nodes and neighbor nodes
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

        co_embeddings = center_embeddings + cc_embeddings + cn_embeddings
        no_embeddings = neighbor_embeddings + nn_embeddings + nc_embeddings
        return co_embeddings, no_embeddings  # code_num x code_size, code_num x code_size

    def forward(self, code_x, neighbor, c_embeddings, n_embeddings):
        """
        :param code_x: center code, shape code_num
        :param neighbor: neighbor code, shape code_num
        :param c_embeddings: center embeddings, shape code_num x code_size
        :param n_embeddings: neighbor embeddings, shape code_num x code_size
        """
        co_embeddings, no_embeddings = self.get_embedding(code_x, neighbor, c_embeddings, n_embeddings)
        co_embeddings = self.activation(self.dense(co_embeddings)) # projection and non-linear activation
        no_embeddings = self.activation(self.dense(no_embeddings)) # projection and non-linear activation
        return co_embeddings, no_embeddings  # code_num x graph_size, code_num x graph_size


class GATConv(nn.Module):
    """
    Graph Convolution Layer for Graph aggregation
    """
    def __init__(self, input_size, output_size, attention_size=64, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.attention_size = attention_size
        self.W = nn.Linear(input_size, attention_size, bias=False)
        self.a_s = nn.Linear(attention_size, 1, bias=False)
        self.a_t = nn.Linear(attention_size, 1, bias=False)
        self.activation = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(dropout)

        self.out_proj = nn.Linear(attention_size, output_size)

    def get_attention_score(self, hs, ht, mask=None):
        """
        Attention score calculation
        :param hs: shape n x hidden_size
        :param ht: shape m x hidden_size
        :param mask: shape n x m
        :return scores: shape n x m
        """
        source_scores = self.a_s(hs)  # n x 1
        target_scores = self.a_t(ht)  # m x 1
        scores = source_scores + target_scores.mT  # n x m
        scores = self.activation(scores)
        if mask is not None:
            attention_mask = -9e16 * torch.ones_like(mask).to(scores.device)
            attention_mask = torch.where(mask > 0, 0, attention_mask)
            scores = scores + attention_mask
        scores = torch.softmax(scores, dim=-1)
        return scores

    def forward(self, x_s, x_t, mask=None):
        """
        Aggregate target node embedding into source node embeddings
        :param x_s: source node embeddings, shape n x input_size
        :param x_t: target node e, shape m x input_size
        :param mask: mask, shape n x m
        """
        hs = self.W(x_s)
        ht = self.W(x_t)
        score = self.get_attention_score(hs, ht, mask)
        score = self.drop(score)
        h_prime = torch.matmul(score, ht)
        h_prime = self.out_proj(h_prime)
        return h_prime


class GATGraphLayer(nn.Module):
    def __init__(self, adj, code_size, hidden_size, graph_size, attention_size=64, dropout=0.1):
        """
        Graph Attention Layer for Graph aggregation
        :param adj: adjacency matrix, shape code_num x code_num
        :param code_size: size of code embeddings
        :param graph_size: size of graph embeddings
        """
        super().__init__()
        self.adj = adj
        self.c_attn = GATConv(code_size, hidden_size, attention_size, dropout=dropout)
        self.n_attn = GATConv(code_size, hidden_size, attention_size, dropout=dropout)
        self.dense = nn.Linear(hidden_size, graph_size)
        self.activation = nn.LeakyReLU()

    def create_mask(self, source_mask, target_mask):
        """
        Create attention mask based on source nodes, target nodes connectivity and the adjacent matrix
        :param source_mask: shape code_num x 1
        :param target_mask: shape code_num x 1
        """
        conn_mask = (self.adj > 0).to(source_mask.device, dtype=torch.float32)
        source_mask = (source_mask > 0).to(torch.float32)
        target_mask = (target_mask > 0).to(torch.float32)
        mask = source_mask * conn_mask * target_mask.mT
        return mask

    def get_embeddings(self, code_x, neighbor, c_embeddings, n_embeddings):
        center_mask = code_x > 0
        neighbor_mask = neighbor > 0
        center_embeddings = c_embeddings[center_mask, :]
        neighbor_embeddings = n_embeddings[neighbor_mask, :]
        cc_embeddings = 0
        cn_embeddings = 0
        nn_embeddings = 0
        nc_embeddings = 0
        if center_embeddings.size(0) > 0:
            cc_embeddings = self.c_attn(center_embeddings, center_embeddings,
                                        mask=self.create_mask(code_x.unsqueeze(-1), code_x.unsqueeze(-1))[center_mask,
                                             :][:, center_mask])
        if neighbor_embeddings.size(0) > 0:
            nn_embeddings = self.n_attn(neighbor_embeddings, neighbor_embeddings,
                                        mask=self.create_mask(neighbor.unsqueeze(-1), neighbor.unsqueeze(-1))[
                                             neighbor_mask, :][:,
                                             neighbor_mask])
        if center_embeddings.size(0) > 0 and neighbor_embeddings.size(0) > 0:
            cn_embeddings = self.c_attn(center_embeddings, neighbor_embeddings,
                                        mask=self.create_mask(code_x.unsqueeze(-1), neighbor.unsqueeze(-1))[center_mask,
                                             :][:,
                                             neighbor_mask])
            nc_embeddings = self.n_attn(neighbor_embeddings, center_embeddings,
                                        mask=self.create_mask(neighbor.unsqueeze(-1), code_x.unsqueeze(-1))[
                                             neighbor_mask, :][:,
                                             center_mask])
        new_c_embeddings = torch.zeros_like(c_embeddings, device=c_embeddings.device, dtype=c_embeddings.dtype)
        new_n_embeddings = torch.zeros_like(n_embeddings, device=n_embeddings.device, dtype=n_embeddings.dtype)
        new_c_embeddings[center_mask, :] = center_embeddings + cc_embeddings + cn_embeddings
        new_n_embeddings[neighbor_mask, :] = neighbor_embeddings + nn_embeddings + nc_embeddings

        c_embeddings = c_embeddings * (1 - center_mask.float().unsqueeze(-1)) + new_c_embeddings
        n_embeddings = n_embeddings * (1 - neighbor_mask.float().unsqueeze(-1)) + new_n_embeddings
        return c_embeddings, n_embeddings

    def forward(self, code_x, neighbor, c_embeddings, n_embeddings):
        """
        :param code_x: center code, shape code_num
        :param neighbor: neighbor code, shape code_num
        :param c_embeddings: center embeddings, shape code_num x code_size
        :param n_embeddings: neighbor embeddings, shape code_num x code_size
        """
        c_embeddings, n_embeddings = self.get_embeddings(code_x, neighbor, c_embeddings, n_embeddings)
        co_embeddings = self.activation(self.dense(c_embeddings))
        no_embeddings = self.activation(self.dense(n_embeddings))
        return co_embeddings, no_embeddings  # code_num x graph_size, code_num x graph_size


class FusionGraphLayer(nn.Module):
    def __init__(self, adj, code_size, graph_size, attention_size=64, dropout=0.1):
        """
        Deprecated, Combined Graph GCN and Graph attention for graph aggregation
        :param adj: adjacency matrix, shape code_num x code_num
        :param code_size: size of code embeddings
        :param graph_size: size of graph embeddings
        """
        super().__init__()
        self.adj = adj
        self.gcn = GraphLayer(adj, code_size, graph_size)
        self.gcn.dense = nn.Identity()
        self.gcn.activation = nn.Identity()
        self.gat = GATGraphLayer(adj, code_size, code_size, graph_size, attention_size, dropout)
        self.gat.activation = nn.Identity()
        self.gat.dense = nn.Identity()
        self.dense = nn.Linear(code_size, graph_size)
        self.activation = nn.LeakyReLU()
        self.w_c = nn.Parameter(data=nn.init.zeros_(torch.empty(self.adj.size(0), 1)), requires_grad=True)
        self.w_n = nn.Parameter(data=nn.init.zeros_(torch.empty(self.adj.size(0), 1)), requires_grad=True)

    def forward(self, code_x, neighbor, c_embeddings, n_embeddings):
        gcn_c_embeddings, gcn_n_embeddings = self.gcn.get_embedding(code_x, neighbor, c_embeddings,
                                                                    n_embeddings)  # code_num x code_size
        gat_c_embeddings, gat_n_embeddings = self.gat.get_embeddings(code_x, neighbor, gcn_c_embeddings,
                                                                     gcn_n_embeddings)  # code_num x graph_size

        w_c = self.w_c
        w_n = self.w_n

        c_embeddings = gcn_c_embeddings * (1 - w_c) + gat_c_embeddings * w_c  # weighted residual
        n_embeddings = gcn_n_embeddings * (1 - w_n) + gat_n_embeddings * w_n  # weighted residual
        co_embeddings = self.activation(self.dense(c_embeddings))
        no_embeddings = self.activation(self.dense(n_embeddings))
        return co_embeddings, no_embeddings  # code_num x graph_size, code_num x graph_size


class TransitionLayer(nn.Module):
    def __init__(self, code_num, graph_size, hidden_size, t_attention_size, t_output_size):
        """
        Transition layer for temporal graph/admission embedding transition
        :param code_num: number of diseases
        :param graph_size: size of graph embeddings
        :param hidden_size: size of GRU hidden states
        :param t_attention_size: size of attention layer
        :param t_output_size: size of output features
        """
        super().__init__()
        self.gru = nn.GRUCell(input_size=graph_size, hidden_size=hidden_size)
        self.single_head_attention = SingleHeadAttentionLayer(graph_size, graph_size, t_output_size, t_attention_size)
        self.activation = nn.Tanh()

        self.code_num = code_num
        self.hidden_size = hidden_size

    def forward(self, t, co_embeddings, divided, no_embeddings, unrelated_embeddings, hidden_state=None):
        """
        Temporal transition layer, persistent diseases and emerging diseases are processed separately.
        Persistent diseases are processed by GRU, emerging diseases are processed by attention mechanism.
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


class TransitionNoteAttentionLayer(nn.Module):
    def __init__(self, code_num, graph_size, hidden_size, t_attention_size, t_output_size, n_attention_size,
                 note_size=768, n_attention_heads=8):
        """
        Deprecated, Transition layer with note attention mechanism at the end to aggregate admission notes into the final embedding
        """
        super().__init__()
        self.gru = nn.GRUCell(input_size=graph_size, hidden_size=hidden_size)
        self.single_head_attention = SingleHeadAttentionLayer(graph_size, graph_size, t_output_size, t_attention_size)
        self.note_attention_layer = MultiHeadAttentionLayer(note_size, hidden_size, hidden_size, n_attention_size,
                                                            output_size=hidden_size, num_heads=n_attention_heads)
        self.activation = nn.Tanh()
        self.norm = nn.LayerNorm(hidden_size)
        self.out_activation = nn.LeakyReLU()
        self.code_num = code_num
        self.hidden_size = hidden_size

    def forward(self, t, co_embeddings, divided, no_embeddings, unrelated_embeddings, note_embeddings,
                hidden_state=None):
        """
        :param t: visit t
        :param co_embeddings: diagnosed disease embeddings, shape code_num x graph_size
        :param divided: divided, shape code_num x 3
        :param no_embeddings: previous undiagnosed neighbor embeddings, shape code_num x graph_size
        :param unrelated_embeddings: unrelated disease embeddings, shape code_num x graph_size
        :param note_embeddings: note embeddings, shape 1 x note_size
        :param hidden_state: hidden state, shape code_num x hidden_size
        """
        m1, m2, m3 = divided[:, 0], divided[:, 1], divided[:, 2]  # code_num, code_num, code_num
        m1_index = torch.where(m1 > 0)[0]  # disease id that appear in visit t and t-1
        m2_index = torch.where(m2 > 0)[
            0]  # disease id that appear in visit t and is an undiagnosed neighbor in visit t-1
        m3_index = torch.where(m3 > 0)[0]  # disease id that appear in visit t and is an unrelated disease in visit t-1
        h_new = torch.zeros((self.code_num, self.hidden_size), dtype=co_embeddings.dtype).to(co_embeddings.device)
        attention_mask = torch.zeros(self.code_num, device=co_embeddings.device)

        if len(m1_index) > 0:  # embedding of persistent diseases
            m1_embedding = co_embeddings[m1_index]  # select m1 embeddings, len(m1_index) x graph_size
            h = hidden_state[
                m1_index] if hidden_state is not None else None  # select corresponding hidden states, len(m1_index) x hidden_size
            h_m1 = self.gru(m1_embedding, h)  # len(m1_index) x hidden_size
            h_new[m1_index] = h_m1  # fill in the result
            attention_mask[m1_index] = 1
        if t > 0 and len(m2_index) + len(m3_index) > 0:  # embedding of emerging disease
            q = torch.vstack([no_embeddings[m2_index], unrelated_embeddings[
                m3_index]])  # (len(m2_index) + len(m3_index)) x graph_size, neighbor embeddings of last visit + unrelated embeddings of last visit
            v = torch.vstack([co_embeddings[m2_index], co_embeddings[
                m3_index]])  # (len(m2_index) + len(m3_index)) x graph_size, diagnosed emerging disease embeddings of this visit
            h_m23 = self.activation(self.single_head_attention(q, q, v))
            h_new[m2_index] = h_m23[:len(m2_index)]
            h_new[m3_index] = h_m23[len(m2_index):]
            attention_mask[m2_index] = 1
            attention_mask[m3_index] = 1
        output = self.note_attention_layer(note_embeddings, h_new, h_new, attention_mask)
        output = self.norm(output)
        output = self.out_activation(output)
        return output, h_new
