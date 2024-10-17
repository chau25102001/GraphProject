import torch
from torch import nn

from models.layers import EmbeddingLayer, GraphLayer, TransitionLayer, TransitionNoteAttentionLayer, GraphLayer2, GATGraphLayer, FusionGraphLayer
from models.utils import DotProductAttention, MultiHeadAttentionWithResidualLayer


class Classifier(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.0, activation=None):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        output = self.dropout(x)
        output = self.linear(output)
        if self.activation is not None:
            output = self.activation(output)
        return output


class Model(nn.Module):
    def __init__(self, code_num, code_size, adj, graph_size, hidden_size, t_attention_size, t_output_size, output_size,
                 dropout_rate, activation, graph_layer_type='gat'):
        """
        :param code_num: number of diseases
        :param code_size: size of disease code embeddings
        :param adj: adjacency matrix
        :param graph_size: size of graph embeddings
        :param hidden_size: size of GRU hidden states
        :param t_attention_size: size of attention in transition layer
        :param t_output_size: size of output in transition layer
        :param output_size: size of output
        :param dropout_rate: dropout rate
        """
        super().__init__()

        self.embedding_layer = EmbeddingLayer(code_num, code_size, graph_size)
        if graph_layer_type == "gcn":
            self.graph_layer = GraphLayer(adj, code_size, graph_size)
        elif graph_layer_type == 'gat':
            self.graph_layer = GATGraphLayer(adj, code_size, code_size, graph_size, attention_size=64, dropout=0.1)
        elif graph_layer_type == 'fusion':
            self.graph_layer = FusionGraphLayer(adj, code_size, graph_size, attention_size=64, dropout=0.1)
        else:
            raise ValueError(f"Invalid graph layer type: {graph_layer_type}")
        self.transition_layer = TransitionLayer(code_num, graph_size, hidden_size, t_attention_size, t_output_size)
        self.attention = DotProductAttention(hidden_size, 32)
        self.classifier = Classifier(hidden_size, output_size, dropout_rate, activation)

    def forward(self, code_x, divided, neighbors, lens):
        """
        :param code_x: B x max_admission_num x code_num: binary tensor, code_x[i][j][k] = 1 if patient i has disease k at visit j
        :param divided: B x max_admission_num x code_num x 3: binary tensor,
                divided[i][j][k][0] = 1 if patient i has disease k at visit j and j-1
                divided[i][j][k][1] = 1 if patient i has disease k at visit j and k is an undiagnosed neighbor disease in visit j - 1
                divided[i][j][k][2] = 1 if patient i has disease k at visit j and k is an unrelated disease in visit j - 1 (neither diagnosed nor a neighbor)
        :param neighbors: B x max_admission_num x code_num: binary tensor, neighbors[i][j][k] = 1 if disease k is an undiagnosed neighbor of patient i at visit j
        :param lens: B: list of number of visits for each patient
        """
        embeddings = self.embedding_layer()
        c_embeddings, n_embeddings, u_embeddings = embeddings  # c_embeddings: code_num x code_size, n_embeddings: code_num x code_size, u_embeddings: code_num x graph_size
        output = []
        for code_x_i, divided_i, neighbor_i, len_i in zip(code_x, divided, neighbors, lens):  # patient i
            no_embeddings_i_prev = None
            output_i = []
            h_t = None
            for t, (c_it, d_it, n_it, len_it) in enumerate(
                    zip(code_x_i, divided_i, neighbor_i, range(len_i))):  # visit t
                # c_it: code_num, d_it: code_num x 3, n_it: code_num
                co_embeddings, no_embeddings = self.graph_layer(c_it, n_it, c_embeddings,
                                                                n_embeddings)  # co_embeddings: code_num x graph_size, no_embeddings: code_num x graph_size
                output_it, h_t = self.transition_layer(t, co_embeddings, d_it, no_embeddings_i_prev, u_embeddings, h_t)
                no_embeddings_i_prev = no_embeddings
                output_i.append(output_it)
            output_i = self.attention(torch.vstack(output_i))  # graph size
            output.append(output_i)
        output = torch.vstack(output)  # B x graph_size
        output = self.classifier(output)
        return output


class ModelNoteAttentionV2(nn.Module):
    def __init__(self, code_num, code_size, adj, graph_size, hidden_size, t_attention_size, t_output_size, output_size,
                 dropout_rate, activation,
                 n_attention_size, note_size, n_attention_heads=8):
        super().__init__()
        self.embedding_layer = EmbeddingLayer(code_num, code_size, graph_size)
        self.graph_layer = GraphLayer(adj, code_size, graph_size)
        self.transition_layer = TransitionLayer(code_num, graph_size, hidden_size, t_attention_size, t_output_size)
        self.attention = DotProductAttention(hidden_size, 32)
        self.classifier = Classifier(hidden_size, output_size, dropout_rate, activation)
        self.note_visit_attention = MultiHeadAttentionWithResidualLayer(hidden_size, note_size, note_size,
                                                                        n_attention_size, hidden_size,
                                                                        n_attention_heads)

    def forward(self, code_x, divided, neighbors, lens, note_embeddings, note_attention_mask=None):
        """
        :param code_x: B x max_admission_num x code_num: binary tensor, code_x[i][j][k] = 1 if patient i has disease k at visit j
        :param divided: B x max_admission_num x code_num x 3: binary tensor,
                divided[i][j][k][0] = 1 if patient i has disease k at visit j and j-1
                divided[i][j][k][1] = 1 if patient i has disease k at visit j and k is an undiagnosed neighbor disease in visit j - 1
                divided[i][j][k][2] = 1 if patient i has disease k at visit j and k is an unrelated disease in visit j - 1 (neither diagnosed nor a neighbor)
        :param neighbors: B x max_admission_num x code_num: binary tensor, neighbors[i][j][k] = 1 if disease k is an undiagnosed neighbor of patient i at visit j
        :param lens: B: list of number of visits for each patient
        :param note_embeddings: list of tensor of shape visit_len x note_size, length B
        """
        embeddings = self.embedding_layer()
        c_embeddings, n_embeddings, u_embeddings = embeddings  # c_embeddings: code_num x code_size, n_embeddings: code_num x code_size, u_embeddings: code_num x graph_size
        output = []
        for i, (code_x_i, divided_i, neighbor_i, len_i, note_embeddings_i) in enumerate(zip(code_x, divided, neighbors, lens,
                                                                             note_embeddings)):  # patient i
            no_embeddings_i_prev = None
            output_i = [] # len = visit_len, shape = hidden_size
            h_t = None
            mask = None
            if note_attention_mask is not None:
                mask = note_attention_mask[i]

            for t, (c_it, d_it, n_it, len_it) in enumerate(
                    zip(code_x_i, divided_i, neighbor_i, range(len_i))):  # visit t
                # c_it: code_num, d_it: code_num x 3, n_it: code_num
                co_embeddings, no_embeddings = self.graph_layer(c_it, n_it, c_embeddings,
                                                                n_embeddings)  # co_embeddings: code_num x graph_size, no_embeddings: code_num x graph_size
                output_it, h_t = self.transition_layer(t, co_embeddings, d_it, no_embeddings_i_prev, u_embeddings, h_t)
                no_embeddings_i_prev = no_embeddings
                output_i.append(output_it)
            # output_i = self.note_visit_attention(torch.vstack(output_i), note_embeddings_i, note_embeddings_i,
            #                                         )
            # output_i = self.attention(output_i)  # graph size
            output_i = self.attention(torch.vstack(output_i))  # graph size

            output.append(output_i)
        output = torch.vstack(output)  # B x graph_size
        output = self.classifier(output)
        return output


class ModelNoteAttention(nn.Module):
    def __init__(self, code_num, code_size, adj, graph_size, hidden_size, t_attention_size, t_output_size, output_size,
                 dropout_rate, activation,
                 n_attention_size, note_size, n_attention_heads=8):
        """
        :param code_num: number of diseases
        :param code_size: size of disease code embeddings
        :param adj: adjacency matrix
        :param graph_size: size of graph embeddings
        :param hidden_size: size of GRU hidden states
        :param t_attention_size: size of attention in transition layer
        :param t_output_size: size of output in transition layer
        :param output_size: size of output
        :param dropout_rate: dropout rate
        :param n_attention_size: size of attention in note attention layer
        :param note_size: size of note embeddings
        :param n_attention_heads: number of heads in note attention layer
        """
        super().__init__()
        self.embedding_layer = EmbeddingLayer(code_num, code_size, graph_size)
        self.graph_layer = GraphLayer(adj, code_size, graph_size)
        self.transition_layer = TransitionNoteAttentionLayer(code_num, graph_size, hidden_size, t_attention_size,
                                                             t_output_size, n_attention_size, note_size,
                                                             n_attention_heads)
        self.attention = DotProductAttention(hidden_size, 32)  # final attention layer
        self.classifier = Classifier(hidden_size, output_size, dropout_rate, activation)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.graph_layer_norm = nn.LayerNorm(graph_size)

    def forward(self, code_x, divided, neighbors, lens, note_embeddings):
        """
        :param code_x: B x max_admission_num x code_num: binary tensor, code_x[i][j][k] = 1 if patient i has disease k at visit j
        :param divided: B x max_admission_num x code_num x 3: binary tensor,
                divided[i][j][k][0] = 1 if patient i has disease k at visit j and j-1
                divided[i][j][k][1] = 1 if patient i has disease k at visit j and k is an undiagnosed neighbor disease in visit j - 1
                divided[i][j][k][2] = 1 if patient i has disease k at visit j and k is an unrelated disease in visit j - 1 (neither diagnosed nor a neighbor)
        :param neighbors: B x max_admission_num x code_num: binary tensor, neighbors[i][j][k] = 1 if disease k is an undiagnosed neighbor of patient i at visit j
        :param lens: B: list of number of visits for each patient
        :param note_embeddings: list of tensor of shape visit_len x note_size, length B
        """
        embeddings = self.embedding_layer()
        c_embeddings, n_embeddings, u_embeddings = embeddings  # c_embeddings: code_num x code_size, n_embeddings: code_num x code_size, u_embeddings: code_num x graph_size
        output = []
        for code_x_i, divided_i, neighbor_i, len_i, note_embeddings_i in zip(code_x, divided, neighbors, lens,
                                                                             note_embeddings):  # patient i
            no_embeddings_i_prev = None
            output_i = []
            h_t = None
            for t, (c_it, d_it, n_it, len_it, ne_it) in enumerate(
                    zip(code_x_i, divided_i, neighbor_i, range(len_i), note_embeddings_i)):  # visit t
                # c_it: code_num, d_it: code_num x 3, n_it: code_num, ne_it: note_size
                ne_it = ne_it.unsqueeze(-2)
                co_embeddings, no_embeddings = self.graph_layer(c_it, n_it, c_embeddings, n_embeddings)
                co_embeddings = self.graph_layer_norm(co_embeddings)
                output_it, h_t = self.transition_layer(t, co_embeddings, d_it, no_embeddings_i_prev, u_embeddings,
                                                       ne_it, h_t)
                no_embeddings_i_prev = no_embeddings
                output_i.append(output_it)
            output_i = self.attention(torch.vstack(output_i))  # graph size
            output.append(output_i)
        output = torch.vstack(output)  # B x graph_size
        output = self.batch_norm(output)
        output = self.classifier(output)
        return output
