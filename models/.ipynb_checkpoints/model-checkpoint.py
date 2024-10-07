import torch
from torch import nn
from models.layers import EmbeddingLayer, GraphLayer, TransitionLayer
from models.utils import DotProductAttention

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
    def __init__(self, code_num, code_size, adj, graph_size, hidden_size, t_attention_size, t_output_size, output_size, dropout_rate, activation, use_text_embeddings = False):
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
        self.use_text_embeddings=  use_text_embeddings

        self.embedding_layer = EmbeddingLayer(code_num, code_size, graph_size, self.use_text_embeddings)
        self.graph_layer = GraphLayer(adj, code_size, graph_size)
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
        c_embeddings, n_embeddings, u_embeddings = embeddings # c_embeddings: code_num x code_size, n_embeddings: code_num x code_size, u_embeddings: code_num x graph_size
        output = []
        for code_x_i, divided_i, neighbor_i, len_i in zip(code_x, divided, neighbors, lens): # patient i
            no_embeddings_i_prev = None
            output_i = []
            h_t = None
            for t, (c_it, d_it, n_it,len_it) in enumerate(zip(code_x_i, divided_i, neighbor_i, range(len_i))): # visit t
                # c_it: code_num, d_it: code_num x 3, n_it: code_num
                co_embeddings, no_embeddings = self.graph_layer(c_it, n_it, c_embeddings, n_embeddings) # co_embeddings: code_num x graph_size, no_embeddings: code_num x graph_size
                output_it, h_t = self.transition_layer(t, co_embeddings, d_it, no_embeddings_i_prev, u_embeddings, h_t)
                no_embeddings_i_prev = no_embeddings
                output_i.append(output_it)
            output_i = self.attention(torch.vstack(output_i)) # graph size
            output.append(output_i)
        output = torch.vstack(output) # B x graph_size
        output = self.classifier(output)
        return output

