import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadAttentionLayer(nn.Module):
    def __init__(self, query_size, key_size, value_size, attention_size):
        """
        Single head attention layer
        :param query_size: int, size of input query embedding
        :param key_size: int, size of input key embedding
        :param value_size: int, size of input value embedding
        """
        super().__init__()
        self.attention_size = attention_size
        self.dense_q = nn.Linear(query_size, attention_size)
        self.dense_k = nn.Linear(key_size, attention_size)
        self.dense_v = nn.Linear(query_size, value_size)

    def forward(self, q, k, v):
        query = self.dense_q(q)
        key = self.dense_k(k)
        value = self.dense_v(v)
        g = torch.div(torch.matmul(query, key.T), math.sqrt(self.attention_size))
        score = torch.softmax(g, dim=-1)
        output = torch.sum(torch.unsqueeze(score, dim=-1) * value, dim=-2)
        return output


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, query_size, key_size, value_size, attention_size, output_size, num_heads):
        """
        Multi head attention layer with attention mask
        :param query_size: int, size of input query embedding
        :param key_size: int, size of input key embedding
        :param value_size: int, size of input value embedding
        :param attention_size: int, size of attention hidden layer
        :param output_size: int, size of output embedding
        :param num_heads: int, number of heads, must be divisible by attention_size
        """
        super().__init__()
        assert attention_size % num_heads == 0, "Attention size must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = attention_size // num_heads

        self.dense_q = nn.Linear(query_size, attention_size)
        self.dense_k = nn.Linear(key_size, attention_size)
        self.dense_v = nn.Linear(value_size, output_size)

    def forward(self, q, k, v, attention_mask=None):
        """
        q: n x query_size
        k: m x key_size
        v: m x value_size
        attention_mask: m or None
        """
        num_dim = len(q.shape)
        if len(q.shape) == 2:
            q = q.unsqueeze(0)
        if len(k.shape) == 2:
            k = k.unsqueeze(0)
        if len(v.shape) == 2:
            v = v.unsqueeze(0)

        batch_size = q.size(0)

        # Linear projections
        query = self.dense_q(q)  # n x attention_size
        key = self.dense_k(k)  # m x attention_size
        value = self.dense_v(v)  # m x attention_size

        # Reshape to (batch_size, num_heads, seq_length, head_dim)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,
                                                                                    2)  # (n, heads, seq_len, head_dim)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (m, heads, seq_len, head_dim)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,
                                                                                    2)  # (m, heads, seq_len, head_dim)

        # Scaled dot-product attention
        g = torch.div(torch.matmul(query, key.transpose(-2, -1)),
                      math.sqrt(self.head_dim))  # (n, heads, seq_len_q, seq_len_k)
        if attention_mask is not None:
            g[:, :, :, attention_mask == 0] = float('-inf')  # mask attention

        score = torch.softmax(g, dim=-1)  # (n, heads, seq_len_q, seq_len_k)
        output = torch.matmul(score, value)  # (n, heads, seq_len_q, head_dim)

        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1,
                                                          self.num_heads * self.head_dim)  # (n, seq_len_q, attention_size)

        if num_dim == 2:
            output = output.squeeze(0)  # remove batch dim
        return output


class DotProductAttention(nn.Module):
    def __init__(self, value_size, attention_size):
        """
        Dot product attention layer, input embeddings are aggregated by attending to a hidden context embedding
        """
        super().__init__()
        self.attention_size = attention_size
        self.context = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(attention_size, 1)))
        self.dense = nn.Linear(value_size, attention_size)

    def forward(self, x):
        """
        x: graph embeddings, n x graph_size
        """
        t = self.dense(x)  # n x attention_size
        vu = torch.matmul(t, self.context).squeeze()  # n
        score = torch.softmax(vu, dim=-1)  # n
        output = torch.sum(x * torch.unsqueeze(score, dim=-1), dim=-2)  # graph_size
        return output


class MultiHeadAttentionWithResidualLayer(nn.Module):
    def __init__(self, query_size, key_size, value_size, attention_size, output_size, num_heads):
        """
        Multi head attention layer with residual connection
        :param query_size: int, size of input query embedding
        :param key_size: int, size of input key embedding
        :param value_size: int, size of input value embedding
        :param attention_size: int, size of attention hidden layer
        :param output_size: int, size of output embedding
        :param num_heads: int, number of heads, must be divisible by attention_size
        """
        super().__init__()
        self.num_heads = num_heads
        self.attention_size = attention_size
        self.head_size = attention_size // num_heads
        assert self.head_size * num_heads == attention_size, "Attention size must be divisible by num_heads"

        self.dense_q = nn.Linear(query_size, attention_size)
        self.dense_k = nn.Linear(key_size, attention_size)
        self.dense_v = nn.Linear(value_size, attention_size)

        self.output_proj = nn.Linear(attention_size, output_size)
        self.norm = nn.LayerNorm(output_size)

    def scaled_dot_product_attention(self, queries, keys, values, mask=None):

        # Scaled dot-product attention
        d_k = queries.size(-1)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)

            scores = scores + ((1 - mask) * float('-inf'))  # mask the score
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, values)
        return attention_output, attention_weights

    def forward(self, q, k, v, attention_mask=None):
        """
                q: n x query_size
                k: m x key_size
                v: m x value_size
                attention_mask: m or None
                """
        num_dim = len(q.shape)
        if len(q.shape) == 2:
            q = q.unsqueeze(0)
        if len(k.shape) == 2:
            k = k.unsqueeze(0)
        if len(v.shape) == 2:
            v = v.unsqueeze(0)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(0)
        batch_size, num_q, embed_dim = q.size()
        batch_size, num_kv, embed_dim = k.size()
        queries = self.dense_q(q)
        keys = self.dense_k(k)
        values = self.dense_v(v)
        # print(queries.shape, keys.shape, values.shape, attention_mask.shape)
        # Reshape to (batch_size, num_heads, seq_len, head_dim)
        queries = queries.view(batch_size, num_q, self.num_heads, self.head_size).transpose(1, 2)
        keys = keys.view(batch_size, num_kv, self.num_heads, self.head_size).transpose(1, 2)
        values = values.view(batch_size, num_kv, self.num_heads, self.head_size).transpose(1, 2)

        attention_output, _ = self.scaled_dot_product_attention(queries, keys, values, mask=attention_mask)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, num_q, self.attention_size)
        output = self.output_proj(attention_output)
        if num_dim == 2:
            output = output.squeeze(0)
            q = q.squeeze(0)
        return self.norm(output) + q


if __name__ == "__main__":
    attn = MultiHeadAttentionWithResidualLayer(32, 32, 64, 256, output_size=32, num_heads=8)
    q = torch.randn(6, 32)
    k = torch.randn(1, 32)
    v = torch.randn(1, 64)
    print(attn(q, k, v).shape)
    print(attn(q, k, v))
