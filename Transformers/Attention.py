import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        """_summary_

        Args:
            embed_size (int): 임베딩 size
            heads (int): 헤드 수
        """
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        #예외처리 
        assert (self.head_dim * heads == embed_size), 'Embded size needs to be div by heads'

        self.values = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]


        # split embedding into self.heads
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = values.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd, nkhd -> nhqk", [queries, keys])
        #queries shape : (N, query_len, heads, heads_dim)
        #keys shape : (N, key_len, heads, heads_dim)
        #enery shape : (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        #attention 
        #self.embed_size = root dk
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        out = torch.einsum("nhql, nlhd -> nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim)
        # attention shape : (N, heads, query_len, key_len)
        # values shape : (N, value_len, heads, heads_dim)
        # (N, query_len, heads, head_dim) then flatten last two dimension

        out = self.fc_out(out)
        return out
    
