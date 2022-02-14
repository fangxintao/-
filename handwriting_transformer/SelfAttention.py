import torch
import torch.nn as nn
import math


class SelfAtttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAtttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), 'Embed size needs to be div by heads'
        # 传入query, key, value, 通过线性层？
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # values==keys==query the shape of input is [batch_size, sequence_len, embed_size]
        N = query.shape[0]
        # it means how many examples are sent in at the same time
        query_len, key_len, value_len = query.shape[1], keys.shape[1], values.shape[1]
        # the length of key, value, query, correspend to source sentence length
        # Split embedding [N, value_len, embed_size] into self.heads piece
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # muitiple the queries with the keys
        energy = torch.einsum("nqhd,nkhd -> nhqk", [queries, keys])
        # queriess shape: (N, query_len, heads, heads_dim)
        # keys shape: (N, key_len, heads, head_dim)
        # energy shape : (N, heads, query_len, key_len)
        # torch.dmm

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
            # shape of mask : (N, 1, 1, src_len)
            # The mean to close the mask is replace the element with a float, set to a very small value
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        # dim = 3 means that we are normalizing across the key length, 此时query为souce sentence，key为target sentence
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )  # do the concatenation part by reshape
        # query的长度是否和 key,value一样
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, head_dim)
        # after einsum: (N, query_len, heads, head_dim)
        out = self.fc_out(out)
        # map the embedding_size to embedding_size
        return out


class TransfomerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransfomerBlock, self).__init__()
        self.attention = SelfAtttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        # do the normalization
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            # in papaer , forward_expansion is set as 4
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        # nn.sequential: A sequential container. Modules will be added to it in the order they are passed in the construct
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        # shape of scr_mask : (N, 1, 1, src_len)
        attention = self.attention(value, key, query, mask)
        # we gonna to send in skip connection
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            embed_size,
            num_layer,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        # src_vocab_size=10, embed_size=256
        self.position_embedding = nn.Embedding(max_length, embed_size)
        # max_length=100
        self.layers = nn.ModuleList(  # 和sequential有什么区别
            [
                TransfomerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
            for _ in range(num_layer)
            ]
        )
        # nn.ModuleList: holds submodules in a list
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # shape of scr_mask : (N, 1, 1, src_len)
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        # position 0,1,2,3... shape of position [2,9]
        # word_embedding = self.word_embedding(x)
        # pos_embedding = self.position_embedding(positions)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAtttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransfomerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)
        # dropout 0.0

    def forward(self, x, value, key, src_mask, trg_mask):
        # src_mask for select true text from paded text
        # trg_mask for multi-head attention
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))  # 需要理解残差网络
        out = self.transformer_block(value, key, query, src_mask)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # print(position.shape)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # e^{2i * -log(10000)/d} == {10000}^{-2i/d}
        # print(div_term.shape)
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0).transpose(0, 1)
        # shape of pe : [seq_len, batch_size, d_model]
        # self.register_buffer('pe', pe)

    def forward(self, x):

        # x: [seq_len, batch_size, d_model]

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Decoder(nn.Module):
    def __init__(self,
                 trg_vocab_size,
                 embed_size,
                 num_layer,
                 heads,
                 forward_expansion,
                 dropout,
                 device,
                 max_length,
                 ):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        # self.position_embedding = nn.Embedding(max_length, embed_size)
        self.position_encodding = PositionalEncoding(embed_size)

        self.layers = nn.ModuleList(  # 和sequential有什么区别
            [
                DecoderBlock(
                    embed_size,
                    heads,
                    forward_expansion=forward_expansion,
                    dropout=dropout,
                    device=device
                )
                for _ in range(num_layer)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out, src_mask, trg_mask):
        word_embedding = self.word_embedding(x)
        # shape of word_embedding : [batch_size, seq_len, d_model]
        pos_embedding = self.position_encodding(word_embedding.transpose(0, 1)).transpose(0, 1)
        # shape of pos_embedding : [batch_size, seq_len, d_model]
        x = self.dropout(word_embedding + pos_embedding)
        for layer in self.layers:
            x = layer(x, encoder_out, encoder_out, src_mask, trg_mask)
        out = self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
             # src_pad_idx and trg_pad_idx for compute the mask
            trg_pad_idx,
            embed_size=256,
            num_layers=6,
            forward_expansion=4,
            heads=8,
            dropout=0,
            device='cpu',
            max_length=100
            ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # shape of scr_mask : (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        # shape of trg_mask (N, 1, trg_len, trg_len)

        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        # shape of scr_mask : (N, 1, 1, src_len)
        # shape of trg_mask (N, 1, trg_len, trg_len)
        # shape of scr : (N, src_len)
        # shape of trg : (N, trg_len)
        trg_mask = self.make_trg_mask(trg)
        encoder_src = self.encoder(src, src_mask)
        out = self.decoder(trg, encoder_src, src_mask, trg_mask)
        return out


if __name__ == '__main__':
    device = torch.device('cpu')

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0, 0, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2, 0, 0]]).to(
        device
    )
    # shape of x : [2, 11]

    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0, 0], [1, 5, 6, 2, 4, 7, 6, 2, 0]]).to(device)
    # shape of trg : [2, 9]
    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    # print(trg[:, :-1])
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)
    out = model(x, trg)
    # target doesn't have the end of sentence token, because we want to learn to predict the end of sentence
    # print(out)
    print(out.shape)
    # print(torch.__version__)
    # print(torch.cuda.is_available())