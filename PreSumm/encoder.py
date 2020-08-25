import math
import torch
import torch


class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.sigmoid(h)

        return sent_scores

    # 왜 hidden_size를 input으로 받는가?
    # 어차피 CLS TOKEN만 연관있는거 아니야?


class PositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.aragne(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if step:
            emb = emb + self.pe[:, step][:, None, :]
        else:
            emb = emb + self.pe[:, : emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, : emb.size(1)]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, heads, ffn_dim, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attention = MultiHeadedAttention(heads, embed_dim, droptout=dropout)
        self.feed_forward = PositionwiseFeedFoward(embed_dim, ffn_dim, dropout)
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iteration, query, inputs, mask):
        # why use query?
        if iteration != 0:
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs
        # 최초의 input은 normalization 하지 않음

        mask = mask.unsqueeze(1)
        context = self.self_attention(input_norm, input_norm, input_norm, mask=mask)
        out = self.dropout(context) + inputs
        out = self.feed_forward(out)

        return out

        """[Transformer Encoder 순서]
        1. Multihead-Attention
        2. Skip-Connection
        3. Feed-Foward
        4. Skip-Connection
        """


class ExtTransformerEncoder(nn.Module):
    def __init__(self, embed_dim, ffn_dim, heads, dropout, num_inter_layers=0):
        super(ExtTransformerEncoder, self).__init()
        self.embed_dim = embed_dim
        self.num_inter_layers = num_inter_layers
        self.position_embedding = PositionalEncoding(dropout, embed_dim)
        self.transformer_inter = nn.ModuleList(
            [
                TransformerEncoderLayer(embed_dim, heads, ffn_dim, dropout)
                for _ in range(num_inter_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.wo = nn.Linear(embed_dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        # top_vecs: last hidden state of BERT

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        position_embdding = self.position_embedding.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + position_embdding

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, 1 - mask)
            # transformer_inter => TransfoermerEncoderLayer Stacks
            # transformer_inter[0] = first layer
            # transformer_inter[num_inter_layers-1] = last layer

        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores
