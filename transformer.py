import torch
import torch.nn as nn
from typing import List, Dict, Tuple


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads) -> None:
        # embed size가 256, heads가 8이라면 8*32로 나누어질 수 있다 -> 어떻게 나누어지는가?
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size  # embed_size = 256
        self.heads = heads  # heads = 8
        self.head_dim = embed_size // heads  # head_dim = 32
        # 당연히 head_dim은 integer이므로 assert로 확인한다!

        assert (
            self.head_dim * heads == embed_size
        ), "Embed size needs to be div by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

        """
        * 차원을 head_dim(32)로 나누는 이유는 결국 나중에 Multihead attention으로 전환하기 위해서이다.
        * query / value / key는 모두 같은 차원
        * query = attention의 수혜자 (단어 하나)
        * keys = attention의 대상 (전체 단어 / 어디에 집중할 것이여!)
        * values = softmax 값을 얹는 대상
        * fc_out = heads들을 모두 concat하여 fc_out에 집어넣는다!
        """

    def forward(self, values, keys, query, mask) -> torch.tensor:
        N = query.shape[0]  # How many inputs we are going to send at the same time
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        # These Three will be correspond to source / target sentence length
        # 단순히 차원수이며, src / trg에 비례하는 이유는 heads 갯수에 따라 달라질 거기 때문에

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        """
        values, keys, query는 모두 밑에서 out, 즉 word_embedding이 적용된 encoder_input이다.
        encoder_input의 꼴은 다음과 같다.
        [3,4,2,6,7,1,8,8,6,6],
        [36,11,264,3,2,5,3,5]
        ....
        여기에 word_embedding이 적용이 되면, |Batch_Size * Sequence_Length * Embedding_Dimension|
        
        query는 values / keys와 다르게 복수형이 아니다 -> 즉, 단일한 값이라는 거다.
        """

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        # |values| = Batch_Size * Sequence_Length * Heads * Head_Dim
        # |keys|   = Batch_Size * Sequence_Length * Heads * Head_Dim
        # |queries|= Batch_Size * Sequence_Length * Heads * Head_Dim

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # n: batch_size
        # q: query_len
        # h: heads
        # d: heads dimension
        # k: key_len
        # einsum: bmm을 좀 더 쉽게 할 수 있다.
        # 차원을 알파벳으로 구성한 후 내가 원하는 차원의 Form으로 BMM으로 구성

        # |energy| = Batch_Size * Heads * Query_Length * Key_Length

        """
        matrix multiplication for various dimensions
        queries shape: (N, query_len, heads, heads_dim)
        keys shape: (N, key_len, heads, head_dim)
        energy shape: (N, heads, query_len, key_len)
        --> query_len: target / src sentence , key_len: src sentence
        query len이 얼만큼 key_len에 집중할 것인가?
        """
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # encoder에 mask가 필요한지 여쭈어봐도 되겠습니까?
        # -> Pad index에 대하여 weights를 0으로 치환

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # |attention| = Batch_Size * Heads * Query_Length * Key_Length

        # now multiply with value

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # |out| = Batch_Size * Query_Length * Embedding_Size

        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out return value = (N, query_len, heads, head_dim)

        # after einsum (N, query_len, heads, head_dim) then flatten last two dimensions

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion) -> None:
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # Layernorm - BatchNorm 은 매우 유사
        # BatchNorm은 배치의 평균을 구한 뒤 Normalize
        # Layer는 각 예시의 평균을 구한 뒤 Normalize
        # LayerNorm이 Computation이 더 많다

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            # forward_expansion - Node라는 개념과 연관, value = 4
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        add_norm = self.norm1(attention + query)
        x = self.dropout(add_norm)

        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

        # attention + query 자체가 skip connection
        # skip connection => 입력과 출력을 다음 레이어의 입력으로 보내는 것을 의미함


class Encoder(nn.Module):
    def __init(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
        # related to positional embedding
        # positional embedding은 문장 길이에 영향을 받으므로, 문장의 최대 길이를 input으로 넣어줘야 한다.
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        # range(a, b) = a부터 b까지
        # arange(a, b) = a부터 b-1까지
        # 왜 positional embedding에서 sinusidal function을 사용하지 않는 것이요?

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()

        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        # trg_mask는 필수이지만 src_mask는 선택이다.
        # src_mask는 <pad>에 대하여 masking을 하여 계산을 하지 않기 위해서이기 때문이다.

        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=256,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cuda",
        max_length=100,
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
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsequeeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

        # tril => triangular lower

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    """
    <PAD> = 0
    <SOS> = 1
    <EOS> = 2

    """

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(
        device
    )
    out = model(x, trg[:, :-1])
    print(out.shape)
