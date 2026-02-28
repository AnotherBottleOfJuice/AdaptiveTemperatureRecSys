import torch
from torch import nn


class SASRec(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 seq_len: int,
                 unk: int,
                 embedding_dim: int = 256,
                 num_heads: int = 4,
                 num_layers: int = 4,
                 transformer_dim=2048,
                 transformer_dropout=0.1,
                 linear_dropout=0.1,
                 reuse_embeddings: bool = False):
        super(SASRec, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.input_embedding = nn.Embedding(num_embeddings=num_embeddings + 1, embedding_dim=embedding_dim)
        self.position_embedding = nn.Embedding(num_embeddings=seq_len, embedding_dim=embedding_dim)

        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=transformer_dim,
                dropout=transformer_dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
            enable_nested_tensor=False
        )

        self.linear = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.dropout = nn.Dropout(p=linear_dropout)

        if reuse_embeddings:
            self.output_embedding = self.input_embedding
        else:
            self.output_embedding = nn.Embedding(num_embeddings=num_embeddings + 1, embedding_dim=embedding_dim)

        self.pad_id = 0
        self.UNK = unk

    def forward(self, x):

        device = x.device

        input_embeddings = self.input_embedding(x)

        seq_len = input_embeddings.shape[1]

        positions = (torch.arange(seq_len, dtype=torch.long, device=device)
                     .unsqueeze(0).expand(x.size(0), seq_len))
        position_embeddings = self.position_embedding(positions)

        embeddings = input_embeddings + position_embeddings

        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)

        attention = self.transformer(embeddings,
                                     mask=causal_mask,
                                     is_causal=True)
        self.dropout(attention)
        return self.linear(attention)

