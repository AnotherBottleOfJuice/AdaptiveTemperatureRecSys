import torch

from model import Graph
from train import train_loop
from dataset import (TrainingDataset, get_train_histories, get_train_events,
                     get_general_data, get_item_to_token)

train, test, embeddings, artists, test_targets = get_general_data()
item_to_token = get_item_to_token(train)

train_events = get_train_events(train, item_to_token)
train_histories = get_train_histories(train_events)

dataloader = TrainingDataset(train_histories, batch_size=32, seq_len=100, shuffle=True, device='cuda')

graph = Graph(
    vocab_size=30001,
    max_seq_len=100,
    n_layers=4,
    dropout=0.1
).cuda()
compiled_graph = torch.compile(graph, mode="default")

optimizer = torch.optim.AdamW(
    compiled_graph.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
    fused=True
)

train_loop(
    compiled_graph,
    num_epochs=5,
    train_dataloader=dataloader,
    log_dir='logs/test',
    optimizer=optimizer,
    grad_clip=1.0,
    grad_accum_steps=1
)
