import torch
from torch.nn.functional import binary_cross_entropy_with_logits

import matplotlib.pyplot as plt

from tqdm import tqdm

from model import SASRec

def show_losses(train_losses, val_losses):
    plt.figure()
    plt.plot([i for i in range(len(train_losses))], train_losses, label="train loss")
    plt.plot([i for i in range(len(train_losses))], val_losses, label="val loss")
    plt.legend()
    plt.show()

def train_epoch(model : SASRec, train_loader, optimizer):
    model.train()
    sum_loss = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, (positives, negatives) in enumerate(tqdm(train_loader, desc="Batch")):
        optimizer.zero_grad()
        positives, negatives = positives.to(device), negatives.to(device)

        model_input = positives[:, :-1]  # B, S, E
        positives = positives[:, 1:]
        negatives = negatives[:, 1:, :] # B, S, N, E
        neg_embeddings = model.input_embedding(negatives)
        pos_embeddings = model.input_embedding(positives)

        output = model(model_input)
        neg_logits = torch.einsum("bse, bsne -> bsn", output, neg_embeddings)
        pos_logits = torch.einsum("bse, bse -> bs", output, pos_embeddings)

        logits = torch.cat([neg_logits, pos_logits], dim=0)
        gt = torch.cat([torch.zeros_like(neg_logits), torch.ones(pos_logits)], dim=0)

        loss = binary_cross_entropy_with_logits(logits, gt)
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()

    return sum_loss / len(train_loader)
