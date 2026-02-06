import torch
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

from model import SASRec


def show_losses(train_losses, val_losses, name):
    plt.figure(num=name)
    plt.plot([i for i in range(len(train_losses))], train_losses, label="train")
    plt.plot([i for i in range(len(train_losses))], val_losses, label="val")
    plt.legend()
    plt.show()


def train_epoch(model: SASRec, train_loader, optimizer):
    model.train()
    sum_loss = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, (positives, negatives) in enumerate(tqdm(train_loader, desc="Batch")):
        optimizer.zero_grad()
        positives, negatives = positives.to(device), negatives.to(device)

        model_input = positives[:, :-1]  # B, S, E
        positives = positives[:, 1:]
        negatives = negatives[:, 1:, :]  # B, S, N, E
        neg_embeddings = model.output_embedding(negatives)
        pos_embeddings = model.output_embedding(positives)

        output = model(model_input)
        neg_logits = torch.einsum("bse, bsne -> bsn", output, neg_embeddings)
        pos_logits = torch.einsum("bse, bse -> bs", output, pos_embeddings)

        pos_logits = pos_logits.unsqueeze(-1)

        logits = torch.cat([neg_logits, pos_logits], dim=-1)
        gt = torch.cat([torch.zeros_like(neg_logits), torch.ones_like(pos_logits)], dim=-1)

        loss = binary_cross_entropy_with_logits(logits, gt)
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()

    return sum_loss / len(train_loader)


def validate_epoch(model: SASRec, val_loader):
    model.eval()

    sum_loss = 0
    top1_acc = 0
    top5_acc = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_embeddings = model.output_embedding.weight

    for i, labels in enumerate(tqdm(val_loader, desc="Batch")):
        labels = labels.to(device)

        model_input = labels[:, :-1]
        labels = labels[:, 1:]

        model_output = model(model_input)

        logits = torch.einsum("bse, ve -> bsv", model_output, all_embeddings)

        logits = logits.reshape(-1, logits.size(-1))
        labels = labels.reshape(-1)

        loss = cross_entropy(logits, labels)
        sum_loss += loss.item()

        top1_logits = logits.topk(1, dim=-1).indices
        top5_logits = logits.topk(5, dim=-1).indices

        gt = (labels != model.pad_id)
        num_predicts = gt.sum().item()

        top1_acc += ((labels == top1_logits) & gt).sum().item() / num_predicts
        top5_acc += ((labels.unsqueeze(-1) == top5_logits).any(dim=-1) & gt).sum().item() / num_predicts

    sum_loss /= len(val_loader)
    top1_acc /= len(val_loader)
    top5_acc /= len(val_loader)
    return sum_loss, top1_acc, top5_acc


def train(model: SASRec, train_loader, val_loader, optimizer, epochs):
    train_losses, val_losses = [], []
    top1_accs = []
    top5_accs = []

    for _ in tqdm(range(epochs)):
        train_losses.append(train_epoch(model, train_loader, optimizer))
        sum_loss, top1_acc, top5_acc = validate_epoch(model, val_loader)

        val_losses.append(sum_loss)
        top1_accs.append(top1_acc)
        top5_accs.append(top5_acc)

        show_losses(train_losses, val_losses, "losses")
        show_losses(top1_accs, top5_acc, "accuracy")
