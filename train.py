import matplotlib.pyplot as plt
import torch
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
from tqdm.notebook import tqdm

from model import SASRec


def show_metrics(metric1, metric2, label1, label2, name):
    plt.figure(num=name)
    plt.plot(range(len(metric1)), metric1, label=label1)
    plt.plot(range(len(metric2)), metric2, label=label2)
    plt.legend()
    plt.grid(True)
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
        neg_embeddings = model.output_embedding(negatives)
        pos_embeddings = model.output_embedding(positives)

        output = model(model_input)
        neg_logits = torch.matmul(output, neg_embeddings.t())
        pos_logits = torch.einsum("bse, bse -> bs", output, pos_embeddings)

        pos_logits = pos_logits.unsqueeze(-1)

        logits = torch.cat([neg_logits, pos_logits], dim=-1)
        gt = torch.cat([torch.zeros_like(neg_logits), torch.ones_like(pos_logits)], dim=-1)

        loss = binary_cross_entropy_with_logits(logits, gt, reduction="none")

        mask = (positives != model.pad_id)
        mask = mask.unsqueeze(-1).float()

        loss = loss * mask
        loss = loss.sum() / mask.sum()

        loss.backward()
        optimizer.step()

        sum_loss += loss.item()

    return sum_loss / len(train_loader)


@torch.no_grad()
def validate_epoch(model: SASRec, val_loader):
    model.eval()

    losses = []
    top1_correct = 0
    top5_correct = 0
    total = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_embeddings = model.output_embedding.weight  # V, E

    for labels in tqdm(val_loader, desc="Batch"):
        labels = labels.to(device)

        model_input = labels[:, :-1]
        targets: torch.Tensor = labels[:, 1:]

        model_output: torch.Tensor = model(model_input)  # B, S, E

        model_output_flat = model_output.flatten(end_dim=1)  # B*S, E
        targets_flat = targets.flatten()  # B*S

        mask = (targets_flat != model.pad_id)

        if sum(mask) == 0:
            continue

        valid_output = model_output_flat[mask]  # N, E
        valid_targets = targets_flat[mask]  # N

        logits = torch.matmul(valid_output, all_embeddings.t())  # N, V
        loss = cross_entropy(logits, valid_targets)
        losses.append(loss.item())

        top5_indices = torch.topk(logits, k=5, dim=-1).indices  # N, 5
        top1_pred = logits.argmax(dim=-1)  # N

        top1_correct += (top1_pred == valid_targets).sum().item()
        top5_correct += (top5_indices == valid_targets.unsqueeze(-1)).any(dim=-1).sum().item()
        total += mask.sum().item()

    avg_loss = sum(losses) / len(losses)
    top1_acc = top1_correct / total
    top5_acc = top5_correct / total

    return avg_loss, top1_acc, top5_acc


def train(model: SASRec, train_loader, val_loader, optimizer,
          epochs, scheduler=None):
    train_losses, val_losses = [], []
    top1_accs = []
    top5_accs = []

    for epoch in tqdm(range(epochs)):
        train_losses.append(train_epoch(model, train_loader, optimizer))

        sum_loss, top1_acc, top5_acc = validate_epoch(model, val_loader)

        val_losses.append(sum_loss)
        top1_accs.append(top1_acc)
        top5_accs.append(top5_acc)

        print(f"Epoch {epoch}: train_loss={train_losses[-1]:.4f}, val_loss={val_losses[-1]:.4f}, "
              f"top1={top1_acc:.4f}, top5={top5_acc:.4f}")

        show_metrics(train_losses, val_losses, "train", "val", "losses")
        show_metrics(top1_accs, top5_accs, "top-1", "top-5", "accuracy")

        if scheduler is not None:
            scheduler.step()
