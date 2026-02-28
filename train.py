import matplotlib.pyplot as plt
import torch
from torch.nn.functional import cross_entropy
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
    # all_embeddings = model.output_embedding.weight  # V, E

    for i, (positives, negatives) in enumerate(tqdm(train_loader, desc="Training")):
        optimizer.zero_grad()
        positives, negatives = positives.to(device), negatives.to(device)

        model_input = positives[:, :-1]  # B, S, E
        positives = positives[:, 1:]
        neg_embeddings = model.output_embedding(negatives)
        pos_embeddings = model.output_embedding(positives)

        output = model(model_input)  # B, S, E
        # print(neg_embeddings.shape)
        neg_logits = torch.matmul(output, neg_embeddings.T)
        pos_logits = (output * pos_embeddings).sum(dim=-1, keepdim=True)
        # all_logits = torch.matmul(output, all_embeddings.t())

        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        logits = logits.reshape(logits.size(0) * logits.size(1), -1)
        loss = cross_entropy(logits, torch.zeros(logits.shape[0],
                                                 device=logits.device,
                                                 dtype=torch.long))

        # loss = cross_entropy(all_logits.flatten(end_dim=1), positives.flatten())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        sum_loss += loss.item()

    return sum_loss / len(train_loader)


@torch.no_grad()
def validate_epoch(model: SASRec, val_loader):
    model.eval()

    losses = []
    top100_correct = 0
    top10_correct = 0
    total = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_embeddings = model.output_embedding.weight  # V, E

    for labels in tqdm(val_loader, desc="Validation"):
        labels = labels.to(device)

        model_input = labels[:, :-1]
        targets: torch.Tensor = labels[:, 1:]

        model_output: torch.Tensor = model(model_input)  # B, S, E

        model_output_flat = model_output.flatten(end_dim=1)  # B*S, E
        targets_flat = targets.flatten()  # B*S

        mask = (targets_flat != model.pad_id)  # B*S

        if sum(mask) == 0:
            continue

        valid_output = model_output_flat[mask]  # N, E
        valid_targets = targets_flat[mask]  # N

        logits = torch.matmul(valid_output, all_embeddings.t())  # N, V
        loss = cross_entropy(logits, valid_targets)
        losses.append(loss.item())

        top100_indices = torch.topk(logits, k=100, dim=-1).indices  # N, 10
        top10_indices = top100_indices[:, :10]  # N, 10

        top10_correct += (top10_indices == valid_targets.unsqueeze(-1)).any(dim=-1).sum().item()
        top100_correct += (top100_indices == valid_targets.unsqueeze(-1)).any(dim=-1).sum().item()

        total += mask.sum().item()

    avg_loss = sum(losses) / len(losses)
    top10_acc = top10_correct / total
    top100_acc = top100_correct / total

    return avg_loss, top100_acc, top10_acc


def train(model: SASRec, train_loader, val_loader, optimizer,
          epochs, scheduler=None):
    train_losses, val_losses = [], []
    top100_accs = []
    top10_accs = []

    for epoch in tqdm(range(epochs)):
        train_losses.append(train_epoch(model, train_loader, optimizer))

        sum_loss, top100_acc, top10_acc = validate_epoch(model, val_loader)

        val_losses.append(sum_loss)
        top100_accs.append(top100_acc)
        top10_accs.append(top10_acc)

        print(f"Epoch {epoch}: train_loss={train_losses[-1]:.4f}, val_loss={val_losses[-1]:.4f}, "
              f"top10={top10_acc:.4f}, top100={top100_acc:.4f}")

        show_metrics(train_losses, val_losses, "train", "val", "losses")
        show_metrics(top100_accs, top10_accs, "top-100", "top-10", "accuracy")

        if scheduler is not None:
            scheduler.step()