from typing import Dict

import torch


def get_metrics(targets: torch.Tensor, candidates: torch.Tensor, topk: int) -> Dict[str, float]:
    targets = targets.flatten().to(dtype=torch.long)
    candidates = candidates.flatten().to(dtype=torch.long)
    relevant = torch.isin(candidates, targets)
    recall_denominator = max(1, min(topk, targets.numel()))

    if relevant.any():
        ranks = torch.nonzero(relevant, as_tuple=False).flatten().to(dtype=torch.float32)
        ndcg_numerator = torch.sum(1.0 / torch.log2(ranks + 2.0))
    else:
        ndcg_numerator = torch.tensor(0.0, device=candidates.device)

    ndcg_denominator = torch.sum(
        1.0 / torch.log2(torch.arange(1, recall_denominator + 1, device=candidates.device, dtype=torch.float32) + 1.0)
    )

    return {
        'hitrate': relevant.any().to(dtype=torch.float32).item(),
        'recall': (relevant.to(dtype=torch.float32).sum() / recall_denominator).item(),
        'ndcg': (ndcg_numerator / ndcg_denominator).item() if ndcg_denominator.item() > 0 else 0.0,
    }


def evaluate(
    targets: Dict[int, torch.Tensor],
    candidates: Dict[int, torch.Tensor],
    catalog_size: int,
    topk: int = 100,
) -> Dict[str, float]:
    hitrates = []
    recalls = []
    ndcgs = []

    covered_candidates = []

    for i in candidates:
        candidate_ids = candidates[i].flatten().to(device="cpu", dtype=torch.long)
        covered_candidates.append(candidate_ids)
        metrics = get_metrics(targets[i], candidates[i], topk)
        hitrates.append(metrics['hitrate'])
        recalls.append(metrics['recall'])
        ndcgs.append(metrics['ndcg'])

    if covered_candidates:
        coverage = torch.unique(torch.cat(covered_candidates)).numel() / max(catalog_size, 1)
    else:
        coverage = 0.0

    return {
        'hitrate': sum(hitrates) / len(hitrates),
        'recall': sum(recalls) / len(recalls),
        'ndcg': sum(ndcgs) / len(ndcgs),
        'coverage': coverage
    }
