import math
from typing import Dict, List


def get_metrics(targets: List[int], candidates: List[int], topk: int) -> Dict[str, float]:
    return {
        'hitrate': min(1, sum(1 if i in targets else 0 for i in candidates)),
        'recall': sum(1 if i in targets else 0 for i in candidates)
                  / min(topk, len(targets)),
        'ndcg': sum(1.0 / math.log2(i + 2) if j in targets else 0
                    for i, j in enumerate(candidates)) /
                sum(1 / math.log2(j + 2) for j in range(min(topk, len(targets))))
    }


def evaluate(
        targets: Dict[int, List[int]],
        candidates: Dict[int, List[int]],
        catalog_size: int,
        topk: int = 100,
) -> Dict[str, float]:
    hitrates = []
    recalls = []
    ndcgs = []

    s = set()

    for i in candidates:
        s = s.union(set(candidates[i]))
        metrics = get_metrics(targets[i], candidates[i], topk)
        hitrates.append(metrics['hitrate'])
        recalls.append(metrics['recall'])
        ndcgs.append(metrics['ndcg'])

    return {
        'hitrate': sum(hitrates) / len(hitrates),
        'recall': sum(recalls) / len(recalls),
        'ndcg': sum(ndcgs) / len(ndcgs),
        'coverage': len(s) / catalog_size
    }
