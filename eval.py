import polars as pl
import tqdm
import torch.nn.functional as F
import torch
from typing import Dict, List

from dataset import TestDataset
from model import Graph
from metrics_utils import evaluate
from config import TOPK, VOCAB_SIZE


def eval_loop(
        graph: Graph,
        test_histories: pl.DataFrame,
        test_targets: Dict[int, List[int]],
        item_to_token: pl.DataFrame) -> Dict[str, float]:
    ds = TestDataset(test_histories, batch_size=128)

    with torch.inference_mode():
        all_candidates = []

        for batch in tqdm.tqdm(ds):
            with torch.autocast(device_type='cuda', dtype=torch.float):
                hidden_states = graph.gpt(batch.token_ids)
                last_hidden_state = hidden_states[
                    torch.arange(len(batch.lengths)),
                    batch.lengths - 1
                ]

                weights = F.normalize(graph.head.weight, dim=-1)
                last_hidden_state_norm = F.normalize(last_hidden_state, dim=-1)

                logits = last_hidden_state_norm @ weights.T

            logits[:, 0] = -torch.inf

            _, indices = torch.topk(logits, k=TOPK)
            all_candidates.append(indices.cpu())

    candidates = torch.cat(all_candidates, dim=0)

    candidates_df = pl.DataFrame({"uid": test_histories['uid'], "token_id": candidates})

    candidates_df = candidates_df.explode("token_id")

    candidates_df = candidates_df.join(item_to_token, on="token_id", how="left")

    candidates_df = candidates_df.group_by("uid", maintain_order=True).agg(pl.col("item_id"))

    return evaluate(
        targets=test_targets,
        candidates=dict(candidates_df.iter_rows()),
        catalog_size=VOCAB_SIZE,
        topk=TOPK
    )
