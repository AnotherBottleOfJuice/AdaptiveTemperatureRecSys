from comet_ml import Experiment
import polars as pl
import tqdm
import torch.nn.functional as F
import torch
from typing import Dict, List

from model import Graph
from metrics_utils import evaluate
from config import TOPK, VOCAB_SIZE


def eval_loop(
        graph: Graph,
        test_dataset,
        test_histories: pl.DataFrame,
        test_targets: Dict[int, List[int]],
        item_to_token: pl.DataFrame,
        writer: Experiment | None = None
) -> Dict[str, float]:
    with torch.inference_mode():
        all_candidates = []

        for batch in tqdm.tqdm(test_dataset):
            with torch.autocast(device_type='cuda', dtype=torch.float):
                hidden_states = graph.gpt(batch.token_ids)
                last_hidden_state = hidden_states[
                    torch.arange(len(batch.lengths)),
                    batch.lengths - 1
                ]

                weights = graph.head.weight

                if graph.is_cosine_similarity:
                    weights = F.normalize(weights, dim=-1)
                    last_hidden_state = F.normalize(last_hidden_state, dim=-1)

                logits = last_hidden_state @ weights.T

            logits[:, 0] = -torch.inf

            _, indices = torch.topk(logits, k=TOPK)
            all_candidates.append(indices.cpu())

    candidates = torch.cat(all_candidates, dim=0)

    candidates_df = pl.DataFrame({"uid": test_histories['uid'], "token_id": candidates})

    candidates_df = candidates_df.explode("token_id")

    candidates_df = candidates_df.join(item_to_token, on="token_id", how="left")

    candidates_df = candidates_df.group_by("uid", maintain_order=True).agg(pl.col("item_id"))

    result = evaluate(
        targets=test_targets,
        candidates=dict(candidates_df.iter_rows()),
        catalog_size=VOCAB_SIZE,
        topk=TOPK
    )

    if writer is not None:
        for name, value in result.items():
            writer.log_metric(f"valid/{name}", value=value)

    return result
