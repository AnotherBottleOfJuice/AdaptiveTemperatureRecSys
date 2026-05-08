import polars as pl
import torch.nn.functional as F
import torch
from typing import Dict, List

from model import Graph
from metrics_utils import evaluate
from config import TOPK, VOCAB_SIZE


class Evaluator:
    def __init__(self,
                test_dataloader,
                test_histories,
                test_targets, 
                item_to_token, 
                vocab_size=VOCAB_SIZE,
                device="cuda",
                topk=TOPK
            ):
        self.test_dataloader = test_dataloader
        self.test_histories = test_histories
        self.test_targets = test_targets
        self.item_to_token = item_to_token
        self.vocab_size = vocab_size
        self.device = device
        self.topk = topk

    @torch.inference_mode()
    def _generate_candidates(self, graph: Graph) -> List[torch.Tensor]:
        all_candidates = []

        for batch in self.test_dataloader:
            with torch.autocast(device_type="cuda", dtype=torch.float):
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
        
        return all_candidates
    
    def _prepare_candidates(self, candidates: List[torch.Tensor]) -> pl.DataFrame:
        return (
            pl.DataFrame({"uid": self.test_histories['uid'], "token_id": candidates})
            .explode("token_id")
            .join(self.item_to_token, on="token_id", how="left")
            .group_by("uid", maintain_order=True)
            .agg(pl.col("item_id"))
        )

    def __call__(self, graph: Graph) -> Dict[str, float]:
        graph.eval()
        candidates = torch.concat(self._generate_candidates(graph), dim=0)
        candidates_df = self._prepare_candidates(candidates)

        targets = {uid: torch.as_tensor(targets, dtype=torch.long) for uid, targets in self.test_targets.items()}
        candidates_map = {
            uid: torch.as_tensor(item_ids, dtype=torch.long)
            for uid, item_ids in candidates_df.iter_rows()
        }

        return evaluate(
            targets=targets,
            candidates=candidates_map,
            catalog_size=self.vocab_size,
            topk=self.topk
        )