import tqdm
import torch
import random
import numpy as np
from dataclasses import dataclass, field, asdict
from comet_ml import Experiment
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

from yambdadataset import (
    TrainingDataset,
    TestDataset,
    get_train_histories,
    get_train_events,
    get_general_data,
    get_item_to_token,
    get_test_histories,
    get_item_to_freq,
)
from .model import Graph, tau
from .eval import Evaluator


def _ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group("nccl", rank=rank, world_size=world_size)

def set_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _as_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


class Trainer:
    def __init__(
        self,
        graph: Graph,
        train_dataloader,
        optimizer,
        scheduler,
        num_epochs: int,
        grad_clip: float,
        eval_every: int,
        evaluator: Evaluator | None,
        logging: bool,
        comet_api_key: str,
        config: "ExperimentConfig | None" = None,
    ):
        self.graph = graph
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.grad_clip = grad_clip
        self.eval_every = eval_every
        self.evaluator = evaluator
        self.logging = logging
        self.comet_api_key = comet_api_key
        self.config = config
        self.ddp = False
        self.writer = None

    def _init_writer(self):
        self.writer = Experiment(
            api_key=self.comet_api_key,
            project_name="AdaptiveTemperature",
            workspace="maksim-bessolitsyn",
        )
        if self.ddp:
            self.writer.set_name(
                self.graph.module.tau.experiment_name() + f" Epochs:{self.num_epochs}"
            )
        else:
            self.writer.set_name(
                self.graph.tau.experiment_name() + f" Epochs:{self.num_epochs}"
            )
        
        self._log_training_params()

    def _log_training_params(self):
        """Log training parameters to Comet."""
        if self.writer is None or self.config is None:
            return
        
        params_dict = asdict(self.config)
        
        flat_params = {}
        for section_name, section_config in params_dict.items():
            if isinstance(section_config, dict):
                for param_name, param_value in section_config.items():
                    flat_params[f"{section_name}_{param_name}"] = param_value
            else:
                flat_params[section_name] = section_config
        
        if self.optimizer is not None:
            flat_params["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        
        self.writer.log_parameters(flat_params)

    def ddp_setup(self, rank, world_size):
        self.graph = DDP(self.graph, device_ids=[rank], output_device=rank)
        self.train_dataloader.ddp_setup(rank, world_size)
        self.ddp = True
        if self.logging and rank == 0:
            self._init_writer()

    def _run_batch(self, batch):
        graph = self.graph.module if self.ddp else self.graph

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast("cuda", torch.bfloat16):
            loss, metrics = self.graph(batch)

        loss.backward()

        if self.grad_clip > 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.graph.parameters(), self.grad_clip
            )
            if self.ddp:
                torch.distributed.all_reduce(
                    grad_norm, op=torch.distributed.ReduceOp.SUM
                )
                grad_norm /= torch.distributed.get_world_size()
            metrics["optim/grad_norm"] = _as_float(grad_norm)

        self.optimizer.step()

        return float(loss.detach()), metrics

    def _run_epoch(self):
        graph = self.graph.module if self.ddp else self.graph

        train_loss = 0.0
        curr_tokens = 0
        num_batches = 0
        epoch_metrics = {}

        for batch in self.train_dataloader:
            loss, metrics = self._run_batch(batch)
            train_loss += loss
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
            curr_tokens += batch.size
            num_batches += 1

        for k in epoch_metrics:
            epoch_metrics[k] /= num_batches

        if self.ddp:
            for k, v in epoch_metrics.items():
                v_t = torch.tensor(v, device=self.graph.device)
                torch.distributed.all_reduce(v_t, op=torch.distributed.ReduceOp.SUM)
                epoch_metrics[k] = (v_t / torch.distributed.get_world_size()).item()

        if self.scheduler is not None:
            self.scheduler.step()
            if self.writer is not None:
                self.writer.log_metric(
                    "optim/lr",
                    value=_as_float(self.optimizer.param_groups[0]["lr"]),
                    step=graph.tokens_passed,
                )

        if self.writer is not None:
            for k, v in epoch_metrics.items():
                self.writer.log_metric(k, value=v, step=graph.tokens_passed)

        return train_loss / num_batches

    def _get_metrics(self, graph) -> dict[str, float]:
        self.graph.eval()
        with torch.inference_mode():
            metrics = self.evaluator(graph=graph)
        self.graph.train()
        if self.ddp:
            for k, v in metrics.items():
                v_t = torch.tensor(v, device=self.graph.device)
                torch.distributed.all_reduce(v_t, op=torch.distributed.ReduceOp.SUM)
                if self.writer is not None:
                    metrics[k] = (v_t / torch.distributed.get_world_size()).item()
        return metrics

    def run(self) -> dict[str, float]:
        graph = self.graph.module if self.ddp else self.graph

        if not self.ddp and self.logging:
            self._init_writer()

        self.graph.train()

        epoch_iter = tqdm.tqdm(
            range(self.num_epochs), desc="Epochs", disable=self.writer is None
        )

        for epoch in epoch_iter:
            train_loss = self._run_epoch()

            if self.writer is not None:
                epoch_iter.set_postfix({"train_loss": f"{train_loss:.4f}"})

            if self.ddp:
                loss_t = torch.tensor(train_loss, device=self.graph.device)
                torch.distributed.all_reduce(loss_t, op=torch.distributed.ReduceOp.SUM)
                loss_t /= torch.distributed.get_world_size()
                train_loss = loss_t.item()

            if self.writer is not None:
                self.writer.log_metric(
                    "train/loss", value=train_loss, step=graph.tokens_passed
                )

            if epoch % self.eval_every == 0 and self.eval_every != -1:
                metrics = self._get_metrics(graph)
                if self.writer is not None and self.evaluator is not None:
                    for k, v in metrics.items():
                        self.writer.log_metric(
                            f"valid/{k}", value=_as_float(v), step=graph.tokens_passed
                        )

        self.graph.eval()
        metrics = {}
        if self.evaluator is not None:
            with torch.inference_mode():
                metrics = self._get_metrics(graph)

        return metrics



@dataclass
class ExperimentConfig:
    @dataclass
    class GraphConfig:
        n_layers: int
        d_model: int
        n_heads: int
        dropout: float
        log_q_correction: float
        is_cosine_similarity: bool

    @dataclass
    class DataConfig:
        vocab_size: int
        max_seq_len: int
        bos: int
        path_interactions: str
        path_embeddings: str
        path_artists: str
        core_min_interaction_per_user: int
        test_interval_seconds: int
        max_train_events_per_user: int

    @dataclass
    class TauConfig:
        class_name: str
        json_args: dict[str, object] = field(default_factory=dict)

    @dataclass
    class TrainingDatasetConfig:
        batch_size: int
        device: str
        chunk_rows: int
        shuffle: bool
        seed: int | None
        pin_memory: bool
        uniform_negative_items: int
        in_batch_negative_items: int

    @dataclass
    class TestDatasetConfig:
        batch_size: int
        device: str

    @dataclass
    class OptimizerConfig:
        class_name: str
        json_args: dict[str, object] = field(default_factory=dict)

    @dataclass
    class SchedulerConfig:
        class_name: str | None = None
        json_args: dict[str, object] = field(default_factory=dict)

    @dataclass
    class TrainingConfig:
        num_epochs: int
        grad_clip: float
        eval_every: int
        logging: bool
        comet_api_key: str
        seed: int | None = None

    @dataclass
    class EvaluatorConfig:
        topk: int

    graph: GraphConfig
    data: DataConfig
    tau: TauConfig
    training_dataset: TrainingDatasetConfig
    test_dataset: TestDatasetConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    training: TrainingConfig
    evaluator: EvaluatorConfig

    def build_train_dataloader(self, train_histories) -> TrainingDataset:
        return TrainingDataset(
            train_histories,
            **self.training_dataset.__dict__,
            seq_len=self.data.max_seq_len,
            vocab_size=self.data.vocab_size,
        )

    def build_test_dataloader(self, test_histories) -> TestDataset:
        return TestDataset(test_histories, **self.test_dataset.__dict__)

    def build_optimizer(self, model_parameters) -> torch.optim.Optimizer:
        return getattr(torch.optim, self.optimizer.class_name)(
            model_parameters, **self.optimizer.json_args
        )

    def build_scheduler(
        self, optimizer
    ) -> torch.optim.lr_scheduler._LRScheduler | None:
        if self.scheduler.class_name is None:
            return None
        return getattr(torch.optim.lr_scheduler, self.scheduler.class_name)(
            optimizer,
            **self.scheduler.json_args,
        )

    def build_evaluator(
        self, test_dataloader, test_histories, test_targets, item_to_token
    ) -> Evaluator:
        return Evaluator(
            test_dataloader,
            test_histories,
            test_targets,
            item_to_token,
            vocab_size=self.data.vocab_size,
            device=self.test_dataset.device,
            topk=self.evaluator.topk,
        )

    def build_tau(self):
        return getattr(tau, self.tau.class_name)(**self.tau.json_args)

    def build_graph(self) -> Graph:
        return Graph(
            **self.graph.__dict__,
            vocab_size=self.data.vocab_size,
            max_seq_len=self.data.max_seq_len,
            tau=self.build_tau(),
        )


def prepare_data(config: ExperimentConfig):
    train, test, embeddings, artists, test_targets = get_general_data(
        path_interactions=config.data.path_interactions,
        path_embeddings=config.data.path_embeddings,
        path_artists=config.data.path_artists,
        core_min_interaction_per_user=config.data.core_min_interaction_per_user,
        test_interval_seconds=config.data.test_interval_seconds,
    )
    item_to_token = get_item_to_token(train, vocab_size=config.data.vocab_size)
    item_to_freq = get_item_to_freq(train)

    train_events = get_train_events(
        train, item_to_token, item_to_freq, config.data.max_train_events_per_user
    )
    train_histories = get_train_histories(train_events, config.data.bos)

    test_histories = get_test_histories(
        test, train_events, config.data.bos, config.data.max_seq_len
    )

    train_dataloader = config.build_train_dataloader(train_histories)

    test_dataset = config.build_test_dataloader(test_histories)

    return train_dataloader, test_dataset, test_histories, test_targets, item_to_token


def run_training_on_device(rank: int, world_size: int, config: ExperimentConfig):
    _ddp_setup(rank, world_size)
    torch.cuda.set_device(rank)

    set_seed(config.training.seed)

    config.training_dataset.device = rank
    config.test_dataset.device = rank

    train_dataloader, test_dataloader, test_histories, test_targets, item_to_token = (
        prepare_data(config)
    )

    graph: Graph = config.build_graph().to(rank)

    optimizer = config.build_optimizer(graph.parameters())
    scheduler = config.build_scheduler(optimizer)

    evaluator = config.build_evaluator(
        test_dataloader,
        test_histories,
        test_targets,
        item_to_token,
    )

    trainer = Trainer(
        graph=graph,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config.training.num_epochs,
        grad_clip=config.training.grad_clip,
        eval_every=config.training.eval_every,
        evaluator=evaluator,
        logging=config.training.logging,
        comet_api_key=config.training.comet_api_key,
        config=config,
    )

    trainer.ddp_setup(rank, world_size)
    metrics = trainer.run()

    destroy_process_group()

    if rank == 0:
        print("--------------------------------")
        print("Experiment name:", graph.tau.experiment_name())
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print("--------------------------------")
        


def run_ddp_training(config: ExperimentConfig, world_size: int):
    assert world_size > 0, (
        "CUDA is unavailable. DDP training requires at least one GPU."
    )

    mp.spawn(
        run_training_on_device, args=(world_size, config), nprocs=world_size, join=True
    )
