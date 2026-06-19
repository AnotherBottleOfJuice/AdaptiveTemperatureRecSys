import tqdm
import torch
import random
import numpy as np
import mlflow
from dataclasses import dataclass, field, asdict
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

from recdata import (
    TrainingDataset,
    TestDataset,
    SequentialDataset,
    build_dataset,
    get_train_histories,
    get_train_events,
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

def _as_step(value) -> int:
    if torch.is_tensor(value):
        return int(value.detach().cpu().item())
    return int(value)


class _MlflowWriter:
    """Thin wrapper over mlflow's active-run API matching the call sites in Trainer."""

    def set_name(self, name: str) -> None:
        mlflow.set_tag("mlflow.runName", name)

    def log_parameters(self, params: dict) -> None:
        safe = {k: str(v) for k, v in params.items() if v is not None}
        mlflow.log_params(safe)

    def log_metric(self, key: str, value: float, step=None) -> None:
        mlflow.log_metric(key, value, step=_as_step(step) if step is not None else None)

    def end(self) -> None:
        mlflow.end_run()


class Trainer:
    def __init__(
        self,
        graph: Graph,
        train_dataset,
        optimizer,
        scheduler,
        num_epochs: int,
        grad_clip: float,
        eval_every: int,
        evaluator: Evaluator | None,
        logging: bool,
        log_dir: str,
        config: "ExperimentConfig | None" = None,
    ):
        self.graph = graph
        self.train_dataset = train_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.grad_clip = grad_clip
        self.eval_every = eval_every
        self.evaluator = evaluator
        self.logging = logging
        self.log_dir = log_dir
        self.config = config
        self.ddp = False
        self.writer = None

    def _init_writer(self):
        db_path = os.path.join(os.path.abspath(self.log_dir), "mlflow.db")
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        experiment_name = (
            self.config.config_name
            if self.config is not None and self.config.config_name
            else "AdaptiveTemperature"
        )
        mlflow.set_experiment(experiment_name)
        mlflow.start_run()
        self.writer = _MlflowWriter()

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
        self.train_dataset.ddp_setup(rank, world_size)
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

        for batch in self.train_dataset:
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
            self.train_dataset.set_epoch(epoch)
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

        if self.writer is not None:
            self.writer.end()

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
    class DatasetConfig:
        class_name: str
        json_args: dict[str, object] = field(default_factory=dict)

    @dataclass
    class DataConfig:
        vocab_size: int
        max_seq_len: int
        bos: int
        max_train_events_per_user: int
        dataset: "ExperimentConfig.DatasetConfig"

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
        log_dir: str
        console_logging: bool = True
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
    config_name: str | None = None

    def build_dataset(self) -> SequentialDataset:
        return build_dataset(
            self.data.dataset.class_name, **self.data.dataset.json_args
        )

    def build_train_dataset(self, train_histories) -> TrainingDataset:
        return TrainingDataset(
            train_histories,
            **self.training_dataset.__dict__,
            seq_len=self.data.max_seq_len,
            vocab_size=self.data.vocab_size,
        )

    def build_test_dataset(self, test_histories) -> TestDataset:
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
        self, test_dataset, test_histories, test_targets, item_to_token
    ) -> Evaluator:
        return Evaluator(
            test_dataset,
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
    train, test, test_targets = config.build_dataset().load()

    item_to_token = get_item_to_token(train, vocab_size=config.data.vocab_size)
    item_to_freq = get_item_to_freq(train)

    train_events = get_train_events(
        train, item_to_token, item_to_freq, config.data.max_train_events_per_user
    )
    train_histories = get_train_histories(train_events, config.data.bos)

    test_histories = get_test_histories(
        test, train_events, config.data.bos, config.data.max_seq_len
    )

    train_dataset = config.build_train_dataset(train_histories)

    test_dataset = config.build_test_dataset(test_histories)

    return train_dataset, test_dataset, test_histories, test_targets, item_to_token


def run_training_on_device(
    rank: int, world_size: int, config: ExperimentConfig, return_queue: mp.Queue = None
):
    _ddp_setup(rank, world_size)
    torch.cuda.set_device(rank)

    set_seed(config.training.seed)

    config.training_dataset.device = rank
    config.test_dataset.device = rank

    train_dataset, test_dataset, test_histories, test_targets, item_to_token = (
        prepare_data(config)
    )

    graph: Graph = config.build_graph().to(rank)

    optimizer = config.build_optimizer(graph.parameters())
    scheduler = config.build_scheduler(optimizer)

    evaluator = config.build_evaluator(
        test_dataset,
        test_histories,
        test_targets,
        item_to_token,
    )

    trainer = Trainer(
        graph=graph,
        train_dataset=train_dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config.training.num_epochs,
        grad_clip=config.training.grad_clip,
        eval_every=config.training.eval_every,
        evaluator=evaluator,
        logging=config.training.logging,
        log_dir=config.training.log_dir,
        config=config,
    )

    trainer.ddp_setup(rank, world_size)
    metrics = trainer.run()

    destroy_process_group()

    if rank == 0:
        if config.training.console_logging:
            print("--------------------------------")
            print("Experiment name:", graph.tau.experiment_name())
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")
            print("--------------------------------")

        if return_queue is not None:
            return_queue.put(metrics)



def run_ddp_training(config: ExperimentConfig, world_size: int):
    assert world_size > 0, (
        "CUDA is unavailable. DDP training requires at least one GPU."
    )

    ctx = mp.get_context("spawn")
    return_queue = ctx.Queue()

    mp.spawn(
        run_training_on_device,
        args=(world_size, config, return_queue),
        nprocs=world_size,
        join=True,
    )

    return return_queue.get()
