import tqdm
import torch
from dataclasses import dataclass
from comet_ml import Experiment
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

from dataset import (TrainingDataset, TestDataset, get_train_histories,
                     get_train_events, get_general_data, get_item_to_token,
                     get_test_histories, get_item_to_freq)
from model import Graph, tau
from eval import Evaluator
from config import BOS, COMET_API_KEY, TOPK, VOCAB_SIZE, UNIFORM_NEGATIVES_NUM, IN_BATCH_NEGATIVES_NUM


def _ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group("nccl", rank=rank, world_size=world_size)

def _as_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)

class Trainer:
    def __init__(self, 
                 graph : Graph, 
                 train_dataloader,
                 optimizer,
                 scheduler=None,
                 num_epochs: int = 1,
                 grad_clip: float = 0.0,
                 eval_every: int = -1,
                 evaluator: Evaluator | None = None,
                 logging: bool = True,
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
        self.ddp = False
        self.writer = None

    def _init_writer(self):
        self.writer = Experiment(
            api_key=COMET_API_KEY,
            project_name="AdaptiveTemperature",
            workspace="maksim-bessolitsyn",
        )
        if self.ddp:
            self.writer.set_name(self.graph.module.tau.experiment_name() + f" Epochs:{self.num_epochs}")
        else:
            self.writer.set_name(self.graph.tau.experiment_name() + f" Epochs:{self.num_epochs}")

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
            loss = self.graph(batch, writer=self.writer)

        loss.backward()

        if self.grad_clip > 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.grad_clip)
            if self.ddp:
                torch.distributed.all_reduce(grad_norm, op=torch.distributed.ReduceOp.SUM)
                grad_norm /= torch.distributed.get_world_size()
            if self.writer is not None:
                step = int(graph.tokens_passed.item())
                self.writer.log_metric("optim/grad_norm", value=_as_float(grad_norm), step=step)

        self.optimizer.step()

        return float(loss.detach())
    
    def _run_epoch(self):
        graph = self.graph.module if self.ddp else self.graph

        train_loss = 0.0
        curr_tokens = 0
        num_batches = 0

        for batch in self.train_dataloader:
            train_loss += self._run_batch(batch)
            curr_tokens += batch.size
            num_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()
            if self.writer is not None:
                self.writer.log_metric("optim/lr", value=_as_float(self.optimizer.param_groups[0]["lr"]),
                               step=graph.tokens_passed)

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
            range(self.num_epochs), 
            desc="Epochs", 
            disable=self.writer is None
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
                self.writer.log_metric("train/loss", value=train_loss,
                                       step=graph.tokens_passed)

            if epoch % self.eval_every == 0 and self.eval_every != -1:
                metrics = self._get_metrics(graph)
                if self.writer is not None and self.evaluator is not None:
                    metrics_str = ", ".join([f"{k}: {_as_float(v):.4f}" for k, v in metrics.items()])
                    tqdm.tqdm.write(f"\n[Epoch {epoch}] Train Loss: {train_loss:.4f} | Validation: {metrics_str}")
                    for k, v in metrics.items():
                        self.writer.log_metric(f"valid/{k}", value=_as_float(v), step=graph.tokens_passed)

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
        vocab_size: int
        max_seq_len: int
        n_layers: int = 4
        d_model: int = 256
        n_heads: int = 4
        dropout: float = 0
        log_q_correction: float = 1.0
        is_cosine_similarity: bool = True

    @dataclass
    class TauConfig:
        type: str
        initial_tau: float = 0.45
        tau_min: float = 0.045
        tau_max: float = 0.05
        num_epochs: int = 5
        num_tokens_per_epoch: int = 4_019_032

    @dataclass
    class TrainingDatasetConfig:
        batch_size: int
        device: str = "cuda"
        chunk_rows: int = 64000
        shuffle: bool = True
        seed: int | None = 42
        pin_memory: bool = True
        vocab_size: int = VOCAB_SIZE
        uniform_negative_items: int = UNIFORM_NEGATIVES_NUM
        in_batch_negative_items: int = IN_BATCH_NEGATIVES_NUM

    @dataclass
    class TestDatasetConfig:
        batch_size : int
        device : str = "cuda"

    @dataclass
    class OptimizerConfig:
        lr: float
        weight_decay: float = 0.0
    
    @dataclass
    class SchedulerConfig:
        type: str
        # TODO parse scheduler-specific parameters here, e.g. for StepLR: step_size, gamma, etc.
    
    @dataclass
    class TrainingConfig:
        num_epochs: int
        grad_clip: float = 0.0
        eval_every: int = -1
        logging: bool = True

    @dataclass
    class EvaluatorConfig:
        topk : int = TOPK    
    
    graph: GraphConfig
    tau: TauConfig
    training_dataset: TrainingDatasetConfig
    test_dataset: TestDatasetConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    training: TrainingConfig
    evaluator: EvaluatorConfig

    vocab_size : int = VOCAB_SIZE
    max_seq_len : int = 100
    bos : int = BOS

    def build_train_dataloader(self, train_histories) -> TrainingDataset:
        return TrainingDataset(train_histories, **self.training_dataset.__dict__, seq_len=self.max_seq_len)

    def build_test_dataloader(self, test_histories) -> TestDataset:
        return TestDataset(test_histories, **self.test_dataset.__dict__)
    
    def build_optimizer(self, model_parameters) -> torch.optim.Optimizer:
        return torch.optim.AdamW(model_parameters, **self.optimizer.__dict__)
    
    def build_scheduler(self, optimizer) -> torch.optim.lr_scheduler._LRScheduler | None:
        if self.scheduler.type is None:
            return None
        scheduler = getattr(torch.optim.lr_scheduler, self.scheduler.type)(
            optimizer,
            **(self.scheduler.__dict__ - {"type"})
        )
        return scheduler
    
    def build_evaluator(self, test_dataloader, test_histories, test_targets, item_to_token) -> Evaluator:
        return Evaluator(test_dataloader, test_histories, test_targets, item_to_token, vocab_size=self.vocab_size,
                        topk=self.evaluator.topk)
    
    def build_tau(self):
        return getattr(tau, self.tau.type)(
            initial_tau=self.tau.initial_tau,
            tau_min=self.tau.tau_min,
            tau_max=self.tau.tau_max,
            num_epochs=self.tau.num_epochs,
            num_tokens_per_epoch=self.tau.num_tokens_per_epoch
        )

    def build_graph(self) -> Graph:
        return Graph(**self.graph.__dict__, tau=self.build_tau())

def prepare_data(config : ExperimentConfig):
    train, test, embeddings, artists, test_targets = get_general_data()
    item_to_token = get_item_to_token(train, vocab_size=config.vocab_size)
    item_to_freq = get_item_to_freq(train)

    train_events = get_train_events(train, item_to_token, item_to_freq, config.max_seq_len)
    train_histories = get_train_histories(train_events, config.bos)

    test_histories = get_test_histories(test, train_events, config.bos)

    train_dataloader = config.build_train_dataloader(train_histories)
    
    test_dataset = config.build_test_dataloader(test_histories)

    return train_dataloader, test_dataset, test_histories, test_targets, item_to_token


def run_training_on_device(rank : int, world_size : int, config : ExperimentConfig):
    _ddp_setup(rank, world_size)
    torch.cuda.set_device(rank)

    config.training_dataset.device = rank
    config.test_dataset.device = rank

    train_dataloader, test_dataloader, test_histories, test_targets, item_to_token = prepare_data(config)

    graph : Graph = config.build_graph().to(rank)

    optimizer = config.build_optimizer(graph.parameters())
    scheduler = config.build_scheduler(optimizer)

    evaluator = config.build_evaluator(test_dataloader, test_histories, test_targets, item_to_token)

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
    )

    trainer.ddp_setup(rank, world_size)
    metrics = trainer.run()

    destroy_process_group()

    if rank == 0:
        print("Name:",graph.tau.experiment_name(), "Final metrics:", metrics)

def run_ddp_training(config : ExperimentConfig, world_size=None):
    if world_size is None:
        world_size = torch.cuda.device_count()
        assert world_size > 0, "CUDA is unavailable. DDP training requires at least one GPU."

    mp.spawn(run_training_on_device, args=(world_size, config),
              nprocs=world_size, join=True)
