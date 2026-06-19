import argparse
import copy
import itertools
import statistics
import sys
from datetime import datetime
from pathlib import Path
import yaml
import torch
from sasrec import run_ddp_training, ExperimentConfig


class _Tee:
    """Writes to both the original stdout and a log file simultaneously."""
    def __init__(self, log_file):
        self._file = log_file
        self._stdout = sys.__stdout__

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()


def _set_nested(d: dict, dotted_key: str, value) -> None:
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def _build_experiment_config(raw: dict, config_name: str) -> ExperimentConfig:
    return ExperimentConfig(
        graph=ExperimentConfig.GraphConfig(**raw["graph"]),
        data=ExperimentConfig.DataConfig(**raw["data"]),
        tau=ExperimentConfig.TauConfig(**raw["tau"]),
        training_dataset=ExperimentConfig.TrainingDatasetConfig(**raw["training_dataset"]),
        test_dataset=ExperimentConfig.TestDatasetConfig(**raw["test_dataset"]),
        optimizer=ExperimentConfig.OptimizerConfig(**raw["optimizer"]),
        scheduler=ExperimentConfig.SchedulerConfig(**raw["scheduler"]),
        training=ExperimentConfig.TrainingConfig(**raw["training"]),
        evaluator=ExperimentConfig.EvaluatorConfig(**raw["evaluator"]),
        config_name=config_name,
    )


def _build_grouped_configs(raw: dict, config_name: str) -> tuple[list[list[ExperimentConfig]], list[str], list[tuple]]:
    """Return (groups, swept_keys, combo_vals_per_group).

    Each group is a list of rerun_count ExperimentConfig objects sharing the same hyperparams.
    """
    sweep = raw.pop("sweep", {})
    default_seed = raw.get("training", {}).get("seed")
    seeds = sweep.get("seeds", [default_seed])
    grid = sweep.get("grid", {})

    keys = list(grid.keys())
    value_lists = [v if isinstance(v, list) else [v] for v in grid.values()]

    groups, combo_vals_per_group = [], []
    for combo in (itertools.product(*value_lists) if keys else [()]):
        cfg = copy.deepcopy(raw)
        for key, val in zip(keys, combo):
            _set_nested(cfg, key, val)
        seeded_configs = []
        for seed in seeds:
            seeded = copy.deepcopy(cfg)
            seeded["training"]["seed"] = seed
            seeded_configs.append(_build_experiment_config(seeded, config_name))
        groups.append(seeded_configs)
        combo_vals_per_group.append(combo)

    return groups, keys, combo_vals_per_group, seeds


def _run(groups, swept_keys, combo_vals_per_group, seeds, world_size):
    n_combos = len(groups)
    print(f"Sweep: {n_combos} combo(s) × {len(seeds)} seed(s) {seeds} on {world_size} GPU(s)\n")

    SEP = "--------------------------------"

    summary = []
    for configs, combo_vals in zip(groups, combo_vals_per_group):
        run_metrics = []
        for seed, config in zip(seeds, configs):
            metrics = run_ddp_training(config, world_size=world_size)
            run_metrics.append(metrics)
            print(SEP)
            name = config.build_tau().experiment_name()
            suffix = f"  (seed={seed})" if len(seeds) > 1 else ""
            print(f"Experiment name: {name}{suffix}")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")
            print(SEP)

        summary.append((configs[0], combo_vals, run_metrics))

    if n_combos > 1 or len(seeds) > 1:
        print(f"\n{SEP}")
        print("Sweep summary (mean ± std across reruns)")
        print(SEP)
        for config, combo_vals, run_metrics in summary:
            print(f"Experiment name: {config.build_tau().experiment_name()}")
            for key, val in zip(swept_keys, combo_vals):
                print(f"{key}: {val}")
            for mk in run_metrics[0].keys():
                vals = [r[mk] for r in run_metrics]
                mean = statistics.mean(vals)
                std = statistics.stdev(vals) if len(vals) > 1 else 0.0
                print(f"{mk}: {mean:.4f} ± {std:.4f}")
            print(SEP)


def main():
    parser = argparse.ArgumentParser(description="Run AdaptiveTemperatureRecSys experiment")
    parser.add_argument("config", type=str, help="Path to YAML experiment config file")
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all available)",
    )
    args = parser.parse_args()
    config_name = Path(args.config).stem

    with open(args.config) as f:
        raw = yaml.safe_load(f)

    groups, swept_keys, combo_vals_per_group, seeds = _build_grouped_configs(raw, config_name)

    world_size = args.gpus if args.gpus is not None else torch.cuda.device_count()
    if world_size == 0:
        print("ERROR: No CUDA devices found. This project requires at least one GPU.", file=sys.stderr)
        sys.exit(1)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = results_dir / f"{config_name}_{timestamp}.txt"

    with open(log_path, "w") as log_file:
        sys.stdout = _Tee(log_file)
        try:
            _run(groups, swept_keys, combo_vals_per_group, seeds, world_size)
        finally:
            sys.stdout = sys.__stdout__

    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
