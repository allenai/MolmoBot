"""
Run SynthVLA evaluation using molmo_spaces.

This script is the entry point invoked by gantry for evaluation jobs.
It calls run_evaluation from molmo_spaces with the provided arguments.

Usage:
    python launch_scripts/run_eval.py \
        --checkpoint_path /path/to/checkpoint \
        --benchmark_path /path/to/benchmark \
        --eval_config_cls olmo.eval.configure_molmo_spaces:SynthVLAFrankaBenchmarkEvalConfig
"""

import argparse
import importlib
from pathlib import Path

from molmo_spaces.evaluation.eval_main import run_evaluation


def main():
    parser = argparse.ArgumentParser(
        description="Run SynthVLA evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint to evaluate",
    )
    parser.add_argument(
        "--benchmark_path",
        type=str,
        required=True,
        help="Path to the benchmark directory",
    )
    parser.add_argument(
        "--eval_config_cls",
        type=str,
        required=True,
        help="Evaluation config class (module:ClassName)",
    )
    parser.add_argument(
        "--task_horizon",
        type=int,
        default=600,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for eval results",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel eval workers",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable wandb logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="mjthor-online-eval",
        help="WandB project name",
    )
    parser.add_argument(
        "--use_filament",
        action="store_true",
        help="Use filament renderer instead of legacy OpenGL",
    )
    parser.add_argument(
        "--environment_light_intensity",
        type=float,
        default=None,
        help="Intensity of default environmental light (filament only)",
    )
    args = parser.parse_args()

    # Resolve module:ClassName string to actual class so the sim package uses __name__
    # (not the full "module:ClassName" string) when constructing the output directory.
    eval_config_cls = args.eval_config_cls
    if isinstance(eval_config_cls, str) and ":" in eval_config_cls:
        module_path, class_name = eval_config_cls.split(":")
        eval_config_cls = getattr(importlib.import_module(module_path), class_name)

    results = run_evaluation(
        eval_config_cls=eval_config_cls,
        benchmark_dir=Path(args.benchmark_path),
        checkpoint_path=Path(args.checkpoint_path),
        task_horizon_steps=args.task_horizon,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        use_filament=args.use_filament,
        environment_light_intensity=args.environment_light_intensity,
    )

    print(f"Success rate: {results.success_rate:.1%}")
    for r in results.episode_results:
        print(f"{r.house_id}/ep{r.episode_idx}: {'pass' if r.success else 'fail'}")


if __name__ == "__main__":
    main()

