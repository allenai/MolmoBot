"""Serve MolmoBot RBY1 multitask policy over WebSocket.

Supports door+open and pick+pnp tasks via --task_type.

Usage:
    cd mm_olmo

    # Door+open (from HuggingFace):
    PYTHONPATH=. python launch_scripts/serve_molmobot_rby1_multitask.py \
        --task_type door_plus_open --hf_repo anonymous/MolmoBot-RBY1Multitask

    # Pick+pnp:
    PYTHONPATH=. python launch_scripts/serve_molmobot_rby1_multitask.py \
        --task_type pick_pnp --hf_repo anonymous/MolmoBot-RBY1Multitask

    # From local path:
    PYTHONPATH=. python launch_scripts/serve_molmobot_rby1_multitask.py \
        --task_type door_plus_open --checkpoint_path /path/to/checkpoint
"""

import argparse
import logging

from olmo.eval.real_robot_molmobot_rby1_multitask import (
    MolmoBotRBY1DoorPlusOpenPolicyConfig,
    MolmoBotRBY1MultitaskPolicy,
    MolmoBotRBY1PickPnPPolicyConfig,
)
from launch_scripts.serve_molmobot_rby1_door import download_model_from_hf, download_model_from_s3
from olmo.eval.websocket_server import WebsocketPolicyServer

logging.basicConfig(level=logging.INFO)

TASK_CONFIGS = {
    "door_plus_open": MolmoBotRBY1DoorPlusOpenPolicyConfig,
    "pick_pnp": MolmoBotRBY1PickPnPPolicyConfig,
}


def main():
    parser = argparse.ArgumentParser(
        description="Serve MolmoBot RBY1 multitask policy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task_type",
        choices=list(TASK_CONFIGS.keys()),
        required=True,
        help="Which multitask policy to serve",
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--hf_repo", type=str, help="HuggingFace repo containing the model")
    source_group.add_argument("--s3_path", type=str, help="S3 path to checkpoint")
    source_group.add_argument("--checkpoint_path", type=str, help="Local checkpoint path")
    parser.add_argument("--port", type=int, default=8000, help="WebSocket server port")
    args = parser.parse_args()

    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    elif args.hf_repo:
        local_folder = args.hf_repo.split("/")[-1]
        local_path = f"ckpts/molmobot/{local_folder}"
        logging.info(f"Downloading model from HF {args.hf_repo} to {local_path}")
        checkpoint_path = download_model_from_hf(args.hf_repo, local_path)
        logging.info(f"Model downloaded to: {checkpoint_path}")
    elif args.s3_path:
        local_folder = args.s3_path.rstrip("/").split("/")[-1]
        local_path = f"ckpts/molmobot/{local_folder}"
        logging.info(f"Downloading model from S3 {args.s3_path} to {local_path}")
        checkpoint_path = download_model_from_s3(args.s3_path, local_path)
        logging.info(f"Model downloaded to: {checkpoint_path}")
    else:
        parser.error("One of --hf_repo, --s3_path, or --checkpoint_path is required")

    policy_config_cls = TASK_CONFIGS[args.task_type]
    config = policy_config_cls(checkpoint_path=checkpoint_path)
    policy = MolmoBotRBY1MultitaskPolicy(config)

    print(f"Serving {args.task_type} policy on ws://0.0.0.0:{args.port}")
    server = WebsocketPolicyServer(
        [policy], f"molmobot-rby1-{args.task_type.replace('_', '-')}", port=args.port,
        metadata={"task_type": args.task_type, "checkpoint_path": checkpoint_path,
                  "hf_repo": args.hf_repo, "s3_path": args.s3_path},
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
