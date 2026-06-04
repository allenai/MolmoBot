"""Serve MolmoBot RBY1 door opening policy over WebSocket.

Usage:
    cd mm_olmo

    # From HuggingFace (recommended):
    PYTHONPATH=. python launch_scripts/serve_molmobot_rby1_door.py \
        --hf_repo anonymous/MolmoBot-RBY1DoorOpening

    # From local path:
    PYTHONPATH=. python launch_scripts/serve_molmobot_rby1_door.py \
        --checkpoint_path /path/to/checkpoint

    # From S3:
    PYTHONPATH=. python launch_scripts/serve_molmobot_rby1_door.py \
        --s3_path s3://bucket/path/to/checkpoint/ --port 8001
"""

import argparse
import logging
from pathlib import Path

from olmo.eval.real_robot_molmobot_rby1_door import (
    MolmoBotRBY1DoorPolicy,
    MolmoBotRBY1DoorPolicyConfig,
)
from olmo.eval.websocket_server import WebsocketPolicyServer

logging.basicConfig(level=logging.INFO)


def download_model_from_hf(hf_repo: str, local_path: str) -> str:
    from huggingface_hub import snapshot_download
    snapshot_download(hf_repo, local_dir=local_path)
    return local_path


def download_model_from_s3(s3_path: str, local_path: str) -> str:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    if not s3_path.startswith("s3://"):
        raise ValueError(f"S3 path must start with 's3://': {s3_path}")

    s3_path_parts = s3_path[5:].split("/", 1)
    bucket_name = s3_path_parts[0]
    s3_key_prefix = s3_path_parts[1] if len(s3_path_parts) > 1 else ""

    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    local_dir = Path(local_path)
    local_dir.mkdir(parents=True, exist_ok=True)

    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_key_prefix)

    downloaded_files = []
    for page in pages:
        if "Contents" in page:
            for obj in page["Contents"]:
                s3_key = obj["Key"]
                relative_path = s3_key[len(s3_key_prefix):].lstrip("/")
                if relative_path:
                    local_file_path = local_dir / relative_path
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)
                    logging.info(f"Downloading {s3_key} to {local_file_path}")
                    s3_client.download_file(bucket_name, s3_key, str(local_file_path))
                    downloaded_files.append(str(local_file_path))

    if not downloaded_files:
        raise ValueError(f"No files found at S3 path: {s3_path}")

    logging.info(f"Downloaded {len(downloaded_files)} files to {local_dir}")
    return str(local_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Serve MolmoBot RBY1 door opening policy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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

    config = MolmoBotRBY1DoorPolicyConfig(checkpoint_path=checkpoint_path)
    policy = MolmoBotRBY1DoorPolicy(config)

    print(f"Serving {checkpoint_path} policy on ws://0.0.0.0:{args.port}")
    server = WebsocketPolicyServer(
        [policy], "molmobot-rby1-door", port=args.port,
        metadata={"checkpoint_path": checkpoint_path, "hf_repo": args.hf_repo, "s3_path": args.s3_path},
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
