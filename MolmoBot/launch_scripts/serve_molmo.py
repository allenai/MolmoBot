import os
import boto3
import botocore.client
import argparse
from pathlib import Path
import logging

os.environ["JAX_ENABLE_X64"] = "0"  # 64-bit jax messes with openpi inference

import torch
from huggingface_hub import snapshot_download

from molmo_spaces.policy.base_policy import InferencePolicy
from olmo.device_util import default_device_type, device_count
from olmo.eval.configure_real_robot import RealRobotVLAPolicy, RealRobotVLAPolicyConfig
from olmo.eval.websocket_server import WebsocketPolicyServer

logging.basicConfig(level=logging.INFO)


def download_model_from_hf(hf_repo: str, local_path: str) -> str:
    snapshot_download(hf_repo, local_dir=local_path)
    return local_path


def download_model_from_s3(s3_path: str, local_path: str) -> str:
    if not s3_path.startswith('s3://'):
        raise ValueError(f"S3 path must start with 's3://': {s3_path}")

    s3_path_parts = s3_path[5:].split('/', 1)  # Remove 's3://' prefix
    bucket_name = s3_path_parts[0]
    s3_key_prefix = s3_path_parts[1] if len(s3_path_parts) > 1 else ""

    # Get AWS credentials from environment
    aws_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")

    if not aws_key or not aws_secret:
        print("No AWS credentials found in environment variables, using unsigned requests")
        boto_config = botocore.client.Config(signature_version=botocore.UNSIGNED)
    else:
        boto_config = None

    # Create S3 client
    s3_client = boto3.client('s3', aws_access_key_id=aws_key, aws_secret_access_key=aws_secret, config=boto_config)

    # Create local directory
    local_dir = Path(local_path)
    local_dir.mkdir(parents=True, exist_ok=True)

    # List and download all objects with the prefix
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_key_prefix)

    downloaded_files = []
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                s3_key = obj['Key']
                # Create relative path from the prefix
                relative_path = s3_key[len(s3_key_prefix):].lstrip('/')
                if relative_path:  # Skip if empty (directory marker)
                    local_file_path = local_dir / relative_path
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)

                    logging.info(f"Downloading {s3_key} to {local_file_path}")
                    s3_client.download_file(bucket_name, s3_key, str(local_file_path))
                    downloaded_files.append(str(local_file_path))

    if not downloaded_files:
        raise ValueError(f"No files found at S3 path: {s3_path}")

    logging.info(f"Downloaded {len(downloaded_files)} files to {local_dir}")
    return str(local_dir)


def load_molmo(checkpoint_path: str = None, action_type: str = None, no_max_delta: bool = False) -> InferencePolicy:
    """Load SynthVLA policy with SynthVLAPolicyConfig."""

    # Create a mock config object that contains the policy_config
    class MockConfig:
        def __init__(self, policy_config):
            self.policy_config = policy_config

    # Convert DictConfig to SynthVLAPolicyConfig
    # config_dict = OmegaConf.to_container(policy_cfg, resolve=True)
    synthvla_config = RealRobotVLAPolicyConfig()

    # Set checkpoint path if provided
    if checkpoint_path:
        synthvla_config.checkpoint_path = checkpoint_path

    if no_max_delta:
        synthvla_config.relative_max_joint_delta = None

    if action_type:
        if action_type == "joint_pos":  # joint_pos_rel is anyway default
            synthvla_config.action_type = action_type
            synthvla_config.action_keys['arm'] = "joint_pos"

    mock_config = MockConfig(synthvla_config)

    # Create and return SynthVLAPolicy
    policy = RealRobotVLAPolicy(config=mock_config, task_type="manipulation")

    return policy


def get_args():
    parser = argparse.ArgumentParser(description="Serve MolmoBot policy")

    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument("--s3-path", type=str, help="S3 path to model")
    source_group.add_argument("--hf-repo", type=str, help="HuggingFace repo which contains the model")

    parser.add_argument("--local-path", type=str, help="Local path to model, put in ckpts/molmobot/ if not provided")
    parser.add_argument("--action-type", choices=["joint_pos", "joint_pos_rel"], help="Action type", default="joint_pos")
    parser.add_argument("--no-max-delta", action="store_true", help="Disable safety clamp on joint deltas")

    return parser.parse_args()


def main():
    args = get_args()

    if not args.local_path and not args.s3_path and not args.hf_repo:
        raise ValueError("No model source specified!")

    if args.local_path:
        local_path = args.local_path
        checkpoint_path = local_path

    elif args.s3_path:
        local_folder = args.s3_path.rstrip("/").split("/")[-1]
        local_path = f"ckpts/molmobot/{local_folder}"
        logging.info(f"Downloading model from {args.s3_path} to {local_path}")
        checkpoint_path = download_model_from_s3(args.s3_path, local_path)
        logging.info(f"Model downloaded to: {checkpoint_path}")

    elif args.hf_repo:
        local_folder = args.hf_repo.split("/")[-1]
        local_path = f"ckpts/molmobot/{local_folder}"
        logging.info(f"Downloading model from {args.hf_repo} to {local_path}")
        checkpoint_path = download_model_from_hf(args.hf_repo, local_path)
        logging.info(f"Model downloaded to: {checkpoint_path}")

    else:
        raise ValueError("No model source specified!")

    action_type = args.action_type
    policy = load_molmo(checkpoint_path, action_type, args.no_max_delta)
    print(f"Serving {checkpoint_path} policy on {device_count()} {default_device_type()} devices")

    server = WebsocketPolicyServer([policy], "synthvla", port=8000,
                                   metadata={"local_path": checkpoint_path, "s3-path": args.s3_path, "hf_repo": args.hf_repo})
    server.serve_forever()


if __name__ == "__main__":
    main()
