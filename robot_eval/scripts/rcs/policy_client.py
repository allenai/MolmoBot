"""Client for MolmoBot's WebsocketPolicyServer (MolmoBot/olmo/eval/websocket_server.py)."""

import hashlib
import os
from pathlib import Path

import msgpack_numpy
from websockets.sync.client import connect


class PolicyClient:
    """One connection per episode: the server resets the policy's state on every new connection."""

    def __init__(self, host: str, port: int):
        self._ws = connect(f"ws://{host}:{port}", compression=None, max_size=None, open_timeout=10)
        self._packer = msgpack_numpy.Packer()
        self.metadata: dict = msgpack_numpy.unpackb(self._ws.recv())

    def infer(self, obs: dict) -> dict:
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            # The server sends a traceback as text before closing on error.
            raise RuntimeError(f"Policy server error:\n{response}")
        return msgpack_numpy.unpackb(response)

    def close(self):
        self._ws.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def checkpoint_fingerprint(checkpoint_path: str) -> str | None:
    """Content fingerprint of a checkpoint dir.

    Small files are hashed fully; large files contribute their size and first/last 16 MiB, which
    identifies different weights without reading multi-GB files.
    Returns None if the directory isn't readable from this machine.
    """
    root = Path(checkpoint_path)
    if not root.is_dir():
        return None
    chunk = 16 * 1024 * 1024
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and ".cache" not in p.parts):
        size = path.stat().st_size
        h.update(f"{path.relative_to(root)}:{size}\n".encode())
        with path.open("rb") as f:
            if size <= 2 * chunk:
                h.update(f.read())
            else:
                h.update(f.read(chunk))
                f.seek(-chunk, os.SEEK_END)
                h.update(f.read(chunk))
    return h.hexdigest()


# serve_molmo.py is launched from MolmoBot/MolmoBot, so relative checkpoint paths resolve there.
SERVER_DIR = Path(__file__).resolve().parents[3] / "MolmoBot"


def resolve_checkpoint_path(metadata: dict, override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    local_path = metadata.get("local_path")
    if not local_path:
        raise RuntimeError("Server metadata has no local_path; set policy.checkpoint_path in the config.")
    path = Path(local_path).expanduser()
    return path if path.is_absolute() else (SERVER_DIR / path).resolve()


def policy_identity(metadata: dict, name_override: str | None, checkpoint_override: str | None) -> dict:
    """Build a stable identity for the served policy from the server's metadata."""
    ckpt = resolve_checkpoint_path(metadata, checkpoint_override)
    source = metadata.get("hf_repo") or metadata.get("s3-path") or str(ckpt)
    name = name_override or Path(str(source).rstrip("/")).name
    fingerprint = checkpoint_fingerprint(str(ckpt))
    if fingerprint is None:
        raise RuntimeError(
            f"Cannot read checkpoint dir {ckpt} to fingerprint it. If the server was launched from a "
            "different directory, set policy.checkpoint_path to the checkpoint dir it loaded."
        )
    return {
        "policy_id": f"{name}-{fingerprint[:8]}",
        "name": name,
        "source": source,
        "local_path": str(ckpt),
        "checkpoint_fingerprint": fingerprint,
        "server_metadata": metadata,
    }
