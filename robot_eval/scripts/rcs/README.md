# MolmoBot on DROID with Robot Control Stack

Runs MolmoBot-DROID on a DROID setup (FR3 + Robotiq 2F-85 + ZED Mini wrist + ZED 2/2i exo) driven directly from the
inference workstation through [Robot Control Stack](https://github.com/RobotControlStack/robot-control-stack) (RCS).
No NUC or polymetis is needed: the Franka control box, the Robotiq (USB) and both ZEDs plug into the workstation.

Two processes, two environments:

| Process | Environment | What it does |
|---|---|---|
| `MolmoBot/launch_scripts/serve_molmo.py` | MolmoBot uv venv | Loads the model on the GPU, serves it over a websocket |
| `robot_eval/scripts/rcs/run_policy.py` | RCS venv | Drives the robot and cameras, queries the server, logs to W&B |

## Install (RCS venv)

Install RCS from source with the `rcs_fr3`, `rcs_robotiq2f85` and `rcs_zed` extensions, plus the ZED SDK Python
bindings (`python /usr/local/zed/get_python_api.py` with the RCS venv active). Then:

```bash
source ~/robot-control-stack/.venv/bin/activate
uv pip install -r robot_eval/scripts/rcs/requirements.txt
```

Permissions: add yourself to `dialout` (Robotiq) and `video,plugdev` (ZED), then log out and back in.

Find serials with `python -m rcs_robotiq2f85 serials` and `python -m rcs_zed serials`, and fill them into
[config/rcs_droid.yaml](../../config/rcs_droid.yaml) along with the robot IP, or pass them on the command line.

## Real-time kernel

libfranka expects a PREEMPT_RT kernel. With `robot.ignore_realtime: true` (the default) it runs on a stock kernel, but
heavy load can cause `communication_constraints_violation` reflexes. RCS recovers automatically and the episode ends
with `end_reason=robot_error`. If that happens often, use an RT kernel or reduce load on the control cores.

## Running

1. In Franka Desk, unlock the joints and activate FCI. Alternatively, keep `python -m rcs_fr3 fci <fr3_ip> -u`
   running in its own terminal (needs `DESK_USERNAME`/`DESK_PASSWORD` in a `.env`).
2. Start the policy server:
   ```bash
   cd MolmoBot/MolmoBot && source .venv/bin/activate
   PYTHONPATH=. python launch_scripts/serve_molmo.py --hf-repo allenai/MolmoBot-DROID --action-type joint_pos
   ```
3. Start the client:
   ```bash
   cd robot_eval && source ~/robot-control-stack/.venv/bin/activate
   python -X faulthandler scripts/rcs/run_policy.py task="put the red mug in the black bowl" \
       robot.ip=<fr3_ip> robot.gripper.serial=<robotiq_serial> \
       robot.cameras.wrist_camera=<zed_mini_serial> robot.cameras.exo_camera_1=<zed2_serial> \
       wandb.eval_name=kitchen_v1
   ```

One episode per run, like [the polymetis client](../droid/run_policy.py):

1. The client connects to the policy server, then opens the robot, gripper and cameras.
2. Press Enter to move home (this also opens the gripper).
3. Press Enter to start. Ctrl+C ends the episode; it also ends at `episode.max_duration_s`. A second Ctrl+C aborts
   without saving.
4. The arm holds its last pose while you answer `y`/`n` for success (Ctrl+C at this prompt counts as failure).
5. The episode is saved locally and logged to W&B, then the robot, gripper and cameras are released. Saving
   first means a crash during shutdown can't lose the episode.

Ctrl+C is not an emergency stop: the arm finishes moving toward its last target. Use the stop button for anything
unsafe.

## Point-conditioned policies

For policies trained with `use_point_prompts=True`, set `point_conditioning.enabled=true`. After moving home, a
window shows the `point_conditioning.camera` frame. Click one point per entry in `point_conditioning.labels`, then
press Enter (`r` restarts, Esc aborts). The points are appended to the task exactly as in training
(`SynthmanipDataset._format_point_prompt`), e.g. `task <points coords="1 1 423 512">pickup_obj</points>`, with
0-1000 coordinates normalized to that camera's image. Labels must match the object names used in training.

`point_conditioning.append_to_prompt=false` keeps the GUI, the video dot and the logging but sends the task unchanged,
for trying the workflow with a policy that wasn't trained on points.

Logged in addition: the points drawn on the point camera's panel of `video/agent_view`, `point_selection` (the frame
you clicked on, full resolution), and `prompt` / `points` in the run summary.

Training with `use_point_prompts=True` also appends the point camera's frame at the time the points were taken (the
conditioning image) as an extra model input. `serve_molmo.py` / `RealRobotVLAPolicy` don't do this yet, so the
server needs that support before a point-trained policy gets the inputs it was trained with.

## What gets logged

Each episode is one W&B run in `wandb.project`:

- **Policy identity.** `group` and the `policy_id` tag are `<name>-<fingerprint>`. The fingerprint is a hash of the
  checkpoint directory the server loaded (all small files, plus the size and first/last 16 MiB of large ones), so it
  is the same for a given checkpoint wherever it came from: HF, S3 or a local dir. The client reads the server's
  `local_path` from its metadata, resolved relative to `MolmoBot/MolmoBot`; set `policy.checkpoint_path` if the server
  was launched elsewhere. `policy.name` overrides the readable part of the id.
  `config.policy` holds the full server metadata.
- **Agent view.** `video/agent_view` shows the exact images sent to the policy each step, side by side. Set
  `recording.camera_videos=true` to also log full-resolution camera videos.
- **Outcome and timing.** `success`, `end_reason`, episode length, actual control rate, inference latency.
- **Trajectory.** `trajectory.npz` with per-step joint positions, gripper state, actions and latencies.
- **Reproducibility.** `config.robot`, `config.image_input` and `config.client` (MolmoBot git commit and dirty flag,
  hostname, RCS version).

To compare policies, group the runs table by `group` (or `config.policy.policy_id`) and average `success`. Filter by
`config.eval_name` or `config.task` for a specific eval set.

Local copies of everything are in `robot_eval/outputs/rcs/<date>/<time>/`.
