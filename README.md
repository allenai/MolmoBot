<div align="center">
  <img src="assets/MolmoBot.png" alt="MolmoSpaces Logo" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/></br>
  Large-Scale Simulation Enables Zero-Shot Manipulation
  <div align="center">
    <a href="https://arxiv.org/pdf/2603.16861" target="_blank" rel="noopener noreferrer"><img src="assets/button_paper.svg" /></a>
    <a href="https://allenai.github.io/MolmoBot" target="_blank" rel="noopener noreferrer"><img src="assets/button_website.svg" /></a>
    <a href="https://github.com/allenai/molmospaces" target="_blank" rel="noopener noreferrer"><img src="assets/button_code_data.svg" /></a>
    <a href="https://huggingface.co/collections/allenai/molmobot-models" target="_blank" rel="noopener noreferrer"><img src="assets/button_models.svg" /></a>
    <a href="https://huggingface.co/datasets/allenai/MolmoBot-Data" target="_blank" rel="noopener noreferrer"><img src="assets/button_data.svg" /></a>
  </div>
  </h1>
</div>
<br>

Code and website for "MolmoB0T: Large-Scale Simulation Enables Zero-Shot Manipulation".

# Getting started

MolmoBot policies have strong demonstrated sim-to-real transfer to a wide variety of novel scenes, objects, and camera viewpoints. Try it out for yourself on your DROID platform with MolmoBot-DROID!

MolmoBot-DROID uses only the wrist camera and 1 exo camera. Don't worry about camera placement, MolmoBot policies are robust to arbitrary camera viewpoints!

## Trying it out in simulation

See [here](MolmoBot/README.md#demo-notebook) to try out MolmoBot interactively! Modify the scene and task to test policy behavior.

## Set up and run MolmoBot-DROID

1. Set up MolmoBot-DROID by following the [installation instructions](MolmoBot/README.md).
2. See [these instructions](robot_eval/scripts/droid/README.md) for detailed instructions on setting up and running the policy on your DROID! Any existing DROID or polymetis setups will work easily.

   Briefly, after starting the polymetis robot and gripper servers:

   ```bash
   # In one terminal
   cd MolmoBot/MolmoBot
   source .venv/bin/activate
   PYTHONPATH=. python launch_scripts/serve_molmo.py --hf-repo allenai/MolmoBot-DROID --action-type joint_pos
   ```

   ```bash
   # in another terminal
   cd MolmoBot/robot_eval
   conda activate molmobot
   python scripts/droid/run_policy.py robot.robot_host=<nuc_ip> robot.cameras.wrist_camera.id=<wrist_id> robot.cameras.exo_camera_1.id=<exo_id> task="put the red mug in the black bowl"
   ```


# Using MolmoBot Data

To use MolmoBot-Data for training experiments, you will need to download it from hugging face using [bulk_download.py](https://huggingface.co/datasets/allenai/molmobot-data/blob/main/bulk_download.py). See [here](https://github.com/allenai/molmospaces/blob/main/docs/data_format.md) for further documentation on the data format.

## Data postprocessing

Before using any dataset implementations in this repo, you will need to run a [postprocessing script](https://huggingface.co/datasets/allenai/molmobot-data/blob/main/validate_trajectories.py). This filters out any corrupted trajectories, and can optionally check for visibility of certain objects in a given camera. Below is some example usage of the script. See [here](https://github.com/allenai/molmospaces/blob/main/docs/data_processing.md) for more information.

Example usage:
```bash
python validate_trajectories.py RBY1OpenDataGenConfig/part0/train --check-visibility head_camera door_handle

python validate_trajectories.py RBY1PickAndPlaceDataGenConfig/part0/train --check-visibility head_camera pickup_obj --check-visibility head_camera place_receptacle

python validate_trajectories.py FrankaPickAndPlaceOmniCamConfig/part0/train --check-visibility droid_shoulder_light_randomization pickup_obj --check-visibility droid_shoulder_light_randomization place_receptacle
```

## Data statistics

Before training (and after data postprocessing), you should also calculate aggregate statistics with [calculate_stats.py](https://huggingface.co/datasets/allenai/molmobot-data/blob/main/calculate_stats.py). Example usage:

```bash
python calculate_stats.py FrankaPickAndPlaceOmniCamConfig/part0/train --keys actions obs/agent/qpos

python calculate_stats.py RBY1OpenDataGenConfig/part0/train --keys actions obs/agent/qpos

python calculate_stats.py RBY1PickAndPlaceDataGenConfig/part0/train --keys actions obs/agent/qpos
```

# BibTeX

```
@inproceedings{deshpande2026molmobot,
    title={MolmoB0T: Large-Scale Simulation Enables Zero-Shot Manipulation},
    author={Abhay Deshpande and Maya Guru and Rose Hendrix and Snehal Jauhri and Ainaz Eftekhar and Rohun Tripathi and Max Argus and Jordi Salvador and Haoquan Fang and Matthew Wallingford and Wilbert Pumacay and Yejin Kim and Quinn Pfeifer and Ying-Chun Lee and Piper Wolters and Omar Rayyan and Mingtong Zhang and Jiafei Duan and Karen Farley and Winson Han and Eli Vanderbilt and Dieter Fox and Ali Farhadi and Georgia Chalvatzaki and Dhruv Shah and Ranjay Krishna},
    booktitle={10th Annual Conference on Robot Learning},
    year={2026},
    url={https://openreview.net/forum?id=c4Z0Y8gvnI}
}
```
