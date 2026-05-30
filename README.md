# Deployment Code of Extreme Parkour on Unitree Go2

This repository provides an **unofficial implementation** for deploying the project [Extreme Parkour with Legged Robots](https://github.com/chengxuxin/extreme-parkour) on the **Unitree Go2** quadrupted robot. The original work was developed for A1 robots and does not provide the deployment code. 

## Key Contributions

- Add detailed comments throughout the training code. Documented the previously unexplained **observation** vector.

- ~~Add camera randomization during training to accout for Go2's movable camera (unlike A1's fixed camera).~~

- Provide train weights and deployment code for Unitree Go2.

## Deployment Instructions

#### Environment Setup
Make sure the environment is properly set up on your Go2 robot, including rclpy, torch and unitree sdk.

#### Hardware Setup
Install the **Intel RealSense D435i** depth camera on the Go2. Verify that the captured images resemble the simulation (can be checked using `rviz`).

#### Deployment Steps
1. Connect to the Go2 robot wirelessly via SSH (wired connection is also ok).
```bash
ssh unitree@<go2_ip_address>
```

2. In the first terminal, start the visual node:
```bash
python3 visual_extreme_parkour.py --logdir traced
```

This script retrieves depth images from the D435i camera and publish them at 100Hz to the appropriate ROS topic.

3. In a second terminal, start the controller node:
```bash
python3 run_extreme_parkour.py --logdir traced --mode parkour --nodryrun
```
This script fuses the depth image and proprioception data. Now the robot is in the sport mode:
- Press **R1** to stand up.
- Press **R2** to lie down.
- Press **L1** to disable the builtin sport service and execute the stand policy.
- After turning off the builtin sport service, press **Y** to start executing the parkour policy.
- When finished, press **L2** to exit the parkour mode and re-enable native motion control.

## Notes and Tips

#### Heightmap Student Deployment (No Depth Camera)
If your student model is distilled from **heightmap points** (e.g. LiDAR elevation samples) and does not rely on depth camera encoder, use:
```bash
python3 run_extreme_parkour_heightmap.py \
	--logdir traced \
	--heightmap_model your_heightmap_jit.pt \
	--heightmap_topic /heightmap_points \
	--n_points 132 \
	--mode parkour --nodryrun
```

The topic `--heightmap_topic` should publish `std_msgs/Float32MultiArray` with exactly `n_points` values per message.
This script keeps the same joystick state machine as depth deployment:
- `R1`: stand up
- `R2`: lie down
- `L1`: disable built-in sport mode and switch to stand policy
- `Y`: start parkour policy
- `L2`: exit parkour and re-enable built-in sport mode

#### MuJoCo Heightmap Debug
Use `run_mujoco_heightmap_serial.py` to run the heightmap student policy directly in MuJoCo. The default motor backend is MuJoCo, so no real motor serial port is required.

Example command:
```bash
python3 run_mujoco_heightmap_serial.py \
	--heightmap_model /home/rc_kfs/Extreme-Parkour-Onboard/traced/student_34000-34000-heightmap_jit.pt \
	--device cpu \
	--mode parkour \
	--height_noise_std 0 \
	--command_source mux \
	--keyboard_initial_mode 0 \
	--motor_backend mujoco \
	--debug_items height \
	--debug_every 10 \
	--debug_height_count 20 \
	--visualize_heightmap
```

Common options:
- `--heightmap_model`: traced heightmap student policy.
- `--mujoco_xml`: MuJoCo robot XML. Default is `/home/rc_kfs/extreme-parkour/legged_gym/resources/robots/mybot_v3/xml/mybot_v3.xml`.
- `--command_source`: command input source. `mux` uses UDP first, then keyboard, then fixed command.
- `--keyboard_initial_mode 0`: start in idle mode.
- `--motor_backend mujoco`: send policy targets directly to MuJoCo. Use `sdk` only when testing real motor serial output.
- `--height_noise_std 0`: disable heightmap noise for debugging.
- `--visualize_heightmap`: draw sampled heightmap points in the MuJoCo viewer.

Keyboard controls:
```text
0      idle / zero torque
1      stand up / default pose
2      RL policy mode
3      joint damping
4      return default pose

W/S    increase/decrease vx
Q/E    increase/decrease vy
A/D    increase/decrease yaw rate
R      reset velocity command to zero
Space  emergency stop
Esc    stop keyboard listener
```

Debug output is controlled by:
```bash
--debug_items cmd,base,height,obs,action,target,joint,motor
--debug_every 10
--debug_height_count 20
```

Debug fields:
- `cmd`: current velocity command and estop flag.
- `base`: MuJoCo base position and yaw.
- `height`: heightmap values sent to the student model. `min/max/mean` show the full scan statistics, and `firstN` prints the first `N` points.
- `obs`: proprioceptive observation statistics.
- `action`: raw policy action statistics.
- `target`: target joint positions after action scaling.
- `joint`: current MuJoCo joint position and velocity.
- `motor`: Python Unitree SDK feedback, only available with `--motor_backend sdk` or `both`.

For heightmap debugging, the important check is whether `height min/max` changes when the sampled points pass over obstacles. On flat ground, all height values can be nearly identical. The current MuJoCo debug formula temporarily uses `base_z - measured_terrain_height`; the original training formula is `base_z - 0.3 - measured_terrain_height` and is left as a comment in the script.

#### Policy selection:
Modify in `run_extreme_parkour.py`:
```bash
base_model = 'your_base_model.pth'
vision_model = 'your_vision_model.pth'
```

#### Walk Mode:
It is recommended to test the walk mode to verify your model and camera setup:
```bash
python3 run_extreme_parkour.py --logdir traced --mode walk --nodryrun
```
Use ``--mode walk`` or ``--mode parkour`` to switch between **walking** and **parkour** mode, as they were trained as separate tasks in the original work. You can aslo use
```bash
python3 run_extreme_parkour.py --logdir traced --mode walk
```
to perform a more conservative test. This command runs the policy without sending actions to the motors — useful for verifying perception and inference without physical movement.

## Performance 
The Go2 is capable of climbing over obstacles up to **40 cm** in height. Video will be provided soon.

## Acknowledgments
This repository is based on modification of [Robot Parkour Learning](https://github.com/ZiwenZhuang/parkour). Special thanks to the original authors for their open-source contribution.

## Contact
I am a beginner in robotics, and warmly welcome feedback and contributions to improve this repository. For questions, suggestions or collaboration, please open an issue or contact me directly.
