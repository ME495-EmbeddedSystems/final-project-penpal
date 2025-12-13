# Penpal: Vision-Guided Robotic Whiteboard Q&A

__Authors: Conor Hayes, Kyuwon Weon, Amber Handal, Tianhao Zhang__

## Project Overview
PenPal uses a vision-guided robotic system to detect a whiteboard in the environment, read handwritten questions from the board using the Gemini vision-language model, generate concise answers, and physically writes responses back onto the board using a Franka Emika arm.

The system integrates:
- Computer vision (AprilTag-based pose estimation, `OpenCV` preprocessing)
- Vision-language models (`Gemini VLM` for OCR + question answering)
- Robot motion planning (`MoveIt` Cartesian path planning)
- Frame-consistent spatial reasoning (TF trees and rigid-body transforms)

All perception, reasoning, and motion are performed dynamically at runtime, allowing the robot to adapt to changes in board position and orientation.

## Quickstart Guide

1. __Prerequisites__
- Ubuntu 24.04
- Python 3.10+
- ROS 2 (tested with `Kilted Kaiju`)
- MoveIt 2
- Google Gemini API key
- Franka ROS stack
- Realsense D435i

2. __Run Setup__
```bash
# run the following from your ROS2 workspace where
# this repo is in the src folder:

# install dependencies
rosdep install --from-paths src --ignore-src --rosdistro kilted 

# WARNING - the below worked fine on our computers, BUT
# there are warnings not to do this from Ubuntu. Move forward
# at your own risk...
pip install --break-system-packages google-genai torch

colcon build

# set the google API key in your shell:
export GOOGLE_API_KEY='[INSERT YOUR API KEY HERE]'

# build the workspace
`cd ~/ws`
`colcon build --symlink-install`
`source install/setup.bash`

# launch Penpal!
`ros2 launch penpal penpal.launch.py`
```

The robot will:

Detect the board

Read the question

Generate a minimal answer

Plan and execute a writing trajectory

3. __Usage and Commands__
```bash
## ONE-OFF COMMANDS
# to write a single hard-coded message
ros2 action send_goal /write_message penpal_interfaces/action/WriteMessage '{text: "HELLO WORLD"}'

# to pick up the pen
ros2 service call /grab_pen example_interfaces/srv/Trigger "{}"

## CONVERSATIONAL COMMANDS
# begin conversational mode
ros2 service call /wake example_interfaces/srv/Trigger "{}" 

# disable conversational mode
ros2 service call /sleep example_interfaces/srv/Trigger "{}" 

```

## System Architecture
CONOR!!!!!

### Nodes
`penpal.py`
    Top-level orchestrator + FSM. Watches for board visibility, triggers OCR/VLM via a Trigger service client, and sends generated text to the writing stack (planner + controller). Provides wake, sleep, grab_pen services and write_message action. Subscribes to board_info.
`board_detector.py`
	Real board detector. Consumes AprilTag detections and produces penpal_interfaces/BoardInfo (board_info) containing board pose, dimensions, writable area, and detection metadata (e.g., tag count, sequence number).
`ocr_node.py`
	Real OCR + QA node. Provides read_and_answer_board (example_interfaces/Trigger) which captures/uses the latest image and returns a JSON payload containing transcription + a concise answer (and raw debug fields).
`mock_board_detector.py`
	Publishes a fixed/synthetic board_info for development when the camera/AprilTags aren’t running (useful for testing planning/writing).
`mock_ocr_node.py`
	Mock Trigger-service implementation of OCR/QA. Always returns a static JSON payload for end-to-end testing of PenPal without VLM dependencies.

### Launchfiles
+`penpal.launch.py`
Launches the full system (PenPal + MoveIt RViz + optionally vision, or mocks).

Launch arguments:
- `vision` (`default: true`)
- `mock`: start `mock_ocr_node` + `mock_board_detector` instead (no real vision)
- `controller` (`default: moveit`)

+`penpal_vision.launch.py`
Launches the vision system (Realsense + Apriltags + board detection + OCR)

Launch arguments:
- `gemini_api_key` (`default: $GOOGLE_API_KEY`)
- `run_rviz` (`default: true`)

+`moveit_ctl.launch.py`
Integration-test launchfile for the MoveIt-based controller.


## Challenges
### Integration
- Significantly multithreaded code in the penpal node - many actions need to be
done in parallel
- Integrating many distinct functions into one architecture

### WritePlanner & Transforms
- management of many frames, complex trajectories, and the transforms between them
- most of these transforms were handled manually (using numpy, scipy) rather than using the TF tree, due to the sheer amount of information to handle.

## Development Instructions
This repo contains a pre-commit hook that performs lint checks before you're allowed to commit code, and also auto-formats some of those errors for you (i.e. replacing double quotes with single quotes). This is so we don't have to go back and fight with with ament_lint for a million years like we did in the previous project.

It uses the python `pre-commit` framework + the `ruff` formatter+linter to do this; see docs for [pre-commit](https://pre-commit.com/) and [ruff](https://docs.astral.sh/ruff/).

In order to set it up, do the following:

```bash
# install the pre-commit program
sudo apt install pre-commit

# install the pre-commit hooks to the repo (as configured in .pre-commit-config.yaml)
pre-commit install
```

All done! Now every time you commit, it will run lint checks + do some autoformatting to make sure that we 
stick to the ROS2 style guidelines (mostly. it doesn't do everything for us/check everything).
