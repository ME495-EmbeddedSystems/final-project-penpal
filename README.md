# Homework 3 Part 2

Authors: Conor Hayes, Kyuwon Weon, Amber Handal, Tianhao Zhang

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

## Run Setup
Requires ros2 kilted kaiju and Ubuntu 24.04.
May work on other distros, but not tested on them.

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

```

## Usage and Commands
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


## Challenges
### Integration
- Significantly multithreaded code in the penpal node - many actions need to be
done in parallel
- Integrating many distinct functions into one architecture

### WritePlanner & Transforms
- management of many frames, complex trajectories, and the transforms between them
- most of these transforms were handled manually (using numpy, scipy) rather than using the TF tree, due to the sheer amount of information to handle.