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
