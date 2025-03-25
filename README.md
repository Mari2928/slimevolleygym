# NEAT for Slime Volleyball

NEAT for Slime Volleyball implements the NeuroEvolution of Augmenting Topologies (NEAT) approach to train the agent in the OpenAI Gym environemnt.

### Requirements
This project requires the following packages and some depend on the specific versions. The conda environment was built with Python==3.11.9.
```text
pyglet==1.5.11
gym==0.19.0
pickle
graphviz
matplotlib
numpy
math
neat-python
```
## Installation

Install from the repository to access a demo, training scripts, and pre-trained models. Installation of `graphviz` might depend on your OS.

```
git clone https://github.com/Mari2928/slimevolleygym.git
cd slimevolleygym
pip install -e .
```

## Basic Usage

After installing from the repo, you can play the game against the NEAT agent by running:

```
python test_state.py
```

You can control the agent on the right using the arrow keys, or the agent on the left using (A, W, D).

## Evaluating the NEAT agent against baseline

You can run the NEAT agent against baseline using the following command:

```
python eval_agents.py --left baseline --right neat --render
```

<p align="left">
  <img width="50%" src="http://lovelyn.biz/slime.GIF"></img>
  <br/><i>Evaluating NEAT agent (right) against baseline (left).</i>
</p>

## Leaderboard

Below are scores achieved by various algorithms and links to their implementations. Feel free to add yours here:

### SlimeVolley-v0

| Method                                                                         |Average Score|Episodes|Other Info
|--------------------------------------------------------------------------------|--|---|---|
| Maximum Possible Score                                                         |5.0|  | 
| PPO                                                                            | 1.377 ± 1.133 | 1000 | [link](https://github.com/hardmaru/slimevolleygym/blob/master/TRAINING.md)
| CMA-ES                                                                         | 1.148 ± 1.071 | 1000 | [link](https://github.com/hardmaru/slimevolleygym/blob/master/TRAINING.md)
| GA (Self-Play)                                                                 | 0.353 ± 0.728 | 1000 | [link](https://github.com/hardmaru/slimevolleygym/blob/master/TRAINING.md)
| CMA-ES (Self-Play)                                                             | -0.071 ± 0.827 | 1000 | [link](https://github.com/hardmaru/slimevolleygym/blob/master/TRAINING.md)
| PPO (Self-Play)                                                                | -0.371 ± 1.085 | 1000 | [link](https://github.com/hardmaru/slimevolleygym/blob/master/TRAINING.md)
| Random Policy                                                                  | -4.866 ± 0.372 | 1000 |
| NEAT                                                                           | -1.854 ± 1.745 | 1000 | 
| [Add Method](https://github.com/hardmaru/slimevolleygym/edit/master/README.md) |  |  |

### SlimeVolley-v0 (Against Other Agents)

Table of average scores achieved versus agents other than the default baseline policy ([1000 episodes](https://github.com/hardmaru/slimevolleygym/blob/master/eval_agents.py)). 
The score against PPO is unavailable, as `stable_baselines` was outdated due to incompatibility with the old Tensorflow version.

| Method                                                                         |Baseline|PPO|CMA-ES|GA (Self-Play)| NEAT      | Other Info
|--------------------------------------------------------------------------------|---|---|---|---|-----------|---|
| PPO                                                                            |  1.377 ± 1.133 | — |  0.133 ± 0.414 | -3.128 ± 1.509 | —         | [link](https://github.com/hardmaru/slimevolleygym/blob/master/TRAINING.md)
| CMA-ES                                                                         | 1.148 ± 1.071 | -0.133 ± 0.414 | — | -0.301 ± 0.618 | -2.155 ± 1.5          | [link](https://github.com/hardmaru/slimevolleygym/blob/master/TRAINING.md)
| GA (Self-Play)                                                                 | 0.353 ± 0.728 | 3.128 ± 1.509 | 0.301 ± 0.618 | — | -1.519 ± 1.349 | [link](https://github.com/hardmaru/slimevolleygym/blob/master/TRAINING.md)
| CMA-ES (Self-Play)                                                             | -0.071 ± 0.827 |  -0.749 ± 0.846 |  -0.351 ± 0.651 |  -4.923 ± 0.342 | —            | [link](https://github.com/hardmaru/slimevolleygym/blob/master/TRAINING.md)
| PPO (Self-Play)                                                                | -0.371 ± 1.085 | 0.119 ± 1.46 |  -2.304 ± 1.392 |  -0.42 ± 0.717 | —         | [link](https://github.com/hardmaru/slimevolleygym/blob/master/TRAINING.md)
| NEAT                                                                           | -1.854 ± 1.745  | — | -2.155 ± 1.5 |  -1.519 ± 1.349 | —         | [link](https://github.com/hardmaru/slimevolleygym/blob/master/TRAINING.md)
| [Add Method](https://github.com/hardmaru/slimevolleygym/edit/master/README.md) |  |  | |

It is interesting to note that while NEAT did not perform well against the baseline policy compared to GA, it achieves a better score if evaluated against it that outperforms baseline.
