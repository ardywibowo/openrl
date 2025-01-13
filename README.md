# OpenRL


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code for fast iteration and performantly implementing various reinforcement learning ideas. Various supervised learning tasks can also be framed in this framework as a special case of reinforcement learning.

## Methods Implemented

- [**Proximal Policy Optimization (PPO)**](https://arxiv.org/abs/1707.06347): The main reinforcement learning algorithm implemented in this codebase.
- [**VinePPO**](https://arxiv.org/abs/2410.01679): A variant of PPO that uses rollouts to compute MC advantages.
- [**Linguistic Calibration**](https://arxiv.org/abs/2404.00474): A method to calibrate the language model to the task.
- [**Generative Verifiers**](https://arxiv.org/abs/2408.15240): A method that models the verifier as a generative model.
- [**Reward Modeling**](https://arxiv.org/abs/2203.02155): A reward model SFT trainer.
- **Behavior Cloning (Imitation Learning)**: A maximum likelihood estimation trainer to do behavior cloning.

## Coming Soon

- [**Free Process Rewards without Process Labels**](https://arxiv.org/abs/2412.01981): Trains a reward model a la DPO and uses it to perform PPO.

## Quick Start

### Installation

Firstly, add your keys in a file named `private/keys.sh`. An example is as follows:

```bash
# Set environment variables
export APP_SEED="[Seed for reproducibility]"
export WANDB_API_KEY="[Wandb API Key]"
huggingface-cli login --token "[Huggingface Token]"
```

This project is implemented based torch, Huggingface, FlashAttention, DeepSpeed, and vLLM libraries. To obtain the dependencies, run the `setup.sh` script. This script installs the necessary libraries, frameworks, and downloads example datasets for finetuning models for math to get you started:

```bash
bash setup.sh
```

*Optional: You can also use the following [Dockerfile](https://github.com/ardywibowo/openrl/blob/main/Dockerfile) to build your own image*

### Running Experiments

Once you have selected the configuration file, you can run various experiments configured in the [configs/experiments](https://github.com/ardywibowo/openrl/configs/experiments) folder. An example script is provided in `run.sh`. Please replace the `HF_TOKEN` and `WANDB_API_KEY` with your own tokens.

```bash
bash run.sh
```

*Refer to `src/treetune/runtime/policy_iteration_runtime.py` if you'd like to start reading the codebase.*

## Code Structure
- [`configs`](https://github.com/ardywibowo/openrl/tree/main/configs): Contains Jsonnet files for configuring experiment settings.
- [`configs/experiments`](https://github.com/ardywibowo/openrl/tree/main/configs/experiments): The location where you should organize all of your different experiment configurations.
- [`src/treetune`](https://github.com/ardywibowo/openrl/tree/main/src/treetune): The main directory for source code, encompassing:
    - [`models`](https://github.com/ardywibowo/openrl/tree/main/src/treetune/models): Contains model loading, with [`pretrained.py`](https://github.com/ardywibowo/openrl/tree/main/src/treetune/models/pretrained.py) the central piece to load HF models.
    - [`episode_generators`](https://github.com/ardywibowo/openrl/tree/main/src/treetune/episode_generators): Manages the episode generation pipelines. The [`math_episode_generator.py`](https://github.com/ardywibowo/openrl/tree/main/src/treetune/episode_generators/math_episode_generator.py) script is key for PPO episode generation and [`math_episode_generator_with_mc_advantages.py`](https://github.com/ardywibowo/openrl/tree/main/src/treetune/episode_generators/math_episode_generator_with_mc_advantages.py) creates the episodes for VinePPO.
    - [`trainers`](https://github.com/ardywibowo/openrl/tree/main/src/treetune/trainers): Contains trainer classes, with [`ppo_trainer.py`](https://github.com/ardywibowo/openrl/tree/main/src/treetune/trainers/ppo_trainer.py) is the main PPO trainer which is shared between PPO and VinePPO.
    - [`runtime`](https://github.com/ardywibowo/openrl/tree/main/src/treetune/runtime): Integrates components and implements training and evaluation procedures. The [`policy_iteration_runtime.py`](https://github.com/ardywibowo/openrl/tree/main/src/treetune/runtime/policy_iteration_runtime.py) script is the **starting point for running experiments.**
- [`src/guidance`](https://github.com/ardywibowo/openrl/tree/main/src/treetune): We ship the [guidance](https://github.com/guidance-ai/guidance) module directly with the codebase. 

### Important Representative Files
Trainers:
- [`ppo_trainer.py`](https://github.com/ardywibowo/openrl/tree/main/src/treetune/trainers/ppo_trainer.py): A PPO trainer.
- [`binary_classification_trainer.py`](https://github.com/ardywibowo/openrl/tree/main/src/treetune/trainers/binary_classification_trainer.py): An example of how you can do binary classification SFT with the repo.
- [`reward_modeling_trainer.py`](https://github.com/ardywibowo/openrl/tree/main/src/treetune/trainers/reward_modeling_trainer.py): A reward model SFT trainer.
- [`mle_trainer.py`](https://github.com/ardywibowo/openrl/tree/main/src/treetune/trainers/mle_trainer.py): A maximum likelihood estimation trainer to do behavior cloning (imitation learning).

Episode Generators:
- [`math_episode_generator.py`](https://github.com/ardywibowo/openrl/tree/main/src/treetune/episode_generators/math_episode_generator.py): The PPO episode generator.
- [`binary_classification_episode_generator.py`](https://github.com/ardywibowo/openrl/tree/main/src/treetune/episode_generators/binary_classification_episode_generator.py): An example of how you can do binary classification SFT with the repo.
- [`sft_episode_generator.py`](https://github.com/ardywibowo/openrl/tree/main/src/treetune/episode_generators/binary_classification_episode_generator.py)

Tasks:
- [`math.py`](https://github.com/ardywibowo/openrl/tree/main/src/treetune/tasks/math.py): The main task file for MATH dataset.
- [`linguistic_calibration/`](https://github.com/ardywibowo/openrl/tree/main/src/treetune/tasks/linguistic_calibration): Contains training tasks for the various steps needed to implement the linguistic calibration method (Paragraph Generation, Answer Extraction, Probability Forecasting, and final RL training).
- [`reward_modeling/reward_bench.py`](https://github.com/ardywibowo/openrl/tree/main/src/treetune/tasks/reward_modeling/reward_bench.py): Contains training tasks for RewardBench, a benchmark dataset for evaluating reward modeling performance.


## Updates
- (Jan 8th, 2025) First public release.

## Acknowledgement

This codebase heavily takes pieces from the [VinePPO](https://github.com/ardywibowo/openrl/) repo, which takes pieces from the [guidance](https://github.com/guidance-ai/guidance), [OpenAI PRM Repo](https://github.com/openai/prm800k), and [DeepSeekMath](https://github.com/deepseek-ai/DeepSeek-Math) repos. My additional contributions minimally help to organize vastly different experiments (such as linguistic calibration, reward model training, generative verifiers etc.) as the original repo was meant for a specialized experiment setting (Math RL).
