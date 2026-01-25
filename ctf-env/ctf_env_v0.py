from ray.rllib.algorithms import PPOConfig, PPO
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune import register_env
from env.ctf_env import CTFEnv
from env.ctf_env import get_team
import warnings
import os
import argparse
from ray.tune.callback import Callback

warnings.filterwarnings("ignore", category=DeprecationWarning)

NUM_OF_ITERATIONS = 4000
RUN_CONFIG_NAME = 'ppo_ctf_training'


class RewardLoggerCallback(Callback):
    def on_trial_result(self, iteration, trials, trial, result, **info):
        """Called after each trial reports results."""
        mean_reward = result.get("env_runners/episode_return_mean", None)
        if mean_reward is not None:
            print(f"Trial {trial.trial_id} - Iteration {iteration}: Mean Reward = {mean_reward:.2f}")


def betas_tensor_to_float(learner):
    param_grp = next(iter(learner._optimizer_parameters.keys())).param_groups[0]
    param_grp["betas"] = tuple(beta.item() for beta in param_grp["betas"])


def env_creator(config):
    # create petting zoo ctf environment
    env = CTFEnv(config['width'], config['height'], config['num_of_team_agents'], config['render_mode'],
                 config['max_steps'])

    return ParallelPettingZooEnv(env)


def evaluate(results):
    """Return best checkpoint based on episode_mean_return."""
    best_result = results.get_best_result(
        metric="env_runners/episode_return_mean",
        mode="max"
    )

    print(f"Best trial config: {best_result.config}")
    print(f"Best trial final return: {best_result.metrics['env_runners']['episode_return_mean']}")
    print(f"Best checkpoint: {best_result.checkpoint}")
    return best_result


def train_algorithm(algo, checkpoint_path):
    for i in range(NUM_OF_ITERATIONS):
        result = algo.train()
        print(f"ITERATION {i}: reward={result['env_runners']['episode_return_mean']}, metadata={result['env_runners']}")
        if "evaluation" in result:
            # TODO eval_reward = result['evaluation']['env_runners']['episode_return_mean']
            print(f"EVAL DONE")

        if i % 10 == 0:
            checkpoint = algo.save(checkpoint_path)
            print(f"Checkpoint saved at {checkpoint}")


def init(render_mode, field_size, model_path, max_steps, execution_mode):
    data_path = os.path.abspath(model_path)
    env_config = {
        "width": field_size,
        "height": field_size,
        "num_of_team_agents": 2,
        "render_mode": render_mode,
        "max_steps": max_steps
    }

    register_env("ctf_env", env_creator)

    config = (
        PPOConfig()
        .resources(num_gpus=1, num_gpus_per_learner_worker=1)
        .environment(env="ctf_env", env_config=env_config)
        .multi_agent(
            policies={
                "red_policy": (None, None, None, {}),
                "blue_policy": (None, None, None, {}),
            },
            policy_mapping_fn=lambda agent_id, episode, **kwargs: (
                f"{get_team(agent_id)}_policy"
            ),
        )
        .env_runners(
            num_env_runners=1,
            sample_timeout_s=240,
            explore=True,
        )
        .training(
            train_batch_size=6000,
            entropy_coeff=[
                [0, 0.05],
                [2000000, 0.02],
                [6000000, 0.01],
                [20000000, 0.0]
            ],
            lr=[
                [0, 3e-4],
                [2000000, 1e-4],
                [10000000, 5e-5]
            ]
        )
        .evaluation(
            evaluation_num_env_runners=1,
            evaluation_interval=10,
            evaluation_duration=5,
            evaluation_force_reset_envs_before_iteration=True,
            evaluation_parallel_to_training=True,
            evaluation_config={
                "env_config": {
                    **env_config,
                    "render_mode": "human"  # override render mode for evaluation
                },
                "explore": False  # deterministic evaluation
            }
        )
        .debugging(log_level="INFO")
    )

    if execution_mode == "retrain":
        algo = config.build()
        train_algorithm(algo, f"{data_path}/{RUN_CONFIG_NAME}")

    elif execution_mode == "train":
        algo = PPO.from_checkpoint(f"{data_path}/{RUN_CONFIG_NAME}")

        def fix_learner_optimizer(learner):
            import torch

            # Access optimizers in learner
            if hasattr(learner, '_optimizer_parameters'):
                for optimizer, params in learner._optimizer_parameters.items():
                    if hasattr(optimizer, 'param_groups'):
                        for group in optimizer.param_groups:
                            if 'betas' in group and torch.is_tensor(group['betas'][0]):
                                group['betas'] = (float(group['betas'][0]), float(group['betas'][1]))

        algo.learner_group.foreach_learner(fix_learner_optimizer)
        train_algorithm(algo, f"{data_path}/{RUN_CONFIG_NAME}")

    elif execution_mode == "evaluate":
        algo = PPO.from_checkpoint(f"{data_path}/{RUN_CONFIG_NAME}")
        algo.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        epilog="python ctf_env_v0.py --render_mode human --field_size 84 --model_path ./data --max_steps 1200"
    )

    parser.add_argument(
        '--render_mode',
        type=str,
        default='non_human',
        help='Set environment render mode (non-human, human)'
    )

    parser.add_argument(
        '--field_size',
        type=int,
        default=84,
        help='The size of the field to use (n, n)'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default='./data',
        help='Path to the saved model'
    )

    parser.add_argument(
        '--max_steps',
        type=int,
        default=1200,
        help='The maximum number of steps until the environment resets'
    )

    parser.add_argument(
        '--execution_mode',
        type=str,
        default='train',
        help='The execution mode (train, evaluate)'
    )

    args = parser.parse_args()

    init(args.render_mode, args.field_size, args.model_path, args.max_steps, args.execution_mode)
