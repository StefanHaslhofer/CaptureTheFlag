from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms import PPOConfig, Algorithm
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune import register_env
from ray.tune import Tuner
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from env.ctf_env import CTFEnv
from env.ctf_env import get_team
import numpy as np
import warnings
import os
import argparse
import torch

warnings.filterwarnings("ignore", category=DeprecationWarning)

NUM_OF_ITERATIONS = 100
RUN_CONFIG_NAME = 'ppo_ctf_training'


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
    return best_result.checkpoint


def test_env(checkpoint, env_config):
    """Generate and render a test environment for checkpoint evaluation by human reviewers."""
    algo = Algorithm.from_checkpoint(checkpoint)
    env = env_creator(env_config)
    policies = {
        'red_policy': algo.env_runner.module["red_policy"],
        'blue_policy': algo.env_runner.module["blue_policy"]
    }

    obs, info = env.reset()

    episode_rewards = {agent: 0 for agent in env.agents}
    terminated = {agent: False for agent in env.agents}
    truncated = {agent: False for agent in env.agents}

    while not all(terminated.values()) and not all(truncated.values()):
        action_dict = {}
        for agent_id, agent_obs in obs.items():
            if not (terminated.get(agent_id, False) or truncated.get(agent_id, False)):
                policy = policies[f"{get_team(agent_id)}_policy"]

                obs_batch = {"obs": torch.from_numpy(np.array([agent_obs])).float()}
                module_output = policy.forward_inference(obs_batch)
                action = module_output["action_dist_inputs"][0]

                # RLlib concatenates logits for all action components
                action_dict[agent_id] = {
                    "move": np.argmax(action[:5]),
                    "tag": np.argmax(action[5:])
                }

        obs, rewards, terminated, truncated, info = env.step(action_dict)

        # Track rewards
        for agent_id, reward in rewards.items():
            episode_rewards[agent_id] += reward

    print(f"Episode reward: {episode_rewards}")
    env.close()


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
            sample_timeout_s=240
        )
        .training(train_batch_size=4000, entropy_coeff=0.01)
    )

    scheduler = ASHAScheduler(
        metric="env_runners/episode_return_mean",
        mode="max",
        max_t=100,
        grace_period=10,
        reduction_factor=3,
    )

    results = {}

    if execution_mode == "retrain":
        tuner = tune.Tuner(
            "PPO",
            param_space=config.to_dict(),
            run_config=tune.RunConfig(
                name=RUN_CONFIG_NAME,
                stop={
                    "training_iteration": 100,
                },
                checkpoint_config=tune.CheckpointConfig(
                    checkpoint_at_end=True,
                    checkpoint_frequency=5
                ),
                storage_path=data_path,
            ),
            tune_config=tune.TuneConfig(
                num_samples=1,
                scheduler=scheduler,
                max_concurrent_trials=2
            ),
        )

        results = tuner.fit()
        evaluate(results)

    elif execution_mode == "evaluate":
        tuner = Tuner.restore(f"{data_path}/{RUN_CONFIG_NAME}", trainable="PPO")
        results = tuner.get_results()

        best_checkpoint = evaluate(results)
        test_env(best_checkpoint, env_config)


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
