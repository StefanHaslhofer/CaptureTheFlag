from ray.rllib.algorithms import PPOConfig
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune import register_env
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from env.ctf_env import CTFEnv
from env.ctf_env import get_team
import ray
import warnings
import os
import argparse

warnings.filterwarnings("ignore", category=DeprecationWarning)

NUM_OF_ITERATIONS = 100


def env_creator(config):
    # create petting zoo ctf environment
    env = CTFEnv(config['width'], config['height'], config['num_of_team_agents'], config['render_mode'],
                 config['max_steps'])

    return ParallelPettingZooEnv(env)


def init(render_mode, field_size, model_path, max_steps, execution_mode):
    data_path = os.path.abspath(model_path)
    env_config = {
        "width": field_size,
        "height": field_size,
        "num_of_team_agents": 2,
        "render_mode": render_mode,
        "max_steps": max_steps
    }

    ray.init()
    register_env("ctf_env", env_creator)

    config = (
        PPOConfig()
        .resources(num_gpus=1, num_gpus_per_learner_worker=0.25)
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
            num_env_runners=4,
            sample_timeout_s=120
        )
        .training(train_batch_size=tune.choice([4000, 8000, 16000]), entropy_coeff=0.01)
    )

    scheduler = ASHAScheduler(
        metric="env_runners/episode_return_mean",
        mode="max",
        max_t=2000,
        grace_period=200,
        reduction_factor=3,
    )

    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=tune.RunConfig(
            name="ppo_ctf_training",
            stop={
                "training_iteration": 2000,
                "env_runners/episode_return_mean": 500,
            },
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_at_end=True,
                checkpoint_frequency=10,
                num_to_keep=3
            ),
            storage_path=data_path,
        ),
        tune_config=tune.TuneConfig(
            num_samples=10,
            scheduler=scheduler,
        ),
    )

    results = tuner.fit()

    # Get the best trial
    best_result = results.get_best_result(
        metric="env_runners/episode_return_mean",
        mode="max"
    )

    print(f"Best trial config: {best_result.config}")
    print(f"Best trial final return: {best_result.metrics['env_runners/episode_return_mean']}")
    print(f"Best checkpoint: {best_result.checkpoint}")


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
