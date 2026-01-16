from ray.rllib.algorithms import PPOConfig, Algorithm, PPO
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.env.multi_agent_env_runner import MultiAgentEnvRunner
from ray.tune import register_env
from ray.tune import Tuner
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from env.ctf_env import CTFEnv
from env.ctf_env import get_team
import warnings
import os
import argparse
from ray.tune.callback import Callback

warnings.filterwarnings("ignore", category=DeprecationWarning)

NUM_OF_ITERATIONS = 1000
RUN_CONFIG_NAME = 'ppo_ctf_training'


class RewardLoggerCallback(Callback):
    def on_trial_result(self, iteration, trials, trial, result, **info):
        """Called after each trial reports results."""
        mean_reward = result.get("env_runners/episode_return_mean", None)
        if mean_reward is not None:
            print(f"Trial {trial.trial_id} - Iteration {iteration}: Mean Reward = {mean_reward:.2f}")


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


def test_env(checkpoint, env_config):
    """Generate and render a test environment for checkpoint evaluation by human reviewers."""
    algo = Algorithm.from_checkpoint(checkpoint)

    config = algo.config.copy(copy_frozen=False)
    config.env_config = env_config
    config.env_runners(num_envs_per_env_runner=1)

    env_runner = MultiAgentEnvRunner(
        config=config,
    )

    env_runner.set_state(algo.get_state())

    samples = env_runner.sample(
        num_episodes=10,
        explore=False,
    )

    print(f"Episode returns: {samples.env_runner_results}")


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
            explore=True
        )
        .training(train_batch_size=4000, entropy_coeff=0.01, lr=[
            [0, 5e-4],
            [1000000, 3e-4],
            [3000000, 1e-4]
        ])
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
                    "training_iteration": NUM_OF_ITERATIONS,
                },
                checkpoint_config=tune.CheckpointConfig(
                    checkpoint_at_end=True,
                    checkpoint_frequency=5
                ),
                storage_path=data_path,
                callbacks=[RewardLoggerCallback()],
            ),
            tune_config=tune.TuneConfig(
                num_samples=1,
                scheduler=scheduler,
                max_concurrent_trials=2
            ),
        )

        results = tuner.fit()
        evaluate(results)

    elif execution_mode == "train":
        tuner = Tuner.restore(
            path=f"{data_path}/{RUN_CONFIG_NAME}",
            trainable="PPO",
            resume_errored=True,
            restart_errored=False,
            resume_unfinished=True,
            param_space=config.to_dict(),
        )

        # continue training
        results = tuner.fit()
        evaluate(results)

    elif execution_mode == "evaluate":
        tuner = Tuner.restore(f"{data_path}/{RUN_CONFIG_NAME}", trainable="PPO")
        results = tuner.get_results()

        best_result = evaluate(results)
        test_env(best_result.checkpoint, env_config)


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
