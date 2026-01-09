from ray.rllib.algorithms import PPOConfig
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune import register_env
from env.ctf_env import CTFEnv
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
        .resources(num_gpus=1, num_gpus_per_learner_worker=1)
        .environment(env="ctf_env", env_config=env_config)
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .env_runners(
            num_env_runners=1,
            sample_timeout_s=120
        )
        .training(train_batch_size=4000)
    )

    algo = config.build()
    # load checkpoint if execution_mode is set to evaluate
    if execution_mode != "retrain":
        algo.restore(data_path)

    for i in range(NUM_OF_ITERATIONS):
        results = algo.train()

        if i % 5 == 0 and execution_mode == "train":
            print(f"Saving data...")
            cp = algo.save(data_path)
            print(f"Checkpoint saved to {cp.checkpoint.path}")

        print(f"Iteration {i}: reward {0}")

    algo.stop()
    ray.shutdown()


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
