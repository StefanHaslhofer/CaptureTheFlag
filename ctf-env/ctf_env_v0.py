from ray.rllib.algorithms import PPOConfig
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune import register_env
from env.ctf_env import CTFEnv
import ray
import warnings
import os
import argparse

warnings.filterwarnings("ignore", category=DeprecationWarning)

# TODO move to arguments
NUM_OF_ITERATIONS = 100
ENV_WIDTH = 84
ENV_HEIGHT = 84


def env_creator(config):
    # create petting zoo ctf environment
    env = CTFEnv(config)

    return ParallelPettingZooEnv(env)

def init(render_mode, field_size, model_path, max_steps):
    data_path = os.path.abspath(model_path)
    env_config = {
        "width": field_size,
        "height": field_size,
        "render_mode": render_mode,
        "num_of_team_agents": 2,
        "max_steps": max_steps
    }

    ray.init()
    register_env("ctf_env", env_creator)

    config = (
        PPOConfig()
        .environment(env="ctf_env", env_config=env_config)
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .env_runners(num_env_runners=1)
        .training(train_batch_size=1024)
    )

    algo = config.build()

    for i in range(NUM_OF_ITERATIONS):
        results = algo.train()

        if i % 5 == 0:
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
    args = parser.parse_args()

    init(args.render_mode, args.field_size, args.model_path, args.max_steps)
