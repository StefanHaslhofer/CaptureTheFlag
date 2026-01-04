from ray.rllib.algorithms import PPOConfig
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune import register_env
import gymnasium.spaces as spaces
import numpy as np

from env.ctf_env import CTFEnv

# TODO move to arguments
NUM_OF_ITERATIONS = 100
ENV_WIDTH = 96
ENV_HEIGHT = 96


def env_creator(_config):
    # create petting zoo ctf environment
    env = CTFEnv(width=ENV_WIDTH, height=ENV_HEIGHT)
    return ParallelPettingZooEnv(env)


if __name__ == "__main__":
    register_env("ctf_env", env_creator)

    obs_space =  spaces.Box(0, 1, (ENV_HEIGHT, ENV_WIDTH, 5), np.float32)

    config = (
        PPOConfig()
        .environment("ctf_env")
        .multi_agent(
            policies={
                "shared_policy": (None, obs_space, None, {})
            }
        )
        .env_runners(num_env_runners=2)
    )

    algo = config.build()

    for i in range(NUM_OF_ITERATIONS):
        train_result = algo.train()
        None
