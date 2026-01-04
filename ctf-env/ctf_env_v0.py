from ray.rllib.algorithms import PPOConfig
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune import register_env
from env.ctf_env import CTFEnv
import ray
import warnings
import os
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS


warnings.filterwarnings("ignore", category=DeprecationWarning)

# TODO move to arguments
NUM_OF_ITERATIONS = 100
ENV_WIDTH = 84
ENV_HEIGHT = 84
DATA_PATH = os.path.abspath("./data")


def env_creator(_config):
    # create petting zoo ctf environment
    env = CTFEnv(width=ENV_WIDTH, height=ENV_HEIGHT, render_mode="non_human")

    return ParallelPettingZooEnv(env)

def init():
    ray.init()
    register_env("ctf_env", env_creator)

    config = (
        PPOConfig()
        .environment("ctf_env")
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .env_runners(num_env_runners=1, num_gpus_per_env_runner=0.5)
        #.training(train_batch_size=1024)
    )

    algo = config.build()

    for i in range(NUM_OF_ITERATIONS):
        results = algo.train()

        if i % 10 == 0:
            cp = algo.save(DATA_PATH)
            print(f"Checkpoint saved to {cp.checkpoint.path}")

        print(f"Iteration {i}: reward {0}")

    algo.stop()
    ray.shutdown()

if __name__ == "__main__":
    init()
