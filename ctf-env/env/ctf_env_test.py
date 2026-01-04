from pettingzoo.test import parallel_api_test
from ctf_env import CTFEnv

if __name__ == "__main__":
    env = CTFEnv()
    parallel_api_test(env, num_cycles=1_000_000)