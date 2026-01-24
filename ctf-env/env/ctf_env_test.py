from pettingzoo.test import parallel_api_test
from ctf_env import CTFEnv

if __name__ == "__main__":
    env = CTFEnv(width=18, height=18, num_of_team_agents=2, render_mode="human", max_steps=1200)
    parallel_api_test(env, num_cycles=1_000_000)