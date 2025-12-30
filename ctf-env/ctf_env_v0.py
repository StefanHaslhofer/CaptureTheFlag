from environment/ctf_env import CTFEnv

ctf_env = CTFEnv()
ctf_env.reset()

while True:
    actions = {
        agent: {
            "move": ctf_env.action_spaces[agent]["move"].sample(),
            "tag": ctf_env.action_spaces[agent]["tag"].sample(),
        }
        for agent in ctf_env.agents
    }
    observations, rewards, terminations, truncations, infos = ctf_env.step(actions)

ctf_env.close()