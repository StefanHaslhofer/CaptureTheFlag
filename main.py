from ctf_env import CTFEnv

ctf_env = CTFEnv()
ctf_env.reset()

while True:
    # TODO random movement + no tagging
    actions = {
        agent: ctf_env.action_spaces(agent).sample()
        for agent in ctf_env.agents
    }
    observations, rewards, terminations, truncations, infos = ctf_env.step(actions)
