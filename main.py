from ctf_env import CTFEnv

ctf_env = CTFEnv()
ctf_env.reset()

while True:
    ctf_env.step(None)