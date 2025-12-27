from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gymnasium.spaces as spaces
import numpy as np


class CTFEnv(MultiAgentEnv):

    def __init__(self, width=10, height=10, num_of_agents=2, config=None):
        super().__init__()

        self.width = width
        self.height = height

        # define all agent IDs
        self.agents = self.possible_agents = [f"agent_{i}" for i in range(num_of_agents)]
        # create observation spaces for all agents
        self.observation_spaces = {
            agent: spaces.Dict({
                # Grid observation with different channels for different entity types:
                # binary (0.0 = empty cell, 1.0 = entity present)
                #   Channel 0: red team agent positions
                #   Channel 1: blue team agent positions
                #   Channel 2: red flag location
                #   Channel 3: blue flag location
                #   Channel 4: obstacle location
                "grid": spaces.Box(0, 1, (width, height, 5), np.int32),
                # Flag status observation: 0 or 1 if respective flag has been picked up
                "red_flag_picked": spaces.Discrete(2),
                "blue_flag_picked": spaces.Discrete(2)
            })
            for agent in self.agents
        }
        # create action spaces for all agents
        self.action_spaces = {
            agent: spaces.Dict({
                # 5 possible movements: stay, up, down, left, right
                "move": spaces.Discrete(5),
                # 5 possible tag actions: none, tag up, tag down, tag left, tag right
                "tag": spaces.Discrete(5)
            })
            for agent in self.agents
        }
        None

    def reset(self, *, seed=None, options=None):
        # return observation dict and infos dict.
        None

    def step(self, action_dict):
        # return observation dict, rewards dict, termination/truncation dicts, and infos dict
        None


ctf_env = CTFEnv()
