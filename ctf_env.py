import pygame
from pettingzoo import ParallelEnv
import gymnasium.spaces as spaces
import numpy as np


class CTFEnv(ParallelEnv):

    def __init__(self, width=1000, height=1000, num_of_agents=2, config=None):
        super().__init__()

        self.width = width
        self.height = height
        # initialize empty object position dicts
        self.agent_positions = {}
        self.flag_positions = {}

        # rendering setup
        self.render_mode = "human"
        self.screen = None
        self.clock = None

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
        # TODO maybe introduce direction, otherwise agents would always tag one another
        self.action_spaces = {
            agent: spaces.Dict({
                # 5 possible movements: stay, up, down, left, right
                "move": spaces.Discrete(5),
                # 5 possible tag actions: none, tag up, tag down, tag left, tag right
                "tag": spaces.Discrete(5)
            })
            for agent in self.agents
        }

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # set initial flag positions
        self.flag_positions = {
            "flag_red": np.array([np.random.randint(0, high=self.width), np.random.randint(0, high=self.height)]),
            "flag_blue": np.array([np.random.randint(0, high=self.width), np.random.randint(0, high=self.height)])
        }

        # set initial agent positions
        self.agent_positions = {
            agent: np.array([np.random.randint(0, high=self.width), np.random.randint(0, high=self.height)])
            for agent in self.agents
        }

        # TODO set obstacle positions

        # set rewards to 0

        # set initial agent observations --> TODO move to function
        observations = {
            agent: self.observation_space(agent)
            for agent in self.agents
        }

        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, action_dict):
        self.render()

        # return observation dict, rewards dict, termination/truncation dicts, and infos dict
        None

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

        # set white background
        self.screen.fill((255, 255, 255))

        # draw agents
        for i, agent in enumerate(self.agents):
            pos = self.agent_positions[agent]
            pygame.draw.circle(self.screen, "red", (pos[0], pos[1]), 40)

        # draw flags
        # TODO

        pygame.display.flip()
        self.clock.tick(10)