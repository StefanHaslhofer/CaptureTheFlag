import pygame
from pettingzoo import ParallelEnv
import gymnasium.spaces as spaces
import numpy as np

# TODO implement flag carrying -> first version = instant capture

def get_team(agent):
    """Derive team name from agent name."""
    return agent.split("_")[0]

def get_enemy_team(agent):
    return "red" if get_team(agent) == "blue" else "blue"


class CTFEnv(ParallelEnv):
    ENTITY_SIZE = 15

    def __init__(self, width=2000, height=1000, num_of_team_agents=2, config=None):
        super().__init__()

        self.width = width
        self.height = height
        # initialize empty object position dicts
        self.agent_positions = None
        self.flag_positions = None
        # TODO create Flag class that hold information about flag status and carrier
        self.red_flag_status = 0
        self.blue_flag_status = 0
        self.flag_carrier = None

        # define all agent IDs
        self.agents = self.possible_agents = (
                [f"red_{i}" for i in range(num_of_team_agents)] +
                [f"blue_{i}" for i in range(num_of_team_agents)]
        )
        # create observation spaces for all agents
        self.observation_spaces = {
            # TODO maybe add flag carrier to observation?
            agent: spaces.Dict({
                # Grid observation with different channels for different entity types:
                # (0.0 = empty cell, 0.5 = agent itself, 1.0 = entity present)
                #   Channel 0: red team agent positions
                #   Channel 1: blue team agent positions
                #   Channel 2: red flag location
                #   Channel 3: blue flag location
                #   Channel 4: obstacle location
                "grid": spaces.Box(0, 1, (self.height, self.width, 5), np.float32),
                # Flag status observation: 0 = normal, 1 = picked up, 2 = captured
                "red_flag_status": spaces.Discrete(3),
                "blue_flag_status": spaces.Discrete(3)
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

        # rendering setup
        self.render_mode = "human"
        self.screen = None
        self.clock = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # Set initial flag positions.
        # Randomly place red_flag on the left and blue_flag on the right.
        self.flag_positions = {
            "red_flag": np.array([np.random.randint(0, high=self.width // 6), np.random.randint(0, high=self.height)]),
            "blue_flag": np.array(
                [np.random.randint(self.width // 6 * 5, high=self.width), np.random.randint(0, high=self.height)])
        }

        # set initial agent positions
        self.agent_positions = {
                                   agent: np.array(
                                       [np.random.randint(0, high=self.width // 6),
                                        np.random.randint(0, high=self.height)])
                                   for agent in self.agents if get_team(agent) == "red"
                               } | {
                                   agent: np.array(
                                       [np.random.randint(self.width // 6 * 5, high=self.width),
                                        np.random.randint(0, high=self.height)])
                                   for agent in self.agents if get_team(agent) == "blue"
                               }

        # TODO set obstacle positions

        # set initial agent observations
        observations = {
            agent: self._get_obs(agent)
            for agent in self.agents
        }

        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, action_dict):
        # 🎬 carry out actions
        for agent in self.agents:
            action = action_dict[agent]
            self._move(agent, action["move"])
            self._tag(agent, action["tag"])

        # 🚩 check if an agent picked up or captured the enemy flag
        for agent in self.agents:
            self._update_flag_status(agent)

        # 🔬 get new observations after movements
        observations = {
            agent: self._get_obs(agent)
            for agent in self.agents
        }

        # 🏅 calculate rewards for each agent
        rewards = {
            agent: self._calculate_reward(agent)
            for agent in self.agents
        }

        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        self.render()

        # return observation dict, rewards dict, termination/truncation dicts, and infos dict
        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

        # set white background
        self.screen.fill((255, 255, 255))

        # draw agents
        self._draw_agents()

        # draw flags
        self._draw_flags()

        pygame.display.flip()
        self.clock.tick(10)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def _move(self, agent, action):
        pos = self.agent_positions[agent].copy()
        if action == 1: # up
            pos[1] = max(0, pos[1] - 1)
        elif action == 2: # down
            pos[1] = min(self.height, pos[1] + 1)
        elif action == 3: # left
            pos[0] = max(0, pos[0] - 1)
        elif action == 4: # right
            pos[0] = min(self.width, pos[0] + 1)

        self.agent_positions[agent] = pos

    def _tag(self, agent, action):
        # TODO
        None

    def _update_flag_status(self, agent):
        """Check the status of the enemy flag based on the positions of the agent and flags.

        The flag is considered picked up if the agent's current position matches the enemy flag's position
        and the flag has not already been picked up.

        The flag is considered captured if the enemy flag's position matches the position of the own flag.
        """
        team = get_team(agent)
        self.blue_flag_status = self.red_flag_status = 0

        if (np.array_equal(self.agent_positions[agent], self.flag_positions[f"{get_enemy_team(agent)}_flag"])
                and self.flag_carrier is None):
            self.flag_carrier = agent
            if team == "red":
                self.blue_flag_status = 1
            else:
                self.red_flag_status = 1

    def _calculate_reward(self, agent):
        """Calculate the reward of the given agent."""
        reward = 0
        team = get_team(agent)

        # [1] positive reward if an agent of the team picks up the flag
        if team == "red" and self.blue_flag_status == 1:
            reward += 10
        elif team == "blue" and self.red_flag_status == 1:
            reward += 10
        # [2] negative reward if the enemy team picks up the flag
        if team == "red" and self.red_flag_status == 1:
            reward -= 10
        elif team == "blue" and self.blue_flag_status == 1:
            reward -= 10

        # [3] positive reward for tagging an enemy TODO
        # [4] negative reward for being tagged TODO
        # [5] positive reward for moving toward the enemy flag
        if team == "red":
            reward += 1 / (np.linalg.norm(self.agent_positions[agent] - self.flag_positions["blue_flag"]) + 1)
        elif team == "blue":
            reward += 1 / (np.linalg.norm(self.agent_positions[agent] - self.flag_positions["red_flag"]) + 1)
        # [6] negative reward for moving away from the own flag TODO maybe drop this
        # TODO maybe add small time penalty?
        return reward

    def _get_obs(self, agent):
        """Get observations for an agent."""
        # initialize grid observations
        red_agents = blue_agents = red_flag = blue_flag = obstacles = np.zeros((self.height, self.width),
                                                                               dtype=np.float32)

        for a, pos in self.agent_positions.items():
            # 🔴 check for red team agents
            if a.startswith("red"):
                red_agents[pos[1], pos[0]] = 1.0 if a != agent else 0.5
            # 🔵 check for blue team agents
            if a.startswith("blue"):
                blue_agents[pos[1], pos[0]] = 1.0 if a != agent else 0.5

        red_flag_pos = self.flag_positions["red_flag"]
        blue_flag_pos = self.flag_positions["blue_flag"]

        # 🚩 check for red flag
        red_flag[red_flag_pos[1], red_flag_pos[0]] = 1.0
        # 🔷 check for blue flag
        blue_flag[blue_flag_pos[1], blue_flag_pos[0]] = 1.0

        # TODO implement flag pick up logic ->
        #   maybe a class wide dict that handles which agent has picked up the flag?
        #   maybe flag carrier part of observation?
        # TODO check for obstacle

        return {
            "grid": np.stack((red_agents, blue_agents, red_flag, blue_flag, obstacles)),
            "red_flag_status": self.red_flag_status,
            "blue_flag_status": self.blue_flag_status
        }

    def _draw_agents(self):
        for i, agent in enumerate(self.agents):
            pos = self.agent_positions[agent]
            color = get_team(agent)
            pygame.draw.circle(self.screen, color, (pos[0], pos[1]), self.ENTITY_SIZE)

    def _draw_flags(self):
        for flag in self.flag_positions:
            pos = self.flag_positions[flag]
            color = get_team(flag)
            pygame.draw.circle(self.screen, color, (pos[0], pos[1]), self.ENTITY_SIZE + 2, 2)
