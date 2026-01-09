import pygame
from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np


# TODO implement flag carrying -> first version = instant capture

def get_team(agent):
    """Derive team name from agent name."""
    return agent.split("_")[0]


def get_enemy_team(agent):
    return "red" if get_team(agent) == "blue" else "blue"


def get_enemy_flag(agent):
    return "red_flag" if get_team(agent) == "blue_flag" else "blue_flag"


class CTFEnv(ParallelEnv):
    SCALE_FACTOR = 15

    def __init__(self, width=84, height=84, num_of_team_agents=2, render_mode="human", max_steps=1200):
        super().__init__()

        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.current_step = 0
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
            # Grid observation with different channels for different entity types:
            # (0.0 = empty cell, 0.5 = agent itself, 1.0 = entity present)
            #   Channel 0: red team agent positions
            #   Channel 1: blue team agent positions
            #   Channel 2: red flag location
            #   Channel 3: blue flag location
            #   Channel 4: obstacle location
            agent: spaces.Box(0, 1, (self.height, self.width, 5), np.float16)
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
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

    def reset(self, *, seed=None, options=None):
        print("RESET")
        self.current_step = 0
        self.flag_carrier = None
        self.blue_flag_status = self.red_flag_status = 0

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
        self.current_step += 1
        d = {}
        # 🎬 carry out actions
        for agent in self.agents:
            action = action_dict[agent]
            d[agent] = self._move(agent, action["move"])
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
            agent: self._calculate_reward(agent, d[agent])
            for agent in self.agents
        }

        # ⛔ terminate agents if flag has been captured or maximum number of steps have been reached
        terminations = {
            agent: self.blue_flag_status == 1 or self.red_flag_status == 1 or self.current_step >= self.max_steps
            for agent in self.agents
        }
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        if self.render_mode == "human":
            self.render()
        else:
            print(f"STEP {self.current_step}")

        # return observation dict, rewards dict, termination/truncation dicts, and infos dict
        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width * self.SCALE_FACTOR, self.height * self.SCALE_FACTOR))
            self.clock = pygame.time.Clock()

        # 🥶 clear the event queue to stop the game from freezing
        for _e in pygame.event.get():
            pass

        # set white background
        self.screen.fill((255, 255, 255))

        # draw agents
        self._draw_agents()

        # draw flags
        self._draw_flags()

        pygame.display.flip()
        self.clock.tick(20)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def _move(self, agent, action):
        """ Moves the agent and calculates the change in distance to the enemy flag.

        Returns
        -------
        d : distance delta
        """
        pos = self.agent_positions[agent].copy()
        if action == 1:  # up
            pos[1] = max(0, pos[1] - 1)
        elif action == 2:  # down
            pos[1] = min(self.height - 1, pos[1] + 1)
        elif action == 3:  # left
            pos[0] = max(0, pos[0] - 1)
        elif action == 4:  # right
            pos[0] = min(self.width - 1, pos[0] + 1)

        efp = self.flag_positions[get_enemy_flag(agent)]
        d = (np.linalg.norm(self.agent_positions[agent] - efp) - np.linalg.norm(pos - efp))

        self.agent_positions[agent] = pos
        return d

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

        if (np.array_equal(self.agent_positions[agent], self.flag_positions[f"{get_enemy_team(agent)}_flag"])
                and self.flag_carrier is None):
            print("CAPTURE")
            self.flag_carrier = agent
            if team == "red":
                self.blue_flag_status = 1
            else:
                self.red_flag_status = 1

    def _calculate_reward(self, agent, delta_distance):
        """Calculate the reward of the given agent."""
        team = get_team(agent)

        # [1] small time penalty
        reward = -0.01

        # [2] positive reward if an agent of the team picks up the flag
        if team == "red" and self.blue_flag_status == 1:
            reward += 10
        elif team == "blue" and self.red_flag_status == 1:
            reward += 10
        # [3] negative reward if the enemy team picks up the flag
        if team == "red" and self.red_flag_status == 1:
            reward -= 10
        elif team == "blue" and self.blue_flag_status == 1:
            reward -= 10

        # [4] positive reward for tagging an enemy TODO
        # [5] negative reward for being tagged TODO
        # [6] positive reward for moving toward the enemy flag
        if team == "red":
            reward += delta_distance * 0.1
        elif team == "blue":
            reward += delta_distance * 0.1
        # [7] negative reward for moving away from the own flag TODO maybe drop this

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

        return np.stack((red_agents, blue_agents, red_flag, blue_flag, obstacles), axis=-1)

    def _draw_agents(self):
        for i, agent in enumerate(self.agents):
            pos = self.agent_positions[agent]
            color = get_team(agent)
            pygame.draw.circle(self.screen, color, (pos[0] * self.SCALE_FACTOR, pos[1] * self.SCALE_FACTOR),
                               self.SCALE_FACTOR)

    def _draw_flags(self):
        for flag in self.flag_positions:
            pos = self.flag_positions[flag]
            color = get_team(flag)
            pygame.draw.circle(self.screen, color, (pos[0] * self.SCALE_FACTOR, pos[1] * self.SCALE_FACTOR),
                               self.SCALE_FACTOR + 2, 2)
