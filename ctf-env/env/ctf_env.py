import pygame
from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


def get_team(agent):
    """Derive team name from agent name."""
    return agent.split("_")[0]


def get_enemy_team(agent):
    return "red" if get_team(agent) == "blue" else "blue"


def get_flag(agent):
    """Derive team flag from agent name."""
    return f"{get_team(agent)}_flag"


def get_enemy_flag(agent):
    return f"{get_enemy_team(agent)}_flag"


def print_heatmap(a):
    plt.imshow(a, cmap='hot', interpolation='nearest')
    plt.show()


def visualize_channel(obs, channel):
    grid = obs[:, :, channel]
    print(grid)

    return grid


class CTFEnv(ParallelEnv):
    SCALE_FACTOR = 15
    CAPTURE_RADIUS = 4
    TAG_RADIUS = 2
    RESPAWN_TIME = 60

    def __init__(self, width=84, height=84, num_of_team_agents=2, render_mode="human", max_steps=1200):
        super().__init__()

        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.current_step = 0
        self.reward_heatmap = np.zeros((height, width))
        # initialize empty object position dicts
        self.agent_positions = None
        self.flag_positions = None
        self.disabled_queue = {}
        self.scorer = None
        self.flag_carrier = None
        self.flag_states = {
            "red_flag": 0,
            "blue_flag": 0
        }

        # define all agent IDs
        self.agents = self.possible_agents = (
                [f"red_{i}" for i in range(num_of_team_agents)] +
                [f"blue_{i}" for i in range(num_of_team_agents)]
        )

        self.cumulative_rewards = {
            agent: 0
            for agent in self.agents
        }

        # create observation spaces for all agents
        self.observation_spaces = {
            # Grid observation with different channels for different entity types:
            #   Channel 0: red team agent positions (0.0 = empty cell, 1.0 = entity present, 5.0 = flag carrier)
            #   Channel 1: blue team agent positions (0.0 = empty cell, 1.0 = entity present, 5.0 = flag carrier)
            #   Channel 2: red flag location (0.0 = empty cell, 0.5 = capture area, 1.0 = entity present)
            #   Channel 3: blue flag location (0.0 = empty cell, 0.5 = capture area, 1.0 = entity present)
            #   Channel 4: self position (0.0 = empty cell, 1.0 = entity present)
            agent: spaces.Box(0, 1, (self.height, self.width, 5), np.float32)
            for agent in self.agents
        }
        # create action spaces for all agents
        self.action_spaces = {
            # 6 possible movements: stay, up, down, left, right, tag
            agent: spaces.Discrete(6)
            for agent in self.agents
        }

        # rendering setup
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

    def reset(self, *, seed=None, options=None):
        print("RESET")
        self.current_step = 0
        self.scorer = None
        self.flag_carrier = None
        self.flag_states = {
            "red_flag": 0,
            "blue_flag": 0
        }
        self.disabled_queue = {}
        self.reward_heatmap = np.zeros((self.height, self.width))
        if self.render_mode == "debug":
            print_heatmap(self.reward_heatmap)

        # Set initial flag positions.
        # Randomly place red_flag on the left and blue_flag on the right.
        self.flag_positions = {
            "red_flag": self._random_flag_position("red_flag"),
            "blue_flag": self._random_flag_position("blue_flag")
        }

        self.cumulative_rewards = {
            agent: 0
            for agent in self.agents
        }

        # set initial agent positions
        self.agent_positions = {
            agent: self._random_agent_position(agent)
            for agent in self.agents
        }

        # TODO set obstacle positions

        # set initial agent observations
        observations = {
            agent: self._get_obs(agent)
            for agent in self.agents
        }

        if self.render_mode == "debug":
            visualize_channel(observations['red_0'], 1)
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, action_dict):
        self.current_step += 1
        delta, d, t, flag_state_changed = {}, {}, {}, False
        # 🎬 carry out actions
        for agent in self.agents:
            # skip agent if it is disabled
            is_disabled = agent in self.disabled_queue
            action = action_dict[agent]
            # call move logic
            delta[agent], d[agent] = self._move(agent, action, is_disabled)
            # action index == 5 -> tag
            if action == 5:
                t[agent] = self._tag(agent, is_disabled)

        # 🚩 check if an agent picked up or captured the enemy flag
        for agent in self.agents:
            if self._update_flag_status(agent):
                flag_state_changed = True

        # 🪢 update flag position if it has been picked up
        for flag, state in self.flag_states.items():
            self._update_flag_positions(flag, state)

        # 🔬 get new observations after movements
        observations = {
            agent: self._get_obs(agent)
            for agent in self.agents
        }

        # 🏅 calculate rewards for each agent
        rewards = {
            agent: self._calculate_reward(agent, delta[agent], d[agent], t, flag_state_changed)
            for agent in self.agents
        }

        self.scorer = None

        # 🐣 re-enable agents after timer has run out
        self._enable_agents()

        # ⛔ terminate if maximum number of steps have been reached
        terminations = {
            agent: self.current_step >= self.max_steps
            for agent in self.agents
        }
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        if self.render_mode == "human":
            self.render()

        # 🖨️ print rewards if environment is about to be terminated
        if all(terminations.values()):
            print(self.cumulative_rewards)

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

        # draw flags
        self._draw_flags()

        # draw agents
        self._draw_agents()

        pygame.display.flip()
        self.clock.tick(30)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def _update_flag_positions(self, flag, state):
        if state == 1:
            if self.flag_carrier is not None:
                # set flag position to position of flag carrier if it has been picked up
                self.flag_positions[flag] = self.agent_positions[self.flag_carrier].copy()
        elif state == 2 or state == 3:
            # set flag state = 0 and reset flag to random starting positions if it has been captured or returned
            self.flag_positions[flag] = self._random_flag_position(flag)
            self.flag_states[flag] = 0

    def _move(self, agent, action, is_disabled):
        """ Moves the agent (if not disabled) and calculates the change in distance to the enemy flag.

        Returns
        -------
        delta : distance delta to the enemy flag
        d     : distance to the enemy flag after move
        """
        pos = self.agent_positions[agent].copy()
        if not is_disabled:
            if action == 1:  # up
                pos[1] = max(0, pos[1] - 1)
            elif action == 2:  # down
                pos[1] = min(self.height - 1, pos[1] + 1)
            elif action == 3:  # left
                pos[0] = max(0, pos[0] - 1)
            elif action == 4:  # right
                pos[0] = min(self.width - 1, pos[0] + 1)

        efp = self.flag_positions[get_enemy_flag(agent)]
        d = np.linalg.norm(pos - efp)
        delta = (np.linalg.norm(self.agent_positions[agent] - efp) - np.linalg.norm(pos - efp))

        self.agent_positions[agent] = pos
        return delta, d

    def _enable_agents(self):
        for agent, time in self.disabled_queue.copy().items():
            self.disabled_queue[agent] -= 1
            if time <= 0:
                self.disabled_queue.pop(agent)

    def _tag(self, agent, is_disabled):
        """
        Check if an agent successfully tagged an enemy.

        Returns
        -------
        The enemy agent if the tag was successful, otherwise the original agent itself.
        """
        for other in self.agents:
            # agent has to be in enemy team and both agents have to be enabled and within tagging distance
            if (get_team(agent) != get_team(other)
                    and not is_disabled
                    and other not in self.disabled_queue
                    and np.linalg.norm(
                        self.agent_positions[agent] - self.agent_positions[other]) < self.TAG_RADIUS):
                print(f"TAGGED {agent} -> {other} at STEP {self.current_step}")
                # reset agent if tagged
                self.agent_positions[other] = self._random_agent_position(other)
                # add agent to respawn queue
                self.disabled_queue = {
                    other: self.RESPAWN_TIME
                }
                if other == self.flag_carrier:
                    self.flag_carrier = None
                return other

        return agent

    def _update_flag_status(self, agent):
        """Check the status of the enemy flag based on the positions of the agent and flags.

        The flag is considered picked up if the agent's current position matches the enemy flag's position
        and the flag has not already been picked up.

        The flag is considered captured if the enemy flag's position is within the capture radius of the own flag.

        If the flag was picked up (status != 0) and the carrier was tagged (carrier is None),
        and the agent is within the flag’s capture radius, the flag is considered returned.

        Returns
        -------
        state_changed : bool
            Indicates whether the flag state has changed. This is used for reward
            calculation, as rewards are only computed when a flag-related event occurs.
        """
        state_changed = False
        team_flag = get_flag(agent)
        enemy_flag = get_enemy_flag(agent)

        # 3 = flag return
        if (self.flag_states[team_flag] == 1
                and self.flag_carrier is None
                and np.linalg.norm(
                    self.agent_positions[agent] - self.flag_positions[team_flag]) < self.CAPTURE_RADIUS):
            print(f"FLAG RETURNED by {agent} at STEP {self.current_step}")
            self.flag_states[team_flag] = 3
            state_changed = True

        # 2 = flag capture
        elif (np.linalg.norm(
                self.flag_positions[get_enemy_flag(agent)] - self.flag_positions[team_flag])
              < self.CAPTURE_RADIUS
              and self.flag_carrier == agent):
            print(f"CAPTURED by {agent} at STEP {self.current_step}")
            self.scorer = agent
            self.flag_carrier = None
            self.flag_states[enemy_flag] = 2
            state_changed = True

        # 1 = flag pickup
        elif (np.linalg.norm(
                self.agent_positions[agent] - self.flag_positions[get_enemy_flag(agent)]) < self.CAPTURE_RADIUS
              and self.flag_carrier is None):
            print(f"PICKED UP by {agent} at STEP {self.current_step}")
            self.scorer = agent
            self.flag_carrier = agent
            self.flag_states[enemy_flag] = 1
            state_changed = True

        return state_changed

    def _calculate_reward(self, agent, delta_distance, _dist, tags, flag_state_changed):
        """Calculate the reward of the given agent."""
        team_flag = get_flag(agent)
        enemy_flag = get_enemy_flag(agent)

        # [0] small time penalty
        reward = -0.001

        if flag_state_changed:
            # [1] positive team reward if an agent of the team captures the flag
            if self.flag_states[enemy_flag] == 2:
                # additional individual reward for flag capture
                if self.scorer == agent:
                    reward += 20
                reward += 10
            # [2] negative team reward if the enemy team captures the flag
            if self.flag_states[team_flag] == 2:
                reward -= 10

            # [3] positive team reward if an agent of the team picks up the flag
            if self.flag_states[enemy_flag] == 1:
                # additional individual reward for pickup
                if self.scorer == agent:
                    reward += 2
                reward += 5
            # [4] negative team reward if the enemy team picks up the flag
            if self.flag_states[team_flag] == 1:
                reward -= 5

            # [5] positive team reward if flag is returned
            if self.flag_states[team_flag] == 3:
                reward += 2
            # [6] negative team reward if the enemy team returns the flag
            if self.flag_states[enemy_flag] == 3:
                reward -= 2

        # [7] positive reward for tagging an enemy (bonus if enemy is the flag carrier)
        if agent in tags and tags[agent] != agent:
            reward += 2
        # [8] negative reward for being tagged (exclude agent key because an agent cannot tag itself)
        for val in list([v for k, v in tags.items() if k != agent]):
            if val == agent:
                reward -= 1

        # [7] penalty for not moving
        if delta_distance == 0:
            reward -= 0.005

        self.cumulative_rewards[agent] += reward
        return reward

    def _get_obs(self, agent):
        """Get observations for an agent."""
        # initialize grid observations
        red_agents, blue_agents, red_flag, blue_flag, self_pos = [
            np.zeros((self.height, self.width), dtype=np.float32)
            for _ in range(5)
        ]

        # 🤖 set self positioning
        sp = self.agent_positions[agent].copy()
        self_pos[sp[1], sp[0]] = 1.0

        for a, pos in self.agent_positions.items():
            # 🔴 check for red team agents
            if get_team(a) == "red":
                red_agents[pos[1], pos[0]] = 1.0 if a != self.flag_carrier else 5
            # 🔵 check for blue team agents
            if get_team(a) == "blue":
                blue_agents[pos[1], pos[0]] = 1.0 if a != self.flag_carrier else 5

        red_flag_pos = self.flag_positions["red_flag"].copy()
        blue_flag_pos = self.flag_positions["blue_flag"].copy()

        # 🚩 check for red flag
        red_flag[red_flag_pos[1], red_flag_pos[0]] = 1.0
        # 🔷 check for blue flag
        blue_flag[blue_flag_pos[1], blue_flag_pos[0]] = 1.0

        return np.stack((red_agents, blue_agents, red_flag, blue_flag, self_pos), axis=-1)

    def _draw_agents(self):
        for i, agent in enumerate(self.agents):
            pos = self.agent_positions[agent]
            color = get_team(agent)

            # set agent color to gray if disabled
            if agent in self.disabled_queue:
                color = "gray"

            pygame.draw.circle(self.screen, color, (pos[0] * self.SCALE_FACTOR, pos[1] * self.SCALE_FACTOR),
                               self.SCALE_FACTOR)
            # draw tag radius
            pygame.draw.circle(self.screen, 'gray', (pos[0] * self.SCALE_FACTOR, pos[1] * self.SCALE_FACTOR),
                               self.TAG_RADIUS * self.SCALE_FACTOR, 1)

    def _draw_flags(self):
        for flag in self.flag_positions:
            pos = self.flag_positions[flag]
            color = get_team(flag)
            pygame.draw.circle(self.screen, color, (pos[0] * self.SCALE_FACTOR, pos[1] * self.SCALE_FACTOR),
                               self.SCALE_FACTOR)
            # draw capture radius
            pygame.draw.circle(self.screen, color, (pos[0] * self.SCALE_FACTOR, pos[1] * self.SCALE_FACTOR),
                               self.CAPTURE_RADIUS * self.SCALE_FACTOR, 2)

    def _random_agent_position(self, agent):
        if get_team(agent) == "red":
            return np.array(
                [np.random.randint(0, high=self.width // 6), np.random.randint(0, high=self.height)]
            )

        if get_team(agent) == "blue":
            return np.array(
                [np.random.randint(self.width // 6 * 5, high=self.width), np.random.randint(0, high=self.height)]
            )

        return None

    def _random_flag_position(self, flag):
        if flag == "red_flag":
            return np.array([
                np.random.randint(2, high=self.width // 6),
                np.random.randint(self.height // 4, high=self.height // 4 * 3)
            ])

        if flag == "blue_flag":
            return np.array([
                np.random.randint(self.width // 6 * 5, high=self.width - 2),
                np.random.randint(self.height // 4, high=self.height // 4 * 3)
            ])

        return None
