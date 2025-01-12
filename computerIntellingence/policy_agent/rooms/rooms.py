"""
rooms:
    - 2d grid world with obstacles
    - agents move 1 grid per step (north, east, south, west)
    - goal is reached if agent 0 (moving) finds agent 1 (fixed)
    - reward is granted only for reaching the goal
    - granted reward decreases linearly by 1/max_steps per step

authors:
    thomy.pha@ifi.lmu.de
    fabian.ritz@ifi.lmu.de
    sarah.gerner

version:
    2023
"""

import pathlib
from typing import Any

import numpy

import gymnasium
from gymnasium import spaces

import pygame
from pygame.locals import QUIT, K_ESCAPE

MOVE_NORTH = 0
MOVE_EAST = 1
MOVE_SOUTH = 2
MOVE_WEST = 3
ACTIONS = [MOVE_NORTH, MOVE_EAST, MOVE_SOUTH, MOVE_WEST]
OBSERVATION_LENGTH = 4  # [agent_0_x, agent_0_y, agent_1_x, agent_1_y]]
OBSERVATION_MIN = 0
OBSERVATION_MAX = 1
WINDOW_NAME = "Rooms"
FPS = 5


class RoomsEnv(gymnasium.Env):

    def __init__(self, layout, initial_positions=None, max_steps=100, seed=0, stochastic=False):
        self.seed = seed; self.rng = numpy.random.default_rng(self.seed)
        
        self.layout = layout  # environment
        self.height, self.width = len(layout), len(layout[0].strip().split())
        self.obstacles = [ [x,y] for y, line in enumerate(layout) 
          for x, cell in enumerate(line.strip().split()) if cell == '#'
        ]

        self.initial_positions = initial_positions
        self.current_step = 0
        self.max_steps = max_steps
        self.stochastic = stochastic
        self.score = 0
        self.agent_positions = []
        self.done = False
        # actions and observations
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(OBSERVATION_MIN, OBSERVATION_MAX, (OBSERVATION_LENGTH, ), numpy.float32)
        # visualization
        self.viewer = None
        self.close = False
        self.view_scale = 50
        self.reset()

    def get_copy_without_rendering(self):
        copy = RoomsEnv(self.layout, self.agent_positions.copy(), self.max_steps, self.seed, self.stochastic)
        copy.current_step = self.current_step
        copy.score = self.score
        return copy

    def reset(self, *, seed: int = None, options = None):
        if seed:
            self.rng = numpy.random.default_rng(seed=seed)
        self.done = False
        if self.initial_positions:
            self.agent_positions = self.initial_positions.copy()
        else:
            self.agent_positions = [self._get_random_agent_position(), self._get_random_agent_position()]
        self.current_step = 0
        self.score = 0
        return self._observation(), {}

    def step(self, action):
        if self.stochastic and self.rng.random() < 0.25:
            action = self.rng.choice(ACTIONS)
        return self._step_with_action(action)

    def render(self, mode='human', close=False) -> None:
        if close or self.close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        if self.viewer is None:
            window_height = self.height * self.view_scale
            window_width = self.width * self.view_scale
            self.viewer = View(window_width, window_height, self.view_scale, self.view_scale, WINDOW_NAME, FPS)
        self.close = self.viewer.draw(self.obstacles, self.agent_positions)

    def _get_random_agent_position(self):
        random_position = [self.rng.integers(low=1, high=self.width-2), self.rng.integers(low=1, high=self.height-2)]
        if random_position not in self.obstacles and random_position not in self.agent_positions:
            return random_position
        return self._get_random_agent_position()

    def _observation(self):
        observation = []
        for agent_position in self.agent_positions:
            observation.append(scale(agent_position[0], 0, self.width-1, OBSERVATION_MIN, OBSERVATION_MAX))
            observation.append(scale(agent_position[1], 0, self.height-1, OBSERVATION_MIN, OBSERVATION_MAX))
        return numpy.array(observation, dtype=numpy.float32)

    def _step_with_action(self, action):
        self.current_step += 1
        reward = 0
        # agent 0 may move, the goal (agent 1) is fixed
        agent_index = 0
        x, y = self.agent_positions[agent_index]
        if action == MOVE_NORTH and y + 1 < self.height:
            self._set_position_if_no_obstacle([x, y + 1], agent_index)
        elif action == MOVE_EAST and x + 1 < self.width:
            self._set_position_if_no_obstacle([x + 1, y], agent_index)
        elif action == MOVE_SOUTH and y - 1 >= 0:
            self._set_position_if_no_obstacle([x, y - 1], agent_index)
        elif action == MOVE_WEST and x - 1 >= 0:
            self._set_position_if_no_obstacle([x - 1, y], agent_index)
        # agent 0 needs to find agent 1
        goal_reached = self.agent_positions[0] == self.agent_positions[1]
        if goal_reached:
            # reward decreases linear w.r.t. the maximum number of steps
            reward = (self.max_steps - self.current_step) / self.max_steps
        self.score += reward
        self.done = goal_reached or self.current_step >= self.max_steps
        return self._observation(), reward, self.done, False, {}

    def _set_position_if_no_obstacle(self, new_position, agent_index):
        if new_position not in self.obstacles:
            self.agent_positions[agent_index] = new_position


class View:
    def __init__(self, width, height, scale_width, scale_height, caption, fps):
        self.width = width
        self.height = height
        self.scale_width = scale_width
        self.scale_height = scale_height
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.font = pygame.font.Font(None, 25)
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.agent_colors = [pygame.Color('Green'), pygame.Color('Red')]
        pygame.display.set_caption(caption)

    def draw(self, obstacles, agents):
        # draw background
        self.screen.fill(pygame.Color('Gray33'))
        # scaled size of elements
        size_x = numpy.ceil(self.scale_width)
        size_y = numpy.ceil(self.scale_height)
        # draw obstacles
        for obstacle in obstacles:
            center_x = obstacle[0] * self.scale_width
            center_y = obstacle[1] * self.scale_height
            pygame.draw.rect(self.screen, pygame.Color('Black'), (center_x, center_y, size_x, size_y))
        # draw agents
        for index, agent in enumerate(agents):
            a_center_x = int(agent[0] * self.scale_width)
            a_center_y = int(agent[1] * self.scale_height)
            pygame.draw.ellipse(self.screen, self.agent_colors[index], (a_center_x, a_center_y, size_x, size_y))
        # draw FPS
        fps = self.clock.get_fps()
        fps_string = self.font.render(str(int(fps)), True, pygame.Color('White'))
        self.screen.blit(fps_string, (1, 1))
        # show screen
        pygame.display.flip()
        self.clock.tick(self.fps)
        return self.check_for_interrupt()

    @staticmethod
    def check_for_interrupt():
        key_state = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == QUIT or key_state[K_ESCAPE]:
                return True
        return False

    @staticmethod
    def close():
        pygame.quit()


def scale(value, input_min, input_max, output_min, output_max):
    """" https://stackoverflow.com/questions/929103/convert-a-number-range-to-another-range-maintaining-ratio """
    input_range = input_max - input_min
    output_range = output_max - output_min
    return (((value - input_min) * output_range) / input_range) + output_min
