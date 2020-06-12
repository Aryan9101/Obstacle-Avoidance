import math
import torch
import pygame
import os
import time

from pymunk.pygame_util import DrawOptions as draw

from agent import *
from boulder import *


os.environ['SDL_VIDEODRIVER'] = 'dummy'

width = 1000
height = 600
pygame.init()
screen = pygame.display.set_mode((width, height))
screen.set_alpha(None)


class Environment:
    def __init__(self):
        self.crashed = False

        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)

        # Create the car.
        self.starting_pos = width/2, height/2
        self.agent = Agent(10, 0.5, self.starting_pos)
        self.agent.set_angle(-0.6 * math.pi)
        self.space.add(self.agent.body, self.agent.shape)

        # Create walls.
        self.wall_offset = 10  # for wall visibility
        offset = self.wall_offset
        self.boundaries = [self.create_wall((offset, offset), (offset, height - offset)),
                           self.create_wall((offset, height - offset), (width - offset, height - offset)),
                           self.create_wall((width - offset, height - offset), (width - offset, offset)),
                           self.create_wall((width - offset, offset), (offset, offset))]

        self.boulders = [Boulder((120, 250), 90),
                         Boulder((750, 390), 150),
                         Boulder((420, 165), 90),
                         Boulder((375, 465), 80),
                         Boulder((750, 100), 80),
                         Boulder((135, 465), 40)]

        for boulder in self.boulders:
            self.space.add(boulder.body, boulder.shape)

        self.num_sensor_per_array = 16
        self.sensor_gap = 8
        self.sensor_grid = list()

    def create_wall(self, endpoint1, endpoint2):
        wall = pymunk.Segment(self.space.static_body, endpoint1, endpoint2, 1)
        wall.friction = 1
        wall.collision = 1
        wall.color = THECOLORS['blue']
        self.space.add(wall)
        return wall

    def create_sensor_array(self, tilt, gap, length):  # include tilt because not every arm will go out straight
        x, y = self.agent.get_pos()
        r = self.agent.get_radius()
        array_points = []
        for i in range(1, length + 1):
            sensor = (x + r + (gap * i), y)
            array_points.append(self.rotate_about_point(self.agent.get_pos(), sensor, tilt))
        return array_points

    def create_sensor_grid(self):
        array_left_extreme = self.create_sensor_array(math.pi/2, self.sensor_gap, self.num_sensor_per_array)
        array_left_tilt = self.create_sensor_array(math.pi/4, self.sensor_gap, self.num_sensor_per_array)
        array_middle = self.create_sensor_array(0, self.sensor_gap, self.num_sensor_per_array)
        array_right_tilt = self.create_sensor_array(-math.pi/4, self.sensor_gap, self.num_sensor_per_array)
        array_right_extreme = self.create_sensor_array(-math.pi/2, self.sensor_gap, self.num_sensor_per_array)
        return [array_left_extreme, array_left_tilt, array_middle, array_right_tilt, array_right_extreme]

    def display_boundaries(self):
        for wall in self.boundaries:
            pygame.draw.aaline(screen, wall.color, wall.a, wall.b)

    def display_boulders(self):
        for boulder in self.boulders:
            display_x, display_y = boulder.get_pos()
            display_radius = int(boulder.get_radius())
            pygame.draw.circle(screen, THECOLORS['green'], (int(display_x), int(display_y)), display_radius)

    def display_agent(self):
        display_x, display_y = self.agent.get_pos()
        pygame.draw.circle(screen, self.agent.get_color(), (int(display_x), int(display_y)), self.agent.get_radius())

    def display_sensors(self):
        for array in self.sensor_grid:
            for sensor in array:
                pos_x, pos_y = self.rotate_about_point(self.agent.get_pos(), sensor, self.agent.get_angle())
                pygame.draw.circle(screen, (255, 255, 255), (pos_x, pos_y), 2)

    def display_environent(self):
        # time.sleep(0.02)  # Add a delay between each epoch to increase environment visibility or it will go vroom
        screen.fill(THECOLORS['black'])
        draw(screen)
        self.display_agent()
        self.display_sensors()
        self.display_boulders()
        self.display_boundaries()
        pygame.display.update()

    def update_state(self, angle, speed):
        self.agent.set_angle(self.agent.get_angle() + angle)
        driving_direction = Vec2d(speed, 0).rotated(self.agent.get_angle())
        self.agent.set_velocity(100 * driving_direction)
        self.space.step(0.1)

    def get_state(self, readings):
        # Apply mean normalization by setting range of sensor values from 0 to 1
        max_distance = self.sensor_gap * self.num_sensor_per_array
        state = torch.tensor([(reading[0])/max_distance for reading in readings])
        return state

    def get_reward(self, readings):
        # Add a distance bias so environment exploration is encouraged
        reward = sum([reading[0] for reading in readings]) + self.distance_from_start() / 10
        return reward

    def act(self, action):
        self.display_environent()

        """
        Turn roughly pi/15 degrees in either direction. A change less than 10 degrees seemed too slow but anything above
        15 degrees seemed too jittery to me so 12 degrees should be a nice compromise.

        Normally, a counterclockwise rotation would correspond to a positive angle (in radians) but the screen is
        flipped here
        """
        if action == 0:
            self.update_state(-math.pi/15, 1)
        elif action == 1:
            self.update_state(math.pi/15, 1)
        else:  # action == 2
            self.update_state(0, 1)

        # I have to recreate all of the sensors since their placement is based on the agent's position
        # Idk if I can optimize this. At the moment, this is not slowing anything down so it's not a high priority
        self.sensor_grid = self.create_sensor_grid()
        readings = self.get_sensor_readings()
        state = self.get_state(readings)

        self.agent.set_color(THECOLORS['yellow'])
        self.crashed = False

        reward = self.get_reward(readings)

        if any([readings[i][0] <= self.sensor_gap and readings[i][1] == -1 for i in range(5)]):
            # Crashed into a wall so flip around
            self.agent.set_angle(self.agent.get_angle() + math.pi/2)
            # To prevent the agent from being stuck in a corner, turn around and move a tiny bit
            driving_direction = Vec2d(1, 0).rotated(self.agent.get_angle())
            self.agent.set_velocity(100 * driving_direction)
            self.agent.set_pos(self.agent.get_pos() + 5 * driving_direction)
        elif any([readings[i][0] <= self.sensor_gap and readings[i][1] == 1 for i in range(5)]):
            # Crashed into a boulder
            reward = -100000
            self.agent.set_color(THECOLORS['red'])
            self.crashed = True

        # Weird edge case: sometimes the agent glitches out of the boundary. No clue why
        if self.agent.get_pos()[0] < self.wall_offset or self.agent.get_pos()[0] > width - self.wall_offset or \
                self.agent.get_pos()[1] < self.wall_offset or self.agent.get_pos()[1] > height - self.wall_offset:
            self.reset()

        print("State: ", state, "\t| Reward: ", reward, "\t| Action: ", action)
        return state, reward, self.crashed

    def get_sensor_readings(self):
        readings = []
        # Rotate them and get readings. First non-zero value is the true sensor reading.
        for i in range(len(self.sensor_grid)):
            sensor_count = 0
            for sensor in self.sensor_grid[i]:
                sensor_count += 1
                distance_to_sensor = sensor_count * self.sensor_gap
                sensor_pos = self.rotate_about_point(self.agent.get_pos(), sensor, self.agent.get_angle())
                if sensor_pos[0] <= self.wall_offset or sensor_pos[0] >= width - self.wall_offset or \
                        sensor_pos[1] <= self.wall_offset or sensor_pos[1] >= height - self.wall_offset:
                    readings.append([distance_to_sensor, -1])
                    break
                else:
                    if screen.get_at(sensor_pos) == THECOLORS['green']:
                        readings.append([distance_to_sensor, 1])
                        break
            if len(readings) == i:
                max_distance = self.num_sensor_per_array * self.sensor_gap
                readings.append([max_distance, 0])
        return readings

    def distance_from_start(self):
        diff_x = self.starting_pos[0] - self.agent.get_pos()[0]
        diff_y = self.starting_pos[1] - self.agent.get_pos()[1]
        return math.sqrt(diff_x**2 + diff_y**2)

    def rotate_about_point(self, pos1, pos2, radians):  # Linear Algebra REEEEEEEEEE
        x_1 = pos1[0]
        y_1 = pos1[1]
        x_2 = pos2[0]
        y_2 = pos2[1]
        rot_x = (x_2 - x_1) * math.cos(radians) - (y_2 - y_1) * math.sin(radians)
        rot_y = (x_2 - x_1) * math.sin(radians) + (y_2 - y_1) * math.cos(radians)
        new_x = x_1 + rot_x
        new_y = y_1 + rot_y
        return int(new_x), int(new_y)

    def reset(self):
        self.agent.set_pos(self.starting_pos)
        self.agent.set_angle(random.uniform(-1, 1) * math.pi)


if __name__ == "__main__":
    game_state = Environment()
    state, reward, is_crashed = game_state.act(-1)
    while True:
        state, reward, is_crashed = game_state.act(random.randint(0, 2))

