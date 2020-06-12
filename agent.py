import random
from pygame.color import THECOLORS

import pymunk
from pymunk.vec2d import Vec2d


class Agent:
    def __init__(self, radius=10, mass=1, starting_pos=(0, 0)):
        self.radius = radius
        self.mass = mass
        self.rotational_inertia = pymunk.moment_for_circle(mass, 0, self.radius)
        self.body = pymunk.Body(self.mass, self.rotational_inertia)
        self.body.position = starting_pos
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.color = THECOLORS['white']
        self.shape.elasticity = 1.0  # perfectly elastic collisions
        self.body.angle = random.randrange(0, 4)
        driving_direction = Vec2d(1, 0).rotated(self.body.angle)
        self.body.apply_impulse_at_local_point(driving_direction)   # Starts agent motion

    def get_radius(self):
        return self.radius

    def get_pos(self):
        return self.body.position[0], self.body.position[1]

    def get_angle(self):
        return self.body.angle

    def get_color(self):
        return self.shape.color

    def set_pos(self, pos):
        self.body.position = pos

    def set_velocity(self, velocity):
        self.body.velocity = velocity

    def set_color(self, color):
        self.shape.color = color

    def set_angle(self, angle):
        self.body.angle = angle
