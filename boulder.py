from pygame.color import THECOLORS

import pymunk


class Boulder:
    def __init__(self, pos, radius):
        self.body = pymunk.Body(10000000, 10000000, body_type=pymunk.Body.STATIC)
        self.radius = radius
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.elasticity = 1.0
        self.body.position = pos
        self.shape.friction = 1
        self.shape.collision_type = 1
        self.shape.color = THECOLORS['green']

    def get_radius(self):
        return self.radius

    def get_pos(self):
        return self.body.position[0], self.body.position[1]
