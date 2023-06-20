from numba.experimental import jitclass
from numba import float32
import math
import numpy as np

vec_spec = [
    ('x', float32),
    ('y', float32)
]

@jitclass(vec_spec)
class Vec2D(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        else:
            raise IndexError("Index out of range.")

    def __add__(self, other):
        return Vec2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2D(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        return Vec2D(self.x * other, self.y * other)

    def __abs__(self):
        return (self.x**2 + self.y**2) ** 0.5

    # def __repr__(self):
    #     return "(%.2f,%.2f)" % (self.x, self.y)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def rotate(self, angle):
        """rotate self counterclockwise by angle
        """
        perp = Vec2D(-self.y, self.x)
        angle = angle * math.pi / 180.0
        c, s = math.cos(angle), math.sin(angle)
        return Vec2D(self.x*c+perp.x*s, self.y*c+perp.y*s)

    
# test this class
if __name__ == '__main__':
    u = Vec2D(1, 2)
    v = Vec2D(3, 4)
    
    # test __add__
    print(u + v)
    assert u + v == Vec2D(4, 6)
    
    # # test __sub__
    # print(u - v)
    # assert u - v == Vec2D(-2, -2)
    
    # # test __mul__
    # print(u * 2)
    # assert u * 2 == Vec2D(2, 4)
    
    # # test __abs__
    # print(abs(u))
    # assert abs(u) == 2.23606797749979
    
    # # # test __repr__
    # # print(u.__repr__())
    # # assert u.__repr__() == "(1.00,2.00)"
    
    # # test __eq__
    # assert u == Vec2D(1, 2)
    
    # # test rotate
    # print(u.rotate(90))
    # assert u.rotate(90) == Vec2D(-2, 1)
