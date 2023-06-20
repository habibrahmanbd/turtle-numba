import numba as nb
import math
import numpy as np
from numba import int32, float32, float64, int64
from numba.experimental import jitclass
import constants
from skimage.draw import line
import cv2
from Vec2D import Vec2D

spec = [
    ('_angleOffset', nb.float64),
    ('_angleOrient', nb.int64),
    ('_mode', nb.int64),
    ('_penmode', nb.int64),
    ('_pen_color', nb.types.boolean),
    ('_canvas_width', int64),
    ('_canvas_height', int64),
    ('_canvas', nb.boolean[:,:]),
    ('_line_lengths', nb.types.ListType(float64)),
    ('_angles', nb.types.ListType(float64)),
    ('_position', Vec2D.class_type.instance_type),
    ('_orient', Vec2D.class_type.instance_type),
    ('_fullcircle', nb.float64),
    ('_degreesPerAU', nb.float64),
]

DEFAULT_MODE = 2
DEFAULT_ANGLEOFFSET = 0
DEFAULT_ANGLEORIENT = 1
DEFAULT_PEN_UP = 0
DEFAULT_PEN_DOWN = 1
DEFAULT_PEN_MODE = DEFAULT_PEN_DOWN

@jitclass(spec)
class TNavigator:
    def __init__(self, mode: int = 2, penmode: int = 1):
        self._angleOffset: float = DEFAULT_ANGLEOFFSET
        self._angleOrient: int = DEFAULT_ANGLEORIENT
        self._mode: int = mode
        self.degrees()
        self._penmode: int = penmode
        self._setmode(mode, penmode)
        self._pen_color: bool = False
        self._canvas_width: int = 128
        self._canvas_height: int = 128
        self._canvas: np.ndarray = np.full((self._canvas_width, self._canvas_width), True, dtype=np.bool_)
        self._line_lengths: np.ndarray = nb.typed.List.empty_list(nb.float64)
        self._angles: np.ndarray = nb.typed.List.empty_list(nb.float64)
        self.reset()
    
    def reset(self):
        self._position = Vec2D(constants.start_x, constants.start_y)
        self._orient =   Vec2D(0.0, 1.0) if self._mode == 2 else Vec2D(1.0, 0.0)
        self._penmode = DEFAULT_PEN_MODE
    
    def _setmode(self, mode=None, penmode=None):
        if mode is None:
            return self._mode
        if mode not in [0, 2, 1]:
            return
        self._mode = mode
        if mode in [0, 1]:
            self._angleOffset = 0
            self._angleOrient = 1
        else: # mode == 2:
            self._angleOffset = self._fullcircle/4.
            self._angleOrient = -1
        
        if penmode is None:
            return
        if penmode not in [DEFAULT_PEN_UP, DEFAULT_PEN_DOWN]:
            return
        self._penmode = penmode

    def _setDegreesPerAU(self, fullcircle):
        self._fullcircle = fullcircle
        self._degreesPerAU = 360/fullcircle
        if self._mode == 0:
            self._angleOffset = 0
        else:
            self._angleOffset = fullcircle/4.
    
    def degrees(self, fullcircle=360.0):
        self._setDegreesPerAU(fullcircle)

    def radians(self):
        self._setDegreesPerAU(2*math.pi)

    def _go(self, distance):
        ende = self._position + self._orient * distance
        ende = Vec2D(int(round(ende[0])), int(round(ende[1])))
        self._goto(ende)
    
    def _rotate(self, angle):
        angle *= self._degreesPerAU
        self._orient = self._orient.rotate(angle)
    
    def _goto(self, end):
        """move turtle to position end."""
        end_x = int(round(end._x))
        end_y = int(round(end._y))

        if self._penmode == DEFAULT_PEN_DOWN:
            
            start_point = Vec2D(max(0, min(self._canvas_width-1, self._position[0])), max(0, min(self._canvas_height-1, self._position[1])))
            end_point = Vec2D(max(0, min(self._canvas_width-1, end._x)), max(0, min(self._canvas_height-1, end._y)))
            
            points = self._get_line(start_point, end_point)
            try:
                points.remove(self._position)
            except:
                _ = 0
            if len(points) != 0:
                self._canvas[tuple(zip(*points))] = self._pen_color
            
                if 0 <= end_x < self._canvas_width and 0 <= end_y < self._canvas_height:
                    self._canvas[end_x, end_y] = self._pen_color

        self._position = Vec2D(end_x, end_y)
        
    def forward(self, distance, angle = 0):
        self._line_lengths.append(distance)
        self._angles.append(angle)
        self._go(distance)
        self._rotate(angle)
    
    def backward(self, distance):
        self._go(-distance)
    
    def right(self, angle):
        self._rotate(-angle)
    
    def left(self, angle):
        self._rotate(angle)
    
    def pos(self):
        return self._position
    
    def xcor(self):
        return self._position[0]
    
    def ycor(self):
        return self._position[1]
    
    def goto(self, x, y=None):
        if y is None:
            self._goto(Vec2D(int(round(*x))))
        else:
            self._goto(Vec2D(int(round(x)), int(round(y))))

    def move_goto(self, x, y=None):
        self._penmode = DEFAULT_PEN_UP
        if y is None:
            self._goto(Vec2D(int(round(*x))))
        else:
            self._goto(Vec2D(int(round(x)), int(round(y))))
        self._penmode = DEFAULT_PEN_DOWN

    def home(self):
        self._penmode = DEFAULT_PEN_UP
        self.move_goto(int(self._canvas_width/2.), int(self._canvas_height/2.))
        self._penmode = DEFAULT_PEN_DOWN
    
    def setx(self, x):
        self._penmode = DEFAULT_PEN_UP
        self._goto(Vec2D(int(round(x)), int(round(self._position[1]))))
        self._penmode = DEFAULT_PEN_DOWN
    
    def sety(self, y):
        self._penmode = DEFAULT_PEN_UP
        self._goto(Vec2D(int(round(self._position[0])), int(round(y))))
        self._penmode = DEFAULT_PEN_DOWN

    def distance(self, x, y=None):
        if y is not None:
            pos = Vec2D(x, y)
        if isinstance(x, Vec2D):
            pos = x
        elif isinstance(x, tuple):
            pos = Vec2D(*x)
        elif isinstance(x, TNavigator):
            pos = x._position
        return abs(pos - self._position)

    def towards(self, x, y=None):
        if y is not None:
            pos = Vec2D(x, y)
        if isinstance(x, Vec2D):
            pos = x
        elif isinstance(x, tuple):
            pos = Vec2D(*x)
        elif isinstance(x, TNavigator):
            pos = x._position
        x, y = pos - self._position
        result = round(math.atan2(y, x)*180.0/math.pi, 10) % 360.0
        result /= self._degreesPerAU
        return (self._angleOffset + self._angleOrient*result) % self._fullcircle

    def heading(self):
        x, y = self._orient
        result = round(math.atan2(y, x)*180.0/math.pi, 10) % 360.0
        result /= self._degreesPerAU
        return (self._angleOffset + self._angleOrient*result) % self._fullcircle
    
    def setheading(self, to_angle):
        angle = (to_angle - self.heading())*self._angleOrient
        full = self._fullcircle
        angle = (angle+full/2.)%full - full/2.
        self._rotate(angle)
     
    def penup(self):
        self._penmode = DEFAULT_PEN_UP
    
    def pendown(self):
        self._penmode = DEFAULT_PEN_DOWN
    
    def _get_line(self, cor1, cor2):
        return list(line(int(cor1._x), int(cor1._y), int(cor2._x), int(cor2._y)))
    
    def _get_image_cv2(self):
        uint_img = np.array(self._canvas, dtype = np.uint8)*255
        return uint_img
    
    def _save_image_cv2(self, filename):
        cv2.imwrite(filename=filename, img = self._get_image_cv2())

if __name__ == "__main__":

    def demo2():
        """Demo of some new features."""
        turtle = TNavigator()
        # turtle.reset()
        turtle.forward(20)
        turtle.left(120)
        # turtle.backward(20)
        # turtle.left(120)
        # turtle.backward(20)
        # turtle.circle(20, steps=10000)
        cv2.imwrite(filename="img/art_Numba.jpg", img = turtle._get_image_cv2())
    import timeit
    # call the demo2 for 10000 times and log the time
    print("Time Taken using Numba: ", timeit.timeit("demo2()", setup="from __main__ import demo2", number=10000))