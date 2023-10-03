import math
import cv2
import numpy as np
from skimage.draw import line
import constants
from Vec2D import Vec2D
from bresenham import bresenham

class TNavigator(object):
    """Navigation part of the RawTurtle.
    Implements methods for turtle movement.
    """
    START_ORIENTATION = {
        "standard": Vec2D(1.0, 0.0),
        "world"   : Vec2D(1.0, 0.0),
        "logo"    : Vec2D(0.0, 1.0)  }
    DEFAULT_MODE = "logo"
    DEFAULT_ANGLEOFFSET = 0
    DEFAULT_ANGLEORIENT = 1
    DEFAULT_PEN_UP = "up"
    DEFAULT_PEN_DOWN = "down"
    DEFAULT_PEN_MODE = DEFAULT_PEN_DOWN

    def __init__(self, mode=DEFAULT_MODE, penmode=DEFAULT_PEN_MODE):
        self._angleOffset = self.DEFAULT_ANGLEOFFSET
        self._angleOrient = self.DEFAULT_ANGLEORIENT
        self._mode = mode
        self.degrees()
        self._mode = None
        self._penmode = penmode
        self._setmode(mode, penmode)
        self._pen_color = False
        self._canvas_width = 128
        self._canvas_height = 128
        self._canvas = np.full((self._canvas_width,self._canvas_width), True)
        self._line_lengths = []
        self._angles = []
        TNavigator.reset(self)

    def reset(self):
        """reset turtle to its initial values

        Will be overwritten by parent class
        """
        # self._position = Vec2D(int(self._canvas_width/2.), int(self._canvas_height/2.))
        self._position = Vec2D(int(constants.start_x), int(constants.start_y))
        self._orient =  TNavigator.START_ORIENTATION[self._mode]
        self._penmode = TNavigator.DEFAULT_PEN_MODE

    def _setmode(self, mode=None, penmode=None):
        """Set turtle-mode to 'standard', 'world' or 'logo'.
           Set Penmode to 'Up' or 'Down'.
        """
        if mode is None:
            return self._mode
        if mode not in ["standard", "logo", "world"]:
            return
        self._mode = mode
        if mode in ["standard", "world"]:
            self._angleOffset = 0
            self._angleOrient = 1
        else: # mode == "logo":
            self._angleOffset = self._fullcircle/4.
            self._angleOrient = -1
        
        if penmode is None:
            return
        if penmode not in [TNavigator.DEFAULT_PEN_UP, TNavigator.DEFAULT_PEN_DOWN]:
            return
        self._penmode = penmode

    def _setDegreesPerAU(self, fullcircle):
        """Helper function for degrees() and radians()"""
        self._fullcircle = fullcircle
        self._degreesPerAU = 360/fullcircle
        if self._mode == "standard":
            self._angleOffset = 0
        else:
            self._angleOffset = fullcircle/4.

    def degrees(self, fullcircle=360.0):
        """ Set angle measurement units to degrees.

        Optional argument:
        fullcircle -  a number

        Set angle measurement units, i. e. set number
        of 'degrees' for a full circle. Dafault value is
        360 degrees.

        Example (for a Turtle instance named turtle):
        >>> turtle.left(90)
        >>> turtle.heading()
        90

        Change angle measurement unit to grad (also known as gon,
        grade, or gradian and equals 1/100-th of the right angle.)
        >>> turtle.degrees(400.0)
        >>> turtle.heading()
        100

        """
        self._setDegreesPerAU(fullcircle)

    def radians(self):
        """ Set the angle measurement units to radians.

        No arguments.

        Example (for a Turtle instance named turtle):
        >>> turtle.heading()
        90
        >>> turtle.radians()
        >>> turtle.heading()
        1.5707963267948966
        """
        self._setDegreesPerAU(2*np.pi)

    def _go(self, distance):
        """move turtle forward by specified distance"""
        # if distance < 0.0:
        #     distance += 1.0
        # elif distance > 0.0:
        #     distance -= 1.0
        ende = self._position + self._orient * distance
        ende = Vec2D(int(round(ende[0])), int(round(ende[1])))
        # print("Start: {}, End: {}, Distance: {}, Orient: {}".format(self._position, ende, distance, self._orient))
        self._goto(ende)
    
    def _get_end_point(self):
        """Return the end point of the turtle's line."""
        distance = max(self._canvas_width, self._canvas_height)
        ende = self._position + self._orient * distance
        ende = Vec2D(int(round(ende[0])), int(round(ende[1])))
        return ende
    
    def _rotate(self, angle):
        """Turn turtle counterclockwise by specified angle if angle > 0."""
        angle *= self._degreesPerAU
        self._orient = self._orient.rotate(angle)

    def _goto(self, end):
        """move turtle to position end."""
        
        rr, cc = [], []
        if self._penmode == TNavigator.DEFAULT_PEN_DOWN:
            # Getting the line for drawing in the 2D matrix (using skimage: pip install scikit-image)
            rr, cc = self._get_line(self._position, end)
            # filtering out the points that are outside the canvas
            mask = ((rr >= 0) & (rr < self._canvas_height) & (cc >= 0) & (cc < self._canvas_width))
            rr, cc = rr[mask], cc[mask]
            if len(rr):
                current_point_color = self._canvas[self._position]
                self._canvas[rr, cc] = self._pen_color
                self._canvas[self._position] = current_point_color

        self._position = end if (len(rr) >0 and len(cc) > 0) else Vec2D(rr[-1], cc[-1])

    def forward(self, distance, angle = 0):
        """Move the turtle forward by the specified distance.

        Aliases: forward | fd

        Argument:
        distance -- a number (integer or float)

        Move the turtle forward by the specified distance, in the direction
        the turtle is headed.

        Example (for a Turtle instance named turtle):
        >>> turtle.position()
        (0.00, 0.00)
        >>> turtle.forward(25)
        >>> turtle.position()
        (25.00,0.00)
        >>> turtle.forward(-75)
        >>> turtle.position()
        (-50.00,0.00)
        """
        self._line_lengths.append(distance)
        self._angles.append(angle)
        self._go(distance)
        self._rotate(angle)

    def backward(self, distance):
        """Move the turtle backward by distance.

        Aliases: back | backward | bk

        Argument:
        distance -- a number

        Move the turtle backward by distance ,opposite to the direction the
        turtle is headed. Do not change the turtle's heading.

        Example (for a Turtle instance named turtle):
        >>> turtle.position()
        (0.00, 0.00)
        >>> turtle.backward(30)
        >>> turtle.position()
        (-30.00, 0.00)
        """
        self._go(-distance)

    def right(self, angle):
        """Turn turtle right by angle units.

        Aliases: right | rt

        Argument:
        angle -- a number (integer or float)

        Turn turtle right by angle units. (Units are by default degrees,
        but can be set via the degrees() and radians() functions.)
        Angle orientation depends on mode. (See this.)

        Example (for a Turtle instance named turtle):
        >>> turtle.heading()
        22.0
        >>> turtle.right(45)
        >>> turtle.heading()
        337.0
        """
        self._rotate(-angle)

    def left(self, angle):
        """Turn turtle left by angle units.

        Aliases: left | lt

        Argument:
        angle -- a number (integer or float)

        Turn turtle left by angle units. (Units are by default degrees,
        but can be set via the degrees() and radians() functions.)
        Angle orientation depends on mode. (See this.)

        Example (for a Turtle instance named turtle):
        >>> turtle.heading()
        22.0
        >>> turtle.left(45)
        >>> turtle.heading()
        67.0
        """
        self._rotate(angle)

    def pos(self):
        """Return the turtle's current location (x,y), as a Vec2D-vector.

        Aliases: pos | position

        No arguments.

        Example (for a Turtle instance named turtle):
        >>> turtle.pos()
        (0.00, 240.00)
        """
        return self._position

    def xcor(self):
        """ Return the turtle's x coordinate.

        No arguments.

        Example (for a Turtle instance named turtle):
        >>> reset()
        >>> turtle.left(60)
        >>> turtle.forward(100)
        >>> print turtle.xcor()
        50.0
        """
        return self._position[0]

    def ycor(self):
        """ Return the turtle's y coordinate
        ---
        No arguments.

        Example (for a Turtle instance named turtle):
        >>> reset()
        >>> turtle.left(60)
        >>> turtle.forward(100)
        >>> print turtle.ycor()
        86.6025403784
        """
        return self._position[1]

    def goto(self, x, y=None):
        """Move turtle to an absolute position.

        Aliases: setpos | setposition | goto:

        Arguments:
        x -- a number      or     a pair/vector of numbers
        y -- a number             None

        call: goto(x, y)         # two coordinates
        --or: goto((x, y))       # a pair (tuple) of coordinates
        --or: goto(vec)          # e.g. as returned by pos()

        Move turtle to an absolute position. If the pen is down,
        a line will be drawn. The turtle's orientation does not change.

        Example (for a Turtle instance named turtle):
        >>> tp = turtle.pos()
        >>> tp
        (0.00, 0.00)
        >>> turtle.setpos(60,30)
        >>> turtle.pos()
        (60.00,30.00)
        >>> turtle.setpos((20,80))
        >>> turtle.pos()
        (20.00,80.00)
        >>> turtle.setpos(tp)
        >>> turtle.pos()
        (0.00,0.00)
        """
        # self._penmode = TNavigator.DEFAULT_PEN_UP
        if y is None:
            self._goto(Vec2D(int(round(*x))))
        else:
            self._goto(Vec2D(int(round(x)), int(round(y))))
        # self._penmode = TNavigator.DEFAULT_PEN_DOWN
    
    def move_goto(self, x, y=None):
        """Move turtle to an absolute position without drawing.

        Aliases: setpos | setposition | goto:

        Arguments:
        x -- a number      or     a pair/vector of numbers
        y -- a number             None

        call: goto(x, y)         # two coordinates
        --or: goto((x, y))       # a pair (tuple) of coordinates
        --or: goto(vec)          # e.g. as returned by pos()

        Move turtle to an absolute position. If the pen is down,
        a line will be drawn. The turtle's orientation does not change.

        Example (for a Turtle instance named turtle):
        >>> tp = turtle.pos()
        >>> tp
        (0.00, 0.00)
        >>> turtle.setpos(60,30)
        >>> turtle.pos()
        (60.00,30.00)
        >>> turtle.setpos((20,80))
        >>> turtle.pos()
        (20.00,80.00)
        >>> turtle.setpos(tp)
        >>> turtle.pos()
        (0.00,0.00)
        """
        self._penmode = TNavigator.DEFAULT_PEN_UP
        if y is None:
            self._goto(Vec2D(int(round(*x))))
        else:
            self._goto(Vec2D(int(round(x)), int(round(y))))
        self._penmode = TNavigator.DEFAULT_PEN_DOWN

    
    def home(self):
        """Move turtle to the origin, where turtle started drawing with current orientation.

        No arguments.

        Move turtle to the origin - coordinates (0,0) and set its
        heading to its start-orientation (which depends on mode).

        Example (for a Turtle instance named turtle):
        >>> turtle.home()
        """
        self._penmode = TNavigator.DEFAULT_PEN_UP
        self.move_goto(int(self._canvas_width/2.), int(self._canvas_height/2.))
        self._penmode = TNavigator.DEFAULT_PEN_DOWN
        # self.setheading(0)

    def setx(self, x):
        """Set the turtle's first coordinate to x

        Argument:
        x -- a number (integer or float)

        Set the turtle's first coordinate to x, leave second coordinate
        unchanged.

        Example (for a Turtle instance named turtle):
        >>> turtle.position()
        (0.00, 240.00)
        >>> turtle.setx(10)
        >>> turtle.position()
        (10.00, 240.00)
        """
        self._penmode = TNavigator.DEFAULT_PEN_UP
        self._goto(Vec2D(int(round(x)), int(round(self._position[1]))))
        self._penmode = TNavigator.DEFAULT_PEN_DOWN

    def sety(self, y):
        """Set the turtle's second coordinate to y

        Argument:
        y -- a number (integer or float)

        Set the turtle's first coordinate to x, second coordinate remains
        unchanged.

        Example (for a Turtle instance named turtle):
        >>> turtle.position()
        (0.00, 40.00)
        >>> turtle.sety(-10)
        >>> turtle.position()
        (0.00, -10.00)
        """
        self._penmode = TNavigator.DEFAULT_PEN_UP
        self._goto(Vec2D(int(round(self._position[0])), int(round(y))))
        self._penmode = TNavigator.DEFAULT_PEN_DOWN

    def distance(self, x, y=None):
        """Return the distance from the turtle to (x,y) in turtle step units.

        Arguments:
        x -- a number   or  a pair/vector of numbers   or   a turtle instance
        y -- a number       None                            None

        call: distance(x, y)         # two coordinates
        --or: distance((x, y))       # a pair (tuple) of coordinates
        --or: distance(vec)          # e.g. as returned by pos()
        --or: distance(mypen)        # where mypen is another turtle

        Example (for a Turtle instance named turtle):
        >>> turtle.pos()
        (0.00, 0.00)
        >>> turtle.distance(30,40)
        50.0
        >>> pen = Turtle()
        >>> pen.forward(77)
        >>> turtle.distance(pen)
        77.0
        """
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
        """Return the angle of the line from the turtle's position to (x, y).

        Arguments:
        x -- a number   or  a pair/vector of numbers   or   a turtle instance
        y -- a number       None                            None

        call: distance(x, y)         # two coordinates
        --or: distance((x, y))       # a pair (tuple) of coordinates
        --or: distance(vec)          # e.g. as returned by pos()
        --or: distance(mypen)        # where mypen is another turtle

        Return the angle, between the line from turtle-position to position
        specified by x, y and the turtle's start orientation. (Depends on
        modes - "standard" or "logo")

        Example (for a Turtle instance named turtle):
        >>> turtle.pos()
        (10.00, 10.00)
        >>> turtle.towards(0,0)
        225.0
        """
        if y is not None:
            pos = Vec2D(x, y)
        if isinstance(x, Vec2D):
            pos = x
        elif isinstance(x, tuple):
            pos = Vec2D(*x)
        elif isinstance(x, TNavigator):
            pos = x._position
        x, y = pos - self._position
        result = round(np.arctan2(y, x)*180.0/np.pi, 10) % 360.0
        result /= self._degreesPerAU
        return (self._angleOffset + self._angleOrient*result) % self._fullcircle

    def heading(self):
        """ Return the turtle's current heading.

        No arguments.

        Example (for a Turtle instance named turtle):
        >>> turtle.left(67)
        >>> turtle.heading()
        67.0
        """
        x, y = self._orient
        result = round(np.arctan2(y, x)*180.0/np.pi, 10) % 360.0
        result /= self._degreesPerAU
        return (self._angleOffset + self._angleOrient*result) % self._fullcircle

    def setheading(self, to_angle):
        """Set the orientation of the turtle to to_angle.

        Aliases:  setheading | seth

        Argument:
        to_angle -- a number (integer or float)

        Set the orientation of the turtle to to_angle.
        Here are some common directions in degrees:

         standard - mode:          logo-mode:
        -------------------|--------------------
           0 - east                0 - north
          90 - north              90 - east
         180 - west              180 - south
         270 - south             270 - west

        Example (for a Turtle instance named turtle):
        >>> turtle.setheading(90)
        >>> turtle.heading()
        90
        """
        angle = (to_angle - self.heading())*self._angleOrient
        full = self._fullcircle
        angle = (angle+full/2.)%full - full/2.
        self._rotate(angle)

    def circle(self, radius, extent = None, steps = None):
        """ Draw a circle with given radius.

        Arguments:
        radius -- a number
        extent (optional) -- a number
        steps (optional) -- an integer

        Draw a circle with given radius. The center is radius units left
        of the turtle; extent - an angle - determines which part of the
        circle is drawn. If extent is not given, draw the entire circle.
        If extent is not a full circle, one endpoint of the arc is the
        current pen position. Draw the arc in counterclockwise direction
        if radius is positive, otherwise in clockwise direction. Finally
        the direction of the turtle is changed by the amount of extent.

        As the circle is approximated by an inscribed regular polygon,
        steps determines the number of steps to use. If not given,
        it will be calculated automatically. Maybe used to draw regular
        polygons.

        call: circle(radius)                  # full circle
        --or: circle(radius, extent)          # arc
        --or: circle(radius, extent, steps)
        --or: circle(radius, steps=6)         # 6-sided polygon

        Example (for a Turtle instance named turtle):
        >>> turtle.circle(50)
        >>> turtle.circle(120, 180)  # semicircle
        """
        if extent is None:
            extent = self._fullcircle
        if steps is None:
            frac = abs(extent)/self._fullcircle
            steps = 1+int(min(11+abs(radius)/6.0, 59.0)*frac)
        w = 1.0 * extent / steps
        w2 = 0.5 * w
        l = 2.0 * radius * math.sin(w2*np.pi/180.0*self._degreesPerAU)
        if radius < 0:
            l, w, w2 = -l, -w, -w2
        self._rotate(w2)
        for i in range(steps):
            self._go(l)
            self._rotate(w)
        self._rotate(-w2)
        
    def penup(self):
        """Pull the pen up -- no drawing when the turtle moves.

        No argument.

        Pull the pen up, which means that the turtle does not draw
        when it moves. The default is down (not up).

        Example (for a Turtle instance named turtle):
        >>> turtle.penup()
        """
        self._penmode = TNavigator.DEFAULT_PEN_UP
    
    def pendown(self):
        """Push the pen down -- drawing when the turtle moves.

        No argument.

        Push the pen down, which means that the turtle draws
        when it moves. The default is down (not up).

        Example (for a Turtle instance named turtle):
        >>> turtle.pendown()
        """
        self._penmode = TNavigator.DEFAULT_PEN_DOWN
    
    def _get_line(self, cor1, cor2):
        """Return a line between two coordinates using bresenham algorithm."""
        return list(line(int(cor1[0]), int(cor1[1]), int(cor2[0]), int(cor2[1])))
    
    def _get_line_from_current_to_end(self):
        """Return a line between current position and end point in the heading direction using bresenham algorithm."""
        cor = self._get_end_point()
        rr, cc = self._get_line(self._position, cor)
        
        # If not, convert it using points = np.array(points)
        points = np.array([rr, cc]).T

        # Filter the points based on the condition
        mask = ((points[:, 0] >= 0) & (points[:, 0] < self._canvas.shape[0]) &
                (points[:, 1] >= 0) & (points[:, 1] < self._canvas.shape[1]))

        # Apply the mask to the points array
        points = points[mask]
        return points
    
    def __get_line(self, cor1, cor2):
        """Return a line between two coordinates using bresenham algorithm."""
        return list(bresenham(int(cor1[0]), int(cor1[1]), int(cor2[0]), int(cor2[1])))
    
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
        turtle.forward(20)
        turtle.left(120)
        turtle.forward(20)
        # print("Turtle Canvas: {}".format(turtle._canvas))
        # turtle.circle(20, steps=10000)
        # print("Current Position: {} Heading: {}".format(turtle._position, turtle.heading()))
        # points = turtle._get_line_from_current_to_end()
        # print("Line from current position to end point: {}".format(points))
        # img = turtle._get_image_cv2()
        # cv2.imwrite(filename="img/art_NonNumba.jpg", img = img)
        
        # """points = [[ 64  64]
        #         [ 65  63]
        #         [ 66  63]
        #         [ 67  62]
        #         [ 68  62]
        #         [ 69  61]]
        # """
        # # mark the points on the image as 0
        # img[points[:, 0], points[:, 1]] = 0
        
        # cv2.imwrite(filename="img/art_NonNumba_extension.jpg", img = img)
        
        # img = np.ones((128, 128), dtype = np.uint8)*255
        # # generate strainght line points from (64,64) to (64, 130)
        # rr, cc = line(64, 64, 0, 0)
        
        # points = rr, cc
        
        # # filter out the points which are outside the canvas
        # mask = ((points[0] >= 0) & (points[0] < img.shape[0]) & (points[1] >= 0) & (points[1] < img.shape[1]))
        # points = points[0][mask], points[1][mask]
        # img[points[0], points[1]] = 0
        
        # cv2.imwrite(filename="img/art_NonNumba_extension2.jpg", img = img)
        
        
    import timeit

    # call the demo2 for 1000000 times and log the time
    print("Time taken in Non-numba (in seconds): {}".format(timeit.timeit("demo2()", setup="from __main__ import demo2", number=100000)))