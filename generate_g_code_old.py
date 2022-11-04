import numpy as np
import scipy as sp
from scipy import optimize as opt
import matplotlib.pyplot as plt
from typing import Callable
import sympy as smp

# function to calculate the length of a given vector
def length(v: np.ndarray):
    return np.sqrt(v[0] ** 2 + v[1] ** 2)


def arc_length(start_angle: float, end_angle: float, R: float):
    return R * (end_angle - start_angle)


# function to calculate the derivative vector of a given function
def nabla2(
    x: Callable[[float], float], y: Callable[[float], float], t: float
) -> np.ndarray:
    h = 1e-6
    return np.array([(x(t + h) - x(t - h)) / 2 / h, (y(t + h) - y(t - h)) / 2 / h])

 
# function to calculate the perpendicular vector of a given vector v
def perpendicular(v: np.ndarray):
    return np.array([-v[1], v[0]])



class Segment:
    n: int = 0
    start_angle = 0
    end_angle = 0
    start: np.ndarray = property(
        lambda self: np.array([self.u(self.start_angle), self.v(self.start_angle)])
    )
    end: np.ndarray = property(
        lambda self: np.array([self.u(self.end_angle), self.v(self.end_angle)])
    )

    def r(self, phi):
        return np.sqrt(self.u(phi) ** 2 + self.v(phi) ** 2)

    def r_path(self, phi):
        return np.sqrt(self.x(phi) ** 2 + self.y(phi) ** 2)

    def __init__(self, x: Callable[[float], float], y: Callable[[float], float]):
        self.x = x
        self.y = y
        self.n = 1


class Circle_Segment(Segment):
    end_angle = property(lambda self: self.start_angle + self.n * self.alpha_0)
    # take a float or array of floats and return the corresponding x coordinates of the circle

    def u(self, phi):
        return self.center[0] + self.R * np.cos(phi + self.relative_start_angle)

    def v(self, phi):
        return self.center[1] + self.R * np.sin(phi + self.relative_start_angle)

    def circle_center(
        self,
        x: Callable[[float], float],
        y: Callable[[float], float],
        angle: float,
        R: float,
    ):
        perpendicular_vector = perpendicular(nabla2(x, y, angle))
        perpendicular_vector /= length(perpendicular_vector)
        return np.array([x(angle), y(angle)]) + R * perpendicular_vector

    def calculate_start_angle(
        self, x: Callable[[float], float], y: Callable[[float], float], angle: float
    ):
        return np.arctan2(y(angle) - self.center[1], x(angle) - self.center[0])

    phi = property(lambda self: np.linspace(self.start_angle, self.end_angle, 100))
    def plot(self):
        plt.plot(self.u(self.phi), self.v(self.phi))
        # plot point at center
        plt.plot(self.center[0], self.center[1], "x")
        plt.draw()

    def __init__(
        self,
        x: Callable[[float], float],
        y: Callable[[float], float],
        angle: float,
        R: float,
        alpha_0: float,
    ):
        super().__init__(x, y)
        self.alpha_0 = alpha_0
        self.R = R
        self.center = self.circle_center(x, y, angle, R)
        self.start_angle = angle
        self.relative_start_angle = self.calculate_start_angle(x, y, angle)
        self.n_max = round(2 * np.pi / alpha_0)



    def find_intersect(self):

        self.n = 1
        start_sign = np.sign(self.get_delta_r(self.start_angle))  # is the start point closer or further from the center?
        if start_sign == 0:
            start_sign = np.sign(np.sign(self.get_delta_r(self.start_angle+1e-3)))  # if lines intersect at starting point, try a bit further.
        while (
            np.sign(self.get_delta_r(self.end_angle)) == start_sign
            and self.n < self.n_max
        ):
            self.n += 1
    
    def get_delta_r(self,phi):
        diff = self.r_path(phi) - self.r(phi)
        if abs(diff) < 1e-12: # rounding out error due to floating point error, we assume points are intersecting
            diff = 0
        return diff
        

    def find_delta_r_max(self):
        difference_vector = self.r(self.phi) - length(
            [self.x(self.phi), self.y(self.phi)]
        )
        return np.amax(abs(difference_vector))


class Straight_Segment(Segment):
    n: int = 0
    end_angle = property(lambda self: self.parameter_to_angle(self.n * self.l_0))
    start: np.ndarray = property(
        lambda self: self._start, lambda self, start: setattr(self, "_start", start)
    )
    end: np.ndarray = property(
        lambda self: np.array([self.u_cartesian(self.n * self.l_0), self.v_cartesian(self.n * self.l_0)])
    )
    t = property(lambda self: np.linspace(0, self.n*l_0, 100))

    def u_cartesian(self, t):
        return self.start[0] + self.direction[0] * t

    def v_cartesian(self, t):
        return self.start[1] + self.direction[1] * t

    def r_cartesian(self, t):
        return np.sqrt(self.u_cartesian(t) ** 2 + self.v_cartesian(t**2))

    def parameter_to_angle(self, t) -> float:
        return np.arctan(self.v_cartesian(t) / self.u_cartesian(t))

    def u(self, phi):
        x0 = self.start[0]
        y0 = self.start[1]
        x = self.direction[0]
        y = self.direction[1]
        r = (x * y0 - x0 * y) / (x * np.sin(phi) - y * np.cos(phi))
        r = np.nan_to_num(r)  # handle case where we start at 0
        return r * np.cos(phi)

    def v(self, phi):
        x0 = self.start[0]
        y0 = self.start[1]
        x = self.direction[0]
        y = self.direction[1]

        r = (x * y0 - x0 * y) / (x * np.sin(phi) - y * np.cos(phi))
        r = np.nan_to_num(r)  # handle case where we start at 0
        return r * np.sin(phi)

    def plot(self):
        plt.plot(self.u_cartesian(self.t), self.v_cartesian(self.t))
        # plot point at start
        plt.plot(self.start[0], self.start[1], "x")
        plt.draw()

    def find_intersect(self, l_max):
        n_max = l_max / self.l_0
        start_sign = np.sign(
            self.r_path(self.start_angle) - self.r_cartesian(0)
        )  # is the start point closer or further from the center?
        if start_sign < 1e-9: # we don't trust differences that small due to float point error, lines start at same point
            start_sign = np.sign(
                self.r_path(self.start_angle + 1e-3) - self.r_cartesian(1e-3)
            )  # if lines intersect at starting point, try a bit further.
        self.n = 0

        while (
            np.sign(self.r_path(self.start_angle) - self.r_cartesian(0)) == start_sign
            and self.n < n_max
        ):
            self.n += 1




    def find_delta_r_max(self):
        difference_vector = self.r_cartesian(self.t) - self.r_path(self.parameter_to_angle(self.t))
        return np.amax(np.abs(difference_vector))    

    def __init__(
        self,
        x: Callable[[float], float],
        y: Callable[[float], float],
        circle_segment: Circle_Segment,
        l_0: float,
    ):
        super().__init__(x, y)
        self.direction = nabla2(
            circle_segment.u, circle_segment.v, circle_segment.end_angle
        )
        self.direction /= length(self.direction)
        self.start = circle_segment.end
        self.l_0 = l_0


def generate_segment_pair(
    x: Callable[[float], float],
    y: Callable[[float], float],
    start_angle: float,
    end_angle: float,
    R: float,
    alpha_0: float,
    l_0: float,
) -> list[Segment]:

    circle_segment = Circle_Segment(x, y, start_angle, R, alpha_0)
    max_length = calculate_path_length(x, y, start_angle, end_angle)
    optimized_max_distance = np.inf
    optimized_n = None

    def max_distance(n_circle: int):
        circle_segment.n = n_circle
        line_segment = Straight_Segment(x, y, circle_segment, l_0)
        line_segment.find_intersect(max_length)
        circle_segment.plot()
        line_segment.plot()
        plt.draw()
        max_delta = max(
            circle_segment.find_delta_r_max(), line_segment.find_delta_r_max()
        )
        return max_delta

    circle_segment.find_intersect()  # theoretically, the worst solution should be to let the circle intersect and then find the second intersection of the line. We use this as a starting point
    n_range = range(0,circle_segment.n)
    for n in n_range:
        distance = max_distance(n)
        if optimized_max_distance > distance:
            optimized_max_distance = distance
            optimized_n = n

    circle_segment.n = optimized_n

    line_segment = Straight_Segment(x, y, circle_segment, l_0)
    line_segment.find_intersect(max_length)

    return [circle_segment, line_segment]


def calculate_path_length(
    x: Callable[[float], float],
    y: Callable[[float], float],
    start_angle: float,
    end_angle: float,
    alpha_0:float,
):
    # make an array
    phi = np.linspace(start_angle, end_angle, (end_angle-start_angle)/alpha_0*100) #resolution of the path integral to be 100 times larger than the mechanical resolution of the machine
    points = np.transpose(np.array([x(phi), y(phi)]))
    differences = np.diff(points, axis=0)
    distance_between_points = np.sqrt(np.sum(differences**2, axis=1))
    return np.sum(distance_between_points)


# function to generate a path made from circle segments and straight lines approximating a given curve r(phi)
# args: x(phi) - function that returnes the x coordinates of the curve at a given phi
#      y(phi) -  function that returnes the y coordinates of the curve at a given phi
#      x_d[[phi],x_d] - function that returns the x component of the tangent vector at a given phi
#      start_angle - the start angle of the path
#      end_angle - the end angle of the path
#      R - the radius of the circle segments
#      alpha_0 the angle increment allowed for circle segments
#      l_0 the length increment allowed for straight lines
def generate_path_segments(
    x: Callable[[float], float],
    y: Callable[[float], float],
    start_angle: float,
    end_angle: float,
    R: float,
    alpha_0: float,
    l_0: float,
):
    segments = []
    segments.extend(
        generate_segment_pair(x, y, start_angle, end_angle, R, alpha_0, l_0)
    )

    while segments[-1].end_angle < end_angle:
        segments.extend(
            generate_segment_pair(
                x, y, segments[-1].end_angle, end_angle, R, alpha_0, l_0
            )
        )
    return segments


def plot_path(
    x: Callable[[float], float],
    y: Callable[[float], float],
    start_angle: float,
    end_angle: float,
) -> None:
    # create array of values for x with phi between start_angle and end_angle
    phi = np.linspace(start_angle, end_angle, 100)
    # create array of values for y with phi between start_angle and end_angle
    x_values = x(phi)
    y_values = y(phi)
    # plot the path
    plt.plot(x_values, y_values)
    plt.draw()
    return


def plot_segments(
    x: Callable[[float], float],
    y: Callable[[float], float],
    start_angle: float,
    end_angle: float,
    R: float,
    alpha_0: float,
    l_0: float,
):
    segments:list[Segment] = generate_path_segments(x, y, start_angle, end_angle, R, alpha_0, l_0)
    for segment in segments:
        segment.plot()
    
    return





start_angle = 0
end_angle = 2 * np.pi
R_tool = 2  # radius of the bending tool
d_wire = 1  # diameter of the wire
R = R_tool + d_wire / 2
alpha_0 = (
    2 * np.pi / 200
)  # step angle in radiant. this is the value for most standard steppers with 1.8 deg
l_0 = (
    5.5 * alpha_0
)  # radius of the feeding tool multiplied with the minimum step angle of the stepper


t,tau = smp.symbols('t, tau', real=True, min=start_angle, max=end_angle)


x_expr = R*t * smp.cos(t)
y_expr = R*t * smp.sin(t)


dtx_expr = x_expr.diff()
dty_expr = y_expr.diff()
ddtx_expr = dtx_expr.diff()
ddty_expr = dty_expr.diff()

x=smp.lambdify(t, x_expr, 'numpy')
y=smp.lambdify(t, y_expr, 'numpy')
dtx=smp.lambdify(t, dtx_expr, 'numpy')
dty=smp.lambdify(t, dty_expr, 'numpy')

curvature = (dtx_expr*ddty_expr-ddtx_expr*dty_expr)/((dtx_expr**2+dty_expr**2)**(3/2))

arc_length = smp.integrate(smp.sqrt(dtx_expr**2+dty_expr**2),(tau,start_angle,t))

assert (
    x(start_angle) == 0 and y(start_angle) == 0
)  # start must be at center of the coordinate system




# make plot window square
plt.gca().set_aspect("equal", adjustable="box")

plot_path(x,y,start_angle,end_angle)
plt.draw()
plot_segments(x, y, start_angle, end_angle, R, alpha_0, l_0)

print()


###### gcode generation ######	
## stepper is configured to do 200 steps per mm so that 1 mm corresponds to 1 rotation instead.
def bend(n): 
    # format string to 3 decimal places
    return "G91\nG0 X{:.3f}\nG0 X-{:.3f}\nG90".format(n*0.01,n*0.01)

def feed(n):
    return "G0 Y{:.3f}".format(n*0.01)

# open a new text file
file = open("bending-file.gcode", "w")

file.write(bend(10))


