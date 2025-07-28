"""Contains reference trajectories. You could add your own to test it with the controller and ekf setup.
   Each function should return [x, y, xd, yd, xdd, ydd]"""

import sympy as sp

def get_ref_figure_eight(t_current: float) -> list[float]:
    """Given the current time in seconds this helper gives
       current (x, y) positions (in meters), (x, y) velocities (in m/s)
       and (x, y) accelerations (in m/s^2) of the lemniscate of Gerono curve.
       Plotting this reference for all time steps going from 0 to 20seconds
       should resemble number 8.
       
       This helper uses python sympy library to analytically evaluate the velocities
       and accelerations of the curve.
       
       Args: t_current (float): Current time, t in seconds
       Returns: list[float] : [x, y, xd, yd, xdd, ydd]
    """

    # Symbolic variable
    t = sp.symbols('t')
    
    k = sp.Piecewise(
        ((sp.pi * t) / 10 - 0.5 * sp.pi, t <= 20),
        (1.5 * sp.pi, True)
    )

    # Define x(t) and y(t)
    x = -2 * sp.sin(k) * sp.cos(k)
    y = 2 * (sp.sin(k) + 1)

    # First derivatives
    x_d = sp.diff(x, t)
    y_d = sp.diff(y, t)

    # Second derivatives
    x_dd = sp.diff(x_d, t)
    y_dd = sp.diff(y_d, t)

    # Substitute t_current into the expressions
    subs = {t: t_current}
    x_val = float(x.evalf(subs=subs))
    y_val = float(y.evalf(subs=subs))
    x_d_val = float(x_d.evalf(subs=subs))
    y_d_val = float(y_d.evalf(subs=subs))
    x_dd_val = float(x_dd.evalf(subs=subs))
    y_dd_val = float(y_dd.evalf(subs=subs))

    return [x_val, y_val, x_d_val, y_d_val, x_dd_val, y_dd_val]


def get_ref_straight_line(t_current) -> list[float]:
    """Given the current time in seconds this helper gives
       current (x, y) positions (in meters), (x, y) velocities (in m/s)
       and (x, y) accelerations (in m/s^2) of a straight line of the form
       x = 0.25 * t and y = x.
       
       This helper uses python sympy library to analytically evaluate the velocities
       and accelerations of the curve.
       
       Args: t_current (float): Current time, t in seconds
       Returns: list[float] : [x, y, xd, yd, xdd, ydd]
    """
    # Symbolic variable
    t = sp.symbols('t')
    
    # Sanity check for our controller. Can the robot follow a simple straight line?
    x = 0.25 * t
    y = 1 * x
   
    # First derivatives
    x_d = sp.diff(x, t)
    y_d = sp.diff(y, t)

    # Second derivatives
    x_dd = sp.diff(x_d, t)
    y_dd = sp.diff(y_d, t)

    # Substitute t_current into the expressions
    subs = {t: t_current}
    x_val = float(x.evalf(subs=subs))
    y_val = float(y.evalf(subs=subs))
    x_d_val = float(x_d.evalf(subs=subs))
    y_d_val = float(y_d.evalf(subs=subs))
    x_dd_val = float(x_dd.evalf(subs=subs))
    y_dd_val = float(y_dd.evalf(subs=subs))

    return [x_val, y_val, x_d_val, y_d_val, x_dd_val, y_dd_val]

def get_ref_simple_curve(t_current) -> list[float]:
    """Given the current time in seconds this helper gives
       current (x, y) positions (in meters), (x, y) velocities (in m/s)
       and (x, y) accelerations (in m/s^2) of a semi-circle of the form
       x = R * sin((0.1 * pi * t)/2) and y = R * (1 - cos((0.1 * pi * t)/2)).
       
       This helper uses python sympy library to analytically evaluate the velocities
       and accelerations of the curve.
       
       Args: t_current (float): Current time, t in seconds
       Returns: list[float] : [x, y, xd, yd, xdd, ydd]
    """
    t = sp.symbols('t')
    # Radius of the semi-circle. You can play around with this value to see 
    # how steep of a curve can the robot follow
    R = 2
    x = R * sp.sin((sp.pi* 0.1 * t)/2)
    y = R * (1 - sp.cos((sp.pi * 0.1 * t) / 2))
    
    # First derivatives
    x_d = sp.diff(x, t)
    y_d = sp.diff(y, t)

    # Second derivatives
    x_dd = sp.diff(x_d, t)
    y_dd = sp.diff(y_d, t)

    # Substitute t_current into the expressions
    subs = {t: t_current}
    x_val = float(x.evalf(subs=subs))
    y_val = float(y.evalf(subs=subs))
    x_d_val = float(x_d.evalf(subs=subs))
    y_d_val = float(y_d.evalf(subs=subs))
    x_dd_val = float(x_dd.evalf(subs=subs))
    y_dd_val = float(y_dd.evalf(subs=subs))

    return [x_val, y_val, x_d_val, y_d_val, x_dd_val, y_dd_val]