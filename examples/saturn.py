"""
Run ABIE programmatically as a library.
"""
import numpy as np
try:
    from ABIE import ABIE
except ImportError:
    print("astroabie package not installed, falling back to local")
    # Try to run local module, by adding path to directory above 
    # library code
    from sys import path
    from os.path import dirname

    path.append(dirname(path[0]))
    from ABIE import ABIE

from math import pi
from math import sqrt
from ABIE import Tools
from h5 import H5
from display import Display

def main():
    filename = 'saturn.h5'

    execute_simulation(filename)

def get_velocity_magnitude(m1, m2, r, G):
    return sqrt(G * (m1 + m2) / r)

def get_period(m1, m2, r, G):
    return 2 * pi * sqrt(r**3 / (G * (m1 + m2)))

def execute_simulation(output_file):
    # Select integrator. Possible choices are 'GaussRadau15', 'RungeKutta', etc.
    # integrator = 'Euler'
    # integrator = 'LeapFrog'
    # integrator = 'AdamsBashforth'
    # integrator = 'RungeKutta'
    # integrator = 'WisdomHolman'
    integrator = 'GaussRadau15'

    G = 6.67430e-20   # Unit of length is km

    # The underlying implementation of the integrator ('ctypes' or 'numpy')
    acceleration_method = 'ctypes'
    # acceleration_method = 'numpy'

    # Add saturn + Epimetheus & Janus
    names = ['Saturn', 'Epimetheus', 'Janus']
    moon_mult = 1.0   # Change to investigate the effect of larger moons
    mass = np.array([5.684766319852324e26, 5.266e17*moon_mult, 1.8975e18*moon_mult])
 
    # Data for Epimetheus - have swapped semi major axis quoted on Wikipedia with Janus
    xe = 1.5146e5
    Te = get_period(mass[0], mass[1], xe, G)
    vye = get_velocity_magnitude(mass[0], mass[1], xe, G)

    # Data for Janus
    xj = 1.5141e5
    Tj = get_period(mass[0], mass[2], xj, G)
    vyj = get_velocity_magnitude(mass[0], mass[2], xj, G)

    pos = np.array([0.0, 0.0, 0.0,  xe,  0.0, 0.0, -xj, 0.0, 0.0])
    vel = np.array([0.0, 0.0, 0.0, 0.0, -vye, 0.0, 0.0, vyj, 0.0])

    # Correct momentum of Saturn
    vel[0:3] = Tools.balance_momentum(vel, mass, first_particle=True)

    # Trigger simulation
    time_step = 60                  # 1 minute
    out_freq = 60 * 4 * time_step   # log data once every 4 hours
    run_time = 5000 * Te            # Integrate for 5000 periods

    run(mass, pos, vel, names, integrator, acceleration_method, G, 
        time_step, out_freq, run_time, output_file)

    h5 = H5(output_file)

    state = h5.get_state()
    ecc = h5.get_eccentricity()
    time = h5.get_time()

    # Find the point of maximum eccentricity
    peak = np.argmax(ecc[:,1])

    # step back by 10 periods
    back = int(10 * Te / out_freq)  # Convert from seconds to logging frequency

    # Extract the positions and velocities for the start of the focussed simulation
    pos = np.reshape(state[peak-back, 0:9], 9)
    vel = np.reshape(state[peak-back, 9:18], 9)

    # Run the focussed simulation for 20 periods
    time_step = 10              # 10 seconds
    out_freq = 2 * time_step    # Output every 20 seconds
    run_time = 20 * Te          # Integrate for 20 periods
    focus_name = 'focus.h5'

    run(mass, pos, vel, names, integrator, acceleration_method, G, 
        time_step, out_freq, run_time, focus_name)

    print("Period Te {:.3f} Hours, Tj {:.3f} Hours".format(Te/60/60, Tj/60/60))
    print("Initial Radius: E {:.1f} km, J {:.1f} km".format(xe, xj))

    # display the data
    divisor = 365.25*24*60*60           # Convert to years
    units = "Yrs"

    d = Display(h5)
    d.display_2d_data(names=names, title=integrator, scatter=True, equal=True, to_helio=(integrator!='WisdomHolman'))
    d.display_semi_major(names=names, title=integrator, divisor=divisor, units=units)
    d.display_2d_e_and_i(names=names, divisor=divisor, units=units)
    d.display_energy_delta(G=G, divisor=divisor, units=units, to_bary=(integrator=='WisdomHolman'), sim=sim)

    h5.set_data(focus_name)

    box = np.array((-40, 40, 0, 0))

    # Get the semi-major axis to calculate the bounding box
    semi = h5.get_semi_major()
    box[2] = min(np.amin(semi[:,1]), np.amin(semi[:,2])) - 2
    box[3] = max(np.amax(semi[:,1]), np.amax(semi[:,2])) + 2
    
    d.display_2d_data(names=names, title=integrator, to_helio=(integrator!='WisdomHolman'), box=box)
    d.display_semi_major(names=names, title=integrator, divisor=divisor, units=units)
    d.display_2d_e_and_i(names=names, divisor=divisor, units=units)
    d.show()


def run(mass, pos, vel, names, integrator, method, G, time_step, out_freq, end_time, output):
    sim = ABIE()

    # The integration timestep (does not apply to Gauss-Radau15)
    sim.h = time_step

    # The output frequency
    sim.store_dt = out_freq

    sim.buffer_len = 10000
    sim.integrator = integrator
    sim.CONST_G = G
    sim.acceleration_method = method
    sim.output_file = output

    # Add to sim
    for i in range(0,len(mass)):
        sim.add(mass=mass[i], x=pos[3*i], y=pos[3*i+1], z=pos[3*i+2],
                vx=vel[3*i], vy=vel[3*i+1], vz=vel[3*i+2], name=names[i])

    # Run the integrator
    sim.initialize()
    sim.integrate(end_time)
    sim.stop()


if __name__ == "__main__":
    main()
