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
    # create an ABIE instance
    sim = ABIE()

    # Select integrator. Possible choices are 'GaussRadau15', 'RungeKutta', etc.
    # integrator = 'Euler'
    # integrator = 'LeapFrog'
    # integrator = 'AdamsBashforth'
    # integrator = 'RungeKutta'
    integrator = 'WisdomHolman'
    # integrator = 'GaussRadau15'
    sim.integrator = integrator

    sim.CONST_G = 6.67430e-20   # Unit of length is km

    # The underlying implementation of the integrator ('ctypes' or 'numpy')
    sim.acceleration_method = 'ctypes'
    # sim.acceleration_method = 'numpy'

    # Add saturn + Epimetheus & Janus
    mass = np.array([5.684766319852324e26, 5.266e17, 1.8975e18])
 
    # Data for Epimetheus - have swapped semi major axis quoted on Wikipedia with Janus
    xe = 1.5146e5
    Te = get_period(mass[0], mass[1], xe, sim.CONST_G)
    vye = get_velocity_magnitude(mass[0], mass[1], xe, sim.CONST_G)

    # Data for Janus
    xj = 1.5141e5
    Tj = get_period(mass[0], mass[2], xj, sim.CONST_G)
    vyj = get_velocity_magnitude(mass[0], mass[2], xj, sim.CONST_G)

    velocity = np.array([0.0,0.0,0.0, 0.0,-vye,0.0, 0.0,vyj,0.0])

    # Correct momentum of Saturn
    vel = Tools.balance_momentum(velocity, mass, first_particle=True)

    # Add to sim
    sim.add(mass=mass[0], x=0.0, y=0.0, z=0.0, vx=vel[0], vy=vel[1], vz=vel[2], name='Saturn')
    sim.add(mass=mass[1], x=xe, y=0.0, z=0.0, vx=0.0, vy=-vye, vz=0.0, name='Epimetheus')
    sim.add(mass=mass[2], x=-xj, y=0.0, z=0.0, vx=0.0, vy=vyj, vz=0.0, name='Janus')

    # get a list of names back
    names = sim.particles.names

    # The output file name. If not specified, the default is 'data.hdf5'
    sim.output_file = output_file
    sim.collision_output_file = 'abc.collisions.txt'
    sim.close_encounter_output_file = 'abc.ce.txt'

    # The integration timestep (does not apply to Gauss-Radau15)
    sim.h = 60                          # Step every minute

    # The output frequency
    out_frequency = sim.h * 60 * 4
    sim.store_dt = out_frequency        # Log data every 4 hours

    # The size of the buffer. The buffer is full when `buffer_len` outputs are
    # generated. In this case, the collective output will be flushed to the HDF5
    # file, generating a `Step#n` HDF5 group
    sim.buffer_len = 10000

    # initialize the integrator
    sim.initialize()

    # perform the integration for 10000 periods
    seconds = Te * 10000
    divisor = 365.25*24*60*60           # Convert to years
    units = "Yrs"

    sim.integrate(seconds)
    sim.stop()

    h5 = H5(output_file)

    state = h5.get_state()
    ecc = h5.get_eccentricity()
    time = h5.get_time()
    peak = np.argmax(ecc[:,1])

    # step back by 10 periods
    back = int(10 * Te / out_frequency)     # Convert from seconds to logging frequency

    # Extract the positions and velocities for the start of the focussed simulation
    pos = np.reshape(state[peak-back, 0:9], 9)
    vel = np.reshape(state[peak-back, 9:18], 9)

    focus_name = 'focus.h5'

    # Run the focussed simulation for 20 periods
    focus(mass, pos, vel, integrator, sim.acceleration_method,
          sim.CONST_G, 20*Te, focus_name)

    print("Period Te {:.3f} Hours, Tj {:.3f} Hours".format(Te/60/60, Tj/60/60))
    print("Initial Radius: E {:.1f} km, J {:.1f} km".format(xe, xj))

    # display the data
    d = Display(h5)
    d.display_2d_data(names=names, title=integrator, scatter=True, equal=True, to_helio=(integrator!='WisdomHolman'))
    d.display_radius(names=names, title=integrator, divisor=divisor, units=units, to_helio=(integrator!='WisdomHolman'))
    d.display_2d_e_and_i(names=names, divisor=divisor, units=units)
    d.display_energy_delta(G=sim.CONST_G, divisor=divisor, units=units, to_bary=(integrator=='WisdomHolman'))

    h5.set_data(focus_name)

    box = np.array((-40, 40, 0, 0))

    # Get the radii to calculate the bounding box
    radii = h5.get_radii(to_helio=(integrator!='WisdomHolman'))
    box[2] = min(np.amin(radii[:,1]), np.amin(radii[:,2])) - 2
    box[3] = max(np.amax(radii[:,1]), np.amax(radii[:,2])) + 2
    
    d.display_2d_data(names=names, title=integrator, to_helio=(integrator!='WisdomHolman'), box=box)
    d.display_radius(names=names, title=integrator, divisor=divisor, units=units, to_helio=(integrator!='WisdomHolman'))
    d.display_2d_e_and_i(names=names, divisor=divisor, units=units)
    d.show()


def focus(mass, pos, vel, integrator, method, G, end_time, output):
    sim = ABIE()

    # The integration timestep (does not apply to Gauss-Radau15)
    sim.h = 10                          # Step every 10 seconds

    # The output frequency
    sim.store_dt = sim.h * 2            # Log every 2nd point

    # The size of the buffer. The buffer is full when `buffer_len` outputs are
    # generated. In this case, the collective output will be flushed to the HDF5
    # file, generating a `Step#n` HDF5 group
    sim.buffer_len = 10000

    sim.integrator = integrator

    # Use the CONST_G parameter to set units
    sim.CONST_G = G

    # The underlying implementation of the integrator ('ctypes' or 'numpy')
    sim.acceleration_method = method

    # Add to sim
    sim.add(mass=mass[0], x=pos[0], y=pos[1], z=pos[2], vx=vel[0], vy=vel[1], vz=vel[2], name='Saturn')
    sim.add(mass=mass[1], x=pos[3], y=pos[4], z=pos[5], vx=vel[3], vy=vel[4], vz=vel[5], name='Epimetheus')
    sim.add(mass=mass[2], x=pos[6], y=pos[7], z=pos[8], vx=vel[6], vy=vel[7], vz=vel[8], name='Janus')

    # The output file name. If not specified, the default is 'data.hdf5'
    sim.output_file = output

    # Run the integrator
    sim.initialize()
    sim.integrate(end_time)
    sim.stop()


if __name__ == "__main__":
    main()
