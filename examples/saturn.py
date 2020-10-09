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
from display import Display

def main():
    filename = 'animate.h5'

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


    # Use the CONST_G parameter to set units
    sim.CONST_G = 6.67430e-11

    # The underlying implementation of the integrator ('ctypes' or 'numpy')
    sim.acceleration_method = 'ctypes'
    # sim.acceleration_method = 'numpy'

    # Add saturn + Epimetheus & Janus
    mass = np.array([5.684766319852324e26, 5.266e17, 1.8975e18])
 
    # Data for Epimetheus - have swapped semi major axis quoted on Wikipedia with Janus
    xe = 1.5146e8
    Te = get_period(mass[0], mass[1], xe, sim.CONST_G)
    vye = get_velocity_magnitude(mass[0], mass[1], xe, sim.CONST_G)

    # Data for Janus
    xj = 1.5141e8
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
    sim.store_dt = sim.h *60*8          # Log data every 8 hours

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

    pos = np.zeros(shape=(2, 3))
    for i in range(0,2):
        for j in range(0,3):
            pos[i,j] = sim.particles.positions[3*(i+1)+j] - sim.particles.positions[j]

    xe_new = np.linalg.norm(pos[0])
    xj_new = np.linalg.norm(pos[1])

    print("Period Te {:.3f} Hours, Tj {:.3f} Hours".format(Te/60/60, Tj/60/60))
    print("Initial Radius: E {:.1f} km, J {:.1f} km".format(xe/1000.0, xj/1000.0))
    print("Current Radius: E {:.1f} km, J {:.1f} km".format(xe_new/1000.0, xj_new/1000.0))
    print("Delta Radius: E {:.1f} km, J {:.1f} km".format(xe_new-xe, xj_new-xj))

    # display the data
    d = Display(output_file)
    d.display_2d_data(names=names, title=integrator, scatter=True, bary=(integrator!='WisdomHolman'))
    d.display_radius(names=names, title=integrator, divisor=divisor, units=units, bary=(integrator!='WisdomHolman'))
    d.display_3d_data(names=names, title=integrator, scatter=True, bary=(integrator!='WisdomHolman'))
    d.display_energy_delta(g=sim.CONST_G, divisor=divisor, units=units, helio=(integrator=='WisdomHolman'))
    d.show()


if __name__ == "__main__":
    main()
