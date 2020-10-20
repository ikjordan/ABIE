"""
Run ABIE programmatically as a library.
"""
import numpy as np
from datetime import datetime
import os
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

from h5 import H5
from display import Display

def main():
    filename = 'kuiper.h5'
    execute_simulation(filename)


def execute_simulation(output_file):
    # create an ABIE instance
    sim = ABIE()

    # Select integrator. Possible choices are 'GaussRadau15', 'RungeKutta', etc.
    # integrator = 'Euler'
    # integrator = 'LeapFrog'
    # integrator = 'AdamsBashforth'
    # integrator = 'RungeKutta'
    # integrator = 'WisdomHolman'
    integrator = 'GaussRadau15'
    sim.integrator = integrator


    # Use the CONST_G parameter to set units
    G = 4 * np.pi ** 2
    sim.CONST_G = G

    # The underlying implementation of the integrator ('ctypes' or 'numpy')
    sim.acceleration_method = 'ctypes'
    # sim.acceleration_method = 'numpy'

    # Add the sun + Neptune
    sim.add(mass=1.0, x=0, y=0.0, z= 0.0, vx=0.0, vy=0.0, vz=0.0, name='Sun')
    sim.add(mass=5.15e-5, a=30.07, e=0.0086, name='Neptune')

    # Add the kuiper belt particles - initially uniformally distributed
    n_oc = 4000
    start_semi = np.linspace(33.501, 53.501, n_oc, endpoint=False)

    # Circular orbits with no inclination
    for i in range(n_oc):
        sim.add(1e-15, a=start_semi[i], e=0, i=0, primary='Sun', name=('test_particle{}'.format(i)))

    # Set the momentum in the system to 0
    sim.particles.balance_system()
    com_s = sim.particles.get_center_of_mass()

    sim.output_file = output_file
    sim.collision_output_file = os.path.splitext(output_file)[0] + '.collisions.txt'
    sim.close_encounter_output_file = os.path.splitext(output_file)[0] + '.ce.txt'

    # The output frequency
    sim.store_dt = 100              # Log time in years

    # The integration timestep (does not apply to Gauss-Radau15)
    sim.h = 10

    total_time = 1000               # Total integration time in years

    sim.buffer_len = 1000           # Number of logs to store in memory before writing them to file

    # initialize the integrator
    sim.initialize()

    startTime = datetime.now()
    sim.integrate(total_time)
    sim.stop()
    endTime = datetime.now()
    print("Time taken = {} seconds.".format(endTime - startTime))

    # Display Start and Final centre of mass
    com_e = sim.particles.get_center_of_mass()
    print('Initial COM. Mass: {} x: {} y: {} z: {} vx: {} vy: {} vz: {}'.format(com_s.mass, com_s.x, com_s.y, com_s.z, 
                                                                                com_s.vx, com_s.vy, com_s.vz))
    print('Final COM. Mass: {} x: {} y: {} z: {} vx: {} vy: {} vz: {}'.format(com_e.mass, com_e.x, com_e.y, com_e.z, 
                                                                              com_e.vx, com_e.vy, com_e.vz))

    h5 = H5(output_file)

    # Get the data
    semi = h5.get_semi_major()
    ecc = h5.get_eccentricity()

    # display the data
    d = Display(h5)

    # Show initial distribution
    names = ['Initial']
    d.display_histogram(start_semi, 31, 56, 500, names=names, title="Kuiper Initial Distribution", units='AU')

    # Display distribution at mid and full simulation time
    names = ['Mid', 'Full']
    x, y = np.shape(semi)

    d.display_histogram(np.column_stack((semi[(x-1)//2, 2:], semi[-1, 2:])), 31, 56, 500, names=names, title="Kuiper Mid and Full Distribution", units='AU')
    d.display_2d_scatter(np.column_stack((semi[(x-1)//2, 2:], semi[-1, 2:])),
                         np.column_stack((ecc[(x-1)//2, 2:], ecc[-1, 2:])), 
                         names=names, title="$e$ for Distribution", y_units='AU')
    d.display_energy_delta(G=G, sim=sim, to_bary=(integrator=='WisdomHolman'), units="Yr")

    # Display the sun, neptune + n particles with highest eccentricity
    n = 5

    # Get the indexes of the bodies with the largest eccentricities, excluding the sun and neptune
    ind = np.argpartition(ecc[-1, 2:], -n)[-n:] +2
    
    # Add on sun and neptune
    ind = np.concatenate(([0, 1], ind)) * 3
    x0 = np.empty((x, 2 + n))
    y0 = np.empty((x, 2 + n))

    # Extract the x and y positions
    state = h5.get_state()
    x0 = state[:, ind]
    y0 = state[:, (ind + 1)]

    d.display_2d_scatter(x0, y0, title="High Excentricities", equal=True, x_units="AU", y_units="AU")
    d.show()


if __name__ == "__main__":
    main()
