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
    sim.CONST_G = 4 * np.pi ** 2

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
        sim.add(1e-15, a=start_semi[i], e=0, i=0, name=('test_particle{}'.format(i)))

    sim.output_file = output_file
    sim.collision_output_file = os.path.splitext(output_file)[0] + '.collisions.txt'
    sim.close_encounter_output_file = os.path.splitext(output_file)[0] + '.ce.txt'

    # The output frequency
    sim.store_dt = 100              # Log data every 100 years

    # The integration timestep (does not apply to Gauss-Radau15)
    sim.h = 1                       # 1 Step per year

    sim.buffer_len = 1000

    # initialize the integrator
    sim.initialize()

    # perform the integration
    divisor = 1000

    startTime = datetime.now()
    sim.integrate(divisor)
    sim.stop()
    endTime = datetime.now()
    print("Time taken = {} seconds.".format(endTime - startTime))

    h5 = H5(output_file)

    # Get the semi-major axis
    semi = h5.get_semi_major()
    x, y = np.shape(semi)

    ecc = h5.get_eccentricity()

    # display the data
    d = Display(h5)

    # Show initial distribution
    names = ['Initial']
    d.display_histogram(start_semi, 31, 56, 500, names=names, title="Kuiper Initial Distribution", units='AU')

    # Display distribution at mid and full simulation time
    names = ['Mid', 'Full']
    d.display_histogram(np.column_stack((semi[(x-1)//2, 2:], semi[x - 1, 2:])), 31, 56, 500, names=names, title="Kuiper Mid and Full Distribution", units='AU')
    d.display_2d_scatter(np.column_stack((semi[(x-1)//2, 2:], semi[x - 1, 2:])),
                         np.column_stack((ecc[(x-1)//2, 2:], ecc[x - 1, 2:])), 
                         names=names, title="$e$ for Distribution", units='AU')

    d.show()


if __name__ == "__main__":
    main()
