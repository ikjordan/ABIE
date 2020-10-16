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

from datetime import datetime
from h5 import H5
from display import Display

def main():
    filename = 'oort.h5'
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

    # Add the solar system
    sim.add(mass=1.0, x=0, y=0.0, z= 0.0, vx=0.0, vy=0.0, vz=0.0, name='Sun')
    sim.add(mass=9.5458e-4, a=5.02, e=0.05, name='Jupiter')
    sim.add(mass=2.858e-4, a=9.537, e=0.05, name='Saturn')
    sim.add(mass=4.366e-5, a=19.189, e=0.05, name='Uranus')
    sim.add(mass=5.15e-5, a=30.07, e=0.0086, name='Neptune')

    # Add the oort cloud particles
    n_oc = 1000
    semi = np.random.uniform(500, 10000, n_oc)
    ecc = np.random.uniform(0.3, 0.99, n_oc)
    inc = np.random.uniform(0, 2*np.pi, n_oc)

    for i in range(n_oc):
        sim.add(0, a=semi[i], e=ecc[i], i=inc[i], name=('test_particle{}'.format(i)))

    # get a list of names back
    names = sim.particles.names

    # The output file name. If not specified, the default is 'data.hdf5'
    sim.output_file = output_file
    sim.collision_output_file = 'abc.collisions.txt'
    sim.close_encounter_output_file = 'abc.ce.txt'

    # The output frequency
    sim.store_dt = 100          # Log data every 100 years

    # The integration timestep (does not apply to Gauss-Radau15)
    sim.h = 0.1                   # Step 10 times per year

    sim.buffer_len = 10000

    # initialize the integrator
    sim.initialize()

    # perform the integration
    divisor = 1000            # 1000 years
    units = "kYr"

    startTime = datetime.now()
    sim.integrate(divisor)
    sim.stop()
    endTime = datetime.now()
    print("Time taken = {} seconds.".format(endTime - startTime))

    h5 = H5(output_file)

    # Get the semi-major axis
    semi = h5.get_semi_major()
    x, y = np.shape(semi)

    # display the data
    d = Display(h5)
    d.display_plot(np.column_stack((semi[0,5:], semi[x - 1, 5:])), 0, 15000, 60, title="Oort", units='AU')
    d.show()


if __name__ == "__main__":
    main()
