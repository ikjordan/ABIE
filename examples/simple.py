"""
Run ABIE programmatically as a library.
"""
import numpy as np
try:
    from ABIE import ABIE
except ImportError:
    print("Failed")
    # Try to run local module, by adding path to directory above 
    # library code
    from sys import path
    from os.path import dirname

    path.append(dirname(path[0]))
    from ABIE import ABIE

from display import display_3d_data

def main():
    names = execute_simulation('abc.h5')
    display_3d_data('abc.h5', names)

def execute_simulation(output_file):
    # create an ABIE instance
    sim = ABIE()

    # Select integrator. Possible choices are 'GaussRadau15', 'RungeKutta', etc.
    sim.integrator = 'Euler'
    # sim.integrator = 'WisdomHolman'
    # sim.integrator = 'GaussRadau15'
    # sim.integrator = 'RungeKutta'
    # sim.integrator = 'LeapFrog'
    # sim.integrator = 'AdamsBashforth'

    # Use the CONST_G parameter to set units
    #sim.CONST_G = 4 * np.pi ** 2
    #sim.CONST_C = 63198.0
    sim.CONST_G = 1

    # The termination time (optional; can be overridden by the integrate() function)
    sim.t_end = 1000

    # The underlying implementation of the integrator ('ctypes' or 'numpy')
    #sim.acceleration_method = 'ctypes'
    sim.acceleration_method = 'numpy'

    # Add the objects
    sim.close_encounter_distance = 0
    # sim.max_close_encounter_events = 100
    # sim.max_collision_events = 1
    # sim.add(mass=1, x=1.0, y=1.0, z=0.0, vx=0.0, vy=0.7598356856515925, vz=0.0, name='One')
    # sim.add(mass=1, x=-0.5, y=0.8660254037844387, z=0.0, vx=-0.6580370064762463, vy=-0.3799178428257961, vz=0.0, name='Two')
    # sim.add(mass=1, x=-0.5, y=-0.8660254037844385, z=0.0, vx=0.6580370064762461, vy=-0.3799178428257966, vz=0.0, name='Three')

    sim.add(mass=1, x=-1.0, y=0.0, z=0.0, vx=0.0, vy=-0.5, vz=0.0, name='One')
    sim.add(mass=1, x=1.0, y=0.0, z=0.0, vx=0.0, vy=0.5, vz=0.0, name='Two')

    print(sim.particles)
    # The output file name. If not specified, the default is 'data.hdf5'
    sim.output_file = output_file
    sim.collision_output_file = 'abc.collisions.txt'
    sim.close_encounter_output_file = 'abc.ce.txt'

    # Build a dictionary mapping hashes to names
    hash2names = {}
    for particle in sim.particles:
        if particle.name is not None:
            hash2names[particle.hash] = particle.name

    # The output frequency
    sim.store_dt = 0.2

    # The integration timestep (does not apply to Gauss-Radau15)
    sim.h = 0.01

    # The size of the buffer. The buffer is full when `buffer_len` outputs are
    # generated. In this case, the collective output will be flushed to the HDF5
    # file, generating a `Step#n` HDF5 group
    sim.buffer_len = 10000

    # set primary body for the orbital element calculation
    # `#COM#`: use the center-of-mass as the primary body
    # `#M_MAX#`: use the most massive object as the primary object
    # `#M_MIN#`: use the least massive object as the primary object
    # One could also specify the name of the object (e.g, 'Sun', 'planet')
    # or the ID of the object (e.g., 0, 1), or a list of IDs / names defining
    # a subset of particles.
    # sim.particles.primary = [0, 1]

    # initialize the integrator
    sim.initialize()

    # perform the integration
    sim.integrate(18)

    sim.stop()

    return hash2names

if __name__ == "__main__":
    main()
