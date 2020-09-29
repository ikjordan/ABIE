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

from display import display_2d_data

def main():
    execute_simulation('abc.h5')

def execute_simulation(output_file):
    # create an ABIE instance
    sim = ABIE()

    # Select integrator. Possible choices are 'GaussRadau15', 'RungeKutta', etc.
    # integrator = 'Euler'
    # integrator =  'LeapFrog'
    # integrator = 'AdamsBashforth'
    # integrator =  'RungeKutta'
    # integrator = 'WisdomHolman'
    integrator = 'GaussRadau15'
    sim.integrator = integrator

    # Use the CONST_G parameter to set units
    sim.CONST_G = 1

    # The underlying implementation of the integrator ('ctypes' or 'numpy')
    sim.acceleration_method = 'ctypes'
    #sim.acceleration_method = 'numpy'

    # Pure 3 body
    #sim.add(mass=1.0, x= 1.000000000000000, y= 0.000000000000000, z= 0.000000000000000, vx= 0.000000000000000, vy= 0.759835685651593, vz= 0.000000000000000, name='One')
    #sim.add(mass=1.0, x=-0.500000000000000, y= 0.866025403784439, z= 0.000000000000000, vx=-0.658037006476246, vy=-0.379917842825796, vz= 0.000000000000000, name='Two')
    #sim.add(mass=1.0, x=-0.500000000000000, y=-0.866025403784438, z= 0.000000000000000, vx= 0.658037006476246, vy=-0.379917842825797, vz= 0.000000000000000, name='Three')

    # Velocity perturbation on pure 3 body
    sim.add(mass=1.0, x= 1.000000000000000, y= 0.000000000000000, z= 0.000000000000000, vx= 0.000100000000000, vy= 0.759835685651593, vz= 0.000000000000000, name='One')
    sim.add(mass=1.0, x=-0.500000000000000, y= 0.866025403784439, z= 0.000000000000000, vx=-0.658037006476246, vy=-0.379917842825796, vz= 0.000000000000000, name='Two')
    sim.add(mass=1.0, x=-0.500000000000000, y=-0.866025403784438, z= 0.000000000000000, vx= 0.658037006476246, vy=-0.379917842825797, vz= 0.000000000000000, name='Three')

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
    sim.store_dt = 0.01

    # The integration timestep (does not apply to Gauss-Radau15)
    sim.h = 0.001

    # The size of the buffer. The buffer is full when `buffer_len` outputs are
    # generated. In this case, the collective output will be flushed to the HDF5
    # file, generating a `Step#n` HDF5 group
    sim.buffer_len = 10000

    # initialize the integrator
    sim.initialize()

    # perform the integration
    sim.integrate(100)

    sim.stop()

    # display the data
    display_2d_data(output_file, hash2names=hash2names, title=integrator)


if __name__ == "__main__":
    main()
