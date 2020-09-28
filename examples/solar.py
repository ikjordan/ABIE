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
    display_3d_data('abc.h5', names, scatter=True)

def execute_simulation(output_file):
    # create an ABIE instance
    sim = ABIE()

    # Select integrator. Possible choices are 'GaussRadau15', 'RungeKutta', etc.
    #sim.integrator = 'WisdomHolman'
    sim.integrator = 'GaussRadau15'
    #sim.integrator = 'RungeKutta'

    # Use the CONST_G parameter to set units
    sim.CONST_G = 1.48813611629e-34

    # The underlying implementation of the integrator ('ctypes' or 'numpy')
    sim.acceleration_method = 'ctypes'
    # sim.acceleration_method = 'numpy'

    # Add the solar system
    sim.add(mass=1.988475415966534e30, x=-7.136455199990743e-03, y=-2.795724239370603e-03, z= 2.061367448662713e-04, vx= 5.378459295718575e-06, vy=-7.406912670374715e-06, vz=-9.433299568131420e-08, name='Sun')
    sim.add(mass=3.301096181046679e23, x=-1.372300608220831e-01, y=-4.500833421231408e-01, z=-2.439217068677536e-02, vx= 2.137177414060989e-02, vy=-6.455396674574269e-03, vz=-2.487957667804378e-03, name='Mercury')
    sim.add(mass=4.867466257521635e24, x=-7.254387515234633e-01, y=-3.545003280048490e-02, z= 4.122031886718398e-02, vx= 8.034959858448681e-04, vy=-2.030262561468278e-02, vz=-3.235458387131530e-04, name='Venus')
    sim.add(mass=5.972365261370794e24, x=-1.842715550033945e-01, y= 9.644459624383047e-01, z= 2.020509819625348e-04, vx=-1.720224660756253e-02, vy=-3.166189066385779e-03, vz= 1.065953235594308e-08, name='Earth')
    sim.add(mass=6.417120205436417e23, x= 1.383579466546939e+00, y=-1.621204162199320e-02, z=-3.426152623891661e-02, vx= 6.768779529388747e-04, vy= 1.517984118242539e-02, vz= 3.015574313127102e-04, name='Mars')
    sim.add(mass=1.898187239616558e27, x= 3.994940946651115e+00, y= 2.935780101608396e+00, z=-1.015793842267007e-01, vx=-4.562948182122576e-03, vy= 6.435847813493556e-03, vz= 7.548674581187792e-05, name='Jupiter')
    sim.add(mass=5.684766319852324e26, x= 6.399272018423088e+00, y= 6.567193895707603e+00, z=-3.688702929216678e-01, vx=-4.286973417088009e-03, vy= 3.882908784079374e-03, vz= 1.028535168332451e-04, name='Saturn')
    sim.add(mass=8.682168328818365e25, x= 1.442472018423088e+01, y=-1.373711773058018e+01, z=-2.379347191117984e-01, vx= 2.683483979188155e-03, vy= 2.665288849251202e-03, vz=-2.486624055378518e-05, name='Uranus')
    sim.add(mass=1.024339999008106e26, x= 1.680490957344363e+01, y=-2.499455859992035e+01, z= 1.274294623716956e-01, vx= 2.584652466233836e-03, vy= 1.769493937752207e-03, vz=-9.600383320403503e-05, name='Neptune')
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
    sim.store_dt = 400

    # The integration timestep (does not apply to Gauss-Radau15)
    sim.h = 1

    # The size of the buffer. The buffer is full when `buffer_len` outputs are
    # generated. In this case, the collective output will be flushed to the HDF5
    # file, generating a `Step#n` HDF5 group
    sim.buffer_len = 10000

    # initialize the integrator
    sim.initialize()

    # perform the integration
    sim.integrate(365.25*1000)

    sim.stop()

    return hash2names

if __name__ == "__main__":
    main()
