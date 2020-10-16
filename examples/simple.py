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

from h5 import H5
from display import Display

def main():

    # Select integrator. Possible choices are 'GaussRadau15', 'RungeKutta', etc.
    # integrator = 'Euler'
    # integrator =  'LeapFrog'
    # integrator = 'AdamsBashforth'
    # integrator =  'RungeKutta'
    # integrator = 'WisdomHolman'
    # integrator = 'GaussRadau15'


    count = 0
    output_file = []
    output_name = []
    scatter = []

    G = 1
    method = 'ctypes'
    # method = 'numpy'
    time_step = 0.001
    out_freq = 0.01
    end_time = 100

    # Simple two body
    mass = np.array([1.0, 1.0])
    pos = np.array([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    vel = np.array([ 0.0, -0.48, 0.0, 0.0, 0.48, 0.0])
    names = ['One', 'Two']
    output_file.append("simple_2.h5")
    output_name.append("Simple 2 Body")
    scatter.append(True)

    run(mass, pos, vel, names, integrator, method, G, time_step, out_freq, end_time, output_file[count])

    # 3 body figure of 8
    count += 1
    time_step = 0.0001

    mass = np.array([1.0, 1.0, 1.0])
    pos = np.array([0.9700436, -0.24308753, 0.0, -0.9700436, 0.24308753, 0.0, 0.0, 0.0, 0.0])
    vel = np.array([0.466203685, 0.43236573, 0.0, 0.466203685, 0.43236573, 0.0, -0.93240737, -0.86473146, 0.0])
    names = ['One', 'Two', 'Three']
    output_file.append("3_body_8.h5")
    output_name.append("3 Body Fig 8")
    scatter.append(False)

    run(mass, pos, vel, names, integrator, method, G, time_step, out_freq, end_time, output_file[count])

    # Perturbed 3 body
    count += 1
    time_step = 0.0001

    mass = np.array([1.0, 1.0, 1.0])
    pos = np.array([1.0, 0.0, 0.0, -0.5, 0.866025403784439, 0.0, -0.5, -0.866025403784438, 0.0])
    vel = np.array([0.0001, 0.759835685651593, 0.0, -0.658037006476246, -0.379917842825796, 0.0, 0.658037006476246, -0.379917842825797, 0.0])
    names = ['One', 'Two', 'Three']
    output_file.append("3_body_pert.h5")
    output_name.append("3 Body Perturbed")
    scatter.append(False)

    run(mass, pos, vel, names, integrator, method, G, time_step, out_freq, end_time, output_file[count])

    # Pythagorean 3 body
    end_time = 62
    count += 1
    time_step = 0.00000001
    mass = np.array([3.0, 4.0, 5.0])
    pos = np.array([1.0, 3.0, 0.0, -2.0, -1.0, 0.0, 1.0, -1.0, 0.0])
    vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    names = ['One', 'Two', 'Three']
    output_file.append("Pi_3_body.h5")
    output_name.append("Pythagorean 3 Body")
    scatter.append(False)

    run(mass, pos, vel, names, integrator, method, G, time_step, out_freq, end_time, output_file[count])

    # 5 body figure of 8
    end_time = 12
    count += 1
    time_step = 0.0001

    mass = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    pos = np.array([1.657666, 0.0, 0.0, 0.439775, -0.169717, 0.0, -1.268608, -0.267651, 0.0, -1.268608, 0.267651, 0.0, 0.439775, 0.169717, 0.0])
    vel = np.array([0.0, -0.593786, 0.0, 1.822785, 0.128248, 0.0, 1.271564, 0.168645, 0.0, -1.271564, 0.168645, 0.0, -1.822785, 0.128248, 0.0])
    names = ['One', 'Two', 'Three', 'Four', 'Five']
    output_file.append("5_body_8.h5")
    output_name.append("5 Body Fig 8")
    scatter.append(False)

    run(mass, pos, vel, names, integrator, method, G, time_step, out_freq, end_time, output_file[count])

    # Display all of the results
    h5 = H5()
    d = Display(h5)
    for i, file in enumerate(output_file):
        h5.set_data(file)
        title = output_name[i] +  ": " + integrator
        d.display_2d_data(names=names, scatter=scatter[i], title=title)
        d.display_energy_delta(G=G, title="Energy Delta: " + title, to_bary=(integrator=='WisdomHolman'))
    h5.close()
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
