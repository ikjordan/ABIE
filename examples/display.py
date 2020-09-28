import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cycler import cycler
from ABIE import snapshot_convert

def display_3d_data(input_file, hash2names=None, scatter=False):
    # flatten the data
    converted = snapshot_convert(input_file)

    if converted:
        h5f = h5py.File(converted[0], 'r')

        x = h5f['/x'][()]     # Get x for the particles
        y = h5f['/y'][()]     # Get y for the particles
        z = h5f['/y'][()]     # Get z for the particles

        # Build names for the plot
        names = []
        if hash2names is not None:
            for hash in h5f['/hash'][(0)]:
                names.append(hash2names.get(hash,"Unknown"))

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_title('Trajectory')
        ax.prop_cycle : cycler(color='bgrcmyk')
        for i in range(0, len(x[0])): 
            # Plot all the positions for all objects
            if scatter:
                ax.scatter(x[:,i], y[:,i], z[:,i], s=0.2, label=names[i] if hash2names is not None else None)
            else:
                ax.plot(x[:,i], y[:,i], z[:,i], label=names[i] if hash2names is not None else None)

        # Finished with the data
        h5f.close()

        # Request a legend and display
        if hash2names is not None:
            ax.legend()
        plt.show()
