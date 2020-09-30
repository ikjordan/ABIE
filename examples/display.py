import h5py
import math
import numpy 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cycler import cycler
from ABIE import snapshot_convert

def display_3d_data(input_file, hash2names=None, names=None, title="Trajectory", scatter=False):
    # flatten the data
    converted = snapshot_convert(input_file)

    if converted:
        h5f = h5py.File(converted[0], 'r')

        x = h5f['/x'][()]     # Get x for the particles
        y = h5f['/y'][()]     # Get y for the particles
        z = h5f['/y'][()]     # Get z for the particles

        # Use the hash (if provided) in favour of the supplied names
        if hash2names is not None:
            names = []
            for hash in h5f['/hash'][(0)]:
                names.append(hash2names.get(hash,"Unknown"))

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_title(title)
        ax.prop_cycle : cycler(color='bgrcmyk')
        for i in range(0, len(x[0])): 
            # Plot all the positions for all objects
            if scatter:
                ax.scatter(x[:,i], y[:,i], z[:,i], s=0.1, label=names[i] if names is not None else None)
            else:
                ax.plot(x[:,i], y[:,i], z[:,i], label=names[i] if names is not None else None)

        # Finished with the data
        h5f.close()

        # Request a legend and display
        if names is not None:
            ax.legend(markerscale=30 if scatter else 1)
        plt.show()


def display_2d_data(input_file, hash2names=None, names=None, title="Trajectory", scatter=False):
    # flatten the data
    converted = snapshot_convert(input_file)

    if converted:
        h5f = h5py.File(converted[0], 'r')

        x = h5f['/x'][()]     # Get x for the particles
        y = h5f['/y'][()]     # Get y for the particles
        z = h5f['/y'][()]     # Get z for the particles

        # Use the hash (if provided) in favour of the supplied names
        if hash2names is not None:
            names = []
            for hash in h5f['/hash'][(0)]:
                names.append(hash2names.get(hash,"Unknown"))

        fig, ax = plt.subplots()

        ax.set_title(title)
        ax.prop_cycle : cycler(color='bgrcmyk')
        for i in range(0, len(x[0])): 
            # Plot x and y positions for all objects
            if scatter:
                ax.scatter(x[:,i], y[:,i], s=0.1, label=names[i] if names is not None else None)
            else:
                ax.plot(x[:,i], y[:,i], label=names[i] if names is not None else None)

        ax.axis('equal')
        # Finished with the data
        h5f.close()

        # Request a legend and display
        if names is not None:
            ax.legend(markerscale=30 if scatter else 1)
        plt.show()


def display_2d_e_and_i(input_file, hash2names=None, names=None, smooth = False, title="$e$ and $I$"):
    # flatten the data
    converted = snapshot_convert(input_file)

    if converted:
        h5f = h5py.File(converted[0], 'r')

        ecc = h5f['/ecc'][()]     # Get eccentricity
        inc = h5f['/inc'][()]     # Get inclinations 
        time = h5f['/time'][()]   # Get time - in days

        # Convert the time to millions of years
        time = time /(365.25 * 1000000)

        # Convert the inclination to degrees
        inc = inc * 180 / math.pi

        # Use the hash (if provided) in favour of the supplied names
        if hash2names is not None:
            names = []
            for hash in h5f['/hash'][(0)]:
                names.append(hash2names.get(hash,"Unknown"))

        fig, ax = plt.subplots(2, 1, sharex=True)
        fig.subplots_adjust(hspace=0)

        fig.suptitle(title, fontsize=16)
        for i in range(1, len(ecc[0])):
            # Plot eccentricity and incination against time - ignore the first data set
            start = 0
            # Optional smooth eccentricity
            if smooth == True:
                n = 200  
                try:
                    from scipy.signal import lfilter
                    b = [1.0 / n] * n
                    a = 1
                    ecc[:,i] = lfilter(b, a, ecc[:,i])
                    inc[:,i] = lfilter(b, a, inc[:,i])
                    start = n
                except:
                    print("Scipy not installed - falling back to numpy polynomial fit")
                    pol = numpy.polynomial.chebyshev.chebfit(time, ecc[:,i], n)
                    ecc[:,i] = numpy.polynomial.chebyshev.chebval(time, pol)
                    pol = numpy.polynomial.chebyshev.chebfit(time, inc[:,i], n)
                    inc[:,i] = numpy.polynomial.chebyshev.chebval(time, pol)

            ax[0].plot(time[start:], ecc[start:,i], label=names[i] if names is not None else None)
            ax[1].plot(time[start:], inc[start:,i])

        ax[0].set_ylabel('$e_i$')
        ax[1].set_ylabel('$I_i$/deg')
        ax[1].set_xlabel('$t$/Myr')

        # Finished with the data
        h5f.close()

        # Request a legend and display
        if names is not None:
            legend = fig.legend()
            for line in legend.get_lines():
                line.set_linewidth(4.0)
        plt.show()
