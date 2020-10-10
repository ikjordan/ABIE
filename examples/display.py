import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cycler import cycler
from h5 import H5

class Display:
    def __init__(self, h5=None):
        self.next_figure = 1
        self.h5 = h5


    def _get_figure(self):
        # Creates a new figure
        fig = plt.figure(self.next_figure)
        self.next_figure += 1
        return fig


    def set_data(self, h5):
        # Caches the h5 data
        self.h5 = h5


    def show(self):
        # Show all of the figures
        if self.next_figure > 1:
            plt.show()


    def display_3d_data(self, hash2names=None, names=None, title="Trajectory", scatter=False, to_helio=False, equal=False):
        # Use the hash (if provided) in favour of the supplied names
        if hash2names is not None:
            names = self.h5.hash_to_names(hash2names)

        # Convert to heliocentric coords if requested
        states = self.h5.get_state_heliocentric() if to_helio else self.h5.get_state()

        fig = self._get_figure()
        ax = fig.gca(projection='3d')
        ax.set_title(title)

        _, width = states.shape
        for i in range(0, width // 6): 
            # Plot all the positions for all objects
            if scatter:
                ax.scatter(states[:,3*i], states[:,3*i+1], states[:,3*i+2], s=0.1, label=names[i] if names is not None else None)
            else:
                ax.plot(states[:,3*i], states[:,3*i+1], states[:,3*i+2], label=names[i] if names is not None else None)

        if equal:
            # As equal does not work for 3d plots
            left, right = ax.get_xlim()
            bottom, top = ax.get_ylim()
            low, high = ax.get_zlim()

            neg = min(left, bottom, low)
            pos = max(right, top, high)

            ax.set_xlim(neg, pos)
            ax.set_ylim(neg, pos)
            ax.set_zlim(neg, pos)

        # Request a legend
        if names is not None:
            ax.legend(markerscale=30 if scatter else 1)


    def display_2d_data(self, hash2names=None, names=None, title="Trajectory", scatter=False, to_helio=False, equal=True, box=None):
        # Use the hash (if provided) in favour of the supplied names
        if hash2names is not None:
            names = self.h5.hash_to_names(hash2names)

        # Convert to heliocentric coords if requested
        states = self.h5.get_state_heliocentric() if to_helio else self.h5.get_state()

        self._get_figure()
        ax = plt.subplot(111)
        ax.set_title(title)

        _, width = states.shape
        for i in range(0, width // 6): 
            # Plot x and y positions for all objects
            if scatter:
                ax.scatter(states[:,3*i], states[:,3*i+1], s=0.1, label=names[i] if names is not None else None)
            else:
                ax.plot(states[:,3*i], states[:,3*i+1], label=names[i] if names is not None else None)

        # Configure axes
        if box is not None:
            ax.set_xlim(box[0], box[1])
            ax.set_ylim(box[2], box[3])
        elif equal:
            ax.axis('equal')

        # Request a legend and display
        if names is not None:
            ax.legend(markerscale=30 if scatter else 1)


    def display_radius(self, hash2names=None, names=None, title="Trajectory", divisor=1.0, units=None, to_helio=True, ignore_first=True):
        # Get time and convert
        time = self.h5.get_time()
        time = time / divisor

        # Use the hash (if provided) in favour of the supplied names
        if hash2names is not None:
            names = self.h5.hash_to_names(hash2names)

        radii = self.h5.get_radii(to_helio)
        _, particles = radii.shape

        self._get_figure()
        ax = plt.subplot(111)
        ax.set_title(title)
        for i in range(1 if ignore_first else 0, particles): 
            # Plot radius against time for all particles, excluding the first particle
            ax.plot(time, (radii[:,i]), label=names[i] if names is not None else None)
        
        if ignore_first:
            y_lab = "Distance from {} /km"
            ax.set_ylabel(y_lab.format(names[0] if names is not None else 'Centre'))
        else:
            ax.set_ylabel("Distance from centre /km")

        if units is not None:
            x_lab = '$t$/{}'.format(units)
        else:
            x_lab = 'Time'
        ax.set_xlabel(x_lab)

        # Request a legend and display
        if names is not None:
            ax.legend()

    def display_energy_delta(self, title="Energy Delta", G=1.0, divisor=1.0, units=None, to_bary=False):
        # Get time
        time = self.h5.get_time()
        
        # Convert the time
        time = time / divisor

        # Get energy against time
        energy = self.h5.compute_energy(G, to_bary)

        # Plot energy delta
        self._get_figure()
        ax = plt.subplot(111)
        ax.set_title(title)

        # Plot delta against time
        ax.plot(time, (energy - energy[0]) / energy[0])

        ax.set_ylabel('$\Delta$')
        if units is not None:
            x_lab = '$t$/{}'.format(units)
        else:
            x_lab = 'Time'
        ax.set_xlabel(x_lab)


    def display_2d_e_and_i(self, hash2names=None, names=None, smooth = False, title="$e$ and $I$", divisor=1.0, units=None):
        time = self.h5.get_time()
        ecc = self.h5.get_eccentricity()
        inc = self.h5.get_inclination()

        # Convert the time 
        time = time /divisor

        # Convert the inclination to degrees
        inc = inc * 180 / math.pi

        # Use the hash (if provided) in favour of the supplied names
        if hash2names is not None:
            names = self.h5.hash_to_names(hash2names)

        fig = self._get_figure()
        ax0 = plt.subplot(211)
        ax1 = plt.subplot(212, sharex = ax0)
        fig.subplots_adjust(hspace=0)

        fig.suptitle(title, fontsize=16)
        if smooth > 1.0:
            try:
                from scipy.signal import butter, sosfiltfilt 
                sos = butter(6, 1.0/smooth, output='sos')
                filter = True
            except Exception as e:
                print("Scipy not installed - falling back to np polynomial fit")
                print(e)
                filter = False

        for i in range(1, len(ecc[0])):
            # Plot eccentricity and incination against time 
            # Optionally smooth the graphs with a low pass filter
            if smooth > 1.0:
                if filter:
                    ecc[:,i] = sosfiltfilt(sos, ecc[:,i])
                    inc[:,i] = sosfiltfilt(sos, inc[:,i])
                else:
                    pol = np.polynomial.chebyshev.chebfit(time, ecc[:,i], smooth)
                    ecc[:,i] = np.polynomial.chebyshev.chebval(time, pol)
                    pol = np.polynomial.chebyshev.chebfit(time, inc[:,i], smooth)
                    inc[:,i] = np.polynomial.chebyshev.chebval(time, pol)

            ax0.plot(time, ecc[:,i], label=names[i] if names is not None else None)
            ax1.plot(time, inc[:,i])

        ax0.set_ylabel('$e_i$')
        ax1.set_ylabel('$I_i$/deg')

        if units is not None:
            x_lab = '$t$/{}'.format(units)
        else:
            x_lab = 'Time'
        ax1.set_xlabel(x_lab)

        # Request a legend and display
        if names is not None:
            legend = fig.legend()
            for line in legend.get_lines():
                line.set_linewidth(4.0)
