import h5py
import math
import numpy 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cycler import cycler
from ABIE import snapshot_convert
from ABIE import Tools

class Display:
    def __init__(self, input_file=None):
        self.next_figure = 1
        self.h5f = None
        if input_file:
            self.set_data(input_file)


    def _get_figure(self):
        # Creates a new figure
        fig = plt.figure(self.next_figure)
        self.next_figure += 1
        return fig


    def set_data(self, input_file):
        # Close previous file, if it exists
        self.close()

        # Caches the h5 data
        converted = snapshot_convert(input_file)
        if converted:
            self.h5f = h5py.File(converted[0], 'r')
        else:
            self.h5f = None
    
    def close(self):
        # If a file is open, then close it
        if self.h5f:
            self.h5f.close()
            self.h5f = None
            

    def _package(self):
        # Package the positions and velocities
        ticks, particles = self.h5f['/x'][()].shape
        return numpy.concatenate((numpy.dstack((self.h5f['/x'][()], self.h5f['/y'][()], self.h5f['/z'][()])),
                                  numpy.dstack((self.h5f['/vx'][()], self.h5f['/vy'][()], self.h5f['/vz'][()]))), 
                                  axis=1).reshape(ticks, 6*particles) 


    def show(self):
        # Close the data set
        self.close()

        # Show all of the figures
        if self.next_figure > 1:
            plt.show()


    def display_3d_data(self, hash2names=None, names=None, title="Trajectory", scatter=False, bary=False, equal=False):
        # Reform the data for possible converstion
        states = self._package()

        # Use the hash (if provided) in favour of the supplied names
        if hash2names is not None:
            names = []
            for hash in self.h5f['/hash'][(0)]:
                names.append(hash2names.get(hash,"Unknown"))

        # Convert to heliocentric coords if requested
        if bary:
            states = Tools.move_to_helio(states)

        fig = self._get_figure()
        ax = fig.gca(projection='3d')
        ax.set_title(title)

        ax.prop_cycle : cycler(color='bgrcmyk')
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


    def display_2d_data(self, hash2names=None, names=None, title="Trajectory", scatter=False, bary=False):
        # Reform the data for possible converstion
        states = self._package()

        # Use the hash (if provided) in favour of the supplied names
        if hash2names is not None:
            names = []
            for hash in self.h5f['/hash'][(0)]:
                names.append(hash2names.get(hash,"Unknown"))
        
        # Convert to heliocentric coords if requested
        if bary:
            states = Tools.move_to_helio(states)

        self._get_figure()
        plt.subplot(111)
        ax = plt.gca()
        ax.set_title(title)
        ax.prop_cycle : cycler(color='bgrcmyk')
        _, width = states.shape
        for i in range(0, width // 6): 
            # Plot x and y positions for all objects
            if scatter:
                ax.scatter(states[:,3*i], states[:,3*i+1], s=0.1, label=names[i] if names is not None else None)
            else:
                ax.plot(states[:,3*i], states[:,3*i+1], label=names[i] if names is not None else None)

        ax.axis('equal')

        # Request a legend and display
        if names is not None:
            ax.legend(markerscale=30 if scatter else 1)


    def display_energy_delta(self, title="Energy Delta", g=1.0, divisor=1.0, units=None, helio=False):
        # Reform the data for converstion
        states = self._package()

        time = self.h5f['/time'][()]    # Get time
        mass = self.h5f['/mass'][()]    # Get mass - note mass can change if particles combine or are ejected

        # Cater for particle combinations
        mass[numpy.isnan(mass)]=0.0
        states[numpy.isnan(states)]=0.0
        
        # Convert the time
        time = time / divisor

        if helio:
            # Transform to barycentric coords
            states = Tools.helio2bary(states, mass)

        # Calculate the total energy against time
        energy = Tools.compute_energy(states, mass, g)

        # Plot energy delta
        self._get_figure()
        ax = plt.subplot(111)
        ax.set_title(title)

        # Plot delta against time
        ax.plot(time, (energy - energy[0]) / energy[0])
        # Finished with the data

        ax.set_ylabel('$\Delta$')
        if units is not None:
            x_lab = '$t$/{}'.format(units)
        else:
            x_lab = 'Time'
        ax.set_xlabel(x_lab)


    def display_2d_e_and_i(self, hash2names=None, names=None, smooth = False, title="$e$ and $I$", divisor=1.0, units=None):
        ecc = self.h5f['/ecc'][()]     # Get eccentricity
        inc = self.h5f['/inc'][()]     # Get inclinations 
        time = self.h5f['/time'][()]   # Get time - in days

        # Convert the time to millions of years
        time = time /divisor

        # Convert the inclination to degrees
        inc = inc * 180 / math.pi

        # Use the hash (if provided) in favour of the supplied names
        if hash2names is not None:
            names = []
            for hash in self.h5f['/hash'][(0)]:
                names.append(hash2names.get(hash,"Unknown"))

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
                print("Scipy not installed - falling back to numpy polynomial fit")
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
                    pol = numpy.polynomial.chebyshev.chebfit(time, ecc[:,i], n)
                    ecc[:,i] = numpy.polynomial.chebyshev.chebval(time, pol)
                    pol = numpy.polynomial.chebyshev.chebfit(time, inc[:,i], n)
                    inc[:,i] = numpy.polynomial.chebyshev.chebval(time, pol)

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
