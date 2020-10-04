import h5py
import math
import numpy 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cycler import cycler
from ABIE import snapshot_convert
from ABIE import Tools

class Display:
    def __init__(self):
        self.next_figure = 1
        self.input_file = None
        self.converted = None


    def _get_figure(self):
        # Creates a new figure
        fig = plt.figure(self.next_figure)
        self.next_figure += 1
        return fig


    def _convert(self, input_file):
        # Caches the converted file
        if self.input_file == input_file:
            return self.converted
        else:
            self.input_file = input_file
            self.converted = snapshot_convert(input_file)
            return self.converted

    @staticmethod
    def _package(x,y,z,vx,vy,vz):
        ticks, particles = x.shape
        states = numpy.empty(shape=(ticks, 6*particles))

        # Package the positions for all particles, then the velocities for all particles for each time
        for t in range(0, ticks):
            for p in range(0, particles):
                states[t, 3*p] = x[t, p]
                states[t, 3*p+1] = y[t, p]
                states[t, 3*p+2] = z[t, p]

            for p in range(0, particles):
                states[t, 3*(particles+p)] = vx[t, p]
                states[t, 3*(particles+p)+1] = vy[t, p]
                states[t, 3*(particles+p)+2] = vz[t, p]

        return states


    def show(self):
        # Show all of the figures
        if self.next_figure > 1:
            plt.show()
    

    def display_3d_data(self, input_file, hash2names=None, names=None, title="Trajectory", scatter=False, bary=False):
        # flatten the data
        converted = self._convert(input_file)

        if converted:
            h5f = h5py.File(converted[0], 'r')

            x = h5f['/x'][()]     # Get x for the particles
            y = h5f['/y'][()]     # Get y for the particles
            z = h5f['/y'][()]     # Get z for the particles

            vx = h5f['/vx'][()]   # Get x velocity for the particles
            vy = h5f['/vy'][()]   # Get y velocity for the particles
            vz = h5f['/vz'][()]   # Get z velocity for the particles

           # Use the hash (if provided) in favour of the supplied names
            if hash2names is not None:
                names = []
                for hash in h5f['/hash'][(0)]:
                    names.append(hash2names.get(hash,"Unknown"))

            # Package the positions and velocities
            states = Display._package(x, y, z, vx, vy, vz)

            # Convert to heliocentric coords if requested
            if bary:
                states = Tools.move_to_helio(states)

            fig = self._get_figure()
            ax = fig.gca(projection='3d')
            ax.set_title(title)
            ax.prop_cycle : cycler(color='bgrcmyk')
            for i in range(0, len(x[0])): 
                # Plot all the positions for all objects
                if scatter:
                    ax.scatter(states[:,3*i], states[:,3*i+1], states[:,3*i+2], s=0.1, label=names[i] if names is not None else None)
                else:
                    ax.plot(states[:,3*i], states[:,3*i+1], states[:,3*i+2], label=names[i] if names is not None else None)

            # Finished with the data
            h5f.close()

            # Request a legend
            if names is not None:
                ax.legend(markerscale=30 if scatter else 1)


    def display_2d_data(self, input_file, hash2names=None, names=None, title="Trajectory", scatter=False, bary=False):
        # flatten the data
        converted = self._convert(input_file)

        if converted:
            h5f = h5py.File(converted[0], 'r')

            x = h5f['/x'][()]     # Get x position for the particles
            y = h5f['/y'][()]     # Get y position for the particles
            z = h5f['/z'][()]     # Get z position for the particles

            vx = h5f['/vx'][()]   # Get x velocity for the particles
            vy = h5f['/vy'][()]   # Get y velocity for the particles
            vz = h5f['/vz'][()]   # Get z velocity for the particles

            # Use the hash (if provided) in favour of the supplied names
            if hash2names is not None:
                names = []
                for hash in h5f['/hash'][(0)]:
                    names.append(hash2names.get(hash,"Unknown"))
            
            # Package the positions and velocities
            states = Display._package(x, y, z, vx, vy, vz)

            # Convert to heliocentric coords if requested
            if bary:
                states = Tools.move_to_helio(states)

            self._get_figure()
            plt.subplot(111)
            ax = plt.gca()
            ax.set_title(title)
            ax.prop_cycle : cycler(color='bgrcmyk')
            for i in range(0, len(x[0])): 
                # Plot x and y positions for all objects
                if scatter:
                    ax.scatter(states[:,3*i], states[:,3*i+1], s=0.1, label=names[i] if names is not None else None)
                else:
                    ax.plot(states[:,3*i], states[:,3*i+1], label=names[i] if names is not None else None)

            ax.axis('equal')
            # Finished with the data
            h5f.close()

            # Request a legend and display
            if names is not None:
                ax.legend(markerscale=30 if scatter else 1)

    def display_energy_delta(self, input_file, title="Energy Delta", g=1.0, divisor=1.0, units=None, helio=False):
        converted = self._convert(input_file)

        if converted:
            h5f = h5py.File(converted[0], 'r')

            x = h5f['/x'][()]     # Get x position for the particles
            y = h5f['/y'][()]     # Get y position for the particles
            z = h5f['/z'][()]     # Get z position for the particles

            vx = h5f['/vx'][()]   # Get x velocity for the particles
            vy = h5f['/vy'][()]   # Get y velocity for the particles
            vz = h5f['/vz'][()]   # Get z velocity for the particles

            time = h5f['/time'][()]   # Get time
            mass = h5f['/mass'][(0)]

            # Convert the time
            time = time / divisor

            # Package the positions and velocities
            states = Display._package(x, y, z, vx, vy, vz)

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
            ax.set_ylabel('$\Delta$')
            if units is not None:
                x_lab = '$t$/{}'.format(units)
            else:
                x_lab = 'Time'
            ax.set_xlabel(x_lab)


    def display_2d_e_and_i(self, input_file, hash2names=None, names=None, smooth = False, title="$e$ and $I$", divisor=1.0, units=None):
        # flatten the data
        converted = self._convert(input_file)

        if converted:
            h5f = h5py.File(converted[0], 'r')

            ecc = h5f['/ecc'][()]     # Get eccentricity
            inc = h5f['/inc'][()]     # Get inclinations 
            time = h5f['/time'][()]   # Get time - in days

            # Convert the time to millions of years
            time = time /divisor

            # Convert the inclination to degrees
            inc = inc * 180 / math.pi

            # Use the hash (if provided) in favour of the supplied names
            if hash2names is not None:
                names = []
                for hash in h5f['/hash'][(0)]:
                    names.append(hash2names.get(hash,"Unknown"))

            fig = self._get_figure()
            ax0 = plt.subplot(211)
            ax1 = plt.subplot(212, sharex = ax0)
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

                ax0.plot(time[start:], ecc[start:,i], label=names[i] if names is not None else None)
                ax1.plot(time[start:], inc[start:,i])

            ax0.set_ylabel('$e_i$')
            ax1.set_ylabel('$I_i$/deg')

            if units is not None:
                x_lab = '$t$/{}'.format(units)
            else:
                x_lab = 'Time'
            ax1.set_xlabel(x_lab)

            # Finished with the data
            h5f.close()

            # Request a legend and display
            if names is not None:
                legend = fig.legend()
                for line in legend.get_lines():
                    line.set_linewidth(4.0)
