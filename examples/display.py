import numpy as np
import matplotlib.pyplot as plt
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


    def display_3d_data(self, hash2names=None, names=None, title="Trajectory", scatter=False, to_helio=False, equal=False, last=False):
        # Use the hash (if provided) in favour of the supplied names
        if hash2names is not None:
            names = self.h5.hash_to_names(hash2names)

        # Convert to heliocentric coords if requested
        states = self.h5.get_state_heliocentric() if to_helio else self.h5.get_state()

        fig = self._get_figure()
        ax = fig.gca(projection='3d')
        ax.set_title(title)

        time, width = states.shape
        if not last:
            for i in range(0, width // 6): 
                # Plot all the positions for all objects
                if scatter:
                    ax.scatter(states[:,3*i], states[:,3*i+1], states[:,3*i+2], s=0.1, label=names[i] if names is not None else None)
                else:
                    ax.plot(states[:,3*i], states[:,3*i+1], states[:,3*i+2], label=names[i] if names is not None else None)
        else:
            # Plot only the last positions, which implies scatter. Ignore names
            ax.scatter(states[time-1,0:width//2:3], states[time-1,1:width//2+1:3], states[time-1,2:width//2+2:3], s=0.1)

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


    def display_2d_data(self, hash2names=None, names=None, title="Trajectory", scatter=False, to_helio=False, equal=True, box=None, last=False):
        # Use the hash (if provided) in favour of the supplied names
        if hash2names is not None:
            names = self.h5.hash_to_names(hash2names)

        # Convert to heliocentric coords if requested
        states = self.h5.get_state_heliocentric() if to_helio else self.h5.get_state()

        self._get_figure()
        ax = plt.subplot(111)
        ax.set_title(title)

        time, width = states.shape
        if not last:
            for i in range(0, width // 6): 
                # Plot all the positions for all objects
                if scatter:
                    ax.scatter(states[:,3*i], states[:,3*i+1], s=0.1, label=names[i] if names is not None else None)
                else:
                    ax.plot(states[:,3*i], states[:,3*i+1], label=names[i] if names is not None else None)

        else:
            # Plot only the last positions, which implies scatter. Ignore names
            ax.scatter(states[time-1,0:width//2:3], states[time-1,1:width//2+1:3], s=0.1)

        # Configure axes
        if box is not None:
            ax.set_xlim(box[0], box[1])
            ax.set_ylim(box[2], box[3])
        elif equal:
            ax.axis('equal')

        # Request a legend and display
        if names is not None:
            ax.legend(markerscale=30 if scatter else 1)


    def display_semi_major(self, hash2names=None, names=None, title="Trajectory", divisor=1.0, units=None, ignore_first=True):
        # Get time and convert
        time = self.h5.get_time()
        time = time / divisor

        # Use the hash (if provided) in favour of the supplied names
        if hash2names is not None:
            names = self.h5.hash_to_names(hash2names)

        semi = self.h5.get_semi_major()
        _, particles = semi.shape

        fig = self._get_figure()
        ax = plt.subplot(111)
        ax.set_title(title)

        # Plot semi-major axis against time for all particles, optionally excluding the first particle
        for i in range(1 if ignore_first else 0, particles): 
            ax.plot(time, (semi[:,i]), label=names[i] if names is not None else None)
        
        if ignore_first:
            y_lab = "Distance from {} /km"
            ax.set_ylabel(y_lab.format(names[0] if names is not None else 'Centre'))
        else:
            ax.set_ylabel("Distance from centre /km")

        # Display the angle between bodies 2 and 3 on the same axis
        if particles > 2:
            ax2 = ax.twinx()
            ang = self.h5.get_angle()
            col = 'gray'

            ax2.tick_params(axis='y', labelcolor=col)
            ax2.set_ylabel('Angle between objects', color=col)
            ax2.yaxis.set_major_formatter('${x:n}\degree$')
            ax2.plot(time, ang, color=col, label="Angle")
            ax2.legend(loc=1)

        if units is not None:
            x_lab = '$t$/{}'.format(units)
        else:
            x_lab = 'Time'
        ax.set_xlabel(x_lab)

        # Request a legend
        if names is not None:
            ax.legend(loc=2)

        fig.tight_layout()


    def display_energy_delta(self, title="Energy Delta", G=1.0, divisor=1.0, units=None, to_bary=False, sim=None):
        # Get time
        time = self.h5.get_time()
        
        # Convert the time
        time = time / divisor

        # Get energy against time
        energy = self.h5.compute_energy(G, to_bary, sim)

        # Plot energy delta
        self._get_figure()
        ax = plt.subplot(111)
        ax.set_title(title)

        # Plot delta against time
        ax.plot(time, (energy - energy[0]) / energy[0])

        ax.set_ylabel(r'$\Delta\/Energy$')
        if units is not None:
            x_lab = '$t$/{}'.format(units)
        else:
            x_lab = 'Time'
        ax.set_xlabel(x_lab)


    def display_histogram(self, data, b_start, b_end, num=10, names=None, title="Distribution", units=None):

        fig=self._get_figure()
        ax = plt.subplot(111)
        ax.set_title(title)

        # plot the data
        ax.hist(data, num, (b_start, b_end), histtype='bar', rwidth=1.0, label=names)
        ax.set_ylabel(title)
        if units is not None:
            x_lab = '{}'.format(units)
        else:
            x_lab = 'AU'
        ax.set_xlabel(x_lab)

        # Request a legend
        if names is not None:
            legend = ax.legend()


    def display_2d_e_and_i(self, hash2names=None, names=None, smooth=1, title="$e$ and $I$", divisor=1.0, units=None):
        time = self.h5.get_time()
        ecc = self.h5.get_eccentricity()
        inc = self.h5.get_inclination()

        # Convert the time 
        time = time /divisor

        # Convert the inclination to degrees
        inc = np.degrees(inc)

        # Use the hash (if provided) in favour of the supplied names
        if hash2names is not None:
            names = self.h5.hash_to_names(hash2names)

        fig = self._get_figure()
        ax0 = plt.subplot(211)
        ax1 = plt.subplot(212, sharex = ax0)
        fig.subplots_adjust(hspace=0)

        fig.suptitle(title, fontsize=16)
        if smooth > 1:
            try:
                from scipy.signal import butter, sosfiltfilt 
                sos = butter(6, 1.0/smooth, output='sos')
                filter = True
            except Exception as e:
                print("Scipy not installed - falling back to np chebyshev fit")
                print(e)
                filter = False

        for i in range(1, len(ecc[0])):
            # Plot eccentricity and incination against time 
            # Optionally smooth the graphs with a low pass filter
            if smooth > 1:
                if filter:
                    ecc[:,i] = sosfiltfilt(sos, ecc[:,i])
                    inc[:,i] = sosfiltfilt(sos, inc[:,i])
                else:
                    pol = np.polynomial.chebyshev.chebfit(time, ecc[:,i], int(smooth))
                    ecc[:,i] = np.polynomial.chebyshev.chebval(time, pol)
                    pol = np.polynomial.chebyshev.chebfit(time, inc[:,i], int(smooth))
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

        # Request a legend
        if names is not None:
            legend = fig.legend()
            for line in legend.get_lines():
                line.set_linewidth(4.0)


    def display_2d_scatter(self, x, y0, y1=None, names=None, title="$e$ and $I$", equal=False, x_units=None, y_units=None):
        fig = self._get_figure()
        if y1 is not None:
            ax0 = plt.subplot(211)
            ax1 = plt.subplot(212, sharex = ax0)
            fig.subplots_adjust(hspace=0)
        else:
            ax0=plt.subplot(111)

        fig.suptitle(title, fontsize=16)

        _, plots = x.shape

        for i in range(plots):
            # Plot sets of data
            ax0.scatter(x[:,i], y0[:,i], s=0.2, label=names[i] if names is not None else None)

            if y1 is not None:
                ax1.scatter(x[:,i], y1[:,i], s=0.2)
        
        if y_units is not None:
            y_lab = '{}'.format(y_units)
        else:
            y_lab = '$e_i$'
        ax0.set_ylabel(y_lab)

        if y1 is not None:
            ax1.set_ylabel('$I_i$/deg')

        if x_units is not None:
            x_lab = '{}'.format(x_units)
        else:
            x_lab = 'Time'
        ax0.set_xlabel(x_lab)

        if equal:
            ax0.axis('equal')

        # Request a legend
        if names is not None:
            ax0.legend(markerscale=20)
