import numpy as np
from .integrator import Integrator
from .ode import ODE

__integrator__ = 'Euler'


class Euler(Integrator):

    def __init__(self):
        super(Euler, self).__init__()
        self.__initialized = False

    def integrate(self, to_time=None):
        if self.__initialized is False:
            self.initialize()
            self.__initialized = True
        if to_time is not None:
            self.t_end = to_time

        # Allocate dense output
        npts = int(np.floor((self.t_end - self.t_start) / self.h) + 1)

        # Initial state
        x = np.concatenate((self._particles.positions, self._particles.velocities))
        # Vector of times
        sol_time = np.linspace(self.t_start, self.t_start + self.h * (npts - 1), npts)
        energy_init = self.calculate_energy()
        # Launch integration
        count = 1
        for t in sol_time[count:]:
            dxdt = ODE.ode_n_body_first_order(x, self.CONST_G, self._particles.masses)
            # Advance step
            x = x + dxdt * self.h

            # Store solution
            self._particles.positions = x[0:self._particles.N * 3]
            self._particles.velocities = x[self._particles.N * 3:]
            count += 1
            self._t = t
            self.store_state()
            if (count - 1) % self.write_update == 0:
                energy = self.calculate_energy()
                print(('t = %f, E/E0 = %g' % (self.t, np.abs(energy - energy_init) / energy_init)))
        self.buf.close()
        return 0

    def calculate_energy(self):
        # To avoid going to the base class implementation, which assumes a ctypes version may be available
        return self._particles.energy
