import h5py
import numpy as np
import matplotlib.pyplot as plt
from ABIE import snapshot_convert
from ABIE import Tools

class H5:
    def __init__(self, input_file):
        self.h5f = None
        if input_file:
            self.set_data(input_file)


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
            

    def get_state(self):
        # Package the positions and velocities
        ticks, particles = self.h5f['/x'][()].shape
        return np.concatenate((np.dstack((self.h5f['/x'][()], self.h5f['/y'][()], self.h5f['/z'][()])),
                               np.dstack((self.h5f['/vx'][()], self.h5f['/vy'][()], self.h5f['/vz'][()]))), 
                               axis=1).reshape(ticks, 6*particles) 


    def get_time(self):
        # Return time
        return self.h5f['/time'][()] 


    def get_mass(self):
        # Return masses
        return self.h5f['/mass'][()] 


    def get_eccentricity(self):
        # Return eccentricity
        return self.h5f['/ecc'][()] 


    def get_inclination(self):
        # Return inclination
        return self.h5f['/inc'][()] 


    def get_state_heliocentric(self):
        return Tools.move_to_helio(self.get_state())


    def get_state_barycentric(self):
        return Tools.move_to_bary(self.get_state(), self.get_mass())


    def compute_energy(self, G, to_bary=False):
        # Transform if necessary
        state = self.get_state_barycentric() if to_bary else self.get_state()

        # Cater for particle combinations
        mass = self.get_mass()
        mass[np.isnan(mass)]=0.0
        state[np.isnan(state)]=0.0
        
        # Calculate the total energy against time
        return Tools.compute_energy(state, mass, G)


    def get_semi_major(self):
        return self.h5f['/semi'][()]


    def get_distance(self, to_helio=False):
        # Transform if necessary
        state = self.get_state_heliocentric() if to_helio else self.get_state()

        # Calculate the distance for each object from the coordinate centre
        ticks, width = state.shape
        dist = np.zeros(shape=(ticks, width // 6))

        for t in range(0, ticks):
            for i in range(0, width // 6):
                # Calculate the magnitude of the position vector
                dist[t, i] = np.linalg.norm((state[t,3*i], state[t,3*i+1], state[t,3*i+2]))

        return dist


    def hash_to_names(self, hash2names):
        # Return a list of names from the supplied map of hashes to names
        names = []
        for hash in self.h5f['/hash'][(0)]:
            names.append(hash2names.get(hash,"Unknown"))
        return names

