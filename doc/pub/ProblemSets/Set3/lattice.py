"""
Lattice classes and functions
"""

import numpy as np
from scipy.signal import convolve2d

# The Hamiltonian constant strength
# can flip this to -1 to get an anti-ferromagnetic hamiltonian
H_CONSTANT = 1.


class Lattice(object):
    """
    Convenience class so you can do stuff like
    >>> lat = Lattice(N=10, T=5.0)
    >>> print (lat.lattice)
    >>> print (lat.get_energy())
    >>> for _ in range(100): lat.step()
    >>> print (lat.lattice)
    >>> print (lat.get_energy())

    easily.
    """

    def __init__(self, N=10, T=1.0):
        """
        Initializes the lattice. This function is called automatically
        when a new Lattice object is initialized with
        
        >>> lat = Lattice(N=10, T=5.0)
        
        Args:
            N (int): an integer specifying the size of the lattice
            T (float): the temperature of the system
        """
        self.N = N
        self.T = T
        self.lattice = None
        self.neighbor_filter = np.array([
            [0,1,0],
            [1,0,1],
            [0,1,0]
            ])

        self.initialize()

    def initialize(self):
        """
        Initializes lattice points to -1 or 1 randomly
        """
        self.lattice = 2*np.random.randint(2, size=(self.N,self.N))-1

    def step(self):
        """
        Every iteration, select N^2 random points to try a flip attempt.
        A flip attempt consists of checking the change in energy due to a flip.
        If it is negative or U(0,1) less than exp(-E/(k_b*T)), then perform the flip.
        """
        for istep in range(self.N**2):
            ix = np.random.randint(0,self.N)
            iy = np.random.randint(0,self.N)
            s = self.lattice[ix,iy]
            # Note the periodic boundary condition
            neighbor_sum = self.lattice[(ix+1)%self.N,iy] + \
                           self.lattice[(ix-1)%self.N,iy] + \
                           self.lattice[ix,(iy+1)%self.N] + \
                           self.lattice[ix,(iy-1)%self.N]
            # This should be the enegy change for a flipped spin
            dE = H_CONSTANT*2*s*neighbor_sum
            if dE < 0 or np.random.rand() < np.exp(-1.0*dE/self.T):
                s *= -1
            self.lattice[ix,iy] = s

    def get_neighbor_sum_matrix(self):
        """
        While not as efficient as computing the energy once at the beginning
        and adding the dE every step(), this is quite *fast* and elegant.
        Use a 3x3 filter for adjacent neighbors and convolve this across
        the lattice. "wrap" boundary option will handle the periodic BCs.
        This returns a NxN matrix of the sum of neighbor spins for each point.
        """
        return convolve2d(self.lattice,self.neighbor_filter,mode="same",boundary="wrap")

    def get_energy(self):
        """
        We can write the hamiltonian using optimized operations now
        """
        return -H_CONSTANT*(self.lattice*self.get_neighbor_sum_matrix()).sum()
    
    def get_avg_magnetization(self):
        """
        Calculates the average magnetization of the lattice at any point
        """
        return 1.0 * self.lattice.sum() / self.N**2

    def __repr__(self):
        """
        Provides a simple string description of the lattice.
        """
        return str(self.lattice)
