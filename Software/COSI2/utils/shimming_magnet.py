import numpy as np
import scipy.ndimage as cp # for erosion

class shimming_magnet():
    '''shimming magnet object. cube.
    dipolar approximation of the magnetic field.
    position and rtation are given.
    field is calculated in a given area.'''
    
    bRem = 1.35 # [T] remanence field of shim magnets
    mu                  = 1e-7
    magSizeOuter        = 6*1e-3        # [m] size of shim magnets, 
    
    
    
    position = [0,0,0]
    dipole_vector = [0,1,0] #by default moment is pointing along y
    rotation_yz = 0 # rotation of the shim magnet in the ZY plane
    
    def __init__(self,position=[0,0,0], rotation_yz = 0):
        self.position = position
        self.rotation_yz = rotation_yz
        dip_mom = self.magnetization(self.bRem, self.magSizeOuter) 
        self.dipole_vector = self.mu*dip_mom*np.array([0,np.cos(rotation_yz),np.sin(rotation_yz)]) # dipole moment in YZ plane!, initially - along Y
        
        print('magnet created, dipole points to ',self.dipole_vector/np.linalg.norm(self.dipole_vector))

    
    def render_field(self,grid,dsv):
        print('the magnet has a position and the dipole moment. calculate the components of the field on the grid.')
        print('for now place the magnet in 0,0,0')
        # grid is a np.meshgrid with ij indexing, so x,y and z.
        
        

        mx = self.dipole_vector[0]
        my = self.dipole_vector[1]
        mz = self.dipole_vector[2]
        
        x = grid[0]*1e-3 - self.position[0] # [m]
        y = grid[1]*1e-3 - self.position[1] # [m]
        z = grid[2]*1e-3 - self.position[2] # [m]
        
        rvec = np.sqrt(np.square(x)+np.square(y)+np.square(z))
        vec_dot_dip = 3*(np.multiply(mx,x) + np.multiply(my,y) + np.multiply(mz,z))

        B0 = np.zeros(np.shape(x)+(3,), dtype=np.float32)

        B0[:,:,:,0] = 0#np.divide(np.multiply(x,vec_dot_dip),rvec**5)# - np.divide(mx,rvec**3)
        B0[:,:,:,1] = 0#np.divide(np.multiply(y,vec_dot_dip),rvec**5) - np.divide(my,rvec**3)
        B0[:,:,:,2] = np.divide(np.multiply(z,vec_dot_dip),rvec**5) - np.divide(mz,rvec**3)
        
        B0 *= 1e3
        
        #Create a spherical mask for the data
        sphereMask = np.zeros(np.shape(grid[0]), dtype = bool)
        sphereMask[np.square(grid[0]) + np.square(grid[1]) + np.square(grid[2]) <= (dsv/2)**2] = 1 
        sphereMask = sphereMask*(~np.isnan(B0[...,2]))

        # Create a spherical shell mask to consider only data points on the surface of the sphere
        erodedMask = cp.binary_erosion(sphereMask)  # remove the outer surface of the initial spherical mask
        shellMask = np.array(sphereMask^erodedMask, dtype = float)   # create a new mask by looking at the difference between the inital and eroded mask
        shellMask[shellMask == 0] = np.nan  # set points outside mask to 'NaN', works better than setting it to zero for calculating mean fields etc.

        sphereMask = np.asarray(sphereMask, dtype=float)
        sphereMask[sphereMask == 0] = np.nan

        #apply mask to data
        B0_masked = np.multiply(sphereMask, B0[...,2])
        
        self.B0 = B0_masked



    def magnetization(self,bRem, dimensions, shape = 'cube', evalDistance = 1):
        #Use the analytical expression for the z component of a cube magnet to estimate
        #dipole momentstrength for correct scaling. Dipole approximation only valid 
        #far-ish away from magnet, comparison made at 1 meter distance.
        if shape == 'cube':
            b_analytic = (bRem/np.pi) *(np.arctan2(dimensions**2, 2*evalDistance *np.sqrt(4*(evalDistance**2) + 2*(dimensions**2)))-\
                            np.arctan2(dimensions**2, 2*(dimensions + evalDistance)*\
                                    np.sqrt(4*((dimensions+evalDistance)**2)+2*(dimensions**2))))
        
            dip_mom = b_analytic * (dimensions/2 + evalDistance)**3 /(2*self.mu) #strength of the dipole moment
        
        return dip_mom