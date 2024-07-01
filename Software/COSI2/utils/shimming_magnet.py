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
        self.dip_mom = self.magnetization(self.bRem, self.magSizeOuter) 
        self.dipole_vector = self.mu*self.dip_mom*np.array([0,1,0]) # dipole moment in YZ plane!, initially - along Y
        
        print('magnet created, dipole points to ',self.dipole_vector/np.linalg.norm(self.dipole_vector))

    def update_rotation(self,rotation_yz):
        self.dipole_vector = self.mu*self.dip_mom*np.array([0,np.cos(rotation_yz),np.sin(rotation_yz)]) # dipole moment in YZ plane!, initially - along Y
        
    def render_field(self,grid=None):
        '''calculate the magnetic field of the magnet on the coordinate grin grid, leave only values within sphere of d=dsv mm'''
        # grid is a np.meshgrid with ij indexing, so x,y and z.
        
        # if no updated rotation, the field is always pointing in the y direction! render the field in all cube
        
        #print('expensive rendering dip to',self.rotation_yz)
        self.update_rotation(self.rotation_yz)

        mx = self.dipole_vector[0]
        my = self.dipole_vector[1]
        mz = self.dipole_vector[2]

        if grid is not None:

            self.x = grid[0]*1e-3 - self.position[0] # [m]
            self.y = grid[1]*1e-3 - self.position[1] # [m]
            self.z = grid[2]*1e-3 - self.position[2] # [m]
        
            self.rvec = np.sqrt(np.square(self.x)+np.square(self.y)+np.square(self.z))
            self.rvec5 = self.rvec**5
            self.rvec3 = self.rvec**3

            vec_dot_dip = 3*(0 + np.multiply(my,self.y) + np.multiply(mz,self.z)) # mx is 0 always, magnets rotate in yz plane

            self.grid=grid
            self.B0 = np.zeros(np.shape(self.x)+(3,), dtype=np.float32)
            self.B0[:,:,:,0] = 0#np.divide(np.multiply(x,vec_dot_dip),rvec**5)# - np.divide(mx,rvec**3)
            self.B0[:,:,:,1] = 1e3*np.divide(np.multiply(self.y,vec_dot_dip),self.rvec5) - np.divide(my,self.rvec3)
            self.B0[:,:,:,2] = 1e3*np.divide(np.multiply(self.z,vec_dot_dip),self.rvec5) - np.divide(mz,self.rvec3)
        else:
            vec_dot_dip = 3*(0 + np.multiply(my,self.y) + np.multiply(mz,self.z)) # mx is 0 always, magnets rotate in yz plane
            self.B0[:,:,:,0] = 0#np.divide(np.multiply(x,vec_dot_dip),rvec**5)# - np.divide(mx,rvec**3)
            self.B0[:,:,:,1] = 1e3*np.divide(np.multiply(self.y,vec_dot_dip),self.rvec5) - np.divide(my,self.rvec3)
            self.B0[:,:,:,2] = 1e3*np.divide(np.multiply(self.z,vec_dot_dip),self.rvec5) - np.divide(mz,self.rvec3)
            

        print('expensive render')
        return self.B0

            #self.B0 *= 7 # temp. 1 strong instead of 7 weak

    def rotate_field(self,rotation_yz):
        # rotation by rotation_yz from Y to Z

        if self.rotation_yz == 0: # safety
            
            By = self.B0[:,:,:,1]
            Bz = self.B0[:,:,:,2]

            self.B0_rotated_Z = By*np.sin(rotation_yz)+Bz*np.cos(rotation_yz)
            #print(rotation_yz)


    def singleMagnetTom(self,X,Y,Z):
        
        position = self.position

        X = X*1e-3 - position[0]#np.linspace(-simDimensions[0]/2 - position[0], simDimensions[0]/2 - position[0] , int(simDimensions[0]*resolution)+1, dtype=np.float32)
        Y = Y*1e-3 - position[1]#np.linspace(-simDimensions[1]/2 - position[1], simDimensions[1]/2 - position[1] , int(simDimensions[1]*resolution)+1, dtype=np.float32)
        Z = Z*1e-3 - position[2]#]np.linspace(-simDimensions[2]/2 - position[2], simDimensions[2]/2 - position[2] , int(simDimensions[2]*resolution)+1, dtype=np.float32)
        
        print('x vector length in single magnet simulation: ',len(X))
    
        x,y,z = np.meshgrid(X,Y,Z,indexing='ij')

        mx = self.dipole_vector[0]
        my = self.dipole_vector[1]
        mz = self.dipole_vector[2]

        rvec = np.sqrt(np.square(x)+np.square(y)+np.square(z))
        vec_dot_dip = 3*(np.multiply(mx,x) + np.multiply(my,y) + np.multiply(mz,z))
    

        B0 = np.zeros(np.shape(x)+(3,), dtype=np.float32)

        B0[:,:,:,0] = 0#np.divide(np.multiply(x,vec_dot_dip),rvec**5)# - np.divide(mx,rvec**3)
        B0[:,:,:,1] = np.divide(np.multiply(y,vec_dot_dip),rvec**5) - np.divide(my,rvec**3)
        B0[:,:,:,2] = np.divide(np.multiply(z,vec_dot_dip),rvec**5) - np.divide(mz,rvec**3)

        B0 *= 1e3

        return B0



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