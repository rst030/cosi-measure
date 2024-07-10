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
        self.dipole_vector = np.array([0,np.cos(rotation_yz),np.sin(rotation_yz)])*self.mu*self.dip_mom # dipole moment in YZ plane!, initially - along Y
        
        print('magnet created, dipole points to ',self.dipole_vector/np.linalg.norm(self.dipole_vector))

    def update_rotation(self,rotation_yz):
        self.dipole_vector = np.array([0,np.cos(rotation_yz),np.sin(rotation_yz)])*self.mu*self.dip_mom # dipole moment in YZ plane!, initially - along Y
        print(self.dipole_vector)
        
    def render_field(self,grid):
        '''calculate the magnetic field of the magnet on the coordinate grin grid, leave only values within sphere of d=dsv mm'''

        self.update_rotation(self.rotation_yz)


        self.dip_mom = self.magnetization(self.bRem,self.magSizeOuter) 
        mu = 1e-7
        dip_vec = self.dipole_vector#mu*np.array([0,dip_mom,0]) # dipole moment in YZ plane!, initially - along Y
        
        x = grid[0]*1e-3-self.position[0]
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        z = grid[1]*1e-3-self.position[1]
        y = grid[2]*1e-3-self.position[2]
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        mx = dip_vec[0]
        my = dip_vec[1]
        mz = dip_vec[2]

        rvec = np.sqrt(np.square(x)+np.square(y)+np.square(z))
        vec_dot_dip = 3*(np.multiply(mx,x) + np.multiply(my,y) + np.multiply(mz,z))

        B0 = np.zeros((np.shape(x)+(3,)),dtype = np.float32)


        B0[:,:,:,0] = np.divide(np.multiply(x,vec_dot_dip),rvec**5)# - np.divide(mx,rvec**3)
        B0[:,:,:,1] = np.divide(np.multiply(y,vec_dot_dip),rvec**5) - np.divide(my,rvec**3)
        B0[:,:,:,2] = np.divide(np.multiply(z,vec_dot_dip),rvec**5) - np.divide(mz,rvec**3)

        self.B0 = B0

        return B0

    def rotate_field(self,rotation_yz):
        # rotation by rotation_yz from Y to Z

        self.rotation_yz == rotation_yz # aka flag
            
        By = self.B0[:,:,:,1]
        Bz = self.B0[:,:,:,2]

        self.Brot = By*np.sin(rotation_yz)+Bz*np.cos(rotation_yz)
        return self.Brot

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
    
    
    def singleMagnet(self,position,grid, plotFields=False):
        dip_mom = self.magnetization(self.bRem,self.magSizeOuter) 
        mu = 1e-7
        dip_vec = mu*np.array([0,dip_mom,0]) # dipole moment in YZ plane!, initially - along Y
                
        resolution = abs(1e3/(grid[0][1][0][0]-grid[0][0][0][0]))
        
        simDimensions=[0,0,0]
        simDimensions[0] = np.max((grid[0])-np.min(grid[0]))*1e-3
        simDimensions[1] = np.max((grid[1])-np.min(grid[1]))*1e-3
        simDimensions[2] = np.max((grid[2])-np.min(grid[2]))*1e-3

        X = np.linspace(-simDimensions[0]/2-position[0], simDimensions[0]/2-position[0], int(simDimensions[0]*resolution)+2, dtype=np.float32)
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        Z = np.linspace(-simDimensions[1]/2-position[1], simDimensions[1]/2-position[1], int(simDimensions[1]*resolution)+2, dtype=np.float32)
        Y = np.linspace(-simDimensions[2]/2-position[2], simDimensions[2]/2-position[2], int(simDimensions[2]*resolution)+2, dtype=np.float32)
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!        
        print('x vector length in single magnet simulation: ',len(X))
        
        x,y,z = np.meshgrid(X,Y,Z,indexing='ij')
        y2d, z2d = np.meshgrid(Y,Z,indexing='ij')
        

        mx = dip_vec[0]
        my = dip_vec[1]
        mz = dip_vec[2]

        rvec = np.sqrt(np.square(x)+np.square(y)+np.square(z))
        vec_dot_dip = 3*(np.multiply(mx,x) + np.multiply(my,y) + np.multiply(mz,z))

        B0 = np.zeros((np.shape(x)+(3,)),dtype = np.float32)


        B0[:,:,:,0] = np.divide(np.multiply(x,vec_dot_dip),rvec**5)# - np.divide(mx,rvec**3)
        B0[:,:,:,1] = np.divide(np.multiply(y,vec_dot_dip),rvec**5) - np.divide(my,rvec**3)
        B0[:,:,:,2] = np.divide(np.multiply(z,vec_dot_dip),rvec**5) - np.divide(mz,rvec**3)

        self.B0 = B0
        
        if plotFields:
            # plot the field of one shim magnet in the YZ plane 
            from matplotlib import pyplot as plt
            ax = plt.figure().add_subplot()#projection='3d')
            ax.contourf(y2d,z2d,B0[int(len(X)/2),:,:,2],cmap='coolwarm')#,clim=[-1e-5,1e-5])
            ax.set_xlabel('Y')
            ax.set_ylabel('Z')
            
            ax.set_title('FAIR z component of field of a single magnet YZ plane')

            plt.show()
        

        return B0


    
    def OLD_singleMagnet(self,position, dipoleMoment, simDimensions, resolution, plotFields=False):
        #based on the dipole approximation
        #create mesh coordinates
        print('HALLO?!')
        print(position)
    
        X = np.linspace(-simDimensions[0]/2 - position[0], simDimensions[0]/2 - position[0] , int(simDimensions[0]*resolution)+1, dtype=np.float32)
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        Z = np.linspace(-simDimensions[1]/2 - position[1], simDimensions[1]/2 - position[1] , int(simDimensions[1]*resolution)+1, dtype=np.float32)
        Y = np.linspace(-simDimensions[2]/2 - position[2], simDimensions[2]/2 - position[2] , int(simDimensions[2]*resolution)+1, dtype=np.float32)
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!        
        print('x vector length in single magnet simulation: ',len(X))
        
        x,y,z = np.meshgrid(X,Y,Z,indexing='ij')
        y2d, z2d = np.meshgrid(Y,Z,indexing='ij')

        print('calculating field of one shim magnet at ',position[0],position[1],position[2])

        #vec_dot_dip = 3*(x*dipoleMoment[0] + y*dipoleMoment[1]) # was in Tom's script, where the shim magnet was in the xy plane
        
        mx = dipoleMoment[0]
        my = dipoleMoment[1]
        mz = dipoleMoment[2]
        rvec = np.sqrt(np.square(x)+np.square(y)+np.square(z))
        vec_dot_dip = 3*(np.multiply(mx,x) + np.multiply(my,y) + np.multiply(mz,z))
        

        B0 = np.zeros(np.shape(x)+(3,), dtype=np.float32)

        B0[:,:,:,0] = np.divide(np.multiply(x,vec_dot_dip),rvec**5)# - np.divide(mx,rvec**3)
        B0[:,:,:,1] = np.divide(np.multiply(y,vec_dot_dip),rvec**5) - np.divide(my,rvec**3)
        B0[:,:,:,2] = np.divide(np.multiply(z,vec_dot_dip),rvec**5) - np.divide(mz,rvec**3) #1/(x**2+y**2+z**2)#

        

        if plotFields:
            # plot the field of one shim magnet in the YZ plane 
            from matplotlib import pyplot as plt
            ax = plt.figure().add_subplot()#projection='3d')
            ax.contourf(y2d,z2d,B0[int(len(X)/2),:,:,2],cmap='coolwarm')#,clim=[-1e-5,1e-5])
            ax.set_xlabel('Y')
            ax.set_ylabel('Z')
            
            ax.set_title('z component of field of a single magnet YZ plane')

            plt.show()
        
        self.grid = [x,y,z]
        self.field = B0

        return B0
