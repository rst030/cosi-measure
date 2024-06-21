import numpy as np

class shimming_magnet():
    '''shimming magnet object. cube.
    dipolar approximation of the magnetic field.
    position and rtation are given.
    field is calculated in a given area.'''
    
    bRem = 1.35 # T remanence field of shim magnets
    magSizeOuter = 6 # mm - size of shim magnets
    mu                  = 1e-7
    magSizeOuter        = 6*1e-3        #size of shim magnets
    
    
    
    position = [0,0,0]
    dipole = [0,1,0] #by default moment is pointing along y
    rotation_yz = 0 # rotation of the shim magnet in the ZY plane
    
    def __init__(self,position=[0,0,0], dipole_moment = 1, rotation_yz = 0):
        self.position = position
        self.rotation_yz = rotation_yz
        self.dipole = [self.dipole[0]*dipole_moment,self.dipole[1]*dipole_moment*np.cos(rotation_yz),self.dipole[1]*dipole_moment*np.sin(rotation_yz)]
        
        print('magnet created, dipole points to ',self.dipole)
    
    
    def render_field(self,grid):
        print('the magnet has a position and the dipole moment. calculate the components of the field on the grid.')
    
    def singleMagnet(position, dipoleMoment, simDimensions, resolution, plotFields=False):
        #based on the dipole approximation
        #create mesh coordinates
    
        X = np.linspace(-simDimensions[0]/2 - position[0], simDimensions[0]/2 - position[0] , int(simDimensions[0]*resolution)+1, dtype=np.float32)
        Y = np.linspace(-simDimensions[1]/2 - position[1], simDimensions[1]/2 - position[1] , int(simDimensions[1]*resolution)+1, dtype=np.float32)
        Z = np.linspace(-simDimensions[2]/2 - position[2], simDimensions[2]/2 - position[2] , int(simDimensions[2]*resolution)+1, dtype=np.float32)
        
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
        B0[:,:,:,2] = np.divide(np.multiply(z,vec_dot_dip),rvec**5) - np.divide(mz,rvec**3)
        

        if plotFields:
            # plot the field of one shim magnet in the YZ plane 
            from matplotlib import pyplot as plt
            ax = plt.figure().add_subplot(projection='3d')
            ax.plot_surface(y2d,z2d,B0[int(len(X)/2),:,:,2],cmap='coolwarm',clim=[-1e-5,1e-5])
            ax.set_xlabel('Y')
            ax.set_ylabel('Z')
            
            ax.set_title('z component of field of a single magnet YZ plane')

            plt.show()
        
        

        return B0