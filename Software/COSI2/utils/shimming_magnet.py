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
        self.dipole_vector = np.array([0,np.cos(rotation_yz)*self.mu*self.dip_mom,np.sin(rotation_yz)*self.mu*self.dip_mom]) # dipole moment in YZ plane!, initially - along Y
        #self.dipole_vector = np.array([0,np.cos(rotation_yz),np.sin(rotation_yz)])*self.mu*self.dip_mom # dipole moment in YZ plane!, initially - along Y
        print(self.dipole_vector)
        
    def render_field(self,grid):
        '''calculate the magnetic field of the magnet on the coordinate grin grid, leave only values within sphere of d=dsv mm'''
        print("ROTATION OF EXPENSIVE MAGNET:%.0f"%(self.rotation_yz*180/np.pi))
        self.update_rotation(rotation_yz = self.rotation_yz)

        self.dip_mom = self.magnetization(self.bRem,self.magSizeOuter) 
        print(self.bRem)
        

        dip_vec = self.dipole_vector#mu*np.array([0,dip_mom,0]) # dipole moment in YZ plane!, initially - along Y
         
        z = grid[0]*1e-3-self.position[1] #!!!!!!!!! #240916
        x = grid[1]*1e-3-self.position[0] #!!!!!!!!! #240916
        y = grid[2]*1e-3-self.position[2] #!!!!!!!!! #240913
        

        
        # XX = np.linspace(np.nanmin(grid[0])*1e-3-self.position[0],np.nanmax(grid[0])*1e-3-self.position[0],len(grid[0])) #!!!!!!!! 240913
        # YY = np.linspace(np.nanmin(grid[1])*1e-3-self.position[1],np.nanmax(grid[1])*1e-3-self.position[1],len(grid[1]))#!!!!!!!! 240913
        # ZZ = np.linspace(np.nanmin(grid[2])*1e-3-self.position[2],np.nanmax(grid[2])*1e-3-self.position[2],len(grid[2]))#!!!!!!!! 240913
        
        # self.EXPENSIVEmangetgrid = np.meshgrid(XX,ZZ,YY,indexing='ij') #!!!!!!! 240913 
        # x,y,z = self.EXPENSIVEmangetgrid #!!!!!!! 240913
        
        mx = dip_vec[0]
        my = dip_vec[1]
        mz = dip_vec[2]

        rvec = np.sqrt(np.square(x)+np.square(y)+np.square(z))
        vec_dot_dip = 3*(np.multiply(mx,x) + np.multiply(my,y) + np.multiply(mz,z))

        B0 = np.zeros((np.shape(x)+(3,)),dtype = np.float32)


        B0[:,:,:,0] += np.divide(np.multiply(x,vec_dot_dip),rvec**5) - np.divide(mx,rvec**3)
        B0[:,:,:,1] += np.divide(np.multiply(y,vec_dot_dip),rvec**5) - np.divide(my,rvec**3)
        B0[:,:,:,2] += np.divide(np.multiply(z,vec_dot_dip),rvec**5) - np.divide(mz,rvec**3) 
        self.B0 = B0

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
    
    

    
    def OLD_singleMagnet(self,position, dipoleMoment, simDimensions, resolution, plotFields=False):
        #based on the dipole approximation
        #create mesh coordinates
        
        x_ = np.linspace(-simDimensions[0]/2 - position[0], simDimensions[0]/2 - position[0] , int(simDimensions[0]*resolution)+1, dtype=np.float32)
        y_ = np.linspace(-simDimensions[1]/2 - position[1], simDimensions[1]/2 - position[1] , int(simDimensions[1]*resolution)+1, dtype=np.float32)
        z_ = np.linspace(-simDimensions[2]/2 - position[2], simDimensions[2]/2 - position[2] , int(simDimensions[2]*resolution)+1, dtype=np.float32)
   
        print('x vector length in single magnet simulation: ',len(x_))
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!     
        self.OLDmangetgrid = np.meshgrid(y_,x_,z_,indexing='xy') #!!!!!!! 240916 
        y,x,z = self.OLDmangetgrid #!!!!!!! 240916
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!     
        #y2d, z2d = np.meshgrid(Y,Z,indexing='xy')

        print('computing field of one shim magnet at ',position[0],position[1],position[2])

        
        #mx = dipoleMoment[0]
        #my = dipoleMoment[1]
        #mz = dipoleMoment[2]
        
        #rvec = np.sqrt(np.square(x)+np.square(y)+np.square(z))
        #vec_dot_dip = 3*(np.multiply(mx,x) + np.multiply(my,y) + np.multiply(mz,z))
        

        #B0 = np.zeros(np.shape(x)+(3,), dtype=np.float32)

        #B0[:,:,:,0] = np.divide(np.multiply(x,vec_dot_dip),rvec**5) - np.divide(mx,rvec**3)
        #B0[:,:,:,1] = np.divide(np.multiply(y,vec_dot_dip),rvec**5) - np.divide(my,rvec**3)
        #B0[:,:,:,2] = np.divide(np.multiply(z,vec_dot_dip),rvec**5) - np.divide(mz,rvec**3) #1/(x**2+y**2+z**2)#

        vec_dot_dip = 3*(y*dipoleMoment[1] + z*dipoleMoment[2])
    
        #calculate the distance of each mesh point to magnet, optimised for speed
        #for improved memory performance move in to b0 calculations
        vec_mag = np.square(x) + np.square(y) + np.square(z)
        vec_mag_3 = np.power(vec_mag,1.5)
        vec_mag_5 = np.power(vec_mag,2.5)
        del vec_mag
        
        B0 = np.zeros(np.shape(x)+(3,), dtype=np.float32)

        #calculate contributions of magnet to total field, dipole always points in xy plane
        #so second term is zero for the z component
      
        B0[:,:,:,0] = np.divide(np.multiply(x, vec_dot_dip),vec_mag_5) - np.divide(dipoleMoment[0],vec_mag_3)
        B0[:,:,:,1] = np.divide(np.multiply(y, vec_dot_dip),vec_mag_5) - np.divide(dipoleMoment[1],vec_mag_3)
        B0[:,:,:,2] = np.divide(np.multiply(z, vec_dot_dip),vec_mag_5) - np.divide(dipoleMoment[2],vec_mag_3)

        self.field = B0

        return B0