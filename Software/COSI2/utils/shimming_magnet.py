import numpy as np

class shimming_magnet():
    '''shimming magnet object. cube.
    dipolar approximation of the magnetic field.
    position and rtation are given.
    field is calculated in a given area.'''
    
    bRem = 1.26 # T remanence field of shim magnets
    magSizeOuter = 6 # mm - size of shim magnets
    
    position = [0,0,0]
    dipole = [0,1,0] #by default moment is pointing along y
    rotation_yz = 0 # rotation of the shim magnet in the ZY plane
    
    def __init__(self,position=[0,0,0], dipole_moment = 1, rotation_yz = 0):
        self.position = position
        self.rotation_yz = rotation_yz
        self.dipole = [self.dipole[0]*dipole_moment,self.dipole[1]*dipole_moment*np.cos(rotation_yz),self.dipole[1]*dipole_moment*np.sin(rotation_yz)]
        
        print('magnet created, dipole points to ',self.dipole)
    