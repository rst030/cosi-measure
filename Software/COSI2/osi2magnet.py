'''rst@PTB 240409 rst030@protonmail.com'''
import numpy as np
from scipy.spatial.transform import Rotation as R # {‘x’, ‘y’, ‘z’} for extrinsic rotations 


class osi2magnet():
    '''magnet object. created for cosi.'''

    # the position of the magnet is defined by 3 euler angles and 3 coordinates
    alpha = 0
    beta = 0 
    gamma = 0

    origin = [0,0,0]

    vec_length = 32 # vectors are 32 mm long
    bore_radius = 150 # mm - adjust!
    bore_depth = 500
    
    # shim rings
    shim_ring_radius = 276 # mm

    xvector = 0
    yvector = 0
    zvector = 0

    def __init__(self,origin=None,euler_angles_zyx=None):
        # initially the magnet is aligned with the lab frame
        self.set_origin(0,0,0)
        if origin:
            self.set_origin(origin[0],origin[1],origin[2])
        if euler_angles_zyx:
            self.rotate_euler(alpha=euler_angles_zyx[0],beta=euler_angles_zyx[1],gamma=euler_angles_zyx[2])


    def set_origin(self,x,y,z):
        self.origin = np.array([x,y,z])
        self.xvector = self.origin+np.array([1,0,0])*self.bore_radius
        self.yvector = self.origin+np.array([0,1,0])*self.bore_radius
        self.zvector = self.origin+np.array([0,0,1])*self.bore_radius

        self.make_bores()

    def rotate_euler(self,alpha,beta,gamma):

        self.xvector = rotatePoint_zyx(point = self.xvector,origin=self.origin,alpha=alpha,beta=beta,gamma=gamma)
        self.yvector = rotatePoint_zyx(point = self.yvector,origin=self.origin,alpha=alpha,beta=beta,gamma=gamma)
        self.zvector = rotatePoint_zyx(point = self.zvector,origin=self.origin,alpha=alpha,beta=beta,gamma=gamma)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.make_bores()




    def make_bores(self):
        t = np.linspace(0, 2*np.pi, 64)
        # x in the magneti is along the bore axis
        self.bore_front_X = t*0+self.origin[0]-self.bore_depth/2
        self.bore_front_Y = np.sin(t)*self.bore_radius+self.origin[1]
        self.bore_front_Z = np.cos(t)*self.bore_radius+self.origin[2]

        self.bore_back_X = t*0+self.origin[0]+self.bore_depth/2
        self.bore_back_Y = np.sin(t)*self.bore_radius+self.origin[1]
        self.bore_back_Z = np.cos(t)*self.bore_radius+self.origin[2]            

def rotatePoint_zyx(point:np.array, origin:np.array, alpha, beta, gamma):
    # all rotations are extrinsic rotations in the laboratory frame of reference   

    # 1. rotate about z by alpha
    # 2. rotate about y by beta
    # 3. rotate about x by gamma

    r = np.asarray(point)
    #print('init pt: ',r)
    center = np.asarray(origin)
    #print('center: ',center)
    r_centered = r-center
    #print('centered pt: ',r_centered)
    
    a = alpha
    b = beta
    y = gamma

    from scipy.spatial.transform import Rotation

    A = Rotation.from_euler('XYZ',[y, b, a], degrees=True)

    rotation_matrix = A.as_matrix()
    #print('rotation matrix: ',rotation_matrix)
    r_centered_rotated = rotation_matrix@r_centered
    #print('rotated centered pt: ',r_centered_rotated)
    r_centered_rotated_shifted = r_centered_rotated+center
    #print('rotated centered shifted pt: ',r_centered_rotated_shifted)

    return r_centered_rotated_shifted