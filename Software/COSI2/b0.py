'''rst@PTB 240429 rst030@protonmail.com'''

from datetime import datetime
import numpy as np
import pth # path class to create a path object
import osi2magnet # osi2magnet class to create an osi2magnet object 

import scipy.ndimage as cp # for erosion
from scipy.special import sph_harm
from scipy.optimize import least_squares
from scipy.linalg import lstsq

from shimming.scripts import b0V5
from utils import shimming_magnet



# HELPING METHODS
    # helping methods

def cartToSpher(coords):
    r = np.sqrt(np.sum(np.square(coords),axis = -1))    
    #remove r = 0 to avoid divide by zero
    r[r==0] = np.nan
    
    phi = np.arctan2(coords[...,0], coords[...,1]) + np.pi
    theta = np.arccos(coords[...,2]/r)
    return np.stack([r,theta, phi], axis = -1)


def getRealSphericalHarmonics(coords, maxOrder):
    r0          = np.nanmean(coords[...,0])       # Get the mean radius for normalisation
    spherHarm   = np.zeros((np.shape(coords[...,0]) + ((maxOrder + 1)**2,)))
    #r0          = np.nanmean(150)       #Get the mean radius for normalisation
    idx         = 0
    for n in range(maxOrder+1):
        for m in range(-n,n+1):
            if m < 0:
                spherHarm[...,idx] = ((1j/np.sqrt(2))*(np.divide(coords[...,0],r0)**n)*(sph_harm(m,n,coords[...,2], coords[...,1])-((-1)**m)*sph_harm(-m,n,coords[...,2], coords[...,1]))).real
            elif m > 0:
                spherHarm[...,idx] = ((1/np.sqrt(2))*(np.divide(coords[...,0],r0)**n)*(sph_harm(-m,n,coords[...,2], coords[...,1])+((-1)**m)*sph_harm(m,n,coords[...,2], coords[...,1]))).real
            elif m == 0:
                spherHarm[...,idx] = np.multiply(sph_harm(m,n,coords[...,2], coords[...,1]),np.divide(coords[...,0],r0)**n).real
            else:
                print("That wasnt suppoosed to happen!")
            idx += 1
    return spherHarm



def saveTmpData(filename:str,numpyData:np.array):
    fullFileName = './data/tmp/'+filename
    np.save(fullFileName, numpyData)
    print('saved numpy array as %s'%fullFileName) 
# ---------------------------------------------------------------------




class b0():
    '''b0 object. created for cosi data. contains path.'''

    path=None
    b0File = 0 # a file where b0 data is stored
    magnet = None
    datetime = 'timeless'
    fieldDataAlongPath = None
    filename = 'Dummy B0 map'
    
    b0Data = None # tasty, berry, what we need. 3D array. ordered. sliceable. fittable.
    shim_magnets = None
    
    interpolatedField = []
    errorField = []

    vector_of_magnet_rotations = None

    def __init__(self,path_filename='', path:pth.pth = None, b0_filename='', magnet_object = ''):
        
        if path_filename!='':
            self.path = pth.pth(filename=path_filename)
            print('b0 object created with path %s'%self.path.filename)
            
        if path is not None:
            self.path = path
               
        if magnet_object!='':
            self.magnet = magnet_object

        # if filename was given on init, read the b0 from that file.
        if b0_filename != '':
            self.filename = b0_filename
            print("WARNING. do csv reading of path and b0 all together, instead of txt reading.")

            with open(b0_filename) as file:
                raw_B0_data = file.readlines()     
                
            header_lines = raw_B0_data[0:5]    
            self.parse_header_of_B0_file(header_lines)  
            field_lines = raw_B0_data[5:]
            self.parse_field_of_B0_file(field_lines)   
            # self.transfer_coordinates_of_the_path_from_cosi_to_magnet()  
            print('read b0 data')
            print(header_lines)
        
        else:
            # if no filename was given
            # create an instance of a b0 object, populate some fields. be ready to fill in b0 values
            if self.path is None:
                print('No path object given on construction of b0 object.\n b0 instance initialized without path.')
                return
            self.fieldDataAlongPath = np.zeros((len(self.path.r),4)) # bx,by,bz,babs
        
        
        
    ''' path transformation to the coordinate system of the magnet '''    
    def transfer_coordinates_of_the_path_from_cosi_to_magnet(self,filtering=None,stepsize=None,onesign=None,component=0):
        # now does everything, like an entry point. separate.
        
        # is called by btn on gui     
        print('ROTATING THE PATH NOW!')
        # rotate path according to the euler angles of the magnet, but backwards
        #self.path.rotate_euler_backwards(gamma=self.magnet.gamma,beta=self.magnet.beta,alpha=self.magnet.alpha) 
        self.path.rotate_euler(gamma=self.magnet.gamma,beta=self.magnet.beta,alpha=self.magnet.alpha)
        
        # center the path to the origin, as the origin of the path is the origin of the magnet
        self.path.center(origin=self.magnet.origin)
        print('ROTATING THE MAGNET NOW!')
        # rotate the magnet
        #self.magnet.rotate_euler_backwards(gamma=self.magnet.gamma,beta=self.magnet.beta,alpha=self.magnet.alpha) # for the backwards euler rotation rotate by negative values in the reversed order: was zyx, now xyz
        self.magnet.rotate_euler(gamma=self.magnet.gamma,beta=self.magnet.beta,alpha=self.magnet.alpha)
        
        self.magnet.set_origin(0,0,0)    
        
         # now that we have the path and the b0 lets compare number of points in both.
        print('len(path.r)=',len(self.path.r))
        print('len(b0Data)=',len(self.fieldDataAlongPath))

        if len(self.path.r) == len(self.fieldDataAlongPath[:,0]):
            self.reorder_field_to_cubic_grid(filtering=filtering,givenstep=stepsize,onesign=onesign,component=component) # make a cubic grid with xPts, yPts, zPts and define B0 on that
            self.reorder_field_to_cubic_grid(filtering=filtering,givenstep=stepsize,onesign=onesign,component=component) # make a cubic grid with xPts, yPts, zPts and define B0 on that
        else:
            print('LEN of PATH and DATA', len(self.path.r), '   ',len(self.fieldDataAlongPath[:,0]))
        
        
        
        
        
    # ----------------- artificial data generation ----------------- 
    def make_cylindrical_anomaly_along_x(self,yz_of_the_cylinder_center,radius_of_cylinder,intensity,bg):
        path = self.path
        y0 = yz_of_the_cylinder_center[0]
        z0 = yz_of_the_cylinder_center[1]
        
        bg_field = bg
        anomaly_field = intensity

        self.fieldDataAlongPath = np.zeros((len(self.path.r),4)) # bx,by,bz,babs

        for idx in range(len(path.r)):
            self.fieldDataAlongPath[idx,:] = [bg_field,bg_field,bg_field,bg_field]
            if np.sqrt((path.r[idx,1]-y0)**2+(path.r[idx,2]-z0)**2)<radius_of_cylinder:
                self.fieldDataAlongPath[idx,:] = [anomaly_field,anomaly_field,anomaly_field,anomaly_field]
                
        print("cylinder generated")        
        self.reorder_field_to_cubic_grid()





    def make_artificial_field_along_path(self,coordinates_of_singularity,radius_of_singularity,intensity,bg):
        path = self.path
        x0 = coordinates_of_singularity[0]
        y0 = coordinates_of_singularity[1]
        z0 = coordinates_of_singularity[2]
        
        bg_field = bg
        anomaly_field = intensity

        self.fieldDataAlongPath = np.zeros((len(self.path.r),4)) # bx,by,bz,babs
        for idx in range(len(path.r)):
            self.fieldDataAlongPath[idx,:] = [bg_field,bg_field,bg_field,bg_field]
            if np.sqrt((path.r[idx,0]-x0)**2+(path.r[idx,1]-y0)**2+(path.r[idx,2]-z0)**2)<radius_of_singularity:
                self.fieldDataAlongPath[idx,:] = [anomaly_field,anomaly_field,anomaly_field,anomaly_field]
                
        self.reorder_field_to_cubic_grid()

        
    def delete_max_point(self,maxPoint):
        for idx in range(len(self.path.r)):
            if self.fieldDataAlongPath[idx,0] >= maxPoint:
                self.fieldDataAlongPath[idx,:] = self.fieldDataAlongPath[idx-1,:]
                print('max point %d replaced by neighbor'%idx)
                    
    def delete_min_point(self,minPoint):
        for idx in range(len(self.path.r)):
            if self.fieldDataAlongPath[idx,0] <= minPoint:
                self.fieldDataAlongPath[idx,:] = self.fieldDataAlongPath[idx-1,:]
                print('min point %d replaced by neighbor'%idx)
            
    
    
    def reorder_field_to_cubic_grid(self,filtering=None,givenstep=None,onesign=None,component=0):
    def reorder_field_to_cubic_grid(self,filtering=None,givenstep=None,onesign=None,component=0):
        # what we want to do here is make the coordinate grid. A cube, essentially.
        # we know that the path has a fixed distance between the points. This is crucial.
        # but the path is a snake path! it is a 1d line. 
        # we want to parse it into a 3D grid. Would be nice to have this grid stored somewhere 
        # at the stage of path generation, like a tmp file or something.
        # but we did not generate it, although we could. hm.
        # so maybe just going back to the generating script and making the path with the same parameters would 
        # do the job for me and Id go write my batteries?
        # nay lets be fair and *measure* the points on the path.
        
        # limits of the path will give us the siye of the cube for that grid
        x_max = max(self.path.r[:,0])
        x_min = min(self.path.r[:,0])
        y_max = max(self.path.r[:,1])
        y_min = min(self.path.r[:,1])
        z_max = max(self.path.r[:,2])
        z_min = min(self.path.r[:,2])
        
        print(x_min,' < x < ',x_max)
        print(y_min,' < y < ',y_max)
        print(z_min,' < z < ',z_max)
        
        # now lets determine how many points we need per each axis
        # lets emasure the step on x
        step_size_x_list = []
        step_size_y_list = []
        step_size_z_list = []
        
        for idx in range(2,len(self.path.r)):
            step = abs(self.path.r[idx,:] - self.path.r[idx-1,:])
            
            if step[0] > 1e-3:
                step_size_x_list.append(step[0])
            if step[1] > 1e-3:
                step_size_y_list.append(step[1])
            if step[2] > 1e-3:
                step_size_z_list.append(step[2])  
                  
        step_size_x = min(step_size_x_list)
        step_size_y = min(step_size_y_list)
        print(step_size_z_list)
        step_size_z = min(step_size_z_list)
        
        print('given step ',givenstep)
        if givenstep is not None:
            step_size_x = givenstep
            step_size_y = givenstep
            step_size_z = givenstep
            
        print('path step size: ',step_size_x,step_size_y,step_size_z)
        
        # so there are unique_x x values between x_min and x_max
        # lets make a linspace
        self.xPts = np.arange(start=x_min,stop=x_max,step=step_size_x) #linspace(start=x_min,stop=x_max,num=num_steps_x)
        #print("10 xPts: ", self.xPts[0:10])
        self.yPts = np.arange(start=y_min,stop=y_max,step=step_size_y) #linspace(start=y_min,stop=y_max,num=num_steps_y)
        #print("10 yPts: ", self.yPts[0:10])
        self.zPts = np.arange(start=z_min,stop=z_max,step=step_size_z) #linspace(start=z_min,stop=z_max,num=num_steps_z)
        #print("10 zPts: ", self.zPts[0:10])
        
                
        # now we do a trick
        # we will go through the snake. 
        # for each (3-valued) point see snake we 
        # take its 0th value and scan xPts searching 
        # which is the closest.
        # that is, less than epsilon
        
        epsx = step_size_x/2#(self.xPts[1]-self.xPts[0])/3
        epsy = step_size_y/2#(self.yPts[1]-self.yPts[0])/3
        epsz = step_size_z/2#(self.zPts[1]-self.zPts[0])/3

        # then we get the index of xPts
        # and same for z and y
        # the b0Data will be a 3D array
        # indexing is the same for path and b0_values_1D
        
        b0Data = np.zeros((len(self.xPts),len(self.yPts),len(self.zPts),4))
        
        meanField_raw = np.mean((self.fieldDataAlongPath[:,component]))
                   
        for idx in range(np.size(self.path.r,0)):
            x_value_along_path = self.path.r[idx,0]
            y_value_along_path = self.path.r[idx,1]
            z_value_along_path = self.path.r[idx,2]

            xArg = min(np.where(abs(self.xPts - x_value_along_path) < epsx))
            yArg = min(np.where(abs(self.yPts - y_value_along_path) < epsy))
            zArg = min(np.where(abs(self.zPts - z_value_along_path) < epsz))
        
            #print("pth r=[",self.path.r[idx,:],"] closest grid [",xPts[xArg],yPts[yArg],zPts[zArg],"]")
            
                    # get minmax and average
                # cleaning

            
            if self.fieldDataAlongPath[idx,component] == 0:
            if self.fieldDataAlongPath[idx,component] == 0:
                self.fieldDataAlongPath[idx,:] == self.fieldDataAlongPath[idx-1,:] if self.fieldDataAlongPath[idx-1,0] !=0 else meanField_raw#self.fieldDataAlongPath[idx-2,:]
                print('b0 importer: warning! 0 VALUE detected! pt %d, assigning'%(idx),self.fieldDataAlongPath[idx-1,:])
           
            # replacing the max point by neighbor
            if filtering is not None: # by how much can the valiues deviate
                print('mean: ',abs(meanField_raw))
                
                if onesign:
                    if self.fieldDataAlongPath[idx,component]>0:
                        print(self.fieldDataAlongPath[idx,component],'is wrong sign! assigning',meanField_raw, '!!!')
                        self.fieldDataAlongPath[idx,:] = self.fieldDataAlongPath[idx-1,:]
                        print('assigned: ',self.fieldDataAlongPath[idx,:], '<+-+-+-')
                
                print(abs(self.fieldDataAlongPath[idx,component])/meanField_raw)
                if abs(self.fieldDataAlongPath[idx,component])/meanField_raw>filtering:
                    print(self.fieldDataAlongPath[idx,component],'is too high! assigning',self.fieldDataAlongPath[idx-1,:], '!!!')
                    self.fieldDataAlongPath[idx,:] = self.fieldDataAlongPath[idx-1,:]
                    print('assigned: ',self.fieldDataAlongPath[idx,:], '<+++++')
            
            # replacing the min point by neighbor
                if meanField_raw/abs(self.fieldDataAlongPath[idx,component])>filtering:
                    print(self.fieldDataAlongPath[idx,component],'is too low! assigning',self.fieldDataAlongPath[idx-1,:], '!!!')
                    self.fieldDataAlongPath[idx,:] = self.fieldDataAlongPath[idx-1,:]
                    print('assigned: ',self.fieldDataAlongPath[idx,:], '<-----')

            b0Data[xArg,yArg,zArg,:] = [self.fieldDataAlongPath[idx,0],self.fieldDataAlongPath[idx,1],self.fieldDataAlongPath[idx,2],self.fieldDataAlongPath[idx,3]]
            print(b0Data[xArg,yArg,zArg,:])

            
          
        # getting mean field
        #meanField = np.nanmean(b0Data[:,:,:,0])
        #b0Data[b0Data==0]=meanField_raw #!!!!    
        
        b0Data[b0Data==0]=np.nan  
        # homogeniety
        maxField = np.nanmax(b0Data[:,:,:,component])
        minField = np.nanmin(b0Data[:,:,:,component])
               
        meanField = np.nanmean(b0Data[:,:,:,component])

        
        try:
            homogeneity = float(1e6*(maxField-minField)/meanField)
        except:
            homogeneity = 0
            
        print('Mean field <B0> = ',meanField, 'mT')
        print('Max field = ',maxField, 'mT')
        print('Min field = ',minField, 'mT')
        print('homogeniety: %.0f ppm'%homogeneity)


        self.homogeneity = homogeneity
        self.mean_field = meanField
        self.b0Data = b0Data
        print('B0.B0 DATA GENERATED ON A RECT GRID')
        
        print('generating a mesh grid')

    # ------------------- data parsers -------------------
                
    def parse_field_of_B0_file(self,field_lines):
        #-2.842000 48.057000 -2.319000 48.197000
        self.fieldDataAlongPath = np.zeros((len(field_lines),4))
        
        for idx, line in enumerate(field_lines):
            try:
                b0x = float(line.split(' ')[0])
                b0y = float(line.split(' ')[1])
                b0z = float(line.split(' ')[2])
                b0abs = float(line.split(' ')[3])
            except:
                b0x = float(line.split(',')[0])
                b0y = float(line.split(',')[1])
                b0z = float(line.split(',')[2])
                b0abs = float(line.split(',')[3])

            if b0abs == 0 :# sometimes gaussmeter doesnt give the vector
                b0abs = np.sqrt(b0x**2+b0y**2+b0z**2)
                print('OOPS')

            self.fieldDataAlongPath[idx,:] = [b0x,b0y,b0z,b0abs]


        if self.fieldDataAlongPath[0,1] == 0:
            self.fieldDataAlongPath[0,0] = np.nanmean(self.fieldDataAlongPath[1:,0])
            self.fieldDataAlongPath[0,1] = np.nanmean(self.fieldDataAlongPath[1:,1])
            self.fieldDataAlongPath[0,2] = np.nanmean(self.fieldDataAlongPath[1:,2])
            self.fieldDataAlongPath[0,3] = np.nanmean(self.fieldDataAlongPath[1:,3])


                        
    def parse_field_of_CSV_file(self,field_lines,comsol=None):
        # 315.17	152.35	113.75	0	100	0	0
        self.fieldDataAlongPath = np.zeros((len(field_lines),4))
        for idx, line in enumerate(field_lines):
            x = float(line.split(',')[0])
            y = float(line.split(',')[1])
            z = float(line.split(',')[2])
            try:
                b0x = float(line.split(',')[3])
            except:
                b0x = 0
            try:
                b0y = float(line.split(',')[4])
            except:
                b0y = 0
            try:
                b0z = float(line.split(',')[5])
            except:
                b0z = 0
            try:
                b0abs = float(line.split(',')[6])
            except:
                b0abs = 0

            if b0abs == 0 :# sometimes gaussmeter doesnt give the vector
                b0abs = np.sqrt(b0x**2+b0y**2+b0z**2)
                print('OOPS, |Bo|=0')
            
            self.fieldDataAlongPath[idx,:] = [b0x,b0y,b0z,b0abs]
            #self.path.r.append([x,y,z])
            
    def parse_header_of_CSV_file(self,header_lines,eulers=None,comsol=None):
        # COSI2 B0 scan						
        # time 2024-05-17 13:21:45.456312						
        # MAGNET CENTER IN LAB: x 265.170 mm	 y 182.350 mm	 z 163.750 mm				
        # MAGNET AXES WRT LAB: alpha 0.00 deg	 beta 0.00 deg	 gamma 0.00 deg				
        # path: C:/cosi-measure/Software/COSI2/dummies/b0_maps/testcsv.path						
        # X[mm]	Y[mm]	Z[mm]	B0_x[mT]	B0_y[mT|	B0_z[mT]	B0_abs[mT]
        
        if comsol is None:
            self.datetime = header_lines[1].split(' ')[2:3]
            mg_cor_str = header_lines[2].split(':')[1]
            mag_center_x = float(mg_cor_str.split(',')[0].split(' ')[2])
            mag_center_y = float(mg_cor_str.split(',')[1].split(' ')[2])
            mag_center_z= float(mg_cor_str.split(',')[2].split(' ')[2])
            
            # euler angles are rotation of the magnet wrt cosi
            mg_euler_str = header_lines[3].split(':')[1]

            mag_alpha = float(mg_euler_str.split(',')[0].split(' ')[2])
            mag_beta = float(mg_euler_str.split(',')[1].split(' ')[2])
            mag_gamma= float(mg_euler_str.split(',')[2].split(' ')[2])
        
            path_filename_str = str(header_lines[4].split('path:')[1])
            print('warning. path file %s not used. path data taken from csv!'%path_filename_str)
        
        
        else:
            self.datetime=str(datetime.now())
            mag_center_x = 0
            mag_center_y = 0
            mag_center_z = 0
            mag_alpha = 0
            mag_beta = 0
            mag_gamma = 0
            
        
        if eulers is not None:
            mag_alpha = float(eulers[0])
            mag_beta = float(eulers[1])
            mag_gamma= float(eulers[2])

        self.magnet = osi2magnet.osi2magnet(origin=[mag_center_x,mag_center_y,mag_center_z],euler_angles_zyx=[mag_alpha,mag_beta,mag_gamma])


        
    
    
    def parse_header_of_B0_file(self,header_lines):
        self.datetime = header_lines[1]
        # ['COSI2 B0 scan\n', 
        # '2024-04-19 10:25:38.383373\n', 
        # 'MAGNET CENTER IN LAB: x 265.287 mm, y 166.332 mm, z 163.238 mm\n', 
        # 'MAGNET AXES WRT LAB: alpha -90.00 deg, beta -90.00 deg, gamma 0.00 deg\n',
        # 'path: ./data/240418/a00_ball_path_80mm_coarse_5s_FAST.path\n']
        
        mg_cor_str = header_lines[2].split(':')[1]
        mag_center_x = float(mg_cor_str.split(',')[0].split(' ')[2])
        mag_center_y = float(mg_cor_str.split(',')[1].split(' ')[2])
        mag_center_z= float(mg_cor_str.split(',')[2].split(' ')[2])

        mg_euler_str = header_lines[3].split(':')[1]
        mag_alpha = float(mg_euler_str.split(',')[0].split(' ')[2])
        mag_beta = float(mg_euler_str.split(',')[1].split(' ')[2])
        mag_gamma= float(mg_euler_str.split(',')[2].split(' ')[2])

        self.magnet = osi2magnet.osi2magnet(origin=[mag_center_x,mag_center_y,mag_center_z],euler_angles_zyx=[mag_alpha,mag_beta,mag_gamma])




    # DATA MANIPULATION SECTION
     # ------------ fitting B0 with spherical harmonics ---------------

    def fitSphericalHarmonics(self, maxorder:int, dsv:float, resol:float,component=0):
        # fitSphericalHarmonicsShell2_forInterpolation.py
        
        fieldMap = self.b0Data[:,:,:,component] #np.load(r'./data/tmp/b0Data.npy')[...,0]
        
        # order of sph harm to consider
        maxOrder = maxorder
        # Diameter spherical volume over which to do the spherical harmonic decomposition
        self.DSV = dsv
        # resolution for rendering the sph harmonics
        resolution = resol
        
        #Create a cartesian coordinate system of where the data points were acquired, maps were acquired with a 5mm resolution 
        fieldMapDims = [len(self.xPts),len(self.yPts),len(self.zPts)]#np.shape(fieldMap) #!!!! 240911
        
        print(fieldMapDims)
        
        self.xDim_SPH_decomp = self.xPts#np.linspace(0, resolution*(fieldMapDims[0]), fieldMapDims[0]) - resolution*(fieldMapDims[0])/2
        self.yDim_SPH_decomp = self.yPts#np.linspace(0, resolution*(fieldMapDims[1]), fieldMapDims[1]) - resolution*(fieldMapDims[1])/2
        self.zDim_SPH_decomp = self.zPts#np.linspace(0, resolution*(fieldMapDims[2]), fieldMapDims[2]) - resolution*(fieldMapDims[2])/2

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!
        x,y,z = np.meshgrid(self.yDim_SPH_decomp, self.xDim_SPH_decomp, self.zDim_SPH_decomp, indexing='xy')  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 240911
        coord = [x,y,z] #!!!!! 240916 !!! XY indexing - we need to render the magnets!
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!


        #Create a spherical mask for the data
        sphereMask = np.zeros(np.shape(coord[0]), dtype = bool)
        sphereMask[np.square(coord[0]) + np.square(coord[1]) + np.square(coord[2]) <= (self.DSV/2)**2] = 1 
        sphereMask = sphereMask*(~np.isnan(fieldMap))

        # Create a spherical shell mask to consider only data points on the surface of the sphere
        erodedMask = cp.binary_erosion(sphereMask)  # remove the outer surface of the initial spherical mask
        shellMask = np.array(sphereMask^erodedMask, dtype = float)   # create a new mask by looking at the difference between the inital and eroded mask
        shellMask[shellMask == 0] = np.nan  # set points outside mask to 'Nan', works better than setting it to zero for calculating mean fields etc.

        sphereMask = np.asarray(sphereMask, dtype=float)
        sphereMask[sphereMask == 0] = np.nan

        #apply mask to data
        maskedField = np.multiply(sphereMask, fieldMap)
        print("Mean field strength in %i cm sphere: %.2f mT"%(self.DSV/10, np.nanmean(maskedField)))
        print("Inhomogeneity in %i cm sphere: %.0f ppm" %(self.DSV/10, 1e6*(np.nanmax(maskedField) - np.nanmin(maskedField))/np.nanmean(maskedField)))

        #convert cartesian coordinates to spherical coordinates
        #spherCoord = cartToSpher(np.stack((coord[1],coord[0], coord[2]), axis = -1))
        spherCoord = cartToSpher(np.stack((coord[0],coord[1], coord[2]), axis = -1))
        
        #apply mask to field and coordinate arrays and vectorise them
        maskedFieldShell = fieldMap[shellMask == 1]
        maskedCoordShell = spherCoord[shellMask == 1,:]
        
        #Get the spherical harmonics for each of the field points
        spherHarm = getRealSphericalHarmonics(maskedCoordShell, maxOrder)

        #Inital guess for the optimisation
        initialGuess = np.zeros((np.size(spherHarm,-1)))

        def fitSphericalHarmonics(fitVector, args):
            return np.square(maskedFieldShell - np.matmul(spherHarm, fitVector))

        #run the optimisation
        fitData = least_squares(fitSphericalHarmonics, initialGuess, args = (maskedFieldShell,))

        #grab the coefficients from the data array
        spherHarmCoeff = fitData.x


        lsqFit = lstsq(spherHarm, maskedFieldShell)
        #print(lsqFit[:])
        #spherHarmCoeff = lsqFit[0]

        #calculate the field from the spherical harmonic decomposition
        decomposedField = np.matmul(spherHarm, spherHarmCoeff)
        print("Inhomogeneity of fit: %.0f ppm" %(abs(1e6*(np.max(decomposedField) - np.min(decomposedField))/np.mean(decomposedField))))

        #See what the difference is between the two decomposed field
        diffSph =  maskedFieldShell - decomposedField
      
        #generate spherical coordinates over entire sphere, not just shell, for plotting
        spherCoordSphere = np.copy(spherCoord) #!!!!! 240916 instead of np.shape(coord(0))
        spherCoordSphere[spherCoord[...,0] == 0,:] = np.nan
        spherHarm3D = getRealSphericalHarmonics(spherCoordSphere, maxOrder)

        tempSpherHarmCoeff = np.copy(spherHarmCoeff)
        # tempSpherHarmCoeff[4:] = 0
        #calculate the spherical harmonic decomposed field
        decomposedField = np.matmul(spherHarm3D, tempSpherHarmCoeff)*sphereMask

        # calculate difference between decomposed field and measured field
        errorField = maskedField - decomposedField
        print("Error: %.0f ppm" %(1e6*(np.nanmax(errorField) - np.nanmin(errorField))/np.nanmean(maskedField)))

        # --- assigning class variables ---
        self.maskedField = maskedField
        self.decomposedField = decomposedField
        self.errorField = errorField
        self.spherHarmCoeff = spherHarmCoeff
        
        
        # save sph_harm coefficients
        SphHarmDataNumpyFilename = 'SpHData.npy'
        saveTmpData(filename = SphHarmDataNumpyFilename,numpyData=self.spherHarmCoeff)     


    def interpolateField(self, resol:float, dsv:float): # 250 interpolate field with sph harmonics at /higher/ res
        '''interpolate the measured field with higher resolution by the spherical harmonics calculated before'''
        coeffs = self.spherHarmCoeff #np.load(r'./data/tmp/SpHData.npy')
        print('sph coefficients loaded')
        maxOrder = int(np.size(coeffs)**0.5 -1)
        DSV = dsv # diameter of sphere, mm
        resolution = resol # mm
        self.resolution = resolution
    
        print('making a fine coordinate grid')
        
        # Create a cartesian coordinate system of where the data points were acquired, maps were acquired with a 5mm resolution 
        self.xDim_SPH_fine = np.linspace(-DSV/2, DSV/2, int(DSV/resolution+1))
        self.yDim_SPH_fine = np.linspace(-DSV/2, DSV/2, int(DSV/resolution+1))
        self.zDim_SPH_fine = np.linspace(-DSV/2, DSV/2, int(DSV/resolution+1))
        
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        x,y,z = np.meshgrid(self.xDim_SPH_fine, self.yDim_SPH_fine, self.zDim_SPH_fine, indexing='xy')
        self.coord_grid_fine = [x,y,z] #!!!!!!!!!!!!! 240916 !!!!!! gives Y,X,Z in the coord_grid_fine
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #!!!!!!!!!!! 240916 !!!!!!!!!!!!!! xy indexing, we need to render the magnets
        
                #Create a spherical mask for the data
        sphereMask = np.zeros(np.shape(self.coord_grid_fine[0]), dtype = bool)
        sphereMask[np.square(self.coord_grid_fine[0]) + np.square(self.coord_grid_fine[1]) + np.square(self.coord_grid_fine[2]) <= (DSV/2)**2] = 1 
        sphereMask = np.asarray(sphereMask, dtype=np.double)
        sphereMask[sphereMask == 0] = np.nan
  
        self.sphere_mask = sphereMask # for later

        spherCoord = cartToSpher(np.stack((self.coord_grid_fine[0],self.coord_grid_fine[1], self.coord_grid_fine[2]), axis = -1))
        #generate spherical coordinates over entire sphere, not just shell, for plotting
        spherCoordSphere = np.copy(spherCoord)
        spherCoordSphere[spherCoord[...,0] == 0,:] = np.nan
        
        spherHarm3D = getRealSphericalHarmonics(spherCoordSphere, maxOrder)
        #calculate the field from the spherical harmonic decomposition
        decomposedField = np.matmul(spherHarm3D, coeffs)*sphereMask

        print("Inhomogeneity of fit: %.0f ppm" %(abs(1e6*(np.nanmax(decomposedField) - np.nanmin(decomposedField))/np.nanmean(decomposedField))))
        self.interpolatedField = decomposedField
        DecomposedDataNumpyFilename = 'B0_interpolated.npy'
        saveTmpData(filename = DecomposedDataNumpyFilename,numpyData=self.interpolatedField) 
            
        
    def load_shim_magnets(self,fname: str):
        print('TODO: load the positions and rotations of the shim magnets.')
        print('opening file %s'%fname)
        #x[m],y[m],z[m],dirx[m^2A],diry[m^2A],dirz[m^2A],rotation_xy[rad]
        #0.0000,0.2721,-0.0461,0.0000e+00,6.2104e-09,2.2358e-08,1.2999
        print('save magnet positions and rotations to the file')
        positions = []
        rotations = []
        
        with open(fname, 'r') as file:
            lines = file.readlines()
            for idx,line in enumerate(lines):
                if idx>0:
                    x = float(line.split(',')[0])
                    y = float(line.split(',')[1])
                    z = float(line.split(',')[2])
                    alpha = float(line.split(',')[-1])
                    position = [x,y,z]
                    rotation = alpha
                    print('pos=',position)
                    print('rotation=',rotation)
                    positions.append(position)
                    rotations.append(rotation)

        file.close()
        print('imported shim magnets rotations/positions from *txt file')
        print('creating the shim magnets')
        print('positions: ',len(positions))
        print('rotations: ',len(rotations))
        
        self.shim_magnets = []
        for idx,position in enumerate(positions):
            magnet = shimming_magnet.shimming_magnet(position=positions[idx],rotation_yz=rotations[idx])
            self.shim_magnets.append(magnet)
        print(len(self.shim_magnets),' shim magnets generated.')        
        print('TODO: render the fields')
        

    def render_fair_shim_field(self):
        if self.shim_magnets is not None:
            print('the shim magnets are loaded. rendering field.')
            self.shimField = self.coord_grid_fine[0]*0
            for magnet in self.shim_magnets:
                magnet.render_field(grid=self.coord_grid_fine)
                self.shimField+=magnet.B0[:,:,:,2]
        else:
            print('load shim nmagnets first!')
            
    
    def get_shim_positions(self,dsv_for_opt_percent:int,verbose:bool):
        print('COPY CODE FROM JUPYTER')


    def save_for_echo(self):
        if self.vector_of_magnet_rotations is not None:
            np.save('vector_of_magnet_rotations.npy',self.vector_of_magnet_rotations)
        if self.fldsZ is not None:
            np.save('fldsZ.npy',self.fldsZ)
        if self.fldsY is not None:
            np.save('fldsY.npy',self.fldsY)
        if self.interpolatedField_masked is not None:
            np.save('interpolatedField_masked.npy',self.interpolatedField_masked)
        
        
        


    # saving magnet positions into rings
    def save_rings(self,fname:str):
        print('save magnet positions and rotations to the file')
        with open(fname, 'w') as file:
            file.write('x[m],y[m],z[m],dirx[m^2A],diry[m^2A],dirz[m^2A],rotation_xy[rad]\n')
            for magnet in self.shim_magnets:
                 x=magnet.position[0]
                 y=magnet.position[1]
                 z=magnet.position[2]
                 dirx = magnet.dipole_vector[0]
                 diry = magnet.dipole_vector[1]
                 dirz = magnet.dipole_vector[2]     
                 rot  = magnet.rotation_yz

                 file.write('%.4f,%.4f,%.4f,%.4e,%.4e,%.4e,%.4f\n'%(x,y,z,dirx,diry,dirz,rot))
        file.close()
        print('exported magnet rotations/positions as *txt file')
        
    def update_magnet_rotations(self, fname):
        print('loading magnet rotations from file')
        with open(fname, 'r') as file:
            raw_data = file.readlines()[1:]
            vec_mag_rots = []
            for line in raw_data:
                vals = line.split(',')
                rot = float(vals[-1])        
                vec_mag_rots.append(rot)
        file.close()
        self.vector_of_magnet_rotations = np.asarray(vec_mag_rots)
        if self.shim_magnets is not None or self.shim_magnets == []:
            for idx,magnet in enumerate(self.shim_magnets):
                magnet.rotation_yz = self.vector_of_magnet_rotations[idx]
                magnet.render_field(self.coord_grid_fine)            
        else:
            self.shim_magnets = []
            
            for idx,angle in enumerate(self.vector_of_magnet_rotations):
                magnet = shimming_magnet.shimming_magnet(position=[0,0,0],rotation_yz=angle)
                self.shim_magnets.append(magnet)              
        
        


















# ------------------------------ DATA SAVING METHODS -------------------------------

    def save_separately(self,filename:str):
        with open(filename+'.path', 'w') as file:
             for pathpt in self.path.r:
                 file.write('X%.2fY%.2fZ%.2f\n'%(pathpt[0],pathpt[1],pathpt[2]))
        file.close()
        print('exported path as *.path file')
        
        with open(filename+'.txt', 'w') as file:
             for field_pt in self.fieldDataAlongPath:
                 file.write('%.9f	%.9f	%.9f	%.9f	\n'%(abs(field_pt[3]),abs(field_pt[2]),abs(field_pt[1]),abs(field_pt[0])))
        file.close()
        print('exported field as *.txt file')



    def saveAsCsv_for_comsol(self, filename: str):
        # for comsol
        magnet = self.magnet
        with open(filename, 'w') as file:
            
            file.write('# COSI2 B0 scan\n')                    
            # Convert date and time to string
            dateTimeStr = str(datetime.now())
            file.write('# time '+dateTimeStr+'\n')
            file.write('# MAGNET CENTER IN LAB: x %.3f mm, y %.3f mm, z %.3f mm\n'%(magnet.origin[0],magnet.origin[1],magnet.origin[2]))
            file.write('# MAGNET AXES WRT LAB: alpha %.2f deg, beta %.2f deg, gamma %.2f deg\n'%(magnet.alpha,magnet.beta,magnet.gamma))   
            file.write('# path: '+self.path.filename+'\n')
            file.write('# X[mm],Y[mm],Z[mm],B0_x[mT],B0_y[mT|,B0_z[mT],B0_abs[mT]\n')   

            for i in range(len(self.path.r[:,0])):
                ri = self.path.r[i,:]            
                bi = self.fieldDataAlongPath[i,:]
                file.write('%.3f,%.3f,%.3f,%.4f,%.4f,%.4f,%.4f\n'%(ri[0],ri[1],ri[2],bi[0],bi[1],bi[2],bi[3]))
                
    def import_from_csv(self,b0_filename: str,eulers=None,comsol=None,magnet_center=None):
        print('importing b0 object from csv file%s'%b0_filename)

        # make an empty instance of b0 and get the b0 values from the csv file.
        self.__init__()        
        self.filename = b0_filename
        with open(b0_filename) as file:
                raw_B0_data = file.readlines()     
                headerlength = 0
                for line in raw_B0_data:
                    if line[0] == '#' or line[0] == '%':
                        headerlength += 1
                        
                header_lines = raw_B0_data[0:headerlength]    
                field_lines = raw_B0_data[headerlength:]
                self.parse_header_of_CSV_file(header_lines,eulers=eulers,comsol=comsol)
                self.parse_field_of_CSV_file(field_lines,comsol=comsol)
         
        # import the path from the path file
        self.path = pth.pth(csv_filename = b0_filename,magnet_center=magnet_center)

                
        
            
    # def saveAs(self,filename: str):
    #     # open file filename and write comma separated values in it
    #     # experiment parameters
    #     # data
    #     with open(filename, 'w') as file:
    #         file.write('COSI pathfile generator output.')
    #         file.write('Date/Time,%s\n\n\n'%self.datetime)
    #         for pathpt in self.path:
    #             file.write('x%.2f,y%.2f,z%.2f\n'%(pathpt[0],pathpt[1],pathpt[2]))
    #     file.close()