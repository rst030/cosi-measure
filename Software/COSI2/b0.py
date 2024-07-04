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
    def transfer_coordinates_of_the_path_from_cosi_to_magnet(self):
        # now does everything, like an entry point. separate.
        
        # is called by btn on gui     
        print('ROTATING THE PATH NOW!')
        # rotate path according to the euler angles of the magnet, but backwards
        self.path.rotate_euler_backwards(gamma=self.magnet.gamma,beta=self.magnet.beta,alpha=self.magnet.alpha) 
        # center the path to the origin, as the origin of the path is the origin of the magnet
        temp_origin_offset = [0,0,0]
        self.path.center(origin=self.magnet.origin+temp_origin_offset)
        print('ROTATING THE MAGNET NOW!')
        # rotate the magnet
        self.magnet.rotate_euler_backwards(gamma=self.magnet.gamma,beta=self.magnet.beta,alpha=self.magnet.alpha) # for the backwards euler rotation rotate by negative values in the reversed order: was zyx, now xyz
        self.magnet.set_origin(temp_origin_offset[0],temp_origin_offset[1],temp_origin_offset[2])    
        
         # now that we have the path and the b0 lets compare number of points in both.
        print('len(path.r)=',len(self.path.r))
        print('len(b0Data)=',len(self.fieldDataAlongPath))

        if len(self.path.r) == len(self.fieldDataAlongPath[:,0]):
            self.reorder_field_to_cubic_grid() # make a cubic grid with xPts, yPts, zPts and define B0 on that
    
        
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
            
    
    
    def reorder_field_to_cubic_grid(self):
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
            step = self.path.r[idx,:] - self.path.r[idx-1,:]
            if step[0] > 1e-3:
                step_size_x_list.append(step[0])
            if step[1] > 1e-3:
                step_size_y_list.append(step[1])
            if step[2] > 1e-3:
                step_size_z_list.append(step[2])  
                  
        step_size_x = min(step_size_x_list)
        step_size_y = min(step_size_y_list)
        step_size_z = min(step_size_z_list)
        
        print('path step size: ',step_size_x,step_size_y,step_size_z)

        #num_steps_x = round((x_max-x_min)/step_size_x)+1
        #num_steps_y = round((x_max-x_min)/step_size_y)+1
        #num_steps_z = round((x_max-x_min)/step_size_z)+1

        # so there are unique_x x values between x_min and x_max
        # lets make a linspace
        self.xPts = np.arange(start=x_min,stop=x_max,step=step_size_x) #linspace(start=x_min,stop=x_max,num=num_steps_x)
        print("xPts: ", self.xPts[0:10])
        self.yPts = np.arange(start=y_min,stop=y_max,step=step_size_y) #linspace(start=y_min,stop=y_max,num=num_steps_y)
        print("yPts: ", self.yPts[0:10])
        self.zPts = np.arange(start=z_min,stop=z_max,step=step_size_z) #linspace(start=z_min,stop=z_max,num=num_steps_z)
        print("zPts: ", self.zPts[0:10])
        
                
        # now we do a trick
        # we will go through the snake. 
        # for each (3-valued) point see snake we take the its 0th value and scan xPts searching which is the closest.
        # that is, less than epsilon
        
        epsx = (self.xPts[1]-self.xPts[0])/3
        epsy = (self.yPts[1]-self.yPts[0])/3
        epsz = (self.zPts[1]-self.zPts[0])/3

        # then we get the index of xPts
        # and same for z and y
        # the b0Data will be a 3D array
        # indexing is the same for path and b0_values_1D
        
        b0Data = np.zeros((len(self.xPts),len(self.yPts),len(self.zPts),4))
        
        meanField_raw = np.mean(abs(self.fieldDataAlongPath[idx,0]))
                   
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

            
            if self.fieldDataAlongPath[idx,0] == 0:
                self.fieldDataAlongPath[idx,:] == self.fieldDataAlongPath[idx-1,:] if self.fieldDataAlongPath[idx-1,0] !=0 else self.fieldDataAlongPath[idx-2,:]
                print('b0 importer: warning! 0 VALUE detected! pt %d, assigning'%(idx),self.fieldDataAlongPath[idx-1,:])
           
           # replacing the max point by neighbor
            if abs(self.fieldDataAlongPath[idx,0])/meanField_raw>1.25:
                print(self.fieldDataAlongPath[idx,0],'is too big! assigning',self.fieldDataAlongPath[idx-1,:], '!!!')
                self.fieldDataAlongPath[idx,:] = self.fieldDataAlongPath[idx-1,:]
                print('assigned: ',self.fieldDataAlongPath[idx,:], '<+++++')
           
           # replacing the min point by neighbor
            if meanField_raw/abs(self.fieldDataAlongPath[idx,0])>1.25:
                print(self.fieldDataAlongPath[idx,0],'is too small! assigning',self.fieldDataAlongPath[idx-1,:], '!!!')
                self.fieldDataAlongPath[idx,:] = self.fieldDataAlongPath[idx-1,:]
                print('assigned: ',self.fieldDataAlongPath[idx,:], '<-----')


            b0Data[xArg,yArg,zArg,:] = [self.fieldDataAlongPath[idx,0],self.fieldDataAlongPath[idx,1],self.fieldDataAlongPath[idx,2],self.fieldDataAlongPath[idx,3]]

                
            
        b0Data[b0Data==0]=np.NaN    
        # getting mean field
        meanField = np.nanmean(b0Data[:,:,:,0])
        
        # homogeniety
        maxField = np.nanmax(b0Data[:,:,:,0])
        minField = np.nanmin(b0Data[:,:,:,0])
               
        
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

    # ------------------- data parsers -------------------
                
    def parse_field_of_B0_file(self,field_lines):
        #-2.842000 48.057000 -2.319000 48.197000
        self.fieldDataAlongPath = np.zeros((len(field_lines),4))
        for idx, line in enumerate(field_lines):
            b0x = float(line.split(' ')[0])
            b0y = float(line.split(' ')[1])
            b0z = float(line.split(' ')[2])
            b0abs = float(line.split(' ')[3])

            if b0abs == 0 :# sometimes gaussmeter doesnt give the vector
                b0abs = np.sqrt(b0x**2+b0y**2+b0z**2)
                print('OOPS')

            self.fieldDataAlongPath[idx,:] = [b0x,b0y,b0z,b0abs]


        if self.fieldDataAlongPath[0,1] == 0:
            self.fieldDataAlongPath[0,0] = np.nanmean(self.fieldDataAlongPath[1:,0])
            self.fieldDataAlongPath[0,1] = np.nanmean(self.fieldDataAlongPath[1:,1])
            self.fieldDataAlongPath[0,2] = np.nanmean(self.fieldDataAlongPath[1:,2])
            self.fieldDataAlongPath[0,3] = np.nanmean(self.fieldDataAlongPath[1:,3])


                        
    def parse_field_of_CSV_file(self,field_lines):
        # 315.17	152.35	113.75	0	100	0	0
        self.fieldDataAlongPath = np.zeros((len(field_lines),4))
        for idx, line in enumerate(field_lines):
            b0x = float(line.split(',')[3])
            b0y = float(line.split(',')[4])
            b0z = float(line.split(',')[5])
            b0abs = float(line.split(',')[6])

            if b0abs == 0 :# sometimes gaussmeter doesnt give the vector
                b0abs = np.sqrt(b0x**2+b0y**2+b0z**2)
                print('OOPS')
            
            self.fieldDataAlongPath[idx,:] = [b0x,b0y,b0z,b0abs]
            
            
    def parse_header_of_CSV_file(self,header_lines):
        # COSI2 B0 scan						
        # time 2024-05-17 13:21:45.456312						
        # MAGNET CENTER IN LAB: x 265.170 mm	 y 182.350 mm	 z 163.750 mm				
        # MAGNET AXES WRT LAB: alpha 0.00 deg	 beta 0.00 deg	 gamma 0.00 deg				
        # path: C:/cosi-measure/Software/COSI2/dummies/b0_maps/testcsv.path						
        # X[mm]	Y[mm]	Z[mm]	B0_x[mT]	B0_y[mT|	B0_z[mT]	B0_abs[mT]
        self.datetime = header_lines[1].split(' ')[2:3]
        mg_cor_str = header_lines[2].split(':')[1]
        mag_center_x = float(mg_cor_str.split(',')[0].split(' ')[2])
        mag_center_y = float(mg_cor_str.split(',')[1].split(' ')[2])
        mag_center_z= float(mg_cor_str.split(',')[2].split(' ')[2])
        
        mg_euler_str = header_lines[3].split(':')[1]
        mag_alpha = float(mg_euler_str.split(',')[0].split(' ')[2])
        mag_beta = float(mg_euler_str.split(',')[1].split(' ')[2])
        mag_gamma= float(mg_euler_str.split(',')[2].split(' ')[2])

        self.magnet = osi2magnet.osi2magnet(origin=[mag_center_x,mag_center_y,mag_center_z],euler_angles_zyx=[mag_alpha,mag_beta,mag_gamma])

        path_filename_str = str(header_lines[4].split('path:')[1])
        print('warning. path file %s not used. path data taken from csv!'%path_filename_str)
        
        
    
    
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

    def fitSphericalHarmonics(self, maxorder:int, dsv:float, resol:float):
        # fitSphericalHarmonicsShell2_forInterpolation.py
        
        fieldMap = self.b0Data[...,0] #np.load(r'./data/tmp/b0Data.npy')[...,0]
        
        # order of sph harm to consider
        maxOrder = maxorder
        # Diameter spherical volume over which to do the spherical harmonic decomposition
        self.DSV = dsv
        # resolution for rendering the sph harmonics
        resolution = resol
        
        #Create a cartesian coordinate system of where the data points were acquired, maps were acquired with a 5mm resolution 
        fieldMapDims = np.shape(fieldMap)
        
        print(fieldMapDims)
        
        self.xDim_SPH_decomp = np.linspace(0, resolution*(fieldMapDims[0]-1), fieldMapDims[0]) - resolution*(fieldMapDims[0] -1)/2
        self.yDim_SPH_decomp = np.linspace(0, resolution*(fieldMapDims[1]-1), fieldMapDims[1]) - resolution*(fieldMapDims[1] -1)/2
        self.zDim_SPH_decomp = np.linspace(0, resolution*(fieldMapDims[2]-1), fieldMapDims[2]) - resolution*(fieldMapDims[2] -1)/2

        coord = np.meshgrid(self.xDim_SPH_decomp, self.yDim_SPH_decomp, self.zDim_SPH_decomp, indexing='ij')

        #Create a spherical mask for the data
        sphereMask = np.zeros(np.shape(coord[0]), dtype = bool)
        sphereMask[np.square(coord[0]) + np.square(coord[1]) + np.square(coord[2]) <= (self.DSV/2)**2] = 1 
        sphereMask = sphereMask*(~np.isnan(fieldMap))

        # Create a spherical shell mask to consider only data points on the surface of the sphere
        erodedMask = cp.binary_erosion(sphereMask)  # remove the outer surface of the initial spherical mask
        shellMask = np.array(sphereMask^erodedMask, dtype = float)   # create a new mask by looking at the difference between the inital and eroded mask
        shellMask[shellMask == 0] = np.nan  # set points outside mask to 'NaN', works better than setting it to zero for calculating mean fields etc.

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
        #spherHarmCoeff = lsqFit[0]

        #calculate the field from the spherical harmonic decomposition
        decomposedField = np.matmul(spherHarm, spherHarmCoeff)
        print("Inhomogeneity of fit: %.0f ppm" %(abs(1e6*(np.max(decomposedField) - np.min(decomposedField))/np.mean(decomposedField))))

        #See what the difference is between the two decomposed field
        shimmedField = maskedFieldShell - decomposedField
        print("Error: %.0f ppm" %(1e6*(np.max(shimmedField) - np.min(shimmedField))/np.mean(maskedFieldShell)))

        #generate spherical coordinates over entire sphere, not just shell, for plotting
        spherCoordSphere = np.copy(spherCoord)
        spherCoordSphere[spherCoord[...,0] == 0,:] = np.nan
        spherHarm3D = getRealSphericalHarmonics(spherCoordSphere, maxOrder)

        tempSpherHarmCoeff = np.copy(spherHarmCoeff)
        # tempSpherHarmCoeff[4:] = 0
        #calculate the spherical harmonic decomposed field
        decomposedField = np.matmul(spherHarm3D, tempSpherHarmCoeff)*sphereMask

        # calculate difference between decomposed field and measured field
        errorField = maskedField - decomposedField

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
        
        self.coord_grid_fine = np.meshgrid(self.xDim_SPH_fine, self.yDim_SPH_fine, self.zDim_SPH_fine, indexing='ij')
        
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

        print("Inhomogeneity of fit: %.0f ppm" %(abs(1e6*(np.max(decomposedField) - np.min(decomposedField))/np.mean(decomposedField))))
        self.interpolatedField = decomposedField
        DecomposedDataNumpyFilename = 'B0_interpolated.npy'
        saveTmpData(filename = DecomposedDataNumpyFilename,numpyData=self.interpolatedField) 
            
        


    
    def get_shim_positions(self,dsv_for_opt_percent:int,verbose:bool):
        # get the magnets. All magnets. Plot the initial field of all magnets oriented along Y axis.
        from utils import shimming_magnet
        # create array of magnet coordinates in the rings
        # make an array of magnets

        shimRadius          = 276*1e-3# 276*1e-3 <- was set by Tom!      # radius on which the shim magnets are placed
        ringPositions_along_x = [0]#np.linspace(-0.1755,0.1755,4) # <- iter 2          #np.linspace(-0.2295, .2295, 4) #Z positions to place shin rubgs
        magsPerSegment      = 7             # number of magnets peer shim tray segment
        anglePerSegment     = 19.25 #the angular distance in degrees between the furthest magnets in a shim tray (span of magnets in shim tray)
        numSegments         = 12 #corresponds to the number of shim trays

        segmentAngles       = np.linspace(0,360, numSegments, endpoint = False)
        magAngles           = np.linspace(-anglePerSegment/2, anglePerSegment/2, magsPerSegment) 

        positions = []
        self.shim_magnets = []
        
        self.interpolatedField_masked = self.interpolatedField
        self.interpolatedField_masked[self.interpolatedField==np.nan] = 0

        # fields of shim magnets
        initialField = self.interpolatedField
        grid = self.coord_grid_fine

        numMags = 0
        for ringPosition in ringPositions_along_x:
            for segmentAngle in segmentAngles:
                for magAngle in magAngles:
                    numMags+=1
        print('%d magnets to rotate'%numMags)

        for ringPosition in ringPositions_along_x:
            for segmentAngle in segmentAngles:
                for magAngle in magAngles:
                    xpos = ringPosition
                    ypos = shimRadius*np.cos((segmentAngle+magAngle)*np.pi/180)
                    zpos = shimRadius*np.sin((segmentAngle+magAngle)*np.pi/180)
                    positions.append((xpos,ypos,zpos))
                    #initangle = 1.4960 # zeros approximation # all shim magnets are initially along Y #np.random.randint(-3,3)*np.pi/16
                    my_magnet = shimming_magnet.shimming_magnet(position=[xpos,ypos,zpos], rotation_yz = 0)
                    my_magnet.render_field(grid=grid)
                    my_magnet.rotate_field(rotation_yz=0)
                    self.shim_magnets.append(my_magnet)

        print('fields of shim magnets rendered.')
        self.magnet_positions = positions
 
        
        nummagnets= len(self.shim_magnets)
        
        self.fldsY = np.zeros((np.shape(self.shim_magnets[0].B0[:,:,:,1])+(nummagnets,)))
        self.fldsZ = np.zeros((np.shape(self.shim_magnets[0].B0[:,:,:,2])+(nummagnets,)))

        for i,magnet in enumerate(self.shim_magnets):
            self.fldsY[:,:,:,i] = (magnet.B0[:,:,:,1])
            self.fldsZ[:,:,:,i] = (magnet.B0[:,:,:,2])

        
        initial_rotation = 0
        shimField = my_magnet.Brot*0
        for magnet in self.shim_magnets:
            magnet.rotate_field(initial_rotation)
            shimField += my_magnet.Brot

        self.shimField = shimField

        print('shim field computed, all shim magnets point along Y (rotation_yz=%.2f)'%initial_rotation)

        print('shape of magnets field data: ',np.shape(self.fldsY), np.shape(self.fldsZ))
                

        print('masking the magnets fields with the spherical mask')
        
        
        print('masking')
        DSV=self.DSV*dsv_for_opt_percent/100
        print('optimization sphere radius, mm: %.2f'%DSV)
        sphereMask = np.zeros(np.shape(self.coord_grid_fine[0]), dtype = bool)
        sphereMask[np.square(self.coord_grid_fine[0]) + np.square(self.coord_grid_fine[1]) + np.square(self.coord_grid_fine[2]) <= (DSV/2)**2] = 1 
        sphereMask = np.asarray(sphereMask, dtype=np.double)
        sphereMask[sphereMask == 0] = np.nan



        for i,magnet in enumerate(self.shim_magnets):
            self.fldsY[:,:,:,i] = np.multiply(sphereMask, self.fldsY[:,:,:,i])
            self.fldsZ[:,:,:,i] = np.multiply(sphereMask, self.fldsZ[:,:,:,i])


        totfield = self.interpolatedField+self.shimField
        self.errorField = totfield-np.nanmean(totfield)
        
        print('least square optimization of the magnet roatations.')
               
        #self.optimize_magnet_rotations(verbose=2 if verbose else 0)
        self.optimize_magnet_rotations_quickly(dsv_for_opt_percent)
        

    def optimize_magnet_rotations_quickly(self,dsv_for_opt_percent):
        
        maxIter = 100000
        def dataFitting(shimVector):
            # shimVector_stacked = np.hstack((np.cos(shimVector), np.sin(shimVector)))
            shimField = np.matmul(maskedFields,np.hstack((np.cos(shimVector), np.sin(shimVector)))) + initialFieldMasked
            # return (shimField - np.mean(shimField))
            #print(shimVector[0])
            
            if np.nan in shimField:
                print('WTF')
                return 1e23
            
            return np.square(((shimField)/np.mean(shimField)) -1)*1e9
            # return np.std(shimField)
                    
        
        initialField = self.interpolatedField
        numMags = len(self.shim_magnets)
        magnetFields = np.zeros((np.shape(initialField)+(2,numMags)), dtype = np.float32)
        positions = self.magnet_positions
        bRem = 1.35 # [T] remanence field of shim magnets
        mu                  = 1e-7
        magSizeOuter        = 6*1e-3        # [m] size of shim magnets
        dip_mom = b0V5.magnetization(bRem, magSizeOuter)
        dip_vec = mu*np.array([dip_mom, 0]) # along Y
        
        DSV = self.DSV*dsv_for_opt_percent/100
        resolution = 1/self.resolution
        
        
        ''' Mask generation'''


        xDim3D, yDim3D, zDim3D = self.coord_grid_fine
        spherCoord = cartToSpher(np.stack((yDim3D,xDim3D, zDim3D), axis = -1))

        #Apply mask to data
        mask = (np.round(spherCoord[...,0],4) <= (DSV/2)).astype(float)
        # mask = (np.square(xDim3D)/((optVol[0]/2)**2) + np.square(yDim3D)/((optVol[1]/2)**2) + np.square(zDim3D)/((optVol[2]/2)**2)  <= 1).astype(float)
        halfMask = mask#*((zDim3D<=0).astype(float))
        erodedMask = cp.binary_erosion(halfMask.astype(bool))                    # remove the outer surface of the initial spherical mask
        halfMask = np.array(halfMask.astype(bool)^erodedMask, dtype = float)   # create a new mask by looking at the difference between the inital and eroded mask
        halfMask[halfMask == 0] = np.nan    
        mask[mask == 0] = np.nan
                
        
        for idx1, position in enumerate(positions):
            magnetFields[:,:,:,0,idx1] = b0V5.singleMagnet(position, dipoleMoment = dip_vec, simDimensions = (DSV*1e-3,DSV*1e-3,DSV*1e-3), resolution = resolution)[...,1]
            magnetFields[:,:,:,1,idx1] = b0V5.singleMagnet(position, dipoleMoment = dip_vec, simDimensions = (DSV*1e-3,DSV*1e-3,DSV*1e-3), resolution = resolution)[...,2]
        
        print(np.shape(magnetFields))
        print("Field calculation complete")
        print("Number of angles to optimise: %d"%(len(positions)))
        print("Number of field points: %d"%(np.nansum(halfMask)))


        magnetFields *= 1e3
        maskedFields = magnetFields[halfMask == 1, :,:].astype(float)
        maskedFields = np.hstack((maskedFields[:,0,:],maskedFields[:,1,:]))
        
        print('masked fields: ',np.shape(maskedFields))

        initialFieldMasked = initialField[halfMask == 1]
        
        if self.vector_of_magnet_rotations is not None:
            initialGuess = self.vector_of_magnet_rotations
        else:
            initialGuess = np.zeros(int(np.size(maskedFields,-1)/2))

        lsqData = least_squares(dataFitting, initialGuess, ftol=0, xtol=0,max_nfev=maxIter, verbose=2,bounds = (initialGuess*0,initialGuess*0+np.pi*2))
        
        optimized_rotation_vector = lsqData.x
        self.vector_of_magnet_rotations = optimized_rotation_vector
        
        initialField *= mask
        tempField = initialField
        
        optimizedField = np.matmul(magnetFields[...,0,:], np.cos(lsqData.x)) + np.matmul(magnetFields[...,1,:], np.sin(lsqData.x))
        optimizedField *= mask
        
        shimmedField = tempField + optimizedField
        shimmedField *= mask
        
        initialMean = np.nanmean(initialField)
        initialHomogeneity = 1e6*(np.nanmax(initialField) - np.nanmin(initialField))/initialMean

        optimizedMean = np.nanmean(shimmedField)
        optimizedHomogeneity = 1e6*(np.nanmax(shimmedField) - np.nanmin(shimmedField))/optimizedMean

        print("Initial mean %.2f mT, initial homogeneity %d ppm"%(initialMean, initialHomogeneity))
        print("Optimized mean %.2f mT, optimized homogeneity %d ppm"%(optimizedMean, optimizedHomogeneity))
        print("Reduction in standard deviation: %.1f percent"%(100*(1-np.nanstd(shimmedField)/np.nanstd(initialField))))
        
        
        print('now render the expensive field by turning the shim magnets')
        for idx, shim_magnet in enumerate(self.shim_magnets):
            shim_magnet.rotation_yz = optimized_rotation_vector[idx]
            shim_magnet.render_field(grid=self.coord_grid_fine)
            #shim_magnet.rotate_field(optimized_rotation_vector[idx]) # fair rendering
            self.shimField += shim_magnet.B0[:,:,:,2]#shim_magnet.Brot

        self.shimField = np.multiply(self.sphere_mask, self.shimField)
        print('expensive shim field rendered.')



        self.errorField = self.interpolatedField + self.shimField
        print('error field rendered')
        self.homogeneity_shimmed = abs(np.nanmax(self.errorField)-np.nanmin(self.errorField))/np.nanmean(self.errorField)*1e6
        print('optimized homogeneity: %d ppm' %self.homogeneity_shimmed)
        self.mean_field_shimmed = np.nanmean(self.errorField)
        print('optimized mean field: %.3f mT' %self.mean_field_shimmed)
        self.cheapField = np.matmul(self.fldsZ,np.cos(optimized_rotation_vector)) + np.matmul(self.fldsY,np.sin(optimized_rotation_vector)) + self.interpolatedField_masked
        


        print('now save that rotating vector!!!')
        #for rot in optimized_rotation_vector:
        #    print('%.9f,'%rot)

        print('optimized rotations assigned to class variable')
        self.vector_of_magnet_rotations = optimized_rotation_vector
        


    def optimize_magnet_rotations(self,verbose=0):
        # this turns all magnets and looks at the inhomogeneity of the resulting shim.
        # best turning combination gives smallest inhomogeneity.



        def _calculate_shimming_error(vector_of_magnet_rotations):
            '''calculate the shim field of shim magnets that are turned as vector_of_magnet_rotations says'''
            #shimField = np.matmul(self.maskedFields,np.hstack((np.cos(vector_of_magnet_rotations), np.sin(vector_of_magnet_rotations)))) + self.interpolatedField_masked
            #self.cheapField = np.matmul(self.maskedFields,np.hstack((np.sin(vector_of_magnet_rotations), np.cos(vector_of_magnet_rotations)))) + self.interpolatedField_masked
            #return np.square(((self.cheapField)/np.mean(self.cheapField)) -1)*1e9
            self.cheapField = np.matmul(self.fldsZ,np.cos(vector_of_magnet_rotations)) + np.matmul(self.fldsY,np.sin(vector_of_magnet_rotations)) + self.interpolatedField_masked
            self.cheapField = self.cheapField[~np.isnan(self.cheapField)]


            #print((np.square((self.cheapField/np.mean(self.cheapField)-1))*1e14)[0:14])

            return np.square((self.cheapField/np.mean(self.cheapField)-1))*1e9
            
            

        if self.vector_of_magnet_rotations is None:
            print('no vector of magnet rotations given, starting from 1.496 rad for all')
            vector_of_magnet_rotations = np.zeros(len(self.shim_magnets))
    
            for idx, shim_magnet in enumerate(self.shim_magnets): 
                vector_of_magnet_rotations[idx] = 0#1.4960

        else:
            print('vector of rotations is given, starting with optimized values.')
            vector_of_magnet_rotations = self.vector_of_magnet_rotations


        print('magnet rotations captured. %d magnets to rotate'%len(vector_of_magnet_rotations))
        print('initial field inhomogeneity: %d ppm'
              %abs((np.nanmax(self.interpolatedField[:,:,:])-
                   np.nanmin(self.interpolatedField[:,:,:]))/np.nanmean(self.interpolatedField[:,:,:])*1e6))

        #print('ihgomogeneity with 0-approx shimming: %d ppm'%_calculate_shimming_error(vector_of_magnet_rotations))

        print('now do the optimization with least squares.')

        initialGuess = vector_of_magnet_rotations

        lsqData = least_squares(_calculate_shimming_error, initialGuess, ftol=1e-32, xtol=0, max_nfev=64, verbose=verbose, bounds=(vector_of_magnet_rotations*0,vector_of_magnet_rotations*0+2*np.pi))

        optimized_rotation_vector = lsqData.x

        print('optimal rotations found. rendering field.')

        self.shimField = np.zeros(np.shape(self.coord_grid_fine[0]), dtype=np.float32)


        print(optimized_rotation_vector)

        for idx, shim_magnet in enumerate(self.shim_magnets):
            shim_magnet.rotation_yz = optimized_rotation_vector[idx]
            shim_magnet.render_field(grid=self.coord_grid_fine)
            shim_magnet.rotate_field(optimized_rotation_vector[idx]) # fair rendering
            self.shimField += shim_magnet.B0[:,:,:,2]#shim_magnet.Brot

        self.shimField = np.multiply(self.sphere_mask, self.shimField)
        print('expensive shim field rendered.')

        homocheap = (np.nanmax(self.cheapField)-np.nanmin(self.cheapField))/np.nanmean(self.cheapField)*1e6
        print('homo of cheap shimmed field: %.0f ppm'%homocheap)


        self.errorField = self.interpolatedField + self.shimField
        print('error field rendered')
        self.homogeneity_shimmed = abs(np.nanmax(self.errorField)-np.nanmin(self.errorField))/np.nanmean(self.errorField)*1e6
        print('optimized homogeneity: %d ppm' %self.homogeneity_shimmed)
        self.mean_field_shimmed = np.nanmean(self.errorField)
        print('optimized mean field: %.3f mT' %self.mean_field_shimmed)
        self.cheapField = np.matmul(self.fldsZ,np.cos(optimized_rotation_vector)) + np.matmul(self.fldsY,np.sin(optimized_rotation_vector)) + self.interpolatedField_masked
        


        print('now save that rotating vector!!!')
        #for rot in optimized_rotation_vector:
        #    print('%.9f,'%rot)

        print('optimized rotations assigned to class variable')
        self.vector_of_magnet_rotations = optimized_rotation_vector

        def do_zeros_approximation():# zeros approximation gave rotation of 1.4960 for all magnets
            inhomos = []
            angles = []
            for angle in np.linspace(0,2*np.pi,64):
                vector_of_magnet_rotations = vector_of_magnet_rotations*0+angle
                inhomogeneity_of_shimmed_field = _calculate_shimming_error(vector_of_magnet_rotations)
                print('--- all magnets turn by %.4f rad -> %d ppm ---'%(angle,inhomogeneity_of_shimmed_field))
                inhomos.append(inhomogeneity_of_shimmed_field)
                angles.append(angle)
            
            minidx = inhomos.index(min(inhomos))
            try:
                print(min(inhomos),' for turn # ',minidx, 'which is ',angles[minidx])
            except Exception as e:
                print(e)

            return vector_of_magnet_rotations


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
        
    def import_from_csv(self,b0_filename: str):
        print('importing b0 object from csv file%s'%b0_filename)

        # make an empty instance of b0 and get the b0 values from the csv file.
        self.__init__()        
        self.filename = b0_filename
        with open(b0_filename) as file:
                raw_B0_data = file.readlines()     
                headerlength = 0
                for line in raw_B0_data:
                    if line[0] == '#':
                        headerlength += 1
                        
                header_lines = raw_B0_data[0:headerlength]    
                field_lines = raw_B0_data[headerlength:]
                self.parse_header_of_CSV_file(header_lines)
                self.parse_field_of_CSV_file(field_lines) 
         
        # import the path from the path file
        self.path = pth.pth(csv_filename = b0_filename)

                
        
            
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
