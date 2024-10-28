'''this is the b0 class.
    fields:
        b0 magnetic field [mT,mT,mT] 
        probed along the certain path [mm,mm,mm]
    methods:
        transformation of the path coordinates
        peeling of the shell from the volumetric data
    '''
    
import numpy as np

class b0(object):
    '''B0 map object. field along path, at the given coordinates'''
    
    x_coordinate_points = None # 1D array of x coordinates
    y_coordinate_points = None # 1D array of y coordinates
    z_coordinate_points = None # 1D array of z coordinates

    coordinate_grid = None # 3D coordinate grid with indexing [i,j,k]
    magnetic_field = None # values of magnetic field corresponding to B[i,j,k] = B(x=x_i, y=y_j, z=z_k)
    
    filename = None # in case importing from a file
    
    def __init__(self) -> None:
        
        self.x_coordinate_points = np.linspace(start=-1,stop=1,num=25)
        self.y_coordinate_points = np.linspace(start=-2,stop=2,num=36)
        self.z_coordinate_points = np.linspace(start=-4,stop=4,num=49)
        
        self.coordinate_grid = np.meshgrid(self.x_coordinate_points,self.y_coordinate_points,self.z_coordinate_points,indexing='ij')
        
        self.magnetic_field = np.zeros((len(self.x_coordinate_points),len(self.y_coordinate_points),len(self.z_coordinate_points),4))
        
        # generate a default field: x component [0] equals x coordinate.
        #                           y component [1] equals y coordinate.
        #                           z component [2] equals z coordinate. 
                
        self.magnetic_field[:,:,:,0] = self.coordinate_grid[0] # x meshgrid
        self.magnetic_field[:,:,:,1] = self.coordinate_grid[1] # y meshgrid
        self.magnetic_field[:,:,:,2] = self.coordinate_grid[2] # z meshgrid
        
        
    def create_anomaly(self,x=0,y=0,z=0,strength=10,length=0,eps = 0.05):
        '''
        create an extreme value at the given coordinates'''
        x_indices = np.where(abs(self.x_coordinate_points - x) <= eps + length)[0][:]
        y_indices = np.where(abs(self.y_coordinate_points - y) <= eps + length)[0][:]
        z_indices = np.where(abs(self.z_coordinate_points - z) <= eps + length)[0][:]
        
        print(x_indices,y_indices,z_indices)
        for x_idx in x_indices:
            for y_idx in y_indices:
                for z_idx in z_indices:
                    self.magnetic_field[x_idx,y_idx,z_idx,:] = [strength,strength,strength,strength]


    def load_from_csv(self,csv_filename:str):
        '''
        load path data and b0 data from a csv file.
        just get the x,y,z,b0x,b0y,b0z,b0abs values at first.
        '''
        self.filename = csv_filename
        with open(self.filename) as file:
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
        
        