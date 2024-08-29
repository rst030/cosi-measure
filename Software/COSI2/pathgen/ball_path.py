import numpy as np


class gradient_path(object):
    points = np.array([]) # 0,0,0; 1,1,1; 2,2,2; ...
    def __init__(self,filename_input,center_point_input,radius_input,radius_npoints_input,axis) -> None:
        self.filename = filename_input
        self.center = center_point_input
        self.radius = radius_input
        self.radius_npoints = radius_npoints_input
        self.axis = axis

        # writes to a file already
        #self.make_cross(center = self.center, radius = self.radius, radius_npoints = self.radius_npoints)
        self.make_line(center = self.center, radius = self.radius, radius_npoints = self.radius_npoints, axis = self.axis)
    
    def make_line(self, center, radius:float, radius_npoints:int, axis='x'):
        npoints = radius_npoints

        x0 = center[0]
        y0 = center[1]
        z0 = center[2]

        x = np.linspace(x0-radius, x0+radius, 2*npoints)
        y = np.linspace(y0-radius, y0+radius, 2*npoints)
        z = np.linspace(z0-radius, z0+radius, 2*npoints) 

        with open(self.filename, 'w+') as f:
            # if axis==x then a line along x from x0-r to 
            if axis == 'x':
                for _x in x:
                    g0 =  'x%.2f y%.2f z%.2f\n'%(_x,y0,z0)
                    f.write( g0 ) 
            if axis == 'y':
                for _y in y:
                    g0 =  'x%.2f y%.2f z%.2f\n'%(x0,_y,z0)
                    f.write( g0 ) 
            if axis == 'z':
                for _z in z:
                    g0 =  'x%.2f y%.2f z%.2f\n'%(x0,y0,_z)
                    f.write( g0 )     
        print('line along %s written'% axis)           
     
            

    def make_cross(self, center, radius:float, radius_npoints:int):

        npoints = radius_npoints

        x0 = center[0]
        y0 = center[1]
        z0 = center[2]
        
        x = np.linspace(x0-radius, x0+radius, 2*npoints)
        y = np.linspace(y0-radius, y0+radius, 2*npoints)
        z = np.linspace(z0-radius, z0+radius, 2*npoints) 
        
        with open(self.filename, 'w+') as f:
            # first a line along x from x0-r to 
            for _x in x:
                g0 =  'x%.2f y%.2f z%.2f\n'%(_x,y0,z0)
                f.write( g0 ) 
            for _y in y:
                g0 =  'x%.2f y%.2f z%.2f\n'%(x0,_y,z0)
                f.write( g0 ) 
            for _z in z:
                g0 =  'x%.2f y%.2f z%.2f\n'%(x0,y0,_z)
                f.write( g0 )     
                
            print('Cross pathfile is written.')           
                

class rect_path(object):
    points = np.array([]) # 0,0,0; 1,1,1; 2,2,2; ...

    def __init__(self,filename_input,center_point_input,radius_input,radius_npoints_input,check_extremes=None) -> None:
        self.filename = filename_input
        self.center = center_point_input
        self.radius = radius_input
        self.radius_npoints = radius_npoints_input
        self.r = []

        # writes to a file already
        self.make_cube(center = self.center, radius = self.radius, radius_npoints = self.radius_npoints,check_extremes=check_extremes)


    def make_cube(self, center, radius:float, radius_npoints:int,check_extremes: bool):

        def G0(x:float,y:float,z:float):
            g0 =  'x%.2f y%.2f z%.2f\n'%(x,y,z)
            self.r.append([x,y,z])
            return(g0)

        def checkBounds(x,y,z,center,radius):
            """check if coordinates are in sphere
            returns True if x,y,z are in sphere
            
            * center - array of x,y,z coordinates of the center of the sphere
            * radius - radius of the sphere """

            if abs(x-center[0]) < abs(radius):
                if abs(y-center[1]) < abs(radius):
                    if abs(z-center[2]) < abs(radius):
                        return True
            else:
                #print(r2,'>',radius**2)
                return False

        def write_extremes(): # write extreme points of the path to the path file
            xc = center[0]
            yc = center[1]
            zc = center[2]
            
            center_pt   = [xc,yc,zc] # center point
            right_pt    = [xc-radius,yc,zc] # right pt
            left_pt     = [xc+radius,yc,zc] # left pt
            back_pt     = [xc,yc-radius,zc] # back pt
            front_pt    = [xc,yc+radius,zc] # front pt
            top_pt      = [xc,yc,zc+radius] # top pt
            bottom_pt   = [xc,yc,zc-radius] # bottom pt
            
            
            x,y,z = center_pt
            f.write( G0(x=x, y=y, z=z) )
            x,y,z = right_pt
            f.write( G0(x=x, y=y, z=z) )
            x,y,z = left_pt
            f.write( G0(x=x, y=y, z=z) )
            x,y,z = back_pt
            f.write( G0(x=x, y=y, z=z) )
            x,y,z = front_pt
            f.write( G0(x=x, y=y, z=z) )
            x,y,z = top_pt
            f.write( G0(x=x, y=y, z=z) )
            x,y,z = bottom_pt
            f.write( G0(x=x, y=y, z=z) )

        npoints = radius_npoints

        xSteps = np.linspace(center[0]-radius, center[0]+radius, 2*npoints+1)
        ySteps = np.linspace(center[1]-radius, center[1]+radius, 2*npoints+1)
        zSteps = np.linspace(center[2]-radius, center[2]+radius, 2*npoints+1)

        #print(min(xSteps),max(xSteps))
        #print(min(ySteps),max(ySteps))
        #print(min(zSteps),max(zSteps))

        with open(self.filename, 'w+') as f:

            if check_extremes:
                write_extremes()

            xIsReversed = False
            yIsReversed = False
            for z in zSteps:
                if yIsReversed:
                    yIsReversed = False
                    for y in reversed(ySteps):
                        if xIsReversed:
                            xIsReversed = False
                            for x in reversed(xSteps):
                                if checkBounds(x,y,z,center,radius):
                                    f.write( G0(x=x, y=y, z=z) )
                        else:
                            xIsReversed = True
                            for x in xSteps:
                                if checkBounds(x,y,z,center,radius):
                                    f.write( G0(x=x, y=y, z=z) )
                else:
                    yIsReversed = True
                    for y in ySteps:
                        if xIsReversed:
                            xIsReversed = False
                            for x in reversed(xSteps):
                                if checkBounds(x,y,z,center,radius):
                                    f.write( G0(x=x, y=y, z=z) )
                        else:
                            xIsReversed = True
                            for x in xSteps:
                                if checkBounds(x,y,z,center,radius):
                                    f.write( G0(x=x, y=y, z=z) )              
            if check_extremes:
                write_extremes()
                print('extreme points written to path file.')

        print('Cube pathfile is written.')



class ball_path(object):
    # 

    points = np.array([]) # 0,0,0; 1,1,1; 2,2,2; ...

    def __init__(self,filename_input,center_point_input,radius_input,radius_npoints_input,check_extremes=None) -> None:
        self.filename = filename_input
        self.center = center_point_input
        self.radius = radius_input
        self.radius_npoints = radius_npoints_input
        self.r = []

        # writes to a file already
        self.make_ball(center = self.center, radius = self.radius, radius_npoints = self.radius_npoints,check_extremes=check_extremes)


    def make_ball(self, center, radius:float, radius_npoints:int,check_extremes: bool):

        def G0(x:float,y:float,z:float):
            g0 =  'x%.2f y%.2f z%.2f\n'%(x,y,z)
            self.r.append([x,y,z])
            return(g0)

        def checkBounds(x,y,z,center,radius):
            """check if coordinates are in sphere
            returns True if x,y,z are in sphere
            
            * center - array of x,y,z coordinates of the center of the sphere
            * radius - radius of the sphere """

            r2 = abs(x-center[0])**2 + abs(y-center[1])**2 + abs(z-center[2])**2
            if r2 <= abs(radius)**2:
                return True
            else:
                #print(r2,'>',radius**2)
                return False

        def write_extremes(): # write extreme points of the path to the path file
            xc = center[0]
            yc = center[1]
            zc = center[2]
            
            center_pt   = [xc,yc,zc] # center point
            right_pt    = [xc-radius,yc,zc] # right pt
            left_pt     = [xc+radius,yc,zc] # left pt
            back_pt     = [xc,yc-radius,zc] # back pt
            front_pt    = [xc,yc+radius,zc] # front pt
            top_pt      = [xc,yc,zc+radius] # top pt
            bottom_pt   = [xc,yc,zc-radius] # bottom pt
            
            
            x,y,z = center_pt
            f.write( G0(x=x, y=y, z=z) )
            x,y,z = right_pt
            f.write( G0(x=x, y=y, z=z) )
            x,y,z = left_pt
            f.write( G0(x=x, y=y, z=z) )
            x,y,z = back_pt
            f.write( G0(x=x, y=y, z=z) )
            x,y,z = front_pt
            f.write( G0(x=x, y=y, z=z) )
            x,y,z = top_pt
            f.write( G0(x=x, y=y, z=z) )
            x,y,z = bottom_pt
            f.write( G0(x=x, y=y, z=z) )


        npoints = radius_npoints

        xSteps = np.linspace(center[0]-radius, center[0]+radius, 2*npoints+1)
        ySteps = np.linspace(center[1]-radius, center[1]+radius, 2*npoints+1)
        zSteps = np.linspace(center[2]-radius, center[2]+radius, 2*npoints+1)

        #print(min(xSteps),max(xSteps))
        #print(min(ySteps),max(ySteps))
        #print(min(zSteps),max(zSteps))

        with open(self.filename, 'w+') as f:

            if check_extremes:
                write_extremes()

            xIsReversed = False
            yIsReversed = False
            for z in zSteps:
                if yIsReversed:
                    yIsReversed = False
                    for y in reversed(ySteps):
                        if xIsReversed:
                            xIsReversed = False
                            for x in reversed(xSteps):
                                if checkBounds(x,y,z,center,radius):
                                    f.write( G0(x=x, y=y, z=z) )
                        else:
                            xIsReversed = True
                            for x in xSteps:
                                if checkBounds(x,y,z,center,radius):
                                    f.write( G0(x=x, y=y, z=z) )
                else:
                    yIsReversed = True
                    for y in ySteps:
                        if xIsReversed:
                            xIsReversed = False
                            for x in reversed(xSteps):
                                if checkBounds(x,y,z,center,radius):
                                    f.write( G0(x=x, y=y, z=z) )
                        else:
                            xIsReversed = True
                            for x in xSteps:
                                if checkBounds(x,y,z,center,radius):
                                    f.write( G0(x=x, y=y, z=z) )              
            if check_extremes:
                write_extremes()
                print('extreme points written to path file.')

        print('Ball pathfile is written.')