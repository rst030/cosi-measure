import numpy as np

class ball_path(object):
    points = np.array([]) # 0,0,0; 1,1,1; 2,2,2; ...

    def __init__(self,filename_input,center_point_input,radius_input,radius_npoints_input) -> None:
        self.filename = filename_input
        self.center = center_point_input
        self.radius = radius_input
        self.radius_npoints = radius_npoints_input

        # writes to a file already
        self.make_ball(center = self.center, radius = self.radius, radius_npoints = self.radius_npoints)



    def make_ball(self, center, radius:float, radius_npoints:int):

        npoints = radius_npoints

        x = np.linspace(center[0]-radius, center[0]+radius, 2*npoints)
        y = np.linspace(center[1]-radius, center[1]+radius, 2*npoints)
        z = np.linspace(center[2]-radius, center[2]+radius, 2*npoints)

        xx, yy, zz = np.meshgrid(x,y,z)

        res = (xx-center[0])**2+(yy-center[1])**2+(zz-center[2])**2<=radius**2
        #print(np.shape(res))
        #print(res)


        with open(self.filename, 'w+') as f:
            snakeup = True
            for iz in range(len(z)):
                for iy in range(len(y)):
                    snakeup =  not snakeup
                    for ix in range(len(x)):
                        if res[ix,iy,iz]:
                            if snakeup:
                                g0 =  'x%.2f y%.2f z%.2f\n'%(x[ix],y[iy],z[iz])
                                print('slice z=%.2f snake UP\n'%z[iz],g0)
                            else:
                                g0 =  'x%.2f y%.2f z%.2f\n'%(x[len(x)-ix],y[iy],z[iz])
                                print('slice z=%.2f snake DOWN\n'%z[iz],g0)
                            f.write( g0 ) 


        print('Ball pathfile is written.')
