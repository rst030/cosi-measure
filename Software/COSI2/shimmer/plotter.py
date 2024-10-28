'''this is the b0 class.
    fields:
        b0 magnetic field [mT,mT,mT] 
        probed along the certain path [mm,mm,mm]
    methods:
        transformation of the path coordinates
        peeling of the shell from the volumetric data
    '''
    
import numpy as np
from matplotlib import pyplot as plt
import b0

class plotter(object):
    '''plotter object. plots slices of b0 object's magnetic_field'''
    b0map = None # b0 map
    fig = None # mpl figure 
    axes = None # axes of the figure (many)
    
    
    def __init__(self,map_to_plot:None) -> None:
        
        figure = plt.figure(0)
        if map_to_plot is not None:
            self.b0map = map_to_plot
        else:
            self.b0map = b0.b0() # dummy b0 at first
        
        
        x_points = self.b0map.x_coordinate_points
        y_points = self.b0map.y_coordinate_points # y -//-
        z_points = self.b0map.z_coordinate_points # z -//-

        field_to_plot = self.b0map.magnetic_field[:,:,:,2] # b0Data is the raw data of the map. 2nd (=z) component is of interest by default


        self.fig, self.axes = plt.subplots(nrows=5, ncols=5, figsize=(10,8))


        
        # plot the mapped points, slices along X.       
        for i in range(5):
            for j in range(5):

                xxx = self.axes[i,j].imshow(np.transpose(field_to_plot[i*5+j,:,:]),clim=[np.nanmin(field_to_plot),np.nanmax(field_to_plot)],origin = 'lower',extent=[min(y_points),max(y_points),min(z_points),max(z_points)],cmap='viridis')
                #xxx = axes[i,j].contourf(z,y,FieldMeasured_shimmed[i*5+j,:,:],vmin=np.nanmin(FieldMeasured_shimmed),vmax=np.nanmax(FieldMeasured_shimmed))
                
                self.axes[i,j].text(0.5, 0.3, 'X=%.1f'%x_points[i*5+j], horizontalalignment='center', verticalalignment='center', transform=self.axes[i,j].transAxes)
                self.axes[i,j].xaxis.set_visible(False)
                self.axes[i,j].yaxis.set_visible(False)
                if i==4:
                    self.axes[i,j].xaxis.set_visible(True)
                if j==0:
                    self.axes[i,j].yaxis.set_visible(True)
                    
                self.axes[i,j].set_xlabel('Y')
                self.axes[i,j].set_ylabel('Z')
                self.axes[i,j].set_aspect(1)
                
                if i*5+j >=len(x_points)-1:
                    #plt.colorbar(mappable=xxx)
                    self.axes[i,j].text(2, 3, 'Dummy B0 map\n slices along X axis\n', horizontalalignment='center', verticalalignment='center', transform=self.axes[i,j].transAxes)
                    break

        
        plt.xlabel('Y')
        plt.ylabel('Z')
        
        plt.subplots_adjust(hspace=0.3)
        plt.subplots_adjust(wspace=-0.7)

        plt.show()
