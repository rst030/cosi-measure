'''
    Ilia Kulikov
    16 June 2022
    ilia.kulikov@fu-berlin.de
plotter.
matplotlib based.
mpl window is imbedded into the parent that has to be passed to the constructor.
    '''

import matplotlib
# if conflict with 3d axes after running Tom's script
# sudo apt remove python3-matplotlib
# sudo apt remove python3-matplotlib

import numpy as np

import chg
import cv
import tp
import pth
import b0

import cosimeasure # for plotting path irl
import osi2magnet


matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QSizePolicy, QVBoxLayout
from PyQt5.QtWidgets import QWidget


class PlotterCanvas(FigureCanvas):
    '''Plotter based on FigureCanvasQTAgg'''
    xlabel = 'pirates'
    ylabel = 'crocodiles'
    title = 'ultimate grapfh'
    parent = None # parent widget, [have to] pass it on construction for live updates
    plotType = 'GEN' # available: 'GEN,CV,CHG,EPR,TP'
    fig = Figure
    colorbar_object = None

    def __init__(self, plotType:str):
        self.plotType = plotType # assign and dont worry anymore!
        self.fig = Figure(figsize=(16, 16), dpi=100)
        fig = self.fig

        if plotType == 'PTH' or plotType == 'B0M':
            self.axes = fig.add_subplot(1,1,1,projection='3d')
            self.axes.set_aspect("equal")
            self.colorbar_object = None
            self.axes.set_proj_type('persp', focal_length=0.42)  # FOV = 157.4 deg
            self.axes.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            self.axes.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            self.axes.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            
            self.fig.subplots_adjust(left=0.0,right=1.0,
                            bottom=0.0,top=1.0,
                            hspace=0.0,wspace=0.0)
            

        else:
            self.axes = fig.add_subplot(111)
#        fig.subplots_adjust(left = 0.18, right=0.99, top=0.94, bottom=0.1)

        FigureCanvas.__init__(self, fig)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        if plotType == 'TP': # if a tunepicture plotter:
            tightrect = (0.01,0.06,0.99,1)
            self.axes.set_yticks([])
        else:
            tightrect = (0.16, 0.1, 0.99, 0.9)

        fig.tight_layout(rect = tightrect)
        self.compute_initial_figure()

    def parent(self):
        return QWidget()

    def clear(self):
        self.axes.cla()

    def set_title(self,title:str):
        self.title = title
        self.axes.set_title(title)
        self.update_plotter()

    def compute_initial_figure(self):
        self.clear()
        if self.plotType == 'GEN':
            pass
        if self.plotType == 'CV':
            self.preset_CV()
        if self.plotType == 'CHG':
            self.preset_CHG()
        if self.plotType == 'EPR':
            self.preset_EPR()
        if self.plotType == 'TP':
            self.preset_TP()
        if self.plotType == 'PTH':
            self.preset_PTH()
        if self.plotType == 'B0M':
            self.preset_B0M()
        

        self.axes.set_xlabel(self.xlabel)
        self.axes.set_ylabel(self.ylabel)
        self.axes.set_title(self.title)


    def preset_CV(self):
        self.clear()
        self.xlabel = 'Volttage [V]'
        self.ylabel = 'Current [A]'
        self.title = 'CV'
        self.axes.grid()

        # plot sample cv
        cvDummy = cv.cv('./dummies/DEPOSITION_DEMO.csv')
        self.plotCv(cvDummy)


    def preset_CHG(self):
        self.clear()
        self.xlabel = 'Time [s]'
        self.ylabel = 'Voltage [V]'
        self.title = 'CHG'
        self.axes.grid()
        chgDummy = chg.chg('./dummies/lipton_4_CHG_DCG.csv')
        self.plotChg(chgDummy)

    def preset_TP(self):
        self.clear()
        self.xlabel = '$\Delta$ f [MHz]'
        self.ylabel = ''
        self.title = ''
        self.axes.set_yticks([])
        tpDummy = tp.tp('./dummies/TP.csv') #TP!
        self.plotTpData(tpDummy)


    def preset_PTH(self):
        self.clear()
        self.xlabel = 'X COSI'
        self.ylabel = 'Y COSI'
        self.zlabel = 'Z COSI'
        self.title = 'dummy path'
        pthDummy = pth.pth('./dummies/pathfiles/2021-10-14_PathfileTest_Spherical.path')
        self.plotPth(pthDummy)
        
    def preset_B0slice(self):
        self.clear()
        self.xlabel = 'careful MAGNET'
        self.ylabel = 'careful MAGNET'
        self.title = 'plane ?? slice ?'       
        
        
    def preset_B0M(self):
        self.clear()
        self.xlabel = 'X MAGNET'
        self.ylabel = 'Y MAGNET'
        self.zlabel = 'Z MAGNET'
        self.title = 'dummy B0 map'
        b0Dummy = b0.b0(b0_filename='./dummies/b0_maps/a00_ball_R80mm_bvalues_coarse_5s_FAST.txt',path_filename='./dummies/pathfiles/2021-10-14_PathfileTest_Spherical.path')
        b0Dummy.magnet.set_origin(0,0,0)
        b0Dummy.path.center(origin=b0Dummy.magnet.origin)
        self.plotPathWithMagnet(b0map_object=b0Dummy)
        
        
    def plot_magnet(self,magnet:osi2magnet):
        print('plotting a magnet with radius ',magnet.bore_radius,' at',magnet.origin)

        magnet_origin  = magnet.origin
        xvec = magnet.xvector
        yvec = magnet.yvector
        zvec = magnet.zvector
        

        self.axes.quiver(magnet_origin[0],magnet_origin[1],magnet_origin[2], xvec[0]-magnet_origin[0], xvec[1]-magnet_origin[1], xvec[2]-magnet_origin[2], color='r')
        self.axes.quiver(magnet_origin[0],magnet_origin[1],magnet_origin[2], yvec[0]-magnet_origin[0], yvec[1]-magnet_origin[1], yvec[2]-magnet_origin[2], color='g')
        self.axes.quiver(magnet_origin[0],magnet_origin[1],magnet_origin[2], zvec[0]-magnet_origin[0], zvec[1]-magnet_origin[1], zvec[2]-magnet_origin[2], color='b')
        
        #self.axes.plot(magnet.bore_front_X,magnet.bore_front_Y,magnet.bore_front_Z,zdir='z',label='magnet front')
        #self.axes.plot(magnet.bore_back_X,magnet.bore_back_Y,magnet.bore_back_Z,zdir='z',label='magnet back')
        

        #todo: make osi2magnet class and plot here the damn cylinder. with the axes!

    def plot_head_on_path(self,cosimeasure: cosimeasure.cosimeasure,magnet:osi2magnet.osi2magnet):
        xheadpos = cosimeasure.head_position[0]
        yheadpos = cosimeasure.head_position[1]
        zheadpos = cosimeasure.head_position[2]
        
        num_path_current_index = cosimeasure.b0.path.current_index

        
        pathInput = cosimeasure.path
        r = pathInput.r
        self.title = 'head at [%.2f %.2f %.2f] '%(xheadpos,yheadpos,zheadpos)

        self.axes.cla()
        self.axes.set_xlabel(self.xlabel)
        self.axes.set_ylabel(self.ylabel)
        self.axes.set_zlabel(self.zlabel)        
        self.axes.set_title(self.title)
        self.plot_magnet(magnet)
        self.axes.plot(r[:,0],r[:,1],r[:,2],alpha=0.1,marker = 'o',linestyle=':',color='black') # alpha = 0.3 is the 4th argum
        self.axes.plot(r[0:num_path_current_index,0],r[0:num_path_current_index,1],r[0:num_path_current_index,2],alpha=1,marker = 'x',linestyle=':',color='green') # alpha = 0.3 is the 4th argum

        self.axes.plot(xheadpos,yheadpos,zheadpos,'rx',linewidth=10)
        #self.axes.autoscale(True)
        self.update_plotter()




    def plotPathWithMagnet(self,b0map_object:b0.b0,coordinate_system=None):
        self.axes.cla()
        self.xlabel = 'X COSI /tmp'
        self.ylabel = 'Y COSI /tmp'
        self.zlabel = 'Z COSI /tmp'
        if coordinate_system == 'magnet':
            self.xlabel = 'X magnet'
            self.ylabel = 'Y magnet'
            self.zlabel = 'Z magnet'

        self.axes.set_title(str(b0map_object.datetime))
        pth = b0map_object.path
        self.plotPth(pathInput=pth)
        self.plot_magnet(b0map_object.magnet)
        self.update_plotter()
        
    def plotB0slice_2D(self,b0map_object:b0.b0,slice_number_xy=-1,slice_number_zx=-1,slice_number_yz=-1,plot_raw=False,plot_sph=False,plot_shim=False,plot_error=False):
        self.axes.cla()
        slice_color_map='viridis'
        
        #minval_of_b0 = -47.809 + np.nanmean(b0map_object.b0Data[:,:,:,0]) + 47.553 #np.nanmin(b0map_object.b0Data[:,:,:,0])
        #maxval_of_b0 = -47.394 + np.nanmean(b0map_object.b0Data[:,:,:,0]) + 47.553#np.nanmax(b0map_object.b0Data[:,:,:,0])
        minval_of_b0 = np.nanmin(b0map_object.b0Data[:,:,:,0])
        maxval_of_b0 = np.nanmax(b0map_object.b0Data[:,:,:,0])

        if plot_raw:
    
            
            if slice_number_xy >= 0:
                #x, y = np.meshgrid(b0map_object.xPts, b0map_object.yPts)
                z = b0map_object.zPts[slice_number_xy]#(np.ones((len(b0map_object.xPts), len(b0map_object.yPts)))*b0map_object.zPts[slice_number_xy])
                #vals = (b0map_object.b0Data[:,:,slice_number_xy,1])
                imgdata = (b0map_object.b0Data[:,:,slice_number_xy,0])#self.axes.imshow(x,y,vals,cmap=slice_color_map)
                 # Scale colormap to current slice
                minval_of_b0 = np.nanmin(imgdata)                
                maxval_of_b0 = np.nanmax(imgdata)                
                img = self.axes.imshow(imgdata,cmap=slice_color_map,vmin = minval_of_b0, vmax= maxval_of_b0,
                                    origin = 'lower', 
                                    extent=[min(b0map_object.yPts),max(b0map_object.yPts),min(b0map_object.xPts),max(b0map_object.xPts)])
                
                self.title = 'XY slice #%d z=%.2f mm'%(slice_number_xy,z)
                self.xlabel = 'Y [mm]' #!!!!
                self.ylabel = 'X [mm]' #!!!!
                self.update_plotter()

                
            if slice_number_zx >= 0:
                #x, y = np.meshgrid(b0map_object.xPts, b0map_object.yPts)
                y = b0map_object.yPts[slice_number_zx]#(np.ones((len(b0map_object.xPts), len(b0map_object.yPts)))*b0map_object.zPts[slice_number_xy])         
                #vals = (b0map_object.b0Data[:,:,slice_number_xy,1])
                imgdata = (b0map_object.b0Data[:,slice_number_zx,:,0])#self.axes.imshow(x,y,vals,cmap=slice_color_map)
                minval_of_b0 = np.nanmin(imgdata)                
                maxval_of_b0 = np.nanmax(imgdata)           
                #vals = (b0map_object.b0Data[:,:,slice_number_xy,1])                
                img = self.axes.imshow(imgdata,cmap=slice_color_map,vmin = minval_of_b0, vmax= maxval_of_b0,
                                    origin = 'lower',
                                    extent=[min(b0map_object.zPts),max(b0map_object.zPts),min(b0map_object.xPts),max(b0map_object.xPts)])
                
                self.title = 'ZX slice #%d y=%.2f mm'%(slice_number_zx,y)
                self.xlabel = 'Z [mm]' #!!!!
                self.ylabel = 'X [mm]' #!!!!
                self.update_plotter()

                
            if slice_number_yz >= 0:
                x = b0map_object.yPts[slice_number_yz]#(np.ones((len(b0map_object.xPts), len(b0map_object.yPts)))*b0map_object.zPts[slice_number_xy])
                # Scale colormap to current slice

                imgdata = (b0map_object.b0Data[slice_number_yz,:,:,0])
                minval_of_b0 = np.nanmin(imgdata)                
                maxval_of_b0 = np.nanmax(imgdata)           
                #vals = (b0map_object.b0Data[:,:,slice_number_xy,1])
                img = self.axes.imshow(imgdata,cmap=slice_color_map,vmin = minval_of_b0, vmax= maxval_of_b0,
                                    origin = 'lower',
                                    extent=[min(b0map_object.zPts),max(b0map_object.zPts),min(b0map_object.yPts),max(b0map_object.yPts)])
                
                self.title = 'YZ slice #%d x=%.2f mm'%(slice_number_yz,x)
                self.xlabel = 'Z [mm]'
                self.ylabel = 'Y [mm]'
                self.update_plotter()

                
            
                
        if plot_sph:
            minval_of_b0 = np.nanmin(b0map_object.interpolatedField[:,:,:])
            maxval_of_b0 = np.nanmax(b0map_object.interpolatedField[:,:,:])
            
            if slice_number_xy >= 0:
                #x, y = np.meshgrid(b0map_object.xPts, b0map_object.yPts)
                z = b0map_object.xDim_SPH_fine[slice_number_xy]#(np.ones((len(b0map_object.xPts), len(b0map_object.yPts)))*b0map_object.zPts[slice_number_xy])
            
                #vals = (b0map_object.b0Data[:,:,slice_number_xy,1])
                imgdata = (b0map_object.interpolatedField[:,:,slice_number_xy])#self.axes.imshow(x,y,vals,cmap=slice_color_map)
                img = self.axes.imshow(imgdata,cmap=slice_color_map,vmin = minval_of_b0, vmax= maxval_of_b0,
                                    origin = 'lower', 
                                    extent=[min(b0map_object.yDim_SPH_fine),max(b0map_object.yDim_SPH_fine),min(b0map_object.xDim_SPH_fine),max(b0map_object.xDim_SPH_fine)]) #!!! AND ALL ABOVE
                
                self.title = 'FIT. XY slice #%d z=%.2f mm'%(slice_number_xy,z)
                self.xlabel = 'Y [mm]' #!!!
                self.ylabel = 'X [mm]' #!!!
                self.update_plotter()
                
            if slice_number_zx >= 0:
                #x, y = np.meshgrid(b0map_object.xPts, b0map_object.yPts)
                y = b0map_object.zDim_SPH_fine[slice_number_zx]#(np.ones((len(b0map_object.xPts), len(b0map_object.yPts)))*b0map_object.zPts[slice_number_xy])
            
                #vals = (b0map_object.b0Data[:,:,slice_number_xy,1])
                imgdata = (b0map_object.interpolatedField[:,slice_number_zx,:])#self.axes.imshow(x,y,vals,cmap=slice_color_map)
                img = self.axes.imshow(imgdata,cmap=slice_color_map,vmin = minval_of_b0, vmax= maxval_of_b0,
                                    origin = 'lower',
                                    extent=[min(b0map_object.zDim_SPH_fine),max(b0map_object.zDim_SPH_fine),min(b0map_object.xDim_SPH_fine),max(b0map_object.xDim_SPH_fine)])
                
                self.title = 'FIT ZX slice #%d y=%.2f mm'%(slice_number_zx,y)
                self.xlabel = 'Z [mm]'
                self.ylabel = 'X [mm]'
                self.update_plotter()
                
            if slice_number_yz >= 0:
                x = b0map_object.yDim_SPH_fine[slice_number_yz]#(np.ones((len(b0map_object.xPts), len(b0map_object.yPts)))*b0map_object.zPts[slice_number_xy])
            
                #vals = (b0map_object.b0Data[:,:,slice_number_xy,1])
                imgdata = (b0map_object.interpolatedField[slice_number_yz,:,:])
                img = self.axes.imshow(imgdata,cmap=slice_color_map,vmin = minval_of_b0, vmax= maxval_of_b0,
                                    origin = 'lower',
                                    extent=[min(b0map_object.zDim_SPH_fine),max(b0map_object.zDim_SPH_fine),min(b0map_object.yDim_SPH_fine),max(b0map_object.yDim_SPH_fine)])
                
                self.title = 'FIT YZ slice #%d x=%.2f mm'%(slice_number_yz,x)
                self.xlabel = 'Z [mm]'
                self.ylabel = 'Y [mm]'
                self.update_plotter()
                
        if plot_shim or plot_error:
            
            fieldmap = b0map_object.shimField if plot_shim else b0map_object.errorField
            typestr = 'SHIM' if plot_shim else 'ERROR'
            if plot_shim:
                minval_of_b0 = np.nanmin(fieldmap)
                maxval_of_b0 = np.nanmax(fieldmap)
            if plot_error:
                minval_of_b0 = np.nanmin(b0map_object.b0Data[:,:,:,0]) - np.nanmean(b0map_object.b0Data[:,:,:,0]) + np.nanmean(b0map_object.errorField[:,:,:])
                maxval_of_b0 = np.nanmax(b0map_object.b0Data[:,:,:,0]) - np.nanmean(b0map_object.b0Data[:,:,:,0]) + np.nanmean(b0map_object.errorField[:,:,:])
                
                    
            if slice_number_xy >= 0:
                #x, y = np.meshgrid(b0map_object.xPts, b0map_object.yPts)
                z = b0map_object.xDim_SPH_fine[slice_number_xy]#(np.ones((len(b0map_object.xPts), len(b0map_object.yPts)))*b0map_object.zPts[slice_number_xy])
                # Scale colormap to current slice
                minval_of_b0 = np.nanmin(imgdata)                
                maxval_of_b0 = np.nanmax(imgdata)
                            
                #vals = (b0map_object.b0Data[:,:,slice_number_xy,1])
                imgdata = (fieldmap[:,:,slice_number_xy])#self.axes.imshow(x,y,vals,cmap=slice_color_map)
                img = self.axes.imshow(imgdata,cmap=slice_color_map,vmin = minval_of_b0, vmax= maxval_of_b0,
                                    origin = 'lower', 
                                    extent=[min(b0map_object.yDim_SPH_fine),max(b0map_object.yDim_SPH_fine),min(b0map_object.xDim_SPH_fine),max(b0map_object.xDim_SPH_fine)])
                
                
                self.title = '%s. XY slice #%d z=%.2f mm'%(typestr, slice_number_xy,z) 
                self.xlabel = 'Y [mm]'
                self.ylabel = 'X [mm]'
                self.update_plotter()

                
            if slice_number_zx >= 0:
                #x, y = np.meshgrid(b0map_object.xPts, b0map_object.yPts)
                y = b0map_object.zDim_SPH_fine[slice_number_zx]#(np.ones((len(b0map_object.xPts), len(b0map_object.yPts)))*b0map_object.zPts[slice_number_xy])
                # Scale colormap to current slice
                minval_of_b0 = np.nanmin(imgdata)                
                maxval_of_b0 = np.nanmax(imgdata)
                #vals = (b0map_object.b0Data[:,:,slice_number_xy,1])
                imgdata = (fieldmap[:,slice_number_zx,:])#self.axes.imshow(x,y,vals,cmap=slice_color_map)
                img = self.axes.imshow(imgdata,cmap=slice_color_map,vmin = minval_of_b0, vmax= maxval_of_b0,
                                    origin = 'lower',
                                    extent=[min(b0map_object.zDim_SPH_fine),max(b0map_object.zDim_SPH_fine),min(b0map_object.xDim_SPH_fine),max(b0map_object.xDim_SPH_fine)])
                
                self.title = '%s ZX slice #%d y=%.2f mm'%(typestr,slice_number_zx,y)
                self.xlabel = 'Z [mm]'
                self.ylabel = 'X [mm]'
                self.update_plotter()

                
            if slice_number_yz >= 0:
                x = b0map_object.yDim_SPH_fine[slice_number_yz]#(np.ones((len(b0map_object.xPts), len(b0map_object.yPts)))*b0map_object.zPts[slice_number_xy])
            
                #vals = (b0map_object.b0Data[:,:,slice_number_xy,1])
                # Scale colormap to current slice
                minval_of_b0 = np.nanmin(imgdata)                
                maxval_of_b0 = np.nanmax(imgdata)
                
                imgdata = (fieldmap[slice_number_yz,:,:])
                img = self.axes.imshow(imgdata,cmap=slice_color_map,vmin = minval_of_b0, vmax= maxval_of_b0,
                                    origin = 'lower',
                                    extent=[min(b0map_object.zDim_SPH_fine),max(b0map_object.zDim_SPH_fine),min(b0map_object.yim_SPH_fine),max(b0map_object.yDim_SPH_fine)])
                
                self.title = '%s YZ slice #%d x=%.2f mm'%(typestr,slice_number_yz,x)
                self.xlabel = 'Z [mm]'
                self.ylabel = 'Y [mm]'
                self.update_plotter()
               
        
        
         
        try:
            norm = matplotlib.colors.Normalize(vmin=minval_of_b0, vmax=maxval_of_b0)
            print('colorbar from ',minval_of_b0,' to',maxval_of_b0)
            
            if self.colorbar_object is None:
#                ax, _ = matplotlib.colorbar.make_axes(self.axes, shrink=0.25)
#                self.colorbar_object = matplotlib.colorbar.ColorbarBase(ax,cmap=slice_color_map,norm=norm)
                               
                self.colorbar_object = plt.colorbar(matplotlib.cm.ScalarMappable(norm = norm,cmap = slice_color_map) ,ax=self.axes, orientation='horizontal',shrink=0.25)
                self.colorbar_object.norm = norm
                for im in plt.gca().get_images():
                    im.set_clim(minval_of_b0,maxval_of_b0)
                self.colorbar_object.ax.set_xlim([minval_of_b0,maxval_of_b0])
                self.colorbar_object.set_ticks(np.linspace(minval_of_b0,maxval_of_b0,2))
                self.colorbar_object.update_ticks()
                self.fig.tight_layout()
            else:
                print('rescaling the colorbaron the 2d plot')
                self.colorbar_object.mappable = img
                self.colorbar_object.norm = norm
                for im in plt.gca().get_images():
                    im.set_clim(minval_of_b0,maxval_of_b0)
                img.set_clim(minval_of_b0,maxval_of_b0)
                self.colorbar_object.ax.set_xlim([minval_of_b0,maxval_of_b0])
                self.colorbar_object.set_ticks(np.linspace(minval_of_b0,maxval_of_b0,2))
                self.colorbar_object.update_ticks()
                self.fig.tight_layout()
                
            
            self.axes.autoscale(False)
            self.update_plotter()
        except Exception as e:
            print(e)
            print('2d plotter is going nuts')    
            
    
    def plotB0Map(self,b0map_object:b0.b0,slice_number_xy=-1,slice_number_zx=-1,slice_number_yz=-1, show_sphere_radius = None, show_magnet = None,show_rings = None, coordinate_system=None, plot_raw = False, plot_sph = False, plot_shim = False, plot_cheap=False, plot_error = False):
        # plot only one slice of data. Slice at the middle of the scan
        self.axes.cla()
       
        
        self.title = b0map_object.filename
        
        self.xlabel = 'X COSI /tmp'
        self.ylabel = 'Y COSI /tmp'
        self.zlabel = 'Z COSI /tmp'
        if coordinate_system == 'magnet':
            self.xlabel = 'X magnet'
            self.ylabel = 'Y magnet'
            self.zlabel = 'Z magnet'
            #self.axes.view_init(elev=None, azim=None, roll=90, vertical_axis='y', share=False)
        
        if show_sphere_radius is not None:
            u = np.linspace(0, 2 * np.pi, 64)
            v = np.linspace(0, np.pi, 64)
            x = show_sphere_radius * np.outer(np.cos(u), np.sin(v))
            y = show_sphere_radius * np.outer(np.sin(u), np.sin(v))
            z = show_sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
            self.axes.plot_wireframe(x, y, z,alpha=0.1,color='black')
            
        if show_magnet is not None:
            # plot a bloody cylinder 
            def data_for_cylinder_along_z(radius,height):
                x = np.linspace(-height/2, height/2, 50)
                theta = np.linspace(0, 2*np.pi, 50)
                theta_grid, x_grid=np.meshgrid(theta, x)
                z_grid = radius*np.cos(theta_grid)
                y_grid = radius*np.sin(theta_grid)
                return x_grid,y_grid,z_grid

            bore_depth = b0map_object.magnet.bore_depth
            bore_radius = b0map_object.magnet.bore_radius
            
            Xc,Yc,Zc = data_for_cylinder_along_z(radius=bore_radius,
                                                 height=bore_depth)

            self.axes.plot_wireframe(Xc, Yc, Zc, alpha=0.1,color='blue')
            self.axes.text(-bore_depth/2, 0, 0, "FRONT", color='black',zdir='z' )
            self.axes.text(bore_depth/2, 0, 0, "BACK", color='black',zdir='z')
            
        if show_rings is not None:
            # get the shim magnets from the b0 object
            try:
                shimming_magnets = b0map_object.shim_magnets
            except:
                print('generate shim positions first!')
            for my_little_magnet in shimming_magnets:
                # the viewer coordinates are mm
                self.axes.quiver(my_little_magnet.position[0]*1e3,my_little_magnet.position[1]*1e3,my_little_magnet.position[2]*1e3,my_little_magnet.dipole_vector[0]*1e9,my_little_magnet.dipole_vector[1]*1e9,my_little_magnet.dipole_vector[2]*1e9,color='black')
            
            self.update_plotter()
            
            
        minval_of_b0 = np.nanmin(b0map_object.b0Data[:,:,:,0]) #!!!
        maxval_of_b0 = np.nanmax(b0map_object.b0Data[:,:,:,0]) #!!!
        # plot the measured field, contour plots with a color map
        if plot_raw: # if didnt tick or unticked the plot sph checkbox
            print('PLOTTING RAW DATA')

            
            print('--- RAW plotter is called --- ')
            
            slice_color_map='viridis'
            
            nlevels = 32
            ctrf = None

            if slice_number_xy >= 0:
                # if slice number xy given, plot Z slice

                x,y = np.meshgrid(b0map_object.xPts, b0map_object.yPts,indexing='ij')
                z = b0map_object.zPts[slice_number_xy]#(np.ones((len(b0map_object.xPts), len(b0map_object.yPts)))*b0map_object.zPts[slice_number_xy])
            
                vals =(b0map_object.b0Data[:,:,slice_number_xy,0])#(b0map_object.b0Data[:,:,slice_number_xy,1])
            
                
                ctrf = self.axes.contourf(x,y,vals, offset = z, zdir = 'z', alpha=0.5,cmap=slice_color_map,edgecolor='black',vmin = minval_of_b0, vmax = maxval_of_b0,levels=nlevels)
    
                #self.axes.set_zlim(min(b0map_object.zPts), max(b0map_object.zPts))


            if slice_number_zx >= 0:
                # if slice number zx given, plot Y slice

                x,z = np.meshgrid(b0map_object.xPts, b0map_object.zPts,indexing='ij')      
                y = b0map_object.yPts[slice_number_zx]#(np.ones((len(b0map_object.xPts), len(b0map_object.zPts)))*b0map_object.yPts[slice_number_zx])
            
                vals = (b0map_object.b0Data[:,slice_number_zx,:,0])
                
            
                
                ctrf = self.axes.contourf(x,vals,z,zdir = 'y', offset = y, alpha=0.5,cmap=slice_color_map,edgecolor='black',vmin = minval_of_b0, vmax = maxval_of_b0,levels=nlevels)
                #self.axes.set_ylim(min(b0map_object.yPts), max(b0map_object.yPts))
                
            
            if slice_number_yz >= 0:
                # if slice number yz given, plot X slice
                
                y,z = np.meshgrid(b0map_object.yPts, b0map_object.zPts,indexing='ij')
            
                x = b0map_object.xPts[slice_number_yz]#(np.ones((len(b0map_object.yPts), len(b0map_object.zPts)))*b0map_object.xPts[slice_number_yz])
            
                vals = (b0map_object.b0Data[slice_number_yz,:,:,0])
            
                #self.axes.plot_surface(x+vals,y,z,alpha=0.5,cmap='viridis',edgecolor='black',vmin = minval_of_b0+x, vmax = maxval_of_b0+x)
                ctrf = self.axes.contourf(vals,y,z,zdir = 'x', offset = x, alpha=0.5,cmap=slice_color_map,edgecolor='black',vmin = minval_of_b0, vmax = maxval_of_b0,levels=nlevels)
        
                
                
                
            #self.figure.show()
            #self.figure.canvas.draw()
            
            
            self.axes.set_xlim(min(b0map_object.xPts), max(b0map_object.xPts))
            self.axes.set_ylim(min(b0map_object.yPts), max(b0map_object.yPts))
            self.axes.set_zlim(min(b0map_object.zPts), max(b0map_object.zPts))
            
            
        
        if plot_sph: # if ticked the plot sph checkbox,
        # plot the interpolated field (SPH), contour plots with a color map

            self.update_plotter()
            print('getting the sph decomposed field from the b0 object')
            minval_of_b0 = np.nanmin(b0map_object.decomposedField[:,:,:])
            maxval_of_b0 = np.nanmax(b0map_object.decomposedField[:,:,:])
            print('min b0 sph: %.3f mT'%minval_of_b0)
            print('max b0 sph: %.3f mT'%maxval_of_b0)
            print('--- SPH plotter is called --- ')
            
            slice_color_map='viridis'
            
            nlevels = 32
            ctrf = None
            
            if slice_number_xy >= 0:
                # if slice number xy given, plot Z slice
                x,y = np.meshgrid(b0map_object.xDim_SPH_fine, b0map_object.yDim_SPH_fine,indexing='ij')
                z = b0map_object.zDim_SPH_fine[slice_number_xy]
                vals =b0map_object.interpolatedField[:,:,slice_number_xy]
                ctrf = self.axes.contourf(x,y,vals, offset = z, zdir = 'z', alpha=0.5,cmap=slice_color_map,edgecolor='black',vmin = minval_of_b0, vmax = maxval_of_b0,levels=nlevels)
    
            if slice_number_yz >= 0:
                # if slice number yz given, plot X slice
                y,z = np.meshgrid(b0map_object.yDim_SPH_fine, b0map_object.zDim_SPH_fine,indexing='ij')
                x = b0map_object.xDim_SPH_fine[slice_number_yz]
                vals =b0map_object.interpolatedField[slice_number_yz,:,:]
                ctrf = self.axes.contourf(vals,y,z, offset = x, zdir = 'x', alpha=0.5,cmap=slice_color_map,edgecolor='black',vmin = minval_of_b0, vmax = maxval_of_b0,levels=nlevels)

            if slice_number_zx >= 0:
                # if slice number zx given, plot Y slice
                x,z = np.meshgrid(b0map_object.xDim_SPH_fine, b0map_object.zDim_SPH_fine,indexing='ij')
                y = b0map_object.yDim_SPH_fine[slice_number_zx]
                vals =b0map_object.interpolatedField[:,slice_number_zx,:]
                ctrf = self.axes.contourf(x,vals,z, offset = y, zdir = 'y', alpha=0.5,cmap=slice_color_map,edgecolor='black',vmin = minval_of_b0, vmax = maxval_of_b0,levels=nlevels)
    
    

            self.axes.set_xlim(min(b0map_object.xDim_SPH_fine), max(b0map_object.xDim_SPH_fine))
            self.axes.set_ylim(min(b0map_object.yDim_SPH_fine), max(b0map_object.yDim_SPH_fine))
            self.axes.set_zlim(min(b0map_object.zDim_SPH_fine), max(b0map_object.zDim_SPH_fine))
            

            
        if plot_shim or plot_error or plot_cheap:
            # plot the interpolated field (SPH), contour plots with a color map
            self.update_plotter()
            print('getting the field map from the b0 object')    
            
            if plot_shim:
                b0map_object.render_fair_shim_field()
                fieldmap = b0map_object.shimField
                print('--- SHIM plotter is called --- ')
            if plot_error:
                fieldmap = b0map_object.errorField
                print('--- ERROR plotter is called --- ')
            if plot_cheap:
                fieldmap = b0map_object.cheapField
                print('--- CHEAP plotter is called --- ')

            minval_of_b0 = np.nanmin(fieldmap[:,:,:])
            maxval_of_b0 = np.nanmax(fieldmap[:,:,:])
            print('min b0 shim: %.3f mT'%minval_of_b0)
            print('max b0 shim: %.3f mT'%maxval_of_b0)
            
            slice_color_map='viridis'
            
            nlevels = 32
            ctrf = None
            
            if slice_number_xy >= 0:
                # if slice number xy given, plot Z slice
                x,y = np.meshgrid(b0map_object.xDim_SPH_fine, b0map_object.yDim_SPH_fine,indexing='ij')
                z = b0map_object.zDim_SPH_fine[slice_number_xy]
                vals = fieldmap[:,:,slice_number_xy]
                ctrf = self.axes.contourf(x,y,vals, offset = z, zdir = 'z', alpha=0.5,cmap=slice_color_map,edgecolor='black',vmin = minval_of_b0, vmax = maxval_of_b0,levels=nlevels)
    
            if slice_number_yz >= 0:
                # if slice number yz given, plot X slice
                y,z = np.meshgrid(b0map_object.yDim_SPH_fine, b0map_object.zDim_SPH_fine,indexing='ij')
                x = b0map_object.xDim_SPH_fine[slice_number_yz]
                vals =fieldmap[slice_number_yz,:,:]
                ctrf = self.axes.contourf(vals,y,z, offset = x, zdir = 'x', alpha=0.5,cmap=slice_color_map,edgecolor='black',vmin = minval_of_b0, vmax = maxval_of_b0,levels=nlevels)

            if slice_number_zx >= 0:
                # if slice number zx given, plot Y slice
                x,z = np.meshgrid(b0map_object.xDim_SPH_fine, b0map_object.zDim_SPH_fine,indexing='ij')
                y = b0map_object.yDim_SPH_fine[slice_number_zx]
                vals =fieldmap[:,slice_number_zx,:]
                ctrf = self.axes.contourf(x,vals,z, offset = y, zdir = 'y', alpha=0.5,cmap=slice_color_map,edgecolor='black',vmin = minval_of_b0, vmax = maxval_of_b0,levels=nlevels)
    
    

            self.axes.set_xlim(min(b0map_object.xDim_SPH_fine), max(b0map_object.xDim_SPH_fine))
            self.axes.set_ylim(min(b0map_object.yDim_SPH_fine), max(b0map_object.yDim_SPH_fine))
            self.axes.set_zlim(min(b0map_object.zDim_SPH_fine), max(b0map_object.zDim_SPH_fine))
            
            
        self.axes.autoscale(False)
        self.update_plotter()
            
        try:
            norm = matplotlib.colors.Normalize(vmin=minval_of_b0, vmax=maxval_of_b0)

            
            if self.colorbar_object is None:
                self.colorbar_object = plt.colorbar(matplotlib.cm.ScalarMappable(norm = norm,cmap = slice_color_map) ,ax=self.axes, orientation='horizontal',label = '[mT]',shrink=0.7)
                for im in self.axes.get_images():
                    im.set_clim(minval_of_b0,maxval_of_b0)

                self.colorbar_object.ax.set_xlim(minval_of_b0,maxval_of_b0)
                self.fig.tight_layout()
            else:
                self.fig.tight_layout()
                self.colorbar_object.norm = norm
                for im in self.axes.get_images():
                    im.set_clim(minval_of_b0,maxval_of_b0)

                self.colorbar_object.ax.set_xlim(minval_of_b0,maxval_of_b0)
                roundDigits = 2
                if plot_raw or plot_sph or plot_error:
                    tickStep = 0.05
                    roundDigits = 1
                if plot_shim:
                    tickStep = 0.01
                    roundDigits = 3
                
                #self.colorbar_object.set_ticks(np.arange(np.round(minval_of_b0,roundDigits),maxval_of_b0,tickStep))
                    
                #self.colorbar_object.update_ticks()
                
            

        except Exception as e:
            print(e)    
        
        
        

    def plotPth(self,pathInput: pth.pth):
        r = pathInput.r
        
        self.axes.cla()
        self.axes.set_xlabel(self.xlabel)
        self.axes.set_ylabel(self.ylabel)
        self.axes.set_zlabel(self.zlabel)        
        self.axes.set_title(self.title)
        self.axes.plot(r[:,0],r[:,1],r[:,2],'k+:')
        self.axes.autoscale(True)
        self.update_plotter()



    # EMRE electrochemistry
    def plotCvData(self, voltages, currents):
        self.axes.cla()
        self.axes.set_xlabel(self.xlabel)
        self.axes.set_ylabel(self.ylabel)
        self.axes.set_title(self.title)
        self.axes.plot(voltages, currents, 'm+:', linewidth=1)
        self.axes.plot(voltages[-1], currents[-1], 'kx:', linewidth=5)
        self.axes.autoscale(True)
        self.update_plotter()

    def plotChg(self, chgInput: chg.chg):
        xValues = chgInput.time
        yValues = chgInput.voltage
        self.axes.cla()
        self.axes.set_xlabel(self.xlabel)
        self.axes.set_ylabel(self.ylabel)
        self.axes.set_title(chgInput.filename)
        self.axes.plot(xValues, yValues,'o:')
        self.axes.plot(xValues[-1], yValues[-1], 'kx:', linewidth=5)
        self.axes.autoscale(True)
        self.axes.grid()
        self.update_plotter()
        
    def plotCv(self,cvToPlot:cv):
        voltages = cvToPlot.voltage
        currents = cvToPlot.current
        self.title = cvToPlot.filename

        self.axes.cla()
        self.axes.set_xlabel(self.xlabel)
        self.axes.set_ylabel(self.ylabel)
        self.axes.set_title(self.title)
        self.axes.plot(voltages, currents, 'k-', linewidth=1)
        self.axes.autoscale(True)
        self.update_plotter()

    # EMRE Microwave
    def plotTpData(self,tpToPlot:tp):
        times = tpToPlot.time
        frequencies = tpToPlot.frequency
        tunepic = tpToPlot.tunepicture
        self.title = ''
        self.axes.cla()
        self.axes.set_xlabel(self.xlabel)
        self.axes.set_ylabel(self.ylabel)
        self.axes.set_title(self.title)
        self.axes.plot(frequencies, tunepic, 'k-', linewidth=1)
        self.axes.autoscale(True)
        self.update_plotter()

    def plotTpFitData(self,tpToPlot:tp):
        frequencies = tpToPlot.frequencyFit
        tunepicFit = tpToPlot.tunepicFit
        
        self.title = ''
        #self.axes.cla()
        self.axes.set_xlabel(self.xlabel)
        self.axes.set_ylabel(self.ylabel)
        self.axes.set_title(self.title)
        # self.axes.plot(dipFreqtpToPlot.dipFreq, tpToPlot.dip, 'r-', linewidth=2) # dip without bg
        self.axes.plot(frequencies, tunepicFit, 'g--', linewidth=2)  # fit

        self.axes.autoscale(True)
        self.update_plotter()



    def update_plotter(self): # very useful and important method for live plotting.

            
        self.axes.set_xlabel(self.xlabel)
        self.axes.set_ylabel(self.ylabel)
        if self.plotType == 'PTH' or self.plotType == 'B0M':
            self.axes.set_zlabel(self.zlabel) 
        self.axes.set_title(self.title)
        self.figure.canvas.draw()
        #self.figure.canvas.flush_events()



    # a widget class to implement the toolbar
class Plotter(QWidget):
    plotType = 'general' # can be EPR, TP, CV and CHG plotType
    def __init__(self, parent, plotType, *args, **kwargs): # you have to pass the main window here, else crashes on click save
        QWidget.__init__(self, *args, **kwargs)
        self.setLayout(QVBoxLayout())
        self.PlotterCanvas = PlotterCanvas(plotType = plotType) # Plotter is a class defined above. plotType defines which plotter to be created ('CV,CHG,EPR,TP')

        # navigation toolbar
        self.toolbar = NavigationToolbar(self.PlotterCanvas, parent = self)

        '''custom buttons on navigation toolbar'''
        # self.toolbar.clear()
        #
        # a = self.toolbar.addAction(self.toolbar._icon("home.png"), "Home", self.toolbar.home)
        # # a.setToolTip('returns axes to original position')
        # a = self.toolbar.addAction(self.toolbar._icon("move.png"), "Pan", self.toolbar.pan)
        # a.setToolTip("Pan axes with left mouse, zoom with right")
        # a = self.toolbar.addAction(self.toolbar._icon("zoom_to_rect.png"), "Zoom", self.toolbar.zoom)
        # a.setToolTip("Zoom to Rectangle")
        # a = self.toolbar.addAction(self.toolbar._icon("filesave.png"), "Save", self.toolbar.save_figure)
        # a.setToolTip("Save the figure")

        def save_figure():
            print('SAVE THE DATA! - write that method in your free time')

        a = self.toolbar.addAction(self.toolbar._icon("filesave.png"), "Save data", save_figure)
        a.setToolTip("Save data in file")


        'insert plotter'
        self.layout().addWidget(self.PlotterCanvas)
        'insert toolbar'
        self.layout().addWidget(self.toolbar)