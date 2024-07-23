# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 09:47:41 2019

@author: Tom O'Reilly
"""

import numpy as np

class interactivePlot(object):
    def __init__(self,fig, ax, X, plotAxis = 2, axisLabels = None, fov = None, cmap = 'gray'):
        
#        set various event handlers
        fig.canvas.mpl_connect('scroll_event', self.onScroll)
        fig.canvas.mpl_connect('button_press_event', self.onClick)
        fig.canvas.mpl_connect('button_release_event', self.onRelease)
        fig.canvas.mpl_connect('motion_notify_event', self.onMotion)
        fig.canvas.mpl_connect('key_press_event', self.keyPress)
        
        self.fig = fig
        self.plotAxis = plotAxis
        self.ax = ax
        self.ax.set_adjustable('box')
        self.mouseClicked = None
        self.cmapRange = np.nanmax(X) - np.nanmin(X)
        self.cmapCenter = np.nanmin(X) + self.cmapRange/2
        self.tempCmapRange = self.cmapRange
        self.tempCmapCenter = self.cmapCenter
        self.X = X
        self.axisLabels = axisLabels
        self.colorMap = cmap
        
        if fov is None:
            self.fov = (1,1,1)
            self.resolution = (1,1,1)
        else:
            self.fov = fov
            self.resolution = np.divide(fov, np.shape(X))

        self.slices = np.shape(X)[plotAxis]
        self.ind = self.slices//2
        
        ax.set_title('Slice %s/%s' % (self.ind,self.slices))
        
        if self.plotAxis == 0:
            imageData = self.X[self.ind,:,:]
        elif self.plotAxis == 1:
            imageData = self.X[:,self.ind,:]
        elif self.plotAxis == 2:
            imageData = self.X[:,:,self.ind]
        else:
            print("invalid axis")
            return -1
        self.im = ax.imshow(imageData, cmap = self.colorMap,vmin = self.cmapCenter-self.cmapRange/2, vmax= self.cmapCenter+self.cmapRange/2)
        self.updateSlice()

    def keyPress(self, event):
        if event.key == " ": #change orientation on space bar press
            self.plotAxis += 1
            if self.plotAxis > 2:
                self.plotAxis = 0
            self.updateSlice()
                
    def onScroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.updateSlice()
    
    def onClick(self, event):
#        print('Button pressed: %s at X: %s, Y: %s'%(event.button, event.xdata, event.ydata))
        if event.button == 3:   #Reset image on right click
            self.cmapRange = np.nanmax(self.X) - np.nanmin(self.X)
            self.cmapCenter = np.nanmin(self.X) + self.cmapRange/2
            self.im.set_clim(self.cmapCenter-self.cmapRange/2, self.cmapCenter+self.cmapRange/2)
            self.im.axes.figure.canvas.draw()
        elif event.button == 2: #change orientation on scroll wheel click
            self.plotAxis += 1
            if self.plotAxis > 2:
                self.plotAxis = 0
            self.updateSlice()
        else:
            self.mouseClicked = event.xdata, event.ydata
        
    def onRelease(self, event):
        self.mouseClicked = None
        self.cmapRange = self.tempCmapRange
        self.cmapCenter = self.tempCmapCenter
        
    def onMotion(self, event):
        if self.mouseClicked == None: return        #if mouse isn't clicked ignore movement

        dx = event.xdata - self.mouseClicked[0]
        dy = event.ydata - self.mouseClicked[1]
        
        normDx = dx/self.mouseClicked[0]
        normDy = dy/self.mouseClicked[1]
        
        self.tempCmapRange = self.cmapRange*(1+normDy)
        self.tempCmapCenter = self.cmapCenter*(1+normDx)
        
        self.im.set_clim(self.tempCmapCenter-self.tempCmapRange/2, self.tempCmapCenter+self.tempCmapRange/2)
        self.im.axes.figure.canvas.draw()
        
    def updateSlice(self):
        if self.plotAxis == 0:
            if self.ind >= np.size(self.X,0):
                self.ind = np.size(self.X, 0)-1
            imageData = self.X[self.ind,:,:]
            if self.axisLabels != None:
                self.ax.set_xlabel(self.axisLabels[2])
                self.ax.set_ylabel(self.axisLabels[1])
            self.im.set_extent((-0.5,np.size(imageData,1) - 0.5,np.size(imageData,0) - 0.5,-0.5))
            self.ax.set_aspect(self.resolution[1]/self.resolution[2])
            self.slices = np.size(self.X,0)
        elif self.plotAxis == 1:
            if self.ind >= np.size(self.X,1):
                self.ind = np.size(self.X, 1)-1
            imageData = self.X[:,self.ind,:]
            if self.axisLabels != None:
                self.ax.set_xlabel(self.axisLabels[2])
                self.ax.set_ylabel(self.axisLabels[0])
            self.im.set_extent((-0.5,np.size(imageData,1) - 0.5,np.size(imageData,0) - 0.5,-0.5))
            self.ax.set_aspect(self.resolution[0]/self.resolution[2])
            self.slices = np.size(self.X,1)
        else:
            if self.ind >= np.size(self.X,2):
                self.ind = np.size(self.X,2)-1
            imageData = self.X[:,:,self.ind]
            if self.axisLabels != None:
                self.ax.set_xlabel(self.axisLabels[1])
                self.ax.set_ylabel(self.axisLabels[0])
            self.im.set_extent((-0.5,np.size(imageData,1) - 0.5,np.size(imageData,0) - 0.5,-0.5))
            self.ax.set_aspect(self.resolution[0]/self.resolution[1])
            self.slices = np.size(self.X,2)
        # print("Plot axis: %.0f, Array dims: %.0f , %.0f" %(self.plotAxis, np.size(imageData,0), np.size(imageData,1)))
        self.im.set_data(imageData)
        self.ax.set_title('Slice %s/%s' % (self.ind,self.slices))
        self.im.axes.figure.canvas.draw()


