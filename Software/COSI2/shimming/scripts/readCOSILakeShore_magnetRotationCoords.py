# -*- coding: utf-8 -*-
"""
Spyder Editor

This version of the code to read in the data for the cosi measure has been written to match the coordinate system used in the magnet rotations shim code,
it varies slightly from the previous version of the code to correct for a flip in the y axis between the two sets of code.
"""

import numpy as np
import matplotlib.pyplot as plt
import interactive3Dplot as plt3D
 

pathFile = r'20221114 - OSII One CNC - 252mm DSV - 9mm.path'
dataFile = r'20221114 - OSII One CNC - 252mm DSV - 9mm.txt' 

with open(pathFile) as file:
    rawPathData = file.readlines()

fieldData = np.genfromtxt(dataFile, delimiter = '\t')[:,:4]

pathData = np.zeros((len(rawPathData),3))

for idx, point in enumerate(rawPathData):
    splitPoint = point.rstrip("\n\r").split('Z')
    pathData[idx, 2] = splitPoint[1]
    splitPoint = splitPoint[0].split('Y')
    pathData[idx, 1] = splitPoint[1]
    splitPoint = splitPoint[0].split('X')
    pathData[idx, 0] = splitPoint[1]

xPts = np.unique(pathData[:,0])
yPts = np.unique(pathData[:,1])
yPts = np.flip(np.unique(pathData[:,1]))
zPts = np.unique(pathData[:,2])
# zPts = np.flip(np.unique(pathData[:,2]))


xCoords, yCoords, zCoords = np.meshgrid(zPts, yPts, xPts, indexing='ij')
# xCoords, yCoords, zCoords = np.meshgrid(zPts, yPts, xPts, indexing = 'xy')


coords = np.stack((zCoords, yCoords, xCoords), axis = -1)

b0Data = np.zeros((len(zPts), len(yPts), len(xPts),4))
tempCoords = np.zeros((len(zPts), len(yPts), len(xPts),3))

for idx in range(np.size(pathData, 0)):
    xArg = np.where(xPts == pathData[idx, 0])[0][0]
    yArg = np.where(yPts == pathData[idx, 1])[0][0]
    zArg = np.where(zPts == pathData[idx, 2])[0][0]
    if np.size(np.where(zPts == pathData[idx, 1]))>1:
        print("oops")
    b0Data[zArg,yArg,xArg,:] = fieldData[idx,:]
    tempCoords[zArg,yArg,xArg,:] = pathData[idx,:]


b0Data[b0Data == 0] = 'NaN'

meanField = np.nanmean(b0Data[...,0])
homogeneity = 1e6 * (np.nanmax(b0Data[...,0]) - np.nanmin(b0Data[...,0]))/meanField

print("Mean field: %.2f mT"%(meanField))
print("Homogeneity: %i ppm"%(homogeneity))

fig, ax = plt.subplots(1,1)
fig3D = plt3D.interactivePlot(fig, ax, b0Data[...,0], plotAxis = 2, axisLabels = ["x","y","z"], cmap = 'viridis')

# plt.figure()
# plt.plot(b0Data[:,14,14,0 ], label = "x")
# plt.plot(b0Data[14,:,14,0 ], label = "y")
# plt.plot(b0Data[14,14,:,0 ], label = "z")
# plt.legend()

tempData = b0Data[...,0]
fig, ax = plt.subplots(1,3)
ax[0].imshow(tempData[:,:,int(np.size(tempData,2)/2)], vmin = np.nanmin(tempData), vmax = np.nanmax(tempData))
ax[0].set_title("Field map")
ax[1].imshow(tempData[:,int(np.size(tempData,1)/2),:], vmin = np.nanmin(tempData), vmax = np.nanmax(tempData))
ax[1].set_title("Field map")
ax[2].imshow(tempData[int(np.size(tempData,0)/2),:,:], vmin = np.nanmin(tempData), vmax = np.nanmax(tempData))
ax[2].set_title("Field map")

np.save("20221114 - OSII One CNC - 252mm DSV - 9mm.npy", b0Data)





