# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 11:42:39 2022

@author: to_reilly
"""

import numpy as np
import matplotlib.pyplot as plt
import b0V5 as b0
from scipy.optimize import least_squares
import scipy.ndimage as cp

def dataFitting(shimVector):
    # shimVector_stacked = np.hstack((np.cos(shimVector), np.sin(shimVector)))
    shimField = np.matmul(maskedFields,np.hstack((np.cos(shimVector), np.sin(shimVector)))) + initialFieldMasked
    # return (shimField - np.mean(shimField))
    return np.square(((shimField)/np.mean(shimField)) -1)*1e9
    # return np.std(shimField)


def cartToSpher(coords):
    r = np.sqrt(np.sum(np.square(coords),axis = -1))    
    #remove r = 0 to avoid divide by zero
    # r[r==0] = np.nan
    
    phi = np.arctan2(coords[...,1], coords[...,0]) + np.pi
    # theta = np.arccos(coords[...,2]/r)
    return np.stack([r, phi], axis = -1)

resolution          = 200           #resolution of map in points per meter
DSV                 = .25           #size of simulation
DSV_mask            = .25           #size over which to do optimisation
mu                  = 1e-7

magSizeOuter        = 6*1e-3        #size of shim magnets
bRem                = 1.26          #remanence field of shim magnets

#shim tray configuration
shimRadius          = 276*1e-3      # radius on which the shim magnets are placed
ringPositions       = np.linspace(-0.2295, .2295, 4) #Z positions to place shin rubgs
magsPerSegment      = 7             # number of magnets peer shim tray segment
anglePerSegment     = 19.25 #the angular distance in degrees between the furthest magnets in a shim tray (span of magnets in shim tray)
numSegments         = 12 #corresponds to the number of shim trays

initialField = np.load(r'20221114 - OSII One CNC - 250mm DSV - 5mm - sphere - interpolated.npy')

maxIter = 1000


''' Mask generation'''
xDim = np.linspace(-DSV/2, DSV/2, int(DSV*resolution) + 1)
yDim = np.linspace(-DSV/2, DSV/2, int(DSV*resolution) + 1)
zDim = np.linspace(-DSV/2, DSV/2, int(DSV*resolution) + 1)

xDim3D, yDim3D, zDim3D = np.meshgrid(xDim, yDim, zDim)
spherCoord = cartToSpher(np.stack((yDim3D,xDim3D, zDim3D), axis = -1))

#Apply mask to data
mask = (np.round(spherCoord[...,0],4) <= (DSV_mask/2)).astype(float)
# mask = (np.square(xDim3D)/((optVol[0]/2)**2) + np.square(yDim3D)/((optVol[1]/2)**2) + np.square(zDim3D)/((optVol[2]/2)**2)  <= 1).astype(float)
halfMask = mask#*((zDim3D<=0).astype(float))
erodedMask = cp.binary_erosion(halfMask.astype(bool))                    # remove the outer surface of the initial spherical mask
halfMask = np.array(halfMask.astype(bool)^erodedMask, dtype = float)   # create a new mask by looking at the difference between the inital and eroded mask
halfMask[halfMask == 0] = np.nan    
mask[mask == 0] = np.nan

ringCounter = 0
print("Calculating magnetic fields")
''' Field calculation'''


segmentAngles       = np.linspace(0,360, numSegments, endpoint = False)

magAngles           = np.linspace(-anglePerSegment/2, anglePerSegment/2, magsPerSegment) 

positions = []
for ringPosition in ringPositions:
    for segmentAngle in segmentAngles:
        for magAngle in magAngles:
            positions.append((shimRadius*np.cos((segmentAngle+magAngle)*np.pi/180), shimRadius*np.sin((segmentAngle+magAngle)*np.pi/180), ringPosition))

numMags = len(positions)
magnetFields = np.zeros((np.shape(initialField)+(2,numMags)), dtype = np.float32)

#define dipole moment
dip_mom = b0.magnetization(bRem, magSizeOuter)
dip_vec = mu*np.array([dip_mom, 0])

counter = 0
for idx1, position in enumerate(positions):
    magnetFields[...,idx1] = b0.singleMagnet(position, dipoleMoment = dip_vec, simDimensions = (DSV,DSV,DSV), resolution = resolution)[...,:2]
    
print("Field calculation complete")
print("Number of angles to optimise: %d"%(len(positions)))
print("Number of field points: %d"%(np.nansum(halfMask)))

magnetFields *= 1e3
maskedFields = magnetFields[halfMask == 1, :,:].astype(float)
maskedFields = np.hstack((maskedFields[:,0,:],maskedFields[:,1,:]))

initialFieldMasked = initialField[halfMask == 1]

initialGuess = np.zeros(int(np.size(maskedFields,-1)/2))
upperBound = initialGuess + 0.5
lowerBound = initialGuess - 0.5

lsqData = least_squares(dataFitting, initialGuess, max_nfev=maxIter, verbose=0)

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

fig, ax = plt.subplots(3,3)
ax[0,0].imshow(initialField[:,:,int(np.size(initialField,2)/2)], vmin = np.nanmin(initialField), vmax = np.nanmax(initialField))
ax[0,0].set_title("Initial")
ax[1,0].imshow(optimizedField[:,:,int(np.size(optimizedField,2)/2)], vmin = np.nanmin(optimizedField), vmax = np.nanmax(optimizedField))
ax[1,0].set_title("Shim field")
ax[2,0].imshow(shimmedField[:,:,int(np.size(shimmedField,2)/2)], vmin = np.nanmin(shimmedField), vmax = np.nanmax(shimmedField))
ax[2,0].set_title("Combined")
ax[0,1].imshow(initialField[:,int(np.size(initialField,1)/2),:], vmin = np.nanmin(initialField), vmax = np.nanmax(initialField))
ax[0,1].set_title("Initial")
ax[1,1].imshow(optimizedField[:,int(np.size(optimizedField,1)/2),:], vmin = np.nanmin(optimizedField), vmax = np.nanmax(optimizedField))
ax[1,1].set_title("Shim field")
ax[2,1].imshow(shimmedField[:,int(np.size(shimmedField,1)/2),:], vmin = np.nanmin(shimmedField), vmax = np.nanmax(shimmedField))
ax[2,1].set_title("Combined")
ax[0,2].imshow(initialField[int(np.size(initialField,0)/2),:,:], vmin = np.nanmin(initialField), vmax = np.nanmax(initialField))
ax[0,2].set_title("Initial")
ax[1,2].imshow(optimizedField[int(np.size(optimizedField,0)/2),:,:], vmin = np.nanmin(optimizedField), vmax = np.nanmax(optimizedField))
ax[1,2].set_title("Shim field")
ax[2,2].imshow(shimmedField[int(np.size(shimmedField,0)/2),:,:], vmin = np.nanmin(shimmedField), vmax = np.nanmax(shimmedField))
ax[2,2].set_title("Combined")
for idx in range(3):
    for idx2 in range(3):
        ax[idx,idx2].axis('off')
fig.set_tight_layout(True)

plt.figure()
plt.plot((180/np.pi)*lsqData.x)
plt.ylabel("Magnet rotation [Degrees]")
plt.xlabel("Magnet number")

positions = np.array(positions)
plt.figure()
#split in to rings
ringConfigs = []
magPos = 0
for idx in range(len(ringPositions)):
    magInRing = np.sum(positions[:,2] == ringPositions[idx])
    ringConfigs.append(np.hstack((positions[magPos:magPos+magInRing,:], (180/np.pi)*lsqData.x[magPos:magPos+magInRing][:,np.newaxis])))
    magPos += magInRing
    plt.plot(np.linspace(0,360, magInRing,endpoint = False),ringConfigs[idx][:,3], label = 'Ring %d'%idx)
plt.xlabel("Position [degrees]")
plt.ylabel("Angle deviation [degrees]")
plt.legend()

#write shim orientations to file
for idx1 in range(len(ringConfigs)):
    ring = ringConfigs[idx1]
    with open("Iteration 1 - Shim Ring %d.txt"%(idx1), 'w') as file:
        for idx2 in range(np.size(ring,0)):
            file.write("%.2f, %.2f, %.2f, %.2f\n"%(ring[idx2,0]*1e3,ring[idx2,1]*1e3,ring[idx2,2]*1e3, ring[idx2,3]))

# np.save("OSII One CNC - Iteration 1 - Simulation - 4 rings - 175mm.npy", optimizedField)
