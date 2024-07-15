# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:22:14 2019

@author: to_reilly
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm
from scipy.optimize import least_squares
import scipy.ndimage as cp

def cartToSpher(coords):
    r = np.sqrt(np.sum(np.square(coords),axis = -1))    
    #remove r = 0 to avoid divide by zero
    r[r==0] = np.nan
    
    phi = np.arctan2(coords[...,1], coords[...,0]) + np.pi
    theta = np.arccos(coords[...,2]/r)
    return np.stack([r,theta, phi], axis = -1)

def spherToCart(coords):
    x = coords[...,0]*np.sin(coords[...,1])*np.cos(coords[...,2])
    y = coords[...,0]*np.sin(coords[...,1])*np.sin(coords[...,2])
    z = coords[...,0]*np.cos(coords[...,1])
    return np.stack([x,y,z], axis = -1)

def getRealSphericalHarmonics(coords, maxOrder):
    r0          = np.nanmean(coords[...,0])       #Get the mean radius for normalisation
    spherHarm   = np.zeros((np.shape(coords[...,0]) + ((maxOrder + 1)**2,)))
    r0          = np.nanmean(150)       #Get the mean radius for normalisation
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

def fitSphericalHarmonics(fitVector, args):
    return np.square(maskedFieldShell - np.matmul(spherHarm, fitVector))

fieldMap = np.load(r'20221114 - OSII One CNC - 252mm DSV - 9mm - sphere.npy')[...,0]



#The highest order spherical harmonic to fit
maxOrder = 15

#Diameter spherical volume over which to do the spherical harmonic decomposition
DSV = 252
resolution = 9

#Create a cartesian coordinate system of where the data points were acquired, maps were acquired with a 5mm resolution 
fieldMapDims = np.shape(fieldMap)
xDim = np.linspace(0, resolution*(fieldMapDims[0]-1), fieldMapDims[0]) - resolution*(fieldMapDims[0] -1)/2
yDim = np.linspace(0, resolution*(fieldMapDims[1]-1), fieldMapDims[1]) - resolution*(fieldMapDims[1] -1)/2
zDim = np.linspace(0, resolution*(fieldMapDims[2]-1), fieldMapDims[2]) - resolution*(fieldMapDims[2] -1)/2

coord = np.meshgrid(xDim, yDim, zDim)

#Create a spherical mask for the data
sphereMask = np.zeros(np.shape(coord[0]), dtype = bool)
sphereMask[np.square(coord[0]) + np.square(coord[1]) + np.square(coord[2]) <= (DSV/2)**2] = 1 
sphereMask = sphereMask*(~np.isnan(fieldMap))


#Create a spherical shell mask to consider only data points on the surface of the sphere
erodedMask = cp.binary_erosion(sphereMask)                    # remove the outer surface of the initial spherical mask
shellMask = np.array(sphereMask^erodedMask, dtype = float)   # create a new mask by looking at the difference between the inital and eroded mask
shellMask[shellMask == 0] = np.nan                          # set points outside mask to 'NaN', works better than setting it to zero for calculating mean fields etc.

# sphereMask[20,20,20] = 0
sphereMask = np.asarray(sphereMask, dtype=float)
sphereMask[sphereMask == 0] = np.nan
# shellMask = np.copy(sphereMask)


#apply mask to data
maskedField = np.multiply(sphereMask, fieldMap)
print("Mean field strength in %i cm sphere: %.2f mT"%(DSV/10, np.nanmean(maskedField)))
print("Inhomogeneity in %i cm sphere: %.0f ppm" %(DSV/10, 1e6*(np.nanmax(maskedField) - np.nanmin(maskedField))/np.nanmean(maskedField)))


#convert cartesian coordinates to spherical coordinates
spherCoord = cartToSpher(np.stack((coord[1],coord[0], coord[2]), axis = -1))

#apply mask to field and coordinate arrays and vectorise them
maskedFieldShell = fieldMap[shellMask == 1]
maskedCoordShell = spherCoord[shellMask == 1,:]

#Get the spherical harmonics for each of the field points
spherHarm = getRealSphericalHarmonics(maskedCoordShell, maxOrder)


#Inital guess for the optimisation
initialGuess = np.zeros((np.size(spherHarm,-1)))

#run the optimisation
fitData = least_squares(fitSphericalHarmonics, initialGuess, args = (maskedFieldShell,))

#grab the coefficients from the data array
spherHarmCoeff = fitData.x

from scipy.linalg import lstsq

lsqFit = lstsq(spherHarm, maskedFieldShell)
spherHarmCoeff = lsqFit[0]

# spherHarmCoeff[1:4] = 0

#calculate the field from the spherical harmonic decomposition
decomposedField = np.matmul(spherHarm, spherHarmCoeff)
print("Inhomogeneity of fit: %.0f ppm" %(1e6*(np.max(decomposedField) - np.min(decomposedField))/np.mean(decomposedField)))

#See what the difference is between the two decomposed field
shimmedField = maskedFieldShell - decomposedField
print("Error: %.0f ppm" %(1e6*(np.max(shimmedField) - np.min(shimmedField))/np.mean(maskedFieldShell)))
   
#generate spherical coordinates over entire sphere, not just shell, for plotting
spherCoordSphere = np.copy(spherCoord)
spherCoordSphere[spherCoord[...,0] == 0,:] = np.nan

#generate spherical coordinates over entire sphere, not just shell, for plotting
spherHarm3D = getRealSphericalHarmonics(spherCoordSphere, maxOrder)

tempSpherHarmCoeff = np.copy(spherHarmCoeff)
# tempSpherHarmCoeff[4:] = 0
#calculate the spherical harmonic decomposed field
decomposedField = np.matmul(spherHarm3D, tempSpherHarmCoeff)*sphereMask

# calculate difference between decomposed field and measured field
errorField = maskedField - decomposedField

tempSpherHarmCoeff = np.copy(spherHarmCoeff)
tempSpherHarmCoeff[0] = 0

# plot all 3 fields in 3 different planes passing through the centere of the magnet
fig, ax = plt.subplots(3,3)
ax[0,0].imshow(maskedField[:,:,int(np.size(maskedField,2)/2)], vmin = np.nanmin(maskedField), vmax = np.nanmax(maskedField))
ax[0,0].set_title("Field map")
ax[1,0].imshow(decomposedField[:,:,int(np.size(decomposedField,2)/2)], vmin = np.nanmin(decomposedField), vmax = np.nanmax(decomposedField))
ax[1,0].set_title("Spherical harmonic decomposition")
ax[2,0].imshow(errorField[:,:,int(np.size(errorField,2)/2)], vmin = np.nanmin(errorField), vmax = np.nanmax(errorField))
ax[2,0].set_title("Difference")
ax[0,1].imshow(maskedField[:,int(np.size(maskedField,1)/2),:], vmin = np.nanmin(maskedField), vmax = np.nanmax(maskedField))
ax[0,1].set_title("Field map")
ax[1,1].imshow(decomposedField[:,int(np.size(decomposedField,1)/2),:], vmin = np.nanmin(decomposedField), vmax = np.nanmax(decomposedField))
ax[1,1].set_title("Spherical harmonic decomposition")
ax[2,1].imshow(errorField[:,int(np.size(errorField,1)/2),:], vmin = np.nanmin(errorField), vmax = np.nanmax(errorField))
ax[2,1].set_title("Difference")
ax[0,2].imshow(maskedField[int(np.size(maskedField,0)/2),:,:], vmin = np.nanmin(maskedField), vmax = np.nanmax(maskedField))
ax[0,2].set_title("Field map")
ax[1,2].imshow(decomposedField[int(np.size(decomposedField,0)/2),:,:], vmin = np.nanmin(decomposedField), vmax = np.nanmax(decomposedField))
ax[1,2].set_title("Spherical harmonic decomposition")
ax[2,2].imshow(errorField[int(np.size(errorField,0)/2),:,:], vmin = np.nanmin(errorField), vmax = np.nanmax(errorField))
ax[2,2].set_title("Difference")

np.save("20221114 - OSII One CNC - 252mm DSV - 9mm - sphere - spherHarm.npy", spherHarmCoeff)
