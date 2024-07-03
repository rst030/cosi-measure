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

coeffs    = np.load(r'20221114 - OSII One CNC - 252mm DSV - 9mm - sphere - spherHarm.npy')

#The highest order spherical harmonic to fit
maxOrder = int(np.size(coeffs)**0.5 -1)
# maxOrder = 20

#Diameter spherical volume over which to do the interpolation
DSV = 250
resolution = 5

#Create a cartesian coordinate system of where the data points were acquired, maps were acquired with a 5mm resolution 
xDim = np.linspace(-DSV/2, DSV/2, int(DSV/resolution+1))
yDim = np.linspace(-DSV/2, DSV/2, int(DSV/resolution+1))
zDim = np.linspace(-DSV/2, DSV/2, int(DSV/resolution+1))

coord = np.meshgrid(xDim, yDim, zDim, indexing='ij')
# coord = np.meshgrid(xDim, yDim, zDim)


#Create a spherical mask for the data
sphereMask = np.zeros(np.shape(coord[0]), dtype = bool)
sphereMask[np.square(coord[0]) + np.square(coord[1]) + np.square(coord[2]) <= (DSV/2)**2] = 1 
sphereMask = np.asarray(sphereMask, dtype=np.double)
sphereMask[sphereMask == 0] = np.nan

#convert cartesian coordinates to spherical coordinates
spherCoord = cartToSpher(np.stack((coord[1],coord[0], coord[2]), axis = -1))

#generate spherical coordinates over entire sphere, not just shell, for plotting
spherCoordSphere = np.copy(spherCoord)
spherCoordSphere[spherCoord[...,0] == 0,:] = np.nan

#generate spherical coordinates over entire sphere, not just shell, for plotting
spherHarm3D = getRealSphericalHarmonics(spherCoordSphere, maxOrder)

#calculate the field from the spherical harmonic decomposition
decomposedField = np.matmul(spherHarm3D, coeffs[:np.size(spherHarm3D, axis = -1)])*sphereMask
print("Inhomogeneity of fit: %.0f ppm" %(1e6*(np.nanmax(decomposedField) - np.nanmin(decomposedField))/np.nanmean(decomposedField)))

# plot all 3 fields in 3 different planes passing through the centere of the magnet
fig, ax = plt.subplots(1,3)
ax[0].imshow(decomposedField[:,:,int(np.size(decomposedField,2)/2)], vmin = np.nanmin(decomposedField), vmax = np.nanmax(decomposedField))
ax[0].axis('off')
# ax[0].set_title("Field map")
ax[1].imshow(decomposedField[:,int(np.size(decomposedField,1)/2),:], vmin = np.nanmin(decomposedField), vmax = np.nanmax(decomposedField))

ax[1].axis('off')
# ax[1].set_title("Spherical harmonic decomposition")
ax[2].imshow(decomposedField[int(np.size(decomposedField,2)/2),:,:], vmin = np.nanmin(decomposedField), vmax = np.nanmax(decomposedField))
# ax[2].set_title("Difference")
ax[2].axis('off')
fig.set_tight_layout(True)


np.save("20221114 - OSII One CNC - 250mm DSV - 5mm - sphere - interpolated.npy", decomposedField)
