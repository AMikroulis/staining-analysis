import numpy as npy
import nd2reader as ndr
from scipy import signal as ssy
import skimage as skimg
from skimage import segmentation as sks
from skimage.morphology import watershed
from skimage.feature import peak_local_max
import skimage.measure as skmsr
from scipy import ndimage as ndi
import os
import sys
from datetime import date
from tkinter import filedialog



def openimage(fname):
    print (str(fname))
    nd2_img_data = []
    zslices = 0
    fields = 0
    slice_chan = 0
    slice_x = 0
    slice_y = 0
    
    with ndr.ND2Reader(fname) as images:
        print(images.sizes)
        fields = int(images.sizes['v'])
        zslices = int(images.sizes['z'])
        slice_chan = int(images.sizes['c'])
        slice_x = int(images.sizes['x'])
        slice_y = int(images.sizes['y'])
        print(images.metadata['pixel_microns'])
        calibration = float(images.metadata['pixel_microns'])
        images.bundle_axes = 'yx'
        images.iter_axes = 'vzc'
        totalslicies = zslices * slice_chan * fields
        for slicey in range(totalslicies):
            nd2_img_data.append(images[slicey])
    nd2_img_data = npy.asarray(nd2_img_data)

    nd2_img_data = npy.reshape(nd2_img_data,[fields,zslices,slice_chan,slice_x,slice_y])
    def dapi(v,z):
        return(nd2_img_data[v,z,0]) #DAPI channel at c=0 - change accordingly
    def staining(v,z):
        return(nd2_img_data[v,z,1]) #staining channel at c=1 - change accordingly

    def analyse(field,zslice):
        print('v = ' + str(field))
        print('z = ' + str(zslice))
        dapi_f = ssy.medfilt2d(dapi(field,zslice),25)
        dapi_fL = ndi.gaussian_laplace(dapi_f,15,mode = 'nearest') #laplace-gaussian filter for hard edge detection (frame edges)
        print('dapi detection...')
        dapi_d = sks.chan_vese(dapi_f,0.01) #chan-vese segmentation for entire frame
        print('dapi detection -frame edge correction')
        dapi_dL = sks.chan_vese(dapi_fL,0.01,1.,1.,0.001,640)   #chan-vese segmentation for frame edges
        dapi_dC = dapi_d*dapi_dL*dapi_f #combine segmentation results
        dapi_e = ndi.binary_erosion(dapi_dC, iterations = 10)   # reduce overlap between cells
        dapi_distance = ndi.distance_transform_edt(dapi_e)
        locmax_dapi = peak_local_max(dapi_distance, indices=False, min_distance=25, labels=dapi_e) #find distinct nucleus centers
        markers = ndi.label(locmax_dapi)[0]
        labels = watershed(-dapi_distance, markers, mask=dapi_e)    #separate cells by nucleus centers
        regions= skmsr.regionprops(labels)

        
        npy.unique(labels)
        rgns = []
        for rgn in regions:
            rgns.append(npy.asarray(rgn.centroid))
        rgns = npy.asarray(rgns)
        print(str(npy.shape(rgns)))
        print('DAPI (cells): '+ str(npy.shape(rgns)))


        #### staining ###

        staining_f = ssy.medfilt2d(staining(field,zslice),9)
        staining_fL = ndi.gaussian_laplace(staining_f,2,mode = 'nearest')
        print('staining...')
        staining_d = sks.chan_vese(staining_fL,0.01)
        staining_m0 = sks.clear_border((-staining_fL > skimg.filters.threshold_isodata(-staining_fL))*staining_d,7)
        staining_distance = ndi.distance_transform_edt(staining_m0)
        staining_locmax = peak_local_max(staining_distance,10)
        npy.shape(staining_locmax)
        staining_spots = (staining_m0*staining(field,zslice)>npy.average(staining(field,zslice)+2.5*npy.std(staining(field,zslice))))
        staining_spots_max = peak_local_max(ndi.filters.gaussian_filter(1.0 * staining_spots,1),3)
        print(str(len(staining_spots_max)))
        print('staining spots: '+ str(len(staining_spots_max)))

        print('counting:')
        print('DAPI...')
        dapi_c = npy.zeros(len(regions),dtype = 'complex')  #complex for distance calculations

        for n in range(len(regions)):
            dapi_c[n] = npy.complex(rgns.T[1][n],rgns.T[0][n])


        print('staining...')
        stainingp = npy.zeros(len(staining_spots_max), dtype = 'complex')
        for n in range(npy.size(stainingp)):
            stainingp[n] = npy.complex(staining_spots_max.T[1][n],staining_spots_max.T[0][n])


        print('measuring distances')

        dist = npy.zeros((npy.size(dapi_c),npy.size(stainingp)))
        print(npy.shape(dist))
        for dapi_cpoint in range(npy.size(dapi_c)):
            for stainingpoint in range(npy.size(stainingp)):
                dist[dapi_cpoint,stainingpoint] = npy.abs(stainingp[stainingpoint] - dapi_c[dapi_cpoint])


        scdist = dist * calibration

    for field_n in range(0,fields):
        for zslice in range(0,zslices):
            try:
                analyse(field_n,zslice)
            except:
                print ('could not count for field / z-slice ' + str(field_n)+' / '+ str(zslice))

# Copyright © Apostolos Mikroulis and Lund University, 2020
# This software is distributed under the terms of the GNU General Public License v3.
