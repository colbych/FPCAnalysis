#!/usr/bin/env python

import sys
sys.path.append(".")

from FPCAnalysis import *

import os
import math
import numpy as np

try:
    analysisinputflnm = sys.argv[1]
    outdirname = sys.argv[2]
except:
    print("This generates FPC netcdf4 file from Tristan data.")
    print("usage: " + sys.argv[0] + "  analysisinputflnm outdirname")
    print("Warning: Frame num expects leading zeros. This script pads to have 3 digits (i.e. 1 or two leading zeros) but may need to be modified if more are expected.")
    sys.exit()

anlinput = anl.analysis_input(flnm = analysisinputflnm)
path = anlinput['path']
num = anlinput['numframe']
dv = anlinput['dv']
vmax = anlinput['vmax']
dx = anlinput['dx']
xlim = anlinput['xlim']
ylim = anlinput['ylim']
zlim = anlinput['zlim']
resultsdir = anlinput['resultsdir']

#add leading zeros (TODO: go to folder and figure out the number of leading zeros automatically)
_zfilllen = 3
num = str(num).zfill(_zfilllen)

os.system('mkdir '+str(outdirname))
os.system('mkdir '+str(outdirname)+ '/ion')
os.system('mkdir '+str(outdirname)+ '/elec')

#-------------------------------------------------------------------------------
# load data
#-------------------------------------------------------------------------------

print("Loading data...")
inputpath = path+'/input'
path = path+'/output/'
inputs = dtr.load_input(inputpath)

params = dtr.load_params(path,num)
dt = params['c']/params['comp'] #in units of wpe
dt,c = anl.norm_constants_tristanmp1(params,dt,inputs) #in units of wci (\omega_ci) and va respectively- as required by the rest of the scripts

beta0 = anl.compute_beta0_tristanmp1(params,inputs)

dfields = dtr.load_fields(path,num,normalizeFields=True)

#-------------------------------------------------------------------------------
# slice data
#-------------------------------------------------------------------------------
#setup sweeping box
if(xlim == None):
     xlim = [dfields['ex_xx'][0],dfields['ex_xx'][-1]]
if(ylim == None):
     ylim = [dfields['ex_yy'][0],dfields['ex_yy'][-1]]
if(zlim == None):
     zlim = [dfields['ex_zz'][0],dfields['ex_zz'][-1]]


x1 = xlim[0]
x2 = x1+dx
xEnd = xlim[1]
y1 = ylim[0]
y2 = ylim[1]
z1 = zlim[0]
z2 = zlim[1]
dparkeys='p1 p2 p3 x1 x2 x3'.split()

#dtr.load_particles(normalizeVelocity=True) normalizes positions by this same factor
#(see data_tristan.py:load_particles) - xlim/ylim/zlim above are in that normalized
#frame (they default from dfields, loaded with normalizeFields=True), but
#load_particles' x1/x2/y1/y2/z1/z2 bounds filter the RAW (pre-normalization) position
#arrays, so convert bounds to raw units before passing them in. See
#docs/optimization_plan.md Phase 1.
posnorm_scale = params['comp']*np.sqrt(params['massratio'])

while(x2 <= xEnd):
    print("x1: ", x1, "x2: ", x2)

    #load only this slice's particles directly (positions are filtered server-side via
    #h5py boolean selection inside load_particles), instead of loading the whole frame
    #once up front and masking it in memory on every iteration
    dpar_elec, dpar_ion = dtr.load_particles(path,num,normalizeVelocity=True,
                                              x1=x1*posnorm_scale, x2=x2*posnorm_scale,
                                              y1=y1*posnorm_scale, y2=y2*posnorm_scale,
                                              z1=z1*posnorm_scale, z2=z2*posnorm_scale)
    dpar_ion = dtr.format_par_like_dHybridR(dpar_ion) #For now, we rename the particle data keys too look like the keys we used when processing dHybridR data so this data is compatible with our old routines
    dpar_elec = dtr.format_par_like_dHybridR(dpar_elec)

    #write sliced data for ions
    _tempdpar = {key: dpar_ion[key] for key in dparkeys}
    outflnm = outdirname + '/ion/' + '{:012.6f}'.format(x1) + '_' + '{:012.6f}'.format(x2)
    ddhr.write_particles_to_hdf5(_tempdpar,outflnm)

    #write sliced data for elecs
    _tempdpar = {key: dpar_elec[key] for key in dparkeys}
    outflnm = outdirname + '/elec/' + '{:012.6f}'.format(x1) + '_' + '{:012.6f}'.format(x2)
    ddhr.write_particles_to_hdf5(_tempdpar,outflnm)
    x1 += dx
    x2 += dx

print("Done!")
