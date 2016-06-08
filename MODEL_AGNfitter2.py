

"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

             MODEL_AGNfitter.py

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

This script contains all functions which are needed to construct the total model of AGN. 
The functions here translate the parameter space points into total fluxes dependin on the models chosen.

Functions contained here are the following:

pick_STARBURST_template
pick_GALAXY_template
pick_TORUS_template
pick_EBV_grid


STARBURST_nf
BBB_nf
GALAXY_nf
TORUS_nf

"""

import numpy as np
import math
from math import exp,log,pi
import matplotlib.pyplot as plt
from numpy import random,argsort,sqrt
import re
import os
import time
from GENERAL_AGNfitter import adict, NearestNeighbourSimple2D, NearestNeighbourSimple1D, extrap1d
from collections import defaultdict
from scipy.interpolate import interp1d
from numpy import array
from scipy.integrate import quad, trapz
from math import sqrt




"""
==============================
PICKING TEMPLATES
==============================

Functions which are used to map from a parameter space vector into a model component template. This functions are mostly called in PARAMETERSPACE_AGNfitter and PLOTandWRITE_AGNfitter.

"""


def pick_STARBURST_template(ir_lum, filename_0, ir_lum_0):

	idx = (np.abs(ir_lum_0-ir_lum)).argmin()
	filename = filename_0[idx]
	filename = 'models/STARBURST/'+str(filename) 

	return filename

#======================
#   PICK BBB TEMPLATE
#======================


def pick_BBB_template():
	return 'models/BBB/richardsbbb.dat'


#======================
#   PICK GALAXY TEMPLATE
#======================


def pick_GALAXY_template( tau, age, filename_0, tau_sep, age_sep):

    
    idx = (np.abs((tau_sep-tau) + (age_sep-age))).argmin()	
    filename = filename_0[idx]
    filename = 'models/GALAXY/'+str(filename)


    return filename 


def pick_TORUS_template(nh, file_nh, filename_0):
	
	file_nh=file_nh.astype(float)

	idx = (np.abs(file_nh-nh)).argmin()

	filename = filename_0[idx]


	return filename 

def pick_EBV_grid (EBV_array, EBV):

	
	idx = (np.abs(EBV_array-EBV)).argmin()

	EBV_fromgrid  = EBV_array[idx]

	return EBV_fromgrid

#==============================
# MAXIMAL POSSIBLE AGE FOR GALAXY MODEL
#==============================


def maximal_age(z):

	z = np.double(z)
	#Cosmological Constants	
	O_m = 0.266
	O_r =  0.
	O_k= 0.
	O_L = 1. - O_m
	H_0 = 74.3 #km/s/Mpc
	H_sec = H_0 / 3.0857e19 
	secondsinyear = 31556926
	ageoftheuniverse = 13.798e9

	# Equation for the time elapsed since z and now

	a = 1/(1+z)
	E = O_m * (1+z)**3 + O_r *(1+z)**4 + O_k *(1+z) + O_L
	integrand = lambda z : 1 / (1+z)     / sqrt(  O_m * (1+z)**3 + O_r *(1+z)**4 + O_k *(1+z) + O_L  )		

	#Integration
	z_obs = z
	z_cmb = 1089 #As Beta (not cmb). But 1089 (cmb) would be the exagerated maximun possible redshift for the birth 
	z_now = 0


	integral, error = quad( integrand , z_obs, z_cmb) #
	
	#t = ageoftheuniverse - (integral * (1 / H_sec) / secondsinyear)
	t = (integral * (1 / H_sec)) / secondsinyear

	return t



#===================================================
#        
#                                              #        MODELS          #
#
#
#                                     Reading fluxes from model template files,
#                                                               &
#                  interpolate arrays of fluxes to make them comparable to data arrays
#===================================================


#==== STARBURST (Dale&Helou, Chary&Elbaz,) ======


def STARBURST_read (fn):

	"""
	This function computes interpolated fluxes of the model STARBURST at the observed frequencies of the DATA 
	with _nf NO FILTERING

	## inputs:
	- fn: file name for template wavelengths and fluxes
	- data_nu : data frequencies
	- z: redshift of the source

	## output:
	- returns fluxes for the starburst model contribution 

	## comments:
	- the redshift is just a shift of the frequencies before the interpolation

	## improvements todo:
	- change to nuFnu= lambdaFlambda, intead of Flambda 2 Fnu
	"""

	# in file F_lambda = ergs /s / cm2 / A
	# They have been converted in flux as                              
	#            Flbda[erg/s/cm2/A] = nuLnu/lbda * fac                      
	#            where  fac= Lo/(4.pi.D^2)  with  D=10pc                    
	#        Lsol=3.826e33[erg/s]  ,   pc=3.086e18[cm]                     
	#        fac= Lsol/(4*3.141593*100*pc**2)= 3.197004226e-07	

	#reading
	c = 2.997e10
	c_Angst = 3.34e-19 #(1/(c*Angstrom)
	
	dh_wl_rest, dh_Flambda =  np.loadtxt(fn, usecols=(0,1),unpack= True)
	dh_wl = dh_wl_rest 
	dh_nu_r = np.log10(c / (dh_wl * 1e-8)) 
	dh_Fnu = dh_Flambda * (dh_wl**2. )* c_Angst

	#reverse , in order to have increasing frequency
	dh_nus= dh_nu_r[::-1]
	dh_Fnu = dh_Fnu[::-1]

	return dh_nus, dh_Fnu



def STARBURST_interp2data(dh_nu, dh_Fnu, data_nu, z):

	#interpolation
	dh_nu_obs= dh_nu/(1+z)
	dh = interp1d(dh_nu, dh_Fnu, bounds_error=False, fill_value=0.)
	dh_x = 10**(data_nu)
	dh_y = dh(dh_x)
	#This output are observed monochromatic flux (not strictly, just assumed because they are templates)

	return dh_y



def BBB_nf(fn, BBebv, data_nu, str_catalog, sourceline,z ):
	"""
	This function computes interpolated fluxes of the model BBB at the observed frequencies (data_nu) 
	with _nf NO FILTERING.

	## inputs:
	- fn: file name for template wavelengths and fluxes
	- data_nu : data frequencies observed
	- z: redshift of the source

	## output:
	- returns fluxes for the starburst model contribution 

	## comments:
	- 

	## bugs:
	- change to nuFnu= lambdaFlambda, intead of Flambda 2 Fnu
	"""


	distance = z2Dlum(z)

	bbb_nu_log_rest, bbb_nuLnu_log = np.loadtxt(fn, usecols=(0,1),unpack= True)
	bbb_nu = 10**(bbb_nu_log_rest) 
	bbb_nuLnu= 10**(bbb_nuLnu_log)
	bbb_Lnu = bbb_nuLnu / bbb_nu



	#Reddening
	RV= 2.72

	#converting freq to wavelenght, to be able to use prevots function instead on simple linera interpolation 
	redd_x =  2.998 * 1e8 / (bbb_nu)* 1e10
	redd_x= redd_x[::-1]
	
	#Define prevots function for the reddenin law redd_k
	
	def function_prevot(x, RV):
   		y=1.39*pow((pow(10.,-4.)*x),-1.2)-0.38 ;
   		return y 

	bbb_k = function_prevot(redd_x, RV)

	#converting back  wavelenght to freq
	redd_f_r= 2.998 * 1e8 / (redd_x) * 1e10
	redd_f = redd_f_r[::-1]	
	bbb_k= bbb_k[::-1]


	bbb_Lnu_red = bbb_Lnu * 10**(-0.4 * bbb_k * BBebv)


	bbb_nu_obs = bbb_nu /(1+z)
	# interpolate

	bbb = interp1d(bbb_nu_obs, bbb_Lnu_red, bounds_error=False, fill_value=0.)
	bbb2 = extrap1d(bbb)	
	bbb_x = 10**(data_nu)
	
	bbb_y = bbb2(bbb_x)
	
	# This output is a restframe monochromatic luminosity, to have the same output as Beta
	return bbb_y



#========== BBB (Richards) with reddening =============

def BBB_read(fn):
	"""
	This function just reads the model template files and five nu and Lnu

	"""

	bbb_nu_log_rest, bbb_nuLnu_log = np.loadtxt(fn, usecols=(0,1),unpack= True)
	bbb_nu_exp = 10**(bbb_nu_log_rest) 
	bbb_nu = np.log10(10**(bbb_nu_log_rest) )
	bbb_nuLnu= 10**(bbb_nuLnu_log)
	bbb_Lnu = bbb_nuLnu / bbb_nu

	bbb_x = bbb_nu
	bbb_y =	bbb_nuLnu  / bbb_nu_exp

	return bbb_x, bbb_y



def BBB_nf2(bbb_x, bbb_y, BBebv, z ):

	"""
	This function aplies  the reddening on bbb_y
	"""

	#Application of reddening - reading E(B-V) from MCMC sampler
	RV= 2.72

	#converting freq to wavelenght, to be able to use prevots function instead on simple linera interpolation 
	redd_x =  2.998 * 1e10 / (10**(bbb_x)* 1e-8)
	redd_x= redd_x[::-1]

	#	Define prevots function for the reddenin law redd_k	
	def function_prevot(x, RV):
   		y=1.39*pow((pow(10.,-4.)*x),-1.2)-0.38 ;
   		return y 

	bbb_k = function_prevot(redd_x, RV)

	bbb_k= bbb_k[::-1]

        bbb_Lnu_red = bbb_y * 10**(-0.4 * bbb_k * BBebv)

	return bbb_x, bbb_Lnu_red

def BBB_interp2data(bbb_x, bbb_Lnu_red, data_nu, z):

	"""
	This function interpolates to observed frequencies of data
	"""

	bbb_nu_obs = (10**bbb_x) /(1+z)

	# interpolate
	bbb = interp1d(bbb_nu_obs, bbb_Lnu_red, bounds_error=False, fill_value=0.)
	bbb2 = extrap1d(bbb)	
	bbb_x = 10**(data_nu)
	
	bbb_y = bbb2(bbb_x)

	# This output is a restframe monochromatic luminosity, to have the same output as Beta
	return bbb_y











def GALAXY_nf(galaxy_file,  GAebv, data_nu, str_catalog, sourceline, z):


	gal_wl_rest, gal_flux_la = np.loadtxt(galaxy_file, skiprows=2, usecols=(0,1),unpack= True)
	gal_Fnu_r= gal_flux_la * 3.34e-19 * gal_wl_rest**2.  

	gal_nu_rest =2.998 * 1.e8 / gal_wl_rest * 1.e10

	gal_nu= gal_nu_rest[::-1]
	gal_Fnu= gal_Fnu_r[::-1]
	
        
	RV = 4.05		
	wl = np.arange(0.122, 2.18, 0.02)
	redd_k=[]
	for 	i in range(len(wl)):
		if (wl[i]>0.12 and wl[i]<0.63):
			k =   2.659*(-2.156+(1.509/wl[i])-(0.198/(wl[i]**2))+(0.011/(wl[i]**3)) )+RV
		elif (wl[i]>0.63and wl[i]<2.2):
			k =  2.659*(-1.857+(1.040/wl[i]))+RV
		redd_k.append(k)
	
	micron2cm = 1e-4
	redd_k= np.array(redd_k)
	redd_wl_rest = wl*micron2cm	
	redd_wl = redd_wl_rest 

	redd_f_r= 2.998 * 1e10 / (redd_wl)
	redd_f_r = np.log10(redd_f_r) 


	redd_f = redd_f_r[::-1]
	redd_k= redd_k[::-1]

	reddening = interp1d(redd_f, redd_k, bounds_error=True)
	reddening2 = extrap1d(reddening)
	redd_x = np.log10(gal_nu)
	gal_k = reddening2(redd_x)

	gal_Fnu_red = gal_Fnu* 10**(-0.4 * gal_k * GAebv)


	gal_nu_obs =gal_nu /(1+z)

	gal = interp1d(gal_nu_obs, gal_Fnu_red, bounds_error=False, fill_value=0.)

	gal_x = 10**data_nu
	gal_y = gal(gal_x) 

	return gal_y


#============================================================

def GALAXY_read(galaxy_file):

	gal_wl_rest, gal_flux_la = np.loadtxt(galaxy_file, skiprows=2, usecols=(0,1),unpack= True)
	gal_Fnu_r= gal_flux_la * 3.34e-19 * gal_wl_rest**2.  

	gal_nu_rest =2.998 * 1.e8 / gal_wl_rest * 1.e10
	#converting to nuFnu

	# reverse
	gal_nu= gal_nu_rest[::-1]
	gal_Fnu= gal_Fnu_r[::-1]

	return gal_nu, gal_Fnu




#==== INTERPOLATING AND REDDENING


def GALAXY_nf2( gal_nu, gal_Fnu,GAebv):
	
	
	RV = 4.05		
	wl = np.arange(0.122, 2.18, 0.02)
	redd_k=[]
	for 	i in range(len(wl)):
		if (wl[i]>0.12 and wl[i]<0.63):
			k =   2.659*(-2.156+(1.509/wl[i])-(0.198/(wl[i]**2))+(0.011/(wl[i]**3)) )+RV
		elif (wl[i]>0.63and wl[i]<2.2):
			k =  2.659*(-1.857+(1.040/wl[i]))+RV
		redd_k.append(k)
	
	micron2cm = 1e-4
	redd_k= np.array(redd_k)
	redd_wl_rest = wl*micron2cm	
	redd_wl = redd_wl_rest 

	redd_f_r= 2.998 * 1e10 / (redd_wl)
	redd_f_r = np.log10(redd_f_r) 
	redd_f = redd_f_r[::-1]
	redd_k= redd_k[::-1]
	reddening = interp1d(redd_f, redd_k, bounds_error=True)


	reddening2 = extrap1d(reddening)
	
	
	if (np.amax(gal_nu) - np.amax(redd_f)) <= 1e7:
		redd_x = gal_nu	
		
	else: 	
		redd_x = np.log10(gal_nu)
		
	gal_k = reddening2(redd_x)

	gal_Fnu_red = gal_Fnu* 10**(-0.4 * gal_k * GAebv)

	
	return gal_nu, gal_Fnu_red

def GALAXY_interp2data(gal_nu, gal_Fnu_red, data_nu, z):

	gal_nu_obs =gal_nu /(1+z)
	gal = interp1d(gal_nu_obs, gal_Fnu_red, bounds_error=False, fill_value=0.)
	gal_x = 10**data_nu
	gal_y = gal(gal_x)
	
	return gal_y


#====
def TORUS_read(tor_file, z):

#The files (torus_5nh) give log of freq, so        f= 10**inputfreq
#The files give monochromatic luminosities
#so just multiply by the       nuL_nu = inputlum *f
#The files (qso_21.5... etc) have the format "lambda, nu,nu L_nu", so just read 

	#distance value is just a normalization value to convert to Flux (z is no more needed!!!)
	distance= 1e27###z2Dlum(z)

	tor_nu_rest, tor_nuLnu = np.loadtxt(tor_file, skiprows=0, usecols=(0,1),unpack= True)
	tor_Lnu = tor_nuLnu / 10**(tor_nu_rest)	
	tor_Fnu = tor_Lnu /(4. * pi * distance**2.) 

	return tor_nu_rest, tor_Fnu 

	
def TORUS_interp2data(tor_nu, tor_Fnu, data_nu):
	

	tor = interp1d(tor_nu, tor_Fnu, bounds_error=False, fill_value=0.)

	tor_x = 10**data_nu
	tor_Fnu= tor(tor_x)
	#Coverting to fluxes
	tor_y = tor_Fnu 	

	return tor_y



#============================================================

#SEPARATING READIN AND  REDDENING IN TWO STEPS





#============================================================
#
#				#		OTHER PHYSICAL OUTPUTS                  #
#
#
# # Interprete MCMC output to calculate estimates of other physical parameters
#
#============================================================

#==============================================================
#						STAR FORMATION INFORMATION
#==============================================================




#==================
#CONSTANTS
#==================


c = 2.99792458e8
Angstrom = 1e10




def z2Dlum(z):
	
	#Cosmo Constants
	
	O_m = 0.266
	O_r =  0.
	O_k= 0.
	O_L = 1. - O_m
	H_0 = 70. #km/s/Mpc
	H_sec = H_0 / 3.0857e19 
	c = 2.997e10

	# equation

	a = 1/(1+z)
	E = O_m * (1+z)**3 + O_r *(1+z)**4 + O_k *(1+z) + O_L
	integrand = lambda z : 1 / sqrt(O_m * (1+z)**3 + O_r *(1+z)**4 + O_k *(1+z) + O_L)	

	#integration

	z_obs = z
	z_now = 0
	
	integral = quad( integrand , z_now, z_obs)	

	dlum_cm = (1+z)*(c / H_sec) * integral[0] 
	dlum_Mpc = dlum_cm/3.08567758e24

	return dlum_cm
      



#===================================================
#        
#                                              #        MODELS   #
#
#
#                                     Reading models from files,
#                                                               &
#                  interpolate arrays of models to make them comparable to data
#===================================================


def STARBURST2(dh_nus, dh_Fnus, z, filterdict):

	bands, dh_Fnu_filtered = filters1(dh_nus, dh_Fnus, filterdict, z)	

	dh_Fnu_filtered = dh_Fnu_filtered.reshape(np.shape(bands)) 

	return bands, dh_Fnu_filtered



#========== BBB (Richards) with reddening =============


def BBB2(bbb_nu, bbb_y_red, filterdict,z):


	bands, bbb_y_red_filtered =  filters1(bbb_nu, bbb_y_red, filterdict, z)	

	bbb_y_red_filtered = bbb_y_red_filtered.reshape(np.shape(bands)) 

	# This output is a restframe monochromatic luminosity, to have the same output as Beta
	return bands, bbb_y_red_filtered



#==== GALAXY (BC03, solLum Angstrom-1) =====

def GALAXY2(gal_nu, gal_Fnu_red,  filterdict, z):

	#convert to log for consistency with function GALAXY

	gal_nu = np.log10(gal_nu)
	bands, gal_y_red_filtered =  filters1(gal_nu, gal_Fnu_red, filterdict, z)	

	gal_y_red_filtered = gal_y_red_filtered.reshape(np.shape(bands)) 

	return bands, gal_y_red_filtered



#==== TORUS (sed_data, solLum Angstrom-1) =====

def TORUS2(tor_nu, tor_Fnu, z, filterdict):

	bands, tor_Fnu_filtered =  filters1(tor_nu, tor_Fnu, filterdict, z )
	
	tor_Fnu_filtered = tor_Fnu_filtered.reshape(np.shape(bands)) 

	return bands, tor_Fnu_filtered



#============================================================
#
#				#		OTHER PHYSICAL OUTPUTS                  #
#
#
# # Interprete MCMC output to calculate estimates of other physical parameters
#
#============================================================


#==============================================================
#		 NEAREST NEIGHBOUR INTERPOLATION ALGORITHM : SIMPLE 1D
#==============================================================



def stellar_info(chain, data):

    stellar_templist= 'models/GALAXY/input_template_hoggnew.dat'	    
    listlines = np.arange(len(stellar_templist)) 	
    tau_mcmc = chain[:,0] 	
    age_mcmc = chain[:,1] 
    GA = chain[:,6] - 18. #1e18 is the common normalization factor used in parspace.ymodel in order to have comparable NORMfactors	

    tau_column, age_column = np.loadtxt(stellar_templist,  usecols=(2,3), skiprows=0, unpack= True)  

    z = data.z
    distance = z2Dlum(z)

   #constants
    Mpc2cm = 3.086e24
    solarlum = 3.839e33
    solarmass = 1.9891e30	
    
    Mstar_list=[]
    SFR_list=[]
    SFR_file_list=[]	

    for i in range (len (tau_mcmc)):		

	N = 10**GA[i]* 4* pi* distance**2 / (solarlum)/ (1+z)

	# Reading tau and age of the galaxy template file chosen 
	y = np.array([tau_mcmc[i]]) 
	a = np.array([age_mcmc[i]])

	nntau = NearestNeighbourSimple1D(y, tau_column , 1)
        nnage = NearestNeighbourSimple1D(10**a, age_column , 1)
    
        mm= [(tau_column==tau_column[nntau]) & (age_column==age_column[nnage])]
        index=np.arange(len(tau_column))[mm]
		
        tau = tau_column[index]
        age = age_column[index]
 
	agelog = np.log10(age)

	#comparing tau, and chose the right file to read agelog, mstar and sfr 



	if tau<0.05:
		agelog_file, mstar_file, sfr_file = np.loadtxt('models/GALAXYinfo/bc2003_lr_m62_chab_const_ifort.4color', usecols=(0,6,9), skiprows=0, unpack= True)
	elif tau<0.2:
		agelog_file,  mstar_file, sfr_file= np.loadtxt('models/GALAXYinfo/bc2003_lr_m62_chab_tau01_ifort.4color', usecols=(0,6,9), skiprows=0,unpack= True)
	elif tau<0.45:
		agelog_file,  mstar_file, sfr_file = np.loadtxt('models/GALAXYinfo/bc2003_lr_m62_chab_tau03_ifort.4color', usecols=(0,6,9),skiprows=0, unpack= True)
	elif tau<0.8:
		agelog_file,  mstar_file, sfr_file = np.loadtxt('models/GALAXYinfo/bc2003_lr_m62_chab_tau06_ifort.4color', usecols=(0,6,9),skiprows=0, unpack= True)
	elif tau<1.5:
		agelog_file, mstar_file, sfr_file = np.loadtxt('models/GALAXYinfo/bc2003_lr_m62_chab_tau1_ifort.4color', usecols=(0,6,9), skiprows=0,unpack= True)
	elif tau<2.5:
		agelog_file, mstar_file, sfr_file= np.loadtxt('models/GALAXYinfo/bc2003_lr_m62_chab_tau2_ifort.4color', usecols=(0,6,9), skiprows=0,unpack= True)
	elif tau<4:
		agelog_file,  mstar_file, sfr_file= np.loadtxt('models/GALAXYinfo/bc2003_lr_m62_chab_tau3_ifort.4color', usecols=(0,6,9),skiprows=0, unpack= True)
	elif tau<7.5:
		agelog_file,  mstar_file, sfr_file= np.loadtxt('models/GALAXYinfo/bc2003_lr_m62_chab_tau5_ifort.4color', usecols=(0,6,9), skiprows=0,unpack= True)
	elif tau<12.5:
		agelog_file, mstar_file, sfr_file= np.loadtxt('models/GALAXYinfo/bc2003_lr_m62_chab_tau10_ifort.4color', usecols=(0,6,9), skiprows=0,unpack= True)
	elif tau<22.5:
		agelog_file,  mstar_file, sfr_file= np.loadtxt('models/GALAXYinfo/bc2003_lr_m62_chab_tau15_ifort.4color', usecols=(0,6,9), skiprows=0,unpack= True)
	else: 
		agelog_file, mstar_file, sfr_file= np.loadtxt('models/GALAXYinfo/bc2003_lr_m62_chab_tau30_ifort.4color', usecols=(0,6,9), skiprows=0,unpack= True)

	#comparing agelog and reading right line of mstar and sfr
	
	x = np.array([agelog]) 
	
	nn= NearestNeighbourSimple1D(agelog, agelog_file , 1)
	nn= int(nn)
	mstar_line = mstar_file[nn]
	SFR_line = sfr_file[nn]


	#Calculate Mstar
	Mstar = N * mstar_line

	Mstar_list.append(Mstar)	
	
	#Calculate SFR. output is in [Msun/yr]
	SFR = N *  exp(-(10**agelog_file[nn]/ 1e9)/ tau) / (tau* 1e9)

	SFR_list.append(SFR)	

	#Calculate SFR from file to doublecheck
	SFR_file = N* SFR_line
	SFR_file_list.append(SFR_file)	

    Mstar = np.array(Mstar_list)	
    SFR = np.array(SFR_list)
    SFR_file = np.array(SFR_file_list)	


    return Mstar, SFR, SFR_file


def stellar_info_array(chain_flat, data, Nthin_compute):

    Ns, Npar = np.shape(chain_flat)  
    chain_thinned = chain_flat[0:Ns:int(Ns/Nthin_compute),:]
    
    Mstar, SFR, SFR_file = stellar_info(chain_thinned, data)
    Mstar_list = []
    SFR_list = []
    SFR_file_list = []

    for i in range(Nthin_compute):
        for j in range(int(Ns/Nthin_compute)):
            Mstar_list.append(Mstar[i])
            SFR_list.append(SFR[i])
            SFR_file_list.append(SFR_file[i])

    Mstar1 = np.array(Mstar_list)        
    SFR1 = np.array(SFR_list)
    SFR_file1 = np.array(SFR_file_list)

    return Mstar1, SFR1, SFR_file1



def sfr_IR(logL_IR):
	#calculate SFR in solar M per year 

	#for an array ofluminosities
	if len(logL_IR)>1:
		SFR_IR_list =[]

		for i in range(len(logL_IR)):
			SFR = 3.88e-44* (10**logL_IR[i])
			SFR_IR_list.append(SFR)
		SFR_IR_array = np.array(SFR_IR_list)
		return SFR_IR_array
	#or just for one luminosity
	else:		
		SFR = 3.88e-44* (10**logL_IR)
		return SFR

def stellar_info_best(best_fit_par, data):

    stellar_templist= 'models/GALAXY/input_template_hoggnew.dat'	    
    listlines = np.arange(len(stellar_templist)) 	
    tau_mcmc = best_fit_par[0] 	
    age_mcmc = best_fit_par[1] 
    GA = best_fit_par[6] - 18 #1e18 is the common normalization factor used in parspace.ymodel in order to have comparable NORMfactors	

    tau_column, age_column = np.loadtxt(stellar_templist,  usecols=(2,3), skiprows=0, unpack= True)  

    z = data.z
    distance_cm = z2Dlum(z) #in cm
    distance = distance_cm/100
   #constants
    Mpc2cm = 3.086e24
    solarlum = 3.839e33
    solarmass = 1.9891e30	
    
    Mstar_list=[]
    SFR_list=[]
    SFR_file_list=[]	

    for i in range (1):		

	N = 10**GA* 4* pi* distance**2 / (solarlum)/ (1+z)

	# Reading tau and age of the galaxy template file chosen 
	y = np.array([tau_mcmc]) 
	a = np.array([age_mcmc]) 
	nntau = NearestNeighbourSimple1D(y, tau_column , 1)
        nnage = NearestNeighbourSimple1D(10**a, age_column , 1)
    
        mm= [(tau_column==tau_column[nntau]) & (age_column==age_column[nnage])]
        index=np.arange(len(tau_column))[mm]
		
        tau = tau_column[index]
        age = age_column[index]
 
	agelog = np.log10(age)

	#comparing tau, and chose the right file to read agelog, mstar and sfr 

	if tau<0.05:
		agelog_file, mstar_file, sfr_file = np.loadtxt('models/GALAXYinfo/bc2003_lr_m62_chab_const_ifort.4color', usecols=(0,6,9), skiprows=0, unpack= True)
	elif tau<0.2:
		agelog_file,  mstar_file, sfr_file= np.loadtxt('models/GALAXYinfo/bc2003_lr_m62_chab_tau01_ifort.4color', usecols=(0,6,9), skiprows=0,unpack= True)
	elif tau<0.45:
		agelog_file,  mstar_file, sfr_file = np.loadtxt('models/GALAXYinfo/bc2003_lr_m62_chab_tau03_ifort.4color', usecols=(0,6,9),skiprows=0, unpack= True)
	elif tau<0.8:
		agelog_file,  mstar_file, sfr_file = np.loadtxt('models/GALAXYinfo/bc2003_lr_m62_chab_tau06_ifort.4color', usecols=(0,6,9),skiprows=0, unpack= True)
	elif tau<1.5:
		agelog_file, mstar_file, sfr_file = np.loadtxt('models/GALAXYinfo/bc2003_lr_m62_chab_tau1_ifort.4color', usecols=(0,6,9), skiprows=0,unpack= True)
	elif tau<2.5:
		agelog_file, mstar_file, sfr_file= np.loadtxt('models/GALAXYinfo/bc2003_lr_m62_chab_tau2_ifort.4color', usecols=(0,6,9), skiprows=0,unpack= True)
	elif tau<4:
		agelog_file,  mstar_file, sfr_file= np.loadtxt('models/GALAXYinfo/bc2003_lr_m62_chab_tau3_ifort.4color', usecols=(0,6,9),skiprows=0, unpack= True)
	elif tau<7.5:
		agelog_file,  mstar_file, sfr_file= np.loadtxt('models/GALAXYinfo/bc2003_lr_m62_chab_tau5_ifort.4color', usecols=(0,6,9), skiprows=0,unpack= True)
	elif tau<12.5:
		agelog_file, mstar_file, sfr_file= np.loadtxt('models/GALAXYinfo/bc2003_lr_m62_chab_tau10_ifort.4color', usecols=(0,6,9), skiprows=0,unpack= True)
	elif tau<22.5:
		agelog_file,  mstar_file, sfr_file= np.loadtxt('models/GALAXYinfo/bc2003_lr_m62_chab_tau15_ifort.4color', usecols=(0,6,9), skiprows=0,unpack= True)
	else: 
		agelog_file, mstar_file, sfr_file= np.loadtxt('models/GALAXYinfo/bc2003_lr_m62_chab_tau30_ifort.4color', usecols=(0,6,9), skiprows=0,unpack= True)

	#comparing agelog and reading right line of mstar and sfr
	
	x = np.array([agelog]) 
	
	nn= NearestNeighbourSimple1D(agelog, agelog_file , 1)
	nn= int(nn)
	mstar_line = mstar_file[nn]
	SFR_line = sfr_file[nn]	
	
	#Calculate Mstar
	Mstar = N * mstar_line
	#Calculate SFR. output is in [Msun/yr]
	SFR = N *  exp(-(10**agelog_file[nn]/ 1e9)/ tau) / (tau* 1e9)
	SFR_file = N* SFR_line #Calculate SFR from file to doublecheck


    return Mstar, SFR, SFR_file




"""=========================================================="""




def filters1( model_nus, model_fluxes, filterdict, z ):	

	bands, files_dict, lambdas_dict, factors_dict = filterdict
	filtered_model_Fnus = []

# For each data point
 	for iband in bands:
	
		# Read  filter info for each data point
		lambdas_filter = np.array(lambdas_dict[iband])
		factors_filter = np.array(factors_dict[iband])
		iband_angst = nu2lambda_angstrom(np.array([iband]))
			
		# Interpolate the model to the wavelengths given by the filter files
		#Ugly conversions from fluxes in lambda to freque and other way round and miltiple times checked to be right
		model_lambdas = nu2lambda_angstrom(model_nus) * (1+z)
		model_lambdas = 	model_lambdas[::-1]
		model_fluxes_nu = model_fluxes[::-1]
			
		model_fluxes_lambda = fluxnu_2_fluxlambda(model_fluxes_nu, model_lambdas) 
	
		model_lambdas_observed = model_lambdas 

		mod2filter_interpol = interp1d(model_lambdas_observed, model_fluxes_lambda, bounds_error=False, fill_value=0.)	
		
		modelfluxes_at_filterlambdas = mod2filter_interpol(lambdas_filter)
			
		# Compute the flux ratios, equivalent to the filtered fluxes in Angstron: F = int(model)/int(filter)
	
		integral_model = trapz(modelfluxes_at_filterlambdas*factors_filter, x= lambdas_filter)
		integral_filter = trapz(factors_filter, x= lambdas_filter) 
	
		filtered_modelF_lambda = (integral_model/integral_filter)
			

		# Convert all from lambda, F_lambda  to Fnu and nu	
		filtered_modelFnu_atfilter_i = fluxlambda_2_fluxnu(filtered_modelF_lambda, iband_angst)
		filtered_model_Fnus.append(filtered_modelFnu_atfilter_i)
			
	filtered_model_Fnus= np.array(filtered_model_Fnus)
	
	return bands, filtered_model_Fnus
#==================================================================

def interpolate_DictandData(bands, filtered_model_Fnus, data_nus):

	filtered_model_Fnus_at_data = []

	for nu in data_nus:

		nu = np.array([nu])		
		datapoint = bands[int(NearestNeighbourSimple1D(nu, bands, 1))] 
		datapoint = np.array(datapoint) #has to be an array in order to use np.where
		filtered_model_Fnus_at_data.append(filtered_model_Fnus[np.where(bands==datapoint)])

	filtered_model_Fnus_at_data= np.array(filtered_model_Fnus_at_data)

	return filtered_model_Fnus_at_data


#==================================================================

















#----------------------CONVERT TO NU AND INTRPOL
def interp_models_2_filterlambdas (model_nus, model_fluxes, lambdas_filter):

	t0 = time.time
	model_lambdas = nu2lambda_angstrom(model_nus)



	model_lambdas = 	model_lambdas[::-1]
	model_fluxes_nu = model_fluxes[::-1]
	
	#convert model F_nu to model F_lambda per angstrom
	model_fluxes_lambda = fluxnu_2_fluxlambda(model_fluxes_nu, model_nus) 

	mod2filter_interpol = interp1d(model_lambdas, model_fluxes_lambda, bounds_error=False, fill_value=0.)
	modelfluxes_at_filterlambdas = mod2filter_interpol(lambdas_filter)


	return modelfluxes_at_filterlambdas




def filtered_modelpoint(lambdas_filter, factors_filter, modelfluxes_at_filterlambdas):

	
	integral_model = trapz(modelfluxes_at_filterlambdas*factors_filter, x= lambdas_filter)
	integral_filter = trapz(factors_filter, x= lambdas_filter) 



	filtered_modelfluxes = (integral_model)#/integral_filter)

	return filtered_modelfluxes

def fluxlambda_2_fluxnu (flux_lambda, wl_angst):

	c = 2.99792458e8

	flux_nu = flux_lambda * (wl_angst**2. ) / c /Angstrom
	return flux_nu

def fluxnu_2_fluxlambda (flux_nu, wl_angst):

	c = 2.99792458e8 
	 
	flux_lambda = flux_nu / wl_angst**2 *c * Angstrom

	return flux_lambda #in angstrom

def nu2lambda_angstrom(nus):

	c = 2.99792458e8#ms
	lambdas_list = []

	for i in nus:
		lambdas = c / (10**i) *Angstrom
		lambdas_list.append(lambdas)  
	lambdas = np.array(lambdas_list)

	return lambdas












def STARBURST_read_4plotting (fn, all_model_nus):

	#reading
	c = 2.997e10
	c_Angst = 3.34e-19 #(1/(c*Angstrom)
	

	dh_wl_rest, dh_Flambda =  np.loadtxt(fn, usecols=(0,1),unpack= True)
	dh_wl = dh_wl_rest 
	dh_nu_r = np.log10(c / (dh_wl * 1e-8)) 
	dh_Fnu_r = dh_Flambda * (dh_wl**2. )* c_Angst

	#reverse , in order to have increasing frequency
	dh_nus= dh_nu_r[::-1]
	dh_Fnu = dh_Fnu_r[::-1]

	SB = interp1d(10**dh_nus, dh_Fnu, bounds_error=False, fill_value=0.)

	dh_Fnus = SB(10**all_model_nus)

	return all_model_nus, dh_Fnus


#====
def TORUS_read_4plotting(tor_file,z, all_model_nus):

#The files (torus_5nh) give log of freq, so        f= 10**inputfreq
#The files give monochromatic luminosities
#so just multiply by the       nuL_nu = inputlum *f
#The files (qso_21.5... etc) have the format "lambda, nu,nu L_nu", so just read 

	distance= 1e27###z2Dlum(z)
	tor_nu_rest, tor_nuLnu = np.loadtxt(tor_file, skiprows=0, usecols=(0,1),unpack= True)

	tor_Lnu = tor_nuLnu / 10**(tor_nu_rest)	
	tor_Fnu = tor_Lnu /(4. * pi * distance**2.)# *(1+z)

	#TO = interp1d(tor_nu_rest, tor_Fnu, bounds_error=False, fill_value=0.)
	TO = interp1d(tor_nu_rest, tor_Fnu, bounds_error=False, fill_value=0.)

	tor_Fnus = TO(all_model_nus)

	return all_model_nus, tor_Fnus



def GALAXY_read_4plotting(galaxy_file, all_model_nus):

	gal_wl_rest, gal_flux_la = np.loadtxt(galaxy_file, skiprows=2, usecols=(0,1),unpack= True)
	gal_Fnu_r= gal_flux_la * 3.34e-19 * gal_wl_rest**2.  

	gal_nu_rest =2.998 * 1.e8 / gal_wl_rest * 1.e10
	#converting to nuFnu

	# reverse
	gal_nu= gal_nu_rest[::-1]
	gal_Fnu= gal_Fnu_r[::-1]

	GA = interp1d(gal_nu, gal_Fnu, bounds_error=False, fill_value=0.)
	gal_Fnus = GA(10**all_model_nus)

	return all_model_nus, gal_Fnus


def BBB_read_4plotting(fn, all_model_nus):


	bbb_nu_log_rest, bbb_nuLnu_log = np.loadtxt(fn, usecols=(0,1),unpack= True)
	bbb_nu_exp = 10**(bbb_nu_log_rest) 
	bbb_nu = np.log10(10**(bbb_nu_log_rest) )
	bbb_nuLnu= 10**(bbb_nuLnu_log)
	bbb_Lnu = bbb_nuLnu / bbb_nu

	bbb_x = bbb_nu
	bbb_y =	bbb_nuLnu  / bbb_nu_exp

	BB= interp1d(10**bbb_x, bbb_y, bounds_error=False, fill_value=0.)
	bbb_Fnus = BB(10**all_model_nus)

	return all_model_nus, bbb_Fnus























#===================================================
#        OLD NOT USED ANYMORE FUNCTIONS
#
#
#===================================================


#==== STARBURST (Dale&Helou, Chary&Elbaz,) ======


def STARBURST (sbfile, data_nu, z, filterset, bands, files_dict, lambdas_dict, factors_dict):


	c = 2.997e10
	c_Angst = 3.34e-19 #(1/(c*Angstrom)
	

	dh_wl_rest, dh_Flambda =  np.loadtxt(sbfile, usecols=(0,1),unpack= True)
	dh_wl = dh_wl_rest 
	dh_nu_r = np.log10(c / (dh_wl * 1e-8)) 
	dh_Fnu = dh_Flambda * (dh_wl**2. )* c_Angst

	#reverse , in order to have increasing frequency
	dh_nus= dh_nu_r[::-1]
	dh_Fnu = dh_Fnu[::-1]

	dh_Fnu_filtered = filters(data_nu, dh_nus, dh_Fnu, filterset, z,bands, files_dict, lambdas_dict, factors_dict)	

	dh_Fnu_filtered = dh_Fnu_filtered.reshape(np.shape(data_nu)) 
	return dh_Fnu_filtered



#========== BBB (Richards) with reddening =============


def BBB(str, BBebv, data_nu, z, filterset,bands, files_dict, lambdas_dict, factors_dict):

	bbb_nu_log_rest, bbb_nuLnu_log = np.loadtxt(str, usecols=(0,1),unpack= True)
	bbb_nu_exp = 10**(bbb_nu_log_rest) 
	bbb_nu = np.log10(10**(bbb_nu_log_rest) )
	bbb_nuLnu= 10**(bbb_nuLnu_log)

	bbb_x = bbb_nu
	bbb_y =	bbb_nuLnu  / bbb_nu_exp

#	Application of reddening - reading E(B-V) from MCMC sampler
	RV= 2.72

	#converting freq to wavelenght, to be able to use prevots function instead on simple linera interpolation 
	redd_x =  2.998 * 1e10 / (10**(bbb_x)* 1e-8)
	redd_x= redd_x[::-1]

	redd_inter_x = []


	for 	i in range(len(redd_x)):
		redd_inter_x.append(redd_x[i])
	redd_inter_x = np.array(redd_inter_x)


#	Define prevots function for the reddenin law redd_k
	
	def function_prevot(x, RV):
   		y=1.39*pow((pow(10.,-4.)*x),-1.2)-0.38 ;
   		return y 
	redd_k = function_prevot(redd_inter_x, RV)

# 	Making interpolation and extrapolation

	reddening = interp1d(redd_inter_x, redd_k, bounds_error=True)
	reddening2 = extrap1d(reddening)
	bbb_k = reddening2(redd_x)

	#converting back  wavelenght to freq
	redd_f_r= 2.998 * 1e10 / (redd_x * 1e-8)
	redd_f = redd_f_r[::-1]
	
	bbb_k= bbb_k[::-1]
	redd_f = np.log10(redd_f) 

	bbb_y_red = bbb_y * 10**(-0.4 * bbb_k * BBebv)	


	bbb_y_red_filtered =  filters(data_nu, bbb_nu, bbb_y_red, filterset, z,bands, files_dict, lambdas_dict, factors_dict)	

	bbb_y_red_filtered = bbb_y_red_filtered.reshape(np.shape(data_nu)) 

	# This output is a restframe monochromatic luminosity, to have the same output as Beta
	return bbb_y_red_filtered

#MUST REDDENING ALSO BE REDSHIFTED?


#==== GALAXY (BC03, solLum Angstrom-1) =====


def GALAXY(fn, GAebv, data_nu,  z, filterset, bands, files_dict, lambdas_dict, factors_dict):
	"""
	this function does XXX

	## inputs:
	- fn : file name for text file with YYY in it
	- GAebv : reddening from YYY

	## output:
	- returns foo; which we use as input to get_bar_blah()

	## comments:
	- equations from Hennawi et al 2004 http:/...

	## bugs:
	- does CCC very slowly.
	- ignores QQQ.
	- not sure that integral for RRR is correct.
	"""

	gal_wl_rest, gal_flux_la = np.loadtxt(fn, skiprows=2, usecols=(0,1),unpack= True)

	gal_nu =2.998 * 1.e10 / gal_wl_rest * 1.e8
	gal_nu_r = np.log10(gal_nu)

	#converting to nuFnu
	gal_Fnu_r= gal_flux_la * 3.34e-19 * gal_wl_rest**2.  
	
	# reverse
	gal_nu= gal_nu_r[::-1]
	gal_Fnu= gal_Fnu_r[::-1]

	gal_x = gal_nu #log
	gal_y = gal_Fnu
        
#---------------Reddening

	#in cm 
	
	RV = 4.05		
	wl = np.arange(0.122, 2.18, 0.02)
	redd_k=[]
	for 	i in range(len(wl)):
		if (wl[i]>0.12 and wl[i]<0.63):
			k =   2.659*(-2.156+(1.509/wl[i])-(0.198/(wl[i]**2))+(0.011/(wl[i]**3)) )+RV
		elif (wl[i]>0.63and wl[i]<2.2):
			k =  2.659*(-1.857+(1.040/wl[i]))+RV
		redd_k.append(k)
	
	micron2cm = 1e-4
	redd_k= np.array(redd_k)
	redd_wl = wl*micron2cm	

	redd_f_r= 2.998 * 1e10 / (redd_wl)
	redd_f_r = np.log10(redd_f_r) 

	redd_f = redd_f_r[::-1]
	redd_k= redd_k[::-1]
	# interpolate
	reddening = interp1d(redd_f, redd_k, bounds_error=True)
	reddening2 = extrap1d(reddening)
	redd_x = gal_x
	gal_k = reddening2(redd_x)

    	gal_Fnu_red = gal_y * 10**(-0.4 * gal_k * GAebv)

	gal_y_red_filtered =  filters(data_nu, gal_nu, gal_Fnu_red, filterset, z,bands, files_dict, lambdas_dict, factors_dict)	


	gal_y_red_filtered = gal_y_red_filtered.reshape(np.shape(data_nu)) 

	return gal_y_red_filtered



#==== TORUS (sed_data, solLum Angstrom-1) =====

def TORUS(str, data_nu, str_catalog, sourceline, z, filterset, bands, files_dict, lambdas_dict, factors_dict):


	distance = z2Dlum(z)

	tor_nu_rest, tor_nuLnu = np.loadtxt(str, skiprows=0, usecols=(0,1),unpack= True)
	tor_nu = np.log10(10**tor_nu_rest )
	tor_Lnu = tor_nuLnu / 10**(tor_nu)

	#Coverting to fluxes
	tor_Fnu = tor_Lnu /(4. * pi * distance**2.)# *(1+z)

#=================================
#FILTER

	tor_Fnu_filtered =  filters(data_nu, tor_nu, tor_Fnu, filterset, z,bands, files_dict, lambdas_dict, factors_dict)	
	tor_Fnu_filtered = tor_Fnu_filtered.reshape(np.shape(data_nu)) 

	return tor_Fnu_filtered


