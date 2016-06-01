

"""%%%%%%%%%%%%%%%%%

            DATA_AGNFitter.py

%%%%%%%%%%%%%%%%%%

This script contains all functions which are needed to construct the total model of AGN. 
The functions here translate the parameter space points into total fluxes dependin on the models chosen.

Functions contained here are the following:


DATA
NAME
DISTANCE
REDSHIFT

MOCKdata
MOCKerrors

"""

import numpy as np
import math
from math import exp,log,pi
import matplotlib.pyplot as plt
from numpy import random,argsort,sqrt
import re
import os
import time
from GENERAL_AGNfitter import NearestNeighbourSimple2D, NearestNeighbourSimple1D, extrap1d
from scipy import interpolate 
from collections import defaultdict
from scipy.interpolate import interp1d, UnivariateSpline
from numpy import array
from scipy.integrate import quad, trapz
from math import sqrt


"""
	DATA
=============

Function to read different formats of DATA catalogs

INPUT
	- Filename of the catalog (till now three options, Eduardos, Jonathans or Betas (XMMCOSMOS)
	- Line of the source beeing fitted
OUTPUT
	- frequency, Fluxes and fluxes errors
"""


def DATA (str, sourceline):

	"""
	This function constructs the data array.

	## inputs:
	- catalog file name
	- sourceline

	## output:
	- x : log observed frequency
	- y : Flux _nu
	- yerr: error on Flux_nu
	"""
	#=================================
	# BOOTES FIELD
	#=================================

	#if str == 'data/catalog_Bootes_hotcold_AGN.txt':
#	if str == 'data/catalog_WENDY_all_FIR.txt':
	if str == 'data/catalog_WENDY_FIR_missing.txt':
		#opening table	
		data = open(str, 'r') 
		#create lists (data_f -> frequency)
		data_nu = []
		data_flux = []
		data_fluxerr = []
		#ignoring header and reading line
	
		header = data.readline()

		for i in range(0, sourceline):
			header = data.readline()

		line = data.readline()

		#extract freq anf fluxes in all 19? 20? filters and make list
		column = line.strip().split()

		c = 2.997e8
		Angst2m= 1e-10
		z =  float(column[1])
		Dlum= z2Dlum(z)
		for i in range(18):



			nu_Angstrom=float(column[2+3*i])#observed

			nu = np.log10(c/ (Angst2m * nu_Angstrom)*(1+z))
			nu_exp = c/ (Angst2m * nu_Angstrom)
			
			flux_Jansky= float(column[3+3*i])
			flux= flux_Jansky * 1e-23				
			fluxerr = float(column[4+3*i])	

			if flux<0:
				if flux_Jansky > -99.:
					flux= 0.5 * (-1*flux_Jansky * 1e-23)
					fluxerr = flux

			if fluxerr != -99:
				fluxerr = fluxerr * 1e-23
			else: 
				fluxerr = flux * 0.1


	
			nu=float(nu)
			
			flux=float(flux)
			fluxerr=float(fluxerr)
			data_nu.append(nu)
			data_flux.append(flux)
			data_fluxerr.append(fluxerr)	

	#=================================
	# BOOTES FIELD
	#=================================

	#if str == 'data/catalog_Bootes_hotcold_AGN.txt':
#	elif str == 'data/catalog_DEEP_all_FIR.txt':
	elif str == 'data/catalog4AGNfitter_DEEP2.txt':	
		#opening table	
		data = open(str, 'r') 
		#create lists (data_f -> frequency)
		data_nu = []
		data_flux = []
		data_fluxerr = []
		#ignoring header and reading line
	
		header = data.readline()

		for i in range(0, sourceline):
			header = data.readline()

		line = data.readline()

		#extract freq anf fluxes in all 19? 20? filters and make list
		column = line.strip().split()

		c = 2.997e8
		Angst2m= 1e-10
		z =  float(column[1])
		Dlum= z2Dlum(z)
		for i in range(18):



			nu_Angstrom=float(column[2+3*i])#observed

			nu = np.log10(c/ (Angst2m * nu_Angstrom)*(1+z))
			nu_exp = c/ (Angst2m * nu_Angstrom)
			
			flux_Jansky= float(column[3+3*i])
			flux= flux_Jansky * 1e-23				
			fluxerr = float(column[4+3*i])	

			if flux<0:
				if flux_Jansky > -99.:
					flux= 0.5 * (-1*flux_Jansky * 1e-23)
					fluxerr = flux

			if fluxerr != -99:
				fluxerr = fluxerr * 1e-23
			else: 
				fluxerr = flux * 0.1


	
			nu=float(nu)
			
			flux=float(flux)
			fluxerr=float(fluxerr)
			data_nu.append(nu)
			data_flux.append(flux)
			data_fluxerr.append(fluxerr)	

	#=================================
	# FOR JONATHANS SAMPLE FORMAT
	#=================================

	elif str == 'data/jonathan_sample_right.txt':

		#opening table	
		data = open(str, 'r') 
		#create lists (data_f -> frequency)
		data_nu = []
		data_flux = []
		data_fluxerr = []
		#ignoring header and reading line
	
		header = data.readline()
		for i in range(0, sourceline):
			header = data.readline()

		line = data.readline()

		#extract freq anf fluxes in all 19? 20? filters and make list
		column = line.strip().split()

		c = 2.997e8
		Angst2m= 1e-10
		z =  float(column[1])
		Dlum= z2Dlum(z)
		for i in range(15): 	
		  
		    flag_l = float(column[9+5*i])
		    flag_e = float(column[11+5*i])	
		    flag_l = int(flag_l) 
		    flag_e = int(flag_e) 		
		    if flag_l ==1:	
			nu_Angstrom=float(column[8+5*i])#observed	
			print 'WAVELENGTH', nu_Angstrom
			nu = np.log10(c/ (Angst2m * nu_Angstrom))
			print 'NU', nu, 'REDSHIFT', z
			nu_exp = c/ (Angst2m * nu_Angstrom)
			lum= float(column[10+5*i])
			print 'LUMINOSITY:', lum
			flux= lum /(4. * pi * Dlum**2.) /nu_exp
			print 'FLUX', flux
			if flag_e == 1:
			        fluxerr = float(column[12+5*i]) / (4. * pi * Dlum**2.) /nu_exp
			else: 
				fluxerr = flux * 0.1
	    		if flag_l == 1:
	
				nu=float(nu)
				flux=float(flux)
				fluxerr=float(fluxerr)
				data_nu.append(nu)
				data_flux.append(flux)
				data_fluxerr.append(fluxerr)	

	#=================================
	# FOR EDUARDOS SAMPLE FORMAT
	#=================================
	elif str == 'data/eduardo_sample.txt':
	#opening table	
		data = open(str, 'r') 
		#create lists (data_f -> frequency)
		data_nu = []
		data_flux = []
		data_fluxerr = []
		#ignoring header and reading line
	
		header = data.readline()
		for i in range(0, sourceline):
			header = data.readline()

		line = data.readline()

		#extract freq anf fluxes in all 19? 20? filters and make list
		column = line.strip().split()

		c = 2.997e8
		Angst2m= 1e-10
		z =  float(column[1])
	
		Dlum= z2Dlum(z)
		for i in range(14): 	
	
		    flag_l = float(column[9+5*i])
		    flag_e = float(column[11+5*i])	
		    flag_l = int(flag_l) 
		    flag_e = int(flag_e) 		
		    if flag_l ==1:	
			nu_Angstrom=float(column[8+5*i])#observed	
			print 'WAVELENGTH', nu_Angstrom
			nu = np.log10(c/ (Angst2m * nu_Angstrom) )
			print 'NU', nu, np.log10(c/ (Angst2m * nu_Angstrom) ) ,'REDSHIFT', z
			nu_exp = c/ (Angst2m * nu_Angstrom)
			flux_Jansky= float(column[10+5*i])
			flux= flux_Jansky * 1e-21
			print 'FLUX', flux
			if flag_e == 1:
			        fluxerr = float(column[12+5*i]) * 1e-21
			else: 
				fluxerr = flux * 0.1
	    		if flag_l == 1:
		#	print i  	
				nu=float(nu)
				flux=float(flux)
				fluxerr=float(fluxerr)
				data_nu.append(nu)
				data_flux.append(flux)
				data_fluxerr.append(fluxerr)	


	#=================================
	# FOR MOCK
	#=================================
	elif str == 'data/MOCKagntype1.txt':
	#opening table	
		data = open(str, 'r') 
		#create lists (data_f -> frequency)
		data_nu = []
		data_flux = []
		data_fluxerr = []
		#ignoring header and reading line
	
		header = data.readline()
		for i in range(0, sourceline):
			header = data.readline()

		line = data.readline()

		#extract freq anf fluxes in all 19? 20? filters and make list
		column = line.strip().split()

		z =  float(column[1])

		for i in range(20): 	
	
	
			nu=np.log10(10**float(column[2+i])*(1+z))#restframe here. later back to obs	

			flux= float(column[22+i])

			fluxerr= float(column[42+i])

			data_nu.append(nu)
			data_flux.append(flux)
			data_fluxerr.append(fluxerr)	


	#=================================
	# FOR BETA'S XMM-COSMOS SAMPLE FORMAT
	#=================================
	elif str =='beta':
		#opening table	
		data = open(str, 'r') 
		#create lists (data_f -> frequency)
		data_nu = []
		data_flux = []
		data_fluxerr = []
		#ignoring header and reading line
	
		header = data.readline()
	
		for i in range(0, sourceline):
			header = data.readline()

		line = data.readline()

		#extract freq anf fluxes in all 19? 20? filters and make list
		column = line.strip().split()
		for i in range(20):
			flux= column[1+4*i]
			nu=column[2+4*i]	
			fluxerr = column[3+4*i]
			flag = column[4+4*i]
			flag = int(flag) 
		    	if flag == 0:
	
				nu=float(nu)
				flux=float(flux)
				fluxerr=float(fluxerr)
				data_nu.append(nu)
				data_flux.append(flux)
				data_fluxerr.append(fluxerr)
			elif flag == -1:
				nu=float(nu)
				flux1=0.5*float(flux)
				fluxerr=flux1
				data_nu.append(nu)
				data_flux.append(flux1)
				data_fluxerr.append(fluxerr)			

		name = column[0]
		

        #DISTANCE  in cm (to convert to lum)
		distance_Mpc = column[82]	
		distance_Mpc = float(distance_Mpc)
		distance = distance_Mpc * 3.08567758e24
	#print distance

	#REDSHIFT
		z=column[81]
		z=float(z)

	#------------------------------------------------
	
	else:	
		#opening table	
		data = open(str, 'r') 
		#create lists (data_f -> frequency)
		data_nu = []
		data_flux = []
		data_fluxerr = []
		#ignoring header and reading line
	
		header = data.readline()

		for i in range(0, sourceline):
			header = data.readline()

		line = data.readline()

		#extract freq anf fluxes in all 19? 20? filters and make list
		column = line.strip().split()

		c = 2.997e8
		Angst2m= 1e-10
		z =  float(column[1])
		Dlum= z2Dlum(z)
		for i in range(18):



			nu_Angstrom=float(column[2+3*i])#observed

			nu = np.log10(c/ (Angst2m * nu_Angstrom)*(1+z))
			nu_exp = c/ (Angst2m * nu_Angstrom)
			
			flux_Jansky= float(column[3+3*i])
			flux= flux_Jansky * 1e-23				
			fluxerr = float(column[4+3*i])	

			if flux<0:
				if flux_Jansky > -99.:
					flux= 0.5 * (-1*flux_Jansky * 1e-23)
					fluxerr = flux

			if fluxerr != -99:
				fluxerr = fluxerr * 1e-23
			else: 
				fluxerr = flux * 0.1


	
			nu=float(nu)
			
			flux=float(flux)
			fluxerr=float(fluxerr)
			data_nu.append(nu)
			data_flux.append(flux)
			data_fluxerr.append(fluxerr)	


	#NUMBER OF SOURCE

	#convert list2array
	data_nu = np.array(data_nu)
	data_Fnu = np.array(data_flux)
	data_Fnu_err = np.array(data_fluxerr)
	

	#Converting FLUX2LUMINOSiTY
	# Fluxes in erg s-1 cm-2 Hz-1

	sorted_indices = data_nu.argsort()	
	data_nu = data_nu[sorted_indices] 

	data_nu_obs = np.log10(10**(data_nu) / (1.+z))

	#L_nu*nu     (nu has to be in observed frame! therefore, nu_obs= nu_res/(1+z))

	data_Fnu = data_Fnu[sorted_indices] 

	data_Fnu_err = data_Fnu_err[sorted_indices]


	return data_nu_obs, data_Fnu, data_Fnu_err

	


def NAME (str, sourceline):

	"""
	function to read out/calculate the name of the source from the catalog
	INPUT:
		- filename of catalog
		- line of the source
	"""

	data = open(str, 'r') 
	#ignoring header and reading line	
	header = data.readline()
	
	for i in range(0, sourceline):
		header = data.readline()

	line = data.readline()
	column = line.strip().split()
	
	#NUMBER OF SOURCE
	name = column[0]
	return name

#=================================


def DISTANCE (str, sourceline):

	"""
	function to read out/calculate the distance of the source from the catalog
	INPUT:
		- filename of catalog
		- line of the source
	
	"""


	if str == 'data/jonathan_sample_right.txt':
	
		data = open(str, 'r') 		
		#ignoring header and reading line
		header = data.readline()
		for i in range(0, sourceline):
			header = data.readline()

		line = data.readline()
		column = line.strip().split()

		z =  float(column[1])
		distance= z2Dlum(z)

	elif str == 'data/eduardo_sample.txt':
		
		data = open(str, 'r') 		
		#ignoring header and reading line
		header = data.readline()
		for i in range(0, sourceline):
			header = data.readline()

		line = data.readline()
		column = line.strip().split()

		z =  float(column[1])
		distance= z2Dlum(z)


	else:
		
		data = open(str, 'r') 
		#ignoring header and reading line
		header = data.readline()
	
		for i in range(0, sourceline):
			header = data.readline()

		line = data.readline()
		column = line.strip().split()
	
        	#DISTANCE (to convert to lum)
		distance_Mpc = column[82]	
		distance_Mpc = float(distance_Mpc)
		distance = distance_Mpc * 3.08567758e24
		
	return distance

#===============================
#
#===============================

def REDSHIFT(str, sourceline):

	"""
	function to read out the redshift of the source from the catalog
	INPUT:
		- filename of catalog
		- line of the source
	"""


#	if str == 'data/catalog_Bootes_hotcold_AGN.txt':
#	if str == 'data/catalog_WENDY_all_FIR.txt':	
	if str == 'data/catalog_WENDY_FIR_missing.txt':
		data = open(str, 'r') 		
		#ignoring header and reading line
		header = data.readline()
		for i in range(0, sourceline):
			header = data.readline()

		line = data.readline()
		column = line.strip().split()

		z =  float(column[1])

	
	elif str == 'data/jonathan_sample_right.txt':
		
		data = open(str, 'r') 		
		#ignoring header and reading line
		header = data.readline()
		for i in range(0, sourceline):
			header = data.readline()

		line = data.readline()
		column = line.strip().split()

		z =  float(column[1])

	elif str == 'data/eduardo_sample.txt':
			
		data = open(str, 'r') 		
		#ignoring header and reading line
		header = data.readline()

		for i in range(0, sourceline):
			header = data.readline()

		line = data.readline()
		column = line.strip().split()

		z =  float(column[1])



#	if str == 'data/catalog_Bootes_hotcold_AGN.txt':
	elif str == 'data/catalog4AGNfitter_DEEP2.txt':#'data/catalog_DEEP_all_FIR.txt':	

		data = open(str, 'r') 		
		#ignoring header and reading line
		header = data.readline()
		for i in range(0, sourceline):
			header = data.readline()

		line = data.readline()
		column = line.strip().split()

		z =  float(column[1])

	
	elif str == 'data/jonathan_sample_right.txt':
		
		data = open(str, 'r') 		
		#ignoring header and reading line
		header = data.readline()
		for i in range(0, sourceline):
			header = data.readline()

		line = data.readline()
		column = line.strip().split()

		z =  float(column[1])

	elif str == 'data/eduardo_sample.txt':
			
		data = open(str, 'r') 		
		#ignoring header and reading line
		header = data.readline()

		for i in range(0, sourceline):
			header = data.readline()

		line = data.readline()
		column = line.strip().split()

		z =  float(column[1])




	elif str == 'data/MOCKagntype1.txt':

		str = '/data2/calistro/AGNfitter/data/MOCKagntype1.txt'
		data = open(str, 'r') 

		#ignoring header and reading line
	
		header = data.readline()
		for i in range(0, sourceline):
			header = data.readline()

		line = data.readline()

		column = line.strip().split()

		z =  float(column[1])
		
	elif str=='betas':
			
		data = open(str, 'r') 
		#ignoring header and reading line
		header = data.readline()
		for i in range(0, sourceline):
			header = data.readline()

		line = data.readline()
		column = line.strip().split()
	
		z=column[81]
		z=float(z)
	else:#'data/catalog_DEEP_all_FIR.txt':	

		data = open(str, 'r') 		
		#ignoring header and reading line
		header = data.readline()
		for i in range(0, sourceline):
			header = data.readline()

		line = data.readline()
		column = line.strip().split()

		z =  float(column[1])

	return z




#====================#
#MOCK ERRORS
#=========================

def MOCKerrors(str, sourceline):
	
	data = open(str, 'r') 
	#create lists (data_f -> frequency)
	data_nu = []
	data_flux = []
	data_fluxerr = []
	#ignoring header and reading line
	header = data.readline()
	
	for i in range(0, sourceline):
		header = data.readline()

	line = data.readline()

	#extract freq anf fluxes in all 19? 20? filters and make list
	column = line.strip().split()
	for i in range(20):
		flux= column[1+4*i]
		nu=column[2+4*i]	
		fluxerr = column[3+4*i]
		flag = column[4+4*i]
		flag = int(flag) 
	    	if flag == 0:
		
			nu=float(nu)
			flux=float(flux)
			fluxerr=float(fluxerr)
			data_nu.append(nu)
			data_flux.append(flux)
			data_fluxerr.append(fluxerr)
		elif flag == -1:
			nu=float(nu)
			flux1=0.5*float(flux)
			fluxerr=flux1
			data_nu.append(nu)
			data_flux.append(flux1)
			data_fluxerr.append(fluxerr)				

	#NUMBER OF SOURCE
	name = column[0]

	#REDSHIFT
	z=column[81]
	z=float(z)

	#convert list2array
	data_nu_rest = np.array(data_nu)
	
	errors_XMMCOSMOS_bands_orderoffile = np.array([0.093, 0.099, 0.092, 0.098, 0.095, 0.097, 0.093, 0.094, 0.0795, 0.080, 0.088, 0.100, 0.277, 0.105, 0.1290, 0.045, 0.065, 0.055, 0.369, 0.235 ])
	#Converting FLUX2LUMINOSiTY
	# Fluxes in erg s-1 cm-2 Hz-1

	sorted_indices = data_nu_rest.argsort()	

	data_nu_rest = data_nu_rest[sorted_indices]  
	data_nu_obs = np.log10(10**data_nu_rest /(1+z))

	errors_XMMCOSMOS_bands = errors_XMMCOSMOS_bands_orderoffile[sorted_indices] 


	return data_nu_obs, errors_XMMCOSMOS_bands







def z2Dlum(z):
	
	z = np.double(z)
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
      

