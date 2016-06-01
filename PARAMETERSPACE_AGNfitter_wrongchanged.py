
"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        PARAMETERSPACE_AGNfitter.py

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

This script contains functions used by the MCMC machinery to explore the parameter space
of AGNfitter.

It contains:

* Initializing a point on the parameter space
*	Calculating the likelihood
*	Making the next step
*	Deciding when the burn-in is finished and start MCMC sampling

"""


from __future__ import division
import pylab as pl
import numpy as np
from math import exp,log,pi
from collections import defaultdict
from scipy.interpolate import interp1d
from scipy.integrate import simps, trapz, romberg
import time

from DATA_AGNfitter import DATA, NAME, DISTANCE, REDSHIFT
import MODEL_AGNfitter as model
from GENERAL_AGNfitter import adict, writetxt, loadobj




def Pdict (catalog, sourceline):

    """
	This function constructs a dictionary P with keys. The value of every key is a tuple with the
same length (the number of model parameters)

    name  : parameter names
    min   : minimum allowed parameter values
    max   : maximum allowed parameter values

	## inputs:
	- catalog file name
	- sourcelines

	## output:
	- dictionary P with all parameter characteristics

	## comments:
	- 

	## bugs:

    """
    P = adict()

    #Constrains on the age of the galaxy:
    z = REDSHIFT(catalog, sourceline)



    # ----------------------------|--------------|--------------|---------------|-----------|-----------|------------|-----------|------------|-------------|------------|
    P.names =  	'tau'	,     'age',	'nh',	'irlum' ,	 'SB',	 'BB',	 'GA',	 'TO',	'BBebv',   'GAebv'
    # -------------------------|-------------|-------------|---------------|-----------|-----------|------------|-----------|------------|-------------|------------| For F_nu
    P.min = 		 0 , 	 	6,	 21,		7, 		  0,  	  0,  	    0,		 0, 		 0, 		  0
    P.max = 		 3.5,	 	np.log10(model.maximal_age(z)),		25,		15, 		10,		   10, 	  10,	  	10,   	1,	  	 0.5
    # -------------------------|-------------|-------------|-----------|-----------|------------|-----------|------------|-------------|------------|



    Npar = len(P.names)
    #

    return P	


"""
CONSTRUCT THE MODEL
"""

float_formatter = lambda x: "%.2f" % x



def ymodel(data_nus, z, dict_modelsfiles, dict_modelfluxes, *par):

  """
	This function constructs the model from the parameter values

	## inputs:
	- v: frequency 
	- catalog file name
	- sourcelines

	## output:
	- dictionary P with all parameter characteristics

	## comments:
	- 

	## bugs:

  """

  all_tau, all_age, all_nh, all_irlum, filename_0_galaxy, filename_0_starburst, filename_0_torus = dict_modelsfiles
  STARBURSTFdict , BBBFdict, GALAXYFdict, TORUSFdict, EBVbbb_array, EBVgal_array= dict_modelfluxes


  # Call parameters from Emcee
  tau, agelog, nh, irlum, SB ,BB, GA,TO, BBebv, GAebv= par[0:10]
  age = 10**agelog


  # Pick templates for physical parameters
  SB_filename = model.pick_STARBURST_template(irlum, filename_0_starburst, all_irlum)
  GA_filename = model.pick_GALAXY_template(tau, age, filename_0_galaxy, all_tau, all_age)
  TOR_filename = model.pick_TORUS_template(nh, all_nh, filename_0_torus)
  BB_filename = 'models/BBB/richardsbbb.dat'

  EBV_bbb_0 = model.pick_EBV_grid(EBVbbb_array, BBebv)
  EBV_bbb = (  str(int(EBV_bbb_0)) if  float(EBV_bbb_0).is_integer() else str(EBV_bbb_0))
  EBV_gal_0 = model.pick_EBV_grid(EBVgal_array,GAebv)
  EBV_gal = (  str(int(EBV_gal_0)) if  float(EBV_gal_0).is_integer() else str(EBV_gal_0))


  try:	
        bands, gal_Fnu = GALAXYFdict[GA_filename, EBV_gal]
	bands, sb_Fnu= STARBURSTFdict[SB_filename] 
	bands, bbb_Fnu = BBBFdict[BB_filename, EBV_bbb]	
	bands, tor_Fnu= TORUSFdict[TOR_filename]
  except ValueError:
    print 'Error: Dictionary does not contain TORUS file:'+TOR_filename

  sb_Fnu *= 10**(SB)*1e-20#e50  
  bbb_Fnu *= 10**(BB)*1e-60#e90
  gal_Fnu *= 10**(GA)*1e-18
  tor_Fnu *=  10**(TO)*1e40


  # Sum components

  lum = sb_Fnu+ bbb_Fnu+ gal_Fnu + tor_Fnu

  lum = lum.reshape((np.size(lum),))	

  return lum



def ln_prior(dict_modelsfiles, dict_modelfluxes, z, P, pars):

  """
  Add priors on the parameters
  """

  for i,p in enumerate(pars):
    if not (P.min[i] < p < P.max[i]):
      return -np.inf

  # Bband expectations
  B_band_expected, B_band_thispoint = galaxy_Lumfct_prior(dict_modelsfiles, dict_modelfluxes, z, *pars )

  #if Bband magnitude in this trial is brighter than expected by the luminosity function, dont accept this one
  if B_band_thispoint < (B_band_expected - 5):#2.5):
      return -np.inf
  return 0.



def ln_likelihood(pars, x, y, ysigma, z, dict_modelsfiles, dict_modelfluxes):

  
    y_model = ymodel(x,z,dict_modelsfiles,dict_modelfluxes,*pars)

    #x_valid:
    #only frequencies with existing data (no detections nor limits F = -99)    
    #Consider only data free of IGM absorption. Lyz = 15.38 restframe    
    array = np.arange(len(x))
    ly_a = np.log10(10**(15.38)/(1+z))
    #x_valid = array[(x< ly_a) & (y>-99.)]

    resid = np.divide(np.subtract(y,y_model),ysigma)[x<ly_a]
    #resid = [(y[i] - y_model[i])/ysigma[i] for i in x_valid]
    return -0.5 * np.dot(resid, resid)


#POSTERIOR

def ln_probab(pars, x, y, ysigma, z, dict_modelsfiles, dict_modelfluxes, P):


  lnp = ln_prior(dict_modelsfiles, dict_modelfluxes, z, P, pars)

  if np.isfinite(lnp):	
    
    	
    posterior = lnp + ln_likelihood(pars, x,y, ysigma, z, dict_modelsfiles, dict_modelfluxes)

    return posterior
  return -np.inf



#============================================
#                                     INITIAL POSITIONS
#============================================



def get_initial_positions(nwalkers, P):

    # uniform distribution between parameter limits
    Npar = len(P.names)	
    p0 = np.random.uniform(size=(nwalkers, Npar))

    for i in range(Npar):

	p0[:, i] =  0.5*(P.max[i] + P.min[i]) + (2* p0[:, i] - 1) * (1)
	
    
    return p0


def get_initial_positions_PT(ntemps, nwalkers, P):

    # uniform distribution between parameter limits
    Npar = len(P.names) 
    p0 = np.random.uniform(size=(ntemps,nwalkers, Npar))

    for i in range(Npar):
      p0[:, :, i] =  0.5*(P.max[i] + P.min[i]) + (2* p0[:, :, i] - 1) * (0.00001)
  
    return p0

#============================================
#                                     BEST POSITIONS
#============================================



def get_best_position(filename, nwalkers, P):
  Npar = len(P.names)	
	#all saved vectors	
  samples = loadobj(filename)
	#index for the largest likelihood		
  i = samples['lnprob'].ravel().argmax()
	#the values for the parameters at this index
  P.ml= samples['chain'].reshape(-1, Npar)[i]

  p1 = np.random.normal(size=(nwalkers, Npar))

  for i in range(Npar):
    p = P.ml[i]
    
    p1[:, i] =  p + 0.00001 * p1[:, i]

  return p1


	
def get_best_position_PT(ntemps, filename, nwalkers, P):

  Npar = len(P.names) 
  #all saved vectors  
  samples = loadobj(filename)
  #index for the largest likelihood   
  i = samples['lnprob'].ravel().argmax()
  #the values for the parameters at this index
  P.ml= samples['chain'].reshape(-1, Npar)[i]

  p1 = np.random.normal(size=(ntemps,nwalkers, Npar)) 

  for i in range(Npar):
    p = P.ml[i]
    print i, P.names
    p1[:, :, i] =  p + 0.00001 * p1[:,:, i]

  return p1


def get_best_position_4mcmc(filename, nwalkers, P):
  Npar = len(P.names) 
  #all saved vectors  
  samples = loadobj(filename)
  #index for the largest likelihood   
  i = samples['lnprob'].ravel().argmax()
  #the values for the parameters at this index
  P.ml= samples['chain'].reshape(-1, Npar)[i]


  p1 = np.random.normal(size=(nwalkers, Npar))  
  for i in range(Npar):
    p = P.ml[i]

    p1[:, i] =  p + 0.00001 * p1[:, i]  

  return p1

def galaxy_Lumfct_prior(dict_modelsfiles, dict_modelfluxes, z, *par):

  # Calculated B-band at this parameter space point
  h_70 = 1.

  distance = model.z2Dlum(z)#/3.08567758e24

  lumfactor = (4. * pi * distance**2.)
  bands, gal_flux = galaxy_flux(dict_modelsfiles, dict_modelfluxes, *par)

#  array = np.arange(len(bands))
#  x_B = np.int(array[(14.87 > bands > 14.80)])
 
  flux_B = gal_flux[(14.87 > bands > 14.80)]

#  mag1= -2.5 * np.log10(flux_B) - 48.6
#  distmod = -5.0 * np.log10((distance/3.08567758e24 *1e6)/10) 
#  abs_mag1 = mag1 + distmod
#  thispoint1 = abs_mag1


  lum_B = lumfactor * flux_B
  thismag = 51.6 - 2.5 *np.log10(lum_B)
  

  # Expected B-band calculation
  
  expected = -20.3 - (5 * np.log10(h_70) )- (1.1 * z)


  return expected,thismag


def galaxy_flux(dict_modelsfiles, dict_modelfluxes, *par):

  all_tau, all_age, _, _, filename_0_galaxy, _, _ = dict_modelsfiles
  _, _, GALAXYFdict, _, _, EBVgal_array= dict_modelfluxes

  # calling parameters from Emcee
  tau, agelog, _, _, _ ,_, GA,_, _, GAebv= par[0:10]
 
  age = 10**agelog

  GA_filename = model.pick_GALAXY_template(tau, age, filename_0_galaxy, all_tau, all_age)
  EBV_gal_0 = model.pick_EBV_grid(EBVgal_array,GAebv)
  EBV_gal = (  str(int(EBV_gal_0)) if  float(EBV_gal_0).is_integer() else str(EBV_gal_0))


  try:
    bands, gal_Fnu = GALAXYFdict[GA_filename, EBV_gal]
  except ValueError:
    print 'Error: Dictionary does not contain key of ', GA_filename, EBV_gal, ' or the E(B-V) grid or the DICTIONARIES_AGNfitter file does not match when the one used in PARAMETERSPACE_AGNfitter/ymodel.py'
  
  gal_Fnu *= 1e-18 * 10**(GA)


  return bands, gal_Fnu
