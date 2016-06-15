
"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        PARAMETERSPACE_AGNfitter.py

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

This script contains all  functions used by the MCMC machinery to explore the parameter space
of AGNfitter.

It contains:

* Initializing a point on the parameter space
* Calculating the likelihood
* Making the next step
* Deciding when the burn-in is finished and start MCMC sampling


"""
from __future__ import division
import pylab as pl
import numpy as np
from math import pi
import time
import pickle
import MODEL_AGNfitter2 as model



def Pdict (data):

    """
    Constructs a dictionary P with keys. The value of every key is a tuple with the
    same length (the number of model parameters)

    name  : parameter names
    min   : minimum value allowed for parameter 
    max   : maximum value allowed for parameter 

    ## inputs:
    - catalog file name
    - sourceline

    ## output:
    - dictionary P with all parameter characteristics

    """
    P = adict()

    # ----------------------------|--------------|--------------|---------------|-----------|-----------|------------|-----------|------------|-------------|------------|
    P.names =   r'$\tau$' ,     'age',  r'N$_{\rm H}$', 'irlum' ,  'SB',   'BB',   'GA',   'TO',  r'E(B-V)$_{bbb}$',    r'E(B-V)$_{gal}$'
    # -------------------------|-------------|-------------|---------------|-----------|-----------|------------|-----------|------------|-------------|------------| For F_nu
    P.min =      0 ,    6,   21,    7,      0,      0,        0,     0,      0,       0.
    P.max =      3.5,   np.log10(model.maximal_age(data.z)),   25,   15,     10,     1,    10,     10,     0.1,       1.5
    # -------------------------|-------------|-------------|-----------|-----------|------------|-----------|------------|-------------|------------|

    Npar = len(P.names)

    return P  



def ymodel(data_nus, z, dictkey_arrays, dict_modelfluxes, *par):
  """
  This function constructs the model from the parameter values

  ## inputs:
  -

  ## output:
  - the total model amplitude

  """
  STARBURSTFdict , BBBFdict, GALAXYFdict, TORUSFdict,_,_,_,_= dict_modelfluxes
  gal_do,  irlum_dict, nh_dict, BBebv_dict= dictkey_arrays

  # Call MCMC-parameter values 
  tau, agelog, nh, irlum, SB ,BB, GA,TO, BBebv, GAebv= par[0:10]
  age = 10**agelog

  # Pick dictionary key-values, nearest to the MCMC- parameter values
  irlum_dct = model.pick_STARBURST_template(irlum, irlum_dict)
  nh_dct = model.pick_TORUS_template(nh, nh_dict)
  ebvbbb_dct = model.pick_BBB_template(BBebv, BBebv_dict)

  gal_do.nearest_par2dict(tau, age, GAebv)
  tau_dct, age_dct, ebvg_dct=gal_do.t, gal_do.a,gal_do.e

  # Call fluxes from dictionary using keys-values
  try: 
    bands, gal_Fnu = GALAXYFdict[tau_dct, age_dct,ebvg_dct]   
    bands, sb_Fnu= STARBURSTFdict[irlum_dct] 
    bands, bbb_Fnu = BBBFdict[ebvbbb_dct] 
    bands, tor_Fnu= TORUSFdict[nh_dct]
 
  except ValueError:
    print 'Error: Dictionary does not contain some values'

  # Renormalize to have similar amplitudes. Keep these fixed!
    
  sb_Fnu_norm = sb_Fnu.squeeze()/ 1e20  
  bbb_Fnu_norm = bbb_Fnu.squeeze() / 1e60
  gal_Fnu_norm = gal_Fnu.squeeze() / 1e18
  tor_Fnu_norm = tor_Fnu.squeeze()/  1e-40


  # Total SED sum
  #---------------------------------------------------------------------------------------------------------------------------------------------------------#

  lum =    10**(SB)* sb_Fnu_norm      +     10**(BB)*bbb_Fnu_norm    +     10**(GA)*gal_Fnu_norm     +     (10**TO) *tor_Fnu_norm 

  #-----------------------------------------------------------------------------------------------------------------------------------------------------------

  lum = lum.reshape((np.size(lum),))  

  return lum



def ln_prior(dict_modelsfiles, dict_modelfluxes, z, P, pars):
  """
  This function constructs the model from the parameter values

  ## inputs:
  -

  ## output:
  - the total model amplitude

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



def ln_likelihood(pars, x, y, ysigma, z, dictkey_arrays, dict_modelfluxes):
    """
    This function constructs the model from the parameter values

    ## inputs:
    -

    ## output:
    - the total model amplitude

    """

    y_model = ymodel(x,z,dictkey_arrays,dict_modelfluxes,*pars)

    #x_valid:
    #only frequencies with existing data (no detections nor limits F = -99)    
    #Consider only data free of IGM absorption. Lyz = 15.38 restframe    
    array = np.arange(len(x))
    x_valid = array[(x< np.log10(10**(15.38)/(1+z))) & (y>-99.)]
  
    
    resid = [(y[i] - y_model[i])/ysigma[i] for i in x_valid]


    return -0.5 * np.dot(resid, resid)




def ln_probab(pars, x, y, ysigma, z, dictkey_arrays, dict_modelfluxes, P):

  """
  This function constructs the model from the parameter values

  ## inputs:
  -

  ## output:
  - the total model amplitude

  """

  lnp = ln_prior(dictkey_arrays, dict_modelfluxes, z, P, pars)

  if np.isfinite(lnp):  
    
    posterior = lnp + ln_likelihood(pars, x,y, ysigma, z, dictkey_arrays, dict_modelfluxes)   
    return posterior

  return -np.inf



"""--------------------------------------
Functions to obtain initial positions
--------------------------------------"""



def get_initial_positions(nwalkers, P):

    """
    This function constructs the model from the parameter values

    ## inputs:
    -

    ## output:
    - the total model amplitude

    """
    Npar = len(P.names) 
    p0 = np.random.uniform(size=(nwalkers, Npar))

    for i in range(Npar):

      p0[:, i] =  0.5*(P.max[i] + P.min[i]) + (2* p0[:, i] - 1) * (1)
  
    
    return p0


def get_best_position(filename, nwalkers, P):
  """
  This function constructs the model from the parameter values

  ## inputs:
  -

  ## output:
  - the total model amplitude

  """
  Npar = len(P.names) 
  #all saved vectors  
  f = open(filename, 'rb')
  samples = pickle.load(f)
  f.close()

  #index for the largest likelihood   
  i = samples['lnprob'].ravel().argmax()
  #the values for the parameters at this index
  P.ml= samples['chain'].reshape(-1, Npar)[i]

  p1 = np.random.normal(size=(nwalkers, Npar))

  for i in range(Npar):
    p = P.ml[i]
    
    p1[:, i] =  p + 0.00001 * p1[:, i]

  return p1




def get_best_position_4mcmc(filename, nwalkers, P):
  Npar = len(P.names) 

  f = open(filename, 'rb')
  samples = pickle.load(f)
  f.close()

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
  distance = model.z2Dlum(z)
  lumfactor = (4. * pi * distance**2.)

  bands, gal_flux = galaxy_flux(dict_modelsfiles, dict_modelfluxes, *par)


  bands = np.array(bands)
  flux_B = gal_flux[(14.790 < bands)&(bands < 14.870)]
  mag1= -2.5 * np.log10(flux_B) - 48.6
  distmod = -5.0 * np.log10((distance/3.08567758e24 *1e6)/10) 
  abs_mag1 = mag1 + distmod
  thispoint1 = abs_mag1


  lum_B = lumfactor * flux_B
  abs_mag = 51.6 - 2.5 *np.log10(lum_B)
  thispoint = abs_mag

  # Expected B-band calculation
  
  expected = -20.3 - (5 * np.log10(h_70) )- (1.1 * z)


  return expected,thispoint


def galaxy_flux(dictkey_arrays, dict_modelfluxes, *par):
  # call dictionary
  gal_do,  _,_,_= dictkey_arrays  
  STARBURSTFdict , BBBFdict, GALAXYFdict, TORUSFdict,_,_,_,_= dict_modelfluxes
  # calling parameters from Emcee
  tau, agelog, nh, irlum, SB ,BB, GA,TO, BBebv, GAebv= par[0:10]
  age = 10**agelog
  #nearest dict-key value to MCMC value
  gal_do.nearest_par2dict(tau, age, GAebv)
  tau_dct, age_dct, ebvg_dct=gal_do.t, gal_do.a,gal_do.e

  # Call fluxes from dictionary using keys-values
  bands, gal_Fnu = GALAXYFdict[tau_dct, age_dct,ebvg_dct] 

  gal_Fnu_norm = gal_Fnu / 1e18

  gal_flux = 10**(GA)*gal_Fnu_norm

  return bands, gal_flux



class adict(dict):

    """ A dictionary with attribute-style access. It maps attribute
    access to the real dictionary.
    This class has been obtained from the Barak package by Neil Chrighton)
    """

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    # the following two methods allow pickling
    def __getstate__(self):
        """Prepare a state of pickling."""
        return self.__dict__.items()

    def __setstate__(self, items):
        """ Unpickle. """
        for key, val in items:
            self.__dict__[key] = val

    def __setitem__(self, key, value):
        return super(adict, self).__setitem__(key, value)

    def __getitem__(self, name):
        return super(adict, self).__getitem__(name)

    def __delitem__(self, name):
        return super(adict, self).__delitem__(name)

    def __setattr__(self, key, value):
        if hasattr(self, key):
            # make sure existing methods are not overwritten by new
            # keys.
            return super(adict, self).__setattr__(key, value)
        else:
            return super(adict, self).__setitem__(key, value)

    __getattr__ = __getitem__

    def copy(self):
        """ Return a copy of the attribute dictionary.

        Does not perform a deep copy
        """
        return adict(self)

