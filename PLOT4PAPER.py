

"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      PLOT4PAPER.py

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

This script contains all  functions used to create the plots in paper number 1
"""

from __future__ import division
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
#matplotlib.use('Agg')

import sys, os, pdb, pprint
import pylab as pl
from math import exp,log,pi      
import numpy as np
import triangle
import time
import pickle
from scipy.interpolate import interp1d
from scipy.integrate import simps, trapz, romberg
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, DrawingArea, HPacker


scipy = True
try:
    from scipy.stats import gaussian_kde
    from scipy.spatial import Delaunay
    from scipy.optimize import minimize
except ImportError:
    scipy = False

import GENERAL_AGNfitter as general
import MODEL_AGNfitter as model
from DATA_AGNfitter import DATA, NAME, REDSHIFT
import DICTIONARIES_AGNfitter as dicts
import PARAMETERSPACE_AGNfitter as parspace



def general_plot1(filename, catalog, sourceline,  P, folder, opt, dict_modelsfiles, filterdict, path_AGNfitter, dict_modelfluxes):


    data_nus, ydata, ysigma = DATA(catalog, sourceline)
    z = REDSHIFT(catalog, sourceline)
    sourcename = NAME(catalog, sourceline)		
    path = os.path.abspath(__file__).rsplit('/', 1)[0]

    if not os.path.lexists(folder+str(sourcename)+'/samples_mcmc.sav'):
	print 'Error: The MCMC sampling has not been perfomed yet, or the chains were not saved properly.'

#==============================================

    samples = general.loadobj(filename)
    #nwalkers, nsamples, npar = samples['chain'].shape

    #ntemps, nwalkers, nsamples, npar = samples['chain'].shape
    nwalkers, nsamples, npar = samples['chain'].shape



    mean_accept =  samples['accept'].mean()
    print 'Mean acceptance fraction', mean_accept 
        

#======================================

    if filename.startswith(folder+str(sourcename)+'/samples_mcmc'):

        #Thinning
        Ns, Nt = opt['Nsample'], opt['Nthinning']

        assert Ns * Nt <= nsamples 

        chain_flat = samples['chain'][:,0:Ns*Nt:Nt,:].reshape(-1, npar)
                     #mix all walkers' steps into one chain
                     #of shape [(nwalkers*steps), parameters]
        lnprob = samples['lnprob'][:,0:Ns*Nt:Nt].ravel()
        isort = (-lnprob).argsort() #sort parameter vector for likelihood
        lnprob_sorted = np.reshape(lnprob[isort],(-1,1))
        lnprob_max = lnprob_sorted[0]
        chain_flat_sorted = chain_flat[isort]

        best_fit_par = chain_flat[isort[0]]
        total_length_chain = int(len(chain_flat))

        Nthin_compute = opt['realizations2compute'] #thinning chain to compute small
                                                #number of luminosities
        Nrealizations = opt['realizations2plot'] #thinning it even more
                                                 #for doing fewer superposed plots
	
        # calculate the model fluxes, mapping parameter space values to observables.

#=====================THE INPUT PARAMETERS=================================
    if catalog == 'data/MOCKagntype1.txt':
        
        mock_input, z_t1 = mock1.MOCKdata_chain()


    if catalog == 'data/MOCKagn.txt':
        
        mock_input, z_t1 = mock1.MOCKdata_chain()
        
    elif catalog == 'data/MOCKagntype2.txt':
        mock_input, z_t2 = mock2.MOCKdata_chain()
    
    

    all_nus, FLUXES4plotting, filtered_modelpoints, ifiltered_modelpoints = fluxes_arrays(data_nus, catalog, sourceline, dict_modelsfiles, filterdict, chain_flat_sorted, Nthin_compute,path_AGNfitter, dict_modelfluxes, mock_input)
    distance = model.z2Dlum(z)

    chain_best_sigma =[]

    for i in range(int(0.65*len(chain_flat_sorted))):	
        chain_best_sigma.append(chain_flat_sorted[i])	
    
    chain_best_sigma = np.array(chain_best_sigma)
	
#===================PLOT SEDS=======================


    plot_nr1 = PLOT_nr1(sourcename, catalog, data_nus, ydata, ysigma,  z, all_nus, FLUXES4plotting, filtered_modelpoints, ifiltered_modelpoints, Nrealizations, mock_input)


    print '************ =) ***********'



def fluxes_arrays(data_nus, catalog, sourceline, dict_modelsfiles, filterdict, chain, Nrealizations, path, dict_modelfluxes, mock_input):
    """
	This function constructs the luminosities arrays for many realizations from the parameter values

	## inputs:
	- v: frequency 
	- catalog file name
	- sourcelines

	## output:
	- dictionary P with all parameter characteristics
    """
 


#LIST OF OUTPUT

    SBFnu_list = []
    BBFnu_list = []
    GAFnu_list= []
    TOFnu_list = []
    TOTALFnu_list = []
    BBFnu_deredd_list = []
    filtered_modelpoints_list = []


    STARBURSTFdict , BBBFdict, GALAXYFdict, TORUSFdict, EBVbbb_array, EBVgal_array = dict_modelfluxes


    all_tau, all_age, all_nh, all_irlum, filename_0_galaxy, filename_0_starburst, filename_0_torus = dict_modelsfiles


    nsample, npar = chain.shape
    source = NAME(catalog, sourceline)

#CALL PARAMETERS FROM INPUT

    itau, iage, inh, iirlum, iSB ,iBB, iGA ,iTO, iBBebv, iGAebv= mock_input[sourceline] #calling parameter
    

#CALL PARAMETERS OF OUTPUT
    tau, agelog, nh, irlum, SB ,BB, GA,TO, BBebv0, GAebv0= [ chain[:,i] for i in range(npar)] #calling parameters


    
    z = REDSHIFT(catalog, sourceline)
    age = 10**agelog

    agelog = np.log10(age)	
    iagelog = np.log10(iage)

#LEFT PLOT

    for inp in range(1):
        

        SB_filename = path + model.pick_STARBURST_template(iirlum, filename_0_starburst, all_irlum)
        GA_filename = path + model.pick_GALAXY_template(itau, iage, filename_0_galaxy, all_tau, all_age)
        TO_filename = path + model.pick_TORUS_template(inh, all_nh, filename_0_torus)
        BB_filename = path + model.pick_BBB_template()

        all_model_nus = np.arange(12, 16, 0.001)#np.log10(dicts.stack_all_model_nus(filename_0_galaxy, filename_0_starburst, filename_0_torus, z, path ))  

        gal_nu, gal_nored_Fnu = model.GALAXY_read_4plotting( GA_filename, all_model_nus)
        gal_nu, gal_Fnu_red = model.GALAXY_nf2( gal_nu, gal_nored_Fnu, iGAebv )
        all_gal_nus, all_gal_Fnus =gal_nu, gal_Fnu_red

        sb_nu0, sb_Fnu0 = model.STARBURST_read_4plotting(SB_filename, all_model_nus)
        all_sb_nus, all_sb_Fnus = sb_nu0, sb_Fnu0

        bbb_nu, bbb_nored_Fnu = model.BBB_read_4plotting(BB_filename, all_model_nus)
        all_bbb_nus, all_bbb_Fnus = model.BBB_nf2(bbb_nu, bbb_nored_Fnu, iBBebv, z )
        all_bbb_nus, all_bbb_Fnus_deredd =all_bbb_nus, bbb_nored_Fnu

        tor_nu0, tor_Fnu0 = model.TORUS_read_4plotting(TO_filename, z, all_model_nus)
        all_tor_nus, all_tor_Fnus = tor_nu0, tor_Fnu0

        par_input = itau, iagelog, inh, iirlum, iSB ,iBB, iGA ,iTO, iBBebv, iGAebv


        ifiltered_modelpoints = parspace.ymodel(data_nus, z, dict_modelsfiles, dict_modelfluxes, *par_input)


        if len(all_gal_nus)==len(all_sb_nus) and len(all_sb_nus)==len(all_bbb_nus) and len(all_tor_nus)==len(all_bbb_nus) :
            nu= all_gal_nus

            all_sb_Fnus_norm = all_sb_Fnus /1e20
            all_bbb_Fnus_norm = all_bbb_Fnus / 1e60
            all_gal_Fnus_norm = all_gal_Fnus/ 1e18
            all_tor_Fnus_norm = all_tor_Fnus/  1e-40
            all_bbb_Fnus_deredd_norm = all_bbb_Fnus_deredd / 1e60


            iSBFnu =   all_sb_Fnus_norm *10**float(iSB) 
            iBBFnu =  all_bbb_Fnus_norm * 10**float(iBB) 
            iGAFnu =   all_gal_Fnus_norm * 10**float(iGA) 
            iTOFnu =   all_tor_Fnus_norm * 10**float(iTO)# /(1+z)
            iBBFnu_deredd = all_bbb_Fnus_deredd_norm* 10**float(iBB)

            iTOTALFnu =    iSBFnu + iBBFnu + iGAFnu + iTOFnu
         

#RIGHT PLOT
    			
    for gi in range(Nrealizations): #LOOP for a 100th part of the realizations

        g= gi*(nsample/Nrealizations)



        BBebv1 = model.pick_EBV_grid(EBVbbb_array, BBebv0[g])
        BBebv2 = (  str(int(BBebv1)) if  float(BBebv1).is_integer() else str(BBebv1))
        GAebv1 = model.pick_EBV_grid(EBVgal_array,GAebv0[g])
        GAebv2 = (  str(int(GAebv1)) if  float(GAebv1).is_integer() else str(GAebv1))
    

        SB_filename = path + model.pick_STARBURST_template(irlum[g], filename_0_starburst, all_irlum)
        GA_filename = path + model.pick_GALAXY_template(tau[g], age[g], filename_0_galaxy, all_tau, all_age)
        TO_filename = path + model.pick_TORUS_template(nh[g], all_nh, filename_0_torus)
        BB_filename = path + model.pick_BBB_template()


        all_model_nus = np.arange(12, 16, 0.001)#np.log10(dicts.stack_all_model_nus(filename_0_galaxy, filename_0_starburst, filename_0_torus, z, path ))  
        gal_nu, gal_nored_Fnu = model.GALAXY_read_4plotting( GA_filename, all_model_nus)
        gal_nu, gal_Fnu_red = model.GALAXY_nf2( gal_nu, gal_nored_Fnu, float(GAebv2))
        all_gal_nus, all_gal_Fnus =gal_nu, gal_Fnu_red


        sb_nu0, sb_Fnu0 = model.STARBURST_read_4plotting(SB_filename, all_model_nus)
        all_sb_nus, all_sb_Fnus = sb_nu0, sb_Fnu0

        bbb_nu, bbb_nored_Fnu = model.BBB_read_4plotting(BB_filename, all_model_nus)
        bbb_nu0, bbb_Fnu_red = model.BBB_nf2(bbb_nu, bbb_nored_Fnu, float(BBebv2), z )
        all_bbb_nus, all_bbb_Fnus = bbb_nu0, bbb_Fnu_red
        all_bbb_nus, all_bbb_Fnus_deredd = bbb_nu0, bbb_nored_Fnu

        tor_nu0, tor_Fnu0 = model.TORUS_read_4plotting(TO_filename, z, all_model_nus)
        all_tor_nus, all_tor_Fnus = tor_nu0, tor_Fnu0

        par1 = tau[g], agelog[g], nh[g], irlum[g], SB[g] ,BB[g], GA[g] ,TO[g], float(BBebv2), float(GAebv2)
        filtered_modelpoints = parspace.ymodel(data_nus, z, dict_modelsfiles, dict_modelfluxes, *par1)


        if len(all_gal_nus)==len(all_sb_nus) and len(all_sb_nus)==len(all_bbb_nus) and len(all_tor_nus)==len(all_bbb_nus) :
            nu= all_gal_nus

           
            all_sb_Fnus_norm = all_sb_Fnus /1e20
            all_bbb_Fnus_norm = all_bbb_Fnus / 1e60
            all_gal_Fnus_norm = all_gal_Fnus/ 1e18
            all_tor_Fnus_norm = all_tor_Fnus/  1e-40
            all_bbb_Fnus_deredd_norm = all_bbb_Fnus_deredd / 1e60


            SBFnu =   all_sb_Fnus_norm *10**float(SB[g]) 
            BBFnu =  all_bbb_Fnus_norm * 10**float(BB[g]) 
            GAFnu =   all_gal_Fnus_norm * 10**float(GA[g])
            TOFnu =   all_tor_Fnus_norm * 10**float(TO[g]) #/(1+z)
            BBFnu_deredd = all_bbb_Fnus_deredd_norm* 10**float(BB[g])

            TOTALFnu =    SBFnu + BBFnu + GAFnu + TOFnu
            SBFnu_list.append(SBFnu)
            BBFnu_list.append(BBFnu)
            GAFnu_list.append(GAFnu)
            TOFnu_list.append(TOFnu)
            TOTALFnu_list.append(TOTALFnu)
            BBFnu_deredd_list.append(BBFnu_deredd)
            filtered_modelpoints_list.append(filtered_modelpoints)


    	else: 
	    	print 'Error:'
	    	print 'The frequencies in the MODELdict_plot dictionaries are not equal for all models and could not be added.'
	    	print 'Check that the dictionary_plot.py stacks all frequencies (bands+galaxy+bbb+sb+torus) properly.'	


	       
    SBFnu_array = np.array(SBFnu_list)
    BBFnu_array = np.array(BBFnu_list)
    GAFnu_array = np.array(GAFnu_list)
    TOFnu_array = np.array(TOFnu_list)
    TOTALFnu_array = np.array(TOTALFnu_list)
    BBFnu_array_deredd = np.array(BBFnu_deredd_list)	
    filtered_modelpoints = np.array(filtered_modelpoints_list)
    

    FLUXES4plotting = (SBFnu_array, BBFnu_array, GAFnu_array, TOFnu_array, TOTALFnu_array,BBFnu_array_deredd, iSBFnu, iBBFnu, iGAFnu, iTOFnu, iTOTALFnu,iBBFnu_deredd)


    return all_model_nus, FLUXES4plotting, filtered_modelpoints, ifiltered_modelpoints



def PLOT_SED_bestfit(source, data_nus, data_flux, data_errors,  z, all_nus, FLUXES4plotting, filtered_modelpoints, ifiltered_modelpoints1):

    data_nus_obs = 10**data_nus
    data_nus_rest =10**data_nus*(1+z) #rest
    data_nus =np.log10(data_nus_rest)
    all_nus_rest = 10**all_nus
    all_nus_obs = 10**all_nus/(1+z) #observed 
    distance= model.z2Dlum(z)
    lumfactor = (4. * pi * distance**2.)
    data_nuLnu_rest = data_flux* data_nus_obs *lumfactor
    data_errors_rest= data_errors * data_nus_obs * lumfactor

    
    SBnuLnu, BBnuLnu, GAnuLnu, TOnuLnu, TOTALnuLnu, BBnuLnu_deredd, _, _, _, _,_, _  = [f[0] for f in FLUXES2nuLnu_4plotting(all_nus_rest, FLUXES4plotting, z)]
    _, _, _, _, _, _,iSBnuLnu, iBBnuLnu, iGAnuLnu, iTOnuLnu, iTOTALnuLnu,iBBnuLnu_deredd =FLUXES2nuLnu_4plotting(all_nus_rest, FLUXES4plotting, z)

    filtered_modelpoints_best = filtered_modelpoints * data_nus_obs * lumfactor
    ifiltered_modelpoints = ifiltered_modelpoints1* data_nus_obs * lumfactor
    return SBnuLnu, BBnuLnu, GAnuLnu, TOnuLnu, TOTALnuLnu, BBnuLnu_deredd,  filtered_modelpoints_best, iSBnuLnu, iBBnuLnu, iGAnuLnu, iTOnuLnu, iTOTALnuLnu,iBBnuLnu_deredd, ifiltered_modelpoints

def PLOT_SED_manyrealizations(source, data_nus, ydata, yerror, z, all_nus, FLUXES4plotting,  filtered_modelpoints1, Nrealizations):



	#data nuLnu
    data_nus_obs = 10**data_nus 
    data_nus_rest = 10**data_nus * (1+z) 
    data_nus = np.log10(data_nus_rest)
    all_nus_rest = 10**all_nus 
    all_nus_obs =  10**all_nus / (1+z) #observed
    distance= model.z2Dlum(z)
    lumfactor = (4. * pi * distance**2.)
    data_nuLnu_rest = ydata* data_nus_obs *lumfactor
    data_errors_rest=yerror * data_nus_obs * lumfactor

    SBnuLnu, BBnuLnu, GAnuLnu, TOnuLnu, TOTALnuLnu, BBnuLnu_deredd, _, _, _, _,_, _   = FLUXES2nuLnu_4plotting(all_nus_rest, FLUXES4plotting, z)
    #data_errors_rest= data_errors * data_nus_obs * lumfactor
    filtered_modelpoints = filtered_modelpoints1* data_nus_obs * lumfactor
    
    return SBnuLnu, BBnuLnu, GAnuLnu, TOnuLnu, TOTALnuLnu, BBnuLnu_deredd, filtered_modelpoints




def SED_colors(combination = 'a'):
    
    if combination=='a':   
        steelblue = '#4682b4'
        darkcyan ='#009acd'
        deepbluesky = '#008b8b'
        seagreen = '#2E8B57'	
        lila = '#68228B'
        darkblue='#123281'
        grey = '#D3D3D3'
 	
	return seagreen, darkblue, 'orange', lila, 'red', grey
	





def SED_plotting_settings(x, ydata):

    #fig = plt.figure()

    fig, (ax1, ax3) = plt.subplots(1,2,   sharey=True, figsize=(20,10))

#==============================================================================================
# FIRST PLOT 0,1  
#==============================================================================================
  


    #ax1 = fig.add_subplot(111)
    x2 = (2.98e8) / x / (1e-6) # Wavelenght axis
    ax2 = ax1.twiny()
    ax2.plot(x2, np.ones(len(x2)), alpha=0)

#-- Latex -------------------------------------------------
    rc('text', usetex=True)
    rc('font', family='serif')
    rc('axes', linewidth=1)
#-------------------------------------------------------------

#    ax1.set_title(r"\textbf{SED of Type 2}" + r"\textbf{ AGN }"+ "Source Nr. "+ source + "\n . \n . \n ." , fontsize=17, color='k')    
    ax1.set_xlabel(r'rest-frame frequency $\mathbf{log \  \nu} [\mathtt{Hz}] $', fontsize=18)
    ax2.set_xlabel(r'rest-frame wavelength $\mathbf{\lambda} [\mathtt{\mu m}] $', fontsize=18)
   # ax1.set_ylabel(r'luminosity $\mathbf{\nu L(\nu) [\mathtt{erg \ } \mathtt{ s}^{-1}]}$',fontsize=15)
    ax1.yaxis.tick_right()
  #  ax1.yaxis.set_label_position("right")

    ax1.set_autoscalex_on(True) 
    ax1.set_autoscaley_on(True) 
    ax1.set_xscale('linear')
    ax1.set_yscale('log')

    ax1.set_xlim([12,16])
    mediandata = np.median(ydata)
    ax1.set_ylim(mediandata /100.,mediandata * 50.)


    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_ylim( mediandata /100., mediandata * 50.)

    ax2.set_xlim([2.98e8 / (10**12) / (1e-6) , (2.98e8) / (10**16) / (1e-6)])
    ax2.set_xticks([100, 10,1, 0.1]) 
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())


#==============================================================================================
#   SECOND PLOT
#==============================================================================================

    
    x2 = (2.98e8) / x / (1e-6) # Wavelenght axis
    ax4 = ax3.twiny()
    ax4.plot(x2, np.ones(len(x2)), alpha=0)

#-- Latex -------------------------------------------------
    rc('text', usetex=True)
    rc('font', family='serif')
    
#-------------------------------------------------------------

#    ax1.set_title(r"\textbf{SED of Type 2}" + r"\textbf{ AGN }"+ "Source Nr. "+ source + "\n . \n . \n ." , fontsize=17, color='k')    
    ax3.set_xlabel(r'rest-frame frequency $\mathbf{log \  \nu} [\mathtt{Hz}] $', fontsize=18)
    ax4.set_xlabel(r'rest-frame wavelength $\mathbf{\lambda} [\mathtt{\mu m}] $', fontsize=18)
    ax3.set_ylabel(r'luminosity $\mathbf{\nu L(\nu) [\mathtt{erg \ } \mathtt{ s}^{-1}]}$',fontsize=18)
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")


    ax3.set_autoscalex_on(True) 
    ax3.set_autoscaley_on(True) 
    ax3.set_xscale('linear')
    ax3.set_yscale('log')

    ax3.set_xlim([12,16])
    ax3.set_xticks(np.arange(12.5, 16, 0.5)) 
    
    mediandata = np.median(ydata)
    ax3.set_ylim(mediandata /100.,mediandata * 50.)

    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_ylim( mediandata /100., mediandata * 50.)

    ax4.set_xlim([2.98e8 / (10**12) / (1e-6) , (2.98e8) / (10**16) / (1e-6)])
    ax4.set_xticks([100, 10,1, 0.1]) 
    ax4.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())



    


    
   
#==============================================================================================
#   THIRD PLOT
#==============================================================================================



    return fig, ax1, ax2, ax3, ax4


def PLOT_nr1(source, catalog, data_nus, data_flux, data_errors,  z, all_nus, FLUXES4plotting, filtered_modelpoints, ifiltered_modelpoints, Nrealizations, mock_input):

    # Two subplots, unpack the output array immediately


    
    data_nus_obs = 10**data_nus 
    data_nus_rest = 10**data_nus * (1+z) 
    data_nus_rest_log = np.log10(data_nus_rest)

    all_nus_rest = 10**all_nus 
    all_nus_obs =  10**all_nus / (1+z) #observed
    distance= model.z2Dlum(z)
    lumfactor = (4. * pi * distance**2.)
    data_nuLnu_rest = data_flux* data_nus_obs *lumfactor
    data_errors_rest=data_errors * data_nus_obs * lumfactor

    fig, ax1, ax2, ax3, ax4 = SED_plotting_settings(all_nus_rest, data_nuLnu_rest)
    plt.subplots_adjust(top=0.6, bottom=0, wspace=0)

#=================================
# INPUT AND BEST OUTPUT SED
#=================================

    SBnuLnu, BBnuLnu, GAnuLnu, TOnuLnu, TOTALnuLnu, BBnuLnu_deredd, filtered_modelpoints_best, iSBnuLnu, iBBnuLnu, iGAnuLnu, iTOnuLnu, iTOTALnuLnu,iBBnuLnu_deredd, ifiltered_modelpoints_lum = PLOT_SED_bestfit(source, data_nus, data_flux, data_errors,  z, all_nus, FLUXES4plotting, filtered_modelpoints, ifiltered_modelpoints)




    SBcolor, BBcolor, GAcolor, TOcolor, TOTALcolor, grey= SED_colors(combination = 'a')
    lw =2
 

    ip1= ax1.plot( all_nus, iTOTALnuLnu, marker="None", linewidth=lw, linestyle='--', label="1 /sigma", color= TOTALcolor, alpha= 1.0)
    ip2=ax1.plot(all_nus, iSBnuLnu, marker="None", linewidth=lw, linestyle='--',label="1 /sigma", color= SBcolor, alpha = 1)
    ip3=ax1.plot(all_nus, iBBnuLnu, marker="None", linewidth=lw, linestyle='--',label="1 /sigma",color= BBcolor, alpha = 1)
    ip4=ax1.plot( all_nus, iGAnuLnu,marker="None", linewidth=lw, linestyle='--',label="1 /sigma",color=GAcolor, alpha = 1)
    ip5=ax1.plot( all_nus, iTOnuLnu, marker="None",  linewidth=lw, linestyle='--',label="1 /sigma",color= TOcolor, alpha = 1)
    interp_total= interp1d(all_nus, iTOTALnuLnu, bounds_error=False, fill_value=0.)
    iTOTALnuLnu_at_datapoints = interp_total(data_nus)


    #plot data points. These are read out from PLOR_SED_bestfit
    #ip6 = ax1.plot(data_nus_rest_log, ifiltered_modelpoints_lum ,  marker='*', linestyle="None", markersize=5, color="red", alpha = 1)
    

    p1= ax1.plot( all_nus, TOTALnuLnu, marker="None", linewidth=lw,  label="1 /sigma", color= TOTALcolor, alpha= 1.0)
    p2=ax1.plot(all_nus, SBnuLnu, marker="None", linewidth=lw, label="1 /sigma", color= SBcolor, alpha = 1)
    p3=ax1.plot(all_nus, BBnuLnu, marker="None", linewidth=lw, label="1 /sigma",color= BBcolor, alpha = 1)
    p4=ax1.plot( all_nus, GAnuLnu,marker="None", linewidth=lw, label="1 /sigma",color=GAcolor, alpha = 1)
    p5=ax1.plot( all_nus, TOnuLnu, marker="None",  linewidth=lw, label="1 /sigma",color= TOcolor, alpha = 1)
    interp_total= interp1d(all_nus, TOTALnuLnu, bounds_error=False, fill_value=0.)
    TOTALnuLnu_at_datapoints = interp_total(data_nus)
    p6 = ax1.plot(data_nus_rest_log, filtered_modelpoints_best[0] ,  marker='o', linestyle="None", markersize=5, color="red", alpha = 1)
    

    (_, caps, _) = ax1.errorbar(data_nus_rest_log, data_nuLnu_rest, yerr= data_errors_rest, capsize=4, linestyle="None", linewidth=lw,  marker='o',markersize=5, color="black")

    

#=================================
# MANY REALIZATIONS SED
#=================================
    

    SBnuLnu, BBnuLnu, GAnuLnu, TOnuLnu, TOTALnuLnu, BBnuLnu_deredd, filtered_modelpoints = PLOT_SED_manyrealizations(source, data_nus, data_flux, data_errors, z, all_nus, FLUXES4plotting,  filtered_modelpoints, Nrealizations)

    thinning_4plot = len(TOTALnuLnu) / (Nrealizations)

    for j in range(Nrealizations):

        
        i = j * 10 
        #Settings for model lines
        q2=ax3.plot(all_nus, SBnuLnu[i], marker="None", linewidth=lw, label="1 /sigma", color= SBcolor, alpha = 0.4)
        q3=ax3.plot(all_nus, BBnuLnu[i], marker="None", linewidth=lw, label="1 /sigma",color= BBcolor, alpha = 0.4)
        q4=ax3.plot( all_nus, GAnuLnu[i],marker="None", linewidth=lw, label="1 /sigma",color=GAcolor, alpha = 0.4)
        q5=ax3.plot( all_nus, TOnuLnu[i], marker="None",  linewidth=lw, label="1 /sigma",color= TOcolor ,alpha = 0.4)
        q1= ax3.plot( all_nus, TOTALnuLnu[i], marker="None", linewidth=lw,  label="1 /sigma", color= TOTALcolor, alpha= 0.4)

        interp_total= interp1d(all_nus, TOTALnuLnu[i], bounds_error=False, fill_value=0.)

        TOTALnuLnu_at_datapoints = interp_total(data_nus)


        (_, caps, _) = ax3.errorbar(data_nus_rest_log, data_nuLnu_rest, yerr= data_errors_rest, capsize=4, linestyle="None", linewidth=1.5,  marker='.',markersize=5, color="black", alpha = 0.4)
        

        q6 = ax3.plot(data_nus_rest_log, filtered_modelpoints[i] ,  marker='o', linestyle="None",markersize=5, color="red")

    ax1.annotate(r'\hspace{0.1cm} \\z ='+ str(z)+ r' \\ input = dashed lines \\ output maximum likelihood = solid line ' , xy=(0, 0.5),  xycoords='axes points', xytext=(20, 380), textcoords='axes points', fontsize=16 )
    ax3.annotate(r'output realizations from PDF' , xy=(0, 0.1),  xycoords='axes points', xytext=(20, 380), textcoords='axes points', fontsize=16 )

    print ' => SEDs of '+ str(Nrealizations)+' different realization were plotted.'


#=================================
# TABLE
#=================================

  
    itau, iage, inh, iirlum, iSB ,iBB, iGA, iTO, iBBebv, iGAebv= mock_input[np.float(source)]

    tau_e, age_e, nh_e, irlum_e, SB_e, BB_e, GA_e, TO_e, BBebv_e, GAebv_e, Mstar_e, SFR_e, SFR_file_e, Lfir_e, Lir_e, Lbol_e, Lbol_deredd_e  = np.loadtxt('/data2/calistro/AGNfitter/OUTPUT/'+str(source)+'/parameters_with_errors_'+str(source)+'.txt', usecols=(0,1,2,3,4,5,6,7,8,9,10,11, 12, 13,14,15, 16), unpack=True , dtype = ('S')) 
    iage = '%.3s'%np.log10(np.float(iage))
    sourceline2_0 = [tau_e[0], age_e[0], nh_e[0], irlum_e[0], SB_e[0], BB_e[0], GA_e[0], TO_e[0], BBebv_e[0], GAebv_e[0]]
    sourceline2_1 = [tau_e[1], age_e[1], nh_e[1], irlum_e[1], SB_e[1], BB_e[1], GA_e[1], TO_e[1], BBebv_e[1], GAebv_e[1]]
    sourceline2_2 = [tau_e[2], age_e[2], nh_e[2], irlum_e[2], SB_e[2], BB_e[2], GA_e[2], TO_e[2], BBebv_e[2], GAebv_e[2]]
    outputline_0 = ['%.4s' %i for i in  sourceline2_0]
    outputline_1 = ['%.4s' %i for i in  sourceline2_1]
    outputline_2 = ['%.4s' %i for i in  sourceline2_2]
    outputline = [r'$%.4s ^{%.4s} _{%.4s} $' %(outputline_0[i], outputline_1[i], outputline_2[i]) for i in range(len(sourceline2_0))]


    rows = [r'$\tau$', 'age', r'N$_h$', r'lum$_{IR}$', 'SB', 'BB', 'GA', 'TO', r'E(B-V)$_{bbb}$', r'E(B-V)$_{gal}$']
    columns = [r'input',  r'output']

    tablearray = np.array([(itau, iage, inh, iirlum, iSB ,iBB, iGA,iTO, iBBebv, iGAebv), outputline ])
    tablearray_transpose = tablearray.T
    

    the_table = plt.table(cellText= tablearray_transpose,
                        colWidths=[1,1],
                      rowLabels=rows,
                      colLabels=columns, loc ='center',  bbox=[-1.25, 0, 0.25, 1]  ,cellLoc='center')
    #Changing heights
    table_props = the_table.properties()
    table_cells = table_props['child_artists']
    for cell in table_cells: cell.set_height(0.1)
    the_table.set_fontsize(18)





    
# Adjust layout to make room for the table:
  
    if catalog == 'data/MOCKagntype1.txt':
        print '/data2/calistro/AGNfitter/OUTPUT/MOCKagn1/'+str(source)+'/bigplot.pdf'
        plt.savefig('/data2/calistro/AGNfitter/OUTPUT/'+str(source)+'/bigplot_'+str(source)+'.pdf',bbox_inches='tight')
        plt.close(fig)

    if catalog == 'data/MOCKagn.txt':
        print '/Users/Gabriela/Codes/AGNfitter/OUTPUT/MOCK/'+str(source)+'/bigplot.pdf'
        plt.savefig('/data2/calistro/AGNfitter/OUTPUT/'+str(source)+'/bigplot_'+str(source)+'.pdf',bbox_inches='tight')
        plt.close(fig)


    elif catalog == 'data/MOCKagntype2.txt':
        print '/data2/calistro/AGNfitter/OUTPUT/'+str(source)+'/bigplot.pdf'
        plt.savefig('/data2/calistro/AGNfitter/OUTPUT/'+str(source)+'/bigplot_'+str(source)+'.pdf',bbox_inches='tight')
        plt.close(fig)

def FLUXES2nuLnu_4plotting(all_nus_rest, FLUXES4plotting, z):

    all_nus_rest = all_nus_rest 
    all_nus_obs = all_nus_rest /(1+z) #observed
    distance= model.z2Dlum(z)
    lumfactor = (4. * pi * distance**2.)

    SBnuLnu, BBnuLnu, GAnuLnu, TOnuLnu, TOTALnuLnu, BBnuLnu_deredd, iSBnuLnu, iBBnuLnu, iGAnuLnu, iTOnuLnu, iTOTALnuLnu,iBBnuLnu_deredd  = [ f *lumfactor*all_nus_obs for f in FLUXES4plotting]

    return SBnuLnu, BBnuLnu, GAnuLnu, TOnuLnu, TOTALnuLnu, BBnuLnu_deredd,  iSBnuLnu, iBBnuLnu, iGAnuLnu, iTOnuLnu, iTOTALnuLnu,iBBnuLnu_deredd  



