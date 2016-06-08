

"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      PLOTandWRITE_AGNfitter.py

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

This script contains all  functions used in order to visualize the output of the sampling. Plotting and writing. This function need all to have the samples_mcmc.sav files, which are produced by the MCMC_AGNfitter module.
    
"""

import matplotlib.pyplot as plt
import matplotlib 
#matplotlib.use('Agg')
import sys, os
import math 
import numpy as np
import triangle
import time
import pickle
import scipy
from astropy import units as u

import GENERAL_AGNfitter as general #STILL NEED TO GET RID OF THIS
import MODEL_AGNfitter as model
import DICTIONARIES_AGNfitter as dicts
import PARAMETERSPACE_AGNfitter as parspace



def main(fileending, data, P, out):

    """
    input: 
    bugs:
        - Lyman alpha beeing broken here

    """

    data_nus = data.nus
    ydata = data.fluxes
    ysigma= data.fluxerrs
    
    folder = data.output_folder
    sourceline = data.sourceline
    catalog = data.catalog
    sourcename = data.name
    dict_modelsfiles =data.dict_modelsfiles
    filterdict=data.filterdict
    dict_modelfluxes= data.dict_modelfluxes
    z = data.z 

    path_AGNfitter = data.path
    filename = data.output_folder+str(data.name)+ fileending


 
    array = np.arange(len(data_nus))
    #below Ly-alpha and non detections
    index_dataexist = array[(data_nus< np.log10(10**(15.38)/(1+z))) & (ydata>-99.)]


    path = os.path.abspath(__file__).rsplit('/', 1)[0]
    if not os.path.lexists(folder+str(sourcename)+'/samples_mcmc.sav'):
        print 'Error: The MCMC sampling has not been perfomed yet, or the chains were not saved properly.'

#==============================================
    f = open(folder+str(sourcename)+'/samples_mcmc.sav', 'rb')
    samples = pickle.load(f)
    f.close()

    nwalkers, nsamples, npar = samples['chain'].shape
    mean_accept =  samples['accept'].mean()
    print 'Mean acceptance fraction', mean_accept
    mean_autocorr = samples['acor'].mean()
    print 'Mean autocorrelation time', mean_autocorr
#===================MCMC===================== 

    if out['plot_tracesburn-in'] :   

      if filename.startswith(folder+str(sourcename)+'/samples_burn1-2-3'):
        if out['plot_tracesburn-in'] :    
            print 'Plotting traces of burn-in'
            fig, nwplot = plot_trace_burnin123(P, samples['chain'], samples['lnprob'])
            fig.suptitle('Chain traces for %i steps of %i walkers' % (nwplot,nwalkers))
            fig.savefig(folder+str(sourcename)+'/traces1-2-3.' +  out['plotformat'])
            plt.close(fig)  
      else: 
        print 'Burn-in phase has not finished or not saved in samples_burn1-2-3.sav '

#===================MCMC===================== 

    if filename.startswith(data.output_folder+str(data.name)+'/samples_mcmc'):

        #Thinning
        Ns, Nt = out['Nsample'], out['Nthinning']

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

        Nthin_compute = out['realizations2int'] #thinning chain to compute small
                                                #number of luminosities
        Nrealizations = out['realizations2plot'] #thinning it even more
                                                 #for doing fewer superposed plots
    

        all_nus, FLUXES4plotting, filtered_modelpoints = fluxes_arrays(data, dict_modelsfiles, filterdict, chain_flat_sorted, Nthin_compute, dict_modelfluxes)
         
#===================WRITING====================


        if out['writepar_maxlikelihood']:
            print 'Printing parameters of maximum likelihood'
            distance = model.z2Dlum(z)
            Mstar, SFR, SFR_file = model.stellar_info_best(best_fit_par, catalog, sourceline)
            output = np.hstack((best_fit_par, np.log10(Mstar), SFR, SFR_file, lnprob_max))
            output_names = np.array(['tau ','age',' nhv',' irlum ','SB',' BB ','GA',' TO ','BBebv ','GAebv ','Mstar ','SFR ','SFR_file','ln_likelihood'])                                           
            np.savetxt(folder + str(sourcename)+'/max_likelihood_parameters_'+ str(sourcename) + '.txt' , np.column_stack((output, output_names)), delimiter = " ",fmt="%s" ,header="Maximum likelihood parameters (Chi2Minimization)")


    #=================PLOT_TRACES===================

        if out['plot_tracesmcmc'] : 

            print 'Plotting traces of mcmc'
            fig, nwplot = plot_trace_mcmc(P, samples['chain'],samples['lnprob'])
            fig.suptitle('Chain traces for %i of %i walkers. Main acceptance fraction: %f' % (nwplot,nwalkers,mean_accept))
            fig.savefig(folder+str(sourcename)+'/traces_mcmc.' + out['plotformat'])
            plt.close(fig)

    #===============INTEGRATE LUMINOSITIES=================


        if out['calc_intlum']:
            print 'Computing integrated luminosities Loptuv_GA, LMir_TO, LMir_SB, Loptuv_BB, LFir_8-1000'
            #print 'Saving in: ', folder + str(sourcename)+'/integrated_luminosities_'+str(sourcename)+'.txt'
            L0, L1, L2, L3, L4 , L5=integrated_luminosities_arrays(all_nus, FLUXES4plotting, Nthin_compute, total_length_chain, z, folder, sourcename)
            #Loptuv_BB, Loptuv_GA, LMir_TO, LMir_SB, Loptuv_BB, LFir_8-1000

    #==================PLOT PDF TRIANGLE=================

        if out['plot_posteriortriangle']:
            print 'Plotting triangle of PDFs of parameters.'
            figure = plot_posteriors_triangle_pars(chain_flat, best_fit_par)
            figure.savefig(folder+str(sourcename)+'/posterior_triangle_pars_'+str(sourcename)+'.' + out['plotformat'])
            plt.close(figure)

    ### change this out to out['posteriortriangleofluminosities]
        if out['plot_posteriortrianglewithluminosities'] :

            if not os.path.lexists(folder+str(sourcename)+'/integrated_luminosities_'+str(sourcename)+'.txt'):
                L0, L1, L2, L3, L4, L5 =integrated_luminosities_arrays(all_nus, FLUXES4plotting, Nthin_compute, total_length_chain ,z, folder, sourcename)
                print 'Plotting triangle of PDFs of derived quantities.'
            L0, L1, L2, L3, L4, L5 = np.loadtxt(folder + str(sourcename)+'/integrated_luminosities_'+str(sourcename)+'.txt', usecols=(0,1,2,3,4,5),unpack= True)
            chain_others = np.column_stack((L0, L1, L2, L3, L4, L5))
            chain_luminosities = np.column_stack((L0, L1, L2, L3, L4, L5))  

            figure2= plot_posteriors_triangle_other_quantities(chain_others)
            figure2.savefig(folder+str(sourcename)+'/posterior_triangle_others_'+str(sourcename)+'.'+ out['plotformat'])
            plt.close(figure2)   


    #-==========PRINT PARAMETERS WITH UNCERTAINTIES=============

        if out['writepar_meanwitherrors']:

            if not out['calc_intlum']:
                L0, L1, L2, L3, L4, L5 =integrated_luminosities_arrays(all_nus, FLUXES4plotting, Nthin_compute, total_length_chain ,z, folder, sourcename)
                    
            print 'Printing mean values of the parameters with corresponding errors.'

            Mstar, SFR, SFR_file = stellar_info_array(chain_flat_sorted, catalog, sourceline, Nthin_compute)
            
            SFR_IR = model.sfr_IR(L5)
            
            chain_pars_and_others = np.column_stack((chain_flat_sorted, Mstar, SFR, SFR_file, SFR_IR, L0, L1, L2, L3, L4, L5 ))
            tau_e, age_e, nh_e, irlum_e, SB_e, BB_e, GA_e, TO_e, BBebv_e, GAebv_e,Mstar_e, SFR_e, SFR_file_e, SFR_IR_e, L0_e, L1_e, L2_e, L3_e, L4_e, L5_e= parameters_with_errors(chain_pars_and_others)
            parameters_with_errors_transpose= np.transpose(parameters_with_errors(chain_pars_and_others))

            output_error = np.column_stack((tau_e, age_e, nh_e, irlum_e, SB_e, BB_e, GA_e, TO_e, BBebv_e, GAebv_e, Mstar_e, SFR_e, SFR_file_e, L0_e, L1_e, L2_e, L3_e, L4_e, L5_e, SFR_IR_e)) 
            np.savetxt(folder + str(sourcename)+'/parameters_with_errors_3_'+str(sourcename)+'.txt' , output_error, delimiter = " ",fmt="%1.4f" ,header="tau_e, age_e, nh_e, irlum_e, SB_e, BB_e, GA_e, TO_e, BBebv_e, GAebv_e, Mstar_e, SFR_e, SFR_file_e, L_sb, L_bbb, L_ga, L_to, Lbb_dered, Lfir_SFR ,SFR_IR_e")


        if out['writepar_maxlikelihood']:
            print 'Printing parameters of maximum likelihood'
            distance = model.z2Dlum(z)
            Mstar, SFR, SFR_file = model.stellar_info_best(best_fit_par, catalog, sourceline)
            L0_b = L0[0]
            L1_b = L1[0]
            L2_b= L2[0]
            L3_b= L3[0]
            L4_b = L4[0]
            L5_b = L5[0]
            L5_reshaped=np.reshape(L5[0],(-1,1)) 
            
            SFR_IR =np.reshape((model.sfr_IR(L5_reshaped)), (1))

        
            output = np.hstack((best_fit_par, np.log10(Mstar), SFR, SFR_file , lnprob_max, L0_b, L1_b, L2_b, L3_b, L4_b, L5_b ,SFR_IR))
            output_names = np.array(['tau ','age',' nhv',' irlum ','SB',' BB ','GA',' TO ','BBebv ','GAebv ','Mstar ','SFR ','SFR_file','ln_likelihood','L0_b', 'L1_b', 'L2_b', 'L3_b', 'L4_b', 'L5_b','SFR_IR'])                                           
            np.savetxt(folder + str(sourcename)+'/max_likelihood_parameters_2_'+ str(sourcename) + '.txt' , np.column_stack((output, output_names)), delimiter = " ",fmt="%s" ,header="Maximum likelihood parameters (Chi2Minimization)")

        #===================PLOT SEDS=======================

        if out['plotSEDbest']:

            fig = PLOT_SED_bestfit(data, all_nus, FLUXES4plotting, filtered_modelpoints, index_dataexist)
            plt.savefig(folder+str(sourcename)+'/SED_best_'+str(sourcename)+'.pdf', format = 'pdf')
            plt.close(fig)

        if out['plotSEDrealizations']:

            fig = PLOT_SED_manyrealizations(data, all_nus, FLUXES4plotting, Nrealizations, filtered_modelpoints, index_dataexist)
            plt.savefig(folder+str(sourcename)+'/SED_realizations_'+str(sourcename)+'.pdf', format = 'pdf')
            plt.close(fig)




#====================================
#           FUNCTIONS
#====================================


def plot_trace_mcmc(P, chain, lnprob, nwplot=50):

    """ Plot the sample trace for a subset of walkers for each parameter.
    """

#-- Latex -------------------------------------------------
    matplotlib.rc('text', usetex=True)
    matplotlib.rc('font', family='serif')
    matplotlib.rc('axes', linewidth=1.5)
#-------------------------------------------------------------

    nwalkers, nsample, npar = chain.shape
    nrows = npar+1
    ncols =1         
    
    fig, axes = general.get_fig_axes(nrows, ncols, npar+1)

    nwplot = min(nsample, nwplot)
    for i in range(npar):
        ax = axes[i]
        for j in range(0, nwalkers, max(1, nwalkers // nwplot)):
            ax.plot(chain[j,:,i], lw=0.5,  color = 'black', alpha = 0.3)
        ax.set_title(r'\textit{Parameter : }'+P.names[i], fontsize=12)  
        ax.set_xlabel(r'\textit{Steps}', fontsize=12)
        ax.set_ylabel(r'\textit{Walkers}',fontsize=12)


    ax = axes[-1]
    for j in range(0, nwalkers, max(1, nwalkers // nwplot)):
        ax.plot(lnprob[j,:], lw=0.5, color = 'black', alpha = 0.3)
    ax.set_title(r'\textit{Likelihood}', fontsize=12)   
    ax.set_xlabel(r'\textit{Steps}', fontsize=12)
    ax.set_ylabel(r'\textit{Walkers}',fontsize=12)

    plt.close(fig)

    return fig, nwplot




def plot_trace_burnin123(P, chain, lnprob, nwplot=50):

    """ Plot the sample trace for a subset of walkers for each parameter.
    """
    nwalkers, nsample, npar = chain.shape
    nrows, ncols = general.get_nrows_ncols(npar+1)
    fig, axes = general.get_fig_axes(nrows, ncols, npar+1)
    # number of walkers to plot
    nwplot = min(nsample, nwplot)
    for i in range(npar):
        ax = axes[i]
        for j in range(0, nwalkers, max(1, nwalkers // nwplot)):
            ax.plot(chain[j,:,i], lw=0.5, alpha=0.5)    
        ax.plot(nsample*0.02, chain[j,nsample*0.02,i], '>b', ms=6, mew=1.5)         

        general.puttext(0.96, 0.02, P.names[i], ax, ha='right', fontsize=17)
    general.puttext(0.65, 0.93, 'Nr. of steps', ax, ha='right', fontsize=12)
    general.puttext(0.97, 0.7, 'Location of 10  walkers', ax, ha='right', fontsize=12, rotation=90)

    ax = axes[-1]
    for j in range(0, nwalkers, max(1, nwalkers // nwplot)):
        ax.plot(lnprob[j,:], lw=0.5, alpha=0.5)
    ax.plot(nsample*10/22, lnprob[j,nsample*10/22], 'xk', ms=6, mew=1)
    ax.plot(nsample*16/22, lnprob[j,nsample*16/22], 'xk', ms=6, mew=1)  
    general.puttext(0.96, 0.02, "ln_Likelihood", ax, ha='right', fontsize=17)

        
    return fig, nwplot


#===============================================
#            INTEGRATED LUMINOSITIES
#===============================================



def integrate_luminosities(x, M1,M2,M3,M4,Mtot):

    """
    This function integrates the luminosities of the components.

    ## inputs:
    - array of frequencies (which should be very dense in order to perform a good integration)
    (Output from modelfluxes_plot)
    - M1, M2, M3, M4 the total luminosities of each of the four components, which are plotted as well
    - Mtot the sum of the four luminosities before

    ## output:
    - Lir_agn, Lfir, Lbol_agn, Lbol_deredd 
      (integrated luminosities in different frequency ranges, as given below)
    - Loptuv_BB, Loptuv_GA, LMir_TO, LMir_SB as well

    """

    #    L_IR DECOMPOSED TORUS

    c = 2.998 * 1e8
    a1 = np.log10(c / (1 * 1e-6))#boundaries for integral
    a2 = np.log10(c / (1000 * 1e-6)) 

    indexa  = ((x >= a2) & (x<= a1))
    x_Lir = 10**(x[indexa])
    TOR_Lir = M4[indexa]/x_Lir  

    Lir_agn = np.log10(scipy.integrate.trapz(TOR_Lir, x=x_Lir))

    #    L_MIR-TORUS

    e1 = np.log10(c / (1 * 1e-6))
    e2 = np.log10(c / (30 * 1e-6)) 

    indexe  = ((x >= e2) & (x<= e1))
    x_Mir = 10**(x[indexe])
    TOR_Mir = M4[indexe]/x_Mir  

    LMir_TO = np.log10(scipy.integrate.trapz(TOR_Mir, x=x_Mir))


       #    L_MIR- STARBURST

    e1 = np.log10(c / (1 * 1e-6))#boundaries for integral
    e2 = np.log10(c / (30 * 1e-6)) 
    
    indexe  = ((x >= e2) & (x<= e1))
    x_Mir = 10**(x[indexe])
    SB_Mir = M1[indexe]/x_Mir  

    LMir_SB = np.log10(scipy.integrate.trapz(SB_Mir, x=x_Mir))


    #   L_FIR

    b1 = np.log10(c / (8 * 1e-6))
    b2 = np.log10(c / (1000 * 1e-6)) 

    indexb  = ((x >= b2) & (x<= b1))
    x_Lfir = 10**(x[indexb])
    Mtot_Lfir = M1[indexb]/x_Lfir

    Lfir = np.log10(scipy.integrate.trapz(Mtot_Lfir, x=x_Lfir))

    #   L_BOL BIG BLUE BUMP 

    c1 = np.log10(c / (1 * 1e-6))

    indexc  = (x>= c1)
    x_Lbol = 10**(x[indexc])
    Lbol_bbb = M2[indexc]/x_Lbol

    Lbol_agn = np.log10(scipy.integrate.trapz(Lbol_bbb, x=x_Lbol))


    #   L_BOL BIG BLUE BUMP (1 micron till 0.1micron)

    d1 = np.log10(c / (0.1 * 1e-6))
    d2 = np.log10(c/ (1 * 1e-6))

    indexd  = ((x >= d2) & (x<= d1))
    x_Lbol_2limits= 10**(x[indexd])
    OPTUV_BB = M2[indexd]/x_Lbol_2limits

    Loptuv_BB = np.log10(scipy.integrate.trapz(OPTUV_BB, x=x_Lbol_2limits))

    #   L_BOL GALAXY (1 micron till 0.1micron)

    d1 = np.log10(c / (0.1 * 1e-6))
    d2 = np.log10(c/ (1 * 1e-6))


    indexd  = ((x >= d2) & (x<= d1))
    x_Lbol_2limits= 10**(x[indexd])

    OPTUV_GA = M3[indexd]/x_Lbol_2limits
    
    Loptuv_GA = np.log10(scipy.integrate.trapz(OPTUV_GA, x=x_Lbol_2limits))





    
    Lfir ="{0:.3f}".format(Lfir)
    Lbol_agn ="{0:.3f}".format(Lbol_agn)
    Lir_agn ="{0:.3f}".format(Lir_agn)


    Loptuv_BB ="{0:.3f}".format(Loptuv_BB)
    Loptuv_GA ="{0:.3f}".format(Loptuv_GA)
    LMir_TO ="{0:.3f}".format(LMir_TO)
    LMir_SB = "{0:.3f}".format(LMir_SB)

    return Loptuv_BB,Loptuv_GA, LMir_TO, LMir_SB 




def integrate_luminosities_deredd( x, M2_deredd):


    #   L_BOL BIG BLUE BUMP (1micron till all UV)
    c = 2.998 * 1e8
    c1 = np.log10(c / (1 * 1e-6))

    indexc  = (x>= c1)
    x_Lbol = 10**(x[indexc])
    Mtot_Lbol = M2_deredd[indexc]/x_Lbol

    Lbol_agn = np.log10(scipy.integrate.trapz(Mtot_Lbol, x=x_Lbol))

    # #   L_BOL BIG BLUE BUMP (1 micron till 0.1micron)
    d = 2.998 * 1e8
    d1 = np.log10(c / (0.1 * 1e-6))
    d2 = np.log10(c/ (1 * 1e-6))

    indexd  = ((x >= d2) & (x<= d1))
    x_Lbol = 10**(x[indexd])
    OPTUV_BB_deredd = M2_deredd[indexd]/x_Lbol

    Loptuv_BB_deredd = np.log10(scipy.integrate.trapz(OPTUV_BB_deredd, x=x_Lbol))

    Lbol_agn ="{0:.3f}".format(Lbol_agn)
    Loptuv_BB_deredd ="{0:.3f}".format(Loptuv_BB_deredd)



    #   L_MIR GALAXY (1 micron till 0.1micron)

    d1 = np.log10(c / (1 * 1e-6))
    d2 = np.log10(c/ (30 * 1e-6))


    indexd  = ((x >= d2) & (x<= d1))
    x_Lbol_2limits= 10**(x[indexd])

    OPTUV_GA = M2_deredd[indexd]/x_Lbol_2limits
    
    Lmir_GA = np.log10(scipy.integrate.trapz(OPTUV_GA, x=x_Lbol_2limits))


    Lmir_GA ="{0:.3f}".format(Lmir_GA)





    return Loptuv_BB_deredd# Lmir_GA


def integrate_luminositires_SFR(x, M1): 

   #   L_FIR
    c = 2.998 * 1e8

    b1 = np.log10(c / (8 * 1e-6))
    b2 = np.log10(c / (1000 * 1e-6)) 

    indexb  = ((x >= b2) & (x<= b1))
    x_Lfir = 10**(x[indexb])
    Mtot_Lfir = M1[indexb]/x_Lfir

    Lfir = np.log10(scipy.integrate.trapz(Mtot_Lfir, x=x_Lfir))

    Lfir ="{0:.3f}".format(Lfir)

    return Lfir

def integrated_luminosities_arrays(all_nus, FLUXES4plotting, Nrealizations, total_length_chain , z, folder, sourcename):

    """
    This function derives quantities (integrated luminosities) for all realizations. 

    These are calculated individually in fct PLOTandWRITE.integrate_luminosities().
    There, you can change its output as you wish, in order to obtain other derived quantities
    eg. integrated luminosities with different integration limits.

    Here, for clarity,  you have to change the names of the luminosities calculated.

    """


    SBnuLnu, BBnuLnu, GAnuLnu, TOnuLnu, TOTALnuLnu, BBnuLnu_deredd = [f for f in FLUXES2nuLnu_4plotting(10**all_nus, FLUXES4plotting, z)]


    L0 = []#Loptuv_SB
    L1 = [] #Loptuv_BB
    L2 = [] #LMir_GA
    L3 = [] #LMir_TO
    L4 = [] #Loptuv_BB_deredd
    L5 = [] #LFIR for SFR calculation   

    for i in range(Nrealizations):


        Loptuv_BB, Loptuv_GA, LMir_TO, LMir_SB  = integrate_luminosities(all_nus, SBnuLnu[i], BBnuLnu[i], GAnuLnu[i], TOnuLnu[i], TOTALnuLnu[i])
        #Loptuv_BB_dered = integrate_luminosities_deredd(all_nus, BBnuLnu_deredd[i])
        Loptuv_BB_dered = integrate_luminosities_deredd(all_nus, GAnuLnu[i]) #actually Lbbb_deredd #notgalaxy
        LFIR_4SFR = integrate_luminositires_SFR(all_nus, SBnuLnu[i])    
 
        for i in range(int(total_length_chain /Nrealizations)):
            L0.append(float(LMir_SB))
            L1.append(float(Loptuv_BB))
            L2.append(float(Loptuv_GA))
            L3.append(float(LMir_TO))  
            L4.append(float(Loptuv_BB_dered))
            L5.append(float(LFIR_4SFR)) 

    L0 = np.array(L0)        
    L1 = np.array(L1)
    L2 = np.array(L2)
    L3 = np.array(L3)   
    L4 = np.array(L4)   
    L5= np.array(L5)   


    luminosities = np.column_stack((L0, L1, L2, L3, L4, L5))
    np.savetxt(folder + str(sourcename)+'/integrated_luminosities_1_'+str(sourcename)+'.txt' ,luminosities , delimiter = " ",fmt="%s" ,header="LMir_SB Loptuv_BB Loptuv_GA LMir_TO Lfir_SFR ")

    return L0, L1, L2, L3, L4, L5




def stellar_info_array(chain_flat,  catalog, sourceline, Nthin_compute):

    Ns, Npar = np.shape(chain_flat)  
    chain_thinned = chain_flat[0:Ns:int(Ns/Nthin_compute),:]
    
    Mstar, SFR, SFR_file = model.stellar_info(chain_thinned,  catalog, sourceline)
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
#====================================
#          CALCULATING FLUXES
#====================================




def fluxes_arrays(data, dict_modelsfiles, filterdict, chain, Nrealizations, dict_modelfluxes):
    """
    This function constructs the luminosities arrays for many realizations from the parameter values

    ## inputs:
    - v: frequency 
    - catalog file name
    - sourcelines

    ## output:
    - dictionary P with all parameter characteristics
    """
    SBFnu_list = []
    BBFnu_list = []
    GAFnu_list= []
    TOFnu_list = []
    TOTALFnu_list = []
    BBFnu_deredd_list = []

    filtered_modelpoints_list = []

    all_tau, all_age, all_nh, all_irlum, filename_0_galaxy, filename_0_starburst, filename_0_torus = dict_modelsfiles

    STARBURSTFdict , BBBFdict, GALAXYFdict, TORUSFdict, EBVbbb_array, EBVgal_array = dict_modelfluxes

    data_nus = data.nus
    catalog = data.catalog
    sourceline = data.sourceline
    path = data.path

    nsample, npar = chain.shape
    source = data.name
    tau, agelog, nh, irlum, SB ,BB, GA,TO, BBebv0, GAebv0= [ chain[:,i] for i in range(npar)] #C




    z = data.z
    
    age = 10**agelog
    agelog = np.log10(age)  



                
    for g in range(Nrealizations): #LOOP for a 100th part of the realizations

        g= g*(nsample/Nrealizations)

        BBebv1 = model.pick_EBV_grid(EBVbbb_array, BBebv0[g])
        BBebv = (  str(int(BBebv1)) if  float(BBebv1).is_integer() else str(BBebv1))
        GAebv1 = model.pick_EBV_grid(EBVgal_array,GAebv0[g])
        GAebv = (  str(int(GAebv1)) if  float(GAebv1).is_integer() else str(GAebv1))


        SB_filename = path + model.pick_STARBURST_template(irlum[g], filename_0_starburst, all_irlum)
        GA_filename = path + model.pick_GALAXY_template(tau[g], age[g], filename_0_galaxy, all_tau, all_age)
        TO_filename = path + model.pick_TORUS_template(nh[g], all_nh, filename_0_torus)
        BB_filename = path + model.pick_BBB_template()

        all_model_nus = np.arange(11.5, 16, 0.001)#np.log10(dicts.stack_all_model_nus(filename_0_galaxy, filename_0_starburst, filename_0_torus, z, path ))  
        gal_nu, gal_nored_Fnu = model.GALAXY_read_4plotting( GA_filename, all_model_nus)
        gal_nu, gal_Fnu_red = model.GALAXY_nf2( gal_nu, gal_nored_Fnu, float(GAebv) )
        all_gal_nus, all_gal_Fnus =gal_nu, gal_Fnu_red

        sb_nu0, sb_Fnu0 = model.STARBURST_read_4plotting(SB_filename, all_model_nus)
        all_sb_nus, all_sb_Fnus = sb_nu0, sb_Fnu0

        bbb_nu, bbb_nored_Fnu = model.BBB_read_4plotting(BB_filename, all_model_nus)
        bbb_nu0, bbb_Fnu_red = model.BBB_nf2(bbb_nu, bbb_nored_Fnu,float(BBebv), z )
        all_bbb_nus, all_bbb_Fnus = bbb_nu0, bbb_Fnu_red
        all_bbb_nus, all_bbb_Fnus_deredd = bbb_nu0, bbb_nored_Fnu

        tor_nu0, tor_Fnu0 = model.TORUS_read_4plotting(TO_filename, z, all_model_nus)

        all_tor_nus, all_tor_Fnus = tor_nu0, tor_Fnu0

        par2 = tau[g], agelog[g], nh[g], irlum[g], SB[g] ,BB[g], GA[g] ,TO[g], float(BBebv), float(GAebv)

        filtered_modelpoints = parspace.ymodel(data_nus, z, dict_modelsfiles, dict_modelfluxes, *par2)


        if len(all_gal_nus)==len(all_sb_nus) and len(all_sb_nus)==len(all_bbb_nus) and len(all_tor_nus)==len(all_bbb_nus) :
            nu= all_gal_nus

            all_sb_Fnus_norm = all_sb_Fnus /1e20#/1e50
            all_bbb_Fnus_norm = all_bbb_Fnus /1e60#/ 1e90
            all_bbb_Fnus_deredd_norm = all_bbb_Fnus_deredd /1e60#/ 1e90
            all_gal_Fnus_norm = all_gal_Fnus/ 1e18
            all_tor_Fnus_norm = all_tor_Fnus/  1e-40

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
    

    FLUXES4plotting = (SBFnu_array, BBFnu_array, GAFnu_array, TOFnu_array, TOTALFnu_array,BBFnu_array_deredd)

    return all_model_nus, FLUXES4plotting, filtered_modelpoints


def FLUXES2nuLnu_4plotting(all_nus_rest, FLUXES4plotting, z):

    """
    input: all_nus_rest (10** , NOT log!!!!)
    """
    all_nus_rest = all_nus_rest 
    all_nus_obs = all_nus_rest /(1+z) #observed
    distance= model.z2Dlum(z)
    lumfactor = (4. * math.pi * distance**2.)

    SBnuLnu, BBnuLnu, GAnuLnu, TOnuLnu, TOTALnuLnu, BBnuLnu_deredd = [ f *lumfactor*all_nus_obs for f in FLUXES4plotting]

    return SBnuLnu, BBnuLnu, GAnuLnu, TOnuLnu, TOTALnuLnu, BBnuLnu_deredd



def plot_posteriors_triangle_pars(chain, best_fit_par):

    npar = chain.shape[-1]
    figure = triangle.corner(chain, labels=[r"$\tau$", r"Age", r"N$_H$", r"L$_{IR}$", r"SB", r"BB", r"GA", r"TO", r"E(B-V)$_\mathrm{BB}$", r"E(B-V)$_{GAL}$"], plot_contours=True, plot_datapoints = False, show_titles=True, quantiles=[0.16, 0.50, 0.84])#, truths = best_fit_par)#truths = [2 , 5e9, 22.7 , 13.26, -16.1, -55.36, -9.23, 47.3, 0.01, 0.01 ]) 
    return figure



def plot_posteriors_triangle_other_quantities(chain):

    figure = triangle.corner(chain, labels=[r"L$_{SB (MIR)}$",r"L$_{BB (OPT-UV)}$",r"L$_{GA (OPT-UV)}$", r"L$_{TO (MIR)}$"],   plot_contours=True, plot_datapoints = False, show_titles=True, quantiles=[0.16, 0.50, 0.84])

    return figure

def parameters_with_errors(chain):

    pars_with_errors_intlums= map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(chain, [16, 50, 84],  axis=0)))


    return pars_with_errors_intlums






#====================================
#           PLOTTING SEDS
#====================================




def PLOT_SED_bestfit(data, all_nus, FLUXES4plotting, filtered_modelpoints, index_dataexist):

    source = data.name
    data_nus_0= data.nus
    ydata_0 = data.fluxes
    yerrors_0 = data.fluxerrs
    z = data.z

    # Choosing only existing data points (source-dependent: not total CATALOG array)
    data_nus = data_nus_0[index_dataexist]
    data_flux = ydata_0[index_dataexist]
    data_errors = yerrors_0[index_dataexist]
  

    data_nus_obs = 10**data_nus
    data_nus_rest =10**data_nus*(1+z) #rest
    data_nus =np.log10(data_nus_rest)
    all_nus_rest = 10**all_nus
    all_nus_obs = 10**all_nus/(1+z) #observed 
    distance= model.z2Dlum(z)
    lumfactor = (4. * math.pi * distance**2.)


    data_nuLnu_rest = data_flux* data_nus_obs *lumfactor
    data_errors_rest= data_errors * data_nus_obs * lumfactor

    fig, ax1, ax2 = SED_plotting_settings(all_nus_rest, data_nuLnu_rest)


    SBnuLnu, BBnuLnu, GAnuLnu, TOnuLnu, TOTALnuLnu, BBnuLnu_deredd = [f[0] for f in FLUXES2nuLnu_4plotting(all_nus_rest, FLUXES4plotting, z)]

#    print np.shape(filtered_modelpoints), np.shape(data_nus_0), np.shape(data_nus)
#    filtered_modelpoints_best = filtered_modelpoints * (10**data_nus_0) * lumfactor



    #Settings for model lines
    SBcolor, BBcolor, GAcolor, TOcolor, TOTALcolor= SED_colors(combination = 'a')
    lw= 1.5
    p1= ax1.plot( all_nus, TOTALnuLnu, marker="None", linewidth=lw,  label="1 /sigma", color= TOTALcolor, alpha= 1.0)
    p2=ax1.plot(all_nus, SBnuLnu, marker="None", linewidth=lw, label="1 /sigma", color= SBcolor, alpha = 0.6)
    p3=ax1.plot(all_nus, BBnuLnu, marker="None", linewidth=lw, label="1 /sigma",color= BBcolor, alpha = 0.6)
    p4=ax1.plot( all_nus, GAnuLnu,marker="None", linewidth=lw, label="1 /sigma",color=GAcolor, alpha = 0.6)
    p5=ax1.plot( all_nus, TOnuLnu, marker="None",  linewidth=lw, label="1 /sigma",color= TOcolor, alpha = 0.6)
    interp_total= scipy.interpolate.interp1d(all_nus, TOTALnuLnu, bounds_error=False, fill_value=0.)
    TOTALnuLnu_at_datapoints = interp_total(data_nus)

    
#    p6 = ax1.plot(np.log10(10**data_nus_0*(1+z)), filtered_modelpoints_best[0] ,  marker='o', linestyle="None", markersize=5, color="red")
    #p6 = ax1.plot(data_nus, TOTALnuLnu_at_datapoints ,  marker='o', linestyle="None", markersize=5, color="red")
    
    (_, caps, _) = ax1.errorbar(data_nus, data_nuLnu_rest, yerr= data_errors_rest, capsize=4, linestyle="None", linewidth=1.5,  marker='o',markersize=5, color="black", alpha = 0.5)

    ax1.annotate(r'XID='+str(source)+r', z ='+ str(z), xy=(0, 1),  xycoords='axes points', xytext=(20, 310), textcoords='axes points' )#+ ', log $\mathbf{L}_{\mathbf{IR}}$= ' + str(Lir_agn) +', log $\mathbf{L}_{\mathbf{FIR}}$= ' + str(Lfir) + ',  log $\mathbf{L}_{\mathbf{UV}} $= '+ str(Lbol_agn)

    #plt.savefig(folder+str(source)+'/SEDbest_'+str(source)+'.pdf', format = 'pdf')#, dpi = 900

    print ' => Best fit SED was plotted.'
    return fig

def PLOT_SED_manyrealizations(data, all_nus, FLUXES4plotting, Nrealizations, filtered_modelpoints, index_dataexist):

    source = data.name
    data_nus_0= data.nus
    ydata_0 = data.fluxes
    yerror_0 = data.fluxerrs
    z = data.z
    # Choosing only existing data points (source-dependent: not total CATALOG array)
 
    data_nus = data_nus_0[index_dataexist]
    ydata = ydata_0[index_dataexist]
    yerror = yerror_0[index_dataexist]
   


    #data nuLnu
    data_nus_obs = 10**data_nus 
    data_nus_rest = 10**data_nus * (1+z) 
    data_nus = np.log10(data_nus_rest)
    all_nus_rest = 10**all_nus 
    all_nus_obs =  10**all_nus / (1+z) #observed
    distance= model.z2Dlum(z)
    lumfactor = (4. * math.pi * distance**2.)
    data_nuLnu_rest = ydata* data_nus_obs *lumfactor
    data_errors_rest= yerror * data_nus_obs * lumfactor

    fig, ax1, ax2 = SED_plotting_settings(all_nus_rest, data_nuLnu_rest)

    SBnuLnu, BBnuLnu, GAnuLnu, TOnuLnu, TOTALnuLnu, BBnuLnu_deredd = FLUXES2nuLnu_4plotting(all_nus_rest, FLUXES4plotting, z)

    SBcolor, BBcolor, GAcolor, TOcolor, TOTALcolor= SED_colors(combination = 'a')
    lw= 1.5

    thinning_4plot = len(TOTALnuLnu) / (Nrealizations)


    for j in range(Nrealizations):

        
        i = j * 10 
        #Settings for model lines
        p2=ax1.plot(all_nus, SBnuLnu[i], marker="None", linewidth=lw, label="1 /sigma", color= SBcolor, alpha = 0.5)
        p3=ax1.plot(all_nus, BBnuLnu[i], marker="None", linewidth=lw, label="1 /sigma",color= BBcolor, alpha = 0.5)
        p4=ax1.plot( all_nus, GAnuLnu[i],marker="None", linewidth=lw, label="1 /sigma",color=GAcolor, alpha = 0.5)
        p5=ax1.plot( all_nus, TOnuLnu[i], marker="None",  linewidth=lw, label="1 /sigma",color= TOcolor ,alpha = 0.5)
        p1= ax1.plot( all_nus, TOTALnuLnu[i], marker="None", linewidth=lw,  label="1 /sigma", color= TOTALcolor, alpha= 0.5)

        interp_total= scipy.interpolate.interp1d(all_nus, TOTALnuLnu[i], bounds_error=False, fill_value=0.)

        TOTALnuLnu_at_datapoints = interp_total(data_nus)


        (_, caps, _) = ax1.errorbar(data_nus, data_nuLnu_rest, yerr= data_errors_rest, capsize=4, linestyle="None", linewidth=1.5,  marker='o',markersize=5, color="black", alpha = 0.5)
        p6 = ax1.plot(data_nus, TOTALnuLnu_at_datapoints ,   marker='o', linestyle="None",markersize=5, color="red")
        
#        p6 = ax1.plot(np.log10(10**data_nus_0 *(1+z)), filtered_modelpoints[i] ,  marker='o', linestyle="None",markersize=5, color="red")

    ax1.annotate(r'XID='+str(source)+r', z ='+ str(z), xy=(0, 1),  xycoords='axes points', xytext=(20, 310), textcoords='axes points' )#+ ', log $\mathbf{L}_{\mathbf{IR}}$= ' + str(Lir_agn) +', log $\mathbf{L}_{\mathbf{FIR}}$= ' + str(Lfir) + ',  log $\mathbf{L}_{\mathbf{UV}} $= '+ str(Lbol_agn)
    print ' => SEDs of '+ str(Nrealizations)+' different realization were plotted.'

    return fig




def SED_colors(combination = 'a'):

    if combination=='a':   
        steelblue = '#4682b4'
        darkcyan ='#009acd'
        deepbluesky = '#008b8b'
        seagreen = '#2E8B57'    
        lila = '#68228B'
        darkblue='#123281'
    
    return seagreen, darkblue, 'orange', lila, 'red'
    





def SED_plotting_settings(x, ydata):

    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    x2 = (2.98e8) / x / (1e-6) # Wavelenght axis
    ax2 = ax1.twiny()
    ax2.plot(x2, np.ones(len(x2)), alpha=0)

#-- Latex -------------------------------------------------
    matplotlib.rc('text', usetex=True)
    matplotlib.rc('font', family='serif')
    matplotlib.rc('axes', linewidth=2)
#-------------------------------------------------------------

#    ax1.set_title(r"\textbf{SED of Type 2}" + r"\textbf{ AGN }"+ "Source Nr. "+ source + "\n . \n . \n ." , fontsize=17, color='k')    
    ax1.set_xlabel(r'rest-frame frequency$\mathbf{log \  \nu} [\mathtt{Hz}] $', fontsize=11)
    ax2.set_xlabel(r'rest-frame wavelength $\mathbf{\lambda} [\mathtt{\mu m}] $', fontsize=11)
    ax1.set_ylabel(r'luminosity $\mathbf{\nu L(\nu) [\mathtt{erg \ } \mathtt{ s}^{-1}]}$',fontsize=11)

    ax1.set_autoscalex_on(True) 
    ax1.set_autoscaley_on(True) 
    ax1.set_xscale('linear')
    ax1.set_yscale('log')

    ax1.set_xlim([11.5,16])
    mediandata = np.median(ydata)
    ax1.set_ylim(mediandata /50.,mediandata * 50.)

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_ylim( mediandata /50., mediandata * 50.)
 
    ax2.set_xlim([2.98e8 / (10**11.5) / (1e-6) , (2.98e8) / (10**16) / (1e-6)])
    ax2.set_xticks([100, 10,1, 0.1]) 
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    return fig, ax1, ax2




if __name__ == 'main':
    main(sys.argv[1:])
