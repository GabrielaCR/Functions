"""

%%%%%%%%%%%%%%%%%%

Ensemble_MCMC_AGNfitter.py

%%%%%%%%%%%%%%%%%%

This script contains all  functions used by the MCMC machinery to explore the parameter space
of AGNfitter.

It contains:

* 	Initializing a point on the parameter space
*	Calculating the likelihood
*	Making the next step
*	Deciding when the burn-in is finished and start MCMC sampling


"""

import emcee
import sys,os
import time
import numpy as np
import pickle
from multiprocessing import Pool
import PARAMETERSPACE_AGNfitter as parspace
from DATANEW_AGNfitter import DATA



#==================================================
# BURN-IN PHASE FUNCTIONS
#==================================================


def run_burn_in(sampler, mc, p0, catalog, sourceline, sourcename, folder, setnr):
    """ Run and save a set of burn-in iterations."""

    print 'Running burn-in nr. '+ str(setnr)+' with %i steps' % mc['Nburn1']
    
    iprint = mc['iprint']

    # note the results are saved in the sampler object.
    for i,(pos, lnprob, state) in enumerate(sampler.sample(p0, iterations=mc['Nburn1'])):
        i += 1
        if not i % iprint:
            print i
    
    print 'Saving results to '+folder+str(sourcename)+'/samples_burn1-2-3.sav'
    
    save_chains(folder+str(sourcename)+'/samples_burn1-2-3.sav', sampler, pos, state)

    return pos, state   



def save_chains(filename, sampler, pos, state):
    f = open(filename, 'wb')
    pickle.dump(dict(
        chain=sampler.chain, accept=sampler.acceptance_fraction,
        lnprob=sampler.lnprobability, final_pos=pos, state=state, acor=sampler.acor), f, protocol=2)
    f.close()

#==================================================
# MCMC SAMPLING FUNCTIONS
#==================================================


def run_mcmc(sampler, pburn, catalog, sourceline, sourcename, folder, mc):
	

    source1=sourcename
    sampler.reset()

    iprint = mc['iprint']
    print "Running MCMC with %i steps" % mc['Nmcmc']

    for i,(pos, lnprob, state) in enumerate(sampler.sample(pburn, iterations=mc['Nmcmc'])):	
        i += 1
        if not i % iprint:
            print i

    print 'Saving results to samples_mcmc.sav'

    save_chains(folder+str(source1)+'/samples_mcmc.sav', sampler, pos, state)	



#============================================
#                MAIN FUNCTION
#============================================


def main(data, P, mc):

    x = data.nus
    ydata = data.fluxes
    ysigma= data.fluxerrs
    
    folder = data.output_folder
    sourceline = data.sourceline
    catalog = data.catalog
    sourcename = data.name
    dict_modelsfiles =data.dict_modelfile
    dict_modelfluxes =data.dict_modelfluxes
    z = data.z


    path = os.path.abspath(__file__).rsplit('/', 1)[0]

    print '......................................................'
    print 'model parameters', P.names
    print 'minimum values', P.min
    print 'maximum values', P.max
    print '......................................................'
    print mc['Nwalkers'], 'walkers'

    Npar = len(P.names)

    sampler = emcee.EnsembleSampler(
        mc['Nwalkers'], Npar, parspace.ln_probab,
        args=[x, ydata, ysigma, z, dict_modelsfiles, dict_modelfluxes,  P],  daemon= True)

# Burn-in process: Just the last point visited here is relevant. Chains are discarted

    if mc['Nburn'] > 0:

        t1 = time.time()
        if not os.path.lexists(folder+str(sourcename)):
            os.mkdir(folder+str(sourcename))

        p_maxlike = parspace.get_initial_positions(mc['Nwalkers'], P)
        Nr_BurnIns = mc['Nburnsets']  

        for i in range(Nr_BurnIns):

            p_maxlike, state = run_burn_in(sampler, mc, p_maxlike, catalog, sourceline, sourcename,folder, i)
            savedfile = folder+str(sourcename)+'/samples_burn1-2-3.sav'
            p_maxlike = parspace.get_best_position(savedfile, mc['Nwalkers'], P)

        print '%.2g min elapsed' % ((time.time() - t1)/60.)


    if mc['Nmcmc'] > 0:

        t2 = time.time()
        run_mcmc(sampler, p_maxlike, catalog, sourceline, sourcename,folder, mc)
        print '%.2g min elapsed' % ((time.time() - t2)/60.)

    del sampler.pool    

if __name__ == 'main':
    main(sys.argv[1:])




