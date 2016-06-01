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
from multiprocessing import Pool
import PARAMETERSPACE_AGNfitter as parspace
import DATA_AGNfitter as data
from GENERAL_AGNfitter import saveobj



#==================================================
# BURN-IN PHASE FUNCTIONS
#==================================================


def run_burn_in(sampler, opt, p0, catalog, sourceline, sourcename, folder, setnr):
    """ Run and save a set of burn-in iterations."""

    print 'Running burn-in nr. '+ str(setnr)+' with %i steps' % opt['Nburn1']
    
    iprint = opt['iprint']

    # note the results are saved in the sampler object.
    for i,(pos, lnprob, state) in enumerate(sampler.sample(p0, iterations=opt['Nburn1'])):
        i += 1
        if not i % iprint:
            print i
    
    print 'Saving results to '+folder+str(sourcename)+'/samples_burn1-2-3.sav'
    
    save_samples1(folder+str(sourcename)+'/samples_burn1-2-3.sav', sampler, pos, lnprob, state)

    return pos, state   



def save_samples1(filename, sampler, pos, lnprob, state):
    saveobj(filename, dict(
        chain=sampler.chain, accept=sampler.acceptance_fraction,
        lnprob=sampler.lnprobability, final_pos=pos, state=state), overwrite=1)

def save_samples2(filename, sampler, pos, state):
    saveobj(filename, dict(
        chain=sampler.chain, accept=sampler.acceptance_fraction,
        lnprob=sampler.lnprobability, final_pos=pos, state=state), overwrite=1)


#==================================================
# MCMC SAMPLING FUNCTIONS
#==================================================


def run_mcmc(sampler, pburn, catalog, sourceline, sourcename, folder, opt):
	

    source1=sourcename
    sampler.reset()

    iprint = opt['iprint']
    print "Running MCMC with %i steps" % opt['Nmcmc']

    for i,(pos, lnprob, state) in enumerate(sampler.sample(pburn, iterations=opt['Nmcmc'])):	
        i += 1
        if not i % iprint:
            print i

    print 'Saving results to samples_mcmc.sav'

    save_samples2(folder+str(source1)+'/samples_mcmc.sav', sampler, pos, state)	



#============================================
#                MAIN FUNCTION
#============================================


def main(catalog, sourceline, P, folder, dict_modelsfiles, dict_modelfluxes, opt):

    x, ydata, ysigma = data.DATA(catalog, sourceline)
#    x, ydata, ysigma = data.MOCKdata(catalog, sourceline,2, dict_modelfluxes,path)
    
    sourcename = data.NAME(catalog, sourceline)
    z = data.REDSHIFT(catalog, sourceline)
    path = os.path.abspath(__file__).rsplit('/', 1)[0]

    print 'model parameters', P.names
    print 'minimum values', P.min
    print 'maximum values', P.max

    print opt['Nthreads'], 'threads'
    print opt['Nwalkers'], 'walkers'

    Npar = len(P.names)

    sampler = emcee.EnsembleSampler(
        opt['Nwalkers'], Npar, parspace.ln_probab,
        args=[x, ydata, ysigma, z, dict_modelsfiles, dict_modelfluxes,  P],  threads=opt['Nthreads'], daemon= True)

# Burn-in process: Just the last point visited here is relevant. Chains are discarted

    if opt['Nburn'] > 0:

        t1 = time.time()
        if not os.path.lexists(folder+str(sourcename)):
            os.mkdir(folder+str(sourcename))

        p_maxlike = parspace.get_initial_positions(opt['Nwalkers'], P)
        Nr_BurnIns = opt['Nburnsets']  

        for i in range(Nr_BurnIns):

            p_maxlike, state = run_burn_in(sampler, opt, p_maxlike, catalog, sourceline, sourcename,folder, i)
            savedfile = folder+str(sourcename)+'/samples_burn1-2-3.sav'
            p_maxlike = parspace.get_best_position(savedfile, opt['Nwalkers'], P)

        print '%.2g min elapsed' % ((time.time() - t1)/60.)


    if opt['Nmcmc'] > 0:

        t2 = time.time()
        run_mcmc(sampler, p_maxlike, catalog, sourceline, sourcename,folder, opt)
        print '%.2g min elapsed' % ((time.time() - t2)/60.)

    del sampler.pool    

if __name__ == 'main':
    main(sys.argv[1:])




