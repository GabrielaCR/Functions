

"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      PLOTandWRITE_AGNfitter.py

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

This script contains all functions used in order to visualize the output of the sampling.
Plotting and writing. 
This function need to have the output files samples_mcmc.sav and samples_bur-in.sav.
"""
#PYTHON IMPORTS
import matplotlib.pyplot as plt
from matplotlib import rc
#matplotlib.use('Agg')
import sys, os
import math 
import numpy as np
import triangle
import time
import scipy
#AGNfitter IMPORTS
import GENERAL_AGNfitter as general
import MODEL_AGNfitter as model
import DICTIONARIES_AGNfitter as dicts
import PARAMETERSPACE_AGNfitter as parspace



class OUTPUT:

	"""
	Class OUTPUT
	input: 
	bugs: 

	"""     

	def __init__(self, outputfilename, opt):
		self.outputfilename = outputfilename
		self.opt = opt

	def chain(self):
		if os.path.lexists(folder+str(sourcename)+'/samples_mcmc.sav'):

			f = open(self.outputfilename, 'rb')
			samples = pickle.load(f)
			f.close()

			nwalkers, nsamples, npar = samples['chain'].shape

			print '_________________________________'
			print 'Some properties of the sampling:'
			mean_accept =  samples['accept'].mean()
			print '- Mean acceptance fraction', mean_accept
			mean_autocorr = samples['acor'].mean()
			print '- Mean autocorrelation time', mean_mean_autocorr


		else:
			'Error: The sampling has not been perfomed yet, or the chains were not saved properly.'



	def plotSED():

		if opt['plotSEDbest']:
			fig = PLOT_SED_bestfit(sourcename, data_nus, ydata, ysigma, z, all_nus, FLUXES4plotting, filtered_modelpoints, index_dataexist)
			plt.savefig(folder+str(sourcename)+'/SED_best_'+str(sourcename)+'.pdf', format = 'pdf')
			plt.close(fig)


