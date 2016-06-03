

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
from astropy import units as u
from astropy import constants as const

#AGNfitter IMPORTS
import GENERAL_AGNfitter as general
import MODEL_AGNfitter as model
import DICTIONARIES_AGNfitter as dicts
import PARAMETERSPACE_AGNfitter as parspace
import pickle

def main(filename, data, P, opt, dict_modelsfiles, filterdict, dict_modelfluxes):
	print 'ALOHA'



class CHAIN:

	"""
	Class OUTPUT
	input: 
	bugs: 

	"""     

	def __init__(self, outputfilename, opt, data):
			self.outputfilename = outputfilename
			self.opt = opt
			self.data = data

	def props(self):
		if os.path.lexists(self.data.output_folder+str(self.data.name)+'/samples_mcmc.sav'):

			f = open(self.outputfilename, 'rb')
			samples = pickle.load(f)
			f.close()

			self.nwalkers, self.nsamples, self.npar = samples['chain'].shape
			self.parametersoutput = [ samples['chain'][:,i] for i in range(self.npar)]

			Ns, Nt = self.opt['Nsample'], self.opt['Nthinning']
			
			lnprob = samples['lnprob'][:,0:Ns*Nt:Nt].ravel()

			isort = (- lnprob).argsort() #sort parameter vector for likelihood
			lnprob_sorted = np.reshape(lnprob[isort],(-1,1))
			self.lnprob_max = lnprob_sorted[0]

			self.flatchain = samples['chain'][:,0:Ns*Nt:Nt,:].reshape(-1, self.npar)
			chain_length = int(len(flatchain))

			flatchain_sorted = flatchain[isort]
			self.best_fit_pars = flatchain[isort[0]]


			print '_________________________________'
			print 'Some properties of the sampling:'
			mean_accept =  samples['accept'].mean()
			print '- Mean acceptance fraction', mean_accept
			mean_autocorr = samples['acor'].mean()
			print '- Mean autocorrelation time', mean_mean_autocorr

		else:
			'Error: The sampling has not been perfomed yet, or the chains were not saved properly.'
	



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



	def plot_trace_mcmc(P, chain, lnprob, nwplot=50):

		""" Plot the sample trace for a subset of walkers for each parameter.
		"""

		#-- Latex -------------------------------------------------
		matplotlib.rc('text', usetex=True)
		matplotlib.rc('font', family='serif')
		matplotlib.rc('axes', linewidth=1.5)
		#-------------------------------------------------------------

		nwalkers, nsample, npar = chain.shape

		#    nrows, ncols = general.get_nrows_ncols(npar+1)
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





class FLUXES_ARRAYS:


	def __init__(self, parameters, out, output_type):
		self.parameters = parameters
		self.output_type = output_type
		self.out = out

	def fluxes(self, dict_modelsfiles, filterdict, dict_modelfluxes):	

		self.props()

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
		if self.output_type == 'plot':
			filtered_modelpoints_list = []


		all_tau, all_age, all_nh, all_irlum, filename_0_galaxy, filename_0_starburst, filename_0_torus = dict_modelsfiles
		STARBURSTFdict , BBBFdict, GALAXYFdict, TORUSFdict, EBVbbb_array, EBVgal_array = dict_modelfluxes

		nsample, npar = self.flatchain.shape
		source = data.name

		#THIS NEEDS TO BE CHANGED INTO RANDOMLY
		if self.output_type == 'plot':
			tau, agelog, nh, irlum, SB ,BB, GA,TO, BBebv0, GAebv0= self.parameters[0:nsample:nsample/(self.out['realizations2plot']),:]
		elif self.output_type == 'intlum':
			tau, agelog, nh, irlum, SB ,BB, GA,TO, BBebv0, GAebv0= self.parameters[0:nsample:nsample/(self.out['realizations2compute']),:]
		elif self.output_type == 'best_fit':
			tau, agelog, nh, irlum, SB ,BB, GA,TO, BBebv0, GAebv0= self.best_fit_pars

		age = 10**agelog 

		for g in range(len(tau)):

			BBebv1 = model.pick_EBV_grid(EBVbbb_array, BBebv0[g])
			BBebv = (  str(int(BBebv1)) if  float(BBebv1).is_integer() else str(BBebv1))
			GAebv1 = model.pick_EBV_grid(EBVgal_array,GAebv0[g])
			GAebv = (  str(int(GAebv1)) if  float(GAebv1).is_integer() else str(GAebv1))

			SB_filename = data.path + model.pick_STARBURST_template(irlum[g], filename_0_starburst, all_irlum)
			GA_filename = data.path + model.pick_GALAXY_template(tau[g], age[g], filename_0_galaxy, all_tau, all_age)
			TO_filename = data.path + model.pick_TORUS_template(nh[g], all_nh, filename_0_torus)
			BB_filename = data.path + model.pick_BBB_template()

			all_nus_rest = np.arange(11.5, 16, 0.001)

			gal_nu, gal_nored_Fnu = model.GALAXY_read_4plotting( GA_filename, all_nus_rest)
			gal_nu, gal_Fnu_red = model.GALAXY_nf2( gal_nu, gal_nored_Fnu, float(GAebv) )
			all_gal_nus, all_gal_Fnus =gal_nu, gal_Fnu_red

			sb_nu0, sb_Fnu0 = model.STARBURST_read_4plotting(SB_filename, all_nus_rest)
			all_sb_nus, all_sb_Fnus = sb_nu0, sb_Fnu0

			bbb_nu, bbb_nored_Fnu = model.BBB_read_4plotting(BB_filename, all_nus_rest)
			bbb_nu0, bbb_Fnu_red = model.BBB_nf2(bbb_nu, bbb_nored_Fnu,float(BBebv), data.z )
			all_bbb_nus, all_bbb_Fnus = bbb_nu0, bbb_Fnu_red
			all_bbb_nus, all_bbb_Fnus_deredd = bbb_nu0, bbb_nored_Fnu

			all_tor_nus, all_tor_Fnus = model.TORUS_read_4plotting(TO_filename, data.z, all_nus_rest)


			if self.output_type == 'plot':
				par2 = tau[g], agelog[g], nh[g], irlum[g], SB[g] ,BB[g], GA[g] ,TO[g], float(BBebv), float(GAebv)
				filtered_modelpoints = parspace.ymodel(data.nus, data.z, dict_modelsfiles, dict_modelfluxes, *par2)


			if len(all_gal_nus)==len(all_sb_nus) and len(all_sb_nus)==len(all_bbb_nus) and len(all_tor_nus)==len(all_bbb_nus) :
				nu= all_gal_nus

				SBFnu =   (all_sb_Fnus /1e20) *10**float(SB[g]) 
				BBFnu = (all_bbb_Fnus /1e60) * 10**float(BB[g]) 
				GAFnu =   (all_gal_Fnus/ 1e18) * 10**float(GA[g]) 
				TOFnu =   (all_tor_Fnus/  1e-40) * 10**float(TO[g])
				BBFnu_deredd = (all_bbb_Fnus_deredd /1e60) * 10**float(BB[g])

				TOTALFnu =    SBFnu + BBFnu + GAFnu + TOFnu
				SBFnu_list.append(SBFnu)
				BBFnu_list.append(BBFnu)
				GAFnu_list.append(GAFnu)
				TOFnu_list.append(TOFnu)
				TOTALFnu_list.append(TOTALFnu)
				BBFnu_deredd_list.append(BBFnu_deredd)
				if self.output_type == 'plot':
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
		if self.output_type == 'plot':
			filtered_modelpoints = np.array(filtered_modelpoints_list)


		FLUXES4plotting = (SBFnu_array, BBFnu_array, GAFnu_array, TOFnu_array, TOTALFnu_array,BBFnu_array_deredd)
		nuLnus4plotting = FLUXES2nuLnu_4plotting(all_nus_rest, FLUXES4plotting, self.data.z)

		if self.output_type == 'plot':
			filtered_modelpoints_nuLnu = FLUXES2nuLnu_4plotting(all_nus_rest,  filtered_modelpoints, self.data.z)
			return all_nus_rest, nuLnus4plotting, filtered_modelpoints_nuLnu
		elif self.output_type == 'intlum':

			return all_nus_rest, nuLnus4plotting, integrated_luminosities

		elif self.output_type == 'best_fit':
		 	return all_nus_rest, nuLnus4plotting

	def FLUXES2nuLnu_4plotting(all_nus_rest, FLUXES4plotting, z):

		"""
		input: all_nus_rest (give in 10^lognu, NOT log!!!!)
		"""
		all_nus_obs = all_nus_rest /(1+z) 
		distance= model.z2Dlum(z)
		lumfactor = (4. * math.pi * distance**2.)

		SBnuLnu, BBnuLnu, GAnuLnu, TOnuLnu, TOTALnuLnu, BBnuLnu_deredd = [ f *lumfactor*all_nus_obs for f in FLUXES4plotting]

		return SBnuLnu, BBnuLnu, GAnuLnu, TOnuLnu, TOTALnuLnu, BBnuLnu_deredd


	def integrated_luminosities(all_nus_rest, nuLnus4plotting, out):

		np.shape()





	# def plotSED():


	# 	if opt['plotSEDbest']:
	# 		fig = PLOT_SED_bestfit(sourcename, data_nus, ydata, ysigma, z, all_nus, FLUXES4plotting, filtered_modelpoints, index_dataexist)
	# 		plt.savefig(folder+str(sourcename)+'/SED_best_'+str(sourcename)+'.pdf', format = 'pdf')
	# 		plt.close(fig)


