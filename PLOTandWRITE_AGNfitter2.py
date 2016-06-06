

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



def main(data, P, out):


	"""
	Main function of PLOTandWRITE_AGNfitter.

	##input:

	- data object
	- parameter space settings dictionary P
	- output settings-dictionary out
	
	"""

	chain_burnin = CHAIN(data.output_folder+str(data.name)+ '/samples_burnin.sav', data, out)
	chain_mcmc = CHAIN(data.output_folder+str(data.name)+ '/samples_mcmc.sav', data, out)	
	output = OUTPUT(chain_mcmc)

	if out['plot_tracesburn-in']:
		chain_burnin.plot_trace()

	if out['plot_tracesmcmc']:
		chain_mcmc.plot_trace()

	if out['plot_posteriortriangle'] :
		output.plot_PDFtriangle('10pars')

	if out['writepar_meanwitherrors']:
		output.write_parameters_outputvalues()

	if out['plot_posteriortrianglewithluminosities']: 
		output.plot_PDFtriangle('luminosities') 

	if out['plotSEDbest']:
		output.plot_bestfit_SED()

	if out['plotSEDrealizations']:
		output.plot_manyrealizations_SED()





"""=========================================================="""




class OUTPUT:

	"""
	Class OUTPUT

	Includes the functions that return all output products.

	##input: 
	- object of the CHAIN class

	##bugs: 

	"""    

	def __init__(self, chain_obj, data_obj):

		self.chain = chain_obj
		self.out = chain_obj.out
		self.data = data_obj
		fluxobj_withintlums = FLUXES_ARRAYS('intlum')
		fluxobj_4SEDplots = FLUXES_ARRAYS('plot')

		if out['calc_lum']:
			fluxobj_withintlums.fluxes()
			self.nuLnus = fluxobj_withintlums.nuLnus4plotting
			self.allnus = fluxobj_withintlums.all_nus_rest
			self.int_lums = fluxobj_withintlums.int_lums
		else:
			fluxobj_4SEDplots.fluxes()
			self.nuLnus = fluxobj_4SEDplots.nuLnus4plotting
			self.allnus = fluxobj_4SEDplots.all_nus_rest

	def write_parameters_outputvalues(self, modelfluxes):		


			Mstar, SFR, SFR_file = stellar_info_array(self.chain.flatchain_sorted, data, out['realizations2int'])

			if out['calc_lum']:			
				chain_pars_and_others = np.column_stack((self.chain.flatchain_sorted, Mstar, SFR, SFR_file, SFR_IR))				
				tau_e, age_e, nh_e, irlum_e, SB_e, BB_e, GA_e, TO_e, BBebv_e, GAebv_e,Mstar_e, SFR_e, SFR_file_e, SFR_IR_e = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(chain_pars_and_others, [16, 50, 84],  axis=0)))
				output_error = np.column_stack((tau_e, age_e, nh_e, irlum_e, SB_e, BB_e, GA_e, TO_e, BBebv_e, GAebv_e, Mstar_e, SFR_e, SFR_file_e, SFR_IR_e, int_lums)) 
			else:
				chain_pars_and_others = np.column_stack((self.chain.flatchain_sorted, Mstar, SFR, SFR_file, SFR_IR))
				tau_e, age_e, nh_e, irlum_e, SB_e, BB_e, GA_e, TO_e, BBebv_e, GAebv_e,Mstar_e, SFR_e, SFR_file_e, SFR_IR_e = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(chain_pars_and_others, [16, 50, 84],  axis=0)))
				output_error = np.column_stack((tau_e, age_e, nh_e, irlum_e, SB_e, BB_e, GA_e, TO_e, BBebv_e, GAebv_e, Mstar_e, SFR_e, SFR_file_e)) 

			pars_with_errors_intlums= map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(chain, [16, 50, 84],  axis=0)))

			np.savetxt(folder + str(sourcename)+'/parameters_with_errors_3_'+str(sourcename)+'.txt' , output_error, delimiter = " ",fmt="%1.4f" ,header="tau_e, age_e, nh_e, irlum_e, SB_e, BB_e, GA_e, TO_e, BBebv_e, GAebv_e, Mstar_e, SFR_e, SFR_file_e, L_sb, L_bbb, L_ga, L_to, Lbb_dered, Lfir_SFR ,SFR_IR_e")



		# self.objfluxes.modelfluxes(self.dataobj.dict1, 3)

		
		# objfluxes.intlum(objfluxes.all_nus_rest,objfluxes.nuLnu)



		#print 'p2.5, p16, p50_median, p84, p97.5, pmean, pbest_fit  X 10 par'

	def plot_PDFtriangle_10pars():		

		npar = chain.shape[-1]
		figure = triangle.corner(chain, labels=[r"$\tau$", r"Age", r"N$_H$", r"L$_{IR}$", r"SB", r"BB", r"GA", r"TO", r"E(B-V)$_\mathrm{BB}$", r"E(B-V)$_{GAL}$"], plot_contours=True, plot_datapoints = False, show_titles=True, quantiles=[0.16, 0.50, 0.84])#, truths = best_fit_par)#truths = [2 , 5e9, 22.7 , 13.26, -16.1, -55.36, -9.23, 47.3, 0.01, 0.01 ]) 
		return figure


	def plot_PDFtriangle_intlums(self, ):		

		figure = triangle.corner(chain, labels=[r"L$_{SB (MIR)}$",r"L$_{BB (OPT-UV)}$",r"L$_{GA (OPT-UV)}$", r"L$_{TO (MIR)}$"],   plot_contours=True, plot_datapoints = False, show_titles=True, quantiles=[0.16, 0.50, 0.84])
		return figure

	def plot_PDFtriangle_stellar():	


		print 'plot'

	def plot_bestfit_SED():		

		# xpb, ypb = objfluxes.4plot_bestfit(objfluxes.all_nus_rest,objfluxes.nuLnu)


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


		print 'plot'

	def plot_manyrealizations_SED():	

		# xmr, ymr = objfluxes.4plot_manyrealizations(objfluxes.all_nus_rest,objfluxes.nuLnu)


		self.fluxes_plot = FLUXES_ARRAYS.fluxes(self.chain, self.out, 'plot' )

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





"""=========================================================="""





class CHAIN:

	"""
	Class CHAIN

	##input: 
	- name of file, where chain was saved
	- dictionary of ouput setting: out

	##bugs: 

	"""     

	def __init__(self, outputfilename, out):
			self.outputfilename = outputfilename
			self.out = out

	def props(self):
		if os.path.lexists(self.outputfilename):

			f = open(self.outputfilename, 'rb')
			samples = pickle.load(f)
			f.close()

			self.nwalkers, self.nsamples, self.npar = samples['chain'].shape

			Ns, Nt = self.out['Nsample'], self.out['Nthinning']		
			lnprob = samples['lnprob'][:,0:Ns*Nt:Nt].ravel()

			isort = (- lnprob).argsort() #sort parameter vector for likelihood
			lnprob_sorted = np.reshape(lnprob[isort],(-1,1))
			self.lnprob_max = lnprob_sorted[0]


			self.flatchain = samples['chain'][:,0:Ns*Nt:Nt,:].reshape(-1, self.npar)
			self.parametersoutput = [self.flatchain[:,1] for i in range(self.npar)]

			chain_length = int(len(self.flatchain))

			self.flatchain_sorted = self.flatchain[isort]
			self.best_fit_pars = self.flatchain[isort[0]]


			print '_________________________________'
			print 'Some properties of the sampling:'
			mean_accept =  samples['accept'].mean()
			print '- Mean acceptance fraction', mean_accept
			mean_autocorr = samples['acor'].mean()
			print '- Mean autocorrelation time', mean_autocorr

		else:
			'Error: The sampling has not been perfomed yet, or the chains were not saved properly.'
	
	def write_totalchain():

		print 'later'


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




"""=========================================================="""



class FLUXES_ARRAYS:

	"""
	This class constructs the luminosities arrays for many realizations from the parameter values
	Outout is return by FLUXES_ARRAYS.fluxes() 
	and depends on which is the output product being produced, set by self.output_type.
	
	## inputs:
	- object of class CHAIN
	- dictionary of output settings, out
	- str giving output_type: ['plot','intlum', 'bestfit']

	## output:
	- frequencies and nuLnus + ['filteredpoints', 'integrated luminosities', - ]
	"""


	def __init__(self, chain_obj, out, output_type):
		self.chain_obj = chain_obj
		self.output_type = output_type
		self.out = out

	def fluxes(self, dict_modelsfiles, filterdict, dict_modelfluxes):	

		"""
		This is the main function of the class.
		"""
		self.chain_obj.props()
		self.parameters = np.array(self.chain_obj.parametersoutput)

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

		nsample, npar = self.chain_obj.flatchain.shape
		source = self.chain_obj.data.name

		#THIS NEEDS TO BE CHANGED INTO RANDOMLY
		if self.output_type == 'plot':
			tau, agelog, nh, irlum, SB ,BB, GA,TO, BBebv0, GAebv0= self.parameters[:,0:nsample:nsample/(self.out['realizations2plot'])]
		elif self.output_type == 'intlum':
			tau, agelog, nh, irlum, SB ,BB, GA,TO, BBebv0, GAebv0= self.parameters[:,0:nsample:nsample/(self.out['realizations2int'])]
		elif self.output_type == 'best_fit':
			tau, agelog, nh, irlum, SB ,BB, GA,TO, BBebv0, GAebv0= self.best_fit_pars

		age = 10**agelog

		self.all_nus_rest = np.arange(11.5, 16, 0.001) 

		for g in range(len(tau)):

			BBebv1 = model.pick_EBV_grid(EBVbbb_array, BBebv0[g])
			BBebv = (  str(int(BBebv1)) if  float(BBebv1).is_integer() else str(BBebv1))
			GAebv1 = model.pick_EBV_grid(EBVgal_array,GAebv0[g])
			GAebv = (  str(int(GAebv1)) if  float(GAebv1).is_integer() else str(GAebv1))

			SB_filename = self.chain_obj.data.path + model.pick_STARBURST_template(irlum[g], filename_0_starburst, all_irlum)
			GA_filename = self.chain_obj.data.path + model.pick_GALAXY_template(tau[g], age[g], filename_0_galaxy, all_tau, all_age)
			TO_filename = self.chain_obj.data.path + model.pick_TORUS_template(nh[g], all_nh, filename_0_torus)
			BB_filename = self.chain_obj.data.path + model.pick_BBB_template()


			gal_nu, gal_nored_Fnu = model.GALAXY_read_4plotting( GA_filename, all_nus_rest)
			gal_nu, gal_Fnu_red = model.GALAXY_nf2( gal_nu, gal_nored_Fnu, float(GAebv) )
			all_gal_nus, all_gal_Fnus =gal_nu, gal_Fnu_red

			sb_nu0, sb_Fnu0 = model.STARBURST_read_4plotting(SB_filename, all_nus_rest)
			all_sb_nus, all_sb_Fnus = sb_nu0, sb_Fnu0

			bbb_nu, bbb_nored_Fnu = model.BBB_read_4plotting(BB_filename, all_nus_rest)
			bbb_nu0, bbb_Fnu_red = model.BBB_nf2(bbb_nu, bbb_nored_Fnu,float(BBebv), self.chain_obj.data.z )
			all_bbb_nus, all_bbb_Fnus = bbb_nu0, bbb_Fnu_red
			all_bbb_nus, all_bbb_Fnus_deredd = bbb_nu0, bbb_nored_Fnu

			all_tor_nus, all_tor_Fnus = model.TORUS_read_4plotting(TO_filename, self.chain_obj.data.z, all_nus_rest)


			if self.output_type == 'plot':
				par2 = tau[g], agelog[g], nh[g], irlum[g], SB[g] ,BB[g], GA[g] ,TO[g], float(BBebv), float(GAebv)
				filtered_modelpoints = parspace.ymodel(self.chain_obj.data.nus,self.chain_obj.data.z, dict_modelsfiles, dict_modelfluxes, *par2)


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
		self.nuLnus4plotting = self.FLUXES2nuLnu_4plotting(all_nus_rest, FLUXES4plotting, self.chain_obj.data.z)

		if self.output_type == 'plot':
			self.filtered_modelpoints_nuLnu = self.FLUXES2nuLnu_4plotting(all_nus_rest,  filtered_modelpoints, self.chain_obj.data.z)
			
		elif self.output_type == 'intlum':
			self.int_lums= np.log10(self.integrated_luminosities(self.out ,all_nus_rest, nuLnus4plotting))

		elif self.output_type == 'best_fit':

	def FLUXES2nuLnu_4plotting(self, all_nus_rest, FLUXES4plotting, z):

		"""
		Converts FLUXES4plotting into nuLnu_4plotting.

		##input: 
		- all_nus_rest (give in 10^lognu, not log.)
		- FLUXES4plotting : fluxes for the four models corresponding
							to each element of the total chain
		- source redshift z					
		"""
		all_nus_obs = all_nus_rest /(1+z) 
		distance= model.z2Dlum(z)
		lumfactor = (4. * math.pi * distance**2.)

		SBnuLnu, BBnuLnu, GAnuLnu, TOnuLnu, TOTALnuLnu, BBnuLnu_deredd = [ f *lumfactor*all_nus_obs for f in FLUXES4plotting]

		return SBnuLnu, BBnuLnu, GAnuLnu, TOnuLnu, TOTALnuLnu, BBnuLnu_deredd


	def integrated_luminosities(self,out ,all_nus_rest, nuLnus4plotting):

		"""
		Calculates the integrated luminosities for 
		all model templates chosen by the user in 
		out['intlum_models'], 
		within out['intlum_freqranges'].

		##input: 
		- settings out
		- all_nus_rest
		- nuLnus4plotting: nu*luminosities for the four models corresponding
							to each element of the total chain
		"""

		SBnuLnu, BBnuLnu, GAnuLnu, TOnuLnu, TOTALnuLnu, BBnuLnu_deredd =nuLnus4plotting
		out['intlum_freqranges'] = (out['intlum_freqranges']*out['intlum_freqranges_unit']).to(u.Hz, equivalencies=u.spectral())
		int_lums = []
		for m in range(len(out['intlum_models'])):

			if out['intlum_models'][m] == 'sb':	
				nuLnu= SBnuLnu
			elif out['intlum_models'][m] == 'bbb':	
				nuLnu= BBnuLnu
			elif out['intlum_models'][m] == 'bbbdered':	
				nuLnu=BBnuLnu_deredd
			elif out['intlum_models'][m] == 'gal':	
			 	nuLnu=GAnuLnu
			elif out['intlum_models'][m] == 'tor':	
			 	nuLnu=TOnuLnu
		
#			print all_nus_rest, np.log10(out['intlum_freqranges'][m][1]),np.log10(out['intlum_freqranges'][m][0])
			index  = ((all_nus_rest >= np.log10(out['intlum_freqranges'][m][1].value)) & (all_nus_rest<= np.log10(out['intlum_freqranges'][m][0].value)))			
			all_nus_rest_int = 10**(all_nus_rest[index])
#			print np.shape(nuLnu), np.shape(nuLnu[:,index])
			Lnu = nuLnu[:,index] / all_nus_rest_int
			Lnu_int = scipy.integrate.trapz(Lnu, x=all_nus_rest_int)
			int_lums.append(Lnu_int)

		return np.array(int_lums)



if __name__ == 'main':
    main(sys.argv[1:])

