

"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      PLOTandWRITE_AGNfitter.py

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

This script contains all functions used in order to visualize the output of the sampling.
Plotting and writing. 
This function need to have the output files samples_mcmc.sav and samples_bur-in.sav.
"""
#PYTHON IMPORTS
import matplotlib.pyplot as plt
from matplotlib import rc, ticker
#matplotlib.use('Agg')
import sys, os
import math 
import numpy as np
import triangle #Author: Dan Foreman-Mackey (danfm@nyu.edu)
import time
import scipy
from astropy import units as u
from astropy import constants as const

#AGNfitter IMPORTS
import GENERAL_AGNfitter as general
import MODEL_AGNfitter2 as model
import DICTIONARIES_AGNfitter as dicts
import PARAMETERSPACE_AGNfitter as parspace
import cPickle



def main(data, P, out):


	"""
	Main function of PLOTandWRITE_AGNfitter.

	##input:

	- data object
	- parameter space settings dictionary P
	- output settings-dictionary out
	
	"""

	chain_burnin = CHAIN(data.output_folder+str(data.name)+ '/samples_burnin.sav', out)
	chain_mcmc = CHAIN(data.output_folder+str(data.name)+ '/samples_mcmc.sav',  out)	
	output = OUTPUT(chain_mcmc, data)

	if out['plot_tracesburn-in']:
		fig, nplot=chain_burnin.plot_trace()
		fig.suptitle('Chain traces for %i of %i walkers.' % (nplot,chain_burnin.nwalkers))
		fig.savefig(data.output_folder+str(data.name)+'/traces_burin.' + out['plot_format'])
		plt.close(fig)

	if out['plot_tracesmcmc']:
		fig, nplot = chain_mcmc.plot_trace()
		fig.suptitle('Chain traces for %i of %i walkers. Main acceptance fraction: %f' % (nplot,chain_burnin.nwalkers))
		fig.savefig(folder+str(sourcename)+'/traces_mcmc.' + out['plot_format'])
		plt.close(fig)

	if out['plot_posteriortriangle'] :
		fig = output.plot_PDFtriangle('10pars', P.names)
		fig.savefig(data.output_folder+str(data.name)+'/PDFtriangle_10pars.' + out['plot_format'])
		plt.close(fig)

	if out['plot_posteriortrianglewithluminosities']: 
		fig = output.plot_PDFtriangle('int_lums', out['intlum_names']) 
		fig.savefig(data.output_folder+str(data.name)+'/PDFtriangle_intlums.' + out['plot_format'])
		plt.close(fig)

	if out['writepar_meanwitherrors']:
		outputvalues, outputvalues_header = output.write_parameters_outputvalues(P)
		comments_ouput= ' # Output for source ' +str(data.name) + '\n' +' Rows are: 2.5, 16, 50, 84, 97.5 percentiles # '+'\n'+ '-----------------------------------------------------'+'\n' 
		np.savetxt(data.output_folder + str(data.name)+'/parameter_outvalues_'+str(data.name)+'.txt' , outputvalues, delimiter = " ",fmt= "%1.4f" ,header= outputvalues_header, comments =comments_ouput)


	if out['plotSEDbest']:
		output.plot_bestfit_SED()

	if out['plotSEDrealizations']:
		fig = output.plot_manyrealizations_SED()
		fig.savefig(data.output_folder+str(data.name)+'/SED_manyrealizations_' +str(data.name)+ '.'+out['plot_format'])
		plt.close(fig)





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
		self.chain.props()

		self.out = chain_obj.out
		self.data = data_obj
		fluxobj_withintlums = FLUXES_ARRAYS(chain_obj, self.out,'int_lums')
		fluxobj_4SEDplots = FLUXES_ARRAYS(chain_obj, self.out,'plot')

		if self.out['calc_intlum']:
			fluxobj_withintlums.fluxes( self.data)
			self.nuLnus = fluxobj_withintlums.nuLnus4plotting
			self.allnus = fluxobj_withintlums.all_nus_rest
			self.int_lums = fluxobj_withintlums.int_lums
		else:
			fluxobj_4SEDplots.fluxes(self.data)
			self.nuLnus = fluxobj_4SEDplots.nuLnus4plotting
			self.allnus = fluxobj_4SEDplots.all_nus_rest

	def write_parameters_outputvalues(self, P):		


		Mstar, SFR_opt, _ = model.stellar_info_array(self.chain.flatchain_sorted, self.data, self.out['realizations2int'])
		column_names = np.transpose(np.array(["P025","P16","P50","P84","P975"], dtype='|S3'))
		chain_pars = np.column_stack((self.chain.flatchain_sorted, Mstar, SFR_opt))		
											# np.mean(chain_pars, axis[0]),
											# np.std(chain_pars, axis[0]),
		if self.out['calc_intlum']:			


			SFR_IR = model.sfr_IR(self.int_lums[0]) #check that ['intlum_names'][0] is always L_IR(8-100)
			
			chain_others =np.column_stack((self.int_lums.T, SFR_IR))
			outputvalues = np.column_stack((np.transpose(map(lambda v: (v[0],v[1],v[2],v[3],v[4]), zip(*np.percentile(chain_pars, [2.5,16, 50, 84,97.5], axis=0)))),
											np.transpose(map(lambda v: (v[0],v[1],v[2],v[3],v[4]), zip(*np.percentile(chain_others, [2.5,16, 50, 84,97.5], axis=0))))											)) 


	
			outputvalues_header= ' '.join([ i for i in np.hstack((P.names, 'Mstar', 'SFR_opt', self.out['intlum_names'], 'SFR_IR',))] )

		else:
			outputvalues = np.column_stack((map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(chain_pars, [16, 50, 84],  axis=0))))) 
			outputvalues_header=' '.join( [ i for i in P.names] )
		return outputvalues, outputvalues_header

	def plot_PDFtriangle(self,parameterset, labels):		

		if parameterset=='10pars':
			figure = triangle.corner(self.chain.flatchain, labels= labels, plot_contours=True, plot_datapoints = False, show_titles=True, quantiles=[0.16, 0.50, 0.84])
		elif parameterset == 'int_lums':
			figure = triangle.corner(self.int_lums.T, labels= labels,   plot_contours=True, plot_datapoints = False, show_titles=True, quantiles=[0.16, 0.50, 0.84])
		return figure


	def plot_manyrealizations_SED(self):	


		source = self.data.name
		data_nus= self.data.nus
		ydata = self.data.fluxes
		yerror = self.data.fluxerrs
		z = self.data.z
		all_nus =self.allnus
		Nrealizations = self.out['realizations2plot']

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
		SBcolor, BBcolor, GAcolor, TOcolor, TOTALcolor= SED_colors(combination = 'a')
		lw= 1.5

		SBnuLnu, BBnuLnu, GAnuLnu, TOnuLnu, TOTALnuLnu, BBnuLnu_deredd = self.nuLnus
		thinning_4plot = len(TOTALnuLnu) / (Nrealizations)



		for i in range(Nrealizations):

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

		print 'notyet'




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
			samples = cPickle.load(f)
			f.close()

			self.chain = samples['chain']
			nwalkers, nsamples, npar = samples['chain'].shape

			Ns, Nt = self.out['Nsample'], self.out['Nthinning']		
			self.lnprob = samples['lnprob']
			self.lnprob_flat = samples['lnprob'][:,0:Ns*Nt:Nt].ravel()

			isort = (- self.lnprob_flat).argsort() #sort parameter vector for likelihood
			lnprob_sorted = np.reshape(self.lnprob_flat[isort],(-1,1))
			self.lnprob_max = lnprob_sorted[0]


			self.flatchain = samples['chain'][:,0:Ns*Nt:Nt,:].reshape(-1, npar)
#			self.parametersoutput = [self.flatchain[:,1] for i in range(npar)]

			chain_length = int(len(self.flatchain))

			self.flatchain_sorted = self.flatchain[isort]
			self.best_fit_pars = self.flatchain[isort[0]]


			print '_________________________________'
			print 'Some properties of the sampling:'
			self.mean_accept =  samples['accept'].mean()
			print '- Mean acceptance fraction', self.mean_accept
			self.mean_autocorr = samples['acor'].mean()
			print '- Mean autocorrelation time', self.mean_autocorr

		else:

			'Error: The sampling has not been perfomed yet, or the chains were not saved properly.'
	
	def write_totalchain():

		print 'later'



	def plot_trace(self, P, nwplot=50):

		""" Plot the sample trace for a subset of walkers for each parameter.
		"""

		#-- Latex -------------------------------------------------
		rc('text', usetex=True)
		rc('font', family='serif')
		rc('axes', linewidth=1.5)
		#-------------------------------------------------------------

		self.nwalkers, nsample, npar = self.chain.shape
		nrows = npar + 1
		ncols =1         
		fig, axes = fig_axes(nrows, ncols, npar+1)

		nwplot = min(nsample, nwplot)
		for i in range(npar):
			ax = axes[i]
			for j in range(0, self.nwalkers, max(1, self.nwalkers // nwplot)):
				ax.plot(self.chain[j,:,i], lw=0.5,  color = 'black', alpha = 0.3)
			ax.set_title(r'\textit{Parameter : }'+P.names[i], fontsize=12)  
			ax.set_xlabel(r'\textit{Steps}', fontsize=12)
			ax.set_ylabel(r'\textit{Walkers}',fontsize=12)

		ax = axes[-1]
		for j in range(0, self.nwalkers, max(1, self.nwalkers // nwplot)):
			ax.plot(self.lnprob[j,:], lw=0.5, color = 'black', alpha = 0.3)
		ax.set_title(r'\textit{Likelihood}', fontsize=12)   
		ax.set_xlabel(r'\textit{Steps}', fontsize=12)
		ax.set_ylabel(r'\textit{Walkers}',fontsize=12)

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

	def fluxes(self, data):	

		"""
		This is the main function of the class.
		"""
		self.chain_obj.props()

		SBFnu_list = []
		BBFnu_list = []
		GAFnu_list= []
		TOFnu_list = []
		TOTALFnu_list = []
		BBFnu_deredd_list = []
		if self.output_type == 'plot':
			filtered_modelpoints_list = []


		gal_do,  irlum_dict, nh_dict, BBebv_dict = data.dictkey_arrays
		# Take the last 4 dictionaries, which are for plotting. (the first 4 were at bands)
		_,_,_,_,STARBURSTFdict , BBBFdict, GALAXYFdict, TORUSFdict= data.dict_modelfluxes

		nsample, npar = self.chain_obj.flatchain.shape
		source = data.name

		if self.output_type == 'plot':
			tau, agelog, nh, irlum, SB ,BB, GA,TO, BBebv, GAebv= self.chain_obj.flatchain[np.random.choice(nsample, (self.out['realizations2plot'])),:].T
		elif self.output_type == 'int_lums':
			tau, agelog, nh, irlum, SB ,BB, GA,TO, BBebv, GAebv= self.chain_obj.flatchain[np.random.choice(nsample, (self.out['realizations2int'])),:].T
		elif self.output_type == 'best_fit':
			tau, agelog, nh, irlum, SB ,BB, GA,TO, BBebv, GAebv= self.best_fit_pars

		age = 10**agelog

		self.all_nus_rest = np.arange(11.5, 16, 0.001) 

		for g in range(len(tau)):


			# Pick dictionary key-values, nearest to the MCMC- parameter values
			irlum_dct = model.pick_STARBURST_template(irlum[g], irlum_dict)
			nh_dct = model.pick_TORUS_template(nh[g], nh_dict)
			ebvbbb_dct = model.pick_BBB_template(BBebv[g], BBebv_dict)
			gal_do.nearest_par2dict(tau[g], age[g], GAebv[g])
			tau_dct, age_dct, ebvg_dct=gal_do.t, gal_do.a,gal_do.e

			#Produce model fluxes at all_nus_rest for plotting, through interpolation
			all_gal_nus, gal_Fnus = GALAXYFdict[tau_dct, age_dct,ebvg_dct]   
			GAinterp = scipy.interpolate.interp1d(all_gal_nus, gal_Fnus, bounds_error=False, fill_value=0.)
			all_gal_Fnus = GAinterp(self.all_nus_rest)

			all_sb_nus, sb_Fnus= STARBURSTFdict[irlum_dct] 
			SBinterp = scipy.interpolate.interp1d(all_sb_nus, sb_Fnus, bounds_error=False, fill_value=0.)
			all_sb_Fnus = SBinterp(self.all_nus_rest)

			all_bbb_nus, bbb_Fnus = BBBFdict[ebvbbb_dct] 
			BBinterp = scipy.interpolate.interp1d(all_bbb_nus, bbb_Fnus, bounds_error=False, fill_value=0.)
			all_bbb_Fnus = BBinterp(self.all_nus_rest)

			all_bbb_nus, bbb_Fnus_deredd = BBBFdict['0']
			BBderedinterp = scipy.interpolate.interp1d(all_bbb_nus, bbb_Fnus_deredd, bounds_error=False, fill_value=0.)
			all_bbb_Fnus_deredd = BBderedinterp(self.all_nus_rest)

			all_tor_nus, tor_Fnus= TORUSFdict[nh_dct]
			TOinterp = scipy.interpolate.interp1d(all_tor_nus, np.log10(tor_Fnus), bounds_error=False, fill_value=0.)
			all_tor_Fnus = 10**(TOinterp(self.all_nus_rest))		


			if self.output_type == 'plot':
				par2 = tau[g], agelog[g], nh[g], irlum[g], SB[g] ,BB[g], GA[g] ,TO[g], BBebv[g], GAebv[g]
				filtered_modelpoints = parspace.ymodel(data.nus,data.z, data.dictkey_arrays, data.dict_modelfluxes, *par2)
				

			#Using the costumized normalization 
			SBFnu =   (all_sb_Fnus /1e20) *10**float(SB[g]) 
			BBFnu = (all_bbb_Fnus /1e60) * 10**float(BB[g]) 
			GAFnu =   (all_gal_Fnus/ 1e18) * 10**float(GA[g]) 
			TOFnu =   (all_tor_Fnus/  1e-40) * 10**float(TO[g])
			BBFnu_deredd = (all_bbb_Fnus_deredd /1e60) * 10**float(BB[g])

			TOTALFnu =  SBFnu + BBFnu + GAFnu + TOFnu
			
			#Append to the list for all realizations
			SBFnu_list.append(SBFnu)
			BBFnu_list.append(BBFnu)
			GAFnu_list.append(GAFnu)
			TOFnu_list.append(TOFnu)
			TOTALFnu_list.append(TOTALFnu)
			BBFnu_deredd_list.append(BBFnu_deredd)
			#Only if SED plotting: do the same with the  modelled flux values at each data point 
			if self.output_type == 'plot':
				filtered_modelpoints_list.append(filtered_modelpoints)


		#Convert lists into Numpy arrays
		SBFnu_array = np.array(SBFnu_list)
		BBFnu_array = np.array(BBFnu_list)
		GAFnu_array = np.array(GAFnu_list)
		TOFnu_array = np.array(TOFnu_list)
		TOTALFnu_array = np.array(TOTALFnu_list)
		BBFnu_array_deredd = np.array(BBFnu_deredd_list)    

		#Put them all together to transport
		FLUXES4plotting = (SBFnu_array, BBFnu_array, GAFnu_array, TOFnu_array, TOTALFnu_array,BBFnu_array_deredd)
		#Convert Fluxes to nuLnu
		self.nuLnus4plotting = self.FLUXES2nuLnu_4plotting(self.all_nus_rest, FLUXES4plotting, data.z)

		#Only if SED plotting:
		if self.output_type == 'plot':
			filtered_modelpoints = np.array(filtered_modelpoints_list)
			distance= model.z2Dlum(data.z)
			lumfactor = (4. * math.pi * distance**2.)
			self.filtered_modelpoints_nuLnu = filtered_modelpoints *lumfactor* 10**(data.nus)
		
		#Only if calculating integrated luminosities:	
		elif self.output_type == 'int_lums':
			self.int_lums= np.log10(self.integrated_luminosities(self.out ,self.all_nus_rest, self.nuLnus4plotting))
		# elif self.output_type == 'best_fit':
		# 	self.filtered_modelpoints_nuLnu = self.FLUXES2nuLnu_4plotting(all_nus_rest,  filtered_modelpoints, self.chain_obj.data.z)


	def FLUXES2nuLnu_4plotting(self, all_nus_rest, FLUXES4plotting, z):

		"""
		Converts FLUXES4plotting into nuLnu_4plotting.

		##input: 
		- all_nus_rest (give in 10^lognu, not log.)
		- FLUXES4plotting : fluxes for the four models corresponding
							to each element of the total chain
		- source redshift z					
		"""

		all_nus_obs = 10**all_nus_rest /(1+z) 
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
		
			index  = ((all_nus_rest >= np.log10(out['intlum_freqranges'][m][1].value)) & (all_nus_rest<= np.log10(out['intlum_freqranges'][m][0].value)))			
			all_nus_rest_int = 10**(all_nus_rest[index])
			Lnu = nuLnu[:,index] / all_nus_rest_int
			Lnu_int = scipy.integrate.trapz(Lnu, x=all_nus_rest_int)
			int_lums.append(Lnu_int)

		return np.array(int_lums)



"""
Some stand-alone functions
"""


def fig_axes(nrows, ncols, npar, width=13):
	fig = plt.figure(figsize=(width, width*1.6))#*nrows/ncols))    
	fig.subplots_adjust(hspace=0.9)
	axes = [fig.add_subplot(nrows, ncols, i+1) for i in range(npar)]
	return fig, axes



def SED_plotting_settings(x, ydata):

	"""
	This function produces the setting for the figures for SED plotting.
	**Input:
	- all nus, and data (to make the plot limits depending on the data)
	"""
	fig = plt.figure()

	ax1 = fig.add_subplot(111)
	x2 = (2.98e8) / x / (1e-6) # Wavelenght axis
	ax2 = ax1.twiny()
	ax2.plot(x2, np.ones(len(x2)), alpha=0)

	#-- Latex -------------------------------------------------
	rc('text', usetex=True)
	rc('font', family='serif')
	rc('axes', linewidth=2)
	#-------------------------------------------------------------

	#    ax1.set_title(r"\textbf{SED of Type 2}" + r"\textbf{ AGN }"+ "Source Nr. "+ source + "\n . \n . \n ." , fontsize=17, color='k')    
	ax1.set_xlabel(r'rest-frame frequency$\mathbf{log \  \nu} [\mathtt{Hz}] $', fontsize=11)
	ax2.set_xlabel(r'rest-frame wavelength $\mathbf{\lambda} [\mathtt{\mu m}] $', fontsize=11)
	ax1.set_ylabel(r'luminosity $\mathbf{\nu L(\nu) [\mathtt{erg \ } \mathtt{ s}^{-1}]}$',fontsize=11)

	ax1.set_autoscalex_on(True) 
	ax1.set_autoscaley_on(True) 
	ax1.set_xscale('linear')
	ax1.set_yscale('log')


	mediandata = np.median(ydata)
	ax1.set_ylim(mediandata /50.,mediandata * 50.)

	ax2.set_xscale('log')
	ax2.set_yscale('log')
	ax2.set_ylim( mediandata /50., mediandata * 50.)

	ax2.set_xticks([100, 10,1, 0.1]) 
	ax2.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

	return fig, ax1, ax2

def SED_colors(combination = 'a'):

	if combination=='a':   
		steelblue = '#4682b4'
		darkcyan ='#009acd'
		deepbluesky = '#008b8b'
		seagreen = '#2E8B57'    
		lila = '#68228B'
		darkblue='#123281'

	return seagreen, darkblue, 'orange', lila, 'red'





if __name__ == 'main':
    main(sys.argv[1:])


