
"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    DICTIONARIES_AGNFitter.py

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

This script contains all functions which are needed to construct the total model of AGN. 


"""
import numpy as np
import math
from collections import defaultdict

import MODEL_AGNfitter2 as model
import time
import cPickle
import shelve
from astropy import units as u 



class MODELSDICT:


	"""
	Class MODELSDICT

	##input: 
	- 

	##bugs: 

	"""     

	def __init__(self, filename, path):
		self.filename = filename
		self.path=path
		self.ebvgal_array = np.array(np.arange(0.,100.,5.)/100)

	def build(self):

		f = open(self.filename, 'wb')

		COSMOS_modelsdict = dict()
		for z in frange(0.3,0.9,0.3):#use this range to make the 'z_array_in_dict' in RUN_AGNfitter.
			print' REDSHIFT', z
			z_key = str(z)
			    #Bands with band filters. Write down the version of the band&filterset. 'version1' includes all COSMOS bands. 
			    #You can create a new band&filterset in fct 'filter_dictionaries(filterset)' in this same script.				
			filterset='BOOTES_FIR'		
			filterdict = filter_dictionaries(filterset, self.path)
			dict_modelsfiltered = self.construct_dictionaryarray_filtered(z, filterdict, self.path)
			COSMOS_modelsdict[z_key] = dict_modelsfiltered
			    	
			print 'Dictionary has been created in :', self.filename

		cPickle.dump(COSMOS_modelsdict, f, protocol=2)
		f.close()




	def arrays_of_modelparsandfiles(self,path): #!!!!

	  	"""

		This function contains the file information and all the model parameter matrix available 

		## inputs:
		- 
		## output:
		- array of values available in the grid for each parameter (elements starting with prefix 'all_')
		- array of file names corresponding to each value in the grid (elements starting with prefix 'filename_0_')

	    #this can include index reading

		"""

		#galaxy
		gal_list = path +'models/GALAXY/input_template_hoggnew.dat'
		filename_0_galaxy= np.genfromtxt(gal_list, usecols=(0), dtype = ('S') ) #all galaxy file names
		all_tau, all_age =  np.loadtxt(gal_list, usecols=(2,3),unpack= True) #all taus and ages

		#starburst
		sb_list1= path+'models/STARBURST/DALE.list' 
		sb_list2= path +'models/STARBURST/CHARY_ELBAZ.list'
		filename_01= np.genfromtxt(sb_list1, usecols=(0), dtype = ('S') )
		ir_lum_01= np.genfromtxt(sb_list1, usecols=(2), unpack= True)
		filename_02= np.genfromtxt(sb_list2, usecols=(0), dtype = ('S') )
		ir_lum_02= np.genfromtxt(sb_list2, usecols=(2), unpack= True)

		filename_0_starburst =np.hstack((filename_01,filename_02))
		all_irlum = np.hstack((ir_lum_01, ir_lum_02))

		#torus
		tor_list = path +'models/TORUS/torus_templates_list.dat'
		all_nh, filename_0_torus= np.loadtxt(tor_list , usecols=(0,1), unpack=True, dtype = ('S'))

		return all_tau, all_age, all_nh, all_irlum, filename_0_galaxy, filename_0_starburst, filename_0_torus


	def construct_dictionaryarray_filtered(self, z, filterdict,path):

		GALAXYFdict_filtered = dict()
		STARBURSTFdict_filtered = dict()		
		BBBFdict_filtered = dict()
		TORUSFdict_filtered = dict()

		#Two othere dictionaries, which are not leaving the function
		GALAXY_nored_Fdict = dict()
		BBB_nored_Fdict = dict()

	#OPENING TEMPLATES AND BUILDING DICTIONARIES

	#LISTS OF MODEL TEMPLATES
		
		_,_,_,_, filename_0_galaxy, filename_0_starburst, filename_0_torus = self.arrays_of_modelparsandfiles(path)

		galaxy_object = cPickle.load(file(path + 'models/GALAXY/bc03_v1.pickle', 'rb')) 

		_, ageidx, tauidx, _, _,_ =  np.shape(galaxy_object.SED)

		for taui in range(tauidx):
			for agei in range(ageidx):

				gal_wl, gal_Fwl =  galaxy_object.wave, galaxy_object.SED[:,agei,taui,:,:,:].squeeze()
				gal_nus= gal_wl.to(u.Hz, equivalencies=u.spectral())
				gal_Fnu= gal_Fwl * 3.34e-19 * gal_wl**2.  
				#converting from Flambda to Fnu

				for EBV_gal in self.ebvgal_array:
				
					#Absorption
					gal_nu, gal_Fnu_red = model.GALAXY_nf2( gal_nus.value[0:len(gal_nus):100], gal_Fnu.value[0:len(gal_nus):100], EBV_gal)	
					#Filter: THIS IS EXPENSIVE!
					bands,  gal_Fnu_filtered =  model.filters1(np.log10(gal_nu), gal_Fnu_red, filterdict, z)	
					EBV_gal_key = str(EBV_gal)			
					GALAXYFdict_filtered[str(galaxy_object.tau.value[taui]),str(galaxy_object.tg.value[agei]), EBV_gal_key] = bands, gal_Fnu_filtered
	
#					if np.amax(gal_Fnu_filtered) == 0:
#						print 'Error: something is wrong in the calculation of GALAXY flux'
		print 'GALAXY done'



		for i in range(len(filename_0_starburst)):
			SB_filename = filename_0_starburst[i]
			sb_nu0, sb_Fnu0 = model.STARBURST_read(path + 'models/STARBURST/'+SB_filename)
			bands, sb_Fnu_filtered = model.STARBURST2(sb_nu0, sb_Fnu0, z, filterdict)
			STARBURSTFdict_filtered['models/STARBURST/'+SB_filename] = bands, sb_Fnu_filtered
			if np.amax(sb_Fnu_filtered) == 0:
				print 'Error: something is wrong in the calculation of STARBURST flux'
		
		print  'STARBUST done'	



		for i in range(1):
			BB_filename = 'models/BBB/richardsbbb.dat'
			bbb_nu, bbb_Fnu = model.BBB_read(path +BB_filename)
			BBB_nored_Fdict[BB_filename] = bbb_nu,bbb_Fnu
			EBVbbb_array= []
			for (EBV_bbb) in frange(0,1,0.1):
				bbb_nu, bbb_nored_Fnu = BBB_nored_Fdict[BB_filename]
				bbb_nu0, bbb_Fnu_red = model.BBB_nf2(bbb_nu, bbb_nored_Fnu, EBV_bbb, z )
				bands, bbb_Fnu_filtered = model.BBB2(bbb_nu0, bbb_Fnu_red, filterdict,z)
				EBV_bbb_key = str(EBV_bbb)
				EBVbbb_array.append(EBV_bbb) #to have the list written down
				BBBFdict_filtered[BB_filename, EBV_bbb_key] = bands, bbb_Fnu_filtered
				if np.amax(bbb_Fnu_filtered) == 0:
					print 'Error: something is wrong in the calculation of BBB flux'
			
		print 'BBB done'
		for i in range(len(filename_0_torus)):
			TO_filename = filename_0_torus[i]
			tor_nu0, tor_Fnu0 = model.TORUS_read(path +TO_filename, z)
			bands, tor_Fnu_filtered = model.TORUS2(tor_nu0, tor_Fnu0, z, filterdict)
			TORUSFdict_filtered[TO_filename] = bands, tor_Fnu_filtered
			if np.amax(tor_Fnu_filtered) == 0:
				print 'Error: something is wrong in the calculation of TORUS flux'


		print 'TORUS done'


		EBVbbb_array = np.array(EBVbbb_array)



		return STARBURSTFdict_filtered , BBBFdict_filtered, GALAXYFdict_filtered, TORUSFdict_filtered, EBVbbb_array, self.ebvgal_array





# def stack_all_model_nus(filename_0_galaxy, filename_0_starburst, filename_0_torus, z, path):
#     GA_filename = filename_0_galaxy[0]
#     gal_nu0, _ = model.GALAXY_read(path+'models/GALAXY/'+GA_filename)
#     BB_filename = 'models/BBB/richardsbbb.dat'
#     bbb_nu0, _ = model.BBB_read(path+BB_filename)
#     TO_filename = filename_0_torus[0]
#     tor_nu0, _ = model.TORUS_read(path+TO_filename,z)
#     SB_filename = filename_0_starburst[0]
#     sb_nu0,_ = model.STARBURST_read(path+'models/STARBURST/'+SB_filename)

#     all_models_nus0 = np.hstack((sb_nu0,gal_nu0,bbb_nu0,tor_nu0))
#     indexes = all_models_nus0.argsort()
#     all_models_nus = all_models_nus0[indexes]
#     return all_models_nus





# def combine_all_nus(model_nu, model_Fnu_red,bands, model_Fnu_filtered ):

#     all_model_nus0 = np.hstack((model_nu, bands))
#     all_model_Fnus0 = np.hstack((model_Fnu_red, model_Fnu_filtered) )

#     indixes = all_model_nus0.argsort()
#     all_model_nus = all_model_nus0[indixes]
#     all_model_Fnus = all_model_Fnus0[indixes]						

#     return all_model_nus, all_model_Fnus
 




def filter_dictionaries(filterset, path):

	if filterset == 'COSMOS1':

		bands = [ 12.27250142,  12.47650142,  12.6315,  13.09650142,  13.59050142, 13.72250142,  13.82750142,  13.92950142,  14.14450142, 14.26450142, 14.38150142 , 14.52150142 , 14.59450142  ,14.68250142,  14.74050142, 14.80250142  ,14.82950142 , 14.88450142  ,15.11350142, 15.28650142]

		#INFRAROJO

		H160band_file = path + 'models/FILTERS/HERSCHEL/PACS_160mu.txt'
		H160_lambda, H160_factor =  np.loadtxt(H160band_file, usecols=(0,1),unpack= True)

		H100band_file =path + 'models/FILTERS/HERSCHEL/PACS_100mu.txt'
		H100_lambda, H100_factor =  np.loadtxt(H100band_file, usecols=(0,1),unpack= True)

		M70band_file = path + 'models/FILTERS/SPITZER/mips70.res'
		M70_lambda, M70_factor =  np.loadtxt(M70band_file, usecols=(0,1),unpack= True)

		M24band_file =  path + 'models/FILTERS/SPITZER/mips24.res'
		M24_lambda, M24_factor =  np.loadtxt(M24band_file, usecols=(0,1),unpack= True)

		I4band_file = path + 'models/FILTERS/SPITZER/irac_ch4.res'
		I4_lambda, I4_factor =  np.loadtxt(I4band_file, usecols=(0,1),unpack= True)

		I3band_file =  path + 'models/FILTERS/SPITZER/irac_ch3.res'
		I3_lambda, I3_factor =  np.loadtxt(I3band_file, usecols=(0,1),unpack= True)

		I2band_file = path + 'models/FILTERS/SPITZER/irac_ch2.res'
		I2_lambda, I2_factor =  np.loadtxt(I2band_file, usecols=(0,1),unpack= True)

		I1band_file = path + 'models/FILTERS/SPITZER/irac_ch1.res'
		I1_lambda, I1_factor =  np.loadtxt(I1band_file, usecols=(0,1),unpack= True)

		Kband_file = path + 'models/FILTERS/2MASS/Ks_2mass.res'
		K_lambda, K_factor =  np.loadtxt(Kband_file, usecols=(0,1),unpack= True)

		Hband_file = path + 'models/FILTERS/2MASS/H_2mass.res'
		H_lambda, H_factor =  np.loadtxt(Hband_file, usecols=(0,1),unpack= True)

		Jband_file = path + 'models/FILTERS/2MASS/J_2mass.res'
		J_lambda, J_factor =  np.loadtxt(Jband_file, usecols=(0,1),unpack= True)

		zband_file =path + 'models/FILTERS/SUBARU/z_subaru.res'
		z_lambda, z_factor =  np.loadtxt(zband_file, usecols=(0,1),unpack= True)

		iband_file = path + 'models/FILTERS/CHFT/i_megaprime_sagem.res'
		i_lambda, i_factor =  np.loadtxt(iband_file, usecols=(0,1),unpack= True)

		rband_file = path + 'models/FILTERS/SUBARU/r_subaru.res'
		r_lambda,r_factor =  np.loadtxt(rband_file, usecols=(0,1),unpack= True)

		Vband_file = path + 'models/FILTERS/SUBARU/V_subaru.res'
		V_lambda, V_factor =  np.loadtxt(Vband_file, usecols=(0,1),unpack= True)

		gband_file =path + 'models/FILTERS/SUBARU/g_subaru.res'
		g_lambda,g_factor =  np.loadtxt(gband_file, usecols=(0,1),unpack= True)

		Bband_file = path + 'models/FILTERS/SUBARU/B_subaru.res'
		B_lambda, B_factor =  np.loadtxt(Bband_file, usecols=(0,1),unpack= True)

		uband_file = path + 'models/FILTERS/CHFT/u_megaprime_sagem.res'
		u_lambda, u_factor =  np.loadtxt(uband_file, usecols=(0,1),unpack= True)

		NUVband_file = path + 'models/FILTERS/GALEX/galex2500.res'
		NUV_lambda, NUV_factor =  np.loadtxt(NUVband_file, usecols=(0,1),unpack= True)

		FUVband_file = path + 'models/FILTERS/GALEX/galex1500.res'
		FUV_lambda, FUV_factor =  np.loadtxt(FUVband_file, usecols=(0,1),unpack= True)

		
		files = [H160band_file, H100band_file, M70band_file, M24band_file,I4band_file , I3band_file, I2band_file, I1band_file, Kband_file, Hband_file, Jband_file,zband_file  , iband_file, rband_file, Vband_file, gband_file , Bband_file,uband_file, NUVband_file, FUVband_file]

		lambdas = [H160_lambda, H100_lambda, M70_lambda, M24_lambda, I4_lambda, I3_lambda, I2_lambda, I1_lambda,K_lambda,H_lambda , J_lambda,z_lambda, i_lambda,r_lambda ,V_lambda ,g_lambda , B_lambda, u_lambda, NUV_lambda,FUV_lambda]

		factors = [H160_factor, H100_factor, M70_factor, M24_factor, I4_factor, I3_factor, I2_factor, I1_factor, K_factor, H_factor , J_factor, z_factor, i_factor, r_factor ,V_factor ,g_factor , B_factor , u_factor , NUV_factor ,FUV_factor ]

        #dictionaries lambdas_dict, factors_dict

		files_dict = defaultdict(list)
		lambdas_dict = defaultdict(list)
		factors_dict = defaultdict(list)

		for i in range(len(files)):

			files_dict[bands[i]].append(files[i])
			lambdas_dict[bands[i]].append(lambdas[i])
			factors_dict[bands[i]].append(factors[i])

				
	if filterset == 'BOOTES_FIR':

		bands = [ 11.77815 , 11.933053, 12.0791812, 13.09650142,  13.59050142, 13.69897, 13.80919, 13.90609, 14.12996,  14.2499 , 14.3662, 14.4491, 14.5169, 14.5748, 14.6686,  14.8239,  14.91335, 15.1135]

		H500band_file = path + 'models/FILTERS/HERSCHEL/SPIRE_500mu.txt'
		H500_lambda, H500_factor =  np.loadtxt(H500band_file, usecols=(0,1),unpack= True)

		H350band_file = path + 'models/FILTERS/HERSCHEL/SPIRE_350mu.txt'
		H350_lambda, H350_factor =  np.loadtxt(H350band_file, usecols=(0,1),unpack= True)

		H250band_file = path + 'models/FILTERS/HERSCHEL/SPIRE_250mu.txt'
		H250_lambda, H250_factor =  np.loadtxt(H250band_file, usecols=(0,1),unpack= True)

		M24band_file =  path + 'models/FILTERS/SPITZER/mips24.res'
		M24_lambda, M24_factor =  np.loadtxt(M24band_file, usecols=(0,1),unpack= True)


		I4band_file = path + 'models/FILTERS/SPITZER/irac_ch4.res'
		I4_lambda, I4_factor =  np.loadtxt(I4band_file, usecols=(0,1),unpack= True)

		I3band_file =  path + 'models/FILTERS/SPITZER/irac_ch3.res'
		I3_lambda, I3_factor =  np.loadtxt(I3band_file, usecols=(0,1),unpack= True)

		I2band_file = path + 'models/FILTERS/SPITZER/irac_ch2.res'
		I2_lambda, I2_factor =  np.loadtxt(I2band_file, usecols=(0,1),unpack= True)

		I1band_file = path + 'models/FILTERS/SPITZER/irac_ch1.res'
		I1_lambda, I1_factor =  np.loadtxt(I1band_file, usecols=(0,1),unpack= True)

		
		Kband_file = path + 'models/FILTERS/2MASS/Ks_2mass.res'
		K_lambda, K_factor =  np.loadtxt(Kband_file, usecols=(0,1),unpack= True)

		Hband_file = path + 'models/FILTERS/2MASS/H_2mass.res'
		H_lambda, H_factor =  np.loadtxt(Hband_file, usecols=(0,1),unpack= True)

		Jband_file = path + 'models/FILTERS/2MASS/J_2mass.res'
		J_lambda, J_factor =  np.loadtxt(Jband_file, usecols=(0,1),unpack= True)

		Yband_file = path + 'models/FILTERS/VISTA/Y_uv.res'
		Y_lambda, Y_factor =  np.loadtxt(Yband_file, usecols=(0,1),unpack= True)

		zband_file =path + 'models/FILTERS/SUBARU/z_subaru.res'
		z_lambda, z_factor =  np.loadtxt(zband_file, usecols=(0,1),unpack= True)

		iband_file = path + 'models/FILTERS/CHFT/i_megaprime_sagem.res'
		i_lambda, i_factor =  np.loadtxt(iband_file, usecols=(0,1),unpack= True)

		rband_file = path + 'models/FILTERS/SUBARU/r_subaru.res'
		r_lambda,r_factor =  np.loadtxt(rband_file, usecols=(0,1),unpack= True)
		
		
		Bband_file = path + 'models/FILTERS/SUBARU/B_subaru.res'
		B_lambda, B_factor =  np.loadtxt(Bband_file, usecols=(0,1),unpack= True)


		uband_file = path + 'models/FILTERS/CHFT/u_megaprime_sagem.res'
		u_lambda, u_factor =  np.loadtxt(uband_file, usecols=(0,1),unpack= True)

		NUVband_file = path + 'models/FILTERS/GALEX/galex2500.res'
		NUV_lambda, NUV_factor =  np.loadtxt(NUVband_file, usecols=(0,1),unpack= True)

		files = [ H500band_file, H350band_file, H250band_file, M24band_file, I4band_file , I3band_file, I2band_file, I1band_file, Kband_file, Hband_file, Jband_file, Yband_file, zband_file , iband_file, rband_file,  Bband_file,  uband_file, NUVband_file]

		lambdas = [H500_lambda, H350_lambda, H250_lambda, M24_lambda, I4_lambda , I3_lambda, I2_lambda, I1_lambda,  K_lambda, H_lambda, J_lambda, Y_lambda,  z_lambda, i_lambda, r_lambda, B_lambda,  u_lambda, NUV_lambda]

		factors = [ H500_factor, H350_factor, H250_factor, M24_factor, I4_factor , I3_factor, I2_factor, I1_factor, K_factor, H_factor, J_factor, Y_factor, z_factor, i_factor, r_factor,  B_factor,  u_factor, NUV_factor]

        #dictionaries lambdas_dict, factors_dict

		files_dict = defaultdict(list)
		lambdas_dict = defaultdict(list)
		factors_dict = defaultdict(list)

		for i in range(len(files)):

			files_dict[bands[i]].append(files[i])
			lambdas_dict[bands[i]].append(lambdas[i])
			factors_dict[bands[i]].append(factors[i])

			
	if filterset == 'BOOTES':

		bands = [ 13.09650142,  13.59050142, 13.69897, 13.80919, 13.90609, 14.12996,  14.2499 , 14.3662, 14.4491, 14.5169, 14.5748, 14.6686,  14.8239,  14.91335, 15.1135]

		M24band_file =  path + 'models/FILTERS/SPITZER/mips24.res'
		M24_lambda, M24_factor =  np.loadtxt(M24band_file, usecols=(0,1),unpack= True)


		I4band_file = path + 'models/FILTERS/SPITZER/irac_ch4.res'
		I4_lambda, I4_factor =  np.loadtxt(I4band_file, usecols=(0,1),unpack= True)

		I3band_file =  path + 'models/FILTERS/SPITZER/irac_ch3.res'
		I3_lambda, I3_factor =  np.loadtxt(I3band_file, usecols=(0,1),unpack= True)

		I2band_file = path + 'models/FILTERS/SPITZER/irac_ch2.res'
		I2_lambda, I2_factor =  np.loadtxt(I2band_file, usecols=(0,1),unpack= True)

		I1band_file = path + 'models/FILTERS/SPITZER/irac_ch1.res'
		I1_lambda, I1_factor =  np.loadtxt(I1band_file, usecols=(0,1),unpack= True)

		
		Kband_file = path + 'models/FILTERS/2MASS/Ks_2mass.res'
		K_lambda, K_factor =  np.loadtxt(Kband_file, usecols=(0,1),unpack= True)

		Hband_file = path + 'models/FILTERS/2MASS/H_2mass.res'
		H_lambda, H_factor =  np.loadtxt(Hband_file, usecols=(0,1),unpack= True)

		Jband_file = path + 'models/FILTERS/2MASS/J_2mass.res'
		J_lambda, J_factor =  np.loadtxt(Jband_file, usecols=(0,1),unpack= True)

		Yband_file = path + 'models/FILTERS/VISTA/Y_uv.res'
		Y_lambda, Y_factor =  np.loadtxt(Yband_file, usecols=(0,1),unpack= True)

		zband_file =path + 'models/FILTERS/SUBARU/z_subaru.res'
		z_lambda, z_factor =  np.loadtxt(zband_file, usecols=(0,1),unpack= True)

		iband_file = path + 'models/FILTERS/CHFT/i_megaprime_sagem.res'
		i_lambda, i_factor =  np.loadtxt(iband_file, usecols=(0,1),unpack= True)

		rband_file = path + 'models/FILTERS/SUBARU/r_subaru.res'
		r_lambda,r_factor =  np.loadtxt(rband_file, usecols=(0,1),unpack= True)
		
		
		Bband_file = path + 'models/FILTERS/SUBARU/B_subaru.res'
		B_lambda, B_factor =  np.loadtxt(Bband_file, usecols=(0,1),unpack= True)


		uband_file = path + 'models/FILTERS/CHFT/u_megaprime_sagem.res'
		u_lambda, u_factor =  np.loadtxt(uband_file, usecols=(0,1),unpack= True)

		NUVband_file = path + 'models/FILTERS/GALEX/galex2500.res'
		NUV_lambda, NUV_factor =  np.loadtxt(NUVband_file, usecols=(0,1),unpack= True)

		files = [ M24band_file, I4band_file , I3band_file, I2band_file, I1band_file, Kband_file, Hband_file, Jband_file, Yband_file, zband_file , iband_file, rband_file,  Bband_file,  uband_file, NUVband_file]

		lambdas = [M24_lambda, I4_lambda , I3_lambda, I2_lambda, I1_lambda,  K_lambda, H_lambda, J_lambda, Y_lambda,  z_lambda, i_lambda, r_lambda, B_lambda,  u_lambda, NUV_lambda]

		factors = [ M24_factor, I4_factor , I3_factor, I2_factor, I1_factor, K_factor, H_factor, J_factor, Y_factor, z_factor, i_factor, r_factor,  B_factor,  u_factor, NUV_factor]

        #dictionaries lambdas_dict, factors_dict

		files_dict = defaultdict(list)
		lambdas_dict = defaultdict(list)
		factors_dict = defaultdict(list)

		for i in range(len(files)):

			files_dict[bands[i]].append(files[i])
			lambdas_dict[bands[i]].append(lambdas[i])
			factors_dict[bands[i]].append(factors[i])

	# print 'A dictionary has been constructed for the bands: '
	# print bands		
	return bands, files_dict, lambdas_dict, factors_dict


def frange(start, stop, step):
    r = start
    while r < stop:
         yield r
         r += step

