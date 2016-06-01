import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.cbook as cbook
from scipy.integrate import simps, trapz
import os, sys, glob, fnmatch
from math import exp,log,pi       
from DATA_AGNfitter import DISTANCE, NAME, REDSHIFT, DATA
from MODEL_AGNfitter import z2Dlum


AGNfitter_out_witherrors_0= []
AGNfitter_out_witherrors_1= []
AGNfitter_out_witherrors_2= []
AGNfitter_out_witherrors_3= []
AGNfitter_out_max= []


 
 #Get original info from the catalog
catalog = '/data2/calistro/AGNfitter/data/catalog_Bootes_final.txt'


lines, b  = np.loadtxt(catalog, usecols=(0,1), unpack=True) 
linesofcatalog=  np.arange(0, len(lines))#np.loadtxt('data/SelectionTYPE2_15datapoints')
linesofcatalog = linesofcatalog.astype(int)
print 'Number of sources in the catalog', linesofcatalog


#Get original info from the model lists of gal and tor. GENERALIZE!
gal_list= 'models/GALAXY/input_template_hoggnew.dat'
tor_list ='models/TORUS/torus_templates_list.dat'



#==========================================
# INPUT values
#==========================================

#f catalog == '/home/calistro/AGNfitter/data/catalog_type1_softsel.dat':
	
#	folder = '/export/qso3/scratch/calistro/2015/Type1/'


#elif catalog == '/home/calistro/AGNfitter/data/catalog_type2_hardsel.dat':

#	folder = '/export/qso3/scratch/calistro/2015/Type2/'
folder = '/data2/calistro/AGNfitter/OUTPUT/NEW/'





#==========================================
# OUTPUT values
#==========================================



for i in linesofcatalog:

	print 'Gal number' ,i, 'of folder', folder

	sourceline = i
	sourcename = NAME(catalog, sourceline)



#MAXIMUM LIKELIHOOD OUTPUT

	if os.path.lexists(folder+str(sourcename)+'/max_likelihood_parameters_2_'+str(sourcename)+'.txt'):

	
		z = REDSHIFT(catalog,i)
		d = z2Dlum(z)


		"""
		Write complete best-fit output for the whole sample into a file 	
		"""

		values, _ = np.loadtxt(folder+str(sourcename)+'/max_likelihood_parameters_2_'+str(sourcename)+'.txt', usecols=(0,1), unpack= True,  dtype = ('S')) 

		
		sourceline0 = [i, sourcename, z, d] 
		sourceline1 = np.hstack((sourceline0, values))
		AGNfitter_out_max.append(sourceline1)


#PARAMETERS WITH ERROR OUTPUT

	if os.path.lexists(folder+str(sourcename)+'/parameters_with_errors_2_'+str(sourcename)+'.txt'):



		"""
		Write complete many-realizations output for the whole sample into a file 
		"""


		tau_e, age_e, nh_e, irlum_e, SB_e, BB_e, GA_e, TO_e, BBebv_e, GAebv_e, Mstar_e, SFR_e, SFR_file_e, L0_e, L1_e, L2_e, L3_e, L4_e, L5_e, SFR_IR_e  = np.loadtxt(folder+str(sourcename)+'/parameters_with_errors_2_'+str(sourcename)+'.txt', usecols=(0,1,2,3,4,5,6,7,8,9,10,11, 12, 13,14,15, 16, 17,18), unpack=True , dtype = ('S')) 



		sourceline2_0 = [sourcename, z, d, tau_e[0], age_e[0], nh_e[0], irlum_e[0], SB_e[0], BB_e[0], GA_e[0], TO_e[0], BBebv_e[0], GAebv_e[0], Mstar_e[0], SFR_e[0], SFR_file_e[0], L0_e[0], L1_e[0], L2_e[0], L3_e[0], L4_e[0], L5_e[0], SFR_IR_e[0] ]

			

		sourceline2_1 = [sourcename, z, d, tau_e[1], age_e[1], nh_e[1], irlum_e[1], SB_e[1], BB_e[1], GA_e[1], TO_e[1], BBebv_e[1], GAebv_e[1], Mstar_e[1], SFR_e[1], SFR_file_e[1], L0_e[1], L1_e[1], L2_e[1], L3_e[1], L4_e[1], L5_e[1] , SFR_IR_e[1]]



		sourceline2_2 = [sourcename, z, d, tau_e[2], age_e[2], nh_e[2], irlum_e[2], SB_e[2], BB_e[2], GA_e[2], TO_e[2], BBebv_e[2], GAebv_e[2], Mstar_e[2], SFR_e[2], SFR_file_e[2], L0_e[2], L1_e[2], L2_e[2], L3_e[2], L4_e[2], L5_e[1], SFR_IR_e[2]]

		
		AGNfitter_out_witherrors_0.append(sourceline2_0)
		AGNfitter_out_witherrors_1.append(sourceline2_1)
		AGNfitter_out_witherrors_2.append(sourceline2_2)




	



	else: 
		print 'noooooooo', folder+str(sourcename)+'/max_likelihood_parameters_2_'+str(sourcename)+'.txt'
		print folder+str(sourcename)+'/parameters_with_errors_2_'+str(sourcename)+'.txt'



d0 = np.hstack((AGNfitter_out_witherrors_0, AGNfitter_out_witherrors_1, AGNfitter_out_witherrors_2))


filename_witherrors = folder+ 'totaloutput_witherrors_0_plus_minus_2.txt'
np.savetxt(filename_witherrors , d0, delimiter = " ",fmt="%s" ,header="i sourcename z d tau_e age_e nh_e irlum_e SB_e BB_e GA_e TO_e BBebv_e GAebv_e Mstar_e SFR_e SFR_file_e L0_e L1_e L2_e L3_e L4_e  L5_e SFR_e i sourcename z d tau_ep age_ep nh_ep irlum_ep SB_ep BB_ep GA_ep TO_ep BBebv_ep GAebv_ep Mstar_ep SFR_ep SFR_file_ep L0_ep L1_ep L2_ep L3_ep L4_ep L5_ep SFR_IR_ep i sourcename z d tau_em age_em nh_em irlum_em SB_em BB_em GA_em TO_em BBebv_em GAebv_em Mstar_em SFR_em SFR_file_em L0_em L1_em L2_em L3_em L4_em  L5_em SFR_IR_en")


e= np.array(AGNfitter_out_max)
filename_witherrors = folder+ 'totaloutput_max_2.txt'
np.savetxt(filename_witherrors , e, delimiter = " ",fmt="%s" ,header="i sourcename z d tau age nh irlum SB BB GA TO BBebv GAebv Mstar SFR SFR_file ln_likelihood")



