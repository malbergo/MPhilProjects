import numpy as np
#import root_numpy
#import root_pandas
from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize, LogNorm
import torch
from torch.autograd import Variable, grad
import ROOT

#ROOT.gSystem.Load("/usr/local/bin/Delphes-3.3.3/external/ExRootAnalysis/libExRootAnalysis.so")
# ROOT.gROOT.ProcessLine('.include /usr/local/bin/Delphes-3.4.1')
ROOT.gSystem.Load("/usr/local/bin/Delphes-3.3.3/libDelphes.so")



def save_array(data, outfile_name):
	'''
	requires data to be FloatTensor or numpy array
	'''
	if isintance(data,np.ndarray):
		if data.is_cuda == True:
			data = data.cpu().numpy()
	else:
		if data.is_cuda == True:
			data = data.cpu().numpy()
		else:
			data = data.numpy()
	savefile_path = "/home/chris/Documents/MPhilProjects/GANs/MyGANs/data/"
	filename = outfile_name
	np.save(savefile_path + filename, data)
	return


def plot2dHist(tensorVariable_data):

	numpy_data = tensorVariable_data.cpu().data.numpy()
	fig, ax_data = plt.subplots()
	counts,xedges,yedges,im = ax_data.hist2d(numpy_data[:,0], numpy_data[:,1], weights = numpy_data[:,2], bins=np.arange(-2.5,2.5,0.1))
	plt.colorbar(im)
	plt.show()
	return


def feature_normalize(data_arr, given_means_stds=None):

	data_normed = np.zeros(data_arr.shape)

	if given_means_stds != None:

		means = given_means_stds[0]
		stds = given_means_stds[1]
		for col in range(data_arr.shape[1]):
			data_normed[:,col] = (data_arr[:,col] - means[col]) / stds[col]

	else:
		stds = []
		means = []
		for col in range(data_arr.shape[1]):
			col_mean = data_arr[:,col].mean()
			#print col_mean
			col_std = data_arr[:,col].std()
			data_normed[:,col] = (data_arr[:,col] - col_mean) / col_std
			stds.append(col_std)
			means.append(col_mean)

	return data_normed, [means, stds]
#
# def feature_normalize(data_arr, given_means_stds = None):
#
# 	if given_means_stds != None:
# 		means = given_means_stds[0]
# 		stds = given_means_stds[0]
# 	else:
# 		means = np.mean(data_arr,0)
# 		print means
# 		stds = np.std(data_arr,0)
# 		data_normed = data_arr
#
# 	for i in range(data_arr.shape[1]):
# 		#print data_normed[:,i].shape, means[i].shape
# 		data_normed[:,i] = (data_arr[:,i] - means[i]) / stds[i]
#
# 	return data_normed, [means,stds]
#


def feature_unnormalize(data_arr, means_stds):

	data_unnormed = np.zeros(data_arr.shape)
	means = means_stds[0]
#	print means
	stds = means_stds[1]
	for col in range(data_arr.shape[1]):
		#print col
		col_mean = means[col]
		col_std = stds[col]
		data_unnormed[:,col]  = (data_arr[:,col]* col_std) + col_mean

	return data_unnormed
