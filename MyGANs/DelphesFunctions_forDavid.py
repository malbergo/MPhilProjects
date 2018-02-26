#import ROOT
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
from logistics import feature_normalize

#ROOT.gSystem.Load("/usr/local/bin/Delphes-3.3.3/external/ExRootAnalysis/libExRootAnalysis.so")
# ROOT.gROOT.ProcessLine('.include /usr/local/bin/Delphes-3.4.1')
#ROOT.gSystem.Load("/usr/local/bin/Delphes-3.3.3/libDelphes.so")





def load_electron_data(root_file, Normalized=True):
    f_root = ROOT.TFile(root_file)
    chain = ROOT.TChain("Delphes")
    chain.Add(root_file)
    myTree = f_root.Get("Delphes")

    # eta restricted and PT restricted > 10 Mev and -2.5 < eta < 2.5 by Delphes card
    Eta_list = []
    Phi_list = []
    PT_list = []
    for event in myTree:
        for particle in event.Electron:
            #print("Electron eta phi: ", particle.Eta, particle.Phi)
            Eta_list.append(particle.Eta)
            Phi_list.append(particle.Phi)
            PT_list.append(particle.PT)

    Eta_array = np.array(Eta_list)
    Phi_array = np.array(Phi_list)
    PT_array = np.array(PT_list)
    delphes_data = np.column_stack((Eta_array,Phi_array,PT_array))

    if Normalized:
        data, means_stds = feature_normalize(delphes_data)
    else:
        data = delphes_data
        means_stds = [delphes_data.mean(), delphes_data.std()]

    #returns [eta, phi, PT] for electron
    return data, means_stds

# sample the real delphes output, returns cuda variable of it
def sample_delphes_array(data_arr, bs=64, eta_center=None, phi = 0, epsilon=0.1, tensor = True):

    #indices = np.random.randint(low=0, high=data_arr.shape[0], size=bs)
    # get the indices of the data you are trying to make
    if eta_center != None:
        indices = list(np.where((data_arr[:,0] > (eta_center - epsilon)) & (data_arr[:,0] < (eta_center + epsilon)))[0])
        data = data_arr[indices,:]
    else:
        data = data_arr

    #don't try to take more samples than you have left after epsilon range
    if bs > data.shape[0]:
        sample_indices = np.random.randint(low=0, high=data.shape[0], size=data.shape[0])
    else:
        sample_indices = np.random.randint(low=0, high=data.shape[0], size=bs)

    data_sample = data[sample_indices,:]
    if tensor == True:
        data_sample = torch.from_numpy(data_sample).type(torch.FloatTensor)
        return Variable(data_sample.cuda(), requires_grad=True)
    else:
        return data_sample

# def specific_delphes_sample(data_arr, eta_center = 2.3, phi= 0, size=256, Normalized=True, epsilon = 0.1):
#
#     data_max, data_min = data_arr.max(), data_arr.min()
#
#     # get the indices of the data you are trying to make
#     indices = list(np.where((data_arr[:,0] > (eta_center - epsilon)) & (data_arr[:,0] < (eta_center + epsilon)))[0])
#     if Normalized:
#         data, means_stds = feature_normalize(data_arr)
#         data = data[indices,:]
#     else:
#         data = data_arr
#         data = data[indices,:]
#
#     #print data.shape[0]
#
#     #don't try to take more samples than you have left after epsilon range
#     if size > data.shape[0]:
#         sample_indices = np.random.randint(low=0, high=data.shape[0], size=data.shape[0])
#     else:
#         sample_indices = np.random.randint(low=0, high=data.shape[0], size=size)
#
#     data_sample = data[sample_indices,:]
#     return data_sample, means_stds
