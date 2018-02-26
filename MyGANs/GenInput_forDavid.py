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


'''
What Pythia Does, electron gun, uniform in angle
We seek to use this gun to create "real" pythia events that have a true energy, pt, solid angle, etc.
We want to train the gan to produce "detector-smeared" versions of these: e.g. like the pt values
returned for electrons found in the detector by delphes.
Our Pythia+Delphes particle gun will produce events like this too, but then propagate them through
delphes and make us a root file with different particle detections and outputs.
'''
def eParticle_gun_with_noise_eta_bound(E_max = 50000, bs = 1000000, m = 0.511, Normalized=True, eta_center=None, epsilon=0.1, var=False):

    #delphes places 0 resolution on particles with -2.5 < eta < 2.5. We won't use pythia events that do that either.
    eta_range = [-2.5, 2.5]

    # m mass of electron
    counter = 0
    data_sample = np.empty((bs, 3))

    while counter < bs:

        #for eta construction
        cosTheta = 2.* np.random.uniform() - 1
        sinTheta = np.sqrt(1. - cosTheta**2)
        tanHalfTheta = (1 - cosTheta) / sinTheta

        # range 0 to 2* np.pi
        phi = 2.* np.pi * np.random.uniform()

        # make a decaying function to randomly sample energy values from  that matches pythia
        E = np.power(10, 1+(np.log10(E_max) - 1)*np.random.uniform())

        # momentum
        if E**2 > m**2:
            P = np.sqrt(E**2 - m**2)
        else:
            P = m

        # momentum components
        px = P * sinTheta * np.cos(phi)
        py = P * sinTheta * np.sin(phi)
        pz = P * cosTheta

        # to put phi into the same range as in delphes output b/c we are learning smearing in PT not phi now
        phi = phi - np.pi

        #transverser momentum, we need this because it is what the root file outputs for electrons, not energy
        PT_real = np.sqrt(px**2 + py**2)

        ## if you want calculating eta from momentum: https://arxiv.org/pdf/1604.02651.pdf
        #eta = np.arctanh(pz / (np.sqrt(px**2 + py**2 + pz**2 + m**2)))

        # eta from the pointed angles
        eta = -np.log(tanHalfTheta)

        # check if you want eta values within a certain epsilon or the normal [-2.5,2.5] range
        if eta_center != None:
            eta_range = [eta_center - epsilon, eta_center + epsilon]
        else:
            eta_range = eta_range

        # eta requirement that matches delphes
        if eta < eta_range[0] or eta > eta_range[1]:
            continue
        else:
            row = np.column_stack((eta, phi, PT_real))
            data_sample[counter] = row
            counter += 1

    #normalize
    #means_stds = [data_sample.mean(), data_sample.std()]
    if Normalized == True:
        data, means_stds = feature_normalize(data_sample)
    else:
        data = data_sample
        means_stds = [data_sample.mean(), data_sample.std()]

    noise = np.random.normal(size = bs)

    #add noise
    data = np.column_stack((data, noise))
    eta_phi = data[:,:2]
    print data[:,0].min(), data[:,0].max()
    if var ==True:
        eta_phi = torch.from_numpy(eta_phi).type(torch.FloatTensor)
        data = torch.from_numpy(data).type(torch.FloatTensor)
        return (Variable(data.cuda(), requires_grad=True),
            Variable(eta_phi.cuda(), requires_grad=True), means_stds)
    else:
        return data, eta_phi, means_stds


##OLD, don't use! What Pythia Does, electron gun, uniform in angle (use the one w/ eta bound)
# def particle_gun_with_noise(E_max = 50000, bs = 64, m = 0.511):
#
#     # m mass of electron
#     cosTheta = 2.* np.random.uniform(size = bs) - 1
#     sinTheta = np.sqrt(1. - cosTheta**2)
#     tanTheta = sinTheta / cosTheta
#     # range -np.pi to np.pi
#     phi = 2.* np.pi * np.random.uniform(size=bs) - np.pi
#     E = np.power(10, 1+(np.log10(E_max)) - 1)*np.random.uniform(size=bs)
#     P = np.sqrt(E**2 - m**2)
#     px = P * sinTheta * np.cos(phi)
#     py = P * sinTheta * np.sin(phi)
#     pz = P * cosTheta
#
#     ## calculating eta: https://arxiv.org/pdf/1604.02651.pdf
#     eta = np.arctanh(pz / (np.sqrt(px**2 + py**2 + pz**2 + m**2)))
#     data_sample = np.column_stack((cosTheta, sinTheta, phi, E, P))
#     #normalize
#     normalizing_terms = [data_sample.min(), data_sample.max()]
#     data_sample = (data_sample - data_sample.min())/(data_sample.max() - data_sample.min())
#     #noise = np.random.normal(size = bs)
#     noise = np.random.uniform(size=bs)
#     data_sample = np.column_stack((data_sample, noise))
#     data_sample = torch.from_numpy(data_sample)
#
#     return Variable(data_sample.cuda(), requires_grad=true), normalizing_terms



def sample_gun_data(data_arr, bs = 64, eta_center = None, epsilon=0.1, tensor=True, angles = None):

    # get the data from the big dataset that meets your criteria if you have any
    if eta_center != None:
        indices = np.where((angles[:,0] > (eta_center - epsilon)) & (angles[:,0] < (eta_center + epsilon)))[0]
        #print indices.shape
        #print indices
        data_arr = data_arr[indices,:]
        #print data_arr.shape

    #don't try to take more samples than you have left after epsilon range (only a problem if eta_center != None)
    if bs > data_arr.shape[0]:
        sample_indices = np.random.randint(low=0, high=data_arr.shape[0], size=data_arr.shape[0])
    else:
        sample_indices = np.random.randint(low=0, high=data_arr.shape[0], size=bs)

    data_sample = data_arr[sample_indices,:]
    eta_phi = data_sample[:,:2]
    if tensor ==True:

        eta_phi = torch.from_numpy(eta_phi).type(torch.FloatTensor)
        data_sample = torch.from_numpy(data_sample).type(torch.FloatTensor)
        return Variable(data_sample.cuda(), requires_grad = True), Variable(eta_phi.cuda(), requires_grad = True)
    else:
        return data_sample, eta_phi
