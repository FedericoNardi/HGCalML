#!/usr/bin/env python3

import pickle
import mgzip
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_cluster_distances(pred_beta, pred_dist, ccoords, t_d):
    beta_max, beta_max_id= *[pred_beta.max()], *[pred_beta.argmax()]
    #print(beta_max)
    beta_max_coords = ccoords[beta_max_id]
    distances = np.sqrt((ccoords[:,0]-np.ones_like(ccoords[:,0])*beta_max_coords[0])**2 + 
                   (ccoords[:,1]-np.ones_like(ccoords[:,0])*beta_max_coords[1])**2 + 
                   (ccoords[:,2]-np.ones_like(ccoords[:,0])*beta_max_coords[2])**2)
    return (np.where(distances<t_d*pred_dist[beta_max_id],1,0), distances)

energies = [10, 25, 50, 75, 100, 125, 150] #, 175]

# Generate as many subplots as len(energies)
fig, axs = plt.subplots(len(energies), 1, figsize=(8, 15), sharex = False)
fig.text(0.5, 0.08, r'$E_{reco}$/$E_{true}$', ha='center')
fig.text(0.04, 0.5, 'Counts', va='center', rotation='vertical')

full_t_energies = []
full_p_energies = []

dirname = 'pred_cut_t/'
PATH_IN = '/media/disk/photon_data/predictions/'
PATH_OUT = '/home/centos/HGCalML/'

for i in range(len(energies)):

    reco_energies = []
    evt_energies = []

    for fname in os.listdir(PATH_IN+dirname):   
        if str(energies[i])+'GeV' in fname:
            with mgzip.open(PATH_IN+dirname+fname) as predictions:
                predictions = pickle.load(predictions)  
                for event in predictions: 
                    pred_id = event[2]['pred_id'].argmax(axis=1)
                    pred_beta = event[2]['pred_beta']
                    pred_ene = event[2]['rechit_energy']
                    pred_dist = event[2]['pred_dist']
                    ccoords = event[2]['pred_ccoords']
                    cluster_idx, cluster_distances = calculate_cluster_distances(pred_beta, pred_dist, ccoords,1.)
                    mask = cluster_idx==1.
                    reco_energies.append(pred_ene[mask].sum())
                    full_p_energies.append(pred_ene[mask].sum())
                    evt_energies.append( np.unique(event[1]['truthHitAssignedEnergies'])[0] )
                    full_t_energies.append( np.unique(event[1]['truthHitAssignedEnergies'])[0] )
    
    ratios = np.array(reco_energies)/np.array(evt_energies)

    axs[i].hist(ratios, bins=50, alpha=0.25, range=(0.5,1.5), label='Reco/True')
    axs[i].text(0.05,0.8,str(energies[i])+' GeV', transform=axs[i].transAxes)
plt.savefig(PATH_OUT+'analysis/timing/'+'energy_ratios_t.jpg')
# Close figure
plt.close()

from scipy.optimize import curve_fit

unique_energies = np.unique(full_t_energies)

# Generate as many subplots as len(energies)

fig, axs = plt.subplots(len(energies), 1, figsize=(8, 15))
fig.text(0.5, 0.08, r'$E_{reco}$/$E_{true}$', ha='center')
fig.text(0.04, 0.5, 'Counts', va='center', rotation='vertical')

means_t = []
sigmas_t = []

i=0

for energy in unique_energies:
    
    sel = np.array(full_t_energies)
    mask = sel==energy
    hist = axs[i].hist(np.array(full_p_energies)[mask],bins=30, alpha=0.25, range = [0.8*energy, 1.2*energy],density=True)
    axs[i].text(0.05,0.8,str(energy)+' GeV', transform=axs[i].transAxes)
    vals=hist[0]
    centers=hist[1][1:]
    import scipy.stats as stats
    def fit_f(x, beta, m, loc, scale):
        return stats.crystalball.pdf(x, beta, m, loc, scale)
    weight = np.where(vals<10,1000,vals)
    popt, pcov = curve_fit(fit_f, 
                           centers, 
                           vals, 
                           # sigma=np.power(weight,-0.5), 
                           maxfev=100000,
                           p0=[.5,2.,energy,0.1*energy]
                           )
    means_t.append(popt[2])
    sigmas_t.append(popt[3])
    axs[i].plot(centers, fit_f(centers, *popt), 'r-', label='fit')
    i+=1
print("---> MEANS_T: ",means_t)

# plot vertical line over all subplots
i = 0
for ax in axs:
    ax.axvline(means_t[i], color='r', linestyle='--')
    # add shaded region corresponding to rms in each subplot
    ax.axvspan(means_t[i]-sigmas_t[i], means_t[i]+sigmas_t[i], alpha=0.15, color='red')
    i+=1
plt.savefig(PATH_OUT+'analysis/timing/'+'energy_ratios_t_fit.jpg')
plt.close()

# Generate as many subplots as len(energies)
fig, axs = plt.subplots(len(energies), 1, figsize=(8, 15), sharex = False)
fig.text(0.5, 0.08, r'$E_{reco}$/$E_{true}$', ha='center')
fig.text(0.04, 0.5, 'Counts', va='center', rotation='vertical')

energy = [10, 25, 50, 75, 100, 125, 150, 175]

full_t_energies = []
full_p_energies = []

dirname = 'pred_cut_no_t/'
PATH_IN = '/media/disk/photon_data/predictions/'
PATH_OUT = '/home/centos/HGCalML/'

for i in range(len(energies)):

    reco_energies = []
    evt_energies = []

    for fname in os.listdir(PATH_IN+dirname):   
        if str(energies[i])+'GeV' in fname:
            with mgzip.open(PATH_IN+dirname+fname) as predictions:
                predictions = pickle.load(predictions)  
                for event in predictions: 
                    pred_id = event[2]['pred_id'].argmax(axis=1)
                    pred_beta = event[2]['pred_beta']
                    pred_ene = event[2]['rechit_energy']
                    pred_dist = event[2]['pred_dist']
                    ccoords = event[2]['pred_ccoords']
                    cluster_idx, cluster_distances = calculate_cluster_distances(pred_beta, pred_dist, ccoords,1.)
                    mask = cluster_idx==1.
                    reco_energies.append(pred_ene[mask].sum())
                    full_p_energies.append(pred_ene[mask].sum())
                    evt_energies.append( np.unique(event[1]['truthHitAssignedEnergies'])[0] )
                    full_t_energies.append( np.unique(event[1]['truthHitAssignedEnergies'])[0] )
    
    ratios = np.array(reco_energies)/np.array(evt_energies)

    axs[i].hist(ratios, bins=50, alpha=0.25, range=(0.5,1.5), label='Reco/True')
    axs[i].text(0.05,0.8,str(energies[i])+' GeV', transform=axs[i].transAxes)
plt.savefig(PATH_OUT+'analysis/timing/'+'energy_ratios.jpg')
# Close figure
plt.close()


from scipy.optimize import curve_fit

unique_energies = np.unique(full_t_energies)

# Generate as many subplots as len(energies)

fig, axs = plt.subplots(len(energies), 1, figsize=(8, 15))
fig.text(0.5, 0.08, r'$E_{reco}$/$E_{true}$', ha='center')
fig.text(0.04, 0.5, 'Counts', va='center', rotation='vertical')

means = []
sigmas = []

i=0

for energy in unique_energies:
    
    sel = np.array(full_t_energies)
    mask = sel==energy
    hist = axs[i].hist(np.array(full_p_energies)[mask],bins=30, alpha=0.25, range = [0.8*energy, 1.2*energy],density=True)
    axs[i].text(0.05,0.8,str(energy)+' GeV', transform=axs[i].transAxes)
    vals=hist[0]
    centers=hist[1][1:]
    import scipy.stats as stats
    def fit_f(x, beta, m, loc, scale):
        return stats.crystalball.pdf(x, beta, m, loc, scale)
    weight = np.where(vals<10,1000,vals)
    popt, pcov = curve_fit(fit_f, 
                           centers, 
                           vals, 
                           # sigma=np.power(weight,-0.5), 
                           maxfev=100000,
                           p0=[.5,2.,energy,0.1*energy]
                           )
    means.append(popt[2])
    sigmas.append(popt[3])
    axs[i].plot(centers, fit_f(centers, *popt), 'r-', label='fit')
    i+=1

# plot vertical line over all subplots
i = 0
for ax in axs:
    ax.axvline(means[i], color='r', linestyle='--')
    # add shaded region corresponding to rms in each subplot
    ax.axvspan(means[i]-sigmas[i], means[i]+sigmas[i], alpha=0.15, color='red')
    i+=1
plt.savefig(PATH_OUT+'analysis/timing/'+'energy_ratios_fit.jpg')
plt.close()

def resFunction(energies,a,b):
   return np.sqrt( np.square(a*np.power(energies,-0.5)) + b*b*(np.ones_like(energies)) )

resolution = np.abs(sigmas)/np.array(means)
resolution_t = np.abs(sigmas_t)/np.array(means_t)

# popt, pcov = curve_fit(resFunction, unique_energies, resolution, maxfev=5000, p0=[0.1,0.1])

plt.figure()
plt.plot(unique_energies, 100*resolution, 'gx',linewidth=0.5, label='OC values')
plt.plot(unique_energies, 100*resolution_t, 'rx',linewidth=0.5, label='OC values time')
# plt.plot(unique_energies, 100*resFunction(unique_energies, *popt), 'g-', label='OC fit')
# popt_t, pcov_t = curve_fit(resFunction, unique_energies, resolution_t, maxfev=5000, p0=[0.1,0.1])
# plt.plot(unique_energies, 100*resFunction(unique_energies, *popt_t), 'r-', label='OC fit with time')
plt.xlabel('True energy [GeV]')
plt.ylabel(r'$\sigma/<E>$ [%]')
plt.grid(linewidth=0.5,linestyle='--')
# print("OC Resolution fit parameters: ", popt)
# print("OC Resolution fit parameters with time: ", popt_t)

PPFA = np.array([4.23, 3.69, 2.30, 1.76, 1.62, 1.40, 1.23]) #, 1.21])
popt_, pcov_ = curve_fit(resFunction, unique_energies, 0.01*PPFA, maxfev=5000, p0=[0.1,0.1])
plt.plot(unique_energies, PPFA, 'c+',linewidth=0.5, label='PandoraPFA values')
plt.plot(unique_energies, 100*resFunction(unique_energies, *popt_), 'c-', label='PandoraPFA fit')
plt.legend()
print("PPFA Resolution fit parameters: ", popt_)
plt.savefig(PATH_OUT+'analysis/timing/'+'resolution.jpg')