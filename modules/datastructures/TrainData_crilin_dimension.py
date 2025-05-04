from DeepJetCore import TrainData
from DeepJetCore import SimpleArray
import numpy as np
import uproot3 as uproot

import os
import pickle
import gzip
import pandas as pd

from datastructures.TrainData_NanoML import TrainData_NanoML

class TrainData_crilin_dimension(TrainData_NanoML):
   
    def branchToFlatArray(self, b, return_row_splits=False, dtype='float32'):
        
        a = b.array()
        nevents = a.shape[0]
        rowsplits = [0]
        
        for i in range(nevents):
            rowsplits.append(rowsplits[-1] + a[i].shape[0])
        
        rowsplits = np.array(rowsplits, dtype='int64')
        
        if return_row_splits:
            return np.expand_dims(np.array(a.flatten(),dtype=dtype), axis=1),np.array(rowsplits, dtype='int64') 
        else:
            return np.expand_dims(np.array(a.flatten(),dtype=dtype), axis=1)

    def convertFromSourceFile(self, filename, weighterobjects, istraining, treename="Events"):
        
        #fileTimeOut(filename, 10)#wait 10 seconds for file in case there are hiccups
        tree = uproot.open(filename)[treename]
        
        '''
        
        hit_x, hit_y, hit_z: the spatial coordinates of the voxel centroids that registered the hit
        hit_dE: the energy registered in the voxel (signal + BIB noise)
        recHit_dE: the 'reconstructed' hit energy, i.e. the energy deposited by signal only
        evt_dE: the total energy deposited by the signal photon in the calorimeter
        evt_ID: an int label for each event -only for bookkeeping, should not be needed
        isSignal: a flag, -1 if only BIB noise, 0 if there is also signal hit deposition

        '''
        
        hit_x, rs = self.branchToFlatArray(tree["x"], True)
        hit_y = self.branchToFlatArray(tree["y"])
        hit_z = self.branchToFlatArray(tree["z"])
        hit_dE = self.branchToFlatArray(tree["dE"])
        
        zerosf = 0.*hit_dE
        
        #truth
        signalFraction = self.branchToFlatArray(tree["signal_fraction"])
        evt_trueE = self.branchToFlatArray(tree["E0"])
        isSignal = np.where(signalFraction>0.5, 1, 0).astype(np.int32)
        evt_dE = hit_dE*signalFraction
        hit_volume = self.branchToFlatArray(tree["volume"])
        
        zerosi = 0 * isSignal
        ### now we build the same structure as NanoML
        
        farr = SimpleArray(np.concatenate([
            hit_dE,
            zerosf,
            hit_volume,
            hit_x,
            hit_y,
            hit_z,
            zerosf,
            zerosf
            ], axis=-1), rs,name="recHitFeatures")
        
        # print("FARR SHAPE: ",farr.shape())
        # print("CELL_DX: \n", cell_dx)
        # print("FARR: \n", farr)
        
        t = {
            't_idx' : SimpleArray(isSignal, rs), #names are optional
            't_energy' : SimpleArray(evt_trueE, rs),
            't_pos' : SimpleArray(np.concatenate(3*[zerosf],axis=-1), rs), #three coordinates
            't_time' : SimpleArray(zerosf, rs)  ,
            't_pid' : SimpleArray(np.concatenate( [1+zerosi]+5*[zerosi],axis=-1 ), rs) , #6 truth classes
            't_spectator' : SimpleArray(zerosf, rs),
            't_fully_contained' : SimpleArray(zerosf + 1., rs),
            't_rec_energy' : SimpleArray(evt_dE, rs),
            't_is_unique' : SimpleArray(zerosi, rs),
            't_sig_fraction' : SimpleArray(signalFraction, rs)
            }
        
        
        return [farr, 
                t['t_idx'], t['t_energy'], t['t_pos'], t['t_time'], 
                t['t_pid'], t['t_spectator'], t['t_fully_contained'],
                t['t_rec_energy'], t['t_is_unique'], t['t_sig_fraction'] ],[], []

    def interpretAllModelInputs(self, ilist, returndict=True):
        if not returndict:
            raise ValueError('interpretAllModelInputs: Non-dict output is DEPRECATED. PLEASE REMOVE') 
        '''
        input: the full list of keras inputs
        returns: td
         - rechit feature array
         - t_idx
         - t_energy
         - t_pos
         - t_time
         - t_pid :             non hot-encoded pid
         - t_spectator :       spectator score, higher: further from shower core
         - t_fully_contained : fully contained in calorimeter, no 'scraping'
         - t_rec_energy :      the truth-associated deposited 
                               (and rechit calibrated) energy, including fractional assignments)
         - t_is_unique :       an index that is 1 for exactly one hit per truth shower
         - row_splits
         
        '''
        out = {
            'features':ilist[0],
            'rechit_energy': ilist[0][:,0:1], #this is hacky. FIXME
            't_idx':ilist[2],
            't_energy':ilist[4],
            't_pos':ilist[6],
            't_time':ilist[8],
            't_pid':ilist[10],
            't_spectator':ilist[12],
            't_fully_contained':ilist[14],
            'row_splits':ilist[1]
            }
        #keep length check for compatibility
        if len(ilist)>16:
            out['t_rec_energy'] = ilist[16]
        if len(ilist)>18:
            out['t_is_unique'] = ilist[18]
        if len(ilist)>20:
            out['t_sig_fraction'] = ilist[20]
        return out
    