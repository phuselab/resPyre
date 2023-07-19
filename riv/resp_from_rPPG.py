import numpy as np
import pandas as pd
from scipy import signal
from importlib import import_module, util
import os

from scipy.signal import butter, filtfilt
from tqdm import tqdm
import matplotlib.pyplot as plt
from pyVHR.analysis.pipeline import Pipeline
from pyVHR.BVP.filters import apply_filter
from pyVHR.extraction.utils import extract_frames_yield
import utils
from riv.IMS import IMS

class RR_from_rPPG():

    def __init__(self, videoFileName, cuda=True, method='cpu_CHROM', verb=True, fmin=0.1, fmax=0.65, just_one_estimate=True):
        self.videoFileName = videoFileName
        self.cuda = cuda
        self.method = method
        self.verb = verb
        self.fmin = fmin
        self.fmax = fmax
        self.just_one_estimate = just_one_estimate
        self.m = 1


    def get_rPPG(self, wsize=30):
        """
            Extracts rPPG signal from video

            Returns
            -------
                the estimated bvp signal(s)
        """
        self.wsize, self.fps = utils.get_vid_stats(self.videoFileName) # window size in seconds

        if not self.just_one_estimate:
            self.wsize = wsize

        # run
        pipe = Pipeline()          # object to execute the pipeline
        bvps, _, _ = pipe.run_on_video(self.videoFileName,
                                        winsize=self.wsize, 
                                        roi_method='convexhull',
                                        roi_approach='holistic',
                                        method=self.method,
                                        estimate='medians',
                                        patch_size=0, 
                                        RGB_LOW_HIGH_TH=(5,230),
                                        Skin_LOW_HIGH_TH=(5,230),
                                        pre_filt=False,
                                        post_filt=False,
                                        cuda=self.cuda, 
                                        verb=self.verb)

        module = import_module('pyVHR.BVP.filters')
        method_to_call = getattr(module, 'BPfilter')

        bvps = apply_filter(bvps,
                            method_to_call,
                            fps=self.fps,
                            params={'minHz':0.1, 'maxHz':4.0, 'fps':'adaptive', 'order':5})

        self.bvps = np.squeeze(bvps[0])

        return self.bvps

    def extract_RIVs_from_peaks(self):
        """
            This method applies Fiedler et al. algorithm for breath measurement

            Returns
            -------
                the estimated RIVs
        """

        # params
        
        am_sig = []
        bm_halfway_sig = []
        bm_maxima_sig =[]
        bm_minima_sig = []
        fm_max_sig = []
        fm_min_sig =[]
        fm_hr_sig = []

        #import code; code.interact(local=locals())

        #iterate over windows
        #for i in np.arange(len(self.bvps)):
        #shape  = np.asarray(self.bvps[i]).shape
        #bvps = np.asarray(self.bvps)
        #window = bvps[i].reshape(shape[1])
        window = self.bvps

        peaks, _ = signal.find_peaks(window, height=0)
        mins,  = signal.argrelmin(window, axis=0, order=1, mode='clip')

        if(peaks[0]>mins[0]):
            mins = mins[1:]

        lmin = len(mins)
        lmax = len(peaks)

        if(lmin!=lmax):
            if(lmin>lmax):
                mins = mins[:lmax]
            else:
                peaks = peaks[:lmin]

        #AMPLITUDE MODULATION
        am_y = [window[peaks[i]]-window[mins[i]] for i in np.arange(len(peaks))]
        am = signal.resample(am_y, len(window))

        #BASELINE MODULATION
        bm_halfway_y = [(window[peaks[i]]+window[mins[i]])/2 for i in np.arange(len(peaks))]
        bm_halfway = signal.resample(bm_halfway_y, len(window))

        bm_maxima_y = window[peaks]
        bm_maxima = signal.resample(bm_maxima_y, len(window))

        bm_minima_y = window[mins]
        bm_minima = signal.resample(bm_minima_y, len(window))

        #FREQUENCY MODULATION
        fm_max_y = np.diff(peaks)
        fm_min_y = np.diff(mins)
        fm_max = signal.resample(fm_max_y, len(window))
        fm_min = signal.resample(fm_min_y, len(window))

        am_sig.append(am)
        bm_halfway_sig.append(bm_halfway)
        bm_maxima_sig.append(bm_maxima)
        bm_minima_sig.append(bm_minima)
        fm_max_sig.append(fm_max)
        fm_min_sig.append(fm_min)

        res = np.vstack([am_sig, bm_halfway_sig, bm_maxima_sig, bm_minima_sig, fm_max_sig, fm_min_sig])        
        return res

    def extract_RIVs_from_IMS(self):

        par = np.array([0.6, 0.5, 2.0, 1.6])
        ims = IMS(self.m, self.fps, self.fps, self.wsize, int(self.bvps.shape[0]), par, False)
        ims.compute_IMS(self.bvps)
        peak_interval, peak_val_max, amplitude_max, artifacts = ims.get_IMS()

        res = np.vstack([peak_interval, peak_val_max, amplitude_max])
        return res

    def extract_RIVs_from_EMD(self, nIMF=4):
        import emd
        from FIRfilt import FIRfilt

        fc = 1      # FIRfilter cutoff: 1Hz
        ft = 0.2    # FIRfilter transition band: 0.2Hz

        N_div = self.wsize//2-1 if self.wsize % 2 == 0 else (self.wsize-1)//2
        spec = np.zeros((nIMF+1, N_div))   #data structure which stores the spectrum of the IMFs components
        freq = np.zeros((nIMF+1, N_div))   #data structure which stores the frequencies of the spectrum of the IMFs components
        spectrum = True                    #set to TRUE to return the spectrum of the IMFs components

        filt = FIRfilt('LP', fc, ft, self.fps, self.wsize)
        x_filt = filt.filtering(self.bvps)

        IMF = np.transpose(emd.sift.sift(x_filt)[:,:nIMF+1])
        if IMF.shape[0] < nIMF+1:
            IMF_residual = np.zeros((nIMF+1-IMF.shape[0], self.wsize))
            IMF = np.vstack([IMF, IMF_residual])

        return IMF

    def extract_RIVs_from_SSA(self, nGroups=3):
        from pyts.decomposition import SingularSpectrumAnalysis
        
        sig = self.bvps.reshape(1,-1)
        
        ssa = SingularSpectrumAnalysis(window_size=self.wsize, groups=nGroups)
        return ssa.fit_transform(sig)[0]


    def visualize_RIV(self, peak_interval_res, peak_val_max_res, amplitude_max_res):
        plt.figure()
        plt.subplot(3,1,1)
        #plt.plot(np.linspace(0, peak_interval_res.size, peak_interval_res.size), peak_interval_res);
        plt.plot(peak_interval_res)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$PP(t)$')
        plt.title(r'Pulse Peak-Peak periods (tachogram) -> RIFV information')
        
        plt.subplot(3,1,2)
        #plt.plot(np.linspace(0, peak_val_max_res.size, peak_val_max_res.size), peak_val_max_res)
        plt.plot(peak_val_max_res);
        plt.xlabel('$t$')
        plt.ylabel(r'$Max[pPPG(t)]$')
        plt.title(r'Pulse maximum intensity -> RIIV information')
        
        plt.subplot(3,1,3) 
        #plt.plot(np.linspace(0, amplitude_max_res.size, amplitude_max_res.size), amplitude_max_res)
        plt.plot(amplitude_max_res);
        plt.xlabel('$t$')
        plt.ylabel(r'$Max[pPPG(t)]-Min[pPPG(t)]$')
        plt.title(r'Pulse amplitude interval -> RIAV information')
        plt.tight_layout()
        plt.show()
  