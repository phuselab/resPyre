import cv2 as cv
import numpy as np
import mediapipe as mp
from scipy import signal
from scipy.signal import butter, filtfilt
from tqdm import tqdm
from PIL import Image
import time

def DoF(frames, fps, downsample_rate=1):
    """
        This method applies the Difference of Frames (DoF) algorithm for breath measurement

        Parameters
        ----------
            frames : the sequence of frames of the Region of Interest (chest) 

        Returns
        -------
            the estimated respiratory signal
    """
    print("\nEstimating Respiration Waveform via Difference of Frames (DoF)...\n")
    start = time.time()
    #frames_np = np.hstack([f.reshape(-1,1)[::downsample_rate] for f in frames])
    frames_np = np.array(frames).reshape(len(frames),-1)
    dof = np.diff(frames_np, axis=0)
    doft = (dof>100).astype(int)
    sig = np.sum(doft, axis=1)
    end = time.time()

    elapsed = end - start

    return sig, elapsed

def OF(frames, fps):
    """
        This method applies Lin et al. algorithm for breath measurement

        Parameters
        ----------
            frames : the sequence of frames of the Region of Interest (chest) 

        Returns
        -------
            the estimated respiratory signal
    """
    print("\nEstimating Respiration Waveform via Optical Flow (OF)...\n")
    median = []
    t = tqdm(frames)
    for i, curr in enumerate(t): 
        if i == 0:
            prev = curr
            continue
        # Calculates dense optical flow by Farneback method
        flow = cv.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        vert = flow[...,1].flatten()
        median.append(np.median(vert))
        prev = curr
    sig = np.array(median)
    elapsed = t.format_dict["elapsed"]
    
    return sig, elapsed


def profile1D(frames, fps):
    """
        This method applies Bartula et al. algorithm for breath measurement

        Parameters
        ----------
            frames : the sequence of frames of the Region of Interest (chest) 

        Returns
        -------
            the estimated respiratory signal
    """
    print("\nEstimating Respiration Waveform via Correlation of 1D profiles...\n")
    dcp = []    #derivatives of chest position

    t = tqdm(frames)
    for i, curr in enumerate(t): 
        currp = np.diff(0.5*(np.mean(curr, axis=1) + np.std(curr, axis=1)))
        if i == 0:
            prevp = currp 
            continue
        xcorr = np.correlate(currp, prevp, mode='same')
        disp = np.max(xcorr)
        dcp.append(disp)
        prevp = currp 

    sig = np.array(dcp)
    elapsed = t.format_dict["elapsed"]
    
    return sig, elapsed
