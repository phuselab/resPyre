from __future__ import division
import torch
import numpy as np
import mediapipe as mp
from scipy import signal
from matplotlib import pyplot as plt
from scipy.signal._arraytools import even_ext
from numpy.fft import rfft, irfft
from numpy import argmax, sqrt, mean, absolute, linspace, log10, logical_and, average, diff, correlate
from scipy.signal import blackmanharris, fftconvolve
import sys
from tqdm import tqdm
from PIL import Image
import cv2 as cv
import re 


def plot_time_and_freq(list_of_sigs):
    plt.figure()
    n_sigs = len(list_of_sigs)
    plt.subplot(2,n_sigs//2, 1)
    plt.plot(list_of_sigs[0], c='r')
    plt.grid()
    for i in range(2,n_sigs+1):
        plt.subplot(2,n_sigs//2, i)
        plt.plot(list_of_sigs[i-1], c='b')
        plt.grid()
    #WELCH
    #GT Welch plot
    fps = 1000
    win_size = 30 
    nyquistF = fps/2
    fRes = 0.1
    nFFT = max(2048, (60*2*nyquistF) / fRes)
    minF = 0.1
    maxF = 0.5
    plt.figure()
    plt.subplot(2,n_sigs//2,1)
    F, P = signal.welch(list_of_sigs[0], nperseg=win_size*fps, noverlap=fps*(win_size-1), fs=fps, nfft=nFFT)
    plt.plot(F,P)
    plt.axvline(x=0.1, ymin=0, ymax=1, c='r')
    plt.axvline(x=maxF, ymin=0, ymax=1, c='r')
    plt.xlim([0,maxF+0.5])
    plt.title("Max frequency GT: "+str(round(F[np.argmax(P)],2))+" Hz, "+str(round(F[np.argmax(P)]*60,2))+" resp/min")
    for i in range(2,n_sigs+1):
        fps = 25
        win_size = 30 
        nyquistF = fps/2
        fRes = 0.1
        nFFT = max(2048, (60*2*nyquistF) / fRes)
        plt.subplot(2,n_sigs//2,i)
        F, P = signal.welch(list_of_sigs[i-1], nperseg=win_size*fps, noverlap=fps*(win_size-1), fs=fps, nfft=nFFT)
        band = np.argwhere((F > minF) & (F < maxF)).flatten()
        plt.plot(F,P)
        plt.axvline(x=0.1, ymin=0, ymax=1, c='r')
        plt.axvline(x=maxF, ymin=0, ymax=1, c='r')
        plt.xlim([0,maxF+0.5])
        plt.title("Max frequency: "+str(round(F[band][np.argmax(P[band])],2))+" Hz, "+str(round(F[np.argmax(P[band])]*60,2))+" resp/min")
    plt.show()

def get_vid_stats(videoFileName):
    cap = cv.VideoCapture(videoFileName)
    fps = cap.get(cv.CAP_PROP_FPS)      # OpenCV v2.x used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    return duration, int(fps)

def sort_nicely(l): 
  """ Sort the given list in the way that humans expect. 
  """ 
  convert = lambda text: int(text) if text.isdigit() else text 
  alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
  l.sort( key=alphanum_key ) 
  return l

def extract_frames_yield(videoFileName):
    """
    This method yield the frames of a video file name or path.
    """
    vidcap = cv.VideoCapture(videoFileName)
    success, image = vidcap.read()
    while success:
        yield image
        success, image = vidcap.read()
    vidcap.release()

def detect_face(img):
    import mediapipe as mp
    image_height, image_width, _ = img.shape
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5) as face_detection:
        detection_result = face_detection.process(img)
    bbox = detection_result.detections[0].location_data.relative_bounding_box
    bbox_pxl = [bbox.xmin*image_width, bbox.ymin*image_height, bbox.width*image_width, bbox.height*image_height]
    xmin = bbox_pxl[0]
    xmax = xmin + bbox_pxl[2]
    ymin = bbox_pxl[1]
    ymax = ymin + bbox_pxl[3]
    centerx = xmax - (xmax - xmin) / 2
    centery = ymax - (ymax - ymin) / 2
    xdist = max(image_width-centerx, centerx)
    ydist = max(image_height-centery, centery)
    d = min(xdist, ydist)
    xmin = int(centerx - d)
    xmax = int(centerx + d)
    ymin = int(centery - d)
    ymax = int(centery + d)
    mybbox = [max(int(xmin), 0), min(int(xmax), img.shape[1]), max(int(ymin), 0), min(int(ymax), img.shape[0])]
    return mybbox

def get_face_ROI(video_path):
    import cv2
    print("\nExtracting face ROIs...")
    i = 0
    frames = []
    t = tqdm(extract_frames_yield(video_path))
    for frame in t:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if (i == 0):
            bbox = detect_face(frame)

        crp = frame[bbox[2]:bbox[3], bbox[0]:bbox[1], :]
        width = crp.shape[1]
        height = crp.shape[0]
        if width >= height:
            crp = crp[:, int(width/2)-int(height/2 + 1):int(height/2)+int(width/2), :]
        else:
            crp = crp[int((height-width)):,:,:]
        frames.append(crp)
        i += 1
    return frames

def get_chest_ROI(video_path, dataset, mp_complexity=2, skip_rate=1):
    print("\nExtracting ROIs...")

    _, fps = get_vid_stats(video_path)

    skip_rate *= fps
    i = 0
    mp_pose = mp.solutions.pose

    frames = []

    #Run MediaPipe Pose and draw pose landmarks.
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=mp_complexity) as pose:
        t = tqdm(extract_frames_yield(video_path))
        for frame in t:
            
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            
            if (i == 0):
            # Estrarre i landmark durante l'esecuzione potrebbe far cambiare la dimensione della ROI
            # e optical flow si arrabbia!!
            #if (i % skip_rate == 0):
                # Process frame with MediaPipe Pose.
                results = pose.process(frame)

            image_height, image_width, _ = frame.shape

            # Get landmark.
            if results.pose_landmarks is None:
                x_left = 0
                y_left = 0
                x_right = 0
                y_right = 0
                print("None landmark")

            else:
                # Get landmark.
                x_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width
                y_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height
                x_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width
                y_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height

            if (i == 0):
                patch_width = x_left - x_right
                patch_height = patch_width * 0.2 # height is 20% of width
                print(patch_width, patch_height)

            im = Image.fromarray(frame)

            left = max(x_right, 0)
            upper = min(y_right, y_left) - patch_height/2
            right = min(x_left, image_width)
            lower = min(min(y_right, y_left) + patch_height/2, image_height)

            if upper > image_height:
                upper = (image_height - 1) - patch_height
                lower = (image_height - 1)

            chest = im.crop(box=(left, upper, right, lower))

            #import code; code.interact(local=locals())

            # Crop chest ROIs
            # if(dataset == 'bp4d'):
            #     x_right = 0
            #     x_left = image_width-1
            #     y_right = image_height-200
            #     y_left = image_height-1
            #     chest = im.crop(box=(x_right, y_right, x_left, y_left))
            #     ### ??? ####
            #     newsize = (752, 144)
            

            # else:
            #     chest = im.crop(box=(round(x_right)+patch_width/6, image_height-patch_height, round(x_right)+5/6*patch_width, image_height))
            #     ### ??? ####
            #     newsize = (224, 144)
            #     chest = chest.resize(newsize)

            frames.append(chest)
            i += 1
    elapsed = t.format_dict["elapsed"]
    return frames, fps, elapsed

def Welch_rpm(resp, fps, winsize, minHz=0.1, maxHz=0.4, fRes=0.1):
    """
    This method computes the spectrum of a respiratory signal

    Parameters
    ----------
        resp: the respiratory signal
        fps: the fps of the video from which signal is estimated
        winsize: the window size used to compute spectrum
        minHz: the lower bound for accepted frequencies
        maxHz: the upper bound for accepted frequencies

    Returns
    -------
        the array of frequencies and the corrisponding PSD
    """
    step = 1
    nperseg=fps*winsize
    noverlap=fps*(winsize-step)

    nyquistF = fps/2
    nfft = max(2048, (60*2*nyquistF) / fRes)

    # -- periodogram by Welch
    F, P = signal.welch(resp, nperseg=nperseg, noverlap=noverlap, fs=fps, nfft=nfft)
    F = F.astype(np.float32)
    P = P.astype(np.float32)
    # -- freq subband
    band = np.argwhere((F > minHz) & (F < maxHz)).flatten()

    Pfreqs = 60*F[band]
    Power = P[:, band]

    return Pfreqs, Power

def sig_to_RPM(sig, fps, winsize, minHz=0.1, maxHz=0.4):
    sig = np.vstack(sig)

    Pfreqs, Power = Welch_rpm(sig, fps, winsize, minHz, maxHz)
    Pmax = np.argmax(Power, axis=1)  # power max
    rpm = Pfreqs[Pmax.squeeze()]

    if (rpm.size == 1):
        return rpm.reshape(1, -1)

    return rpm

def select_component(sig, fps, winsize, minHz=0.1, maxHz=0.4):
    
    cur_pMax = 0

    for d in range(sig.shape[0]):
        Pfreqs, Power = Welch_rpm(sig[d,:][np.newaxis,:], fps, winsize, minHz, maxHz)
        pMax = np.max(Power, axis=1)  # power max
        
        if pMax > cur_pMax:
            cur_pMax = pMax
            cur_d = d

    return sig[cur_d, :][np.newaxis,:]


def average_filter(sig, win_length = 5):
    """
    This method applies to a signal an average filter

    Parameters
    ----------
        sig: the respiratory signal
        win_length: the length of the window used to apply the average filter

    Returns
    -------
        the filtered signal
    """
    res = []
    sig = even_ext(np.array(sig), win_length, axis=-1)
    for i in np.arange(win_length, len(sig)-win_length+1):
        window = np.sum(sig[i-win_length:i+win_length])
        res.append(1/(1+2*win_length)*window)
    return res

def filter_RW(sig, fps, lo=0.1, hi=0.5):
    """
    This method performs posptprocessing steps of fiedler methods; the postprocessing process performs on the signal a normalization, computes the gradient of the signal and applies a band-pass filter

    Parameters
    ----------
        sig: the considered signal
        fps : the fps of the considered video

    Returns
    -------
        the postprocessed signal
    """
    #sig = np.diff(np.asarray(sig), axis=0)
    #sig = np.squeeze(sig)
    if (sig.ndim == 1):
        sig = sig[np.newaxis,:]

    b, a = signal.butter(N=2, Wn=[lo, hi], fs=fps, btype='bandpass')
    filtered_sig = signal.filtfilt(b, a, sig)

    return filtered_sig

def butter_lowpass_filter(data, cutoff, fs, order=6):
    """
    This method applies to a signal a butter lowpass filter

    Parameters
    ----------
        data: the respiratory signal
        cutoff: the cutoff frequency
        fs: the sampling frequency
        order: the order of the filter

    Returns
    -------
        the filtered signal
    """
    b, a = signal.butter(order, cutoff, fs=fs, btype='low', analog=False)
    y = signal.lfilter(b, a, data)
    return y

def plot_mask(mask):
    """
    This method plots the mask given as input

    Parameters
    ----------
        mask: the input mask

    Returns
    -------
        the plotted mask
    """
    plt.imshow(mask, interpolation='nearest')
    plt.show()

def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.

    f is a vector and x is an index for that vector.

    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.

    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.

    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]

    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)

    """
    # Requires real division.  Insert float() somewhere to force it?
    xv = 1/2 * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4 * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

def freq_from_autocorr(sig, fs):
    """Estimate frequency using autocorrelation

    Pros: Best method for finding the true fundamental of any repeating wave,
    even with strong harmonics or completely missing fundamental

    Cons: Not as accurate, currently has trouble with finding the true peak

    """
    # Calculate autocorrelation and throw away the negative lags
    corr = fftconvolve(sig, sig[::-1], mode='full')
    corr = corr[int(len(corr)/2):]

    # Find the first low point
    d = diff(corr)
    start, = np.nonzero(np.ravel(d > 0))
    start = start[0]

    # Find the next peak after the low point (other than 0 lag).  This bit is
    # not reliable, due to peaks that occur between samples.
    peak = argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)

    return fs / px

def freq_from_crossings(sig, fs):
    """Estimatcorr[len(corr)/2:]e frequency by counting zero crossings

    Pros: Fast, accurate (increasing with data length).  Works well for long low-noise sines, square, triangle, etc.

    Cons: Doesn't work if there are multiple zero crossings per cycle, low-frequency baseline shift, noise, etc.

    """
    # Find all indices right before a rising-edge zero crossing
    indices, = np.nonzero(np.ravel((sig[1:] >= 0) & (sig[:-1] < 0)))

    # Naive (Measures 1000.185 Hz for 1000 Hz, for instance)
    #crossings = indices

    # More accurate, using linear interpolation to find intersample
    # zero-crossings (Measures 1000.000129 Hz for 1000 Hz, for instance)
    crossings = [i - sig[i] / (sig[i+1] - sig[i]) for i in indices]

    # Some other interpolation based on neighboring points might be better. Spline, cubic, whatever

    return fs / average(diff(crossings))

def freq_from_fft(sig, fs):
    """Estimate frequency from peak of FFT

    Pros: Accurate, usually even more so than zero crossing counter
    (1000.000003 Hz for 1000 Hz, for instance).  Due to parabolic interpolation
    being a very good fit for windowed log FFT peaks?
    https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
    Accuracy also increases with data length

    Cons: Doesn't find the right value if harmonics are stronger than
    fundamental, which is common.  Better method would try to identify the fundamental

    """
    # Compute Fourier transform of windowed signal
    windowed = sig * blackmanharris(len(sig))
    f = rfft(windowed)

    # Find the peak and interpolate to get a more accurate peak
    i = argmax(abs(f)) # Just use this for less-accurate, naive version
    true_i = parabolic(abs(f), i)[0]

    # Convert to equivalent frequency
    return fs * true_i / len(windowed)

def snr(sig, fs, nperseg, noverlap):
    """
    This method computes the SNR of a signal

    Parameters
    ----------
        sig: the respiratory signal
        fs: the sampling frequency
        nperseg: the length of each segment
        noverlap: the number of points to overlap between segments

    Returns
    -------
        the SNR of the given signal
    """
    freqs, psd = signal.welch(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
    num = 0
    den = 0
    for i in np.arange(len(freqs)):
        if freqs[i]>=0.1 and freqs[i]<=0.4:
            num+=psd[i]
        if freqs[i]>=0 and freqs[i]<=4:
            den+=psd[i]
    if den!=0:
        return num/den
    else:

        return -1

def pad_rgb_signal(sig, fps, win_size):
    """
    This method applies padding to a windowed rgb signal

    Parameters
    ----------
        sig: the respiratory signal
        fps: the sampling frequency
        win_size: the length of each segment

    Returns
    -------
        The padded RGB respiratory signal
    """
    sig = np.swapaxes(sig,0,1)

    nperseg = fps * win_size

    new_sig = []
    for roi in sig:
        red = [frame[0] for frame in roi]
        green = [frame[1] for frame in roi]
        blue = [frame[2] for frame in roi]

        red = even_ext(np.asarray(red), int(nperseg//2), axis=-1)
        green = even_ext(np.asarray(green), int(nperseg//2), axis=-1)
        blue = even_ext(np.asarray(blue), int(nperseg//2), axis=-1)

        new_roi = []
        for i in np.arange(len(red)):
            new_roi.append([red[i], green[i], blue[i]])

        new_sig.append(new_roi)


    return np.swapaxes(new_sig,0,1)

def get_channel(sig, channel):
    """
    This method select from a windowed rgb signal a single channel

    Parameters
    ----------
        sig: the respiratory signal
        channel: the channel index (0:red, 1:green, 2:blue)

    Returns
    -------
        The signal resukting from the selection
    """
    res = []
    for win in sig:
        row = []
        for roi in win:
            row.append(roi[channel])
        res.append(row)
    return res

def get_SNR(RW, reference_rr, fps):
    '''Computes the signal-to-noise ratio of the BVP
    signals according to the method by -- de Haan G. et al., IEEE Transactions on Biomedical Engineering (2013).
    SNR calculated as the ratio (in dB) of power contained within +/- 0.1 Hz
    of the reference heart rate frequency and +/- 0.2 of its first
    harmonic and sum of all other power between 0.5 and 4 Hz.
    Adapted from https://github.com/danmcduff/iphys-toolbox/blob/master/tools/bvpsnr.m
    '''
   
    interv1 = 0.05*60
    
    #Estimations params
    win_size = 30 
    nyquistF_est = fps/2
    fRes = 0.1
    nFFT_est = max(2048, (60*2*nyquistF_est) / fRes)
    minF = 0.05
    maxF = 1.5
   
    F, P = signal.welch(RW, nperseg=win_size*fps, noverlap=fps*(win_size-1), fs=fps, nfft=nFFT_est)
    band = np.argwhere((F > minF) & (F < maxF)).flatten()
    pfreqs = 60*F[band]
    power = P[band]
    GTMask = np.logical_and(pfreqs>=reference_rr-interv1, pfreqs<=reference_rr+interv1)
    FMask = np.logical_not(GTMask)

    SPower = np.sum(power[GTMask])
    allPower = np.sum(power[FMask])
    snr = 10*np.log10(SPower/allPower)

    return snr

def _plot_PSD_snr(pfreqs, power, reference_rr, interv1):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.plot(pfreqs, np.squeeze(p))
    x1 = pfreqs[np.argmin(np.abs(pfreqs-curr_ref))]
    x2 = pfreqs[np.argmin(np.abs(pfreqs-curr_ref))]
    y1 = 0
    y2 = p[np.argmin(np.abs(pfreqs-curr_ref))]
    plt.plot([x1, x2], [y1, y2], color='r', linestyle='-', linewidth=2)
    
    x1 = pfreqs[np.argmin(np.abs(pfreqs-curr_ref-interv1))]
    x2 = pfreqs[np.argmin(np.abs(pfreqs-curr_ref-interv1))]
    y1 = 0
    y2 = p[np.argmin(np.abs(pfreqs-curr_ref-interv1))]
    plt.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=2)
    x1 = pfreqs[np.argmin(np.abs(pfreqs-curr_ref+interv1))]
    x2 = pfreqs[np.argmin(np.abs(pfreqs-curr_ref+interv1))]
    y1 = 0
    y2 = p[np.argmin(np.abs(pfreqs-curr_ref+interv1))]
    plt.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=2)
    plt.grid()
    plt.show()

def sig_windowing(sig, fps, wsize, stride=1):
    """ Performs signal windowing

    Args:
      sig (list/array): full signal
      fps       (float): frames per seconds      
      wsize     (float): size of the window (in seconds)
      stride    (float): stride (in seconds)

    Returns:
      win_sig (list): windowed signal
      timesES (list): times of (centers) windows 
    """
    sig = np.array(sig).squeeze()
    block_idx, timesES = sliding_straded_win_idx(sig.shape[0], wsize, stride, fps)
    sig_win  = []
    for e in block_idx:
        st_frame = int(e[0])
        end_frame = int(e[-1])
        wind_signal = np.copy(sig[st_frame: end_frame+1])
        sig_win.append(wind_signal[np.newaxis, :])

    return sig_win, timesES

def sliding_straded_win_idx(N, wsize, stride, fps):
    """
    This method is used to compute the indices for creating an overlapping windows signal.

    Args:
        N (int): length of the signal.
        wsize (float): window size in seconds.
        stride (float): stride between overlapping windows in seconds.
        fps (float): frames per seconds.

    Returns:
        List of ranges, each one contains the indices of a window, and a 1D ndarray of times in seconds, where each one is the center of a window.
    """
    wsize_fr = wsize*fps
    stride_fr = stride*fps
    idx = []
    timesES = []
    num_win = int((N-wsize_fr)/stride_fr)+1
    s = 0
    for i in range(num_win):
        idx.append(np.arange(s, s+wsize_fr))
        s += stride_fr
        timesES.append(wsize/2+stride*i)
    return idx, np.array(timesES, dtype=np.float32)


