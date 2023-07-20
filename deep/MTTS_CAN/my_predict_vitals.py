import tensorflow as tf
import numpy as np
import scipy.io
import os
import sys
import argparse
from .model import Attention_mask, MTTS_CAN
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter
from .inference_preprocess import preprocess_frames#, detrend

def predict_vitals(frames, batch_size=100):
    img_rows = 36
    img_cols = 36
    frame_depth = 10
    model_checkpoint = 'deep/MTTS_CAN/mtts_can.hdf5'
    batch_size = batch_size
    #sample_data_path = video_path

    '''
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
      tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
      # Invalid device or cannot modify virtual devices once initialized.
      pass
    '''
    
    # Hide GPU from visible devices
    tf.config.set_visible_devices([], 'GPU')

    dXsub = preprocess_frames(frames, dim=36)
    print('dXsub shape', dXsub.shape)

    dXsub_len = (dXsub.shape[0] // frame_depth)  * frame_depth
    dXsub = dXsub[:dXsub_len, :, :, :]

    model = MTTS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
    model.load_weights(model_checkpoint)

    yptest = model.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=1)

    resp_pred = yptest[1]

    return resp_pred

"""
def predict_vitals(video_path, batch_size=100, plot=False):
    img_rows = 36
    img_cols = 36
    frame_depth = 10
    model_checkpoint = 'deep/MTTS_CAN/mtts_can.hdf5'
    batch_size = batch_size
    sample_data_path = video_path

    physical_devices = tf.config.list_physical_devices('GPU')
    try:
      tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
      # Invalid device or cannot modify virtual devices once initialized.
      pass

    dXsub, fs = preprocess_raw_video(sample_data_path, dim=36)
    print('dXsub shape', dXsub.shape)

    dXsub_len = (dXsub.shape[0] // frame_depth)  * frame_depth
    dXsub = dXsub[:dXsub_len, :, :, :]

    model = MTTS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
    model.load_weights(model_checkpoint)

    yptest = model.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=1)

    pulse_pred = yptest[0]
    pulse_pred = detrend(np.cumsum(pulse_pred), 100)
    [b_pulse, a_pulse] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    pulse_pred = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(pulse_pred))

    resp_pred = yptest[1]
    resp_pred = detrend(np.cumsum(resp_pred), 100)
    [b_resp, a_resp] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')
    resp_pred = scipy.signal.filtfilt(b_resp, a_resp, np.double(resp_pred))

    if plot:
        ########## Plot ##################
        plt.subplot(211)
        plt.plot(pulse_pred)
        plt.title('Pulse Prediction')
        plt.subplot(212)
        plt.plot(resp_pred)
        plt.title('Resp Prediction')
        plt.show()

    return pulse_pred, resp_pred

"""
