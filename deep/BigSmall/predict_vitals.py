import torch
from BigSmall import BigSmall
import numpy as np
import cv2
from preprocess import preprocess_frames, extract_raw
import os
from scipy import signal
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from collections import OrderedDict

def format_data_shape(data):
    base_len = 3
    # reshape big data
    data_big = data[0]
    data_big = torch.swapaxes(data_big, 2, 4)
    N, D, C, H, W = data_big.shape
    data_big = data_big.view(N * D, C, H, W)
    # reshape small data
    data_small = data[1]
    data_small = torch.swapaxes(data_small, 2, 4)
    N, D, C, H, W = data_small.shape
    data_small = data_small.view(N * D, C, H, W)
    
    # If using temporal shift module
    #if self.using_TSM:
    data_big = data_big[:(N * D) // base_len * base_len]
    data_small = data_small[:(N * D) // base_len * base_len]
    data[0] = data_big
    data[1] = data_small
    return data


def send_data_to_device(data, device):
    big_data = data[0].to(device)
    small_data = data[1].to(device)
    data = (big_data, small_data)
    return data


def define_model():
    # BigSmall Model
    model = BigSmall(n_segment=3)
    frame_depth = 3
    base_len = 1 * frame_depth 
    return model

##########################################################
######################### CONFIGS ########################
##########################################################

# CONFIGURATION DICTIONARY
config_preprocess = dict()
# Data / Frame Processing
config_preprocess['BIG_DATA_TYPE'] = ["Standardized"] # Default: ["Standardized"]
config_preprocess['BIG_W'] = 144 # Default: 144
config_preprocess['BIG_H'] = 144 # Default: 144
config_preprocess['SMALL_DATA_TYPE'] = ["Normalized"] # Default: ["Normalized"]
config_preprocess['SMALL_W'] = 9 # Default: 9
config_preprocess['SMALL_H'] = 9 # Default: 9
# Resize Parameters
config_preprocess['DYNAMIC_DETECTION'] = False # Default: False
config_preprocess['DYNAMIC_DETECTION_FREQUENCY'] = False # Default: False
config_preprocess['LARGE_FACE_BOX'] = False # Default: False
config_preprocess['CROP_FACE'] = False # Default: False
config_preprocess['LARGE_BOX_COEF'] = False # Default: False
config_preprocess['DATASET'] = 'bp4d'


##########################################################
######################### MAIN ###########################
##########################################################


def predict_vitals(videoFileName):
    frames = extract_raw(videoFileName)
    big_clip, small_clip = preprocess_frames(frames, config_preprocess)
    data = [torch.Tensor(big_clip), torch.Tensor(small_clip)]

    """ Model evaluation on the testing dataset."""
    print("\n=== Loading pretrained weights ===\n")

    model_path = "checkpoints/BP4D_BigSmall_Multitask_Fold1.pth"
    print("Testing uses pretrained model!")
    print('Model path:', model_path)
    if not os.path.exists(model_path):
        raise ValueError("Inference model path error!")

    model = define_model() # define the model

    # LOAD ABOVED SPECIFIED MODEL FOR TESTING
    if torch.cuda.is_available():
        device = torch.device("cuda:0") # set device to primary GPU
    else:
        device = "cpu" # if no GPUs set device is CPU

    weights =  torch.load(model_path, map_location=torch.device(device))
    w = OrderedDict({k.replace('module.', ''): v  for k,v in weights.items()})
    model.load_state_dict(w)
    model = model.to(device)
    model.eval()

    print("\n=== Inference ===\n")

    # MODEL TESTING
    with torch.no_grad():
        # GATHER AND FORMAT BATCH DATA
        data = format_data_shape(data)
        data = send_data_to_device(data, device)

        # GET MODEL PREDICTIONS
        _, _, resp_out = model(data)

    return resp_out


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='processed video path')
    parser.add_argument('--sampling_rate', type=int, default = 30, help='sampling rate of your video')
    args = parser.parse_args()

    sample_data_path = args.video_path

    resp_out = predict_vitals(sample_data_path)

    print("\n=== Plotting ===\n")

    resp_sig = np.squeeze(resp_out.detach().cpu().numpy())
    b, a = butter(N=2, Wn=[0.1, 0.5], fs=25, btype='bandpass')
    filtered_sig = filtfilt(b, a, resp_sig)

    #WELCH Estimations params
    fps = 25
    win_size = 30 
    nyquistF_est = fps/2
    fRes = 0.1
    nFFT_est = max(2048, (60*2*nyquistF_est) / fRes)
    minF = 0
    maxF = 0.65

    F, P = signal.welch(filtered_sig, nperseg=win_size*fps, noverlap=fps*(win_size-1), fs=fps, nfft=nFFT_est)
    band = np.argwhere((F > 0.1) & (F < 0.65)).flatten()
    RR = F[band][np.argmax(P[band])] * 60

    plt.plot(filtered_sig)
    plt.grid()
    plt.figure()
    plt.plot(F,P)
    plt.grid()
    plt.axvline(x=0.1, ymin=0, ymax=1, c='r')
    plt.axvline(x=maxF, ymin=0, ymax=1, c='r')
    plt.xlim([0,maxF+0.5])
    plt.title("RR: "+str(round(RR,2))+" resp/min")
    plt.show()
