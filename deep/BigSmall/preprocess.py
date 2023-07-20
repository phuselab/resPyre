import numpy as np
import cv2
from skimage.util import img_as_float
import math
from tqdm import tqdm


def resize(frames, dynamic_det, det_length,
            w, h, larger_box, crop_face, larger_box_size):
    """
    :param dynamic_det: If False, it will use the only first frame to do facial detection and
                        the detected result will be used for all frames to do cropping and resizing.
                        If True, it will implement facial detection every "det_length" frames,
                        [i*det_length, (i+1)*det_length] of frames will use the i-th detected region to do cropping.
    :param det_length: the interval of dynamic detection
    :param larger_box: whether enlarge the detected region.
    :param crop_face:  whether crop the frames.
    :param larger_box_size: the coefficient of the larger region(height and weight),
                        the middle point of the detected region will stay still during the process of enlarging.
    """
    if dynamic_det:
        det_num = math.ceil(frames.shape[0] / det_length)
    else:
        det_num = 1
    face_region = list()

    # obtain detection region. it will do facial detection every "det_length" frames, totally "det_num" times.
    for idx in range(det_num):
        if crop_face:
            pass
        else:
            # if crop_face:False, the face_region will be the whole frame, namely cropping nothing.
            face_region.append([0, 0, frames.shape[1], frames.shape[2]])
    face_region_all = np.asarray(face_region, dtype='int')
    resize_frames = np.zeros((frames.shape[0], h, w, 3))

    # if dynamic_det: True, the frame under processing will use the (i // det_length)-th facial region.
    # if dynamic_det: False, the frame will only use the first region obtrained from the first frame.
    for i in range(0, frames.shape[0]):
        frame = frames[i]
        if dynamic_det:
            reference_index = i // det_length
        else:
            reference_index = 0
        if crop_face:
            face_region = face_region_all[reference_index]
            frame = frame[max(face_region[1], 0):min(face_region[1] + face_region[3], frame.shape[0]),
                    max(face_region[0], 0):min(face_region[0] + face_region[2], frame.shape[1])]
        resize_frames[i] = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    return resize_frames


def extract_frames_yield(videoFileName):
    """
    This method yield the frames of a video file name or path.
    """
    vidcap = cv2.VideoCapture(videoFileName)
    success, image = vidcap.read()
    while success:
        yield image
        success, image = vidcap.read()
    vidcap.release()

def extract_raw(videoFileName):
    """
    Extracts raw frames from video.
    Args:
        videoFileName (str): video file name or path.
    Returns: 
        ndarray: raw frames with shape [num_frames, height, width, rgb_channels].
    """
    frames = []
    for frame in extract_frames_yield(videoFileName):
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))   # convert to RGB
    return np.array(frames)

'''
#OLD VERSION
def downsample_frame(frame, dim_h=144, dim_w=144, dataset='bp4d', face_detect=True):
    if not face_detect:
        if dataset == 'bp4d':
            if dim_h == dim_w: # square crop
                vidLxL = cv2.resize(img_as_float(frame[int((frame.shape[0]-frame.shape[1])):,:,:]), (dim_h,dim_w), interpolation=cv2.INTER_AREA)
            else:
                vidLxL = cv2.resize(img_as_float(frame), (dim_h,dim_w), interpolation=cv2.INTER_AREA)
        elif dataset == 'cohface':
            width = frame.shape[1]
            height = frame.shape[0]
            frame = frame[:, int(width/2)-int(height/2 + 1):int(height/2)+int(width/2), :]
            vidLxL = cv2.resize(img_as_float(frame), (dim_h,dim_w), interpolation=cv2.INTER_AREA)
    else:
        bbox = detect_face(frame)
        crp = frame[bbox[2]:bbox[3], bbox[0]:bbox[1], :]

        width = crp.shape[1]
        height = crp.shape[0]
        if width >= height:
            crp = crp[:, int(width/2)-int(height/2 + 1):int(height/2)+int(width/2), :]
        else:
            crp = crp[int((height-width)):,:,:] 
        vidLxL = cv2.resize(img_as_float(crp), (dim_h,dim_w), interpolation=cv2.INTER_AREA)

    #return cv2.cvtColor(vidLxL.astype('float32'), cv2.COLOR_BGR2RGB)
    return vidLxL.astype('float32')
'''

def downsample_frame(frame, dim_h=144, dim_w=144):
    vidLxL = cv2.resize(img_as_float(frame), (dim_h,dim_w), interpolation=cv2.INTER_AREA)
    return vidLxL.astype('float32')


def diff_normalize_data(data):
    """Difference frames and normalization data"""
    n, h, w, c = data.shape
    normalized_len = n - 1
    normalized_data = np.zeros((normalized_len, h, w, c), dtype=np.float32)
    for j in range(normalized_len - 1):
        normalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
                data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
    normalized_data = normalized_data / np.std(normalized_data)
    normalized_data[np.isnan(normalized_data)] = 0
    return normalized_data

def standardized_data(data):
    """Difference frames and normalization data"""
    data = data - np.mean(data)
    data = data / np.std(data)
    data[np.isnan(data)] = 0
    return data


def preprocess(frames, config_preprocess):
    #######################################
    ########## PROCESSING FRAMES ##########
    #######################################
    # RESIZE FRAMES TO BIG SIZE  (144x144 DEFAULT)
    frames = resize(
            frames,
            config_preprocess['DYNAMIC_DETECTION'], # dynamic face detection
            config_preprocess['DYNAMIC_DETECTION_FREQUENCY'], # how often to use face detection
            config_preprocess['BIG_W'], # Big width
            config_preprocess['BIG_H'], # Big height
            config_preprocess['LARGE_FACE_BOX'], # larger-than-face bounding box coefficient
            config_preprocess['CROP_FACE'], # use face cropping
            config_preprocess['LARGE_BOX_COEF']) # use larger-than-face bounding box

    # PROCESS BIG FRAMES
    big_data = list()
    for data_type in config_preprocess['BIG_DATA_TYPE']:
        f_c = frames.copy()
        if data_type == "Raw": # Raw Frames
            big_data.append(f_c[:-1, :, :, :])
        elif data_type == "Normalized": # Normalized Difference Frames
            big_data.append(diff_normalize_data(f_c))
        elif data_type == "Standardized": # Raw Standardized Frames
            big_data.append(standardized_data(f_c)[:-1, :, :, :])
        else:
            raise ValueError("Unsupported data type!")
    big_data = np.concatenate(big_data, axis=3)

    # PROCESS SMALL FRAMES
    small_data = list()
    for data_type in config_preprocess['SMALL_DATA_TYPE']:
        f_c = frames.copy()
        if data_type == "Raw": # Raw Frames
            small_data.append(f_c[:-1, :, :, :])
        elif data_type == "Normalized": # Normalized Difference Frames
            small_data.append(diff_normalize_data(f_c))
        elif data_type == "Standardized": # Raw Standardized Frames
            small_data.append(standardized_data(f_c)[:-1, :, :, :])
        else:
            raise ValueError("Unsupported data type!")
    small_data = np.concatenate(small_data, axis=3)

    # RESIZE SMALL FRAMES TO LOWER RESOLUTION (9x9 DEFAULT)
    small_data = resize(
            small_data,
            False,
            False,
            config_preprocess['SMALL_W'],
            config_preprocess['SMALL_H'],
            False,
            False,
            False)
    return np.array([big_data]), np.array([small_data])


def preprocess_frames(frames, config_preprocess):
    ppframes = []
    print("\n=== Preprocessing Video Frames ===\n")
    for img in tqdm(frames):
        dim_h = config_preprocess['BIG_H']
        dim_w = config_preprocess['BIG_W']
        vid_LxL = downsample_frame(img, dim_h=dim_h, dim_w=dim_w) # downsample frames (otherwise processing time becomes WAY TOO LONG)
        # clip image values to range (1/255, 1)
        vid_LxL[vid_LxL > 1] = 1
        vid_LxL[vid_LxL < 1./255] = 1./255
        vid_LxL = np.expand_dims(vid_LxL, axis=0)
        ppframes.append(vid_LxL)
    Xsub = np.vstack(ppframes)
    big_clip, small_clip = preprocess(Xsub, config_preprocess)
    return big_clip, small_clip


'''
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
config_preprocess['DATASET'] = 'cohface'



##########################################################
######################### MAIN ###########################
##########################################################

videoFileName = "data/data.avi"
frames = extract_raw(videoFileName)
big_clip, small_clip = preprocess_frames(frames, config_preprocess)

'''