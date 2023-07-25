import os
import numpy as np
from scipy import signal
import pickle
import utils
import errors
import cv2 as cv
from tqdm import tqdm
import sys, getopt

# Datasets class definitions

class DatasetBase:
	def __init__(self):
		self.data_dir = '/mnt/43fba879-48e4-4e4c-afb2-dcb7e861c868/sftp/datasets/'

	def load_dataset(self):
		raise NotImplementedError("Subclasses must implement load_datset method")

class BP4D(DatasetBase):
	def __init__(self):
		super().__init__()
		self.name = 'bp4d'
		self.path = self.data_dir + 'BP4Ddef/'
		self.fs_gt = 1000
		self.data = [] 

	def load_dataset(self):

		print('\nLoading dataset ' + self.name + '...')
		for sub in utils.sort_nicely(os.listdir(self.path)):
			sub_path = self.path + sub + '/'

			for trial in utils.sort_nicely(os.listdir(sub_path)):
				trial_path = sub_path + trial + '/'
				video_path = trial_path + 'vid.avi'

				if os.path.exists(video_path):
					d = {}
					d['video_path'] = video_path
					d['subject'] = sub
					d['trial'] = trial
					d['chest_rois'] = []
					d['face_rois'] = []
					d['rppg_obj'] = []
					d['gt'] = self.load_gt(trial_path)
					self.data.append(d)

		print('%d items loaded!' % len(self.data))

	def load_gt(self, trial_path):
		#Load GT
		gt = np.loadtxt(trial_path + "/Resp_Volts.txt")
		return gt

	def extract_ROI(self, video_path, region='chest'):
		if region == 'chest':
			rois, _, _ = utils.get_chest_ROI(video_path, self.name, mp_complexity=1, skip_rate=10)
		elif region == 'face':
			rois = utils.get_face_ROI(video_path)
		return rois

	def extract_rppg(self, video_path, method='cpu_CHROM'):
		from riv.resp_from_rPPG import RR_from_rPPG

		rppg_obj = RR_from_rPPG(video_path, method=method)
		rppg_obj.get_rPPG()
		return rppg_obj

class COHFACE(DatasetBase):
	def __init__(self):
		super().__init__()
		self.name = 'cohface'
		self.path = self.data_dir + 'cohface/data/'
		self.fs_gt = 32
		self.data = []

	def load_dataset(self):
		print('\nLoading dataset ' + self.name + '...')
		for sub in utils.sort_nicely(os.listdir(self.path)):
			sub_path = self.path + sub + '/'

			for trial in utils.sort_nicely(os.listdir(sub_path)):
				trial_path = sub_path + trial + '/'
				video_path = trial_path + 'data.avi'

				if os.path.exists(video_path):
					d = {}
					d['video_path'] = video_path
					d['subject'] = sub
					d['trial'] = trial
					d['chest_rois'] = []
					d['face_rois'] = []
					d['rppg_obj'] = []
					d['gt'] = self.load_gt(trial_path)
					self.data.append(d)

		print('%d items loaded!' % len(self.data)) 

	def load_gt(self, trial_path):
		import h5py

		#Load GT
		f = h5py.File(trial_path + '/data.hdf5', 'r')
		gt = np.array(f['respiration'])
		gt = gt[np.arange(0, len(gt), 8)] # ???
		return gt

	def extract_ROI(self, video_path, region='chest'):
		if region == 'chest':
			rois, _, _ = utils.get_chest_ROI(video_path, self.name, mp_complexity=1, skip_rate=10)
		elif region == 'face':
			rois = utils.get_face_ROI(video_path)
		return rois

	def extract_rppg(self, video_path, method='cpu_CHROM'):
		from riv.resp_from_rPPG import RR_from_rPPG

		rppg_obj =  RR_from_rPPG(video_path, method=method)
		rppg_obj.get_rPPG()
		return rppg_obj

class MAHNOB(DatasetBase):
	def __init__(self):
		super().__init__()
		self.name = 'mahnob'
		self.path = self.data_dir + 'MAHNOB/'
		self.data = []

	def load_gt(self, sbj_path):
		import pybdf
		for fn in os.listdir(sbj_path):
			if fn.endswith('.bdf'):
				break
		bdfRec = pybdf.bdfRecording(sbj_path + '/' + fn)
		rec = bdfRec.getData(channels=[44])
		self.fs_gt = bdfRec.sampRate[44]
		gt = np.array(rec['data'][0])
		return gt


	def load_dataset(self):
		print('\nLoading dataset ' + self.name + '...')
		for sub in utils.sort_nicely(os.listdir(self.path)):
			sub_path = self.path + sub + '/'
			for fn in os.listdir(sub_path):
				if fn.endswith('.avi'):
					break
			video_path = sub_path + fn

			if os.path.exists(video_path):
				d = {}
				d['video_path'] = video_path
				d['subject'] = sub
				d['chest_rois'] = []
				d['face_rois'] = []
				d['rppg_obj'] = []
				d['gt'] = self.load_gt(sub_path)
				self.data.append(d)

		print('%d items loaded!' % len(self.data)) 

	def extract_ROI(self, video_path, region='chest'):
		if region == 'chest':
			rois, _, _ = utils.get_chest_ROI(video_path, self.name, mp_complexity=1, skip_rate=10)
		elif region == 'face':
			rois = utils.get_face_ROI(video_path)
		return rois

	def extract_rppg(self, video_path, method='cpu_CHROM'):
		from riv.resp_from_rPPG import RR_from_rPPG

		rppg_obj =  RR_from_rPPG(video_path, method=method)
		rppg_obj.get_rPPG()
		return rppg_obj

# Methods class definitions

class MethodBase:
	def __init__(self):
		self.name = ''
		self.win_size = 30
		self.data_type = ''

	def process(self, data):
		# This class can be used to process either videos or ROIs
		raise NotImplementedError("Subclasses must implement process method")

# Deep models

class MTTS_CAN(MethodBase):
	def __init__(self):
		super().__init__()
		self.name = 'MTTS_CAN'
		self.batch_size = 100
		self.data_type = 'face'

	def process(self, data):
		from deep.MTTS_CAN.my_predict_vitals import predict_vitals

		resp = predict_vitals(frames=data['face_rois'], batch_size=self.batch_size)
		return resp

class BigSmall(MethodBase):
	def __init__(self):
		super().__init__()
		self.name = 'BigSmall'
		self.data_type = 'face'

	def process(self, data):
		from deep.BigSmall.predict_vitals import predict_vitals

		resp = predict_vitals(data['face_rois'])
		return resp

# Motion based

class OF_Deep(MethodBase):

	def __init__(self, model):
		super().__init__()
		self.name = 'OF_Deep' + ' ' + model
		self.data_type = 'chest'
		self.model = model
		self.ckpt = 'things'

	def process(self, data):
		import ptlflow
		import torch
		from PIL import Image
		from ptlflow.utils import flow_utils
		from ptlflow.utils.io_adapter import IOAdapter

		model = ptlflow.get_model(self.model, pretrained_ckpt=self.ckpt)
		device = torch.device("cuda")
		model.to(device)

		s = []
		for i, r in enumerate(data['chest_rois']):

			#hsize = 144
			#wpercent = hsize/float(r.size[1])
			#wsize = int((float(r.size[0])*float(wpercent)))
			newsize = (224, 144)

			r = r.resize(newsize)

			if(i == 0):
				prev = r
				io_adapter = IOAdapter(model, prev.size, cuda = True)
				continue

			curr = r

			images = [
				prev,
				curr
			]

			inputs = io_adapter.prepare_inputs(images)

			predictions = model(inputs)
			predictions = io_adapter.unpad_and_unscale(predictions)

			flow = predictions['flows'][0,0]
			flow = flow.permute(2, 1, 0)
			flow = flow.cpu().detach().numpy()

			vert = flow[...,1].flatten()

			s.append(np.median(vert))

			prev = curr

		s = np.asarray(s)

		del model
		torch.cuda.empty_cache()

		return s

class OF_Model(MethodBase):

	def __init__(self):
		super().__init__()
		self.name = 'OF_Model'
		self.data_type = 'chest'

	def process(self, data):
		from motion.motion import OF
		import cv2 as cv

		# convert rois to grayscale
		g_rois = [cv.cvtColor(np.asarray(x), cv.COLOR_RGB2GRAY) for x in data['chest_rois']];

		# estimate OF
		of, _ = OF(g_rois, data['fps'])
		return of

class DoF(MethodBase):

	def __init__(self):
		super().__init__()
		self.name = 'DoF'
		self.data_type = 'chest'

	def process(self, data):
		from motion.motion import DoF
		import cv2 as cv

		# convert rois to grayscale
		g_rois = [cv.cvtColor(np.asarray(x), cv.COLOR_RGB2GRAY) for x in data['chest_rois']];

		# estimate DoF
		dof, _ = DoF(g_rois, data['fps'])
		return dof

class profile1D(MethodBase):

	def __init__(self):
		super().__init__()
		self.name = 'profile1D'
		self.data_type = 'chest'

	def process(self, data):
		from motion.motion import profile1D
		import cv2 as cv

		# convert rois to grayscale
		g_rois = [cv.cvtColor(np.asarray(x), cv.COLOR_RGB2GRAY) for x in data['chest_rois']];

		# estimate profile1D
		profile, _ = profile1D(g_rois, data['fps'])
		return profile

# RIV based

class peak(MethodBase):

	def __init__(self):
		super().__init__()
		self.name = 'fiedler'
		self.data_type = 'rppg'

	def process(self, data):
		return data['rppg_obj'].extract_RIVs_from_peaks()

class morph(MethodBase):

	def __init__(self):
		super().__init__()
		self.name = 'ims'
		self.data_type = 'rppg'

	def process(self, data):
		return data['rppg_obj'].extract_RIVs_from_IMS()

class bss_ssa(MethodBase):

	def __init__(self):
		super().__init__()
		self.name = 'bss_ssa'
		self.data_type = 'rppg'
		self.nGroups = None

	def process(self, data):
		return data['rppg_obj'].extract_RIVs_from_SSA(self.nGroups)

class bss_emd(MethodBase):

	def __init__(self):
		super().__init__()
		self.name = 'bss_emd'
		self.data_type = 'rppg'
		self.nIMF = 4

	def process(self, data):
		return data['rppg_obj'].extract_RIVs_from_EMD(self.nIMF)

def evaluate(results_dir, metrics, win_size=30, visualize=False):
	print('\n> Loading extracted data from ' + results_dir + '...')

	method_metrics = {}

	files = utils.sort_nicely(os.listdir(results_dir))
	ofdeep_models = ['_raft', '_raft_small', '_gma', '_irr_pwc', '_lcv_raft', '_craft']

	for filepath in tqdm(files, desc="Processing files"):
		tqdm.write("> Processing file %s" % (filepath))

		if 'metrics' in filepath:
			continue

		# Open the file with pickled data
		file = open(results_dir + filepath, 'rb')
		data = pickle.load(file)
		file.close()

		# Extract ground truth data
		fs_gt = data['fs_gt']
		gt = data['gt']

		# Filter ground truth
		filt_gt = utils.filter_RW(gt, fs_gt)

		if win_size == 'video':
			ws = filt_gt.shape[1] / fs_gt
		else:
			ws = win_size

		tqdm.write("> Length: %.2f sec" % (len(gt) / int(fs_gt)))

		# Apply windowing to ground truth
		gt_win, t_gt = utils.sig_windowing(filt_gt, fs_gt, ws)

		# Extract ground truth RPM using Welch with (win_size/1.5)
		gt_rpm = utils.sig_to_RPM(gt_win, fs_gt, int(ws/1.5), 0.2, 0.5)

		# Extract estimation data
		fps = data['fps']

		for i, est in enumerate(data['estimates']):

			cur_method = est['method']
			if cur_method == 'OF_Deep':
				cur_method += ofdeep_models[i]
			elif cur_method == 'OF_Model':
				cur_method = 'OF_Farneback'

			sig = np.squeeze(est['estimate'])

			if win_size == 'video':
				if sig.ndim == 1:
					ws = len(sig) / fps
				else:
					ws = sig.shape[1] / fps
			else:
				ws = win_size

			if (sig.ndim == 1):
				sig = sig[np.newaxis,:]

			# Filter estimated signal over all dimensions
			filt_sig = []
			for d in range(sig.shape[0]):
				filt_sig.append(utils.filter_RW(sig[d,:], fps))

			filt_sig = np.vstack(filt_sig)

			if cur_method in ['bss_emd', 'bss_ssa']:
				filt_sig = utils.select_component(filt_sig, fps, int(ws/1.5), 0.2, 0.5)

			sig_rpm = []
			for d in range(filt_sig.shape[0]):
				# Apply windowing to the estimation
				sig_win, t_sig = utils.sig_windowing(filt_sig[d,:], fps, ws)
				# Extract estimated RPM
				sig_rpm.append(utils.sig_to_RPM(sig_win, fps, int(ws/1.5), 0.2, 0.5))

			sig_rpm = np.mean(sig_rpm, axis=0)

			e = errors.getErrors(sig_rpm, gt_rpm, t_sig, t_gt, metrics)

			method_metrics.setdefault(cur_method, []).append((e))

	if win_size == 'video':
		fn = 'metrics_1w.pkl'
	else:
		fn = 'metrics.pkl'
	# Save the results of the applied methods
	with open(results_dir + fn, 'wb') as fp:
		pickle.dump([metrics, method_metrics] , fp)
		print('> Metrics saved!\n')

def print_metrics(results_dir, unique_window=False):
	from prettytable import PrettyTable
	from errors import concordance_correlation_coefficient

	if unique_window:
		print("Considering one window per video\n")
		fn = 'metrics_1w.pkl'
	else:
		print("Considering time windowing per each video\n")
		fn = 'metrics.pkl'

	# Load the calculated metrics
	with open(results_dir + fn, 'rb') as f: 
		metrics, method_metrics = pickle.load(f)

	t = PrettyTable(['Method'] + metrics)

	for method, metrics_value in method_metrics.items():

		#import code; code.interact(local=locals())

		if unique_window:
			from errors import RMSEerror, MAEerror, MAPEerror, PearsonCorr, LinCorr
			bpmsEst = np.stack([np.squeeze(metric[-1][0]) for metric in metrics_value])[np.newaxis,:]
			bpmsGT = np.stack([np.squeeze(metric[-1][1]) for metric in metrics_value])			
			rmse = RMSEerror(bpmsEst, bpmsGT)
			mae = MAEerror(bpmsEst, bpmsGT)
			mape = MAPEerror(bpmsEst, bpmsGT)
			pcc = PearsonCorr(bpmsEst, bpmsGT)
			ccc = LinCorr(bpmsEst, bpmsGT)			
			vals = [rmse, mae, mape, pcc, ccc]
		else:
			vals = []
			for i, m in enumerate(metrics):
				avg = np.nanmedian([metric[i] for metric in metrics_value])
				std = np.nanstd([metric[i] for metric in metrics_value])
				vals.append(f"%.3f (%.2f)" % (float(avg), float(std)))

		t.add_row([method] + vals)

	print(t)

def extract_respiration(datasets, methods, results_dir):

	for dataset in datasets:

		dataset.load_dataset()
		# Loop over the dataset
		for d in tqdm(dataset.data, desc="Processing files"):

			if 'trial' in d.keys(): 
				outfilename = results_dir + dataset.name + '_' + d['subject'] + '_' + d['trial'] + '.pkl'
			else:
				outfilename = results_dir + dataset.name + '_' + d['subject'] + '.pkl'

			if os.path.exists(outfilename):
				tqdm.write("> File %s already exists! Skipping..." % outfilename)
				continue

			_, d['fps'] = utils.get_vid_stats(d['video_path'])

			results = {'video_path': d['video_path'],
					   'fps': d['fps'],
					   'gt' : d['gt'],
					   'fs_gt': dataset.fs_gt,
					   'estimates': [] }

			if 'trial' in d.keys(): 
				tqdm.write("> Processing video %s/%s\n> fps: %d" % (d['subject'], d['trial'], d['fps']))
			else:
				tqdm.write("> Processing video %s\n> fps: %d" % (d['subject'], d['fps']))

			# Apply every method to each video
			for m in methods:

				tqdm.write("> Applying method %s ..." % m.name)

		 		# If method process rois, extract them first
				if m.data_type == 'chest' and not d['chest_rois']:
					d['chest_rois'] = dataset.extract_ROI(d['video_path'], m.data_type)

				elif m.data_type == 'face' and not d['face_rois']:
					d['face_rois'] = dataset.extract_ROI(d['video_path'], m.data_type)
		 		
		 		# If method process rppg, extract it first
				elif m.data_type == 'rppg' and not d['rppg_obj']:
					d['rppg_obj'] = dataset.extract_rppg(d['video_path'])

				output = {'method': m.name,
						  'estimate': m.process(d)}

				results['estimates'].append(output)

			d['chest_rois'] = []	#release some memory
			d['face_rois'] = []	#release some memory

			# Save the results of the applied methods
			with open(outfilename, 'wb') as fp:
				pickle.dump(results, fp)
				tqdm.write('> Results saved!\n')

def main(argv):
	# Define the path where to save results
	results_dir = 'results/'
	what = 0 # 0: Estimate signals, 1: Perform results evalution, 2: Print metrics

	opts, args = getopt.getopt(argv,"ha:d:",["action=","dir="])
	for opt, arg in opts:
		if opt == '-h':
			print ('run_all.py -a <action> -d <results_dir>')
			sys.exit()
		elif opt in ("-a", "--action"):
			what = int(arg)
		elif opt in ("-d", "--dir"):
			results_dir = arg
	print ('Action is ', what)
	print ('Results dir is ', results_dir)

	if what == 0: 

		# Initialize a list of methods
		#methods = [peak(), morph(), bss_ssa(), bss_emd()]
		methods = [BigSmall(), MTTS_CAN()]

		# Initialize a list of datasets
		datasets = [BP4D(), COHFACE()]

		extract_respiration(datasets, methods, results_dir)

	elif what == 1:

		# Define list of metrics to evaluate
		metrics = ['RMSE', 'MAE', 'MAPE', 'PCC', 'CCC']

		evaluate(results_dir, metrics, win_size='video')

	elif what == 2:

		# Just print the metrics values
		print_metrics(results_dir, unique_window=True)


if __name__ == "__main__":
	main(sys.argv[1:])
