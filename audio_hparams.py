from tensorflow.contrib.training import HParams
from glob import glob
import os, pickle

def _get_image_list(split):
	pkl_file = 'logs/filenames_{}.pkl'.format(split)
	if os.path.exists(pkl_file):
		with open(pkl_file, 'rb') as p:
			return pickle.load(p)
	else:
		filelist = glob('../female_preprocessed/*/*.jpg')
		if split == 'train':
			filelist = filelist[:int(.9 * len(filelist))]
		else:
			filelist = filelist[int(.9 * len(filelist)):]

		with open(pkl_file, 'wb') as p:
			pickle.dump(filelist, p, protocol=pickle.HIGHEST_PROTOCOL)
		return filelist

# Default hyperparameters
hparams = HParams(
	num_mels=80,  # Number of mel-spectrogram channels and local conditioning dimensionality
	#  network
	rescale=True,  # Whether to rescale audio prior to preprocessing
	rescaling_max=0.9,  # Rescaling value

	# For cases of OOM (Not really recommended, only use if facing unsolvable OOM errors, 
	# also consider clipping your samples to smaller chunks)
	max_mel_frames=900,
	# Only relevant when clip_mels_length = True, please only use after trying output_per_steps=3
	#  and still getting OOM errors.
	
	# Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
	# It"s preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
	# Does not work if n_ffit is not multiple of hop_size!!
	use_lws=False,
	
	n_fft=800,  # Extra window size is filled with 0 paddings to match this parameter
	hop_size=200,  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
	win_size=800,  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
	sample_rate=16000,  # 16000Hz (corresponding to librispeech) (sox --i <filename>)
	
	frame_shift_ms=None,  # Can replace hop_size parameter. (Recommended: 12.5)
	
	# Mel and Linear spectrograms normalization/scaling and clipping
	signal_normalization=True,
	# Whether to normalize mel spectrograms to some predefined range (following below parameters)
	allow_clipping_in_normalization=True,  # Only relevant if mel_normalization = True
	symmetric_mels=True,
	# Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, 
	# faster and cleaner convergence)
	max_abs_value=4.,
	# max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not 
	# be too big to avoid gradient explosion, 
	# not too small for fast convergence)
	normalize_for_wavenet=True,
	# whether to rescale to [0, 1] for wavenet. (better audio quality)
	clip_for_wavenet=True,
	# whether to clip [-max, max] before training/synthesizing with wavenet (better audio quality)
	
	# Contribution by @begeekmyfriend
	# Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude 
	# levels. Also allows for better G&L phase reconstruction)
	preemphasize=True,  # whether to apply filter
	preemphasis=0.97,  # filter coefficient.
	
	# Limits
	min_level_db=-100,
	ref_level_db=20,
	fmin=55,
	# Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To 
	# test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
	fmax=7600,  # To be increased/reduced depending on data.
	
	# Griffin Lim
	power=1.5,
	# Only used in G&L inversion, usually values between 1.2 and 1.5 are a good choice.
	griffin_lim_iters=60,
	# Number of G&L iterations, typically 30 is enough but we use 60 to ensure convergence.
	###########################################################################################################################################
	
	# Model params
	builder='nyanko',
	downsample_step=4,
	max_positions=512,
	binary_divergence_weight=0.,
	priority_freq=3000,
	use_guided_attention=True,
    guided_attention_sigma=0.2,

	T=75,
	overlap=25,
	mel_overlap=80,
	mel_step_size=240,
	img_size=48,
	fps=25,
	#all_images=_get_image_list('train'),
	#all_test_images=_get_image_list('test'),
	n_gpu=1,
	resume=False,
	checkpoint_dir = 'checkpoints/',
	checkpoint_path=None,
	best_checkpoint_path='logs/best3d_75.h5',

	batch_size=64,
	adam_beta1=0.5,
    adam_beta2=0.9,
    adam_eps=1e-6,
    amsgrad=False,
    initial_learning_rate=5e-4,
    lr_schedule=None,#"noam_learning_rate_decay",
    lr_schedule_kwargs={},
    nepochs=2000,
    weight_decay=0.0,
    clip_thresh=0.1,
	num_workers=6,
	checkpoint_interval=1000,
    eval_interval=1000,
    save_optimizer_state=True,
)


def hparams_debug_string():
	values = hparams.values()
	hp = ["  %s: %s" % (name, values[name]) for name in sorted(values) if name != "sentences"]
	return "Hyperparameters:\n" + "\n".join(hp)
