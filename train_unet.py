from os import listdir, path
import numpy as np
import scipy
import cv2
import os, sys
from generator import create_model_residual, create_model
from keras.callbacks import ModelCheckpoint, Callback
from glob import glob
import pickle, argparse

def frame_id(fname):
	return int(fname.split('.')[0])

def choose_ip_frame(frames, gt_frame):
	selected_frames = [f for f in frames if np.abs(frame_id(gt_frame) - frame_id(f)) >= 6]
	return np.random.choice(selected_frames)

def datagen(args):
	all_images = args.all_images
	batch_size = args.batch_size

	while(1):
		np.random.shuffle(all_images)
		batches = [all_images[i:i + args.batch_size] for i in range(0, len(all_images), args.batch_size)]

		for batch in batches:
			sys.stderr.flush()
			img_gt_batch = []
			img_ip_batch = []
			mfcc_batch = []
			
			for img_name in batch:
				gt_fname = os.path.basename(img_name)
				dir_name = img_name.replace(gt_fname, '')
				frames = [f for f in os.listdir(dir_name) if f.endswith('.jpg')]
				if len(frames) < 12:
					continue
				mfcc_fname = img_name[:-3]+"npz"
				try:
					mfcc = np.load(mfcc_fname)
				except:
					continue
				mfcc = mfcc['mfcc']
				if sum(np.isnan(mfcc.flatten())) > 0:
					continue

				mfcc_batch.append(mfcc)
				
				img_gt = cv2.imread(img_name)
				img_gt = cv2.resize(img_gt, (args.img_size, args.img_size))
				img_gt_batch.append(img_gt)

				ip_fname = choose_ip_frame(frames, gt_fname)
				img_ip = cv2.imread(os.path.join(dir_name, ip_fname))
				img_ip = cv2.resize(img_ip, (args.img_size, args.img_size))
				img_ip_batch.append(img_ip)

			img_gt_batch = np.asarray(img_gt_batch)
			img_ip_batch = np.asarray(img_ip_batch)
			mfcc_batch = np.expand_dims(np.asarray(mfcc_batch), 3)

			img_gt_batch_masked = img_gt_batch.copy()
			img_gt_batch_masked[:, args.img_size//2:,...] = 0.
			img_ip_batch = np.concatenate([img_ip_batch, img_gt_batch_masked], axis=3)
			
			yield [img_ip_batch/255.0, mfcc_batch], img_gt_batch/255.0

parser = argparse.ArgumentParser(description='Keras implementation of LipGAN')

parser.add_argument('--data_root', type=str, help='LRS2 preprocessed dataset root to train on', required=True)
parser.add_argument('--logdir', type=str, help='Folder to store checkpoints', default='logs/')

parser.add_argument('--model', type=str, help='Model name to use: basic|residual', default='residual')
parser.add_argument('--resume', help='Path to weight file to load into the model', default=None)
parser.add_argument('--checkpoint_name', type=str, help='Checkpoint filename to use while saving', default='unet_residual.h5')
parser.add_argument('--checkpoint_freq', type=int, help='Frequency of checkpointing', default=1000)

parser.add_argument('--n_gpu', type=int, help='Number of GPUs to use', default=1)
parser.add_argument('--batch_size', type=int, help='Single GPU batch size', default=96)
parser.add_argument('--lr', type=float, help='Initial learning rate', default=1e-3)
parser.add_argument('--img_size', type=int, help='Size of input image', default=96)
parser.add_argument('--epochs', type=int, help='Number of epochs', default=20000000)

parser.add_argument('--all_images', default='filenames.pkl', help='Filename for caching image paths')
args = parser.parse_args()

if path.exists(path.join(args.logdir, args.all_images)):
	args.all_images = pickle.load(open(path.join(args.logdir, args.all_images), 'rb'))
else:
	all_images = glob(path.join("{}/train/*/*/*.jpg".format(args.data_root)))
	pickle.dump(all_images, open(path.join(args.logdir, args.all_images), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
	args.all_images = all_images
	
print ("Will be training on {} images".format(len(args.all_images)))

if args.model == 'residual':
	model = create_model_residual(args)
else:
	model = create_model(args)

if args.resume:
	model.load_weights(args.resume)
	print('Resuming from : {}'.format(args.resume))

args.batch_size = args.n_gpu * args.batch_size
train_datagen = datagen(args)

class WeightsSaver(Callback):
    def __init__(self, N, weight_path):
        self.N = N
        self.batch = 0
        self.weight_path = weight_path

    def on_batch_end(self, batch, logs={}):
    	self.batch += 1
        if self.batch % self.N == 0:
            self.model.save_weights(self.weight_path)

callbacks_list = [WeightsSaver(args.checkpoint_freq, path.join(args.logdir, args.checkpoint_name))]
model.fit_generator(train_datagen, steps_per_epoch=len(args.all_images)//args.batch_size, 
					epochs=args.epochs, verbose=1, initial_epoch=0, callbacks = callbacks_list)

