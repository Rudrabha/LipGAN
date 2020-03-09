from os import listdir, path
import numpy as np
import scipy
import cv2
import os, sys
import discriminator as md
import generator as mg
from keras.callbacks import ModelCheckpoint, Callback
from keras.utils import plot_model
from tqdm import tqdm
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
parser.add_argument('--logdir', type=str, help='Folder to store checkpoints & generated images', default='logs/')

parser.add_argument('--model', type=str, help='Model name to use: basic|residual', default='residual')
parser.add_argument('--resume_gen', help='Path to weight file to load into the generator', default=None)
parser.add_argument('--resume_disc', help='Path to weight file to load into the discriminator', default=None)
parser.add_argument('--checkpoint_freq', type=int, help='Frequency of checkpointing', default=1000)

parser.add_argument('--n_gpu', type=int, help='Number of GPUs to use', default=1)
parser.add_argument('--batch_size', type=int, help='Single GPU batch size', default=96)
parser.add_argument('--lr', type=float, help='Initial learning rate', default=1e-4)
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
	gen = mg.create_model_residual(args)
else:
	gen = mg.create_model(args)

disc = md.create_model(args)
comb = mg.create_combined_model(gen, disc, args)

if args.resume_gen:
	gen.load_weights(args.resume_gen)
	print('Resuming generator from : {}'.format(args.resume_gen))
if args.resume_disc:
	disc.load_weights(args.resume_disc)
	print('Resuming discriminator from : {}'.format(args.resume_disc))

args.batch_size = args.n_gpu * args.batch_size
train_datagen = datagen(args)

comb.summary()

for e in range(args.epochs):
	prog_bar = tqdm(range(len(args.all_images) // args.batch_size))
	disc_loss, unsync_loss, sync_loss, gen_loss_mae, gen_loss_adv = 0., 0., 0., 0., 0.
	prog_bar.set_description('Starting epoch {}'.format(e))
	for batch_idx in prog_bar:
		(dummy_faces, audio), real_faces = next(train_datagen)
		real = np.zeros((len(real_faces), 1))
		fake = np.ones((len(real_faces), 1))

		gen_fakes = gen.predict([dummy_faces, audio]) # predict fakes

		### Train discriminator
		if np.random.choice([True, False]):
			disc_loss += disc.train_on_batch([gen_fakes, audio], fake)
			unsync_loss += disc.test_on_batch([real_faces, np.roll(audio, 10, axis=0)], fake)
		else:
			disc_loss += disc.test_on_batch([gen_fakes, audio], fake)
			unsync_loss += disc.train_on_batch([real_faces, np.roll(audio, 10, axis=0)], fake)

		sync_loss += disc.train_on_batch([real_faces, audio], real)

		### Train generator
		total, mae, adv = comb.train_on_batch([dummy_faces, audio], [real_faces, real])
		gen_loss_mae += mae
		gen_loss_adv += adv

		prog_bar.set_description('Disc_loss: {}, Unsynced: {}, Synced: {}, MAE: {} Adv_loss: {}'.format(\
														round(disc_loss / (batch_idx + 1), 3), 
														round(unsync_loss / (batch_idx + 1), 3), 
														round(sync_loss / (batch_idx + 1), 3),
														round(gen_loss_mae / (batch_idx + 1), 3),
														round(gen_loss_adv / (batch_idx + 1), 3)))
		prog_bar.refresh()

		if (batch_idx + 1) % (args.checkpoint_freq // 10) == 0:
			if (batch_idx + 1) % args.checkpoint_freq == 0:
				disc.save(path.join(args.logdir, 'disc.h5'))
				gen.save(path.join(args.logdir, 'gen.h5'))
				comb.save(path.join(args.logdir, 'comb.h5'))

			collage = np.concatenate([dummy_faces[...,:3], real_faces, gen_fakes], axis=2)
			collage *= 255.
			collage = np.clip(collage, 0., 255.).astype(np.uint8)
			
			for i in range(len(collage)):
				cv2.imwrite(path.join(args.logdir, 'gen_faces/{}.jpg'.format(i)), collage[i])
