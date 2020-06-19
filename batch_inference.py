from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse
from scipy.io import loadmat
import dlib, json, h5py, subprocess
from tqdm import tqdm

from keras.callbacks import ModelCheckpoint
from keras.models import Model

parser = argparse.ArgumentParser(description='Code to generate talking face using LipGAN')

parser.add_argument('--checkpoint_path', type=str, help='Name of saved checkpoint to load weights from', required=True)
parser.add_argument('--face_det_checkpoint', type=str, help='Name of saved checkpoint for face detection', 
						default='logs/mmod_human_face_detector.dat')

parser.add_argument('--face', type=str, help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--audio', type=str, help='Filepath of wav file to use as raw audio source', required=True)
parser.add_argument('--mat', type=str, help='Filepath of preprocessed mat file for LipGAN', required=True)
parser.add_argument('--results_dir', type=str, help='Folder to save all results into', default='results/')

parser.add_argument('--static', type=bool, help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='FPS of input video, ignore if image', default=25., required=False)
parser.add_argument('--max_sec', type=float, help='If video, until how many seconds of the clip to use for inference?', default=240.)
parser.add_argument('--pads', nargs='+', type=int, default=[0, 0, 0, 0], help='Padding (top, bottom, left, right)')

parser.add_argument('--face_det_batch_size', type=int, help='Single GPU batch size for face detection', default=64)
parser.add_argument('--lipgan_batch_size', type=int, help='Single GPU batch size for LipGAN', default=256)
parser.add_argument('--n_gpu', help='Number of GPUs to use', default=1)

args = parser.parse_args()
args.img_size = 96

if args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
	args.static = True

def rect_to_bb(d):
	x = d.rect.left()
	y = d.rect.top()
	w = d.rect.right() - x
	h = d.rect.bottom() - y
	return (x, y, w, h)

def calcMaxArea(rects):
	max_cords = (-1,-1,-1,-1)
	max_area = 0
	max_rect = None
	for i in range(len(rects)):
		cur_rect = rects[i]
		(x,y,w,h) = rect_to_bb(cur_rect)
		if w*h > max_area:
			max_area = w*h
			max_cords = (x,y,w,h)
			max_rect = cur_rect	
	return max_cords, max_rect
	
def face_detect(images):
	detector = dlib.cnn_face_detection_model_v1(args.face_det_checkpoint)

	batch_size = args.face_det_batch_size

	predictions = []
	for i in tqdm(range(0, len(images), batch_size)):
		predictions.extend(detector(images[i:i + batch_size]))
	
	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for rects, image in zip(predictions, images):
		(x, y, w, h), max_rect = calcMaxArea(rects)
		if x == -1:
			results.append([None, (-1,-1,-1,-1), False])
			continue
		y1 = max(0, y + pady1)
		y2 = min(image.shape[0], y + h + pady2)
		x1 = max(0, x + padx1)
		x2 = min(image.shape[1], x + w + padx2)
		face = image[y1:y2, x1:x2, ::-1] # RGB ---> BGR

		results.append([face, (y1, y2, x1, x2), True])
	
	del detector # make sure to clear GPU memory for LipGAN inference
	return results 

def datagen(frames, mfccs):
	img_batch, mfcc_batch, frame_batch, coords_batch = [], [], [], []

	if not args.static:
		face_det_results = face_detect([f[...,::-1] for f in frames]) # BGR2RGB for CNN face detection
	else:
		face_det_results = face_detect([frames[0][...,::-1]])

	for i, m in enumerate(mfccs):
		if sum(np.isnan(m.flatten())) > 0:
			print('nan')
			continue

		idx = 0 if args.static else i%len(frames)
		frame_to_save = frames[idx].copy()
		face, coords, valid_frame = face_det_results[idx].copy()
		if not valid_frame:
			print ("Face not detected, skipping frame {}".format(i))
			continue

		face = cv2.resize(face, (args.img_size, args.img_size))

		img_batch.append(face)
		mfcc_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= args.lipgan_batch_size:
			img_batch, mfcc_batch = np.asarray(img_batch), np.asarray(mfcc_batch)

			img_masked = img_batch.copy()
			img_masked[:, args.img_size//2:] = 0

			img_batch = np.concatenate((img_batch, img_masked), axis=3) / 255.
			mfcc_batch = np.reshape(mfcc_batch, [len(mfcc_batch), mfcc_batch.shape[1], mfcc_batch.shape[2], 1])

			yield img_batch, mfcc_batch, frame_batch, coords_batch
			img_batch, mfcc_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mfcc_batch = np.asarray(img_batch), np.asarray(mfcc_batch)

		img_masked = img_batch.copy()
		img_masked[:, args.img_size//2:] = 0

		img_batch = np.concatenate((img_batch, img_masked), axis=3) / 255.
		mfcc_batch = np.reshape(mfcc_batch, [len(mfcc_batch), mfcc_batch.shape[1], mfcc_batch.shape[2], 1])

		yield img_batch, mfcc_batch, frame_batch, coords_batch

fps = float(args.fps)
step_size_in_ms = 1./fps*1000
window_size = 350
mfcc_chunk_size = 35
temp = step_size_in_ms/10
mfcc_step_size = int(temp)
video_step_size_in_ms = temp * 10 # for 25 fps video

if 'residual' in args.checkpoint_path:
	from generator import create_model_residual as create_model
else:
	from generator import create_model as create_model

def main():
	if args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
		full_frames = [cv2.imread(args.face)]
	else:
		video_stream = cv2.VideoCapture(args.face)
		length = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
		print("Number of frames in the input video: " + str(length))
		
		frames = []
		while 1:
			still_reading, frame = video_stream.read()
			if not still_reading:
				video_stream.release()
				break
			frames.append(frame)
			if len(frames) % 2000 == 0: print(len(frames))

			if len(frames) * (1./fps) >= args.max_sec: break

		full_frames = []
		ss = 0.
		es = (ss + (window_size / 1000.))
		mid_second = (ss + es) / 2.

		while int(mid_second * fps) < len(frames):
			full_frames.append(frames[int(mid_second * fps)])

			ss += (video_step_size_in_ms / 1000.)
			es = (ss + (window_size / 1000.))
			mid_second = (ss + es) / 2.

		print ("Number of frames to be used for inference: "+str(len(full_frames)))

	mfccs = loadmat(args.mat)['mfccs']
	mfcc_chunks = []

	i = 0
	time_ms = 0.
	frame_duration = (1./fps) * 1000.
	mfcc_mags = []
	while (i < (len(mfccs[0])-(mfcc_chunk_size - 1))):
		mfcc_chunks.append(mfccs[:,i:i + mfcc_chunk_size])
		mfcc_mags.append(np.sum(mfccs[:, i:i + mfcc_chunk_size]))
		time_ms += frame_duration
		i = int(time_ms // 10)

		if (time_ms / 1000.) > args.max_sec: break

	print ("Length of mfcc chunks: "+str(len(mfcc_chunks)))

	batch_size = args.lipgan_batch_size
	gen = datagen(full_frames.copy(), mfcc_chunks)

	for i, (img_batch, mfcc_batch, frames, coords) in enumerate(tqdm(gen, 
											total=int(np.ceil(float(len(mfcc_chunks))/batch_size)))):
		if i == 0:
			model = create_model(args)
			print ("Model Created")

			model.load_weights(args.checkpoint_path)
			print ("Model loaded")

			frame_h, frame_w = full_frames[0].shape[:-1]
			out = cv2.VideoWriter(path.join(args.results_dir, 'result.avi'), 
									cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

		pred = model.predict([img_batch, mfcc_batch])
		pred = pred * 255
		
		for p, f, c in zip(pred, frames, coords):
			y1, y2, x1, x2 = c
			p = cv2.resize(p, (x2 - x1, y2 - y1))

			f[y1:y2, x1:x2] = p
			out.write(f)
	out.release()
	command = 'ffmpeg -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, path.join(args.results_dir, 'result.avi'), 
														path.join(args.results_dir, 'result_voice.avi'))
	subprocess.call(command, shell=True)

if __name__ == '__main__':
	main()
