LipGAN
===================
*Generate realistic talking faces for any human speech and face identity.*

[[Paper]](https://dl.acm.org/doi/10.1145/3343031.3351066) | [[Project Page]](http://cvit.iiit.ac.in/research/projects/cvit-projects/facetoface-translation)  | [[Demonstration Video]](https://www.dailymotion.com/video/x7pxw54)

![image](https://drive.google.com/uc?export=view&id=1Y2isqWhUmAeYhbwK54tIqYOX0Pb5oH9w)
----------
 Features
---------
 - Can handle in-the-wild face poses and expressions.
 - Can handle speech in any language and is robust to background noise.
 - Paste faces back into the original video with minimal/no artefacts --- can potentially correct lip sync errors in dubbed movies! 
 - Complete multi-gpu training code, pre-trained models available.
 - Fast inference code to generate results from the pre-trained models

Prerequisites
-------------
- Python >= 3.5
- ffmpeg: `sudo apt-get install ffmpeg`
- Matlab R2016a (for audio preprocessing, this dependency will be removed in later versions)
- Install necessary packages using `pip install -r requirements.txt`
- Install keras-contrib `pip install git+https://www.github.com/keras-team/keras-contrib.git`

Getting the weights
----------
Download checkpoints of the folowing models into the `logs/` folder

- CNN Face detection using dlib: [Link](http://dlib.net/files/mmod_human_face_detector.dat.bz2)
- LipGAN [Google Drive](https://drive.google.com/open?id=1ZTIt0XII4ZPulMNZbq2yg0x7zQBG6n9e)

Generating talking face videos using pretrained models (Inference)
-------
LipGAN takes speech features in the form of MFCCs and we need to preprocess our input audio file to get the MFCC features. We use the `create_mat.m` script to create `.mat` files for a given audio. 
```bash
cd matlab
matlab -nodesktop
>> create_mat(input_wav_or_mp4_file, path_to_output.mat) # replace with file paths
>> exit
cd ..
```
#### Usage #1: Generating correct lip motion on a random talking face video
Here, we are given an audio input (as `.mat` MFCC features) and a video of an identity speaking something entirely different. LipGAN can synthesize the correct lip motion for the given audio and overlay it on the given video of the speaking identity (Example #1, #2 in the above image). 
```bash
python batch_inference.py --checkpoint_path <saved_checkpoint> --face <random_input_video> --fps <fps_of_input_video> --audio <guiding_audio_wav_file> --mat <mat_file_from_above> --results_dir <folder_to_save_generated_video>
```
The generated `result_voice.mp4` will contain the input video lip synced with the given input audio.

#### Usage #2: Generating talking video from a single face image
Refer to example #3 in the above picture. Given an audio, LipGAN generates a correct mouth shape (viseme) at each time-step and overlays it on the input image. The sequence of generated mouth shapes yields a talking face video.
```bash
python batch_inference.py --checkpoint_path <saved_checkpoint> --face <random_input_face> --audio <guiding_audio_wav_file> --mat <mat_file_from_above> --results_dir <folder_to_save_generated_video>
```

#### More options
```bash
python batch_inference.py --help
```
Training LipGAN
-------
We illustrate the training pipeline using the LRS2 dataset. Adapting for other datasets would involve small modifications to the code. 
#### Preprocess the dataset
For preprocessing we first use a Matlab code from https://github.com/Hangz-nju-cuhk/Talking-Face-Generation-DAVS
We use this code to generate the MFCC files for all the videos present in the dataset.

```bash
# Please copy the appropriate LRS2 split's filelist.txt to the filelists/ folder
cd preprocess
matlab -nodesktop
run savemfcc.m
```
We preprocess the video files by detecting faces using a face detector from dlib. 
```bash
# Please copy the appropriate LRS2 split's filelist.txt to the filelists/ folder 
python preprocess.py --split [train|pretrain|val] --videos_data_root <root_folder_of_LRS2> --final_data_root <folder_to_store_preprocessed_files>

### More options while preprocessing
python preprocess.py --help
```
#### Train the generator only
As training LipGAN is computationally intensive, we find that first training the generator separately makes the entire training pipeline fast.  
```bash
python train_unet.py --data_root <path_to_preprocessed_dataset>

### Extensive set of training options available. Please run and refer to:
python train_unet.py --help
```
#### Train LipGAN
```bash
python train.py --data_root <path_to_preprocessed_dataset>

### Extensive set of training options available. Please run and refer to:
python train.py --help
```
