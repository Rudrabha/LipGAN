LipGAN
===================
*Generate realistic talking faces for any human speech and face identity.*

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-automatic-face-to-face-translation/talking-face-generation-on-lrw)](https://paperswithcode.com/sota/talking-face-generation-on-lrw?p=towards-automatic-face-to-face-translation)

[[Paper]](https://dl.acm.org/doi/10.1145/3343031.3351066) | [[Project Page]](http://cvit.iiit.ac.in/research/projects/cvit-projects/facetoface-translation)  | [[Demonstration Video]](https://www.youtube.com/watch?v=aHG6Oei8jF0&list=LL2W0lqk_iPaqSlgPZ9GNv6w)

![image](https://drive.google.com/uc?export=view&id=1Y2isqWhUmAeYhbwK54tIqYOX0Pb5oH9w)

# Important Update:
A new, improved work that can produce significantly more accurate and natural results is available here: https://github.com/Rudrabha/Wav2Lip

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
- Install necessary packages using `pip install -r requirements.txt`
- Install keras-contrib `pip install git+https://www.github.com/keras-team/keras-contrib.git`
- git clone `https://github.com/Rudrabha/LipGAN.git` (make sure to check out to branch `fully_pythonic`)

Alternatively, if you would like to try it on Google Colab, please refer to this [notebook](https://colab.research.google.com/drive/1NLUwupCBsB1HrpEmOIHeMgU63sus2LxP) [Credits: [Kirill](https://github.com/KirillR911)]

Getting the weights
----------
Download checkpoints of the folowing models into the `logs/` folder

- CNN Face detection using dlib: [Link](http://dlib.net/files/mmod_human_face_detector.dat.bz2)
- LipGAN [Google Drive](https://drive.google.com/file/d/1DtXY5Ei_V6QjrLwfe7YDrmbSCDu6iru1/view?usp=sharing)

Generating talking face videos using pretrained models (Inference)
-------

#### Usage #1: Generating correct lip motion on a random talking face video
Here, we are given an audio input and a video of an identity speaking something entirely different. LipGAN can synthesize the correct lip motion for the given audio and overlay it on the given video of the speaking identity (Example #1, #2 in the above image).

```bash
python batch_inference.py --checkpoint_path <saved_checkpoint> --model residual --face <random_input_video> --fps <fps_of_input_video> --audio <guiding_audio_wav_file> --results_dir <folder_to_save_generated_video>
```

The generated `result_voice.mp4` will contain the input video lip synced with the given input audio. Note that the FPS parameter is by default `25`, **make sure you set the FPS correctly for your own input video**.

#### Usage #2: Generating talking video from a single face image
Refer to example #3 in the above picture. Given an audio, LipGAN generates a correct mouth shape (viseme) at each time-step and overlays it on the input image. The sequence of generated mouth shapes yields a talking face video.
```bash
python batch_inference.py --checkpoint_path <saved_checkpoint> --model residual --face <random_input_face> --audio <guiding_audio_wav_file> --results_dir <folder_to_save_generated_video>
```
**Please use the --pads argument to correct for inaccurate face detections such as not covering the chin region correctly. This can improve the results further.** 
#### More options
```bash
python batch_inference.py --help
```
Training LipGAN
-------
We illustrate the training pipeline using the LRS2 dataset. Adapting for other datasets would involve small modifications to the code. 

###### LRS2 dataset folder structure
```
data_root (mvlrs_v1)
├── main, pretrain (we use only main folder in this work)
|	├── list of folders
|	│   ├── five-digit numbered video IDs ending with (.mp4)
```

### Preprocess the dataset
We use Python [Librosa](https://librosa.github.io/librosa/) library to save melspectrogram features and perform face detection using dlib.  

```bash
# Please copy the appropriate LRS2 split's filelist.txt to the filelists/ folder. Example below is shown for LRS2. 
python preprocess.py --split [train|pretrain|val] --videos_data_root mvlrs_v1/ --final_data_root <folder_to_store_preprocessed_files>

### More options while preprocessing (like number of workers, image size etc.)
python preprocess.py --help
```
###### Final preprocessed folder structure
```
data_root (mvlrs_v1)
├── main, pretrain (we use only main folder in this work)
|	├── list of folders
|	│   ├── folders with five-digit video IDs 
|	│   |	 ├── 0.jpg, 1.jpg .... (extracted face crops of each frame)
|	│   |	 ├── mels.npz, audio.wav (melspectrogram and raw audio of the whole video)
```

#### Train the generator only
As training LipGAN is computationally intensive, you can just train the generator alone for quick, decent results.  
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

License and Citation
----------
The software is licensed under the MIT License. Please cite the following paper if you have use this code:

```
@inproceedings{KR:2019:TAF:3343031.3351066,
  author = {K R, Prajwal and Mukhopadhyay, Rudrabha and Philip, Jerin and Jha, Abhishek and Namboodiri, Vinay and Jawahar, C V},
  title = {Towards Automatic Face-to-Face Translation},
  booktitle = {Proceedings of the 27th ACM International Conference on Multimedia}, 
  series = {MM '19}, 
  year = {2019},
  isbn = {978-1-4503-6889-6},
  location = {Nice, France},
   = {1428--1436},
  numpages = {9},
  url = {http://doi.acm.org/10.1145/3343031.3351066},
  doi = {10.1145/3343031.3351066},
  acmid = {3351066},
  publisher = {ACM},
  address = {New York, NY, USA},
  keywords = {cross-language talking face generation, lip synthesis, neural machine translation, speech to speech translation, translation systems, voice transfer},
}
```


Acknowledgements
----------
Part of the audio preprocessing code is taken from the [DeepVoice 3](https://github.com/r9y9/deepvoice3_pytorch) implementation. We thank the author for releasing their code.

