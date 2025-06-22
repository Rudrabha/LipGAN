LipGAN
===================
*Generate realistic talking faces for any human speech and face identity.*

# Commercial Version

Create your first lipsync generation in minutes. Please note, the commercial version is of a much higher quality than the old open source model!

## Create your API Key

Create your API key from the [Dashboard](https://sync.so/keys). You will use this key to securely access the Sync API.

## Make your first generation

The following example shows how to make a lipsync generation using the Sync API.

### Python

#### Step 1: Install Sync SDK

```bash
pip install syncsdk
```

#### Step 2: Make your first generation

Copy the following code into a file `quickstart.py` and replace `YOUR_API_KEY_HERE` with your generated API key.

```python
# quickstart.py
import time
from sync import Sync
from sync.common import Audio, GenerationOptions, Video
from sync.core.api_error import ApiError

# ---------- UPDATE API KEY ----------
# Replace with your Sync.so API key
api_key = "YOUR_API_KEY_HERE" 

# ----------[OPTIONAL] UPDATE INPUT VIDEO AND AUDIO URL ----------
# URL to your source video
video_url = "https://assets.sync.so/docs/example-video.mp4"
# URL to your audio file
audio_url = "https://assets.sync.so/docs/example-audio.wav"
# ----------------------------------------

client = Sync(
    base_url="https://api.sync.so", 
    api_key=api_key
).generations

print("Starting lip sync generation job...")

try:
    response = client.create(
        input=[Video(url=video_url),Audio(url=audio_url)],
        model="lipsync-2",
        options=GenerationOptions(sync_mode="cut_off"),
        outputFileName="quickstart"
    )
except ApiError as e:
    print(f'create generation request failed with status code {e.status_code} and error {e.body}')
    exit()

job_id = response.id
print(f"Generation submitted successfully, job id: {job_id}")

generation = client.get(job_id)
status = generation.status
while status not in ['COMPLETED', 'FAILED']:
    print('polling status for generation', job_id)
    time.sleep(10)
    generation = client.get(job_id)
    status = generation.status

if status == 'COMPLETED':
    print('generation', job_id, 'completed successfully, output url:', generation.output_url)
else:
    print('generation', job_id, 'failed')
```

Run the script:

```bash
python quickstart.py
```

#### Step 3: Done!

It may take a few minutes for the generation to complete. You should see the generated video URL in the terminal post completion.

---

### TypeScript

#### Step 1: Install dependencies

```bash
npm i @sync.so/sdk
```

#### Step 2: Make your first generation

Copy the following code into a file `quickstart.ts` and replace `YOUR_API_KEY_HERE` with your generated API key.

```typescript
// quickstart.ts
import { SyncClient, SyncError } from "@sync.so/sdk";

// ---------- UPDATE API KEY ----------
// Replace with your Sync.so API key
const apiKey = "YOUR_API_KEY_HERE";

// ----------[OPTIONAL] UPDATE INPUT VIDEO AND AUDIO URL ----------
// URL to your source video
const videoUrl = "https://assets.sync.so/docs/example-video.mp4";
// URL to your audio file
const audioUrl = "https://assets.sync.so/docs/example-audio.wav";
// ----------------------------------------

const client = new SyncClient({ apiKey });

async function main() {
    console.log("Starting lip sync generation job...");

    let jobId: string;
    try {
        const response = await client.generations.create({
            input: [
                {
                    type: "video",
                    url: videoUrl,
                },
                {
                    type: "audio",
                    url: audioUrl,
                },
            ],
            model: "lipsync-2",
            options: {
                sync_mode: "cut_off",
            },
            outputFileName: "quickstart"
        });
        jobId = response.id;
        console.log(`Generation submitted successfully, job id: ${jobId}`);
    } catch (err) {
        if (err instanceof SyncError) {
            console.error(`create generation request failed with status code ${err.statusCode} and error ${JSON.stringify(err.body)}`);
        } else {
            console.error('An unexpected error occurred:', err);
        }
        return;
    }

    let generation;
    let status;
    while (status !== 'COMPLETED' && status !== 'FAILED') {
        console.log(`polling status for generation ${jobId}...`);
        try {
            await new Promise(resolve => setTimeout(resolve, 10000));
            generation = await client.generations.get(jobId);
            status = generation.status;
        } catch (err) {
            if (err instanceof SyncError) {
                console.error(`polling failed with status code ${err.statusCode} and error ${JSON.stringify(err.body)}`);
            } else {
                console.error('An unexpected error occurred during polling:', err);
            }
            status = 'FAILED';
        }
    }

    if (status === 'COMPLETED') {
        console.log(`generation ${jobId} completed successfully, output url: ${generation?.outputUrl}`);
    } else {
        console.log(`generation ${jobId} failed`);
    }
}

main();
```

Run the script:

```bash
npx tsx quickstart.ts -y
```

#### Step 3: Done!

You should see the generated video URL in the terminal.

---

## Next Steps

Well done! You've just made your first lipsync generation with sync.so!

Ready to unlock the full potential of lipsync? Dive into our interactive [Studio](https://sync.so/login) to experiment with all available models, or explore our [API Documentation](/api-reference) to take your lip-sync generations to the next level!

## Contact
- prady@sync.so
- pavan@sync.so
- sanjit@sync.so

# Non Commercial, Open-source version
[[Paper]](https://dl.acm.org/doi/10.1145/3343031.3351066) | [[Project Page]](http://cvit.iiit.ac.in/research/projects/cvit-projects/facetoface-translation)  | [[Demonstration Video]](https://www.youtube.com/watch?v=aHG6Oei8jF0&list=LL2W0lqk_iPaqSlgPZ9GNv6w)

![image](https://drive.google.com/uc?export=view&id=1Y2isqWhUmAeYhbwK54tIqYOX0Pb5oH9w)

# Important Update:
A new, improved work that can produce significantly more accurate and natural results on moving talking face videos is available here: https://github.com/Rudrabha/Wav2Lip

----------
**Code without MATLAB dependency is now available in `fully_pythonic` branch**. Note that the models in both the branches are not entirely identical and either one may perform better than the other in several cases. The model used at the time of the paper's publication is with the MATLAB dependency and this is the one that has been extensively tested. Please feel free to experiment with the `fully_pythonic` branch if you do not want to have the MATLAB dependency. 
**A Google Colab [notebook](https://colab.research.google.com/drive/1NLUwupCBsB1HrpEmOIHeMgU63sus2LxP) is also available for the `fully_pythonic` branch. [Credits: [Kirill](https://github.com/KirillR911)]**

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
The generated `result_voice.mp4` will contain the input video lip synced with the given input audio. Note that the FPS parameter is by default `25`, **make sure you set the FPS correctly for your own input video**. 

#### Usage #2: Generating talking video from a single face image
Refer to example #3 in the above picture. Given an audio, LipGAN generates a correct mouth shape (viseme) at each time-step and overlays it on the input image. The sequence of generated mouth shapes yields a talking face video.
```bash
python batch_inference.py --checkpoint_path <saved_checkpoint> --face <random_input_face> --audio <guiding_audio_wav_file> --mat <mat_file_from_above> --results_dir <folder_to_save_generated_video>
```
**Please use the --pads argument to correct for inaccurate face detections such as not covering the chin region correctly. This can improve the results further.** 
#### More options
```bash
python batch_inference.py --help
```
Training LipGAN
-------
We illustrate the training pipeline using the LRS2 dataset. Adapting for other datasets would involve small modifications to the code. 
### Preprocess the dataset
We need to do two things: (i) Save the MFCC features from the audio and (ii) extract and save the facial crops of each frame in the video. 

##### LRS2 dataset folder structure
```
data_root (mvlrs_v1)
├── main, pretrain (we use only main folder in this work)
|	├── list of folders
|	│   ├── five-digit numbered video IDs ending with (.mp4)
```
##### Saving the MFCC features
We use MATLAB to save the MFCC files for all the videos present in the dataset. Refer to the [fully_pythonic branch](https://github.com/Rudrabha/LipGAN/tree/fully_pythonic) if you do not want to use MATLAB.  

```bash
# Please copy the appropriate LRS2 train split's filelist.txt to the filelists/ folder. The example below is shown for LRS2.
cd matlab
matlab -nodesktop
>> preprocess_mat('../filelists/train.txt', 'mvlrs_v1/main/') # replace with appropriate file paths for other datasets.
>> exit
cd ..
```

##### Saving the Face Crops of all Video Frames
We preprocess the video files by detecting faces using a face detector from dlib. 
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
|	│   |	 ├── 0.npz, 1.npz .... (mfcc features corresponding to each frame)
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
Part of the MATLAB code is taken from the an implementation of the [Talking Face Generation](https://github.com/Hangz-nju-cuhk/Talking-Face-Generation-DAVS) implementation. We thank the authors for releasing their code.

