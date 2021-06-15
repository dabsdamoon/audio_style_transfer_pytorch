# audio_style_transfer_pytorch

Implementation of Dmitry Ulyanov's neural-style-audio-tf (Audio style transfer) with Pytorch. </br>
Codes are heavily inspired by: </br>
Dmitry's code: https://github.com/DmitryUlyanov/neural-style-audio-tf </br>
Pytorch implementation of image transfer: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html </br>

## Requirements
* Python=3.8
* pytorch==1.8.1
* For others, see requirements.txt.

## Environments
For Linux envnironment, use conde below (You can use codes in pytorch.org for installation on your own os. </br>
<code>
  conda create -n {name of environment} python=3.8
</code>
</br>
<code>
  pip install -r requirements.txt
</code>
</br>
<code>
  conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
</code>

## Training
You have two options:</br>
1. If you want to just try one-by-one, just open jupyter notebook file ("Python3 Audio Style Transfer Run.ipynb") and follow the procedure. </br>
2. If you have multiple contents and styles, set working directorys for content files and style files in configs.py and run the code below: </br>
<code>
  python run_functions.py -p {which phase of spectrogram you are going to use} -s {tag when saving synthesized audio files}
</code> 

## Notes
There are some differences between Dmitry's code and mine that: </br>
  1. Instead of using tensorflow v1.0 like Dmitry, I've used latest pytorch (v1.8.1) for implementation. </br>
  2. I used Prem Seetharaman's STFT class constructed with pytorch. Thus, when synthesizing the sound, one can choose either pase of content or style for synthesis. </br>
  3. I've customized codes so that one can generate combinations of multiple content sounds and multiple style sounds. </br>

June 2021, Dabin Moon
