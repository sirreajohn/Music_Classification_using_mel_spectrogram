# Music_Classification_using_mel_spectrogram

Classifying Music genre using an acoustic time-frequency representation of a sound, a Mel Spectrogram.

**Deployed using streamlit [Here](https://share.streamlit.io/sirreajohn/music_classification_using_mel_spectrogram/music_recognition.py)**

## Description
### Conventional Way
Conventional way of representing Audio is amplitude wave plot which plots Amplitude with respect to time, which looks about like this.


![amp_img](https://github.com/sirreajohn/Music_Classification_using_mel_spectrogram/blob/master/amplitude%20graph.png)


this is a Wave plot of a Classical music.
### Mel Spectrogram
  - An object of type MelSpectrogram represents an acoustic time-frequency representation of a sound: the power spectral density P(f, t). \n
  - It is sampled into a number of points around equally spaced times ti and frequencies fj (on a Mel frequency scale).
Now, lets take the same Example used above and plot its Mel spectrogram.

![mel](https://github.com/sirreajohn/Music_Classification_using_mel_spectrogram/blob/master/spectrogram.png)

## Architecture Details
This is a VGG19 with Custom head trained over 2k Wav files found [here](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification)

![archi](https://github.com/sirreajohn/Music_Classification_using_mel_spectrogram/blob/master/model_architecture.png)

## Working/instructions

- Deployed model can be found [here](https://share.streamlit.io/sirreajohn/music_classification_using_mel_spectrogram/music_recognition.py)
- just upload .wav file of preferably 30sec length 

## Requirements
- Running locally you need,
```
Python==3.9


streamlit==0.85.1
pandas==1.3.0
plotly==5.1.0
scipy==1.7.0
librosa==0.8.1
matplotlib==3.4.2
numpy==1.19.5
tensorflow==2.5.0
ipython==7.26.0
```

or just run 
```
pip install requirements.txt
```

## Files
- Music_Classification.ipynb contains training details and model packing
- music_recognition.py contains deployment code 
- weights are in .h5 file
