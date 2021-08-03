
import streamlit as ss

import pandas as pd
import numpy as np
import librosa 
import librosa.display as ld
import IPython.display as ipd
from scipy.io import wavfile

import matplotlib.pyplot as plt
import plotly.express as px  #plots and graphing lib
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import tensorflow
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image

target_dict = {0:"blues",1:"classical",2:"country",3:"disco",4:"hiphop",
                 5:"jazz",6:"metal",7:"pop",8:"reggae",9:"rock"}

#---------------------------QoL functions----------------------------------
def dic_maker_tuple(tuple_arr):
  """ takes in [(x,y),(a,b)]
      outputs {x:y,a:b} (basically some formatting to make life easier)
  """
  dict_ = {}
  for tuple_ in tuple_arr:
    dict_[target_dict[tuple_[0]]] = tuple_[1]
  return dict_
def dic_maker(arr):
  """ dis takes in arr [[prob(1),prob(2),prob(3)......prob(n)]]
   and outputs [(1,prob(1)),(2,prob(2))]
   (basically some formatting to make life easier)"""
  dict_ = {}
  for ind in range(len(arr[0])):
    dict_[ind] = arr[0][ind]
  return sorted(dict_.items(), key=lambda x: x[1],reverse=True)[:3]

def spect_no_gen_img(spec):
  """ 
  prediction happens in this function
  super important, takes in image_path (/content/test_1/test/111.jpg)
  outputs: {1:prob(1),2:prob(2)}
  """
  image_1 = tensorflow.keras.preprocessing.image.load_img(spec)
  input_arr = tensorflow.keras.preprocessing.image.img_to_array(image_1)
  input_arr = preprocess_input(input_arr)
  input_arr = tensorflow.image.resize(input_arr,size = (288,432))

  input_arr = tensorflow.expand_dims(input_arr, -1)
  input_arr = np.array([input_arr])  # Convert single image to a batch.
  predictions = model.predict(input_arr)
  return dic_maker_tuple(dic_maker(predictions))

def spectrographic_anal(wav_path,name = "test"):
  y, sr = librosa.load(wav_path)
  hiphop, _ = librosa.effects.trim(y)
  n_fft = 2048
  hop_length = 512
  D = np.abs(librosa.stft(hiphop, n_fft = n_fft,  hop_length = hop_length))
  DB = librosa.amplitude_to_db(D, ref = np.max)
  librosa.display.specshow(DB, sr = sr, hop_length = hop_length)
  plt.savefig(f"{name}.png")
  plt.close()

  test_imgs = f"{name}.png"
  fig = make_subplots(rows = 1, cols = 2)
  pred_list = spect_no_gen_img(test_imgs)
  fig.append_trace(go.Image(z = np.array(image.load_img(test_imgs))),1,1)
  fig.append_trace(go.Bar(y = list(pred_list.keys()), x = list(pred_list.values()),orientation='h'),1,2)
  fig.update_layout(height= 400 , width = 1200, title_text="Spectrograph/prediction",showlegend = False)
  return fig
#--------streamlit starts here---------
ss.set_page_config(page_title = "Music Recognition", layout = "wide")
ss.title("Music Classification using Mel Spectrogram.")
model = tensorflow.keras.models.load_model("vgg19_head_50epochs.h5")
ss.markdown('Classifying Music genre using an acoustic time-frequency representation of a sound, a Mel Spectrogram.')

ss.markdown(
'''
### Conventional Way
Conventional way of representing Audio is amplitude wave plot which plots Amplitude with respect to time, which looks about like this.
''')
ss.image('amplitude graph.png')
ss.markdown('''
this is a Wave plot of a Classical music.
### Mel Spectrogram
  - An object of type MelSpectrogram represents an acoustic time-frequency representation of a sound: the power spectral density P(f, t). \n
  - It is sampled into a number of points around equally spaced times ti and frequencies fj (on a Mel frequency scale).
Now, lets take the same Example used above and plot its Mel spectrogram
''')
ss.image("spectrogram.png")

ss.markdown('- upload ur audio files IN .WAV FORMAT(~30Sec long preferably)')
wav_file = ss.file_uploader("drop the Audio file here: ", type = ["wav"])
if wav_file:
  ss.markdown('''
### Model Architecture and Dataset
This is a VGG19 with Custom head trained over 2k Wav files found [here](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification)
''')
  ss.image("model_architecture.png")

  ss.markdown('### Audio I/O')
  ss.audio(wav_file)
  preds = spectrographic_anal(wav_file)
  ss.plotly_chart(preds)

