# Phoneme Recognition

Simple implementation of DNN for phoneme recognition

## Dataset

* [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1)

## Requirement

* python 3.6.8
* tensorflow==1.13.1
* numpy
* matplotlib
* librosa

## Run

#### 1. Spikegram
```
python3 run_spikegram.py
```
#### 2. MFCC
```
python3 run_mfcc.py
```
#### 3. Spectrogram
```
python3 run_spectrogram.py
```
#### 4. Melspectrogram
```
python3 run_melspectrogram.py
```

## Result
#### 1. Broad class  

|   |   |Spikegram|MFCC|Spectrogram|Melspectrogram| 
|:---:|:---:|:---:|:---:|:---:|:---:|
|Obstruent|Stops|57.38|50.97|50.45|49.18|
|Obstruent|Affricate|42.19|30.61|36.06|35.74|
|Obstruent|Fricative|70.66|66.93|66.98|67.23|
|Sonorant|Glides|55.14|56.59|55.98|55.43|
|Sonorant|Nasals|59.15|62.36|60.39|60.09|
|Sonorant|Vowels|53.05|53.38|52.44|53.70|
|Others||92.24|91.96|91.79|91.94|

#### 2. Voice & voiceless

|        |Spikegram|MFCC|Spectrogram|Melspectrogram| 
|:------:|:---:|:---:|:---:|:---:|
|Obstruent|65.76|61.06|61.23|61.06|
|Sonorant|54.12|54.99|53.99|54.75|
|Others|92.24|91.96|91.79|91.94|

#### 3. Non-mute & mute

||Spikegram|MFCC|Spectrogram|Melspectrogram| 
|:---:|:---:|:---:|:---:|:---:|
|Non mute|57.49|56.77|56.11|56.61|
|mute|92.24|91.96|91.79|91.94|

#### 4. Total

|        |Spikegram|MFCC|Spectrogram|Melspectrogram| 
|:------:|:---:|:---:|:---:|:---:|
|Total|65.26|65.50|64.96|65.37|



[detail](https://hanseokhyeon.github.io/phoneme-recognition/)

## Author

Han Seokhyeon