import os
import numpy as np
import librosa
import time
import gzip
import pickle
import requests
import zipfile
import io
import shutil
from logger import *
from pathlib import Path

feature_frame = 400
the_hop_length = 160


def make_mfcc_feature():
    download_timit()

    start = time.time()

    file_type = ['TRAIN', 'TEST_developmentset', 'TEST_coreset']

    for i in range(3):
        input_filename = list(np.loadtxt("dataset/{}_list.csv".format(file_type[i]), delimiter=',', dtype=np.str))

        logger.info("Filename scan complete")

        shared_data = []

        for j, filename in enumerate(input_filename[:]):
            concatenate_feature(shared_data, filename)
            logger.info("{} {} complete".format(j, filename))

        data = np.concatenate(shared_data, axis=1)

        if i == 0:
            data_mean = np.mean(data, axis=1)
            data_std = np.std(data, axis=1)

        data_norm = np.transpose(normalize_data(x=data, data_mean=data_mean, data_std=data_std))

        parent = Path(__file__).parent.parent
        with gzip.open("{}/input/120_spectrogram_{}.pickle".format(parent, file_type[i]), 'wb') as f:
            pickle.dump(data_norm, f, pickle.HIGHEST_PROTOCOL)

        logger.info("%s complete" % (file_type[i]))

        end = time.time() - start
        logger.info("time = %.2f" % end)


def download_timit():
    if os.path.isdir('dataset/TIMIT'):
        logger.info("TIMIT already exists")
    else:
        logger.info("TIMIT downloading")
        r = requests.get('https://ndownloader.figshare.com/files/10256148')
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall('dataset')
        shutil.move('dataset/data/lisa/data/timit/raw/TIMIT', 'dataset')
        shutil.rmtree('dataset/data')


def search(dirname, input_filename):
    try:
        filenames = os.listdir(dirname)
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            if os.path.isdir(full_filename):
                search(full_filename, input_filename)
            else:
                ext = os.path.splitext(full_filename)[-1]
                if ext == '.PHN':
                    input_filename.append(full_filename[:-4])
                    # print(full_filename)
    except PermissionError:
        pass


def concatenate_feature(shared_data, filename):
    # data load
    phn, y = get_data(filename)

    if int(phn[-1][1]) > np.shape(y)[0]:
        end_file = np.shape(y)[0]
    else:
        end_file = int(phn[-1][1])

    y = y[int(phn[0][0]):end_file]

    feature = librosa.feature.mfcc(y,
                                   sr=16000,
                                   n_fft=feature_frame,
                                   hop_length=the_hop_length,
                                   n_mfcc=40,
                                   center=False)
    feature_delta = get_delta(feature, 2)
    feature_deltadelta = get_delta(feature_delta, 2)

    label = phn_label(phn=phn, frame=feature_frame, hop_length=the_hop_length, num_of_frame=feature.shape[1])
    label_idx = set_label_number(label)

    feature = np.concatenate((feature, feature_delta, feature_deltadelta, label_idx.reshape(1, -1)), axis=0)

    shared_data.append(feature)

    return


def get_data(filename):
    phn_filename = filename + ".PHN"
    wav_filename = filename + ".WAV"

    phn = np.loadtxt(phn_filename, dtype=np.unicode)
    y = np.fromfile(wav_filename, dtype=np.int16)
    y = (y[512:] + 0.5) / 32767.5

    return phn, y


def get_delta(x, N):
    pad_x = np.pad(x, ((0, 0), (N, N)), 'edge')
    delta = np.zeros(np.shape(x))
    iterator = [i + 1 for i in range(N)]
    for t in range(np.shape(x)[1]):
        tmp1, tmp2 = 0, 0
        for n in iterator:
            tmp1 += n * (pad_x[:, (t + N) + n] - pad_x[:, (t + N) - n])
            tmp2 += 2 * n * n
        delta[:, t] = np.divide(tmp1, tmp2)

    return delta


def normalize_data(x, data_mean, data_std):
    data = x
    data_std[data_std==0] = 0.00001
    data[:-1] -= data_mean[:-1, None]
    data[:-1] /= data_std[:-1, None]
    return data


def phn_label(phn, frame, hop_length, num_of_frame):
    label = np.empty(num_of_frame, dtype='U5')
    label_number = 0
    idx = int(phn[0][0])
    for i in range(num_of_frame):
        if int(phn[label_number][0]) <= idx < int(phn[label_number][1]):
            label[i] = phn[label_number][2]
        else:
            if idx - int(phn[label_number][1]) <= frame / 2:
                label[i] = phn[label_number][2]
                label_number += 1
            else:
                label_number += 1
                label[i] = phn[label_number][2]

        idx += hop_length
    return label


def set_label_number(label):
    phone_39set = {"iy": 0, "ih": 1, "ix": 1, "eh": 2, "ae": 3, "ah": 4, "ax": 4, "ax-h": 4, "uw": 5, "ux": 5, "uh": 6,
                   "aa": 7, "ao": 7, "ey": 8, "ay": 9, "oy": 10, "aw": 11, "ow": 12, "er": 13, "axr": 13,
                   "l": 14, "el": 14, "r": 15, "w": 16, "y": 17, "m": 18, "em": 18, "n": 19, "en": 19, "nx": 19,
                   "ng": 20, "eng": 20, "dx": 21, "jh": 22, "ch": 23, "z": 24, "s": 25, "sh": 26, "zh": 26,
                   "hh": 27, "hv": 27, "v": 28, "f": 29, "dh": 30, "th": 31, "b": 32, "p": 33, "d": 34, "t": 35,
                   "g": 36, "k": 37, "bcl": 38, "pcl": 38, "dcl": 38, "tcl": 38, "gcl": 38, "kcl": 38, "epi": 38,
                   "pau": 38, "h": 38, "q": 38}

    label_idx = np.zeros(len(label))
    for i in range(len(label)):
        label_idx[i] = phone_39set[label[i]]

    return label_idx


if __name__ == '__main__':
    make_mfcc_feature()
