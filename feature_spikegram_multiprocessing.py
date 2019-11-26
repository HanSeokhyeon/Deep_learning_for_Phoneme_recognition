import os
import numpy as np
import librosa
import time
import gzip
import pickle
from multiprocessing import Process, Manager

spike_frame = 2048 * 6
n_band = 32
n_time = 10
n_feature = 3 * (n_band+n_time) + 1
n_structure = 4

feature_frame = 400
the_hop_length = 160


def make_spikegram_feature():
    start = time.time()

    file_type = ['TRAIN', 'TEST_developmentset', 'TEST_coreset']

    for i in range(3):
        input_filename = []
        dirname = "dataset/spikegram/%s/" % file_type[i]
        search(dirname, input_filename)

        print("Filename scan complete")

        with Manager() as manager:
            shared_data = manager.list()
            processes = []

            for j, filename in enumerate(input_filename[:]):
                p = Process(target=concatenate_feature, args=(shared_data, filename,))
                processes.append(p)
                p.start()
                print("{} {}\t-> feature".format(j, filename))

            for j, p in enumerate(processes):
                p.join()
                print(str(j) + " feature -> list")

            shared_data = list(shared_data)

            end = time.time() - start
            print("time = %.2f" % end)

            data = np.concatenate(shared_data, axis=1)

        if i == 0:
            data_mean = np.mean(data, axis=1)
            data_std = np.std(data, axis=1)

        data_norm = np.transpose(normalize_data(x=data, data_mean=data_mean, data_std=data_std))

        with gzip.open("feature/126_PSNR50_%s.pickle" % file_type[i], 'wb') as f:
            pickle.dump(data_norm, f, pickle.HIGHEST_PROTOCOL)

        print("%s complete" % (file_type[i]))

    end = time.time() - start
    print("time = %.2f" % end)


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
    phn, spikegram = get_data(filename)

    if int(phn[-1][1]) > np.shape(spikegram)[1]:
        end_file = np.shape(spikegram)[1]
    else:
        end_file = int(phn[-1][1])

    spikegram_cut = spikegram[:, int(phn[0][0]):end_file]

    feature = make_feature(y=spikegram_cut,
                           frame=feature_frame,
                           hop_length=the_hop_length)
    feature_delta = get_delta(feature, 2)
    feature_deltadelta = get_delta(feature_delta, 2)

    label = phn_label(phn=phn, frame=feature_frame, hop_length=the_hop_length, num_of_frame=feature.shape[1])
    label_idx = set_label_number(label)

    feature = np.concatenate((feature, feature_delta, feature_deltadelta, label_idx.reshape(1, -1)), axis=0)

    shared_data.append(feature)

    return


def get_data(filename):
    phn_filename = filename + ".PHN"
    raw_filename = filename + "_spike_PSNR50.raw"
    num_filename = filename + "_num_PSNR50.raw"

    phn = np.loadtxt(phn_filename, dtype=np.unicode)
    x = np.fromfile(raw_filename, dtype=np.float64)
    x = np.reshape(x, (-1, n_structure))
    num = np.fromfile(num_filename, dtype=np.int32)

    n_data = np.shape(num)[0]
    acc_num = [sum(num[:i]) for i in range(n_data + 1)]

    for k in range(n_data):
        x[acc_num[k]:acc_num[k + 1], 2] += k * spike_frame

    spikegram = get_spikegram(x=x, num=num, acc_num=acc_num, n_data=n_data)

    return phn, spikegram


def get_delay():
    gammatone_filter = np.fromfile("dataset/Gammatone_Filter_Order4.raw", dtype=np.float64)

    gammatone_filter = np.reshape(gammatone_filter, (n_band, -1))
    gammatone_filter = gammatone_filter[:, 1:-1]

    max_point = np.argmax(np.abs(gammatone_filter), axis=1)

    return max_point


max_point = get_delay()


def get_spikegram(x, num, acc_num, n_data):
    # get spikegram by SNR
    spikegram = np.zeros((n_band, spike_frame * n_data))
    for k in range(n_data):
        for n in range(num[k]):
            spikegram[int(x[acc_num[k] + n, 0])][int(x[acc_num[k] + n, 2])] \
                += np.abs(x[acc_num[k] + n, 1])

    for idx, point in enumerate(max_point):
        spikegram[idx, point:] = spikegram[idx, :-point]

    return spikegram


def make_feature(y, frame, hop_length):
    feature = []
    feature_tmp = np.zeros(n_band + n_time)
    num_of_frame = int((y.shape[1] - frame) / hop_length + 1)
    start, end = 0, frame

    if y.shape[1] % frame != 0:
        y = np.pad(y, ((0, 0), (0, frame - y.shape[1] % frame)), 'constant', constant_values=0)

    for i in range(num_of_frame):
        feature_tmp[:n_band] = librosa.power_to_db(np.sum(y[:, start:end], axis=1) + 1)
        tmp_sum = np.reshape(np.sum(y[:, start:end], axis=0), (n_time, -1))
        feature_tmp[n_band:] = librosa.power_to_db(np.sum(tmp_sum, axis=1) + 1)
        start += hop_length
        end += hop_length
        feature.append(np.copy(feature_tmp.reshape(1, -1)))

    feature = np.concatenate(feature, axis=0).transpose()
    return feature


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
    make_spikegram_feature()
