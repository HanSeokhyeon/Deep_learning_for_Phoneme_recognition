import librosa
import time
import gzip
import pickle
from logger import *
from pathlib import Path
from util import *

spike_frame = 2048 * 6
n_band = 32
n_time = 10
n_structure = 4

feature_frame = 400
the_hop_length = 160


def make_spikegram_feature():
    start = time.time()

    file_type = ['TRAIN', 'TEST_developmentset', 'TEST_coreset']

    for i in range(3):
        input_filename = list(np.loadtxt("dataset/{}_list.csv".format(file_type[i]), delimiter=',', dtype=np.str))

        logger.info("Filename scan complete")

        shared_data = []

        for j, filename in enumerate(input_filename[:]):
            concatenate_feature(shared_data, "dataset/spikegram/{}".format(filename))
            logger.info("{} {} complete".format(j, filename))

        data = np.concatenate(shared_data, axis=1)

        if i == 0:
            data_mean = np.mean(data, axis=1)
            data_std = np.std(data, axis=1)

        data_norm = np.transpose(normalize_data(x=data, data_mean=data_mean, data_std=data_std))

        parent = Path(__file__).parent.parent
        with gzip.open("{}/input/126_spikegram_{}.pickle".format(parent, file_type[i]), 'wb') as f:
            pickle.dump(data_norm, f, pickle.HIGHEST_PROTOCOL)

        logger.info("%s complete" % (file_type[i]))

        end = time.time() - start
        logger.info("time = %.2f" % end)


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
    raw_filename = filename + "_spike.raw"
    num_filename = filename + "_num.raw"

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
    # get spikegram_old by SNR
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
    feature_tmp = np.zeros(n_band+n_time)
    num_of_frame = int((y.shape[1] - frame) / hop_length + 1)
    start, end = 0, frame

    if y.shape[1] % frame != 0:
        y = np.pad(y, ((0, 0), (0, frame - y.shape[1] % frame)), 'constant', constant_values=0)

    for i in range(num_of_frame):
        feature_tmp[:n_band] = librosa.power_to_db(np.sum(y[:, start:end], axis=1)+1)
        tmp_sum = np.reshape(np.sum(y[:, start:end], axis=0), (n_time, -1))
        feature_tmp[n_band:] = librosa.power_to_db(np.sum(tmp_sum, axis=1)+1)
        start += hop_length
        end += hop_length
        feature.append(np.copy(feature_tmp.reshape(1, -1)))

    feature = np.concatenate(feature, axis=0).transpose()
    return feature


if __name__ == '__main__':
    make_spikegram_feature()
