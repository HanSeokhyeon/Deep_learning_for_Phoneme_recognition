import librosa
import time
import gzip
import pickle
from pathlib import Path
from util import *

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
            concatenate_feature(shared_data, "dataset/TIMIT/{}".format(filename))
            logger.info("{} {} complete".format(j, filename))

        data = np.concatenate(shared_data, axis=1)

        if i == 0:
            data_mean = np.mean(data, axis=1)
            data_std = np.std(data, axis=1)

        data_norm = np.transpose(normalize_data(x=data, data_mean=data_mean, data_std=data_std))

        parent = Path(__file__).parent.parent
        with gzip.open("{}/input/120_mfcc_{}.pickle".format(parent, file_type[i]), 'wb') as f:
            pickle.dump(data_norm, f, pickle.HIGHEST_PROTOCOL)

        logger.info("%s complete" % (file_type[i]))

        end = time.time() - start
        logger.info("time = %.2f" % end)


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


if __name__ == '__main__':
    make_mfcc_feature()
