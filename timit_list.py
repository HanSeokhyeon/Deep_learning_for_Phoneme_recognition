import os
import numpy as np


def make_list():
    train_list = []
    search("dataset/spikegram_old/TRAIN", train_list)
    train_list = [fn.replace("_", "/") for fn in train_list]
    train_list = [fn[22:] for fn in train_list]
    np.savetxt("dataset/TRAIN_list.csv", train_list, fmt="%s" ,delimiter=',')

    test_developmentset_list = []
    search("dataset/spikegram_old/TEST_developmentset", test_developmentset_list)
    test_developmentset_list = [fn.replace("_", "/") for fn in test_developmentset_list]
    test_developmentset_list = [fn.replace("TEST/developmentset", "TEST") for fn in test_developmentset_list]
    test_developmentset_list = [fn[22:] for fn in test_developmentset_list]
    np.savetxt("dataset/TEST_developmentset_list.csv", test_developmentset_list, fmt="%s", delimiter=',')

    test_coreset_list = []
    search("dataset/spikegram_old/TEST_coreset", test_coreset_list)
    test_coreset_list = [fn.replace("_", "/") for fn in test_coreset_list]
    test_coreset_list = [fn.replace("TEST/coreset", "TEST") for fn in test_coreset_list]
    test_coreset_list = [fn[22:] for fn in test_coreset_list]
    np.savetxt("dataset/TEST_coreset_list.csv", test_coreset_list, fmt="%s", delimiter=',')

    return


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


if __name__ == '__main__':
    make_list()