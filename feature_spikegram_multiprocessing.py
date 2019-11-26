from multiprocessing import Process, Manager
from feature_spikegram import *
from logger import *

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

        logger.info("Filename scan complete")

        with Manager() as manager:
            shared_data = manager.list()
            processes = []

            for j, filename in enumerate(input_filename[:]):
                p = Process(target=concatenate_feature, args=(shared_data, filename,))
                processes.append(p)
                p.start()
                logger.info("{} {}\t-> feature".format(j, filename))

            for j, p in enumerate(processes):
                p.join()
                logger.info(str(j) + " feature -> list")

            shared_data = list(shared_data)

            end = time.time() - start
            logger.info("time = %.2f" % end)

            data = np.concatenate(shared_data, axis=1)

        if i == 0:
            data_mean = np.mean(data, axis=1)
            data_std = np.std(data, axis=1)

        data_norm = np.transpose(normalize_data(x=data, data_mean=data_mean, data_std=data_std))

        with gzip.open("feature/126_PSNR50_%s.pickle" % file_type[i], 'wb') as f:
            pickle.dump(data_norm, f, pickle.HIGHEST_PROTOCOL)

        logger.info("%s complete" % (file_type[i]))

    end = time.time() - start
    logger.info("time = %.2f" % end)


if __name__ == '__main__':
    make_spikegram_feature()
