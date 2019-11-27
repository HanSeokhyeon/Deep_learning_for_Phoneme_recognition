from multiprocessing import Process, Manager
from feature.mfcc import *


def make_mfcc_feature():
    download_timit()

    start = time.time()

    file_type = ['TRAIN', 'TEST_developmentset', 'TEST_coreset']

    for i in range(3):
        input_filename = list(np.loadtxt("dataset/{}_list.csv".format(file_type[i]), delimiter=',', dtype=np.str))

        logger.info("Filename scan complete")

        with Manager() as manager:
            shared_data = manager.list()
            processes = []

            for j, filename in enumerate(input_filename[:]):
                p = Process(target=concatenate_feature, args=(shared_data, "dataset/TIMIT/{}".format(filename,)))
                processes.append(p)
                p.start()
                logger.info("{} {}\t-> input".format(j, filename))

            for j, p in enumerate(processes):
                p.join()
                logger.info(str(j) + " input -> list")

            shared_data = list(shared_data)

            end = time.time() - start
            logger.info("time = %.2f" % end)

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


if __name__ == '__main__':
    make_mfcc_feature()
