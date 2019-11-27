import os
import shutil
from logger import *


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


def change_TRAIN():
    input_filename = []
    search("dataset/spikegram_old/TRAIN", input_filename)

    for fn in input_filename:
        fname_list = fn.split('/')
        fname_list[-1] = fname_list[-1].split('_')
        fname_list = fname_list[:-1] + fname_list[-1]
        fname_list[1] = 'spikegram'

        speaker_dir = "/".join(fname_list[:5])
        if os.path.isdir(speaker_dir):
            logger.info("{} exists".format(speaker_dir))
        else:
            os.mkdir(speaker_dir)

        fname_list = "/".join(fname_list)

        spike_filename = "{}_spike_PSNR50.raw".format(fn)
        new_spike_filename = "{}_spike.raw".format(fname_list)

        num_filename = "{}_num_PSNR50.raw".format(fn)
        new_num_filename = "{}_num.raw".format(fname_list)

        phn_filename = "{}.PHN".format(fn)
        new_phn_filename = "{}.PHN".format(fname_list)

        shutil.copy(spike_filename, new_spike_filename)
        shutil.copy(num_filename, new_num_filename)
        shutil.copy(phn_filename, new_phn_filename)

        logger.info("{} copy".format(fname_list))


def change():
    file_type = ['TRAIN', 'TEST_developmentset', 'TEST_coreset']
    new_file_type = ['TRAIN', 'TEST', 'TEST']
    for type, new_type in zip(file_type, new_file_type):
        input_filename = []
        search("dataset/spikegram_old/{}".format(type), input_filename)

        for fn in input_filename:
            fname_list = fn.split('/')
            fname_list[-1] = fname_list[-1].split('_')
            fname_list = fname_list[:-1] + fname_list[-1]
            fname_list[1] = 'spikegram'
            fname_list[2] = new_type

            speaker_dir = "/".join(fname_list[:5])
            if os.path.isdir(speaker_dir):
                logger.info("{} exists".format(speaker_dir))
            else:
                os.mkdir(speaker_dir)

            fname_list = "/".join(fname_list)

            spike_filename = "{}_spike_PSNR50.raw".format(fn)
            new_spike_filename = "{}_spike.raw".format(fname_list)

            num_filename = "{}_num_PSNR50.raw".format(fn)
            new_num_filename = "{}_num.raw".format(fname_list)

            phn_filename = "{}.PHN".format(fn)
            new_phn_filename = "{}.PHN".format(fname_list)

            shutil.copy(spike_filename, new_spike_filename)
            shutil.copy(num_filename, new_num_filename)
            shutil.copy(phn_filename, new_phn_filename)

            logger.info("{} copy".format(fname_list))


if __name__ == '__main__':
    change()
