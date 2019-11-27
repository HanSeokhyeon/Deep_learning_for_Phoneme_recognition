import matplotlib.pyplot as plt
import numpy as np


def plot(loss_tr, loss_val, acc_val, acc_test, filename):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax1.plot(loss_val, label='valid')
    ax1.plot(loss_tr, label='train')

    plt.legend()

    ax2 = ax1.twinx()

    ax2.set_ylabel("Accuracy")

    ax2.plot(acc_val, label='valid')

    plt.text(0.85, 0.5, "Test acc:{:.4f}".format(acc_test), ha='center', va='center', transform=ax2.transAxes)

    plt.grid()

    plt.show()

    fig.savefig(filename)

    return


def phonetic_accuracy(confusion_matrix):
    broad_class_shorter = [[32, 34, 36, 33, 35, 37, 21],
                           [22, 23],
                           [25, 26, 24, 29, 31, 28, 30],
                           [14, 15, 16, 17, 27],
                           [18, 19, 20],
                           [0, 1, 2, 8, 3, 7, 11, 9, 4, 10, 12, 6, 5, 13],
                           [38]]

    voice_class = [broad_class_shorter[0] + broad_class_shorter[1] + broad_class_shorter[2],
                   broad_class_shorter[3] + broad_class_shorter[4] + broad_class_shorter[5],
                   broad_class_shorter[6]]

    non_mute = voice_class[0] + voice_class[1]

    n_data = np.sum(confusion_matrix, axis=1)

    correct = confusion_matrix.diagonal()

    phone_acc = []

    for list_class in broad_class_shorter:
        actual_phone = np.sum(n_data[list_class])
        correct_phone = np.sum(correct[list_class])
        phone_acc.append(correct_phone / actual_phone)

    voice_acc = []

    for list_class in voice_class:
        actual_phone = np.sum(n_data[list_class])
        correct_phone = np.sum(correct[list_class])
        voice_acc.append(correct_phone / actual_phone)

    actual_phone_non_mute = np.sum(n_data[non_mute])
    correct_phone_non_mute = np.sum(correct[non_mute])
    phone_acc_non_mute = correct_phone_non_mute / actual_phone_non_mute

    acc = np.sum(correct) / np.sum(confusion_matrix)

    print("Obstruent - Stops : {:.4f}".format(phone_acc[0]))
    print("Obstruent - Affricate : {:.4f}".format(phone_acc[1]))
    print("Obstruent - Fricative : {:.4f}".format(phone_acc[2]))
    print("Sonorant - Glides : {:.4f}".format(phone_acc[3]))
    print("Sonorant - Nasals : {:.4f}".format(phone_acc[4]))
    print("Sonorant - Vowels : {:.4f}".format(phone_acc[5]))
    print("Others : {:.4f}".format(phone_acc[6]))
    print()
    print("Obstruent : {:.4f}".format(voice_acc[0]))
    print("Sonorant : {:.4f}".format(voice_acc[1]))
    print("Others : {:.4f}".format(voice_acc[2]))
    print()
    print("Non-mute : {:.4f}".format(phone_acc_non_mute))
    print("Mute : {:.4f}".format(phone_acc[6]))
    print()
    print("Total : {:.4f}".format(acc))

    return