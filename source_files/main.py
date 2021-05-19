from feature_extraction import *
from mfcc_dl_model import *


def main():
    dataset_path = "audio_files"
    hop_length = 512
    frame_size = 2048

    data_insert(dataset_path, hop_length, frame_size)

    data_preparation()


if __name__ == '__main__':
    main()
