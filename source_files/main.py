from feature_extraction import *
from dl_model import *


def main():
    test_file = "audio_files/Throes of Perdition.flac"
    hop_length = 512
    frame_size = 2048

    mel_spectrogram_extraction(test_file, hop_length, frame_size)

    mfcc_extraction(test_file)


if __name__ == '__main__':
    main()
