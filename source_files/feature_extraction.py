import librosa
import numpy as np


def mel_spectrogram_extraction(input_file, hop_length, frame_size):

    # load audio file into librosa
    new_file, sr = librosa.load(input_file)

    # create mel spectrogram of the audio file Note: look into number of mel bands
    mel_spectrogram = librosa.feature.melspectrogram(new_file, sr=sr, n_fft=frame_size,
                                                     hop_length=hop_length, n_mels=100)

    # apply db scaling to the mel spectrogram to make it more intuitive based on how humans perceive sound
    log_spectrogram = librosa.power_to_db(mel_spectrogram)


def mfcc_extraction(input_file):

    # load audio file into librosa
    new_file, sr = librosa.load(input_file)

    # extract mel frequency cepstral coefficients to inform timbre
    mfccs = librosa.feature.mfcc(new_file, n_mfcc=13, sr=sr)

    # take the first and second derivative
    first_mfccs_derivative = librosa.feature.delta(mfccs)
    second_mfccs_derivative = librosa.feature.delta(mfccs, order=2)

    # aggregate them into a single array for a more comprehensive analysis
    compiled_mfccs = np.concatenate((mfccs, first_mfccs_derivative, second_mfccs_derivative))
