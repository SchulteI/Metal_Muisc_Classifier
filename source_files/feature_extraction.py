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

    # add 2 new axis to make tensor input correct size for CNN such that the input channel size of 1 at axis index of 1
    log_spectrogram = log_spectrogram[np.newaxis, ...]
    np.insert(log_spectrogram, 0, np.size(new_file), axis=0)


def mfcc_extraction(input_file):

    # load audio file into librosa
    new_file, sr = librosa.load(input_file)

    # extract mel frequency cepstral coefficients to inform timbre
    mfccs = librosa.feature.mfcc(new_file, n_mfcc=13, sr=sr)

    # take the first and second derivative
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)

    # aggregate them into a single array for a more comprehensive analysis
    compiled_mfccs = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))

    compiled_mfccs = compiled_mfccs[np.newaxis, ...]
    np.insert(compiled_mfccs, 0, np.size(new_file), axis=0)
