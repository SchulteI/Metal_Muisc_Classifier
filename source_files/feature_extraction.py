import librosa
import numpy as np
import mysql.connector
import os


database = mysql.connector.connect(
    host='localhost',
    user='metallibrary',
    passwd='password',
    database='metal_features'
)


def data_insert(dataset_path, hop_length, frame_size):

    mfcc_cursor = database.cursor()
    mel_cursor = database.cursor()

    mfcc_cursor.execute('CREATE TABLE mfcc_dataset (song_id INT PRIMARY KEY AUTO_INCREMENT,'
                        'mfccs MEDIUMBLOB, genre VARCHAR(50), genre_id INT)')

    mel_cursor.execute('CREATE TABLE mel_dataset (song_id INT PRIMARY KEY AUTO_INCREMENT,'
                       'mel_spectrogram MEDIUMBLOB, genre VARCHAR(50), genre_id INT)')

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            semantic_label = dirpath.split("/")[-1]

            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # load audio file into librosa
                new_file, sr = librosa.load(file_path)

                # create mel spectrogram of the audio file Note: look into number of mel bands
                mel_spectrogram = librosa.feature.melspectrogram(new_file, sr=sr, n_fft=frame_size,
                                                                 hop_length=hop_length, n_mels=100)

                # apply db scaling to the mel spectrogram to make it more intuitive based on how humans perceive sound
                log_spectrogram = librosa.power_to_db(mel_spectrogram)

                # add 2 new axis to make tensor input correct size for CNN such that the input channel size of 1
                # at axis index of 1
                log_spectrogram = log_spectrogram[np.newaxis, ...]
                np.insert(log_spectrogram, 0, np.size(new_file), axis=0)
                string_spectrogram = np.ndarray.dumps(log_spectrogram)

                mel_cursor.execute('INSERT INTO mel_dataset (mel_spectrogram, genre, genre_id) VALUES '
                                   '(%s, %s, %s)', (string_spectrogram, semantic_label, i-1))

                # extract mel frequency cepstral coefficients to inform timbre
                mfccs = librosa.feature.mfcc(new_file, n_mfcc=13, sr=sr)

                # take the first and second derivative
                delta_mfccs = librosa.feature.delta(mfccs)
                delta2_mfccs = librosa.feature.delta(mfccs, order=2)

                # aggregate them into a single array for a more comprehensive analysis
                compiled_mfccs = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))

                compiled_mfccs = compiled_mfccs[np.newaxis, ...]
                np.insert(compiled_mfccs, 0, np.size(new_file), axis=0)
                string_mfcc = np.ndarray.dumps(compiled_mfccs)

                mfcc_cursor.execute('INSERT INTO mfcc_dataset (mfccs, genre, genre_id) VALUES '
                                    '(%s, %s, %s)', (string_mfcc, semantic_label, i - 1))
