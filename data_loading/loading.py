#IMPORT THE LIBRARIES
import pandas as pd
import os

# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later.
import librosa
import librosa.display

import kagglehub

def extract_numeric_part(file_path):
    return [int(s) for s in os.path.basename(file_path).split('.') if s.isdigit()][0]

def dataset_Loading():
    # Download latest version
    path = kagglehub.dataset_download("imsparsh/multimodal-mirex-emotion-dataset")
    dataset_path = path + '/dataset'
    print("Path to dataset files:", dataset_path)

    audio = dataset_path + '/Audio'
    music_directory = os.listdir(audio)
    print(music_directory)

    file_paths = []
    for file_name in music_directory:
        # Create the full file path using os.path.join()
        file_path = os.path.join(audio, file_name)

        # Add the file path to the list
        file_paths.append(file_path)

    # Sort the file paths based on the numeric values in the file names
    file_paths = sorted(file_paths, key=extract_numeric_part)
    # Print the list of file paths
    print(file_paths[0])

    txt_file_path = dataset_path + '/clusters.txt'

    # Create a list to store cluster numbers
    cluster_numbers = []

    # Read the txt file and extract cluster numbers
    with open(txt_file_path, 'r') as file:
        for line in file:
            # Assuming each line contains "Cluster" followed by a space and the cluster number
            if line.startswith('Cluster'):
                _, cluster_number = line.strip().split()
                cluster_numbers.append(int(cluster_number))

    # Print the list of cluster numbers
    print(cluster_numbers[170])

    file_paths_df = pd.DataFrame(file_paths, columns=['Path'])
    cluster_numbers_df = pd.DataFrame(cluster_numbers, columns=['Cluster'])
    data_path = pd.concat([cluster_numbers_df, file_paths_df], axis=1)

    print(data_path.Cluster.value_counts())

    print(data_path.head())
    print("______________________________________________")
    print(data_path.tail())
    print("_______________________________________________")
    #print(path_df.Emotions.value_counts())

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.title('Count of Emotions', size=16)
    sns.countplot(data=data_path, x='Cluster')  # Use x='Cluster' to specify the column name
    plt.ylabel('Count', size=12)
    plt.xlabel('Cluster', size=12)
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.show()

    data, sr = librosa.load(file_paths[0]) # sr -> sampling rate

    # CREATE LOG MEL SPECTROGRAM
    plt.figure(figsize=(10, 5))
    spectrogram = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128,fmax=8000)
    log_spectrogram = librosa.power_to_db(spectrogram) # 데시벨 단위 변환
    librosa.display.specshow(log_spectrogram, y_axis='mel', sr=sr, x_axis='time')
    plt.title('Mel Spectrogram ')
    plt.colorbar(format='%+2.0f dB')
    plt.show()


    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=30)

    # MFCC
    plt.figure(figsize=(16, 10))
    plt.subplot(3,1,1)
    librosa.display.specshow(mfcc, x_axis='time')
    plt.ylabel('MFCC')
    plt.colorbar()
    #plt.show()

    return data_path, data, sr # data, sr은 이하 과정을 보여주는 샘플



