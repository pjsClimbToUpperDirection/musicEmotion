#IMPORT THE LIBRARIES
import numpy as np

# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later.
import librosa
import librosa.display
from data_augmentation.augmentation import noise, pitch


# Zero crossing rate
def zcr(data, frame_length, hop_length):
    zcr=librosa.feature.zero_crossing_rate(data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(zcr) # (N,) shape 의 배열

# Root mean square
def rmse(data, frame_length=2048, hop_length=512):
    rmse=librosa.feature.rms(y=data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(rmse) # (N,) shape 의 배열

# Mel-Frequency Cepstral coefficient
def mfcc(data,sr, flatten:bool=True):
    mfcc=librosa.feature.mfcc(y=data,sr=sr)
    return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T) # (N,) shape 의 배열

# Combine all feature functions
def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    result=np.array([])

    # 각 1차원 array를 axis=1 기준으로 합침(이어붙임)
    result=np.hstack((result,
                      zcr(data,frame_length,hop_length),
                      rmse(data,frame_length,hop_length),
                      mfcc(data,sr)
                      ))
    return result

# Apply data augmentation and extract its features
def get_features(path,duration=28, offset=0.6):
    data,sr=librosa.load(path,duration=duration,offset=offset,mono=True)
    aud=extract_features(data)
    audio=np.array(aud)

    noised_audio=noise(data)
    aud2=extract_features(noised_audio)
    # (1, N) array 로 각 반환값을 변환 후 concatenate
    audio=np.vstack((audio,aud2)) # 증강된 데이터 추가(row wise stacking)

    pitched_audio=pitch(data,sr)
    aud3=extract_features(pitched_audio)
    # (1, N) array 로 각 반환값을 변환 후 concatenate
    audio=np.vstack((audio,aud3)) # 증강된 데이터 추가(row wise stacking)

    pitched_audio1=pitch(data,sr)
    pitched_noised_audio=noise(pitched_audio1)
    aud4=extract_features(pitched_noised_audio)
    # (1, N) array 로 각 반환값을 변환 후 concatenate
    audio=np.vstack((audio,aud4)) # 증강된 데이터 추가(row wise stacking)

    # 수평 스택 형식으로 쌓은 데이터를 증강한 뒤 원본과 함께 수직 스택 형식으로 쌓아 반환
    return audio

