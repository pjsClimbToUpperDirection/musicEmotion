#IMPORT THE LIBRARIES
import pandas as pd
import numpy as np

import os
import sys

# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later.
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM,BatchNormalization , GRU
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD

from data_loading.loading import dataset_Loading
from data_augmentation.augmentation import noise, stretch, shift, pitch
from feature_extraction.featureExtraction import get_features

import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import tensorflow as tf
print("Done")

# 데이터셋을 불러오는 동작
data_path, data, sr = dataset_Loading()

# NORMAL AUDIO
import librosa.display
plt.figure(figsize=(12, 5))
librosa.display.waveshow(y=data, sr=sr)

# AUDIO WITH NOISE
x = noise(data)
plt.figure(figsize=(12,5))
librosa.display.waveshow(y=x, sr=sr)

# STRETCHED AUDIO
x = stretch(data,0.8)
plt.figure(figsize=(12, 5))
librosa.display.waveshow(y=x, sr=sr)

# SHIFTED AUDIO
x = shift(data)
plt.figure(figsize=(12,5))
librosa.display.waveshow(y=x, sr=sr)

# AUDIO WITH PITCH
x = pitch(data, sr)
plt.figure(figsize=(12, 5))
librosa.display.waveshow(y=x, sr=sr)


#
# feature extraction
#
import multiprocessing as mp
print("Number of processors: ", mp.cpu_count())

import timeit
from tqdm import tqdm

# Extract the features.
start = timeit.default_timer()
X,Y=[],[]
for path, emotion, index in tqdm(zip(data_path.Path,data_path.Cluster, range(data_path.Path.shape[0]))):
    features = get_features(path) # 여러 특징 추출법을 통해 반환된 값을 조합한 뒤 증강 기법 사용하여 성능 향상
    if index % 500 == 0:
        print(f'{index} audio has been processed')
    for i in features:
        X.append(i) # 동일한 emotion 라벨에 대하여 오리지널 및 증강된 데이터 append(총 4개)
        Y.append(emotion)
print('Done')
stop = timeit.default_timer()

print('Time: ', stop - start)

len(X), len(Y), data_path.Path.shape

# Save the features in a CSV file
Emotions = pd.DataFrame(X)
Emotions['Emotions'] = Y
Emotions.to_csv('emotion.csv', index=False) # 본 csv 파일 gitignore 처리
Emotions.head()
Emotions.tail()


# checking data integrity for feature preparation(데이터 무결성 확인)
print(Emotions.isna().any())
Emotions=Emotions.fillna(0) # nan 은 0으로 채움
print(Emotions.isna().any())
np.sum(Emotions.isna())


X = Emotions.iloc[: ,:-1].values # 데이터
Y = Emotions['Emotions'].values # (지도 학습을 위한)라벨

from sklearn.preprocessing import StandardScaler, OneHotEncoder
encoder = OneHotEncoder()
# 각 데이터에 대응되는 emotion 값을 1차원 배열로 만든 뒤(a -> [a]) one hot encoding
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

print("X.shape: ", X.shape)
print("Y.shape: ", Y.shape)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0,test_size=0.2, shuffle=True)
print("x_train.shape, y_train.shape, x_test.shape, y_test.shape :", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

#reshape for lstm
X_train = x_train.reshape(x_train.shape[0] , x_train.shape[1] , 1)
X_test = x_test.reshape(x_test.shape[0] , x_test.shape[1] , 1)

# scaling our data with sklearn's Standard scaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) # (x - u) / s, 이때 u는 평균, s는 표준편차(standard deviation)
x_test = scaler.transform(x_test)
print("x_train.shape, y_train.shape, x_test.shape, y_test.shape: ", x_train.shape, y_train.shape, x_test.shape, y_test.shape)


#
# 여기서부터 훈련 영역
#
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
model_checkpoint = ModelCheckpoint('best_model1_weights.h5', monitor='val_accuracy', save_best_only=True)

early_stop=EarlyStopping(monitor='val_acc',mode='auto',patience=5,restore_best_weights=True)
lr_reduction=ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.5,min_lr=0.00001)

x_traincnn =np.expand_dims(x_train, axis=2)
x_testcnn= np.expand_dims(x_test, axis=2)
print("x_traincnn.shape, y_train.shape, x_testcnn.shape, y_test.shape: ", x_traincnn.shape, y_train.shape, x_testcnn.shape, y_test.shape)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPool1D, Dropout, Flatten, Dense

model = Sequential([
    Conv1D(16, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),
    MaxPool1D(pool_size=5, strides=2, padding='same'),

    Conv1D(8,kernel_size=3,strides=1,padding='same',activation='relu'),
    BatchNormalization(),
    MaxPool1D(pool_size=3,strides=2,padding='same'),
    Dropout(0.3),  # Add dropout layer after the fifth max pooling layer

    Flatten(),
    Dense(8,activation='relu'),
    BatchNormalization(),
    Dense(5,activation='softmax')
])
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')
model.summary()

history=model.fit(x_traincnn, y_train,
                  epochs=30,
                  validation_data=(x_testcnn, y_test),
                  batch_size=16,
                  callbacks=[early_stop,lr_reduction,model_checkpoint])

print("Accuracy of our model on test data : " , model.evaluate(x_testcnn,y_test)[1]*100 , "%")


# predicting on test data.
pred_test0 = model.predict(x_testcnn)
y_pred0 = encoder.inverse_transform(pred_test0)
y_test0 = encoder.inverse_transform(y_test)

# Check for random predictions
df0 = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
df0['Predicted Labels'] = y_pred0.flatten()
df0['Actual Labels'] = y_test0.flatten()

df0.head(10)

from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test0, y_pred0)
plt.figure(figsize=(12, 10))
cm = pd.DataFrame(cm, index=[i for i in encoder.categories_], columns=[i for i in encoder.categories_])
#cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='.2f')
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=14)
plt.ylabel('Actual Labels', size=14)
plt.show()
print(classification_report(y_test0, y_pred0))
