# 순환 신경망으로 IMDB 리뷰 분류하기

## 키워드 정리

- **말뭉치** : 자연어 처리에서 사용하는 텍스트 데이터의 모음, 즉 훈련 데이터셋을 일컫습니다.
- **토큰** : 텍스트에서 공백으로 구분되는 문자열을 말합니다. 종종 소문자로 변환하고 구둣점은 삭제합니다.
- **원-핫 인코딩** : 어떤 클래스에 해당하는 원소만 1이고 나머지는 모두 0인 벡터입니다. 정수로 변환된 토큰을 원-핫 인코딩으로 변환하려면 어휘 사전의 크기의 벡터가 만들어집니다.
- **단어 임베딩** : 정수로 변환된 토큰을 비교적 작은 크기의 실수 밀집 벡터로 변환합니다. 이런 밀집 벡터는 단어 사이의 관계를 표현할 수 있기 때문에 자연어 처리에서 좋은 성능을 발휘합니다.

### Tensorflow

- **pad_sequences()** : 시퀀스 길이를 맞추기 위해 패딩을 추가합니다. 이 함수는 (샘플 개수, 타임스텝 개수) 크기의 2차원 배열을 기대합니다.
  - `maxlen` 매개변수로 원하는 시퀀스 길이를 지정할 수 있습니다. 이 값보다 긴 시퀀스는 잘리고 짧은 시퀀스는 패딩 됩니다. 이 매개변수를 지정하지 않으면 가장 긴 시퀀스의 길이가 됩니다.
  - `padding` 매개변수는 패딩을 추가할 위치를 지정합니다. 기본값인 `pre`는 시퀀스 앞에 패딩을 추가하고 `post`는 시퀀스 뒤에 패딩을 추가합니다.
  - `truncating` 매개변수는 긴 시퀀에서 잘라버릴 위치를 지정합니다. 기본값인 `pre`는 시퀀스 앞부분을 잘라내고 `post`는 시퀀스 뒷부분을 잘라냅니다.
- **to_categorical()** : 정수 시퀀스를 원-핫 인코딩으로 변환합니다. 토큰을 원-핫 인코딩하거나 타깃값을 원-핫 인코딩할 때 사용합니다.
  - `num_classes` 매개변수에서 클래스 개수를 지정할 수 있습니다. 지정하지 않으면 데이터에서 자동으로 찾습니다.
- **SimpleRNN** : 케라스의 기본 순환층 클래스입니다.
  - 첫 번째 매개변수에 뉴런의 개수를 지정합니다.
  - `activation` 매개변수에서 활성화 함수를 지정합니다. 기본값은 하이퍼볼릭 탄젠트인 `tanh`입니다.
  - `dropout` 매개변수에서 입력에 대한 드롭아웃 비율을 지정할 수 있습니다.
  - `return_sequences` 매개변수에서 모든 타임스텝의 은닉 상태를 출력할지 결정합니다. 기본값은 `False`입니다.
- **Embedding** : 단어 임베딩을 위한 클래스입니다.
  - 첫 번째 매개변수에서 어휘 사전의 크기를 지정합니다.
  - 두 번째 매개변수에서 `Embedding` 층이 출력할 밀집 벡터의 크기를 지정합니다.
  - `input_length` 매개변수에서 입력 시퀀스의 길이를 지정합니다. 이 매개변수는 `Embedding`층 바로 뒤에 `Flatten`이나 `Dense` 클래스가 올 때 꼭 필요합니다.

## 시작하기 전에

- 1절에서 순환 신경망의 작동 원리를 살펴보았습니다. 이번 절에서는 대표적인 순환 신경망 문제인 IMDB 리뷰 데이터셋을 사용해 가장 간단한 순환 신경망 모델을 훈련해 보겠습니다.
- 이 데이터셋을 두 가지 방법으로 변형하여 순환 신경망에 주입해 보겠습니다. 하나는 원-핫 인코딩이고 또 하나는 단어 임베딩입니다. 이 두 가지 방법의 차이점에 대해 설명하고 순환 신경망을 만들 때 고려해야 할 점을 알아보겠습니다.
- 그럼 먼저 이 절에서 사용할 IMDB 리뷰 데이터셋을 적재해 보겠습니다.

![스크린샷 2025-03-18 오후 9 50 02](https://github.com/user-attachments/assets/2af6c399-1cce-4048-8568-04f8295c13d9)

## IMDB 리뷰 데이터셋

- IMDB 리뷰 데이터셋은 유명한 인터넷 영화 데이터베이스인 `imdb.com`에서 수집한 리뷰를 감상평에 따라 긍정과 부정으로 분류해 놓은 데이터셋입니다. 총 50,000개의 샘플로 이루어져 있고 훈련 데이터와 테스트 데이터에 각각 25,000개씩 나누어져 있습니다.

> **자연어 처리와 말뭉치란 무엇인가요?**<br>**자연어 처리**(natual language processing, NLP)는 컴퓨터를 사용해 인간의 언어를 처리하는 분야입니다. 대표적인 세부 분야로는 음성 인식, 기계 번역, 감성 분석 등이 있습니다. IMDB 리뷰를 감상평에 따라 분류하는 작업은 감성 분석에 해당합니다. 자연어 처리 분야에서는 훈련 데이터를 종종 **말뭉치**(corpus)라고 부릅니다. 예를 들어 IMDB 리뷰 데이터셋이 하나의 말뭉치입니다.

- 사실 텍스트 자체를 신경망에 전달하지는 않습니다. 컴퓨터에서 처리하는 모든 것은 어떤 숫자 데이터입니다. 앞서 합성곱 신경망에서 이미지를 다룰 떄는 특별한 변환을 하지 않았습니다. 이미지가 정수 픽셀값으로 이루어져 있기 때문이죠. 텍스트 데이터의 경우 단어를 숫자 데이터로 바꾸는 일반적인 방법은 데이터에 등장하는 단어마다 고유한 정수를 부여하는 것입니다. 예를 들면 다음과 같습니다.

![스크린샷 2025-03-18 오후 9 55 13](https://github.com/user-attachments/assets/5ea4096f-c938-429d-9b4f-be8704b74f03)



```python
from tensorflow.keras.datasets import imdb

(train_input, train_target), (test_input, test_target) = imdb.load_data(
    num_words=200)
```

```python
print(train_input.shape, test_input.shape)
```

```
(25000,) (25000,)
```

```python
print(len(train_input[0]))
```

```
218
```

```python
print(len(train_input[1]))
```

```
189
```

```python
print(train_input[0])
```

```
[1, 14, 22, 16, 43, 2, 2, 2, 2, 65, 2, 2, 66, 2, 4, 173, 36, 2, 5, 25, 100, 43, 2, 112, 50, 2, 2, 9, 35, 2, 2, 5, 150, 4, 172, 112, 167, 2, 2, 2, 39, 4, 172, 2, 2, 17, 2, 38, 13, 2, 4, 192, 50, 16, 6, 147, 2, 19, 14, 22, 4, 2, 2, 2, 4, 22, 71, 87, 12, 16, 43, 2, 38, 76, 15, 13, 2, 4, 22, 17, 2, 17, 12, 16, 2, 18, 2, 5, 62, 2, 12, 8, 2, 8, 106, 5, 4, 2, 2, 16, 2, 66, 2, 33, 4, 130, 12, 16, 38, 2, 5, 25, 124, 51, 36, 135, 48, 25, 2, 33, 6, 22, 12, 2, 28, 77, 52, 5, 14, 2, 16, 82, 2, 8, 4, 107, 117, 2, 15, 2, 4, 2, 7, 2, 5, 2, 36, 71, 43, 2, 2, 26, 2, 2, 46, 7, 4, 2, 2, 13, 104, 88, 4, 2, 15, 2, 98, 32, 2, 56, 26, 141, 6, 194, 2, 18, 4, 2, 22, 21, 134, 2, 26, 2, 5, 144, 30, 2, 18, 51, 36, 28, 2, 92, 25, 104, 4, 2, 65, 16, 38, 2, 88, 12, 16, 2, 5, 16, 2, 113, 103, 32, 15, 16, 2, 19, 178, 32]
```

```python
print(train_target[:20])
```

```
[1 0 0 1 0 0 1 0 1 0 1 0 0 0 0 0 1 1 0 1]
```

```python
from sklearn.model_selection import train_test_split

train_input, val_input, train_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)
```

```python
import numpy as np

lengths = np.array([len(x) for x in train_input])
```

```python
print(np.mean(lengths), np.median(lengths))
```

```
239.00925 178.0
```

```python
import matplotlib.pyplot as plt

plt.hist(lengths)
plt.xlabel('length')
plt.ylabel('frequency')
plt.show()
```

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_seq = pad_sequences(train_input, maxlen=100)
```

```python
print(train_seq.shape)
```

```
(20000, 100)
```

```python
print(train_seq[0])
```

```
[ 10   4  20   9   2   2   2   5  45   6   2   2  33   2   8   2 142   2
   5   2  17  73  17   2   5   2  19  55   2   2  92  66 104  14  20  93
  76   2 151  33   4  58  12 188   2 151  12   2  69   2 142  73   2   6
   2   7   2   2 188   2 103  14  31  10  10   2   7   2   5   2  80  91
   2  30   2  34  14  20 151  50  26 131  49   2  84  46  50  37  80  79
   6   2  46   7  14  20  10  10   2 158]
```

```python
print(train_input[0][-10:])
```

```
[6, 2, 46, 7, 14, 20, 10, 10, 2, 158]
```

```python
print(train_seq[5])
```

```
[  0   0   0   0   1   2 195  19  49   2   2 190   4   2   2   2 183  10
  10  13  82  79   4   2  36  71   2   8   2  25  19  49   7   4   2   2
   2   2   2  10  10  48  25  40   2  11   2   2  40   2   2   5   4   2
   2  95  14   2  56 129   2  10  10  21   2  94   2   2   2   2  11 190
  24   2   2   7  94   2   2  10  10  87   2  34  49   2   7   2   2   2
   2   2   2   2  46  48  64  18   4   2]
```

```python
val_seq = pad_sequences(val_input, maxlen=100)
```

```python
from tensorflow import keras

model = keras.Sequential()

model.aadd(keras.layers.Input(shape=(100,200)))
model.add(keras.layers.SimpleRNN(8))
model.add(keras.layers.Dense(1, activation='sigmoid'))
```

```python
train_oh = keras.utils.to_categorical(train_seq)
```

```python
print(train_oh.shape)
```

```
(20000, 100, 200)
```

```python
print(train_oh[0][0][:12])
```

```
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
```

```python
print(np.sum(train_oh[0][0]))
```

```
1.0
```

```python
val_oh = keras.utils.to_categorical(val_seq)
```

```python
model.summary()
```

```python
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rmsprop, loss='binary_crossentropy',
              metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-simplernn-model.keras',
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

history = model.fit(train_oh, train_target, epochs=100, batch_size=64,
                    validation_data=(val_oh, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])
```

```
Epoch 1/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 15s 39ms/step - accuracy: 0.4945 - loss: 0.7135 - val_accuracy: 0.4924 - val_loss: 0.7053
Epoch 2/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 12s 18ms/step - accuracy: 0.5020 - loss: 0.7038 - val_accuracy: 0.4960 - val_loss: 0.7010
Epoch 3/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.5059 - loss: 0.6993 - val_accuracy: 0.4962 - val_loss: 0.6984
Epoch 4/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 11s 21ms/step - accuracy: 0.5124 - loss: 0.6964 - val_accuracy: 0.5038 - val_loss: 0.6962
Epoch 5/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.5154 - loss: 0.6943 - val_accuracy: 0.5110 - val_loss: 0.6939
Epoch 6/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.5195 - loss: 0.6925 - val_accuracy: 0.5120 - val_loss: 0.6925
Epoch 7/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 7s 23ms/step - accuracy: 0.5228 - loss: 0.6915 - val_accuracy: 0.5162 - val_loss: 0.6917
Epoch 8/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 9s 20ms/step - accuracy: 0.5255 - loss: 0.6905 - val_accuracy: 0.5198 - val_loss: 0.6910
Epoch 9/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 9s 16ms/step - accuracy: 0.5305 - loss: 0.6897 - val_accuracy: 0.5232 - val_loss: 0.6904
Epoch 10/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.5355 - loss: 0.6888 - val_accuracy: 0.5256 - val_loss: 0.6899
Epoch 11/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 18ms/step - accuracy: 0.5392 - loss: 0.6881 - val_accuracy: 0.5256 - val_loss: 0.6893
Epoch 12/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.5413 - loss: 0.6873 - val_accuracy: 0.5272 - val_loss: 0.6888
Epoch 13/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.5443 - loss: 0.6866 - val_accuracy: 0.5328 - val_loss: 0.6883
Epoch 14/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.5470 - loss: 0.6859 - val_accuracy: 0.5344 - val_loss: 0.6878
Epoch 15/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.5505 - loss: 0.6851 - val_accuracy: 0.5372 - val_loss: 0.6873
Epoch 16/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.5537 - loss: 0.6844 - val_accuracy: 0.5432 - val_loss: 0.6867
Epoch 17/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 16ms/step - accuracy: 0.5544 - loss: 0.6837 - val_accuracy: 0.5466 - val_loss: 0.6861
Epoch 18/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.5559 - loss: 0.6829 - val_accuracy: 0.5478 - val_loss: 0.6856
Epoch 19/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 18ms/step - accuracy: 0.5578 - loss: 0.6821 - val_accuracy: 0.5490 - val_loss: 0.6850
Epoch 20/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 17ms/step - accuracy: 0.5599 - loss: 0.6813 - val_accuracy: 0.5522 - val_loss: 0.6845
Epoch 21/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 9s 29ms/step - accuracy: 0.5630 - loss: 0.6805 - val_accuracy: 0.5524 - val_loss: 0.6839
Epoch 22/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 9s 29ms/step - accuracy: 0.5656 - loss: 0.6797 - val_accuracy: 0.5530 - val_loss: 0.6832
Epoch 23/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.5670 - loss: 0.6789 - val_accuracy: 0.5538 - val_loss: 0.6825
Epoch 24/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.5691 - loss: 0.6780 - val_accuracy: 0.5578 - val_loss: 0.6819
Epoch 25/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 17ms/step - accuracy: 0.5706 - loss: 0.6771 - val_accuracy: 0.5604 - val_loss: 0.6812
Epoch 26/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 17ms/step - accuracy: 0.5730 - loss: 0.6761 - val_accuracy: 0.5622 - val_loss: 0.6804
Epoch 27/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 11s 18ms/step - accuracy: 0.5747 - loss: 0.6750 - val_accuracy: 0.5632 - val_loss: 0.6797
Epoch 28/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.5747 - loss: 0.6740 - val_accuracy: 0.5634 - val_loss: 0.6787
Epoch 29/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 17ms/step - accuracy: 0.5768 - loss: 0.6728 - val_accuracy: 0.5662 - val_loss: 0.6774
Epoch 30/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 17ms/step - accuracy: 0.5796 - loss: 0.6713 - val_accuracy: 0.5716 - val_loss: 0.6759
Epoch 31/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 11s 18ms/step - accuracy: 0.5801 - loss: 0.6695 - val_accuracy: 0.5790 - val_loss: 0.6737
Epoch 32/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 11s 20ms/step - accuracy: 0.5860 - loss: 0.6671 - val_accuracy: 0.5836 - val_loss: 0.6705
Epoch 33/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.5962 - loss: 0.6634 - val_accuracy: 0.5974 - val_loss: 0.6647
Epoch 34/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.6079 - loss: 0.6564 - val_accuracy: 0.6116 - val_loss: 0.6511
Epoch 35/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 16ms/step - accuracy: 0.6356 - loss: 0.6393 - val_accuracy: 0.6616 - val_loss: 0.6188
Epoch 36/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - accuracy: 0.6708 - loss: 0.6103 - val_accuracy: 0.6684 - val_loss: 0.6126
Epoch 37/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 19ms/step - accuracy: 0.6774 - loss: 0.6029 - val_accuracy: 0.6758 - val_loss: 0.6062
Epoch 38/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.6811 - loss: 0.5968 - val_accuracy: 0.6796 - val_loss: 0.6015
Epoch 39/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.6876 - loss: 0.5911 - val_accuracy: 0.6864 - val_loss: 0.5966
Epoch 40/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.6914 - loss: 0.5860 - val_accuracy: 0.6854 - val_loss: 0.5926
Epoch 41/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 19ms/step - accuracy: 0.6976 - loss: 0.5811 - val_accuracy: 0.6904 - val_loss: 0.5889
Epoch 42/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 17ms/step - accuracy: 0.7006 - loss: 0.5762 - val_accuracy: 0.6974 - val_loss: 0.5852
Epoch 43/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 17ms/step - accuracy: 0.7060 - loss: 0.5720 - val_accuracy: 0.6988 - val_loss: 0.5817
Epoch 44/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 16ms/step - accuracy: 0.7100 - loss: 0.5681 - val_accuracy: 0.6990 - val_loss: 0.5787
Epoch 45/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 16ms/step - accuracy: 0.7115 - loss: 0.5654 - val_accuracy: 0.7022 - val_loss: 0.5757
Epoch 46/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 17ms/step - accuracy: 0.7131 - loss: 0.5625 - val_accuracy: 0.7028 - val_loss: 0.5734
Epoch 47/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 17ms/step - accuracy: 0.7149 - loss: 0.5601 - val_accuracy: 0.7058 - val_loss: 0.5717
Epoch 48/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 11s 18ms/step - accuracy: 0.7164 - loss: 0.5574 - val_accuracy: 0.7084 - val_loss: 0.5693
Epoch 49/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 18ms/step - accuracy: 0.7152 - loss: 0.5561 - val_accuracy: 0.7106 - val_loss: 0.5675
Epoch 50/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 19ms/step - accuracy: 0.7186 - loss: 0.5541 - val_accuracy: 0.7120 - val_loss: 0.5662
Epoch 51/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 11s 22ms/step - accuracy: 0.7204 - loss: 0.5524 - val_accuracy: 0.7126 - val_loss: 0.5653
Epoch 52/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.7213 - loss: 0.5511 - val_accuracy: 0.7152 - val_loss: 0.5647
Epoch 53/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.7232 - loss: 0.5497 - val_accuracy: 0.7156 - val_loss: 0.5632
Epoch 54/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 18ms/step - accuracy: 0.7241 - loss: 0.5483 - val_accuracy: 0.7166 - val_loss: 0.5624
Epoch 55/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 16ms/step - accuracy: 0.7257 - loss: 0.5472 - val_accuracy: 0.7204 - val_loss: 0.5615
Epoch 56/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.7263 - loss: 0.5463 - val_accuracy: 0.7210 - val_loss: 0.5605
Epoch 57/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.7265 - loss: 0.5451 - val_accuracy: 0.7202 - val_loss: 0.5601
Epoch 58/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 19ms/step - accuracy: 0.7279 - loss: 0.5440 - val_accuracy: 0.7214 - val_loss: 0.5590
Epoch 59/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 9s 16ms/step - accuracy: 0.7287 - loss: 0.5431 - val_accuracy: 0.7224 - val_loss: 0.5579
Epoch 60/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 16ms/step - accuracy: 0.7301 - loss: 0.5422 - val_accuracy: 0.7210 - val_loss: 0.5584
Epoch 61/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.7279 - loss: 0.5410 - val_accuracy: 0.7238 - val_loss: 0.5574
Epoch 62/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.7286 - loss: 0.5400 - val_accuracy: 0.7270 - val_loss: 0.5558
Epoch 63/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 19ms/step - accuracy: 0.7319 - loss: 0.5391 - val_accuracy: 0.7256 - val_loss: 0.5556
Epoch 64/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 9s 16ms/step - accuracy: 0.7310 - loss: 0.5384 - val_accuracy: 0.7262 - val_loss: 0.5544
Epoch 65/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.7310 - loss: 0.5375 - val_accuracy: 0.7292 - val_loss: 0.5532
Epoch 66/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 18ms/step - accuracy: 0.7332 - loss: 0.5369 - val_accuracy: 0.7298 - val_loss: 0.5530
Epoch 67/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.7335 - loss: 0.5361 - val_accuracy: 0.7324 - val_loss: 0.5524
Epoch 68/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.7333 - loss: 0.5355 - val_accuracy: 0.7312 - val_loss: 0.5516
Epoch 69/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.7334 - loss: 0.5351 - val_accuracy: 0.7324 - val_loss: 0.5515
Epoch 70/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.7343 - loss: 0.5345 - val_accuracy: 0.7330 - val_loss: 0.5510
Epoch 71/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.7349 - loss: 0.5336 - val_accuracy: 0.7322 - val_loss: 0.5504
Epoch 72/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.7360 - loss: 0.5329 - val_accuracy: 0.7300 - val_loss: 0.5501
Epoch 73/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 17ms/step - accuracy: 0.7365 - loss: 0.5324 - val_accuracy: 0.7330 - val_loss: 0.5501
Epoch 74/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 16ms/step - accuracy: 0.7364 - loss: 0.5318 - val_accuracy: 0.7326 - val_loss: 0.5500
Epoch 75/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 19ms/step - accuracy: 0.7363 - loss: 0.5314 - val_accuracy: 0.7298 - val_loss: 0.5494
Epoch 76/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 11s 20ms/step - accuracy: 0.7373 - loss: 0.5307 - val_accuracy: 0.7310 - val_loss: 0.5496
Epoch 77/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 9s 17ms/step - accuracy: 0.7378 - loss: 0.5303 - val_accuracy: 0.7286 - val_loss: 0.5494
Epoch 78/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 17ms/step - accuracy: 0.7385 - loss: 0.5298 - val_accuracy: 0.7316 - val_loss: 0.5491
Epoch 79/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.7382 - loss: 0.5294 - val_accuracy: 0.7308 - val_loss: 0.5493
Epoch 80/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 17ms/step - accuracy: 0.7386 - loss: 0.5291 - val_accuracy: 0.7318 - val_loss: 0.5492
Epoch 81/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 17ms/step - accuracy: 0.7386 - loss: 0.5285 - val_accuracy: 0.7320 - val_loss: 0.5491
Epoch 82/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.7387 - loss: 0.5281 - val_accuracy: 0.7324 - val_loss: 0.5491
Epoch 83/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.7383 - loss: 0.5275 - val_accuracy: 0.7324 - val_loss: 0.5495
Epoch 84/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 11s 19ms/step - accuracy: 0.7389 - loss: 0.5270 - val_accuracy: 0.7330 - val_loss: 0.5497
```

```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```

```python
model2 = keras.Sequential()

model2.add(keras.layers.Embedding(200, 16, input_shape=(100,)))
model2.add(keras.layers.SimpleRNN(8))
model2.add(keras.layers.Dense(1, activation='sigmoid'))

model2.summary()
```

```python
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model2.compile(optimizer=rmsprop, loss='binary_crossentropy',
               metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-embedding-model.keras',
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

history = model2.fit(train_seq, train_target, epochs=100, batch_size=64,
                     validation_data=(val_seq, val_target),
                     callbacks=[checkpoint_cb, early_stopping_cb])
```

```
Epoch 1/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 22ms/step - accuracy: 0.5096 - loss: 0.6937 - val_accuracy: 0.6108 - val_loss: 0.6764
Epoch 2/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 19ms/step - accuracy: 0.6264 - loss: 0.6719 - val_accuracy: 0.6544 - val_loss: 0.6605
Epoch 3/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 17ms/step - accuracy: 0.6600 - loss: 0.6577 - val_accuracy: 0.6786 - val_loss: 0.6466
Epoch 4/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 12s 22ms/step - accuracy: 0.6802 - loss: 0.6438 - val_accuracy: 0.6916 - val_loss: 0.6347
Epoch 5/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 17ms/step - accuracy: 0.6897 - loss: 0.6316 - val_accuracy: 0.6596 - val_loss: 0.6366
Epoch 6/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 11s 18ms/step - accuracy: 0.6973 - loss: 0.6199 - val_accuracy: 0.7086 - val_loss: 0.6108
Epoch 7/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 19ms/step - accuracy: 0.7087 - loss: 0.6071 - val_accuracy: 0.7092 - val_loss: 0.6007
Epoch 8/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 18ms/step - accuracy: 0.7112 - loss: 0.5974 - val_accuracy: 0.7030 - val_loss: 0.5963
Epoch 9/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.7133 - loss: 0.5890 - val_accuracy: 0.7072 - val_loss: 0.5892
Epoch 10/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 11s 19ms/step - accuracy: 0.7199 - loss: 0.5801 - val_accuracy: 0.7044 - val_loss: 0.5841
Epoch 11/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 18ms/step - accuracy: 0.7235 - loss: 0.5722 - val_accuracy: 0.7128 - val_loss: 0.5747
Epoch 12/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.7259 - loss: 0.5665 - val_accuracy: 0.7192 - val_loss: 0.5670
Epoch 13/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 17ms/step - accuracy: 0.7283 - loss: 0.5611 - val_accuracy: 0.7186 - val_loss: 0.5631
Epoch 14/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 17ms/step - accuracy: 0.7310 - loss: 0.5568 - val_accuracy: 0.7182 - val_loss: 0.5598
Epoch 15/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.7324 - loss: 0.5531 - val_accuracy: 0.7210 - val_loss: 0.5565
Epoch 16/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 17ms/step - accuracy: 0.7318 - loss: 0.5498 - val_accuracy: 0.7224 - val_loss: 0.5535
Epoch 17/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.7342 - loss: 0.5469 - val_accuracy: 0.7242 - val_loss: 0.5511
Epoch 18/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.7338 - loss: 0.5445 - val_accuracy: 0.7260 - val_loss: 0.5494
Epoch 19/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 11s 19ms/step - accuracy: 0.7347 - loss: 0.5426 - val_accuracy: 0.7268 - val_loss: 0.5480
Epoch 20/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 11s 20ms/step - accuracy: 0.7345 - loss: 0.5410 - val_accuracy: 0.7284 - val_loss: 0.5466
Epoch 21/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 9s 18ms/step - accuracy: 0.7343 - loss: 0.5395 - val_accuracy: 0.7282 - val_loss: 0.5455
Epoch 22/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 19ms/step - accuracy: 0.7355 - loss: 0.5381 - val_accuracy: 0.7280 - val_loss: 0.5448
Epoch 23/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 19ms/step - accuracy: 0.7356 - loss: 0.5368 - val_accuracy: 0.7290 - val_loss: 0.5442
Epoch 24/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 17ms/step - accuracy: 0.7364 - loss: 0.5356 - val_accuracy: 0.7294 - val_loss: 0.5437
Epoch 25/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 19ms/step - accuracy: 0.7362 - loss: 0.5345 - val_accuracy: 0.7294 - val_loss: 0.5432
Epoch 26/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 17ms/step - accuracy: 0.7363 - loss: 0.5334 - val_accuracy: 0.7296 - val_loss: 0.5427
Epoch 27/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 19ms/step - accuracy: 0.7350 - loss: 0.5323 - val_accuracy: 0.7308 - val_loss: 0.5423
Epoch 28/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 18ms/step - accuracy: 0.7352 - loss: 0.5313 - val_accuracy: 0.7296 - val_loss: 0.5418
Epoch 29/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.7367 - loss: 0.5303 - val_accuracy: 0.7294 - val_loss: 0.5415
Epoch 30/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 17ms/step - accuracy: 0.7369 - loss: 0.5294 - val_accuracy: 0.7290 - val_loss: 0.5415
Epoch 31/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.7387 - loss: 0.5287 - val_accuracy: 0.7306 - val_loss: 0.5413
Epoch 32/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 19ms/step - accuracy: 0.7385 - loss: 0.5279 - val_accuracy: 0.7310 - val_loss: 0.5415
Epoch 33/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 17ms/step - accuracy: 0.7394 - loss: 0.5273 - val_accuracy: 0.7316 - val_loss: 0.5416
Epoch 34/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.7395 - loss: 0.5268 - val_accuracy: 0.7322 - val_loss: 0.5417
```

```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```
