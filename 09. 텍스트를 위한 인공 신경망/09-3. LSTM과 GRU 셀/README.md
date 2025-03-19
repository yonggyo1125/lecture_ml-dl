# LSTM과 GRU 셀

## 키워드 정리

- **LSTM** 셀은 타임스텝이 긴 데이터를 효과적으로 학습하기 위해 고안된 순환층입니다. 입력게이트, 삭제게이트, 출력 게이트의 역할을 하는 작은 셀이 포함되어 있습니다.
- LSTM 셀은 은닉 상태 외에 **셀 상태**를 출력합니다. 셀 상태는 다음 층으로 전달되지 않으며 현재 셀에서만 순환됩니다.
- **GRU** 셀은 LSTM 셀의 간소화 버전으로 생각할 수 있지만 LSTM 셀에 못지않은 성능을 냅니다.

### Tensorflow

- **LSTM**은 LSTM 셀을 사용한 순환층 클래스입니다.
  - 첫 번째 매개변수에 뉴런의 개수를 지정합니다.
  - `dropout` 매개변수에서 입력에 대한 드롭아웃 비율을 지정할 수 있습니다.
  - `return_sequences` 매개변수에서 모든 타임스텝의 은닉 상태를 출력할지 결정합니다. 기본 값은 False입니다.
- **GRU**는 GRU 셀을 사욜한 순환층 클래스입니다.
  - 첫 번째 매개변수에 뉴런의 개수를 지정합니다.
  - `dropout` 매개변수에서 입력에 대한 드롭아웃 비율을 지정할 수 있습니다.
  - `return_sequences` 매개변수에서 모든 타임스텝의 은닉 상태를 출력할지 결정합니다. 기본값은 False입니다.

## 시작하기 전에

- 고급 순환충인 `LSTM`과 `GRU`에 대해 알아보겠습니다. 이런 층들은 2절에서 배웠던 `SimpleRNN`보다 계산이 훨씬 복잡합니다. 하지만 성능이 뛰어나기 때문에 순환 신경망에 많이 채택되고 있습니다.
- 일반적으로 기본 순환층은 긴 시퀀스를 학습하기 어렵습니다. 시퀀스가 길수록 순환되는 은닉 상태에 담긴 정보가 점차 희석되기 떄문입니다. 따라서 멀리 떨어져 있는 단어 정보를 인식하는 데 어려울 수 있습니다. 이를 위해 `LSTM`과 `GRU` 셀이 발명되었습니다.

## LSTM 구조

- `LSTM`은 Long Short-Term Memory의 약자입니다. 말 그대로 단기 기억을 오래 기억하기 위해 고안되었습니다. `LSTM`은 구조가 복잡하므로 단계적으로 설명하겠습니다. 하지만 기본 개념은 동일합니다. `LSTM`에는 가중치를 곱하고 절편을 더해 활성화 함수를 통과시키는 구조를 여러개 가지고 있습니다. 이런 계산 결과는 다음 타임스텝에 재사용됩니다. 이 과정을 하나씩 따라가 보죠.
- 먼저 은닉 상태를 만드는 방법을 알아보죠. 은닉 상태는 입력과 이전 타임스텝의 은닉 상태를 가중치에 곱한 후 활성화 함수를 통과시켜 다음 은닉 상태를 만듭니다. 이때 기본 순환층과는 달리 시그모이드 활성화 함수를 사용합니다. 또 `tanh` 활성화 함수를 통과한 어떤 값과 곱해져서 은닉 상태를 만듭니다. 이 값은 잠시 후에 설명하겠습니다. 다음 그림을 참고하세요.

![스크린샷 2025-03-19 오전 11 51 26](https://github.com/user-attachments/assets/a063a56b-6a18-4a5f-aa99-7228ae76bc6b)

- 이 그림에는 편의상 은닉 상태를 계산할 떄 사용하는 가중치 W<sub>x</sub>와 W<sub>h</sub>를 통틀어 W<sub>o</sub>라고 표시했습니다. 파란색 원은 `tanh` 함수를 나타내고 주황색 원은 시그모이드 함수를 나타냅니다. x는 곱셈을 나타냅니다. 기본 순환층과 크게 다르지 않습니다. 그럼 `tanh`함수를 통과하는 값이 무엇인지 알아보죠.
- `LSTM`에는 순환되는 상태가 2개입니다. 은닉 상태 말고 **셀 상태**<sup>cell state</sup>라고 부르는 값이 또 있죠. 은닉 상태와 달리 셀 상태는 다음 층으로 전달되지 않고 `LSTM` 셀에서 순환만 되는 값입니다. 다음 그림에 초록색으로 순환되는 셀 상태가 표시되어 있습니다.

![스크린샷 2025-03-19 오전 11 56 04](https://github.com/user-attachments/assets/e007f436-4ddc-407a-b386-34e1886e7fc9)

- 셀 상태를 은닉 상태 h와 구분하여 c로 표시했습니다. 셀 상태를 계산하는 과정은 다음과 같습니다.
- 먼저 입력과 은닉 상태를 또 다른 가중치 W<sub>f</sub>에 곱한 다음 시그모이드 함수를 통과시킵니다. 그다음 이전 타임스텝의 셀 상태와 곱하여 새로운 셀 상태를 만듭니다. 이 셀 상태가 오른쪽에서 `tanh`함수를 통과하여 새로운 은닉 상태를 만드는 데 기여합니다.
- `LSTM`은 마치 작은 셀을 여러개 포함하고 있는 큰 셀과 같습니다. 중요한 것은 입력과 은닉 상태에 곱해지는 가중치 W<sub>o</sub>와 W<sub>f</sub>가 다르다는 점입니다. 이 두 작은 셀은 각기 다른 기능을 위해 훈련됩니다. 그런데 `LSTM` 셀은 이게 끝이 아닙니다.
- 여기에 2개의 작은 셀이 더 추가되어 셀 상태를 만드는 데 기여합니다. 다음 그림을 보시죠.

![스크린샷 2025-03-19 오전 11 56 10](https://github.com/user-attachments/assets/6119e7e5-0ffe-4b13-90b2-77ab74dad015)

- 이전과 마찬가지로 입력과 은닉 상태를 각기 다른 가중치에 곱한 다음, 하나는 시그모이드 함수를 통과시키고 다른 하나는 `tanh` 함수를 통과시킵니다. 그다음 두 결과를 곱한 후 이전 셀 상태와 더합니다. 이 결과가 최종적인 다음 셀 상태가 됩니다.
- 다음 그림처럼 세 군데의 곱셈을 왼쪽부터 차례대로 삭제 게이트<sup>gate</sup>, 입력 게이트, 출력 게이트라고 부릅니다.

![스크린샷 2025-03-19 오후 12 05 43](https://github.com/user-attachments/assets/9482bcdf-f850-4add-a313-b0dc79d76ad8)

- 삭제 게이트는 셀 상태에 있는 정보를 제거하는 역할을 하고 입력 게이트는 새로운 정보를 셀 상태에 추가합니다. 출력 게이트를 통해서 이 셀 상태가 다음 은닉 상태로 출력됩니다.
- 물론 이 복잡한 셀 계산을 직접 할 필요는 없습니다. 케라스에는 이미 `LSTM` 클래스가 준비되어 있습니다. 다음 섹션에서 `LSTM` 클래스를 사용해 `LSTM` 순환 신경망을 만들어 보겠습니다.

## LSTM 신경망 훈련하기

- 먼저 이전 절에서처럼 IMDB 리뷰 데이터를 로드하고 훈련 세트와 검증 세트로 나눕니다. 이번에는 500개의 단어를 사용하겠습니다.

```python
from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) = imdb.load_data(
    num_words=500)

train_input, val_input, train_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)
```

- 그 다음 케라스의 `pad_sequences()` 함수로 각 샘플의 길이를 100에 맞추고 부족할 떄는 패딩을 추가합니다.

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_seq = pad_sequences(train_input, maxlen=100)
val_seq = pad_sequences(val_input, maxlen=100)
```

- 이제 `LSTM` 셀을 사용한 순환층을 만들어 보겠습니다. 사실 `SimpleRNN` 클래스를 `LSTM` 클래스로 바꾸기만 하면 됩니다.

```python
from tensorflow import keras

model = keras.Sequential()

model.add(keras.layers.Input(shape=(100,)))
model.add(keras.layers.Embedding(500, 16))
model.add(keras.layers.LSTM(8))
model.add(keras.layers.Dense(1, activation='sigmoid'))
```

- 2절에서 임베딩을 사용했던 순환 신경망 모델과 완전히 동일합니다. 여기에서는 `SimpleRNN` 대신 `LSTM`을 사용합니다. 모델 구조를 출력해 보죠.

```python
model.summary()
```

![스크린샷 2025-03-19 오후 12 11 03](https://github.com/user-attachments/assets/bc138016-7a37-4562-8d11-b76d18a92596)

- `SimpleRNN` 클래스의 모델 파라미터 개수는 200개였습니다. LSTM 셀에는 작은 셸이 4개 있으므로 정확히 4배가 늘어 모델 파라미터의 개수는 800개가 되었습니다.
- 모델을 컴파일하고 훈련해 보겠습니다. 이전과 마찬가지로 배치 크기는 64개, 에포크 횟수는 100으로 지정합니다. 체크포인트와 조기 종료를 위한 코드도 동일합니다.

```python
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rmsprop, loss='binary_crossentropy',
              metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-lstm-model.keras',
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

history = model.fit(train_seq, train_target, epochs=100, batch_size=64,
                    validation_data=(val_seq, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])
```

```
Epoch 1/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 9s 8ms/step - accuracy: 0.5110 - loss: 0.6929 - val_accuracy: 0.5864 - val_loss: 0.6911
Epoch 2/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 7ms/step - accuracy: 0.5784 - loss: 0.6907 - val_accuracy: 0.6258 - val_loss: 0.6874
Epoch 3/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - accuracy: 0.6217 - loss: 0.6858 - val_accuracy: 0.6562 - val_loss: 0.6765
Epoch 4/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - accuracy: 0.6681 - loss: 0.6682 - val_accuracy: 0.7096 - val_loss: 0.6177
Epoch 5/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 4s 7ms/step - accuracy: 0.7180 - loss: 0.6006 - val_accuracy: 0.7248 - val_loss: 0.5760
Epoch 6/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.7282 - loss: 0.5690 - val_accuracy: 0.7404 - val_loss: 0.5555
Epoch 7/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - accuracy: 0.7481 - loss: 0.5465 - val_accuracy: 0.7450 - val_loss: 0.5383
Epoch 8/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 10ms/step - accuracy: 0.7600 - loss: 0.5274 - val_accuracy: 0.7588 - val_loss: 0.5216
Epoch 9/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.7688 - loss: 0.5110 - val_accuracy: 0.7698 - val_loss: 0.5076
Epoch 10/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 7ms/step - accuracy: 0.7777 - loss: 0.4972 - val_accuracy: 0.7746 - val_loss: 0.4959
Epoch 11/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.7831 - loss: 0.4853 - val_accuracy: 0.7802 - val_loss: 0.4862
Epoch 12/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - accuracy: 0.7885 - loss: 0.4750 - val_accuracy: 0.7850 - val_loss: 0.4781
Epoch 13/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 10ms/step - accuracy: 0.7936 - loss: 0.4662 - val_accuracy: 0.7874 - val_loss: 0.4713
Epoch 14/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.7961 - loss: 0.4585 - val_accuracy: 0.7876 - val_loss: 0.4659
Epoch 15/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8009 - loss: 0.4519 - val_accuracy: 0.7904 - val_loss: 0.4614
Epoch 16/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8036 - loss: 0.4461 - val_accuracy: 0.7920 - val_loss: 0.4577
Epoch 17/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8054 - loss: 0.4411 - val_accuracy: 0.7900 - val_loss: 0.4548
Epoch 18/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 11ms/step - accuracy: 0.8083 - loss: 0.4369 - val_accuracy: 0.7898 - val_loss: 0.4524
Epoch 19/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8104 - loss: 0.4332 - val_accuracy: 0.7908 - val_loss: 0.4504
Epoch 20/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8117 - loss: 0.4300 - val_accuracy: 0.7904 - val_loss: 0.4488
Epoch 21/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 10ms/step - accuracy: 0.8128 - loss: 0.4273 - val_accuracy: 0.7902 - val_loss: 0.4475
Epoch 22/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 10ms/step - accuracy: 0.8135 - loss: 0.4249 - val_accuracy: 0.7888 - val_loss: 0.4463
Epoch 23/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8139 - loss: 0.4228 - val_accuracy: 0.7894 - val_loss: 0.4453
Epoch 24/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8139 - loss: 0.4210 - val_accuracy: 0.7886 - val_loss: 0.4444
Epoch 25/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8144 - loss: 0.4194 - val_accuracy: 0.7884 - val_loss: 0.4436
Epoch 26/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 7ms/step - accuracy: 0.8152 - loss: 0.4179 - val_accuracy: 0.7902 - val_loss: 0.4429
Epoch 27/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 4s 11ms/step - accuracy: 0.8148 - loss: 0.4166 - val_accuracy: 0.7914 - val_loss: 0.4422
Epoch 28/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8140 - loss: 0.4154 - val_accuracy: 0.7914 - val_loss: 0.4416
Epoch 29/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 7ms/step - accuracy: 0.8145 - loss: 0.4144 - val_accuracy: 0.7932 - val_loss: 0.4410
Epoch 30/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8139 - loss: 0.4134 - val_accuracy: 0.7940 - val_loss: 0.4405
Epoch 31/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8141 - loss: 0.4125 - val_accuracy: 0.7936 - val_loss: 0.4400
Epoch 32/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 11ms/step - accuracy: 0.8137 - loss: 0.4117 - val_accuracy: 0.7940 - val_loss: 0.4395
Epoch 33/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 4s 7ms/step - accuracy: 0.8138 - loss: 0.4109 - val_accuracy: 0.7936 - val_loss: 0.4391
Epoch 34/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8141 - loss: 0.4102 - val_accuracy: 0.7946 - val_loss: 0.4387
Epoch 35/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8147 - loss: 0.4095 - val_accuracy: 0.7944 - val_loss: 0.4384
Epoch 36/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - accuracy: 0.8151 - loss: 0.4089 - val_accuracy: 0.7946 - val_loss: 0.4381
Epoch 37/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - accuracy: 0.8153 - loss: 0.4083 - val_accuracy: 0.7934 - val_loss: 0.4379
Epoch 38/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 7ms/step - accuracy: 0.8170 - loss: 0.4077 - val_accuracy: 0.7940 - val_loss: 0.4376
Epoch 39/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8170 - loss: 0.4072 - val_accuracy: 0.7932 - val_loss: 0.4375
Epoch 40/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - accuracy: 0.8162 - loss: 0.4067 - val_accuracy: 0.7938 - val_loss: 0.4373
Epoch 41/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 10ms/step - accuracy: 0.8163 - loss: 0.4061 - val_accuracy: 0.7940 - val_loss: 0.4372
Epoch 42/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 4s 7ms/step - accuracy: 0.8160 - loss: 0.4057 - val_accuracy: 0.7940 - val_loss: 0.4370
Epoch 43/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8167 - loss: 0.4052 - val_accuracy: 0.7936 - val_loss: 0.4370
Epoch 44/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - accuracy: 0.8166 - loss: 0.4047 - val_accuracy: 0.7930 - val_loss: 0.4369
Epoch 45/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 10ms/step - accuracy: 0.8167 - loss: 0.4043 - val_accuracy: 0.7934 - val_loss: 0.4368
Epoch 46/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8169 - loss: 0.4038 - val_accuracy: 0.7934 - val_loss: 0.4368
Epoch 47/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8173 - loss: 0.4034 - val_accuracy: 0.7938 - val_loss: 0.4368
Epoch 48/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 7ms/step - accuracy: 0.8173 - loss: 0.4030 - val_accuracy: 0.7940 - val_loss: 0.4368
Epoch 49/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - accuracy: 0.8171 - loss: 0.4025 - val_accuracy: 0.7944 - val_loss: 0.4368
Epoch 50/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 11ms/step - accuracy: 0.8166 - loss: 0.4021 - val_accuracy: 0.7944 - val_loss: 0.4368
Epoch 51/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 4s 7ms/step - accuracy: 0.8164 - loss: 0.4017 - val_accuracy: 0.7942 - val_loss: 0.4368
Epoch 52/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8167 - loss: 0.4013 - val_accuracy: 0.7944 - val_loss: 0.4368
```

- 훈련 손실과 검증 손실 그래프를 그려 보겠습니다.

```python
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```

- 그래프를 보면 기본 순환층보다 `LSTM`이 과대적합을 억제하면서 훈련을 잘 수행한 것으로 보입니다..
- 하지만 경우에 따라서는 과대적합을 더 강하게 제어할 필요가 있습니다. 7장에서 배웠던 드롭아웃을 순환층에도 적용할 수 있을까요? 다음 섹션에서 이에 대해 알아보겠습니다.

## 순환층에 드롭아웃 적용하기

- 완전 연결 신경망과 합성곱 신경망에서는 `Dropout` 클래스를 사용해 드롭아웃을 적용했습니다. 이를 통해 모델이 훈련 세트에 너무 과대적합되는 것을 막았죠. 순환층은 자체적으로 드롭아웃 기능을 제공합니다. `SimpleRNN`과 `LSTM` 클래스 모두 `dropout` 매개변수와 `recurrent_dropout` 매개변수를 가지고 있습니다.

> 드롭아웃은 은닉층에 있는 뉴런의 출력을 랜덤하게 꺼서 과대적합을 막는 기법입니다.

- `dropout` 매개변수는 셀의 입력에 드롭아웃을 적용하고 `recurrent_dropout`은 순환되는 은닉 상태에 드롭아웃을 적용합니다. 하지만 기술적인 문제로 인해 `recurrent_dropout`을 사용하면 `GPU`를 사용하여 모델을 훈련하지 못합니다. 이 떄문에 모델의 훈련 속도가 크게 느려집니다. 따라서 여기에서는 `dropout`만을 사용해 보겠습니다.
- 전체적인 모델 구조는 이전과 동일합니다. `LSTM` 클래스에 `dropout` 매개변수를 0.3으로 지정하여 30%의 입력을 드롭아웃 합니다.

```python
model2 = keras.Sequential()

model2.add(keras.layers.Input(shape=(100,)))
model2.add(keras.layers.Embedding(500, 16))
model2.add(keras.layers.LSTM(8, dropout=0.3))
model2.add(keras.layers.Dense(1, activation='sigmoid'))
```

- 이 모델을 이전과 동일한 조건으로 훈련해 보죠.

```python
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model2.compile(optimizer=rmsprop, loss='binary_crossentropy',
               metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-dropout-model.keras',
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

history = model2.fit(train_seq, train_target, epochs=100, batch_size=64,
                     validation_data=(val_seq, val_target),
                     callbacks=[checkpoint_cb, early_stopping_cb])
```

```
Epoch 1/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 12ms/step - accuracy: 0.5145 - loss: 0.6930 - val_accuracy: 0.5546 - val_loss: 0.6923
Epoch 2/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - accuracy: 0.5619 - loss: 0.6920 - val_accuracy: 0.5872 - val_loss: 0.6909
Epoch 3/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.6026 - loss: 0.6902 - val_accuracy: 0.6164 - val_loss: 0.6879
Epoch 4/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.6257 - loss: 0.6865 - val_accuracy: 0.6406 - val_loss: 0.6809
Epoch 5/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - accuracy: 0.6458 - loss: 0.6773 - val_accuracy: 0.6486 - val_loss: 0.6513
Epoch 6/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 11ms/step - accuracy: 0.6848 - loss: 0.6394 - val_accuracy: 0.7208 - val_loss: 0.6041
Epoch 7/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - accuracy: 0.7198 - loss: 0.5997 - val_accuracy: 0.7310 - val_loss: 0.5860
Epoch 8/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - accuracy: 0.7326 - loss: 0.5804 - val_accuracy: 0.7364 - val_loss: 0.5722
Epoch 9/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - accuracy: 0.7458 - loss: 0.5618 - val_accuracy: 0.7530 - val_loss: 0.5525
Epoch 10/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 10ms/step - accuracy: 0.7549 - loss: 0.5479 - val_accuracy: 0.7516 - val_loss: 0.5480
Epoch 11/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 8ms/step - accuracy: 0.7634 - loss: 0.5337 - val_accuracy: 0.7580 - val_loss: 0.5328
Epoch 12/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.7698 - loss: 0.5203 - val_accuracy: 0.7562 - val_loss: 0.5290
Epoch 13/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.7742 - loss: 0.5099 - val_accuracy: 0.7674 - val_loss: 0.5149
Epoch 14/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 10ms/step - accuracy: 0.7796 - loss: 0.5000 - val_accuracy: 0.7638 - val_loss: 0.5109
Epoch 15/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - accuracy: 0.7813 - loss: 0.4916 - val_accuracy: 0.7728 - val_loss: 0.4965
Epoch 16/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - accuracy: 0.7863 - loss: 0.4829 - val_accuracy: 0.7736 - val_loss: 0.4912
Epoch 17/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - accuracy: 0.7900 - loss: 0.4736 - val_accuracy: 0.7786 - val_loss: 0.4847
Epoch 18/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - accuracy: 0.7908 - loss: 0.4681 - val_accuracy: 0.7790 - val_loss: 0.4792
Epoch 19/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 10ms/step - accuracy: 0.7957 - loss: 0.4608 - val_accuracy: 0.7774 - val_loss: 0.4762
Epoch 20/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - accuracy: 0.7987 - loss: 0.4565 - val_accuracy: 0.7802 - val_loss: 0.4736
Epoch 21/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 8ms/step - accuracy: 0.7961 - loss: 0.4506 - val_accuracy: 0.7836 - val_loss: 0.4689
Epoch 22/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - accuracy: 0.7972 - loss: 0.4478 - val_accuracy: 0.7882 - val_loss: 0.4632
Epoch 23/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 10ms/step - accuracy: 0.8023 - loss: 0.4423 - val_accuracy: 0.7852 - val_loss: 0.4641
Epoch 24/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - accuracy: 0.8029 - loss: 0.4391 - val_accuracy: 0.7856 - val_loss: 0.4571
Epoch 25/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 8ms/step - accuracy: 0.8027 - loss: 0.4375 - val_accuracy: 0.7858 - val_loss: 0.4577
Epoch 26/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8056 - loss: 0.4343 - val_accuracy: 0.7858 - val_loss: 0.4592
Epoch 27/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 10ms/step - accuracy: 0.8060 - loss: 0.4317 - val_accuracy: 0.7866 - val_loss: 0.4538
Epoch 28/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 4s 8ms/step - accuracy: 0.8093 - loss: 0.4307 - val_accuracy: 0.7920 - val_loss: 0.4526
Epoch 29/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - accuracy: 0.8097 - loss: 0.4274 - val_accuracy: 0.7894 - val_loss: 0.4539
Epoch 30/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - accuracy: 0.8052 - loss: 0.4292 - val_accuracy: 0.7904 - val_loss: 0.4473
Epoch 31/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - accuracy: 0.8128 - loss: 0.4233 - val_accuracy: 0.7908 - val_loss: 0.4487
Epoch 32/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 10ms/step - accuracy: 0.8098 - loss: 0.4228 - val_accuracy: 0.7922 - val_loss: 0.4489
Epoch 33/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - accuracy: 0.8109 - loss: 0.4198 - val_accuracy: 0.7902 - val_loss: 0.4500
```

- 검증 손실이 약간 향상된 것 같네요. 훈련 손실과 검증 손실 그래프를 그려 보겠습니다.

```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```

![스크린샷 2025-03-19 오후 12 22 35](https://github.com/user-attachments/assets/c8aeb29f-81d8-4fcb-8c69-03dc1619d438)

- `LSTM` 층에 적용한 드롭아웃이 효과를 발휘한 것 같습니다. 훈련 손실과 검증 손실 간의 차이가 좁혀진 것을 확인할 수 있습니다.
- 밀집층이나 합성곱 층처럼 순환층도 여러 개를 쌓지 않을 이유가 없습니다. 다음 섹션에서는 2개의 순환층을 연결한 모델을 훈련해 보죠.

## 2개의 층을 연결하기

- 순환층을 연결할 때는 한 가지 주의할 점이 있습니다. 앞서 언급했지만 순환층의 은닉 상태는 샘플의 마지막 타임스텝에 대한 은닉 상태만 다음 층으로 전달합니다. 하지만 순환층을 쌓게 되면 모든 순환 층에 순차 데이터가 필요합니다. 따라서 앞쪽의 순환층이 모든 타임스텝에 대한 은닉 상태를 출력해야 합니다. 오직 마지막 순환층만 마지막 타임스텝의 은닉 상태를 출력해야 합니다.

![스크린샷 2025-03-19 오후 12 26 13](https://github.com/user-attachments/assets/f0e0eace-1ad7-42c1-a3b7-d5d2a135bf5c)

- 케라스의 순환층에서 모든 타임스텝의 은닉 상태를 출력하려면 마지막을 제외한 다른 모든 순환층에서 `return_sequences` 매개변수를 True로 지정하면 됩니다. 다음의 코드를 확인해 보세요.

```python
model3 = keras.Sequential()

model3.add(keras.layers.Input(shape=(100,)))
model3.add(keras.layers.Embedding(500, 16))
model3.add(keras.layers.LSTM(8, dropout=0.3, return_sequences=True))
model3.add(keras.layers.LSTM(8, dropout=0.3))
model3.add(keras.layers.Dense(1, activation='sigmoid'))
```

- 2개의 `LSTM` 층을 쌓았고 모두 드롭아웃을 0.3으로 지정했습니다. 그리고 첫 번쨰 `LSTM` 클래스에는 `return_sequences` 매개변수를 True로 지정한 것을 볼 수 있습니다. `summary()` 메서드의 결과를 확인해 보죠

```python
model3.summary()
```

![스크린샷 2025-03-19 오후 12 29 14](https://github.com/user-attachments/assets/7a84fa9d-da24-41c7-add4-0d75938800f6)

- 첫 번쨰 `LSTM` 층이 모든 타임스텝(100개)의 은닉 상태를 출력하기 때문에 출력 크기가 (None, 100, 8)로 표시되었습니다. 이에 반해 두 번쨰 `LSTM` 층의 출력 크기는 마지막 타임스텝의 은닉 상태만 출력하기 때문에 (None, 8)입니다.
- 이 모델을 앞에서와 같이 훈련해 보겠습니다.

```python
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model3.compile(optimizer=rmsprop, loss='binary_crossentropy',
               metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-2rnn-model.keras',
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

history = model3.fit(train_seq, train_target, epochs=100, batch_size=64,
                     validation_data=(val_seq, val_target),
                     callbacks=[checkpoint_cb, early_stopping_cb])
```

```
Epoch 1/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 13ms/step - accuracy: 0.5235 - loss: 0.6927 - val_accuracy: 0.5738 - val_loss: 0.6904
Epoch 2/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 12ms/step - accuracy: 0.5854 - loss: 0.6888 - val_accuracy: 0.6384 - val_loss: 0.6765
Epoch 3/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 11ms/step - accuracy: 0.6495 - loss: 0.6604 - val_accuracy: 0.6932 - val_loss: 0.5969
Epoch 4/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 15ms/step - accuracy: 0.7134 - loss: 0.5717 - val_accuracy: 0.7300 - val_loss: 0.5459
Epoch 5/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 11ms/step - accuracy: 0.7405 - loss: 0.5346 - val_accuracy: 0.7478 - val_loss: 0.5202
Epoch 6/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 11ms/step - accuracy: 0.7571 - loss: 0.5090 - val_accuracy: 0.7558 - val_loss: 0.5120
Epoch 7/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 4s 14ms/step - accuracy: 0.7712 - loss: 0.4930 - val_accuracy: 0.7612 - val_loss: 0.5026
Epoch 8/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 4s 11ms/step - accuracy: 0.7732 - loss: 0.4813 - val_accuracy: 0.7716 - val_loss: 0.4902
Epoch 9/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 11ms/step - accuracy: 0.7836 - loss: 0.4727 - val_accuracy: 0.7662 - val_loss: 0.4928
Epoch 10/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 15ms/step - accuracy: 0.7806 - loss: 0.4711 - val_accuracy: 0.7742 - val_loss: 0.4794
Epoch 11/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 11ms/step - accuracy: 0.7831 - loss: 0.4691 - val_accuracy: 0.7776 - val_loss: 0.4795
Epoch 12/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 11ms/step - accuracy: 0.7909 - loss: 0.4590 - val_accuracy: 0.7828 - val_loss: 0.4705
Epoch 13/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 4s 12ms/step - accuracy: 0.7906 - loss: 0.4552 - val_accuracy: 0.7762 - val_loss: 0.4754
Epoch 14/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 4s 14ms/step - accuracy: 0.7940 - loss: 0.4542 - val_accuracy: 0.7844 - val_loss: 0.4670
Epoch 15/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 4s 11ms/step - accuracy: 0.7940 - loss: 0.4510 - val_accuracy: 0.7806 - val_loss: 0.4688
Epoch 16/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 4s 11ms/step - accuracy: 0.7992 - loss: 0.4484 - val_accuracy: 0.7820 - val_loss: 0.4728
Epoch 17/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 13ms/step - accuracy: 0.7982 - loss: 0.4452 - val_accuracy: 0.7812 - val_loss: 0.4698
```

- 모델이 잘 훈련된 것 같네요. 일반적으로 순환층을 쌓으면 성능이 높아집니다. 이 예에서는 그리 큰 효과를 내지 못했네요. 손실 그래프를 그려서 과대적합이 잘 제어되었는지 확인해 보겠습니다.

```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```

![스크린샷 2025-03-19 오후 12 32 29](https://github.com/user-attachments/assets/1e1a1b33-4408-48df-9290-33c8a307d1bb)

- 그래프를 보면 과대적합을 제어하면서 손실을 최대한 낮춘 것 같습니다. 지금까지 `LSTM` 셀을 사용한 훈련과 드롭아웃을 적용해 보았고 2개의 층을 쌓은 순환 신경망을 만들어 보았습니다. 다음 섹션에서는 유명한 또 다른 셀인 `GRU`셀에 대해 알아보겠습니다.

## GRU 구조

- `GRU`는 Gated Recurrent Unit의 약자입니다. 뉴욕 대학교 조경현 교수가 발명한 셀로 유명합니다. 이 셀은 `LSTM`을 간소화한 버전으로 생각할 수 있습니다. 이 셀은 `LSTM`처럼 셀 상태를 계산하지 않고 은닉 상태 하나만 포함하고 있습니다. 먼저 GRU 셀의 그림을 보죠

![스크린샷 2025-03-19 오후 12 38 54](https://github.com/user-attachments/assets/6ef3d6d8-eec4-4e4f-9895-49d40678be50)

- GRU 셀에는 은닉 상태와 입력에 가중치를 곱하고 절편을 더하는 작은 셀이 3개 들어 있습니다. 2개는 시그모이드 활성화 함수를 사용하고 하나는 `tanh` 활성화 함수를 사용합니다. 여기에서도 은닉 상태와 입력에 곱해지는 가중치를 합쳐서 나타냈습니다.
- 맨 왼쪽에서 W<sub>z</sub>를 사용하는 셀의 출력이 은닉 상태에 바로 곱해져 삭제 게이트 역할을 수행합니다. 이와 똑같은 출력을 1에서 뺀 다음에 가장 오른쪽 W<sub>g</sub>를 사용하는 셀의 출력에 곱합니다. 이는 입력되는 정보를 제어하는 역할을 수행합니다. 가운데 W<sub>t</sub>을 사용하는 셀에서 출력된 값은 W<sub>g</sub>셀이 사용할 은닉 상태의 정보를 제어합니다.
- `GRU` 셀은 `LSTM`보다 가중치가 적기 때문에 계산량이 적지만 `LSTM` 못지않은 좋은 성능을 내는 것으로 알려져 있습니다. 다음 섹션에서 `GRU`셀을 사용한 순환 신경망을 만들어 보겠습니다.

## GRU 신경망 훈련하기

```python
model4 = keras.Sequential()

model4.add(keras.layers.Input(shape=(100,)))
model4.add(keras.layers.Embedding(500, 16))
model4.add(keras.layers.GRU(8))
model4.add(keras.layers.Dense(1, activation='sigmoid'))
```

- `LSTM` 클래스를 `GRU` 클래스로 바뀐 것 외에는 이전 모델과 동일합니다. 이 모델의 구조를 확인해보죠.

```python
model4.summary()
```

![스크린샷 2025-03-19 오후 12 43 44](https://github.com/user-attachments/assets/b187e403-fe74-4990-91e6-17a7a0053c6e)

- `GRU` 층의 모델 파라미터 개수를 계산해 보겠습니다. `GRU`셀에는 3개의 작은 셀이 있습니다. 작은 셀에는 입력과 은닉 상태에 곱하는 가중치와 절편이 있습니다. 입력에 곱하는 가중치는 16 X 8 = 128개이고 은닉 상태에 곱하는 가중치는 8 X 8 = 64개 입니다. 그리고 절편은 뉴런마다 하나씩이므로 8개입니다. 모두 더하면 128 + 64 + 8 = 200개입니다. 이런 작은 셀이 3개이므로 모두 600개의 모델 파라미터가 필요합니다. 그런데 `summary()` 메서드의 출력은 624개네요. 무엇이 잘못되었을까요?
- 사실 텐서플로에 기본적으로 구현된 `GRU` 셀의 계산은 앞의 그림과 조금 다릅니다. `GRU` 셀의 초기 버전은 다음 그림과 같이 계산됩니다. 

```python
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model4.compile(optimizer=rmsprop, loss='binary_crossentropy',
               metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-gru-model.keras',
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

history = model4.fit(train_seq, train_target, epochs=100, batch_size=64,
                     validation_data=(val_seq, val_target),
                     callbacks=[checkpoint_cb, early_stopping_cb])
```

```
Epoch 1/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 4s 8ms/step - accuracy: 0.5052 - loss: 0.6931 - val_accuracy: 0.5206 - val_loss: 0.6928
Epoch 2/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 10ms/step - accuracy: 0.5532 - loss: 0.6925 - val_accuracy: 0.5616 - val_loss: 0.6921
Epoch 3/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.5831 - loss: 0.6917 - val_accuracy: 0.5714 - val_loss: 0.6911
Epoch 4/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 7ms/step - accuracy: 0.5901 - loss: 0.6903 - val_accuracy: 0.5844 - val_loss: 0.6893
Epoch 5/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.6033 - loss: 0.6882 - val_accuracy: 0.5938 - val_loss: 0.6866
Epoch 6/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - accuracy: 0.6136 - loss: 0.6849 - val_accuracy: 0.6036 - val_loss: 0.6822
Epoch 7/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 10ms/step - accuracy: 0.6197 - loss: 0.6796 - val_accuracy: 0.6120 - val_loss: 0.6752
Epoch 8/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 4s 7ms/step - accuracy: 0.6300 - loss: 0.6713 - val_accuracy: 0.6294 - val_loss: 0.6640
Epoch 9/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.6418 - loss: 0.6578 - val_accuracy: 0.6488 - val_loss: 0.6453
Epoch 10/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.6644 - loss: 0.6351 - val_accuracy: 0.6844 - val_loss: 0.6111
Epoch 11/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 4s 11ms/step - accuracy: 0.6996 - loss: 0.5917 - val_accuracy: 0.7326 - val_loss: 0.5492
Epoch 12/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.7397 - loss: 0.5360 - val_accuracy: 0.7484 - val_loss: 0.5242
Epoch 13/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.7523 - loss: 0.5158 - val_accuracy: 0.7634 - val_loss: 0.5107
Epoch 14/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.7617 - loss: 0.5014 - val_accuracy: 0.7724 - val_loss: 0.5008
Epoch 15/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 7ms/step - accuracy: 0.7712 - loss: 0.4883 - val_accuracy: 0.7750 - val_loss: 0.4949
Epoch 16/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 4s 10ms/step - accuracy: 0.7816 - loss: 0.4780 - val_accuracy: 0.7762 - val_loss: 0.4916
Epoch 17/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - accuracy: 0.7873 - loss: 0.4701 - val_accuracy: 0.7766 - val_loss: 0.4870
Epoch 18/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.7921 - loss: 0.4630 - val_accuracy: 0.7782 - val_loss: 0.4798
Epoch 19/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.7938 - loss: 0.4567 - val_accuracy: 0.7816 - val_loss: 0.4736
Epoch 20/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.7963 - loss: 0.4512 - val_accuracy: 0.7846 - val_loss: 0.4702
Epoch 21/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 10ms/step - accuracy: 0.7992 - loss: 0.4466 - val_accuracy: 0.7868 - val_loss: 0.4677
Epoch 22/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - accuracy: 0.8031 - loss: 0.4426 - val_accuracy: 0.7878 - val_loss: 0.4655
Epoch 23/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8064 - loss: 0.4392 - val_accuracy: 0.7868 - val_loss: 0.4636
Epoch 24/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8089 - loss: 0.4364 - val_accuracy: 0.7880 - val_loss: 0.4621
Epoch 25/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8102 - loss: 0.4341 - val_accuracy: 0.7888 - val_loss: 0.4608
Epoch 26/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 10ms/step - accuracy: 0.8112 - loss: 0.4320 - val_accuracy: 0.7894 - val_loss: 0.4597
Epoch 27/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 4s 7ms/step - accuracy: 0.8121 - loss: 0.4303 - val_accuracy: 0.7878 - val_loss: 0.4587
Epoch 28/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8134 - loss: 0.4288 - val_accuracy: 0.7882 - val_loss: 0.4578
Epoch 29/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8133 - loss: 0.4274 - val_accuracy: 0.7890 - val_loss: 0.4570
Epoch 30/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 10ms/step - accuracy: 0.8134 - loss: 0.4262 - val_accuracy: 0.7908 - val_loss: 0.4562
Epoch 31/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 4s 7ms/step - accuracy: 0.8124 - loss: 0.4250 - val_accuracy: 0.7892 - val_loss: 0.4554
Epoch 32/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8128 - loss: 0.4240 - val_accuracy: 0.7900 - val_loss: 0.4547
Epoch 33/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 7ms/step - accuracy: 0.8132 - loss: 0.4230 - val_accuracy: 0.7906 - val_loss: 0.4540
Epoch 34/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - accuracy: 0.8131 - loss: 0.4220 - val_accuracy: 0.7908 - val_loss: 0.4533
Epoch 35/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - accuracy: 0.8144 - loss: 0.4211 - val_accuracy: 0.7894 - val_loss: 0.4527
Epoch 36/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8145 - loss: 0.4203 - val_accuracy: 0.7898 - val_loss: 0.4521
Epoch 37/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 7ms/step - accuracy: 0.8139 - loss: 0.4194 - val_accuracy: 0.7904 - val_loss: 0.4515
Epoch 38/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 7ms/step - accuracy: 0.8147 - loss: 0.4187 - val_accuracy: 0.7898 - val_loss: 0.4509
Epoch 39/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 10ms/step - accuracy: 0.8152 - loss: 0.4179 - val_accuracy: 0.7894 - val_loss: 0.4504
Epoch 40/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 4s 7ms/step - accuracy: 0.8154 - loss: 0.4172 - val_accuracy: 0.7884 - val_loss: 0.4499
Epoch 41/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8158 - loss: 0.4165 - val_accuracy: 0.7884 - val_loss: 0.4495
Epoch 42/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 7ms/step - accuracy: 0.8158 - loss: 0.4159 - val_accuracy: 0.7880 - val_loss: 0.4490
Epoch 43/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 10ms/step - accuracy: 0.8160 - loss: 0.4153 - val_accuracy: 0.7882 - val_loss: 0.4486
Epoch 44/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - accuracy: 0.8159 - loss: 0.4147 - val_accuracy: 0.7868 - val_loss: 0.4481
Epoch 45/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8166 - loss: 0.4141 - val_accuracy: 0.7872 - val_loss: 0.4477
Epoch 46/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8166 - loss: 0.4136 - val_accuracy: 0.7864 - val_loss: 0.4473
Epoch 47/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8169 - loss: 0.4131 - val_accuracy: 0.7866 - val_loss: 0.4469
Epoch 48/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - accuracy: 0.8170 - loss: 0.4126 - val_accuracy: 0.7870 - val_loss: 0.4465
Epoch 49/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - accuracy: 0.8172 - loss: 0.4121 - val_accuracy: 0.7870 - val_loss: 0.4461
Epoch 50/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8171 - loss: 0.4116 - val_accuracy: 0.7870 - val_loss: 0.4457
Epoch 51/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 7ms/step - accuracy: 0.8175 - loss: 0.4112 - val_accuracy: 0.7872 - val_loss: 0.4453
Epoch 52/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8175 - loss: 0.4107 - val_accuracy: 0.7876 - val_loss: 0.4449
Epoch 53/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - accuracy: 0.8179 - loss: 0.4103 - val_accuracy: 0.7878 - val_loss: 0.4445
Epoch 54/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 7ms/step - accuracy: 0.8179 - loss: 0.4099 - val_accuracy: 0.7884 - val_loss: 0.4442
Epoch 55/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 7ms/step - accuracy: 0.8177 - loss: 0.4094 - val_accuracy: 0.7894 - val_loss: 0.4438
Epoch 56/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8181 - loss: 0.4090 - val_accuracy: 0.7898 - val_loss: 0.4434
Epoch 57/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - accuracy: 0.8185 - loss: 0.4086 - val_accuracy: 0.7906 - val_loss: 0.4431
Epoch 58/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 8ms/step - accuracy: 0.8183 - loss: 0.4082 - val_accuracy: 0.7906 - val_loss: 0.4427
Epoch 59/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8186 - loss: 0.4078 - val_accuracy: 0.7912 - val_loss: 0.4424
Epoch 60/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - accuracy: 0.8184 - loss: 0.4075 - val_accuracy: 0.7912 - val_loss: 0.4420
Epoch 61/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - accuracy: 0.8179 - loss: 0.4071 - val_accuracy: 0.7920 - val_loss: 0.4417
Epoch 62/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 7ms/step - accuracy: 0.8181 - loss: 0.4067 - val_accuracy: 0.7924 - val_loss: 0.4413
Epoch 63/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8181 - loss: 0.4063 - val_accuracy: 0.7922 - val_loss: 0.4410
Epoch 64/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8186 - loss: 0.4060 - val_accuracy: 0.7920 - val_loss: 0.4407
Epoch 65/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - accuracy: 0.8190 - loss: 0.4056 - val_accuracy: 0.7922 - val_loss: 0.4404
Epoch 66/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 8ms/step - accuracy: 0.8190 - loss: 0.4052 - val_accuracy: 0.7922 - val_loss: 0.4400
Epoch 67/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8193 - loss: 0.4048 - val_accuracy: 0.7926 - val_loss: 0.4397
Epoch 68/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - accuracy: 0.8195 - loss: 0.4045 - val_accuracy: 0.7930 - val_loss: 0.4394
Epoch 69/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 10ms/step - accuracy: 0.8202 - loss: 0.4041 - val_accuracy: 0.7932 - val_loss: 0.4391
Epoch 70/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - accuracy: 0.8204 - loss: 0.4037 - val_accuracy: 0.7938 - val_loss: 0.4388
Epoch 71/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 8ms/step - accuracy: 0.8205 - loss: 0.4034 - val_accuracy: 0.7932 - val_loss: 0.4385
Epoch 72/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8207 - loss: 0.4030 - val_accuracy: 0.7936 - val_loss: 0.4382
Epoch 73/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - accuracy: 0.8207 - loss: 0.4027 - val_accuracy: 0.7940 - val_loss: 0.4380
Epoch 74/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 10ms/step - accuracy: 0.8206 - loss: 0.4023 - val_accuracy: 0.7942 - val_loss: 0.4377
Epoch 75/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 4s 7ms/step - accuracy: 0.8204 - loss: 0.4019 - val_accuracy: 0.7948 - val_loss: 0.4374
Epoch 76/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8208 - loss: 0.4016 - val_accuracy: 0.7944 - val_loss: 0.4371
Epoch 77/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8206 - loss: 0.4012 - val_accuracy: 0.7948 - val_loss: 0.4369
Epoch 78/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 4s 11ms/step - accuracy: 0.8207 - loss: 0.4009 - val_accuracy: 0.7950 - val_loss: 0.4366
Epoch 79/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8206 - loss: 0.4005 - val_accuracy: 0.7946 - val_loss: 0.4363
Epoch 80/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8208 - loss: 0.4001 - val_accuracy: 0.7952 - val_loss: 0.4361
Epoch 81/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 7ms/step - accuracy: 0.8212 - loss: 0.3998 - val_accuracy: 0.7950 - val_loss: 0.4358
Epoch 82/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8213 - loss: 0.3994 - val_accuracy: 0.7954 - val_loss: 0.4356
Epoch 83/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 4s 11ms/step - accuracy: 0.8215 - loss: 0.3990 - val_accuracy: 0.7960 - val_loss: 0.4353
Epoch 84/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8218 - loss: 0.3987 - val_accuracy: 0.7960 - val_loss: 0.4351
Epoch 85/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 8ms/step - accuracy: 0.8223 - loss: 0.3983 - val_accuracy: 0.7958 - val_loss: 0.4348
Epoch 86/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8222 - loss: 0.3980 - val_accuracy: 0.7962 - val_loss: 0.4346
Epoch 87/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8221 - loss: 0.3976 - val_accuracy: 0.7962 - val_loss: 0.4344
Epoch 88/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 4s 11ms/step - accuracy: 0.8225 - loss: 0.3972 - val_accuracy: 0.7960 - val_loss: 0.4341
Epoch 89/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 4s 7ms/step - accuracy: 0.8224 - loss: 0.3969 - val_accuracy: 0.7964 - val_loss: 0.4339
Epoch 90/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8225 - loss: 0.3965 - val_accuracy: 0.7964 - val_loss: 0.4337
Epoch 91/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 7ms/step - accuracy: 0.8226 - loss: 0.3962 - val_accuracy: 0.7968 - val_loss: 0.4335
Epoch 92/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 4s 10ms/step - accuracy: 0.8226 - loss: 0.3958 - val_accuracy: 0.7970 - val_loss: 0.4333
Epoch 93/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 4s 7ms/step - accuracy: 0.8230 - loss: 0.3954 - val_accuracy: 0.7970 - val_loss: 0.4331
Epoch 94/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8229 - loss: 0.3951 - val_accuracy: 0.7968 - val_loss: 0.4329
Epoch 95/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8231 - loss: 0.3947 - val_accuracy: 0.7970 - val_loss: 0.4327
Epoch 96/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - accuracy: 0.8232 - loss: 0.3944 - val_accuracy: 0.7966 - val_loss: 0.4325
Epoch 97/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - accuracy: 0.8236 - loss: 0.3940 - val_accuracy: 0.7962 - val_loss: 0.4323
Epoch 98/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 7ms/step - accuracy: 0.8236 - loss: 0.3936 - val_accuracy: 0.7970 - val_loss: 0.4321
Epoch 99/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - accuracy: 0.8238 - loss: 0.3933 - val_accuracy: 0.7972 - val_loss: 0.4319
Epoch 100/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - accuracy: 0.8237 - loss: 0.3929 - val_accuracy: 0.7978 - val_loss: 0.4317
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
test_seq = pad_sequences(test_input, maxlen=100)

rnn_model = keras.models.load_model('best-2rnn-model.keras')

rnn_model.evaluate(test_seq, test_target)
```

```
782/782 ━━━━━━━━━━━━━━━━━━━━ 5s 5ms/step - accuracy: 0.7866 - loss: 0.4654
[0.4673371911048889, 0.7847599983215332]
```
