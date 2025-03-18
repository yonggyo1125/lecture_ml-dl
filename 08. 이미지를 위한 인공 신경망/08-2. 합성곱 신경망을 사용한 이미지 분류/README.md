# 합성곱 신경망을 사용한 이미지 분류

## 키워드 정리

### Tensorflow

- **Conv2D** : 입력의 너비와 높이 방향의 합성곱 연산을 구현한 클래스입니다.
  - 첫 번째 매개변수는 합성곱 필터의 개수입니다.
  - **kernel_size** 매개변수는 필터의 커널 크기를 지정합니다. 가로세로 크기가 같은 경우 정수 하나로, 다른 경우 정수의 튜플로 지정할 수 있습니다. 일반적으로 커널의 가로세로 크기는 동일합니다. 커널의 깊이는 입력의 깊이와 동일하게 때문에 따로 지정하지 않습니다.
  - **strides** 매개변수는 필터의 이동 간격을 지정합니다. 가로세로 크기가 같은 경우 정수 하나로, 다른 경우 정수의 튜플로 지정할 수 있습니다. 일반적으로 가로세로 스트라이드 크기는 동일합니다. 기본값은 1입니다.
  - **padding** 매개변수는 입력의 패딩 타입을 지정합니다. 기본값 `valid`는 패딩을 하지 않습니다. `same`은 합성곱 층의 출력의 가로세로 크기를 입력과 동일하게 맞추도록 입력에 패딩을 추가합니다.
  - `activation` 매개변수는 합성곱 층에 적용할 활성화 함수를 지정합니다.
- **MaxPooling2D** : 입력의 너비와 높이를 줄이는 풀링 연산을 구현한 클래스입니다.
  - 첫 번째 매개변수는 풀링의 크기를 지정하며, 가로세로 크기가 같은 경우 정수 하나로, 다른 경우 정수의 튜플로 지정할 수 있습니다. 일반적으로 풀링의 가로세로 크기는 같게 지정합니다.
  - **strides** 매개변수는 풀링의 이동 간격을 지정합니다. 기본값은 풀링의 크기와 동일합니다. 즉 입력 위를 겹쳐서 풀링하지 않습니다.
  - **padding** 매개변수는 입력의 패딩 타입을 지정합니다. 기본값은 `valid`는 패딩을 하지 않습니다. `same`은 합성곱 층의 출력의 가로세로 크기를 입력과 동일하게 맞추도록 입력에 패딩을 추가합니다.
- **plot_model()** : 케라스 모델 구조를 주피터 노트북에 그리거나 파일로 저장합니다.
  - 첫 번째 매개변수에 케라스 모델 객체를 전달합니다.
  - **to_file** 매개변수에 파일 이름을 지정하면 그림을 파일로 저장합니다.
  - **show_shapes** 매개변수를 `True`로 지정하면 층의 입력, 출력 크기를 표시합니다. 기본값은 `False`입니다.
  - **show_layer_names** 매개변수를 `True`로 지정하면 층 이름을 출력합니다. 기본값이 `True`입니다.

### matplotlib

- **bar()**는 막대그래프를 출력합니다.
  - 첫 번쨰 매개변수에 x축의 값을 리스트나 넘파이 배열로 전달합니다.
  - 두 번째 매개변수에 막대의 y축 값을 리스트나 넘파이 배열로 전달합니다.
  - `width` 매개변수에서 막대의 두께를 지정할 수 있습니다. 기본값은 0.8입니다.

## 패션 MNIST 데이터 불러오기

- 먼저 주피터 노트북에서 케라스 API를 사용해 패션 MNIST 데이터를 불러오고 적절히 전처리하겠습니다. 이 작업은 7장에서 했던 것과 아주 비슷합니다. 데이터 스케일을 0\~255사이에서 0\~1 사이로 바꾸고 훈련 세트와 검증 세트로 나눕니다.
- 여기에서는 한 가지 작업이 다릅니다. 완전 연결 신경망에서는 입력 이미지를 밀집층에 연결하기 위해 일렬로 펼쳐야 합니다. 이 작업을 위해 넘파이 `reshape()` 메서드를 사용하거나 `Flatten` 클래스를 사용했습니다. 합성곱 신경망은 2차원 이미지를 그대로 사용하기 때문에 이렇게 일렬로 펼치지 않습니다.
- 다만 8장 1절에서 언급했듯이 입력 이미지는 항상 깊이(채널) 차원이 있어야 합니다. 흑백 이미지의 경우 채널 차원이 없는 2차원 배열이지만 `Conv2D` 층을 사용하기 위해 마지막에 이 채널 차원을 추가해야 합니다. 넘파이 `reshape()` 메서드를 사용해 전체 배열 차원을 그대로 유지하면서 마지막에 차원을 간단히 추가할 수 있습니다.

```python
from tensorflow import keras
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()

train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)
```

- 이제 (48000, 28, 28) 크기인 train_input이 (48000, 28, 28, 1) 크기인 train_scaled가 되었습니다.

![스크린샷 2025-03-18 오후 12 05 26](https://github.com/user-attachments/assets/aacdfc7c-decb-41e6-9335-2fe1c73786a2)

- 그 외 다른 작업은 7장에서 했던 것과 정확히 같습니다. 데이터를 준비했으니 이제 본격적으로 합성곱 신경망을 만들어 보죠.

## 합성곱 신경망 만들기

- 1절에서 설명했듯이 전형적인 합성곱 신경망의 구조는 합성곱 층으로 이미지에서 특징을 감지한 후 밀집층으로 클래스에 따른 분류 확률을 계산합니다. 케라스의 `Sequential` 클래스를 사용해 순서대로 이 구조를 정의해 보겠습니다.
- 먼저 `Sequential` 클래스의 객체를 만들고 첫 번째 합성곱 층인 `Conv2D`를 추가합니다. 이 클래스는 다른 층 클래스와 마찬가지로 `keras.layers` 패키지 아래에 있습니다. 여기에서는 이전 장에서 보았던 모델의 `add()` 메서드를 사용해 층을 하나씩 차례대로 추가하겠습니다.

```python
model = keras.Sequential()
model.add(keras.layers.Input(shape=(28,28,1)))
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu',
                              padding='same'))
```

- 이 코드의 매개변수를 자세히 살펴보겠습니다. 이 합성곱 층은 32개의 필터를 사용합니다. 커널의 크기는 (3,3)이고 렐루 활성화 함수와 세임 패딩을 사용합니다.
- 완전 연결 신경망에서처럼 케라스 신경망 모델의 첫 번째 층에서 입력된 차원을 지정해 주어야 합니다. 앞서 패션 MNIST 이미지를 (28,28)에서 (28,28,1)로 변경했던 것을 기억하시죠? `Input`의 `shape`매개변수를 이 값으로 지정합니다.

- 그다음 풀링 층을 추가합니다. 케라스는 최대 풀링과 평균 풀링을 `keras.layers` 패키지 아래 `MaxPooling2D`와 `AveragePooling2D` 클래스로 제공합니다. 전형적인 풀링 크기인 (2,2) 풀링을 사용해 보죠. `Conv2D` 클래스의 kernel_size처럼 가로세로 크기가 같으면 정수 하나로 지정할 수 있습니다.

```python
model.add(keras.layers.MaxPooling2D(2))
```

- 패션 MNIST 이미지가 (28, 28) 크기에 세임 패딩을 적용했기 때문에 합성곱 층에서 출력된 특성 맵의 가로세로 크기는 입력과 동일합니다. 그 다음 (2,2) 풀링을 적용했으므로 특성 맵의 크기는 절반으로 줄어듭니다. 합성곱 층에서 32개의 필터를 사용했기 때문에 이 특성 맵의 깊이는 32가 됩니다. 따라서 최대 풀링을 통과한 특성 맵의 크기는 (14, 14, 32)가 될 것입니다. 나중에 각 층의 출력 크기를 `summary()`메서드로 확인해 보겠습니다.
- 첫 번째 합성곱-풀링 층 다음에 두 번째 합성곱-풀링 층을 추가해 보겠습니다. 두 번째 합성곱-풀링 층은 첫 번쨰와 거의 동일합니다. 필터의 개수를 64개로 늘린 덤만 다릅니다.

```python
model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu',
                              padding='same'))
model.add(keras.layers.MaxPooling2D(2))
```

- 첫 번째 합성곱-풀링 층과 마찬가지로 이 합성곱 층은 세임 패딩을 사용합니다. 따라서 입력의 가로세로 크기를 줄이지 않습니다. 이어지는 풀링 충에서 이 크기를 절반으로 줄입니다. 64개의 필터를 사용했으므로 최종적으로 만들어지는 특성 맵의 크기는 (7, 7, 64)가 될 것입니다.
- 이제 이 3차원의 특성 맵을 일렬로 펼칠 차례입니다. 이렇게 하는 이유는 마지막에 10개의 뉴런을 가진(밀집) 출력층에서 확률을 계산하기 떄문입니다. 여기에서는 특성 맵을 일렬로 펼쳐서 바로 출력층에 전달하지 않고 중간에 하나의 밀집 은닉층을 하나 더 두도록 하겠습니다. 즉 `Flatten` 클래스 다음에 `Dense` 은닉층, 마지막으로 `Dense` 출력층의 순서대로 구성합니다.

```python
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10, activation='softmax'))
```

- 은닉층과 출력층 사이에 드롭아웃을 넣었습니다. 드롭아웃 층이 은닉층의 과대적합을 막아 성능을 조금 더 개선해 줄 것입니다. 은닉층은 100개의 뉴런을 사용하고 활성화 함수는 합성곱 층과 마찬가지로 렐루 함수를 사용합니다. 패션 MNIST 데이터셋은 클래스 10개를 분류하는 다중 분류 문제이므로 마지막 층의 활성화 함수는 소프트맥스를 사용합니다.
- 이렇게 합성곱 신경망의 구성을 마쳤습니다. 앞 절에서 커널, 패딩, 풀링 등을 잘 이애했다면 케라스 API를 사용해 손쉽게 다양한 구성을 실험해 볼 수 있습니다.
- 케라스 모델의 구성을 마쳤으니 `summary()` 메서드로 모델 구조를 출력해 보겠습니다.

```python
model.summary()
```

![스크린샷 2025-03-18 오후 12 21 26](https://github.com/user-attachments/assets/6c362b31-7c0f-4a2d-ad73-71dbdd5fec63)

- `summary()` 메서드의 출력 결과를 보면 합성곱 층과 풀링 층의 효과가 잘 나타나 있습니다. 첫 번째 합성곱 층을 통과하면서 특성 맵의 깊이는 32가 되고 두 번째 합성곱에서 특성 맵의 크기가 64로 늘어납니다. 반면 특성 맵의 가로세로 크기는 첫 번째 풀링 층에서 절반으로 줄어들고 두 번째 풀링층에서 다시 절반으로 더 줄어듭니다. 따라서 최종 특성 맵의 크기는 (7,7,64)입니다.
- 완전 연결 신경망에서 했던 것처럼 모델 파라미터 개수를 계산해 보죠. 첫 번째 합성공 층은 32개의 필터를 가지고 있고 크기가 (3,3), 깊이가 1입니다. 또 필터마다 하나의 절편이 있습니다.
- 따라서 총 3 X 3 X 1 X 32 + 32 = 320개의 파라미터가 있습니다.
- 두 번째 합성곱 층은 64개의 필터를 사용하고 크기가 (3,3), 깊이가 32입니다. 역시 필터마다 하나의 절편이 있습니다. 따라서 총 3 X 3 X 34 X 64 = 18,496개의 파라미터가 있습니다. 층의 구조를 잘 이해하고 있는지 확인하려면 이렇게 모델 파라미터의 개수를 계산해 보세요.
- `Flatten` 클래스에서 (7, 7, 64) 크기의 특성 맵을 1차원 배열로 펼치면 (3136,) 크기의 배열이 됩니다. 이를 100개의 뉴런과 완전히 연결해야 하므로 은닉층의 모델 파라미터 개수는 3,136 X 100 + 100 = 313,700개 입니다. 마찬가지 방식으로 계산하면 마지막 출력층의 모델 파라미터 개수는 1,010개 입니다.
- 합성곱 신경망 모델을 잘 구성했고 각 층의 파라미터 개수를 검증해 보았습니다. 케라스는 `summary()` 메서드 외에 층의 구성을 그림으로 표현해 주는 `plot_model()` 함수를 `keras.utils` 패키지에서 제공합니다. 이 함수에 앞에서 만든 model 객체를 넣어 호출해 보죠.

```python
keras.utils.plot_model(model)
```

```python
keras.utils.plot_model(model, show_shapes=True)
```

```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.keras',
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2,
                                                  restore_best_weights=True)

history = model.fit(train_scaled, train_target, epochs=20,
                    validation_data=(val_scaled, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])
```

```
Epoch 1/20
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 14s 3ms/step - accuracy: 0.7493 - loss: 0.7043 - val_accuracy: 0.8833 - val_loss: 0.3211
Epoch 2/20
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - accuracy: 0.8726 - loss: 0.3552 - val_accuracy: 0.8975 - val_loss: 0.2714
Epoch 3/20
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - accuracy: 0.8905 - loss: 0.2989 - val_accuracy: 0.9087 - val_loss: 0.2455
Epoch 4/20
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 4s 3ms/step - accuracy: 0.9043 - loss: 0.2608 - val_accuracy: 0.9100 - val_loss: 0.2380
Epoch 5/20
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 6s 4ms/step - accuracy: 0.9113 - loss: 0.2400 - val_accuracy: 0.9136 - val_loss: 0.2326
Epoch 6/20
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 4s 3ms/step - accuracy: 0.9184 - loss: 0.2174 - val_accuracy: 0.9201 - val_loss: 0.2193
Epoch 7/20
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - accuracy: 0.9263 - loss: 0.2022 - val_accuracy: 0.9151 - val_loss: 0.2330
Epoch 8/20
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - accuracy: 0.9314 - loss: 0.1847 - val_accuracy: 0.9208 - val_loss: 0.2206
```

```python
import matplotlib.pyplot as plt
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
model.evaluate(val_scaled, val_target)
```

```
375/375 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9231 - loss: 0.2125
[0.21925583481788635, 0.9200833439826965]
```

```python
plt.imshow(val_scaled[0].reshape(28, 28), cmap='gray_r')
plt.show()
```

```python
preds = model.predict(val_scaled[0:1])
print(preds)
```

```
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 411ms/step
[[8.7171188e-19 1.7660993e-26 4.9542759e-21 1.7801612e-19 6.3786056e-18
  4.7823439e-21 2.4926043e-19 3.2195997e-17 1.0000000e+00 4.5894585e-22]]
```

```python
plt.bar(range(1, 11), preds[0])
plt.xlabel('class')
plt.ylabel('prob.')
plt.show()
```

```python
classes = ['티셔츠', '바지', '스웨터', '드레스', '코트',
           '샌달', '셔츠', '스니커즈', '가방', '앵클 부츠']
```

```python
import numpy as np
print(classes[np.argmax(preds)])
```

```python
test_scaled = test_input.reshape(-1, 28, 28, 1) / 255.0
```

```python
model.evaluate(test_scaled, test_target)
```

```
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.9114 - loss: 0.2617
[0.2516588270664215, 0.9125000238418579]
```
