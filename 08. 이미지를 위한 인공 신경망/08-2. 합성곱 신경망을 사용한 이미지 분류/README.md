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

```python
model = keras.Sequential()
```

```python
model.add(keras.layers.Input(shape=(28,28,1)))
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu',
                              padding='same'))
```

```python
model.add(keras.layers.MaxPooling2D(2))
```

```python
model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu',
                              padding='same'))
model.add(keras.layers.MaxPooling2D(2))
```

```python
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10, activation='softmax'))
```

```python
model.summary()
```

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
