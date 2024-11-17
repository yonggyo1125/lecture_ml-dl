# 신경망 모델 훈련

## 키워드 정리

- **드롭아웃**
  - 은닉층에 있는 뉴런의 출력을 랜덤하게 꺼서 과대적합을 막는 기법입니다.
  - 드롭아웃은 훈련 중에 적용되며 평가나 예측에서는 적용하지 않습니다.
  - 텐서플로는 이를 자동으로 처리합니다.
- **콜백**
  - 케라스 모델을 훈련하는 도중에 어떤 작업을 수행할 수 있도록 도와주는 도구입니다.
  - 대표적으로 최상의 모델을 자동으로 저장해 주거나 검증 점수가 더 이상 향상되지 않으면 일찍 종료할 수 있습니다.
- **조기 종료**
  - 검증 점수가 더 이상 감소하지 않고 상승하여 과대적합이 일어나면 훈련을 계속 진행하지 않고 멈추는 기법입니다.
  - 이렇게 하면 계산 비용과 시간을 절약할 수 있습니다.

## 핵심 패키지와 함수

### TensorFlow

- **Dropout**
  - 드롭아웃 층입니다.
  - 첫 번째 매개변수로 드롭아웃 할 비율(r)을 지정합니다.
  - 드롭아웃 하지 않는 뉴런의 출력은 1 / (1 - r)만큼 증가시켜 출력의 총합이 같도록 만듭니다.
- **save_weights()**는 모든 층의 가중치와 절편을 파일에 저장합니다.
  - 첫 번째 매개변수에 저장할 파일을 지정합니다.
  - `save_format` 매개변수에서 저장할 파일 포맷을 지정합니다. 기본적으로 텐서플로의 `Checkpoint` 포맷을 사용합니다. 이 매개변수를 "h5"로 지정하거나 파일의 확장자가 '.h5'이면 HDF5 포맷으로 저장됩니다.
- **load_weights()**는 모든 층의 가중치와 절편을 파일에 읽습니다.
  - 첫 번째 매개변수에 읽을 파일을 지정합니다.
- **save()**는 모델 구조와 모든 가중치와 절편을 파일에 저장합니다.
  - 첫 번째 매개변수에 저장할 파일을 지정합니다.
  - `save_format` 매개변수에서 저장할 파일 포맷을 지정합니다. 기본적으로 텐서플로의 **SavedModel** 포맷을 사용합니다. 이 매개변수를 'h5'로 지정하거나 파일의 확장자가 '.h5'이면 HDF5 포맷으로 저장됩니다.
- **load_model()**은 `model.save()`로 저장된 모델을 로드합니다.
  - 첫 번째 매개변수에 읽을 파일을 지정합니다.
- **ModelCheckPoint**는 케라스 모델과 가중치를 일정 간격으로 저장합니다.
  - 첫 번째 매개변수에 저장할 파일을 지정합니다.
  - `monitor` 매개변수는 모니터링할 지표를 지정합니다. 기본값은 `val_loss`로 검증 손실을 관찰합니다.
  - `save_weights_only` 매개변수의 기본값은 False로 전체 모델을 저장합니다. True로 지정하면 모델의 가중치와 절편만 저장합니다.
  - `save_best_only` 매개변수를 True로 지정하면 가장 낮은 검증 점수를 만드는 모델을 저장합니다.
- **EarlyStopping**은 관심 지표가 더이상 향상하지 않으면 훈련을 중지합니다.
  - `monitor` 매개변수는 모니터링할 지표를 지정합니다. 기본값은 `val_loss`로 검증 손실을 관찰합니다.
  - `patience` 매개변수에 모델이 더 이상 향상되지 않고 지속할 수 있는 최대 에포크 횟수를 지정합니다.
  - `restore_best_weights` 매개변수에 최상의 모델 가중치를 복원할지 지정합니다. 기본값은 False입니다.

### NumPy

- **argmax**는 배열에서 축을 따라 최대값의 인덱스를 반환합니다.
  - `axis` 매개변수에서 어떤 축을 따라 최댓값을 찾을지 지정합니다. 기본값은 None으로 전체 배열에서 최대값을 찾습니다.

## 손실 곡선

- 2절에서 `fit()` 메서드로 모델을 훈련하면 훈련 과정이 상세하게 출력되어 확인할 수 있었습니다. 여기에는 에포크 횟수, 손실, 정확도 등이 있었습니다.
- 그런데 이 출력의 마지막에 다음과 같은 메시지를 보았을 것

```
<keras.src.callbacks.history.History at 0x7ba278da8df0>
```

- 노트북의 코드 셀은 `print()` 명령을 사용하지 않더라도 마지막 라인의 실행 결과를 자동으로 출력합니다. 즉 이 메시지는 `fit()` 메서드의 실행 결과를 출력한 것입니다. 다시 말해 fit() 메서드가 무엇인지 반환된다는 증거입니다.
- 실은 케라스의 `fit()` 메서드는 **History** 클래스 객체를 반환합니다. 
- **History** 객체에는 훈련 과정에서 계산한 지표, 즉 손실과 정확도 값이 저장되어 있습니다. 이 값을 사용하면 그래프를 그릴 수 있습니다.
- 먼저 이전 절에서 사용했던 것과 같이 패션 MNIST 데이터셋을 적재하고 훈련 세트와 검증 세트로 나눕니다.

```python
from tensorflow import keras
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()

train_scaled = train_input / 255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)
```

- 그다음 모델을 만들겠습니다. 그런데 이전 절과는 달리 모델을 만드는 간단한 함수를 정의하겠습니다. 이 함수는 하나의 매개변수를 가집니다. 

```python
def model_fn(a_layer=None):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(100, activation='relu'))
    if a_layer:
        model.add(a_layer)
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model
```

- if 구문을 제외하면 이 코드는 이전 절에서 만든 것과 동일한 모델을 만듭니다. if 구문의 역할은 `model_fn()` 함수에 (a_layer 매개변수로) 케라스 층을 추가하면 은닉층 뒤에 또 하나의 층을 추가하는 것입니다. 신경망 모델을 만드는 것이 마치 프로그래밍을 하는 것과 같습니다.
- 여기서는 `a_layer` 매개변수로 층을 추가하지 않고 단순하게 `model_fn()` 함수를 호출합니다. 그리고 모델 구조를 출력하면 이전 절과 동일한 모델이라는 것을 확인할 수 있습니다.

```python
model = model_fn()

model.summary()
```

<img width="500" alt="스크린샷 2024-11-17 오후 7 41 33" src="https://github.com/user-attachments/assets/24ee1ef2-4ad5-4132-b1cc-4af9cf0dfd10">

- 이전 절과 동일하게 모델을 훈련하지만 `fit()` 메서드의 결과를 history 변수에 담아 보겠습니다.

```python
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_scaled, train_target, epochs=5, verbose=0)
```

> verbose 매개변수는 훈련 과정 출력을 조절합니다. 기본값은 1로 이전 절에서처럼 에포크마다 진행 막대와 함께 손실등의 지표가 출력됩니다. 2로 바꾸면 진행 막대를 빼고 출력됩니다. 이번 절에서는 훈련 결과를 그래프로 나타내는 대신 verbose 매개변수를 0으로 지정하여 훈련 과정을 나타내지 않겠습니다.

- history 객체에는 훈련 측정값이 담겨 있는 history 딕셔너리가 들어있습니다. 이 딕셔너리에 어떤 값이 들어 있는지 확인해 봅시다.


```python
print(history.history.keys())
```

```
dict_keys(['accuracy', 'loss'])
```

- 손실과 정확도가 포함되어 있습니다. 이전 절에서 언급했듯이 케라스는 기본적으로 에포크마다 손실을 계산합니다. 정확도는 `compile()` 메서드에서 `metrics` 매개변수에 `accuracy`를 추가했기 때문에 `history` 속성에 포함되었습니다.
- `history` 속성에 포함된 손실과 정확도는 에포크마다 계산한 값이 순서대로 나열된 단순한 리스트입니다. 맷플롯립을 사용해 쉽게 그래프로 그릴 수 있습니다.


```python
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
```

<img width="382" alt="스크린샷 2024-11-17 오후 7 50 17" src="https://github.com/user-attachments/assets/3e31002e-f7ca-4946-93ab-70080954e001">



```python
plt.plot(history.history['accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
```

<img width="392" alt="스크린샷 2024-11-17 오후 7 50 29" src="https://github.com/user-attachments/assets/744c91f0-054d-4e0c-a3b6-9b4c4a105307">


```python
model = model_fn()
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_scaled, train_target, epochs=20, verbose=0)
```

```python
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
```


<img width="395" alt="스크린샷 2024-11-17 오후 7 50 41" src="https://github.com/user-attachments/assets/8008b34b-7000-4d5b-8f93-373cd623bc7b">


## 검증 손실

```python
model = model_fn()
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_scaled, train_target, epochs=20, verbose=0,
                    validation_data=(val_scaled, val_target))
```

```python
print(history.history.keys())
```

```
dict_keys(['accuracy', 'loss', 'val_accuracy', 'val_loss'])
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
model = model_fn()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_scaled, train_target, epochs=20, verbose=0,
                    validation_data=(val_scaled, val_target))
```

```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```

## 드롭아웃

```python
model = model_fn(keras.layers.Dropout(0.3))

model.summary()
```

```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_scaled, train_target, epochs=20, verbose=0,
                    validation_data=(val_scaled, val_target))
```

```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```

## 모델 저장과 복원

```python
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_scaled, train_target, epochs=10, verbose=0,
                    validation_data=(val_scaled, val_target))
```

```python
model.save('model-whole.keras')
```

```python
model.save_weights('model.weights.h5')
```

```
!ls -al model*
```

```
-rw-r--r-- 1 root root 971928 Aug  6 06:42 model.weights.h5
-rw-r--r-- 1 root root 975720 Aug  6 06:42 model-whole.keras
```

```python
model = model_fn(keras.layers.Dropout(0.3))

model.load_weights('model.weights.h5')
```

```python
import numpy as np

val_labels = np.argmax(model.predict(val_scaled), axis=-1)
print(np.mean(val_labels == val_target))
```

```python
model = keras.models.load_model('model-whole.keras')

model.evaluate(val_scaled, val_target)
```

## 콜백

```python
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.keras',
                                                save_best_only=True)

model.fit(train_scaled, train_target, epochs=20, verbose=0,
          validation_data=(val_scaled, val_target),
          callbacks=[checkpoint_cb])
```

```python
model = keras.models.load_model('best-model.keras')

model.evaluate(val_scaled, val_target)
```

```python
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.keras',
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2,
                                                  restore_best_weights=True)

history = model.fit(train_scaled, train_target, epochs=20, verbose=0,
                    validation_data=(val_scaled, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])
```

```python
print(early_stopping_cb.stopped_epoch)
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
