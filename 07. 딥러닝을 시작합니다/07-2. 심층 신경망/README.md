# 심층 신경망

## 키워드 정리

- **심층 신경망**
  - 2개 이상의 층을 포함한 신경망입니다.
  - 종종 다층 인공 신경망, 심층 신경망, 딥러닝을 같은 의미로 사용합니다.
- **렐루 함수**
  - 이미지 분류 모델의 은닉층에서 많이 사용하는 활성화 함수입니다.
  - 시그모이드 함수는 층이 많을수록 활성화 함수의 양쪽 끝에서 변화가 작기 때문에 학습이 어려워집니다.
  - 렐루 함수는 이런 문제가 없으며 계산도 간단합니다.
- **옵티마이저**
  - 신경망의 가중치와 절편을 학스바기 위한 알고리즘 또는 방법을 말합니다.
  - 케라스에는 다양한 경사 하강법 알고리즘이 구현되어 있습니다. 대표적으로 SGD, 네스트로프 모멘텀, RMSprop, Adam 등이 있습니다.

## 핵심 패키지와 함수

### TensorFlow

- **add()**
  - 케라스 모델에 층을 추가하는 메서드입니다.
  - 케라스 모델의 `add()` 메서드는 `keras.layers` 패키지 아래에 있는 층의 객체를 입력받아 신경망 모델에 추가합니다.
  - `add()` 메서드를 호출하여 전달한 순서대로 층이 차례대로 늘어납니다.
- **summary()**
  - 케라스 모델의 정보를 출력하는 메서드입니다.
  - 모델에 추가된 층의 종류와 순서, 모델 파라미터 개수를 출력합니다. 층을 만들 때 name 매개변수로 이름을 지정하면 `summary()` 메서드 출력에서 구분하기 쉽습니다.
- **SGD**
  - 기본 경사 하강법 옵티마이저 클래스 입니다.
  - `learning_rate` 매개변수로 학습률을 지정하며 기본값은 0.01입니다.
  - `momentum` 매개변수에 0 이상의 값을 지정하면 모멘텀 최적화를 수행합니다.
  - `nesterov` 매개변수를 True로 설정하면 네트테로프 모멘텀 최적화를 수행합니다.
- **Adagrad**
  - `Adagrad` 옵티마이저 클래스입니다.
  - `learning_rate` 매개변수로 학습률을 지정하며 기본값은 0.001입니다.
  - `Adagrad`는 그레디언트 제곱을 누적하여 학습률을 나눕니다. `initial_accumulator_value` 매개변수에서 누적 초깃값을 지정할 수 있으며 기본값은 0.1입니다.
- **RMSprop**
  - `RMSProp` 옵티마이저 클래스입니다.
  - `learning_rate` 매개변수로 학습률을 지정하며 기본값은 0.001입니다.
  - `Adagrad` 처럼 그레이디언트 제곱으로 학습률을 나누지만 최근 그레이디언트를 사용하기 위해 지수 감소를 사용합니다. `rho` 매개변수에서 감소 비율을 지정하며 기본값은 0.9입니다.
- **Adam**
  - `Adam`은 Adam 옵티마이저 클래스 입니다.
  - `learning_rate` 매개변수로 학습률을 지정하며 기본값은 0.001입니다.
  - 모멘텀 최적화에 있는 그레이디언트의 지수 감소 평균을 조절하기 위해 `beta_1` 매개변수가 있으며 기본값은 0.9입니다.
  - `RMSprop`에 있는 그레이디언트 제곱의 지수 감소 평균을 조절하기 위해 `beta_2` 매개변수가 있으며 기본값은 0.999입니다.

## 2개의 층

- 케라스 API를 사용해서 패션 MNIST 데이터셋을 불러오겠습니다.

```python
from tensorflow import keras

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
```

- 그 다음 이미지의 픽셀값을 0\~255 범위에서 0\~1 사이로 변환하고 28 X 28 크기의 2차원 배열을 784 크기의 1차원 배열로 펼칩니다.
- 마지막으로 사이킷런의 `train_test_split()` 함수로 훈련 세트와 검증 세트로 나눕니다.

```python
from sklearn.model_selection import train_test_split

train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)
```

- 이제 인공 신경망 모델에 층을 2개 추가해 봅니다.
- 여기에서 만들 모델의 대략적인 구조는 다음 그림과 같습니다.

```python
dense1 = keras.layers.Dense(100, activation='sigmoid', input_shape=(784,))
dense2 = keras.layers.Dense(10, activation='softmax')
```

## 심층 신경망 만들기

```python
model = keras.Sequential([dense1, dense2])
```

```python
model.summary()
```

## 층을 추가하는 다른 방법

```python
model = keras.Sequential([
    keras.layers.Dense(100, activation='sigmoid', input_shape=(784,), name='hidden'),
    keras.layers.Dense(10, activation='softmax', name='output')
], name='패션 MNIST 모델')
```

```python
model.summary()
```

```python
model = keras.Sequential()
model.add(keras.layers.Dense(100, activation='sigmoid', input_shape=(784,)))
model.add(keras.layers.Dense(10, activation='softmax'))
```

```python
model.summary()
```

```python
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_scaled, train_target, epochs=5)
```

## 렐루 활성화 함수

```python
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
```

```python
model.summary()
```

```python
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

train_scaled = train_input / 255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)
```

```python
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_scaled, train_target, epochs=5)
```

```python
model.evaluate(val_scaled, val_target)
```

## 옵티마이저

```python
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

```python
sgd = keras.optimizers.SGD()
model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

```python
sgd = keras.optimizers.SGD(learning_rate=0.1)
```

```python
sgd = keras.optimizers.SGD(momentum=0.9, nesterov=True)
```

```python
adagrad = keras.optimizers.Adagrad()
model.compile(optimizer=adagrad, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

```python
rmsprop = keras.optimizers.RMSprop()
model.compile(optimizer=rmsprop, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

```python
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
```

```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_scaled, train_target, epochs=5)
```

```python
model.evaluate(val_scaled, val_target)
```
