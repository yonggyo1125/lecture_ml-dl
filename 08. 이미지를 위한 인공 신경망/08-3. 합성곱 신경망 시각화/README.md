# 합성곱 신경망의 시각화

## 키워드 정리

- **가중치 시각화** : 합성곱 층의 가중치를 출력하는 것을 말합니다. 합성곱 신경망은 주로 이미지를 다루기 때문에 가중치가 시각적인 패턴을 학습하는지 알아볼 수 있습니다.
- **특성 맵 시각화** : 합성곱 층의 활성화 출력층을 이미지로 그리는 것을 말합니다. 가중치 시각화와 함께 비교하여 각 필터가 이미지의 어느 부분을 활성화시키는지 확인할 수 있습니다.
- **함수형 API** : 케라스에서 신경망 모델을 만드는 방법 중 하나입니다. `Model` 클래스에 모델의 입력과 출력을 지정합니다. 전형적으로 입력은 `Input()` 함수를 사용하여 정의하고 출력은 마지막 층의 출력으로 정의합니다.

### TensorFlow

- **Model**은 케라스 모델을 만드는 클래스입니다.
  - 첫 번째 매개변수인 `inputs` 모델의 입력 또는 입력의 리스트를 지정합니다.
  - 두 번째 매개변수인 `outputs`에 모델의 출력 또는 출력의 리스트를 지정합니다.
  - `name` 매개변수에 모델의 이름을 지정할 수 있습니다.

## 시작하기 전에

- 합성곱 신경망은 특히 이미지에 있는 특징을 찾아 압축하는 데 뛰어난 성능을 냅니다. 이번 절에서는 합성곱 층이 이미지에서 어떤 것을 학습했는지 알아보기 위해 합성곱 층의 가중치와 특성 맵을 그림으로 시각화해 보겠습니다. 이를 통해 합성곱 신경망의 동작 원리에 대한 통찰을 키울 수 있습니다.
- 지금까지는 케라스의 `Sequential` 클래스만 사용했습니다. 케라스는 좀 더 복잡한 모델을 만들 수 있는 함수형 API를 제공합니다. 이번 절에서 함수형 API가 무엇인지 살펴보고 합성곱 층의 특성 맵을 시각화하는 데 사용해 보겠습니다.
- 이 절에서는 2절에서 훈련했던 합성곱 신경망의 체크포인트 파일을 사용합니다. 이 파일은 최적의 에포크까지 훈련한 모델 파라미터를 저장하고 있습니다.

## 가중치 시각화

- 합성곱 층은 여러 개의 필터를 사용해 이미지에서 특징을 학습합니다. 각 필터는 커널이라고 부르는 가중치와 절편을 가지고 있죠. 일반적으로 절편은 시각적으로 의미가 있지 않습니다. 가중치는 입력 이미지의 가차원 영역에 적용되어 어떤 특징을 크게 두드러지게 표현하는 역할을 합니다.
- 예를 들어 다음과 같은 가중치는 둥금 모서리가 있는 영역에서 크게 활성화되고 그렇지 않은 영역에서는 낮은 값을 만들 것입니다.

![스크린샷 2025-03-18 오후 1 33 43](https://github.com/user-attachments/assets/883b9627-d39d-4285-b4be-2346efeb202e)

- 이 필터의 가운데 곡선 부분의 가중치 값은 높고 그 외 부분의 가중치 값은 낮을 것입니다. 이렇게 해야 둥근 모서리가 있는 입력과 곱해져서 큰 출력을 만들기 때문입니다.
- 그럼 2절에서 만든 모델이 어떤 가중치를 학습했는지 확인하기 위해 체크포인트 파일을 읽어 들이겠습니다.

```python
from tensorflow import keras
model = keras.models.load_model('best-cnn-model.keras')
```

- 케라스 모델에 추가한 층은 layers 속성에 저장되어 있습니다. 이 속성은 파이썬 리스트입니다. `model.layers`를 출력해 보겠습니다.

```python
model.layers
```

```
[<Conv2D name=conv2d, built=True>,
 <MaxPooling2D name=max_pooling2d, built=True>,
 <Conv2D name=conv2d_1, built=True>,
 <MaxPooling2D name=max_pooling2d_1, built=True>,
 <Flatten name=flatten, built=True>,
 <Dense name=dense, built=True>,
 <Dropout name=dropout, built=True>,
 <Dense name=dense_1, built=True>]
```

- `models.layers` 리스트에 이전 절에서 추가했던 `Conv2D`, `MaxPooling2D` 층이 번갈아 2번 연속 등장합니다. 그 다음 `Flatten`층과 `Dense` 층, `Dropout` 층이 차례대로 등장합니다. 마지막에 `Dense` 출력층이 놓여 있습니다.
- 그럼 첫 번째 합성곱 층의 가중치를 조사해 보겠습니다. 층의 가중치와 절편은 층의 `weights` 속성에 저장되어 있습니다. `weights`도 파이썬 리스트입니다. 다음 코드에서 처럼 layers 속성의 첫 번째 원소를 선택해 `weights`의 첫 번째 원소 (가중치)와 두 번째 원소 (절편)의 크기를 출력해 보죠.

```python
conv = model.layers[0]

print(conv.weights[0].shape, conv.weights[1].shape)
```

```
(3, 3, 1, 32) (32,)
```

- 이전 절에서 커널 크기를 (3,3)으로 지정했던 것을 기억하시죠? 이 합성곱 층에 전달되는 입력의 깊이가 1이므로 실제 커널 크기는 (3,3,1) 입니다. 또 필터 개수가 32개 이므로 `weights`의 첫 번째 원소인 가중치의 크기는 (3, 3, 1, 32)가 되었습니다. `weights`의 두 번째 원소는 절편의 개수를 나타냅니다. 필터마다 1개의 절편이 있으므로 (32,) 크기가 됩니다.
- `weights` 속성은 텐서플로의 다차원 배열인 `Tensor` 클래스의 객체입니다. 여기에서는 다루기 쉽도록 `numpy()` 메서드를 사용해 넘파이 배열로 변환하겠습니다. 그다음 가중치 배열의 평균과 표준편차를 넘파이 `mean()` 메서드와 `std()`메서드로 계산해 보죠.

```python
conv_weights = conv.weights[0].numpy()

print(conv_weights.mean(), conv_weights.std())
```

```
-0.014383553 0.27351653
```

- 이 가중치의 평균 값은 0에 가깝고 표준 편차는 0.27 정도 입니다. 나중에 이 값을 훈련하기 전의 가중치와 비교해 보겠습니다. 이 가중치가 어떤 분포를 가졌는지 직관적으로 이해하기 쉽도록 히스토그램을 그려보겠습니다.

```python
import matplotlib.pyplot as plt
plt.hist(conv_weights.reshape(-1, 1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()
```

![스크린샷 2025-03-18 오후 1 52 26](https://github.com/user-attachments/assets/3e0d7689-fb0c-4597-9df8-1f6e72331f9b)


```python
fig, axs = plt.subplots(2, 16, figsize=(15,2))

for i in range(2):
    for j in range(16):
        axs[i, j].imshow(conv_weights[:,:,0,i*16 + j], vmin=-0.5, vmax=0.5)
        axs[i, j].axis('off')

plt.show()
```

```python
no_training_model = keras.Sequential()

no_training_model.add(keras.layers.Input(shape=(28,28,1)))
no_training_model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu',
                                          padding='same'))
```

```python
no_training_conv = no_training_model.layers[0]

print(no_training_conv.weights[0].shape)
```

```
(3, 3, 1, 32)
```

```python
no_training_weights = no_training_conv.weights[0].numpy()

print(no_training_weights.mean(), no_training_weights.std())
```

```
0.0053191613 0.08463709
```

```pytnon
plt.hist(no_training_weights.reshape(-1, 1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()
```

```python
fig, axs = plt.subplots(2, 16, figsize=(15,2))

for i in range(2):
    for j in range(16):
        axs[i, j].imshow(no_training_weights[:,:,0,i*16 + j], vmin=-0.5, vmax=0.5)
        axs[i, j].axis('off')

plt.show()
```

```python
print(model.inputs)
```

```
[<KerasTensor shape=(None, 28, 28, 1), dtype=float32, sparse=False, name=input_layer>]
```

```python
conv_acti = keras.Model(model.inputs, model.layers[0].output)
```

```python
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
```

```python
plt.imshow(train_input[0], cmap='gray_r')
plt.show()
```

```python
inputs = train_input[0:1].reshape(-1, 28, 28, 1)/255.0

feature_maps = conv_acti.predict(inputs)
```

```python
print(feature_maps.shape)
```

```
(1, 28, 28, 32)
```

```python
fig, axs = plt.subplots(4, 8, figsize=(15,8))

for i in range(4):
    for j in range(8):
        axs[i, j].imshow(feature_maps[0,:,:,i*8 + j])
        axs[i, j].axis('off')

plt.show()
```

```python
conv2_acti = keras.Model(model.inputs, model.layers[2].output)
```

```python
feature_maps = conv2_acti.predict(train_input[0:1].reshape(-1, 28, 28, 1)/255.0)
```

```python
print(feature_maps.shape)
```

```
(1, 14, 14, 64)
```

```python
fig, axs = plt.subplots(8, 8, figsize=(12,12))

for i in range(8):
    for j in range(8):
        axs[i, j].imshow(feature_maps[0,:,:,i*8 + j])
        axs[i, j].axis('off')

plt.show()
```
