# 인공 신경망

## 키워드 정리

- **인경 신경망**
  - 생물학적 뉴런에서 영감을 받아 만든 머신러닝 알고리즘입니다. 이름이 신경망 이만 실제 우리 뇌를 모델링한 것은 아닙니다.
  - 신경망은 기존의 머신러닝 알고리즘으로 다루기 어려웠던 이미지, 음성, 텍스트 분야에서 뛰어난 성능을 발휘하면서 크게 주목받고 있습니다.
  - 인공 신경망 알고리즘을 종종 딥러닝이라고 부릅니다.
- **텐서플로**
  - 구글이 만든 딥러닝 라이브러리로 매우 인기가 높습니다. **CPU**와 **GPU**를 사용해 인공 신경망 모델을 효율적으로 훈련하여 모델 구축과 서비스에 필요한 다양한 도구를 제공합니다.
  - 텐서플로 2.0부터는 신경망 모델을 빠르게 구성할 수 있는 케라스를 핵심 API로 채택하였습니다.
  - 케라스를 사용하면 간단한 모델에서 아주 복잡한 모델한 모델까지 손쉽게 만들 수 있습니다.
- **밀집층**
  - 가장 간단한 인공 신경망의 층입니다.
  - 인공 신경망에는 여러 종류의 층이 있습니다. 밀집층에서는 뉴런들이 모여 연결되어 있기 때문에 완전 연결 층이라고도 부릅니다.
  - 특별히 출력층에 밀집층을 사용할 때는 분류하려는 클래스오 동일한 개수의 뉴런을 사용합니다.
- **원-핫 인코딩**
  - 정수값을 배열에서 해당 정수의 위치의 원소만 1이고 나머지는 모두 0으로 변환합니다.
  - 이런 변환이 필요한 이유는 다중 분류에서 출력층에서 만든 확률과 크로스 엔트로피 손실을 계산하기 위해서 입니다. 텐서플로에서는 `sparse_categorical_entropy` 손실을 지정하면 이런 변환을 수행할 필요가 없습니다.

## 핵심 패키지와 함수

### TensorFlow

- **Dense**
  - 신경망에서 가장 기본 층인 밀집층을 만드는 클래스입니다.
  - 이 층에서 첫 번째 매개변수에는 뉴런의 개수를 지정합니다.
  - `activation` 매개변수에는 사용할 활성화 함수를 지정합니다. 대표적으로 `sigmoid`, `softmax` 함수가 있습니다. 아무것도 지정하지 않으면 활성화 함수를 사용하지 않습니다.
- **Sequential**
  - 케라스에서 신경망 모델을 만드는 클래스입니다.
  - 이 클래스의 객체를 생성할 때 신경망 모델에 추가할 층을 지정할 수 있습니다.
  - 추가할 층이 1개 이상일 경우 파이썬 리스트로 전달합니다.
- **compile()**
  - 모델 객체를 만든 후 훈련하기 전에 사용할 손실 함수와 측정 지표등을 지정하는 메서드입니다.
  - `loss` 매개변수에 손실함수를 지정합니다. 이진 분류일 경우 `binary_crossentropy`, 다중 분류일 경우 `categorical_crossentropy`로 지정합니다. 회귀 모델일 경우 `mean_square_error` 등으로 지정할 수 있습니다.
  - `metrics` 매개변수에 훈련 과정에서 측정하고 싶은 지표를 지정할 수 있습니다. 측정 지표가 1개 이상일 경우 리스트로 전달합니다.
- **fit()**
  - 모델을 훈련하는 메서드입니다.
  - 첫 번째와 두 번째 매개변수에 입력하는 타깃 데이터를 전달합니다.
  - `epochs` 매개변수에 전체 데이터에 대해 반복할 에포크 횟수를 지정합니다.
- **evaluate()**
  - 모델 성능을 평가하는 메서드입니다.
  - 첫 번째와 두 번째 매개변수에 입력과 타깃 데이터를 전달합니다.
  - `compile()` 메서드에서 `loss` 매개변수에 지정한 손실 함수의 값과 `metrics` 매개변수에 지정한 측정 지표를 출력합니다.

## 패션 MNIST

- 패션 MNIST 데이터셋을 사용하겠습니다.
- 이 데이터셋은 10종류의 패션 아이템으로 구성되어 있습니다.

> MNIST
>
> 머신러닝과 딥러닝을 처음 배울 때 많이 사용하는 데이터셋이 있습니다. 머신러닝에서는 붓꽃 데이터셋이 유명합니다. 딥러닝에서는 **MNIST 데이터셋**이 유명합니다. 이 데이터는 손으로 쓴 0\~9까지의 숫자로 이루어져 있습니다. **MNIST**와 크기, 개수가 동일하지만 숫자 대신 패션 아이템으로 이루어진 데이터가 바로 **패션 MNIST**입니다.

- 패션 MNIST 데이터는 워낙 유명하기 때문에 많은 딥러닝 라이브러리에서 이 데이터를 가져올 수 있는 도구를 제공합니다. 여기에서는 **텐서플로**(TensorFlow)를 사용해 이 데이터를 불러오겠습니다.

```python
from tensorflow import keras

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
```

- 다음 명령으로 텐서플로의 **케라스**(Keras) 패키지를 임포트하고 패션 MNIST 데이터를 다운로드 합니다.
- `keras.datasets.fashion_mnist` 모듈 아래 `load_data()` 함수는 친절하게 훈련 데이터와 테스트 데이터를 나누어 반환합니다. 이 데이터는 각각 입력과 타깃의 쌍으로 구성되어 있습니다.

```python
print(train_input.shape, train_target.shape)
```

- 전달받은 데이터의 크기를 확인해 보면

```
(60000, 28, 28) (60000,)
```

- 훈련 데이터는 60,000개의 이미지로 이루어져 있습니다. 각 이미지는 28 X 28 크기 입니다. 타깃도 60,000개의 원소가 있는 1차원 배열입니다.

<img width="522" alt="스크린샷 2024-11-15 오후 1 58 51" src="https://github.com/user-attachments/assets/9649d363-f8e3-46c5-85ee-0c8ae7d1eff3">

```python
print(test_input.shape, test_target.shape)
```

- 테스트 세트의 크기를 확인해 보면

```
(10000, 28, 28) (10000,)
```

- 테스트 세트는 10,000개의 이미지로 이루어져 있습니다.

```python
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 10, figsize=(10,10))
for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r')
    axs[i].axis('off')
plt.show()
```

- 훈련 데이터에서 몇 개의 샘플을 그림으로 출력해 보면 어떤 이미지인지 볼 수 있으므로 문제를 이해하는데 큰 도움이 됩니다.

<img width="542" alt="스크린샷 2024-11-15 오후 2 03 22" src="https://github.com/user-attachments/assets/a16d232e-8b3f-483c-9b90-9506dbc900c7">

- 반전된 흑백 이미지 입니다. 

```python
print([train_target[i] for i in range(10)])
```

- 파이썬의 리스트 내포를 사용해서 처음 10개 샘플 타깃값을 리스트로 만든 후 출력합니다.

```
[9, 0, 0, 3, 0, 2, 7, 2, 5, 5]
```

- 패션 MNIST의 타깃은 0\~9까지의 숫자 레이블로 구성됩니다. 각 숫자의 의미는 아직 모르지만 마지막 2개의 샘플이 같은 레이블 (숫자5)을 가지고 있습니다. 
- 앞서 출력한 이미지를 보면 이 2개의 샘플은 같은 종류의 신발 입니다. 
- 패션 MNIST에 포함된 10개 레이블의 의미는 다음과 같습니다.

<img width="616" alt="스크린샷 2024-11-15 오후 2 07 41" src="https://github.com/user-attachments/assets/98e2487e-de59-40a6-9268-0f749d47f7e7">


```python
import numpy as np

print(np.unique(train_target, return_counts=True))
```

- 넘파이 `unique()` 함수로 레이블 당 샘플의 개수를 확인합니다.

```
(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000]))
```

- 0\~9까지 레이블마다 정확히 6,000개의 샘플이 들어 있는 것을 볼 수 있습니다.

## 로지스틱 회귀로 패션 아이템 분류하기

- 이 훈련 샘플은 60,000개나 되기 때문에 전체 데이터를 한꺼번에 사용하여 모델을 훈련하는 것보다 샘플을 하나씩 꺼내서 모델을 훈련하는 방법이 더 효율적입니다.
- 이런 상황에 잘 맞는 방법은 경사 하강법입니다. - **SGDClassifier**

> 넘파이 배열의 `nbytes` 속성에 실제 해당 배열이 차지하는 바이트 용량이 저장되어 있습니다.

- **SGDClassifier** 클래스의 `log_loss`로 지정하여 로지스틱 손실 함수를 최소화하는 **확률적 경사 하강법 모델**을 만듭니다.
- **SGDClassifier**를 사용할 때 **표준화 전처리된 데이터**를 사용했습니다.
- 그 이유는 확률적 경사 하강법은 여러 특성 중 기울기가 가장 가파른 방향을 따라 이동합니다. 만약 특성마다 값의 범위가 많이 다르면 올바르게 손실 함수의 경사를 내려 올 수 없습니다. 패션 MNIST의 경우 각 픽셀은 0\~255 사이의 정숫값을 가집니다. 이런 이미지의 경우 보통 255로 나누어 0\~1 사이의 값으로 정규화합니다. 이는 표준화는 아니지만 양수 값으로 이루어진 이미지를 전처리할 때 널리 사용하는 방법입니다.

```python
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)
```

- `reshape()` 메서드를 사용해 2차원 배열인 각 샘플을 1차원 배열로 펼칩니다. 
- **SGDClassifier**는 2차원 입력을 다루지 못하기 때문에 각 샘플을 1차원 배열로 만들어야 합니다.
- `reshape()` 메서드의 두 번째 매개변수를 28 X 28 이미지 크기에 맞게 지정하면 첫 번째 차원(샘플 개수)은 변하지 않고 원본 데이터의 두 번째, 세 번째 차원이 1차원으로 합쳐집니다. 

```python
print(train_scaled.shape)
```

- 변환된 `train_scaled`의 크기를 확인해 봅니다.

```
(60000, 784)
```

- 기대한 대로 784개의 픽셀로 이루어진 60,000개의 샘플이 준비되었습니다. 


```python
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier

sc = SGDClassifier(loss='log_loss', max_iter=5, random_state=42)

scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
print(np.mean(scores['test_score']))
```

- **SGDClassifier** 클래스와 `cross_validate` 함수를 사용해 이 데이터에서 교차 검증으로 성능을 확인해 봅시다.


```
0.8196000000000001
```

- 여기에서는 **SGDClassifier**의 반복 횟수(max_iter)를 5번으로 지정했습니다. 반복 횟수를 늘려도 성능이 크게 향상되지는 않습니다.
- 직접 9나 20등의 여러 숫자를 넣어서 테스트해 봅시다.

```
0.8303833333333334  # max_iter=9
0.8436999999999999  # max_iter=20 
```

- 만족할 만한 수준은 아닙니다. 
- 잠시 로지스틱 회귀 공식을 떠올려 보면 


## 인공신경망

### 텐서플로와 케라스

```python
import tensorflow as tf
```

```python
from tensorflow import keras
```

## 인공신경망으로 모델 만들기

```python
from sklearn.model_selection import train_test_split

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)
```

```python
print(train_scaled.shape, train_target.shape)
```

```
(48000, 784) (48000,)
```

```python
print(val_scaled.shape, val_target.shape)
```

```
(12000, 784) (12000,)
```

```python
dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))
```

```python
model = keras.Sequential([dense])
```

## 인공신경망으로 패션 아이템 분류하기

```python
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

```python
print(train_target[:10])
```

```
[7 3 5 8 6 9 3 3 9 9]
```

```python
model.fit(train_scaled, train_target, epochs=5)
```

```python
model.evaluate(val_scaled, val_target)
```

```
[0.44444453716278076, 0.8458333611488342]
```
