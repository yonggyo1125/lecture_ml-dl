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

<img width="577" alt="스크린샷 2024-11-17 오후 12 34 58" src="https://github.com/user-attachments/assets/2756508d-88ce-4df6-b882-2390948c44ed">

- 이전 절에서 만든 신경망 모델과 다른 점은 입력층과 출력층 사이에 밀집층이 추가된 것입니다. 이렇게 입력층과 출력층 사이에 있는 모든 층을 **은닉층**(hidden layers)이라고 부릅니다.
- 은닉층에는 주황색 원으로 활성화 함수가 표시되어 있습니다. 활성화 함수는 신경망 층의 선형 방정식의 계산 값에 적용하는 함수입니다. 이전 절에서 출력층에 적용했던 소프트맥스 함수도 활성화 함수입니다. 출력층에 적용하는 활성화 함수는 종류가 제한되어 있습니다. 이진 분류일 경우 시그모이드 함수를 사용하고 다중 분류일 경우 소프트맥스 함수를 사용합니다. 이에 비해 은닉층의 함수는 비교적 자유롭습니다. 대표적으로 시그모이드 함수와 렐루(ReLU)함수 등을 사용합니다.

> 분류 문제는 클래스에 대한 확률을 출력하기 위해 활성화 함수를 사용합니다. 회귀의 출력은 임의의 어떤 숫자이므로 활성화 함수를 적용할 필요가 없습니다. 즉 출력층의 선형 방정식의 계산을 그대로 출력합니다. 이렇게 하려면 Dense 층의 `activation` 매개변수에 아무런 값을 지정하지 않습니다.

- 그런데 은닉층에 왜 활성화 함수를 적용할까요? 다음 그림에 있는 2개의 선형 방정식을 생각해 보면, 왼쪽의 첫 번쨰 식에서 계산된 b가 두 번째 식에서 c를 계싼하기 위해 쓰입니다. 하지만 두 번째 식에 첫 번째 식을 대입하면 오른쪽처럼 하나로 합쳐질 수 있습니다. 이렇게 되면 b는 사라집니다. b가 하는 일이 없는 셈입니다.

<img width="366" alt="스크린샷 2024-11-17 오후 12 45 11" src="https://github.com/user-attachments/assets/c3af2153-4497-4074-b475-048a6cd13224">

- 신경망도 마찬가지 입니다. 은닉층에서 선형적인 산술 계산만 수행한다면 수행 역할이 없는 셈입니다. 선형 계산을 적당하게 비선형적으로 비틀어 주어야 합니다. 그래야 다음 층의 계산과 단순히 합쳐지지 않고 나름의 역할을 할 수 있습니다. 마치 다음과 같습니다.

<img width="168" alt="스크린샷 2024-11-17 오후 12 45 20" src="https://github.com/user-attachments/assets/22f2b05a-2d1a-4143-817d-2bd85ac5c6ee">

> 인공 신경망을 그림으로 나타낼 때 활성화 함수를 생략하는 경우가 많은데 이는 절편과 마찬가지로 번거로움을 피하기 위해 활성화 함수를 별개의 층으로 생각하지 않고 층에 포함되어 있다고 간주하기 떄문입니다. 그림에서 보이지는 않지만 모든 신경망의 은닉층에는 항상 활성화 함수가 있습니다.

- 많이 사용하는 활성화 함수 중 하나는 앞서 배웠던 시그모이드 함수입니다. 다시 한번 살펴보면

<img width="466" alt="스크린샷 2024-11-17 오후 12 56 17" src="https://github.com/user-attachments/assets/8e8e528a-4eea-47a0-92a8-3837f5682987">

- 이 함수는 뉴런의 출력 z값을 0과 1사이로 압축합니다.
- 그럼 시그모이드 활성화 함수를 사용한 은닉층과 소프트맥스 함수를 사용한 출력층을 케라스의 Dense 클래스로 만들어 보겠습니다.
- 이전 절에서 언급했듯이 케라스에서 신경망의 첫 번쨰 층은 `input_shape` 매개변수로 입력의 크기를 꼭 지정해 주어야 합니다.

```python
dense1 = keras.layers.Dense(100, activation='sigmoid', input_shape=(784,))
dense2 = keras.layers.Dense(10, activation='softmax')
```

- dense1이 은닉층이고 100개의 뉴런을 가진 밀집층입니다. 활성화 함수를 `sigmoid`로 지정했고 `input_shape` 매개변수에서 입력의 크기를 (784,)로 지정했습니다. 은닉층의 뉴런 개수를 정하는데는 특별한 기준이 없습니다. 몇 개의 뉴런을 두어야 할지 판단하기 위해서는 상당한 경험이 필요합니다.
- 여기에서 한가지 제약 사항이 있다면 적어도 출력층의 뉴런보다는 많게 만들어야 합니다. 클래스 10개에 대한 확률을 예측해야 하는데, 이전 은닉층의 뉴런이 10개보다 부족하면 정보가 전달될 것입니다.
- 그 다음 dense2는 출력층입니다. 10개의 클래스를 분류하므로 10개의 뉴런을 두었고 활성화 함수는 소프트맥스 함수로 지정했습니다.

## 심층 신경망 만들기

- 이제 앞에서 만든 dense1과 dense2 객체를 Sequential 클래스에 추가하여 **심층 신경망**(deep neural network, DNN)을 만들어 보겠습니다.

```python
model = keras.Sequential([dense1, dense2])
```

- **Sequential** 클래스의 객체를 만들 때 여러 개의 층을 추가하려면 이와 같이 `dense1`과 `dense2`를 리스트로 만들어 전달합니다. 여기에서 주의할 것은 출력층을 가장 마지막에 두어야 한다는 것입니다. 이 리스트는 가장 처음 등장하는 은닉층에서 마지막 출력층의 순서대로 나열해야 합니다. 


![스크린샷 2024-11-17 오후 3 03 48](https://github.com/user-attachments/assets/229d43d6-c768-40f3-98de-f46d3c1eba51)


- 인공 신경망의 강력한 성능은 바로 이렇게 층을 추가하여 입력 데이터에 대해 연속적인 학습을 진행하는 능력에서 나옵니다. 앞서 배운 선형 회귀, 로지스틱 회귀, 결정 트리 등 다른 머신 러닝 알고리즘들과 대조됩니다. 물론 2개 이상의 층을 추가할 수도 있습니다. 
- 케라스는 모델의 `summary()` 메서드를 호출하면 층에 대한 유용한 정보를 얻을 수 있습니다.

```python
model.summary()
```

![스크린샷 2024-11-17 오후 3 22 03](https://github.com/user-attachments/assets/39405f6a-0f12-434b-9387-01048103cb2a)

- 맨 첫 줄에 모델의 이름이 나옵니다. 그 다음 이 모델이 들어 있는 층이 순서대로 나열됩니다. 이 순서는 맨 처음 추가한 은닉층에서 출력층의 순서대로 나열됩니다.
- 층 마다 층 이름, 클래스, 출력 크기, 모델 파라미터 개수가 출력 됩니다. 층을 만들 때 name 매개변수로 이름을 지정할 수 있습니다. 층 이름을 지정하지 않으면 케라스가 자동으로 `dense`라고 이름을 붙입니다. 
- 출력의 크기를 보면 (None, 100)입니다. 첫 번쨰 차원은 샘플의 개수를 나타냅니다. 샘플 개수가 아직 정의되어 있지 않기 때문에 None 입니다. 이유는 케라스 모델의 `fit()` 메서드에 훈련 데이터를 주입 하면 이 데이터를 한 번에 모두 사용하지 않고 잘게 나누어 여러 번에 걸쳐 경사 하강법 단계를 수행합니다. 즉 미니배치 경사 하강법을 사용하는 것 
- 케라스의 기본 미니배치 크기는 32개입니다. 이 값은 `fit()` 메서드에서 `batch_size` 매개변수로 바꿀 수 있습니다. 따라서 샘플 개수를 고정하지 않고 어떤 배치 크기에도 유연하게 대응할 수 있도록 `None` 으로 설정합니다. 이렇게 신경망 층에 입력되거나 출력되는 배열의 첫 번째 차원을 배치 차원이라고 부릅니다. 
- 두 번째 100은 다음과 같습니다. 은닉층의 뉴런 개수를 100개로 두었으니 100개의 출력이 나옵니다. 즉 샘플마다 784개의 픽셀값이 은닉층을 통과하면서 100개의 특성으로 압축되었습니다.
- 마지막으로 모델 파라미터 개수가 출력됩니다. 이 층은 Dense 층이므로 입력 픽셀 784개와 100개의 모든 조합에 대한 가중치가 있습니다. 그리고 뉴런마다 1개의 절편이 있습니다.


![스크린샷 2024-11-17 오후 3 35 38](https://github.com/user-attachments/assets/78ca27ae-7265-4b54-b5ac-2e1eb9623387)

- 두 번째 층의 출력 크기는 (None, 10)입니다. 배치 차원은 동일하게 None이고 출력 뉴런 개수가 10개이기 때문입니다. 이 층의 모델 파라미터 개수는 몇 개일까요?

![스크린샷 2024-11-17 오후 3 38 18](https://github.com/user-attachments/assets/7d83c227-c2d0-4c0b-b97c-2e893eeeafff)

- 100개의 은닉층 뉴런과 10개의 출력층 뉴런이 모두 연결되고 출력층의 뉴런마다 하나의 절편이 있기 때문에 1,010개의 모델 파라미터가 있습니다.
- `summary()` 메서드의 마지막에는 총 모델 파라미터 개수와 훈련되는 파라미터 개수가 동일하게 79,510개로 나옵니다. 은닉층과 출력층의 파라미터 개수를 합친 값입니다. 그 아래 훈련되지 않는 파라미터(Non-trainable params)는 0으로 나옵니다. 간혹 경사 하강법으로 훈련되지 않는 파라미터를 가진 층이 있습니다. 이런 층의 파라미터 개수가 여기에 나타납니다.

## 층을 추가하는 다른 방법

- 모델을 훈련하기 전에 **Sequential** 클래스에 층을 추가하는 다른 방법을 알아보겠습니다. 앞에서는 **Dense** 클래스의 객체 `dense1`, `dense2`를 만들어 **Sequential** 클래스에 전달했습니다. 이 두 객체를 따로 저장하여 쓸 일이 없기 때문에 다음 처럼 **Sequential** 클래스의 생성자 안에서 바로 **Dense** 클래스의 객체를 만드는 경우가 많습니다. 

```python
model = keras.Sequential([
    keras.layers.Dense(100, activation='sigmoid', input_shape=(784,), name='hidden'),
    keras.layers.Dense(10, activation='softmax', name='output')
], name='패션 MNIST 모델')
```

- 이렇게 작업하면 추가되는 층을 한눈에 쉽게 알아보는 장점이 있습니다. 이전과 달리 이번에는 **Sequential** 클래스의 `name` 매개변수로 모델의 이름을 지정했습니다. 또 **Dense** 층의 `name` 매개변수에 층의 이름을 `hidden`과 `output` 으로 각각 지정했습니다. 모델의 이름과 달리 층의 이름은 반드시 영문이어야 합니다. `summary()` 메서드의 출력에 이름이 잘반영되는지 확인해 봅니다.

```python
model.summary()
```

![스크린샷 2024-11-17 오후 3 48 20](https://github.com/user-attachments/assets/12fbceb2-1e83-4566-aa3d-1a3e0fef66a5)


![스크린샷 2024-11-17 오후 3 48 29](https://github.com/user-attachments/assets/cccbe5ee-914c-43db-82dc-b44cea27aad8)


- 2개의 Dense 층이 이전과 동일하게 추가되었고 파라미터 개수도 같습니다. 바뀐 것은 모델 이름과 층 이름입니다. 여러 모델과 많은 층을 사용할 때 name 매개변수를 사용하면 구분하기 쉽습니다.
- 이 방법이 편리하지만 아주 많은 층을 추가하려면 **Sequential** 클래스 생성자가 매우 길어집니다. 또 이 조건에 따라 층을 추가할 수도 없습니다. **Sequential** 클래스에서 층을 추가할 때 가장 널리 사용하는 방법은 모델의 `add()` 메서드 입니다.
- 이 방법은 다음처럼 **Sequential** 클래스의 객체를 만들과 이 객체의 add() 메서드를 호출하여 층을 추가합니다. 

```python
model = keras.Sequential()
model.add(keras.layers.Dense(100, activation='sigmoid', input_shape=(784,)))
model.add(keras.layers.Dense(10, activation='softmax'))
```

- 여기에서도 **Dense** 클래스의 객체를 따로 변수에 담지 않고 바로 `add()` 메서드로 전달합니다. 이 방법은 한눈에 추가되는 층을 볼 수 있고 프로그램 실행 시 동적으로 층을 선택하여 추가할 수 있습니다. 
- `summary()` 메서드의 결과에서 층과 파라미터 개수는 당연히 동일합니다.

```python
model.summary()
```

- 이제 모델을 훈련해 보겠습니다. `compile()` 메서드의 설정은 앞서 했던 것과 동일합니다. 여기에서도 5번의 에포크 동안 훈련해 봅시다.

```python
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_scaled, train_target, epochs=5)
```

![스크린샷 2024-11-17 오후 3 58 25](https://github.com/user-attachments/assets/c1e779c5-90b5-4842-be9a-e65e541028cc)

- 훈련 세트에 대한 성능을 보면 추가된 층이 성능을 향상시켰다는 것을 잘 알 수 있습니다. 인공 신경망에 몇 개의 층을 추가하더라도 `compile()` 메서드와 `fit()` 메서드의 사용법은 동일합니다. 이것이 케라스 API의 장점입니다. 필요하면 여러 개의 층을 추가하고 실험해 봅니다.  


## 렐루 활성화 함수

- 초창기 인공 신경망의 은닉층에 많이 사용된 활성화 함수는 시그모이드 함수였습니다. 하지만 이 함수에는 단점이 있습니다. 이 함수의 오른쪽과 왼쪽 끝으로 갈수록 그래프가 누워있기 때문에 올바른 출력을 만드는데 신속하게 대응하지 못합니다. 
- 다음 그림을 참고하세요.

![스크린샷 2024-11-17 오후 4 05 37](https://github.com/user-attachments/assets/76381746-753c-4ec6-90e3-b582b7eb3313)

- 특히 층이 많은 심층 신경망일수록 그 효과가 누적되어 학습을 더 어렵게 만듭니다. 이를 개선하기 위해 다른 종류의 활성화 함수가 제안되었습니다. 바로 **렐루**(ReLU) 함수 입니다. 렐루 함수는 아주 간단합니다. 입력이 양수일 경우 마치 활성화 함수가 없는 것처럼 그냥 입력을 통과시키고 음수일 경우에는 0으로 만듭니다. 다음 그림을 참고하세요.


![스크린샷 2024-11-17 오후 4 07 42](https://github.com/user-attachments/assets/a2e222fd-0ff8-4d8c-b582-6c40f2c0ee1f)

- 렐루 함수 `max(0, z)` 와 같이 쓸 수 있습니다. 이 함수는 z가 0보다 크면 z를 출력하고 z가 0보다 작으면 0을 출력합니다. 
- 렐루 함수는 특히 이미지 처리에서 좋은 성능을 낸다고 알려져 있습니다. 은닉층의 활성화 함수에 시그모이드 함수 대신 렐루 함수를 적용하기 전에 케라스에서 제공하는 편리한 층 하나를 더 살펴보겠습니다. 
- 패션 MNIST 데이터는 28 X 28 크기이기 때문에 인공 신경망에 주입하기 위해 넘파이 배열의 `reshape()` 메서드를 사용해 1차원으로 펼쳤습니다. 직접 이렇게 1차원으로 펼쳐도 좋지만 케라스에서는 이를 위한 **Flattern** 층을 제공합니다. 
- 사실 **Flatten** 클래스는 배치 차원을 제외하고 나머지 입력 차원을 모두 일렬로 펼치는 역할만 합니다. 입력에 곱해지는 가중치나 절편이 없습니다. 따라서 인공 신경망의 성능을 이해 기여하는 바는 없습니다. 
- 하지만 **Flatten** 클래스를 층처럼 입력층과 은닉층 사이에 추가하기 때문에 이를 층이라고 부릅니다. **Flatten** 층은 다음 코드처럼 입력층 바로 뒤에 추가합니다.


```python
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
```

- 첫 번째 **Dense** 층에 있던 `input_shape` 매개변수를 **Flatten** 층으로 옮겼습니다. 또 첫 번째 **Dense** 층의 활성화 함수를 `relu`로 바꾼 것을 확인하세요. 하지만 이 신경망을 깊이가 3인 신경망이라고 부르지는 않습니다. **Flatten** 클래스는 학습하는 층이 아니기 때문입니다. 
- 모델의 `summary()` 메서드를 호출해 보면 이런 점을 더 확실히 알 수 있습니다. 

```python
model.summary()
```

![스크린샷 2024-11-17 오후 4 17 36](https://github.com/user-attachments/assets/ec9de3df-5574-43a2-9d8c-f92d183859e9)

- 첫 번째 등장하는 **Flatten** 클래스에 포함된 모델 파라미터는 0개입니다. 케라스의 **Flatten** 층을 신경망 모델에 추가하면 입력값의 차원을 짐작할 수 있는 것이 또 하나의 장점입니다. 
- 앞의 출력에서 784개의 입력이 첫 번째 은닉층에 전달되는 것을 알 수 있는데, 이는 이전에 만들었던 모델에서는 쉽게 눈치채기 어려워습니다. 입력 데이터에 대한 전처리 과정을 가능한 모델에 포함시키는 것이 케라스 API의 철학 중 하나입니다. 
- 그럼 훈련 데이터르 다시 준비해서 모델을 훈련해 보겠습니다. 이전 절의 서두에 있던 코드와 동일하지만 `reshape()` 메서드를 적용하지 않았습니다.


```python
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

train_scaled = train_input / 255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)
```

- 모델을 컴파일하고 훈련하는 것은 다음 코드처럼 이전과 동일합니다.

```python
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_scaled, train_target, epochs=5)
```

![스크린샷 2024-11-17 오후 4 24 03](https://github.com/user-attachments/assets/3bdbd57f-436b-4057-a9da-978bbccb77c1)

![스크린샷 2024-11-17 오후 4 24 13](https://github.com/user-attachments/assets/7e49f70d-0a2d-463c-8a80-4a4541ea81f8)


- 시그모이드 함수를 사용했을 때와 비교하면 성능이 조금 향상되었습니다. 크지 않지만 렐루 함수의 효과를 보았습니다. 
- 검증 세트에서의 성능도 확인해 보겠습니다.


```python
model.evaluate(val_scaled, val_target)
```

![스크린샷 2024-11-17 오후 4 48 28](https://github.com/user-attachments/assets/266e32cd-ccaf-4ebd-89eb-3f37b856dd67)

- 이전 절의 은닉층을 추가하지 않은 경우보다 몇 퍼센트 성능이 향상되었습니다. 지금까지는 모델을 5번의 에포크 동안 훈련했습니다. 이보다 더 훈련하지 않을 이유가 없습니다. 
- 그전에 인공 신경망의 하이퍼파라미터에 대해 잠시 알아보겠습니다.

## 옵티마이저

- 하이퍼파라미터는 모델이 학습하지 않아 사람이 지정해 주어야 하는 파라미터라고 설명했습니다. 신경망에서는 특히 하이퍼파라미터가 많습니다. 
- 이번 절에서는 은닉층을 하나 추가했습니다. 하지만 여러 개의 은닉층을 추가할 수도 있습니다. 추가할 은닉층의 개수는 모델이 학습하는 것이 아니라 우리가 지정해 주어야 할 하이퍼파라미터입니다. 그럼 은닉층의 뉴런 개수도 하이퍼파라미터일까요? 맞습니다. 또 활성화 함수도 선택해야 할 하이퍼파라미터 중 하나입니다. 심지어 층의 종류도 하이퍼파라미터입니다. 이번 장에서는 가장 기본적인 밀집층만 다루지만 다른 종류의 층을 선택할 수도 있습니다. 
- 케라스는 기본적으로 미니배치 경사 하강법을 사용하여 미니배치 개수는 32개입니다. `fit()` 메서드의 `batch_size` 매개변수에서 이를 조정할 수 있으며 역시 하이퍼파라미터입니다. 또한 `fit()` 메서드의 `epochs` 매개변수도 하이퍼파라미터입니다. 반복 횟수에 따라 다른 모델이 만들어집니다.
- 마지막으로 `compile()` 메서드에서는 케라스의 기본 경사 하강법 알고리즈인 `RMSprop` 을 사용했습니다. 케라스는 다양한 종류의 경사 하강법 알고리즘을 제공합니다. 이들을 **옵티마이저**(optimizer)라고 부릅니다. 
- **RMSprop**의 학습률 또한 조정할 하이퍼파라미터 중 하나입니다.
- 처음부터 모델을 구성하고 각종 하이퍼파라미터의 최적값을 찾는 것은 어려운 작업입니다. 여기서는 여러 가지 옵티마이저를 테스트해 보겠습니다. 
- 가장 기본적인 옵티마이저는 확률적 경사 하강법인 **SGD**입니다. 이름이 **SGD** 이지만 1개의 샘플을 뽑아서 훈련하지 않고 앞서 언급한 것처럼 기본적으로 미니배치를 사용합니다.

- **SGD** 옵티마이저를 사용하려면 `compile()` 메서드의 optimizer 매개변수를 `sgd`로 지정합니다. 

```python
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

- 이 옵티마이저는 `tensorflow.keras.optimizers` 패키지 아래 `SGD` 클래스로 구현되어 있습니다. `sgd` 문자열은 이 클래스의 기본 설정 매개변수로 생성한 객체와 동일합니다. 즉 다음 코드는 위의 코드와 정확히 동일합니다.

```python
sgd = keras.optimizers.SGD()
model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

- 만약 SGD 클래스의 학습률 기본값이 0.01 일 때 이를 바꾸고 싶다면 다음과 같이 원하는 학습률을 `learning_rate` 매개변수에 지정하여 사용합니다.

```python
sgd = keras.optimizers.SGD(learning_rate=0.1)
```

- SGD 외에도 다양한 옵티마이저들이 있습니다. 




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
