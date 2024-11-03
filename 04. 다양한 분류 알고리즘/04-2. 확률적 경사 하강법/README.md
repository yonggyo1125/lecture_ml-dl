# 확률적 경사 하강법

## 키워드 정리

- **확률적 경사 하강법**
  - 훈련 세트에서 샘플 하나씩 꺼내 손실 함수의 경사를 따라 최적의 모델을 찾는 알고리즘입니다.
  - 샘플을 하나씩 사용하지 않고 여러 개를 사용하면 미니배치 경사 하강법이 됩니다.
  - 한 번에 전체 샘플을 사용하면 배치 경사 하강법이 됩니다.
- **손실 함수**
  - 확률적 경사 하강법이 최적화할 대상입니다.
  - 대부분의 문제에 잘 맞는 손실 함수가 이미 정의되어 있습니다.
  - 이진 분류에는 로지스틱 회귀(또는 이진 크로스엔트로피) 손실 함수를 사용합니다.
  - 다중 분류에는 크로스엔트로피 손실 함수를 사용합니다.
  - 회귀 문제에는 평균 제곱 오차 손실 함수를 사용합니다.
- **에포크**
  - 확률적 경사 하강법에서 전체 샘플을 모두 사용하는 한 번 반복을 의미합니다.
  - 일반적으로 경사 하강법 알고리즘은 수십에서 수백 번의 에포크를 반복합니다.

## 핵심 패키지와 함수

### scikit-learn

- **SGDClassifier**

  - 확률적 경사 하강법을 사용한 분류 모델을 만듭니다.
  - loss 매개변수는 확률적 경사 하강법으로 최적화할 손실 함수를 지정합니다. 기본값은 서포트 백터 머신을 위한 `hinge` 손실 함수입니다. 로지스틱 회귀를 위해서는 `log_loss`로 지정합니다.
  - `penalty` 매개변수에서 규제의 종류를 지정할 수 있습니다. 기본값은 L2 규제를 위한 'l2' 입니다. L1 규제를 적용하라면 'l1'으로 지정합니다. 규제 강도는 alpha 매개변수에서 지정합니다. 기본값은 0.0001 입니다.
  - `max_iter` 매개변수는 에포크 횟수를 지정합니다. 기본값은 1000입니다.
  - `tol` 매개변수는 반복을 멈출 조건입니다. `n_iter_no_charge` 매개변수에서 지정한 에포크 동안 손실이 tol 만큼 줄어들지 않으면 알고리즘이 중단됩니다. tol 매개변수의 기본값은 0.001 이고 `n_iter_no_charge` 매개변수의 기본값은 5입니다.

- **SGDRegressor**
  - 확률적 경사 하강법을 사용한 회귀모델을 만듭니다.
  - `loss` 매개변수에서 손실 함수를 지정합니다. 기본값은 제곱 오차를 나타내는 `squared_loss` 입니다.
  - 앞의 `SGDClassifier`에서 설명한 매개변수는 모두 `SGDRegressor`에서 동일하게 사용됩니다.

## 점진적인 학습

- 앞서 훈련한 모델을 버리지 않고 새로운 데이터에 대해서만 조금씩 더 훈련하는 방법
- 이런 식의 훈련방법을 **점진적 학습** 또는 온라인 학습이라고 부릅니다.
- 대표적인 학습 알고리즘은 **확률적 경사 하강법**(Stochastic Gradient Descent)입니다.

## 확률적 경사 하강법

- 확률적 경사 하강법에서 확률적이란 말은 **무작위하기** 혹은 **랜덤하게**의 기술적인 표현입니다.
- 경사 하강법은 경사를 따라 내려가는 방법을 말합니다.
- 가장 빠른 길은 경사가 가장 가파른 길이며, 경사 하강법이 바로 이런 방식입니다. 가장 가파른 경사를 따라 원하는 지점에 도달하는 것이 목표입니다.
- 가장 가파른 길을 찾아 내려오지만 조금씩 내려오는 것이 중요합니다. 이렇게 내려오는 과정이 바로 경사 하강법 모델을 훈련하는 것입니다.
- 확률적이라는 말을 이해해보려면 경사 하강법으로 내려올 때 가장 가파른 길을 찾는 방법이 무엇인지 생각해보는 것
- 훈련 세트의 전체 샘플을 사용하지 않고 딱 하나의 샘플을 훈련 세트에서 랜덤하게 골라 가장 가파른 길을 찾습니다. 이처럼 훈련 세트에서 랜덤하게 하나의 샘플을 고르는 것이 바로 **확률적 경사 하강법** 입니다.

> 확률적 경사 하강법은 훈련 세트에서 랜덤하게 하나의 샘플을 선택하여 가파른 경사를 조금 내려갑니다. 그다음 훈련 세트에서 랜덤하게 또 다른 샘플을 하나 선택하여 가파른 경사를 조금 내려갑니다. 그다음 훈련 세트에서 랜덤하게 또 다른 샘플을 하나 선택하여 경사를 조금 내려갑니다. 이런 식으로 모두 사용할 때까지 계속합니다.
>
> 모든 샘플을 다 사용했습니다. 그래도 산을 다 내려오지 못하였다면 다시 처음부터 시작하게 됩니다. 훈련 세트에 모든 샘플을 다시 채워 넣습니다. 그다음 다시 랜덤하게 하나의 샘플을 선택해 이어서 경사를 내려갑니다. 이렇게 만족할 만한 위치에 도달할 때까지 계속 내려가면 됩니다.
>
> 확률적 경사하강법에서 훈련 세트를 한 번 모두 사용하는 과정을 **에포크**(epoch)라고 부릅니다. 일반적으로 경사 하강법은 수십, 수백 번 이상 에포크를 수행합니다.

- **미니배치 경사 하강법**(minibatch gradient descent): 여러개의 샘플을 사용해 경사 하강법을 수행하는 방식
- **배치 경사 하강법**(batch gradient descent)
  - 한 번 경사로를 따라 이동하기 위해 전체 샘플을 사용하는 것
  - 전체 데이터를 사용하기 때문에 가장 안정적인 방법이 될 수 있습니다.
  - 하지만 전체 데이터를 사용하면 그만큼 컴퓨터의 자원을 많이 사용하게 됩니다.
  - 어떤 경우는 데이터가 너무너무 많아 한 번에 전체 데이터를 모두 읽을 수 없을지도 모릅니다.

![스크린샷 2024-11-03 오후 9 36 26](https://github.com/user-attachments/assets/bda256c2-60ea-4642-b0d7-4262cafe3291)

- 확률적 경사 하강법은 훈련 세트를 사용해 산 아래에 있는 최적의 장소로 조금씩 이동하는 알고리즘 입니다. 
- 이 때문에 훈련 데이터가 모두 준비되어 있지 않고 매일매일 업데이트되어도 학습을 계속 이어나갈 수 있습니다. 

> 확률적 경사 하강법을 꼭 사용하는 알고리즘이 있습니다. 바로 신경망 알고리즘 입니다. 신경망은 일반적으로 많은 데이터를 사용하기 때문에 한 번에 모든 데이터를 사용하기 어렵습니다. 또 모델이 매우 복잡하기 때문에 수학적인 방법으로 해답을 얻기 어렵습니다. 신경망 모델이 확률적 경사하강법이나 미니배치 경사 하강법을 사용합니다. 

- 가장 빠른 길을 찾아 내려가려고 하는 산 -> 손실함수

## 손실 함수
- **손실함수**(loss function)는 어떤 문제에서 머신러닝 알고리즘이 얼마나 엉터리인지를 측정하는 기준입니다. 
- 손실 함수의 값이 작을 수록 좋으나 어떤 값이 최솟값인지는 알지 못합니다. 가능한 많이 찾아보고 만족할만한 수준이라면 산을 다 내려왔다고 인정해야 합니다.
- 이 값을 찾아서 조금씩 이동하려면 확률적 경사 하강법이 잘 맞습니다. 
- 다행히도 우리가 다루는 많은 문제에 필요한 손실 함수는 이미 정의되어 있습니다. 

- 분류에서 손실은 정답을 못 맞히는 것 입니다. 
- 도미와 빙어를 구분하는 이진 문제를 예로 들면 도미는 양성 클래스(1), 빙어는 음성 클래스(0)라고 가정해 봅시다.

![스크린샷 2024-11-04 오전 6 02 32](https://github.com/user-attachments/assets/87cabd9c-a73f-4add-9554-ee6f9966d7a3)

- 정확도는 4개의 예측 중에 2개만 맞았으므로 정확도는 1/2 = 0.5 입니다. 
- 정확도를 손실함수로 사용할 수 있습니다. 정확도에 음수를 취하면 -1.0이 가장 낮고, -0.0이 가낭 높습니다. 
- 하지만 정확도는 치명적인 단점이 있습니다. 예를 들어 앞의 그림과 같이 4개의 샘플만 있다면 가능한 정확도는 0, 0.25, 0.5, 0.75, 1 다섯 가지 뿐입니다. 
- 정확도가 이렇게 듬성듬성하다면 경사 하강법을 이용해 조금씩 움질일 수 없습니다. 산의 경사면은 확실히 연속적이어야 합니다. 

![스크린샷 2024-11-04 오전 6 07 22](https://github.com/user-attachments/assets/6c426715-1a18-4fa4-9be4-bc3734f7e97c)



## SGDClassifier

```python
import pandas as pd

fish = pd.read_csv('https://bit.ly/fish_csv_data')
```

```python
fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
fish_target = fish['Species'].to_numpy()
```

```python
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42)
```

```python
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
```

```python
from sklearn.linear_model import SGDClassifier
```

```python
sc = SGDClassifier(loss='log_loss', max_iter=10, random_state=42)
sc.fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
```

```python
sc.partial_fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
```

```
0.8151260504201681
0.85
```

## 에포크와 과대/과소접합

```python
import numpy as np

sc = SGDClassifier(loss='log_loss', random_state=42)

train_score = []
test_score = []

classes = np.unique(train_target)
```

```python
for _ in range(0, 300):
    sc.partial_fit(train_scaled, train_target, classes=classes)

    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))
```

```python
import matplotlib.pyplot as plt

plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
```

```python
sc = SGDClassifier(loss='log_loss', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
```

```
0.957983193277311
0.925
```

```python
sc = SGDClassifier(loss='hinge', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
```

```
0.9495798319327731
0.925
```
