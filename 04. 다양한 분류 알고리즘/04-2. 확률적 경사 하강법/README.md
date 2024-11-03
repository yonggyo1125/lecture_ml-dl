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
