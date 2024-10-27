# 데이터 전처리

## 키워드 정리

- **데이터 전처리**
  - 머신러닝 모델 훈련 데이터를 주입하기 전에 가공하는 단계를 말합니다.
  - 때로는 데이터 전처리에 많은 시간이 소모되고 합니다.
- **표준점수**

  - 훈련 세트의 스케일을 바꾸는 대표적인 방법 중 하나입니다.
  - 표준점수는 특성의 평균을 빼고 표준편차로 나눕니다.
  - 반드시 훈련 세트의 평균과 표준편차로 테스트 세트를 바꿔야 합니다.

- **브로드캐스팅**
  - 크기가 다른 넘파이 배열에서 자동으로 사칙 연산을 모든 행이나 열로 확장하여 수행하는 기능입니다.

## 핵심 패키지와 함수

### scikit-learn

- **train_test_split()**
  - 훈련 데이터를 훈련 세트와 테스트 세트로 나누는 함수입니다.
  - 테스트 세트로 나눌 비율은 test_size 매개변수에서 지정할 수 있으며, 기본값은 0.25(25%)입니다.
  - `shuffle` 매개변수로 훈련 세트와 테스트 세트로 나누기 전에 무작위로 섞을지 여부를 결정할 수 있습니다. 기본값은 True
  - `stratify` 매개변수에 클래스 레이블이 담긴 배열(일반적으로 타깃 데이터)을 전달하면 클래스 비율에 맞게 훈련 세트와 테스트 세트를 나눕니다.
- **kneighbors()**
  - k-최근접 이웃 객체의 메서드입니다.
  - 입력한 데이터에 가장 가까운 이웃을 찾아 거리와 이웃 샘플의 인덱스를 반환합니다.
  - 기본적으로 이웃의 개수는 `KNeighborsClassifier` 클래스의 객체를 생성할 때 지정한 개수를 사용합니다.
  - `n_neighbors`매개변수에서 다르게 지정할 수도 있습니다.
  - `return_distance` 매개변수를 False로 지정하면 이웃 샘플의 인덱스만 반환하고 거리는 반환하지 않습니다. 기본값은 True

## 넘파이로 데이터 준비하기

```python
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
```

- 넘파이를 사용하여 길이와 무게를 2차원 배열로 만들어 봅시다.

```python
import numpy as np
```

- 넘파이를 임포트 합니다.

```python
np.column_stack(([1,2,3], [4,5,6]))
```

- `column_stack()` 함수는 전달받은 리스트를 일렬로 세운 다음 차례대로 나란히 연결합니다.
- 연결할 리스트는 파이썬 튜플(tuple)로 전달합니다.

```
array([[1, 4],
       [2, 5],
       [3, 6]])
```

- \[1,2,3\]과 \[4,5,6\] 두 리스트를 일렬로 세운 다움 나란히 옆으로 붙였습니다.
- 만들어진 배열은 (3, 2) 크기의 배열입니다. 즉, 3개의 행과 2개읠 열

```python
fish_data = np.column_stack((fish_length, fish_weight))
```

- fish_length와 fish_weight를 합칩니다.

```python
print(fish_data[:5])
```

- 처음 5개의 데이터 확인

```
[[ 25.4 242. ]
 [ 26.3 290. ]
 [ 26.5 340. ]
 [ 29.  363. ]
 [ 29.  430. ]]
```

- 5개의 행을 출력했고, 행마다 2개의 열(생선의 길이와 무게)이 있음을 알 수 있습니다.

```python
print(np.ones(5))
```

- 원소가 하나인 리스트 \[1\], \[0\]을 여러번 곱해서 데이터를 만드는데, `np.ones()`와 `np.zeros()` 함수를 이용할 수도 있습니다.

```
[1. 1. 1. 1. 1.]
```

```python
fish_target = np.concatenate((np.ones(35), np.zeros(14)))
```

- 위 두 함수를 사용해 1이 35개인 배열과 0이 14개인 배열을 만듭니다.
- 그 다음 두 배열을 그대로 연결합니다.
- 이때는 `np.column_stack()` 함수를 사용하여 첫 번째 차원을 따라 배열을 연결합니다.
- 연결할 리스트나 배열을 튜플로 전달해야 합니다.

```python
print(fish_target)
```

- 만들어져 있는 데이터 확인

```python
[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0.]
```

## 사이킷런으로 훈련 세트와 테스트 세트 나누기

- 넘파이 배열의 인덱스를 직접 섞어서 훈련 세트와 테스트 세트로 나누는 방법은 번거롭습니다.
- 사이킷런의 `train_test_split()`함수를 이용하면 좀더 세련되게 나눌 수 있습니다.
- 이 함수는 전달되는 리스트나 배열을 비율에 맞게 섞어서 `훈련 세트`와 `테스트 세트`로 나누어 줍니다.
- `train_test_split()` 함수는 사이킷런의 `model_selection` 모듈 아래 있습니다.

```python
from sklearn.model_selection import train_test_split
```

- 위와 같이 임포트 합니다.

```python
train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target, random_state=42)
```

- 훈련 세트와 테스트 세트를 나눕니다.
- fish_data와 fish_target 2개의 배열을 전달했으므로 2개씩 나뉘어 총 4개의 배열이 반환됩니다.
- (train_input, test_input) : 입력 데이터
- (train_target, test_target): 타깃 데이터
- 이 함수는 기본적으로 25%를 테스트 세트로 떼어 냅니다.

> `random_state` : np.random.seed() 함수와 같이 출력 결과와 강의 자료 내용과 같아지도록 만들기 위해서 직접 값을 지정할 수 있음

```
print(train_input.shape, test_input.shape)
```

- 훈련 세트와 테스트 세트가 잘 나뉘었는지 체크
- 넘파이 배열의 shape 속성으로 입력 데이터 크기 출력

```
(36, 2) (13, 2)
```

```python
print(train_target.shape, test_target.shape)
```

```
(36,) (13,)
```

- 훈련 데이터와 테스트 데이터를 각각 36, 13개로 나누었음
- 입력 데이터는 2개의 열이 있는 2차원 배열
- 타깃 데이터는 1차원 배열

```python
print(test_target)
```

- 도미와 방어가 잘 섞였는지 테스트 데이터 출력

```
[1. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
```

- 잘 섞인 것 같지만 빙어의 비율이 조금 모자랍니다. 
- 샘플링 편향이 여기에서도 나타났습니다.
- 특히 일부 클래스의 갯수가 적을때 무작위로 데이터를 나누었을 때 샘플이 골고루 섞이지 않을 수 있습니다. 
- 휸련 세트와 테스트 세트에서 샘플의 클래스 비율이 일정하지 않다면 모델이 일부 샘플을 올바르게 학습할 수 없습니다. 

```python
train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target, stratify=fish_target, random_state=42)
```

- `train_test_split()` 함수의 `stratify` 매개변수에 타깃 데이터를 전달하면 클래스 비율에 맞게 데이터를 나눕니다.
- 훈련 데이터가 작거나 특정 클래스의 샘플 갯수가 적을 때 유용

```python
print(test_target)
```

```
[0. 0. 1. 0. 1. 0. 1. 1. 1. 1. 1. 1. 1.]
```



## 수상한 도미 한마리

```python
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
kn.score(test_input, test_target)
```

```
1.0
```

```python
print(kn.predict([[25, 150]]))
```

```python
import matplotlib.pyplot as plt
```

```python
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

```python
distances, indexes = kn.kneighbors([[25, 150]])
```

```python
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

```python
print(train_input[indexes])
```

```
[[[ 25.4 242. ]
  [ 15.   19.9]
  [ 14.3  19.7]
  [ 13.   12.2]
  [ 12.2  12.2]]]
```

```python
print(train_target[indexes])
```

```
[[1. 0. 0. 0. 0.]]
```

```python
print(distances)
```

```
[[ 92.00086956 130.48375378 130.73859415 138.32150953 138.39320793]]
```

## 기준을 맞춰라

```python
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
plt.xlim((0, 1000))
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

```python
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)
```

```python
print(mean, std)
```

```
[ 27.29722222 454.09722222] [  9.98244253 323.29893931]
```

```python
train_scaled = (train_input - mean) / std
```

## 전처리 데이터로 모델 훈련하기

```python
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(25, 150, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

```python
new = ([25, 150] - mean) / std
```

```python
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

```python
kn.fit(train_scaled, train_target)
```

```python
test_scaled = (test_input - mean) / std
```

```python
kn.score(test_scaled, test_target)
```

```
1.0
```

```python
print(kn.predict([new]))
```

```
[1.]
```

```python
distances, indexes = kn.kneighbors([new])
```

```python
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.scatter(train_scaled[indexes,0], train_scaled[indexes,1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```
