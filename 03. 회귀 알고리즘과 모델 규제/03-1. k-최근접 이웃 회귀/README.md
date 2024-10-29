# k-최근접 이웃 회기

## 키워드 정리

- **회귀** : 임의의 수치를 예측하는 문제입니다. 따라서 타깃값도 임의의 수치가 됩니다.
- **k-최근접 이웃 회귀**

  - k-최근접 이웃 알고리즘을 사용해 회귀 문제를 풉니다.
  - 가장 가까운 이웃 샘플을 찾고 이 샘플의 타깃값을 평균하여 예측으로 삼습니다.

- **결정계수(R^2)**

  - 대표적인 회귀 문제의 성능 측정 도구입니다.
  - 1에 가까울수록 좋고, 0에 가깝다면 성능이 나쁜 모델입니다.

- **과대적합**
  - 모델의 훈련 세트 성능이 테스트 세트 성능보다 훨씬 높을 때 일어납니다.
  - 모델이 훈련 세트에 너무 집착해서 데이터에 내재된 거시적인 패턴을 감지하지 못합니다.
- **과소적합**
  - 훈련 세트와 테스트 세트의 성능이 모두 동일하게 낮거나 테스트 세트 성능이 오히려 더 높을 때 일어납니다.
  - 이런 경우 더 복잡한 모델을 사용해 훈련 세트에 잘 맞는 모델을 만들어야 합니다.

## 핵심 패키지와 함수

### scikit-learn

- **KNeighborsRegressor**
  - k-최근접 이웃 회귀 모델을 만드는 사이킷런 클래스입니다.
  - `n_neighbors` 매개변수로 이웃의 개수를 지정합니다. 기본값은 5입니다.
  - 다른 매개 변수는 `KNeighborClassifier` 클래스와 거의 동일합니다.
- **mean_absolute_error()**
  - 회귀 모델의 평균 절대값 오차를 계산합니다.
  - 첫 번째 매개변수는 타깃, 두 번째 매개변수는 예측값을 전달합니다.
  - 이와 비슷한 함수로는 평균 제곱 오차를 계산하는 `mean_squared_error()`가 있습니다.
  - 이 함수는 타깃과 예측을 뺀 값을 제곱한 다음 전체 샘플에 대해 평균한 값을 반환합니다.

### numpy

- **reshape()**
  - 배열의 크기를 바꾸는 메서드입니다.
  - 바꾸고자 하는 배열의 크기를 매개벼누로 전달합니다. 바꾸기 전후의 배열 원소 개수는 동일해야 합니다.
  - 넘파이는 종종 배열의 메서드와 동일한 함수를 별도로 제공합니다.
  - 함수의 첫 번째 매개변수는 바꾸고자 하는 배열입니다.
  - 예를 들면 `test_array.reshape(2, 2)`는 `np.reshape(test_array, (2,2))`와 같이 바꿔 쓸 수 있습니다.

## k-최근접 이웃 회귀

- 회귀는 클래스 중 하나로 분류하는 것이 아니라 임의의 어떤 숫자를 예측하는 문제입니다.
- 내년도 경제 성장률을 예측하거나 배달이 도착할 시간을 예측하는 것이 회귀 문제입니다.
- 농어의 무게를 예측하는 것도 회귀가 됩니다.
- 회귀는 정해진 클래스가 없고 임의의 수치를 출력합니다.
- 예측하려는 샘플에 가장 가까운 샘플 k개를 선택합니다. 하지만 회귀이기 때문에 이웃한 샘플의 타깃은 어떤 클래스가 아니라 임의의 수치입니다.
- 이웃 샘플의 수치를 사용해 새로운 샘플 X의 타깃을 예측하는 간단한 방법은 이 수치들의 평균을 구하는 것 입니다.
- 이웃한 샘플의 타깃값이 각각 100, 80, 60이고 이를 평균하면 샘플 X의 예측 타깃값은 80이 됩니다.

> 회귀라는 용어를 보고 어떤 알고리즘인지 예측하기 참 어렵습니다. 이 용어는 19세기 통계학자이자 사회학자인 프랜시스 골턴(Francis Galton)이 처음 사용했습니다. 그는 키가 큰 사람의 아이가 부모보다 더 크지 않다는 사실을 관찰하고 이를 '평균으로 회귀한다'라고 표현했습니다. 그 후 **두 변수 사이의 상관관계를 분석하는 방법**을 회귀라 불렀습니다.

## 데이터 준비

```python
import numpy as np
```

```python
perch_length = np.array(
    [8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0,
     21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5,
     22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5,
     27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0,
     36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0,
     40.0, 42.0, 43.0, 43.0, 43.5, 44.0]
     )
perch_weight = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0,
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0,
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0,
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0,
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0,
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0,
     1000.0, 1000.0]
     )
```

```python
import matplotlib.pyplot as plt
plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

- 데이터가 어떤 형태를 띠고 있는지 산점도를 그려봅니다.
- 하나의 특성을 사용기 때문에 특성 데이터를 X축에 놓고 타깃 데이터를 y축에 놓습니다.
- 맷플롯립을 임포트 하고 `scatter()`함수를 사용하여 산점도를 그립니다.
- 농어의 길이가 커짐에 따라 무게도 늘어납니다.

```python
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42)
```

- 농어 데이터를 머신러닝 모델에 사용하기 전에 훈련 세트와 테스트 세트로 나눕니다.
- 결과를 동일하게 유지하기 위해 random_state=42로 지정

```python
print(train_input.shape, test_input.shape)
```

```
(42,) (14,)
```

```python
test_array = np.array([1,2,3,4])
print(test_array.shape)
```

```
(4,)
```

```python
test_array = test_array.reshape(2, 2)
print(test_array.shape)
```

- 특성을 1개만 사용하므로 2차원 배열을 만들어야 합니다. 
- 넘파이 배열은 크기를 바꿀 수 있는 2차원 배열을 만듭니다. 이 때 `reshape()` 메서드를 제공
- (4,) 배열을 (2,2)로 변경하는 예제

```
(2, 2)
```

```python
# 아래 코드의 주석을 제거하고 실행하면 에러가 발생합니다
# test_array = test_array.reshape(2, 3)
```

```python
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
```

```python
print(train_input.shape, test_input.shape)
```

```
(42, 1) (14, 1)
```

## 결정 계수( R^2 )

```python
from sklearn.neighbors import KNeighborsRegressor
```

```python
knr = KNeighborsRegressor()
# k-최근접 이웃 회귀 모델을 훈련합니다
knr.fit(train_input, train_target)
```

```python
knr.score(test_input, test_target)
```

```
0.992809406101064
```

```python
from sklearn.metrics import mean_absolute_error
```

```python
# 테스트 세트에 대한 예측을 만듭니다
test_prediction = knr.predict(test_input)
# 테스트 세트에 대한 평균 절댓값 오차를 계산합니다
mae = mean_absolute_error(test_target, test_prediction)
print(mae)
```

```
19.157142857142862
```

## 과대적합 vs 과소적합

```python
print(knr.score(train_input, train_target))
```

```
0.9698823289099254
```

```python
# 이웃의 갯수를 3으로 설정합니다
knr.n_neighbors = 3
# 모델을 다시 훈련합니다
knr.fit(train_input, train_target)
print(knr.score(train_input, train_target))
```

```
0.9804899950518966
```

```python
print(knr.score(test_input, test_target))
```
