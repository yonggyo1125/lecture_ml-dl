# k-최근접 이웃 회기

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
```

```python
plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

```python
from sklearn.model_selection import train_test_split
```

```python
train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42)
```

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
