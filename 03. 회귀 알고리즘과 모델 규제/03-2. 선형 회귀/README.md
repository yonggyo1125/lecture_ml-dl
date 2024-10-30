# 선형 회귀

## k-최근접 이웃의 한계

```python
import numpy as np

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
from sklearn.model_selection import train_test_split

# 훈련 세트와 테스트 세트로 나눕니다
train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42)
# 훈련 세트와 테스트 세트를 2차원 배열로 바꿉니다
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
```

- 데이터를 훈련 세트와 테스트 세트로 나눕니다.
- 특성 데이터는 2차원 배열로 변환합니다.

```python
from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor(n_neighbors=3)

# k-최근접 이웃 회귀 모델을 훈련합니다
knr.fit(train_input, train_target)
```

- 최근접 이웃 개수를 3으로 하는 모델을 훈련합니다.

```python
print(knr.predict([[50]]))
```

- 이 모델을 사용해 길이가 50cm인 농어의 무게를 예측합니다.

```
[1033.33333333]
```

- 50cm 농어의 무게를 1,033g 정도로 예측했습니다.
- 실제 이 농어의 무게는 훨씬 더 많이 나갑니다. 어딘가 문제가 있는 것 처럼 보입니다.

```python
import matplotlib.pyplot as plt

# 50cm 농어의 이웃을 구합니다
distances, indexes = knr.kneighbors([[50]])

# 훈련 세트의 산점도를 그립니다
plt.scatter(train_input, train_target)
# 훈련 세트 중에서 이웃 샘플만 다시 그립니다
plt.scatter(train_input[indexes], train_target[indexes], marker='D')
# 50cm 농어 데이터
plt.scatter(50, 1033, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

- 훈련 세트와 50cm 농어, 이 농어의 최근접 이웃을 산점도에 표시합니다.
- 사이킷런의 k-최근점 이웃 모델의 kneighbors() 메서드를 사용하면 가장 가까운 이웃까지의 거리와 이웃 샘플의 인덱스를 얻을 수 있습니다.

![스크린샷 2024-10-30 오전 7 00 33](https://github.com/user-attachments/assets/fd95e152-e753-4256-a202-3bd9f10ce0ef)

- 길이가 50cm이고 무게가 1,033g인 농어는 ▲(marker='^')으로 표시되고 그 주변의 샘플은 ♦︎(marker='D') 입니다.
- 이 산점도를 보면 길이가 커질수록 농어의 무게가 증가하는 경향이 있습니다.
- 하지만 50cm 농어에서 가장 가까운 것은 45cm 근방이기 때문에 k-최근접 이웃 알고리즘은 이 샘플들의 무게를 평균합니다.

```python
print(np.mean(train_target[indexes]))
```

```
1033.3333333333333
```

- 모델이 예측했던 값과 정확히 일치
- k-최근접 이웃 회귀는 가장 가까운 샘플을 찾아 타깃을 평균합니다. 따라서 새로운 샘플이 훈련 세트의 범위를 벗어나면 엉뚱한 값을 예측할 수 있습니다.
- 예를 들면 100cm인 농어도 1,033g으로 예측합니다.

```python
print(knr.predict([[100]]))
```

```
[1033.33333333]
```

```python
# 100cm 농어의 이웃을 구합니다
distances, indexes = knr.kneighbors([[100]])

# 훈련 세트의 산점도를 그립니다
plt.scatter(train_input, train_target)
# 훈련 세트 중에서 이웃 샘플만 다시 그립니다
plt.scatter(train_input[indexes], train_target[indexes], marker='D')
# 100cm 농어 데이터
plt.scatter(100, 1033, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

- 한번 더 그래프를 그려봅니다.
- 농어의 무게가 아무리 커도 무게가 늘어나지 않습니다.

## 선형 회귀

- **선형 회귀**(linear regression)는 널리 사용되는 대표적인 알고리즘
- 선형이란 말에서 짐작할 수 있듯이 특성이 하나인 경우 어떤 직선을 학습하는 알고리즘입니다.
- 특성을 가장 잘 나타낼 수 있는 직선을 찾는 것

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
# 선형 회귀 모델 훈련
lr.fit(train_input, train_target)

# 50cm 농어에 대한 예측
print(lr.predict([[50]]))
```

- 사이킷런은 `sklearn.linear_model` 패키지 아래에 `LinearRegression` 클래스로 선형 회귀 알고리즘을 구현해 놓았습니다.
- 사이킷런의 모델 클래스들은 훈련, 평가, 예측하는 메서드의 이름이 모두 동일합니다.
- 즉 `LinearRegression` 클래스에도 `fit()`, `score()`, `predict()` 메서드가 있습니다.

```
[1241.83860323]
```

- k-최근접 이웃 회귀를 사용했을 때와 달리 선형 회귀는 50cm 농어의 무게를 아주 높게 예측했습니다.
- 하나의 직전을 그리려면 기울기와 절편이 있어야 합니다. ![스크린샷 2024-10-31 오전 6 21 53](https://github.com/user-attachments/assets/603fad2c-3a25-4f44-98f8-21a9c5e912a4)
- 여기에서 x를 농어의 길이, y를 농어의 무게로 바꾸면 다음과 같습니다.

![스크린샷 2024-10-31 오전 6 22 07](https://github.com/user-attachments/assets/38f63167-f268-485b-bef6-fe78326c0a98)

- 가장 간단한 직선의 방정식 입니다.
- `LinearRegression` 클래스가 찾은 이 데이터와 가장 잘 맞는 a와 b는 lr 객체의 `coef_`와 `intercept_` 속성에 저장되어 있습니다.

```python
print(lr.coef_, lr.intercept_)
```

```
[39.01714496] -709.0186449535477
```

> coef\_ 속성 이름에서 알수 있듯이 머신러닝에서 기울기를 종종 계수(coefficient) 또는 가중치(weight)라고 부릅니다.

> coef*와 intercept*를 머신러닝 알고리즘이 찾은 값이라는 의미로 **모델 파라미터**(model parameter)라고 부릅니다. 머신러닝 알고리즘의 훈련 과정은 최적의 모델 파라미터를 찾는 것과 같습니다. 이를 **모델 기반 학습**이라고 부릅니다. 앞서 사용한 k-최근접 이웃에는 모델 파라미터가 없습니다. 훈련 세트를 저장하는 것이 훈련의 전부입니다. 이를 **사례 기반 학습**이라고 부릅니다.

```python
# 훈련 세트의 산점도를 그립니다
plt.scatter(train_input, train_target)

# 15에서 50까지 1차 방정식 그래프를 그립니다
plt.plot([15, 50], [15*lr.coef_+lr.intercept_, 50*lr.coef_+lr.intercept_])

# 50cm 농어 데이터
plt.scatter(50, 1241.8, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

- 농어의 길이 15에서 50까지 직선으로 그려봅니다.
- 이 직선을 그리려면 앞에서 구한 기울기와 절편을 사용하여 (15, 15 X 39 - 709)와 (50, 50 X 39 - 709) 두 점을 이으면 됩니다.
- 훈련세트의 산점도를 그립니다.

![스크린샷 2024-10-31 오전 6 32 19](https://github.com/user-attachments/assets/e58e21c0-c127-4611-8a09-aa84ae204c4e)

- 이 직선이 선형 회귀 알고리즘이 이 데이터셋에서 찾은 최적의 직선입니다.
- 길이가 50cm인 농어에 대한 예측은 이 직선의 연장선에 있습니다.
- 이제 훈련 세트 범위를 벗어난 농어의 무게도 예측할 수 있습니다.

```python
print(lr.score(train_input, train_target))  # 훈련 세트
print(lr.score(test_input, test_target))    # 테스트 세트
```

- 훈련 세트와 테스트 세트에 대한 R^2 점수를 확인합니다.

```
0.939846333997604
0.8247503123313558
```

- 훈련 세트와 테스트 세트의 점수가 조금 차이가 납니다. 훈련 세트 점수 역시 높지 않으므로 전체적으로 과소적합 되었다고 볼 수 있습니다.
- 과소적합 말고도 또다른 문제가 있습니다. 


## 다항 회귀

- 선형 회귀가 만든 직선이 왼쪽 아래로 쭉 뻗어 있습니다. 이 직선대로 예측하면 농어의 무게가 0g 이하로 내려갈 수 있는데, 현실에서는 있을 수 없는 일입니다. 
- 농어이 길이와 무게에 대한 산점도를 자세히 보면 일직선이라기보다 왼쪽 위로 조금 구부러진 곡선에 가깝습니다. 
- 최적의 직선을 찾기보다 최적의 곡선을 찾는 것이 적합합니다. 

![스크린샷 2024-10-31 오전 6 46 10](https://github.com/user-attachments/assets/4c09ca84-2419-4384-8f8d-84a0f83cf395)

- 이런 2차 방정식의 그래프를 그리려면 길이를 제곱한 항이 훈련 세트에 추가되어야 합니다. 
- 넘파이를 사용하면 아주 간단하게 만들 수 있습니다. 다음 처럼 농어의 길이를 제곱해서 원래 데이터 앞에 붙여 봅니다.


```python
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))
```

- `column_stack()` 함수를 사용하면 간단합니다. `train_input`을 제곱한 것과 `train_input` 두 배열을 나란히 붙이면 됩니다. 
- `test_input`도 마찬가지 입니다. 
- `train_input ** 2` 식에도 넘파이 브로드캐스팅이 적용됩니다. 즉, train_input에 있는 모든 원소를 제곱합니다.  

```python
print(train_poly.shape, test_poly.shape)
```

- 새롭게 만든 데이터셋의 크기를 확인합니다.

```
(42, 2) (14, 2)
```

- 원래 특성인 길이를 제곱하여 왼쪽 열에 추가했기 때문에 훈련 세트와 테스트 세트 모두 열이 2개로 늘어났습니다.

```python
lr = LinearRegression()
lr.fit(train_poly, train_target)

print(lr.predict([[50**2, 50]]))
```

- `train_poly`를 사용해 선형 회귀 모델을 다시 훈련합니다. 
- 여기에서 주목할 점은 2차 방정식의 그래프를 찾기 위해 훈련 세트에 제곱 항을 추가했지만, 타깃값은 그대로 사용한다는 것, 목표하는 값은 어떤 그래프를 훈련하든지 바꿀 필요가 없습니다. 
- 이 훈련 세트로 선형 회귀 모델을 훈련한 다음 50cm짜리 농어에 대해 무게를 예측해 봅시다. 
- 테스트할 때는 이 모델에 농어 길이의 제곱과 원래 길이를 함께 넣어 주어야 합니다.


```
[1573.98423528]
```

- 앞서 훈련한 모델보다 더 높은 값을 예측하였습니다.


```python
print(lr.coef_, lr.intercept_)
```

- 이 모델이 훈련한 계수와 절편을 출력해 봅시다.

```
[  1.01433211 -21.55792498] 116.0502107827827
```

- 이 모델은 다음과 같은 그래프를 학습했습니다.
  
![스크린샷 2024-10-31 오전 6 57 40](https://github.com/user-attachments/assets/177533e4-c4d0-4b01-8eaf-4f369a261ecc)

- 이런 방정식을 다항식(polynomial)이라 부르며 다항식을 사용한 선형 회귀를 **다항 회귀**(polynomial regression)라고 부릅니다.

```python
# 구간별 직선을 그리기 위해 15에서 49까지 정수 배열을 만듭니다
point = np.arange(15, 50)

# 훈련 세트의 산점도를 그립니다
plt.scatter(train_input, train_target)

# 15에서 49까지 2차 방정식 그래프를 그립니다
plt.plot(point, 1.01*point**2 - 21.6*point + 116.05)

# 50cm 농어 데이터
plt.scatter([50], [1574], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

- 이 2차 방정식의 계수와 절편 a, b, c를 알았으니 이전과 동일하게 훈련 세트의 산점도에 그래프로 그려봅니다.
- 짧은 직선을 이어서 그리면 마치 곡선처럼 표현할 수 있습니다.
- 1씩 짧게 끊어서 그려봅니다.

![스크린샷 2024-10-31 오전 7 02 38](https://github.com/user-attachments/assets/ff4e46af-acac-4b26-a952-396c1a135d81)


```python
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))
```

```
0.9706807451768623
0.9775935108325122
```
