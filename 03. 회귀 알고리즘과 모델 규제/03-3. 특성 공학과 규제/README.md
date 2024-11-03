# 특성 공학과 규제

## 키워드 정리

- **다중 회귀**
  - 여러 개의 특성을 사용하는 회귀 모델입니다.
  - 특성이 많으면 선형 모델은 강력한 성능을 발휘합니다.
- **특성 공학** : 주어진 특성을 조합하여 새로운 특성을 만드는 일련의 작업 과정입니다.
- **릿지**
  - 규제가 있는 선형 회귀 모델 중 하나이며 선형 모델의 계수를 작게 만들어 과대적합을 완화시킵니다.
  - 릿지는 비교적 효과가 좋아 널리 사용하는 규제 방법입니다.
- **라쏘**
  - 또 다른 규제가 있는 선형 회귀 모델입니다.
  - 릿지와 달리 계수 값을 아예 0으로 만들 수도 있습니다.
- **하이퍼파라미터**
  - 머신러닝 알고리즘이 학습하지 않는 파라미터, 이런 파라미터는 사람이 사전에 지정해야 합니다.
  - 릿지와 라쏘의 규제 강도 `alpha 파라미터`입니다.

## 핵심 패키지와 함수

### pandas

- **read_csv()**
  - CSV 파일을 로컬 컴퓨터나 인터넷에서 읽어 판다스 데이터프레임으로 변환하는 함수입니다.
  - 자주 사용하는 매개변수
    - `sep`: CSV 파일의 구분자를 지정합니다. 기본값은 '콤마(,)' 입니다.
    - `header`: 데이터프레임의 열 이름으로 사용할 CSV 파일의 행 번호를 지정합니다. 기본적으로 첫 번째 행을 열의 이름으로 사용합니다.
    - `skiprows`: 파일에서 읽기 전에 건너뛸 행의 개수를 지정합니다.
    - `nrows`: 파일에서 읽을 행의 개수를 지정합니다.

### scikit-learn\*\*

- **PolynomialFeatures**
  - 주어진 특성을 조합하여 새로운 특성을 만듭니다.
  - `degree`는 최고 차수를 지정합니다. 기본값은 2입니다.
  - `interaction_only`가 True이면 거듭제곱 항은 제외되고 특성 간의 곱셈 항만 추가됩니다. 기본값은 False입니다.
  - `include_bias`가 False이면 절편을 위한 특성을 추가하지 않습니다. 기본값은 True입니다.
- **Ridge**
  - 규제가 있는 회귀 알고리즘인 릿지 회귀 모델을 훈련합니다.
  - alpha 매개변수로 규제의 강도를 조절합니다. alpha 값이 클수록 규제가 세집니다. 기본값은 1입니다.
  - **solver** 매개변수에 최적의 모델을 찾기 위한 방법을 지정할 수 있습니다. 기본값은 'auto'이며 데이터에 따라 자동으로 선택됩니다.
    - 사이킷런 0.17버전에 추가된 'sag'는 확률적 평균 경사하강법 알고리즘으로 특성과 샘플 수가 많을 때 성능이 빠르고 좋습니다. 사이킷런 0.19 버전에는 'sag'의 개선 버전인 'saga'가 추가되었습니다.
  - **random_state**는 solver가 'sag'나 'saga'일 때 넘파이 난수 시드값을 지정할 수 있습니다.
- **Lasso**
  - 규제가 있는 회귀 알고리즘인 라쏘 회귀 모델을 훈련합니다.
  - 이 클래스는 최적의 모델을 찾기 위해 좌표측을 따라 최적화를 수행해가는 좌표 하강법(coordinate descent)을 사용합니다.
  - `alpha`와 `random_state` 매개변수는 `Ridge` 클래스와 동일합니다.
  - `max_iter`는 알고리즘의 수행 반복 횟수를 지정합니다. 기본값은 1000입니다.

## 다중 회귀

- 여러 개의 특성을 사용한 선형 회귀를 **다중 회귀**(multiple regression)라고 부릅니다.
- 1개의 특성을 사용했을 때 선형 회귀 모델이 학습하는 것은 직선입니다. 반면 특성이 2개면 선형 회귀는 평면을 학습합니다.

![스크린샷 2024-11-03 오전 11 10 39](https://github.com/user-attachments/assets/9dead331-dd31-4094-9be5-c876ff3bdd99)


## 데이터 준비

```python
import pandas as pd
```

```python
df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy()
print(perch_full)
```

```
[[ 8.4   2.11  1.41]
 [13.7   3.53  2.  ]
 [15.    3.82  2.43]
 [16.2   4.59  2.63]
 [17.4   4.59  2.94]
 [18.    5.22  3.32]
 [18.7   5.2   3.12]
 [19.    5.64  3.05]
 [19.6   5.14  3.04]
 [20.    5.08  2.77]
 [21.    5.69  3.56]
 [21.    5.92  3.31]
 [21.    5.69  3.67]
 [21.3   6.38  3.53]
 [22.    6.11  3.41]
 [22.    5.64  3.52]
 [22.    6.11  3.52]
 [22.    5.88  3.52]
 [22.    5.52  4.  ]
 [22.5   5.86  3.62]
 [22.5   6.79  3.62]
 [22.7   5.95  3.63]
 [23.    5.22  3.63]
 [23.5   6.28  3.72]
 [24.    7.29  3.72]
 [24.    6.38  3.82]
 [24.6   6.73  4.17]
 [25.    6.44  3.68]
 [25.6   6.56  4.24]
 [26.5   7.17  4.14]
 [27.3   8.32  5.14]
 [27.5   7.17  4.34]
 [27.5   7.05  4.34]
 [27.5   7.28  4.57]
 [28.    7.82  4.2 ]
 [28.7   7.59  4.64]
 [30.    7.62  4.77]
 [32.8  10.03  6.02]
 [34.5  10.26  6.39]
 [35.   11.49  7.8 ]
 [36.5  10.88  6.86]
 [36.   10.61  6.74]
 [37.   10.84  6.26]
 [37.   10.57  6.37]
 [39.   11.14  7.49]
 [39.   11.14  6.  ]
 [39.   12.43  7.35]
 [40.   11.93  7.11]
 [40.   11.73  7.22]
 [40.   12.38  7.46]
 [40.   11.14  6.63]
 [42.   12.8   6.87]
 [43.   11.93  7.28]
 [43.   12.51  7.42]
 [43.5  12.6   8.14]
 [44.   12.49  7.6 ]]
```

```python
import numpy as np

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

train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)
```

## 사이킷런의 변환기

```python
from sklearn.preprocessing import PolynomialFeatures
```

```python
poly = PolynomialFeatures()
poly.fit([[2, 3]])
print(poly.transform([[2, 3]]))
```

```
[[1. 2. 3. 4. 6. 9.]]
```

```python
poly = PolynomialFeatures(include_bias=False)
poly.fit([[2, 3]])
print(poly.transform([[2, 3]]))
```

```
[[2. 3. 4. 6. 9.]]
```

```python
poly = PolynomialFeatures(include_bias=False)

poly.fit(train_input)
train_poly = poly.transform(train_input)
```

```python
print(train_poly.shape)
```

```
(42, 9)
```

```python
poly.get_feature_names_out()
```

```
array(['x0', 'x1', 'x2', 'x0^2', 'x0 x1', 'x0 x2', 'x1^2', 'x1 x2',
       'x2^2'], dtype=object)
```

```python
test_poly = poly.transform(test_input)
```

## 다중 회귀 모델 훈련하기

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
```

```
0.9903183436982125
```

```python
print(lr.score(test_poly, test_target))
```

```
0.9714559911594111
```

```python
poly = PolynomialFeatures(degree=5, include_bias=False)

poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
```

```python
print(train_poly.shape)
```

```
(42, 55)
```

```python
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
```

```
0.9999999999996433
```

```python
print(lr.score(test_poly, test_target))
```

```
-144.40579436844948
```

## 규제

```python
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_poly)

train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)
```

## 릿지

```python
from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
```

```
0.9896101671037343
```

```python
print(ridge.score(test_scaled, test_target))
```

```
0.9790693977615387
```

```python
import matplotlib.pyplot as plt

train_score = []
test_score = []
```

```python
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    # 릿지 모델을 만듭니다
    ridge = Ridge(alpha=alpha)
    # 릿지 모델을 훈련합니다
    ridge.fit(train_scaled, train_target)
    # 훈련 점수와 테스트 점수를 저장합니다
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))
```

```python
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()
```

```python
ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)

print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))
```

```
0.9903815817570367
0.9827976465386928
```

## 라쏘

```python
from sklearn.linear_model import Lasso

lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
```

```
0.989789897208096
```

```python
print(lasso.score(test_scaled, test_target))
```

```
0.9800593698421883
```

```python
train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    # 라쏘 모델을 만듭니다
    lasso = Lasso(alpha=alpha, max_iter=10000)
    # 라쏘 모델을 훈련합니다
    lasso.fit(train_scaled, train_target)
    # 훈련 점수와 테스트 점수를 저장합니다
    train_score.append(lasso.score(train_scaled, train_target))
    test_score.append(lasso.score(test_scaled, test_target))
```

```python
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()
```

```python
lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)

print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))
```

```
0.9888067471131867
0.9824470598706695
```

```python
print(np.sum(lasso.coef_ == 0))
```

```
40
```
