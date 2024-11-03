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

### scikit-learn

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

- 오른쪽 그림처럼 특성이 2개면 타깃값과 함께 3차원 공간을 형성하고 선형 회귀 방정식 `타깃 = a X 특성1 + b X 특성2 + 절편`은 평면이 됩니다.
- 특성이 3개일 경우? 3차원 공간 이상을 그리거나 상상하기는 힘듭니다.
- 선형 회귀를 단순한 직선이나 평면으로 생각하여 성능이 무조건 맞다고 오해해서는 안됩니다. 특성이 많은 고차원에서는 선형 회귀가 매우 복잡한 모델을 표현할 수 있습니다.
- 농어의 길이 뿐만 아니라 높이와 두께도 함께 사용합니다. 이와 더불어 3개의 특성을 각각 제곱하여 추가합니다.
- 또한 각 특성을 서로 곱해서 또 다른 특성을 만들겠습니다. 즉 '농어 길이 X 농어 높이'를 새로운 특성으로 만들게 됩니다. 이렇게 기존의 특성을 사용해 사로운 특성을 뽑아내는 작업을 **특성 공학**(feature engineering)이라고 부릅니다.

## 데이터 준비

```python
import pandas as pd
```

```python
df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy()
print(perch_full)
```

- 판다스의 `read_csv()` 함수를 사용하면 인터넷에서 csv 파일을 바로 다운로드하여 사용할 수 있습니다. 다운로드 받은 csv는 **데이터프레임**(dataframe)으로 변환됩니다.

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

- 타깃 데이터를 준비합니다.

```python
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)
```

- perch_full과 perch_weight를 훈련 세트와 테스트 세트로 나눕니다.
- 이 데이터를 사용해 새로운 특성을 만들겠습니다.

## 사이킷런의 변환기

- 사이킷런은 특성을 만들거나 전처리하기 위한 다양한 클래스를 제공합니다. 사이킷런에서는 이런 클래스를 **변환기**(transformer)라고 부릅니다.
- 변환기 클래스는 모두 `fit()`, `transform()` 메서드를 제공합니다.

```python
from sklearn.preprocessing import PolynomialFeatures
```

- 사용할 변환기는 `PolynomialFeatures` 클래스입니다.
- 이 클래스는 `sklearn.preprocessing` 패키지에 포함되어 있습니다.

```python
poly = PolynomialFeatures()
poly.fit([[2, 3]])
print(poly.transform([[2, 3]]))
```

- 2개의 특성 2와 3으로 이루어진 샘플 하나를 적용합니다.
- 이 클래스의 객체를 만든 다음 `fit()`, `transform()` 메서드를 차례대로 호출합니다.
- `fit()` : 새롭게 만들 특성 조합을 찾습니다.
- `transform()` : 실제 데이터로 변환합니다.

```
[[1. 2. 3. 4. 6. 9.]]
```

- `PolynomialFeatures` 클래스는 기본적으로 각 특성을 제곱한 항을 추가하고 특성끼리 서로 곱한 항을 추가합니다.
- 2와 3을 각기 제곱한 4와 9가 추가되었고, 2와 3을 곱한 6이 추가되었습니다.
- 1이 추가된 이유는?

```
무게 = a X 길이 + b X 높이 + c X 두께 + d X 1
```

- 선형 방정식의 절편을 항상 값이 1인 특성과 곱해지는 계수라고 볼 수 있습니다. 이렇게 보면 특성은 (길이, 높이, 두께, 1)이 됩니다. 하지만 사이킷런의 선형 모델은 자동으로 절편을 추가하므로 굳이 이렇게 특성을 만들 필요가 없습니다.

> transform전에 꼭 poly.fit을 사용해야 하나요?
>
> 훈련(fit)을 해야 변환(transform)이 가능합니다. 사이킷런의 일관된 api 때문에 두 단계로 나뉘어져 있습니다. 두 메서드를 하나로 붙인 `fit_transform` 메서드도 있습니다.

```python
poly = PolynomialFeatures(include_bias=False)
poly.fit([[2, 3]])
print(poly.transform([[2, 3]]))
```

- `include_bias=False` 로 지정하여 다시 특성을 변환하겠습니다.

```
[[2. 3. 4. 6. 9.]]
```

- 절편을 위한 항이 제거되고 특성의 제곱과 특성끼리 곱한 항만 추가되었습니다.

> include_bias=False는 꼭 지정해야 하나요?
>
> `include_bias=False`로 지정하지 않아도 사이킷런 모델은 자동으로 특성에 추가된 절편 항을 무시합니다. 하지만 여기에서는 혼돈을 피하기 위해 명시적으로 지정하겠습니다.

```python
poly = PolynomialFeatures(include_bias=False)

poly.fit(train_input)
train_poly = poly.transform(train_input)
print(train_poly.shape)
```

- 이 방식으로 `train_input`에 적용합니다.
- `train_input`을 변환한 데이터를 `train_poly`에 저장하고 이 배열의 크기를 확인해 봅니다.

```
(42, 9)
```

```python
poly.get_feature_names_out()
```

- `PolynomialFeatures` 클래스는 9개의 특성이 어떻게 만들어졌는지 확인하는 방법을 제공합니다.
- `get_feature_names_out()` 메서드를 호출하면 9개의 특성이 각각 어떤 입력의 조합으로 만들어졌는지 확인할 수 있습니다.

```
array(['x0', 'x1', 'x2', 'x0^2', 'x0 x1', 'x0 x2', 'x1^2', 'x1 x2',
       'x2^2'], dtype=object)
```

- `x0`은 첫 번째 특성을 의미, `x0^2` 는 첫 번째 특성의 제곱, `x0 x1`은 첫 번째 특성과 두 번째 특성의 곱을 나타내는 식입니다.

```python
test_poly = poly.transform(test_input)
```

- 테스트 세트를 변환합니다.
- 변환된 특성을 사용하여 다중 회귀 모델을 훈련하겠습니다.

## 다중 회귀 모델 훈련하기

- 다중 회귀 모델을 훈련하는 것은 선형 회귀 모델을 훈련하는 것과 같습니다. 다만 여러 개의 특성을 사용하여 선형 회귀를 수행하는 것뿐
- 사이킷런의 `LinearRegession` 클래스를 임포트하고 앞에서 만든 `train_poly를 사용해 모델을 훈련시킵니다.

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
```

```
0.9903183436982125
```

- 농어의 길이뿐만 아니라 높이와 두께를 모두 사용하였고, 각 특성을 제곱하거나 서로 곱해서 다항 특성을 더 추가했습니다. 높은 점수로 평가되었습니다.
- 특성이 늘어나면 선형 회귀의 능력은 매우 강하다는 것을 알 수 있습니다.

```python
print(lr.score(test_poly, test_target))
```

- 테스트 세트에 대한 점수를 확인합니다.

```
0.9714559911594111
```

- 테스트 세트에 대한 점수는 높아지지 않았지만 농어의 길이만 사용했을 때 과소적합 문제는 더이상 나타나지 않았습니다.

```python
poly = PolynomialFeatures(degree=5, include_bias=False)

poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
print(train_poly.shape)
```

- 특성을 더 많이 추가한다면?
- `PolynomialFeatures` 클래스의 degree 매개변수를 사용하여 필요한 고차항의 최대 차수를 지정할 수 있습니다.
- 5제곱까지 특성을 만들어 출력해 봅시다.

```
(42, 55)
```

- 만들어진 특성의 개수가 55개나 됩니다. train_poly 배열의 열의 개수가 특성의 개수입니다.

```python
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
```

- 이 데이터를 사용해 선형 회귀 모델을 다시 훈련하겠습니다.

```
0.9999999999996433
```

- 거의 완벽한 점수

```python
print(lr.score(test_poly, test_target))
```

- 테스트 세트에 대한 점수는?

```
-144.40579436844948
```

- 음수가 나왔습니다.
- 특성의 개수를 크게 늘리면 선형 모델은 아주 강력해집니다. 훈련 세트에 대해 거의 완벽하게 학습할 수 있습니다.
- 그러나 이런 모델은 훈련 세트에 너무 과대적합되므로 테스트 세트에서는 형편없는 점수를 만듭니다.
- 이 문제를 해결하려면 다시 특성을 줄여야 합니다.

## 규제

- **규제**(regularization)는 머신러닝 모델이 훈련 세트를 너무 과도하게 학습하지 못하도록 훼방하는 것을 말합니다.
- 즉, 모델이 훈련 세트에 과대적합되지 않도록 만드는 것입니다. 선형 회귀 모델의 경우 특성에 곱해지는 계수(또는 기울기)의 크기를 작게 만드는 일입니다.

![스크린샷 2024-11-03 오후 2 01 19](https://github.com/user-attachments/assets/a901145e-61a0-4473-8941-82bdd557c53a)

- 왼쪽은 훈련 세트를 과도하게 학습했고 오른쪽은 기울기를 줄여 보다 보편적인 패턴을 학습하고 있습니다.
- 앞서 55개의 특성으로 훈련한 회귀 모델의 계수를 규제하여 훈련 세트의 점수를 낮추고 대신 테스트 세트의 점수를 높여 보겠습니다.
- 그 전에 특성의 스케일에 대해 잠시 생각해 보면, 특성의 스케일이 정규화되지 않으면 여기에 곱해지는 계수 값도 차이나게 됩니다. 일반적으로 선형 회귀 모델에 규제를 적용할 때 계수 값의 크기가 서로 많이 다르면 공정하게 제어되지 않을 것입니다.
- 그렇다면 규제를 적용하기 전에 먼저 정규화를 해야 합니다. 사이킷런에서 제공하는 `StandardScaler` 클래스를 사용하면 됩니다. 이 클래스도 변환기의 하나 입니다.

```python
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_poly)

train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)
```

- `StandardScaler` 클래스의 객체 ss를 초기화환 후 `PolynomialFeatures` 클래스로 만든 train_poly를 사용해 이 객체를 훈련합니다.
- 반드시 꼭 훈련 세트로 학습한 변환기를 사용해 테스트 세트까지 변환해야 합니다.
- 이제 표준 점수로 변환한 `train_scaled`와 `test_scaled`가 준비되었습니다.

> 훈련 세트에서 학습한 평균과 표준편차는 `StandardScaler` 클래스 객체의 `mean_`, `scale_` 속성에 저장됩니다. 특성마다 계산하므로 55개의 평균과 표준 편차가 들어 있습니다.

- 선형 회귀 모델에 규제를 추가한 모델을 **릿지**(ridge)와 **라쏘**(lasso)라고 부릅니다. 두 모델은 규제를 가하는 방법이 다릅니다.
- 릿지는 계수를 제곱한 값을 기준으로 규제를 적용합니다. 라쏘는 계수의 절대값을 기준으로 규제를 적용합니다.
- 일반적으로 릿지를 조금 더 선호합니다.
- 두 알고리즘 모두 계수의 크기를 줄이지만 라쏘는 아예 0으로 만들 수도 있습니다.
- 사이킷런이 이 두 알고리즘을 모두 제공합니다.

## 릿지

```python
from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
```

- 릿지와 라쏘 모두 `sklearn.linear_model` 패키지 안에 있습니다. 사이킷런 모델을 사용할 때 편리한 점은 훈련하고 사용하는 방법이 같다는 것
- 모델 객체를 만들고 `fit()` 메서드에서 훈련한 다음 `score()` 메서드로 평가합니다.
- 준비한 `train_scaled` 데이터로 릿지 모델을 훈련합니다.

```
0.9896101671037343
```

- 선형 회귀에서는 거의 완벽에 가까웠던 점수가 조금 낮아졌습니다.

```python
print(ridge.score(test_scaled, test_target))
```

- 테스트 세트에 대한 점수를 확인하면

```
0.9790693977615387
```

- 테스트 세트 점수가 정상으로 돌아왔습니다.
- 확실히 많은 특성을 사용했음에도 불구하고 훈련 세트에 너무 과대적합되지 않아 테스트 세트에서도 좋은 성능을 내고 있습니다.

- 릿지와 라쏘 모델을 사용할 떄 규제의 양을 임의로 조절할 수 있습니다. 모델 객체를 만들 떄 `alpha` 매개변수로 규제의 강도를 조절합니다.
- `alpha` 값이 크면 규제 강도가 세지므로 계수 값을 더 줄이고 조금 더 과소적합되도록 유도합니다.
- `alpha` 값이 작으면 계수를 줄이는 역할이 줄어들고 선형 회귀 모델과 유사해지므로 과대적합될 가능성이 큽니다.

> **사람이 직접 지정해야 하는 매개변수**
>
> `alpha` 값은 릿지 모델이 학습하는 값이 아니라 사전에 우리가 지정해야 하는 값입니다. 이렇게 머신러닝 모델이 학습할 수 없고 사람이 알려줘야 하는 파라미터를 **하이퍼파라미터**(hyperparameter)라고 부릅니다. 사이킷런과 같은 머신러닝 라이브러리에서 하이퍼파라미터는 클래스와 메서드의 매개변수로 표현됩니다.

```python
import matplotlib.pyplot as plt

train_score = []
test_score = []
```

- 적절한 `alpha` 값을 찾는 한 가지 방법은 `alpha` 값에 대한 R^2 값의 그래프를 그려 보는 것입니다.
- 훈련 세트와 테스트 세트의 점수가 가장 가까운 지점이 최적의 `alpha` 값이 됩니다.
- 맷플롯립을 임포트하고 `alpha` 값을 바꿀 때마다 `score()` 메서드의 결과를 저장할 리스트를 만듭니다.

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

- `alpha` 값을 0.001에서 100까지 10배씩 늘려가며 릿지 회귀 모델을 훈련한 다음 훈련 세트와 테스트 세트의 점수를 파이썬 리스트에 저장합니다.

```python
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()
```

- `alpha` 값을 0.001부터 10배씩 늘렸기 때문에 이대로 그래프를 그리면 그래프 왼쪽이 너무 촘촘해집니다.
- `alpha_list`에 있는 6개의 값을 동일한 간격으로 나타내기 위해 로그 함수로 바꾸어 지수로 표현하겠습니다.
- 즉 0.001은 -3, 0.01은 -2가 되는 식입니다.

> 넘파이 로그 함수는 np.log()와 np.log10()이 있습니다. 전자는 자연 상수 e를 밑으로 하는 자연로그입니다. 후자는 10을 밑으로 하는 상용로그입니다.

![스크린샷 2024-11-03 오후 2 33 52](https://github.com/user-attachments/assets/52a2899a-aa4b-4ff8-b040-f0043edfe408)


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
