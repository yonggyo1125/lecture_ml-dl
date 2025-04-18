# 로지스틱 회귀

## 키워드 정리

- **로지스틱 회귀**
  - 선형 방정식을 사용한 분류 알고리즘
  - 선형 회귀와 달리 시그모이드 함수나 소프트맥스 함수를 사용하여 클래스 확률을 출력할 수 있습니다.

- **다중 분류**
  - 타깃 클래스가 2개 이상인 분류 문제입니다.
  - 로지스틱 회귀는 다중 분류를 위해 소프트맥스 함수를 사용하여 클래스를 예측합니다.

- **시그모이드 함수**
  - 선형 방정식의 출력을 0과 1사이의 값으로 압축하여 이진 분류를 위해 사용합니다.

- **소프트맥스 함수**
  - 다중 분류에서 여러 선형 방정식의 출력 결과를 정규화하여 합이 1이 되도록 만듭니다.

## 핵심 패키지와 함수
### scikit-learn
- **LogisticRegression**
    - 선형 분류 알고리즘인 로지스틱 회귀를 위한 클래스입니다. 
    - solver 매개변수에서 사용할 알고리즘을 선택할 수 있습니다. 기본값은 `lbfgs` 입니다. 
    - 사이킷런 0.17 버전에 추가된 `sag` 는 확률적 평균 경사 하강법 알고리즘으로 특성과 샘플 수가 많을 때 성능은 빠르고 좋습니다. 
    - 사이킷런 0.19 버전에는 `sag`의 개선 버전인 `saga`가 추가되었습니다. 
    - `penalty` 매개변수에서 L2 규제(릿지 방식)과 L1 규제(라쏘 방식)을 선택할 수 있습니다. 기본값은 L2 규제를 의미하는 `l2` 입니다.
    - `C` 매개변수에서 규제의 강도를 제어합니다. 기본값은 1.0이며 값이 작을수록 규제가 강해집니다.

- **predict_proba()** : 예측 확률을 반환합니다.
    - 이진 분류의 경우에는 샘플마다 음성 클래스와 양성 클래스에 대한 확률을 반환합니다.
    - 다중 분류의 경우에는 샘플마다 모든 클래스에 대한 확률을 반환합니다.

- **decision_function()** : 모델이 학습한 선형 방정식의 출력을 반환합니다.
    - 이진 분류의 경우 양성 클래스의 확률이 반환됩니다. 이 값이 0보다 크면 양성 클래스, 작거나 같으면 음성 클래스로 예측합니다.
    - 다중 분류의 경우 각 클래스마다 선형 방정식을 계산합니다. 가장 큰 클래스가 예측 클래스가 됩니다.

## 럭키백의 확률

- 문제: 럭키백에 들어간 생선의 크기, 무게 등이 주어졌을 때 7개 생선에 대한 확률을 출력해야 합니다. 이번에는 길이, 높이, 두께 외에도 대각선 길이와 무게를 사용할 수 있습니다. 
- 사이킷런의 k-최근접 이웃 분류기로 럭키백에 들어간 생선의 확률을 계산해 보겠습니다.

## 데이터 준비하기

```python
import pandas as pd

fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish.head()
```

- 판다스의 `read_csv()` 함수로 인터넷에서 직접 CSV 데이터를 읽어 데이터 프레임으로 변환한 다음 head() 메서드로 5개 행을 출력합니다.

![스크린샷 2024-11-03 오후 5 40 08](https://github.com/user-attachments/assets/c5ce8cf6-7782-433f-a76b-feef25e8228e)

- 판다스는 CSV 파일의 첫 줄을 자동으로 인식해 열 제목으로 만들어 줍니다.

```python
print(pd.unique(fish['Species']))
```

- 어떤 종류의 생선이 있는지 `Species` 열에서 고유한 값을 추출합니다.
- 판다스의 `unique()` 함수를 사용

```
['Bream' 'Roach' 'Whitefish' 'Parkki' 'Perch' 'Pike' 'Smelt']
```

- 이 데이터프레임에서 `Species` 열을 타깃으로 만들고 나머지 5개 열은 입력 데이터로 사용합니다. 

```python
fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
```

- 데이터프레임에서 원하는 열을 리스트로 나열하면 됩니다. 
- `Species` 열을 뺴고 나머지 5개 열을 선택합니다. 
- 데이터프레임에서 여러 열을 선택하면 새로운 데이터프레임이 반환됩니다. 이를 `to_numpy()` 메서드로 넘파이 배열로 바꾸어 `fish_input`에 저장했습니다.  

```python
print(fish_input[:5])
```

- `fish_input`에 5개의 특성이 잘 저장되어 있는지 처음 5개 행을 출력해 봅시다.

```
[[242.      25.4     30.      11.52     4.02  ]
 [290.      26.3     31.2     12.48     4.3056]
 [340.      26.5     31.1     12.3778   4.6961]
 [363.      29.      33.5     12.73     4.4555]
 [430.      29.      34.      12.444    5.134 ]]
```

```python
fish_target = fish['Species'].to_numpy()
```

- 동일한 방식으로 타깃 데이터를 만듭니다.

```python
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42)
```

- 데이터를 훈련 세트와 테스트 세트로 나눕니다.

```python
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
```

- 사이킷런의 `StandardScaler` 클래스를 사용해 훈련 세트와 테스트 세트를 표준화 전처리합니다. 
- 반드시 훈련 세트의 통계 값으로 테스트 세트를 변환해야 한다는 점


## k-최근접 이웃 분류기의 확률 예측

```python
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)

print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))
```

- `KeighborsClassifier` 클래스 객체를 만들고 훈련 세트로 모델을 훈련한 다음 훈련 세트와 테스트 세트의 점수를 확인 합니다.
- 최근접 이웃 개수인 k를 3으로 지정하여 사용합니다.

```
0.8907563025210085
0.85
```

- 타깃 데이터를 만들 때 `fish['Species']`를 사용해 만들었기 때문에 훈련 세트와 테스트 세트의 타깃 데이터에도 7개의 생선 종류가 들어가 있습니다. 
- 이렇게 타깃 데이터에 2개 이상의 클래스가 포함된 문제를 **다중 분류**(multi-class classification)라고 부릅니다.
- 이진 분류와 모델을 만들고 훈련하는 방식은 동일합니다.
- 이진 분류를 사용했을 때는 양성 클래스와 음성 클래스를 각각 1과 0으로 지정하여 타깃 데이터를 만들었습니다. 다중 분류에서도 타깃값을 숫자로 바꾸어 입력할 수 있지만 사이킷런에서는 편리하게도 문자열로 된 타깃값을 그대로 사용할 수 있습니다.

```python
print(kn.classes_)
```
- 이때 주의할 점, 타깃값을 그대로 사이킷런 모델에 전달하면 순서가 자동으로 알파벳 순으로 매겨집니다. 따라서 `pd.unique(fish['Species'])`로 출력했던 순서와 다릅니다. 
- `KNeighborsClassifier` 에서 정렬된 타깃값은 `classes_` 속성에 저장되어 있습니다. 

```
['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']
```

```python
print(kn.predict(test_scaled[:5]))
```

 - 테스트 세트에 있는 처음 5개의 샘플의 타깃값을 예측해 보면

```
['Perch' 'Smelt' 'Pike' 'Perch' 'Perch']
```

```python
import numpy as np

proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4))
```

- 사이킷런의 분류 모델은 `predict_proba()` 메서드로 클래스별 확률값을 반환합니다. 
- 테스트 세트에 있는 처음 5개의 샘플에 대한 확률을 출력합니다. 
- 넘파이 `round()` 함수는 기본적으로 소수점 첫째 자리에서 반올림을 하는데, decimals 매개변수로 유지할 소수점 아래 자리수를 지정할 수 있습니다. 

```
[[0.     0.     1.     0.     0.     0.     0.    ]
 [0.     0.     0.     0.     0.     1.     0.    ]
 [0.     0.     0.     1.     0.     0.     0.    ]
 [0.     0.     0.6667 0.     0.3333 0.     0.    ]
 [0.     0.     0.6667 0.     0.3333 0.     0.    ]]
```

- `predict_proba()` 메서드의 출력 순서는 앞서 보았던 `classes_` 속성과 같습니다. 
- 즉, 첫 번째 열이 `Bream`에 대한 확률, 두 번째 열이 `Parkki`에 대한 확률

```python
distances, indexes = kn.kneighbors(test_scaled[3:4])
print(train_target[indexes])
```

- 이 모델이 계산한 확률이 가장 가까운 이웃의 비율이 맞는지 확인합니다. 
- 네 번째 샘플의 최근접 이웃의 클래스를 확인해 봅니다.

```
[['Roach' 'Perch' 'Perch']]
```

- 3개의 최근접 이웃을 사용하기 떄문에 가능한 확률은 0/3, 1/3, 2/3, 3/3이 전부 입니다. 확률이라고 하기엔 어색합니다.
- 더 좋은 방법이 필요 -> 로지스틱 회귀

## 로지스틱 회귀

- **로지스틱 회귀**(logistic regression)는 이름은 회귀이지만 분류 모델입니다. 이 알고리즘은 선형 회귀와 동일하게 선형 방정식을 학습합니다.
  
![스크린샷 2024-11-03 오후 6 19 18](https://github.com/user-attachments/assets/51fc1f8e-1ff3-4dc4-87d7-7c8539a3a76a)

- a, b, c, d, e는 가중치 혹은 계수입니다. 특성은 늘어났지만 다중 회귀를 위한 선형 방정식과 같습니다. 
- z는 어떤 값도 가능합니다. 하지만 확률이 되려면 0\~1(또는 0\~100%) 사이 값이 되어야 합니다. 
- z가 아주 큰 음수일 때 0이 되고, z가 아주 큰 양수일 때 1이 되도록 바꾸는 방법 -> **시그모이드 함수** sigmoid function (또는 **로지스틱 함수**(logistic function))를 사용하면 가능합니다.


![스크린샷 2024-11-03 오후 6 26 05](https://github.com/user-attachments/assets/1d2defea-1c1d-4411-8f69-96b8c0321140)


- z가 무한하게 큰 음수일 경우 이 함수는 0에 가까워지고, z가 무한하게 큰 양수가 될 때는 1에 가까워집니다. z가 0이 될 때는 0.5가 됩니다.
- z가 어떤 값이 되더라도 절대로 0\~1 사이의 범위를 벗어날 수 없습니다. 즉 0\~1 사이 값을 0\~100% 까지 확률로 해석할 수 있습니다. 


```python
import numpy as np
import matplotlib.pyplot as plt

z = np.arange(-5, 5, 0.1)
phi = 1 / (1 + np.exp(-z))

plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()
```

- 넘파이를 사용하면 그래프를 간단히 그릴 수 있습니다. 
- -5와 5 사이에 0.1 간격으로 배열 z를 만든 다음 z 위치마다 시그모이드 함수를 계산합니다. 
- 지수 함수 계산은 `np.exp()` 함수를 사용합니다.

![스크린샷 2024-11-03 오후 6 31 31](https://github.com/user-attachments/assets/bab31cb0-196d-4875-8b14-d5d62b0a581f)

- 시그모이드 함수의 출력은 0에서 1까지 변합니다. 
- 

## 로지스틱 회귀로 이진 분류 수행하기

- 사이킷 런에는 로지스틱 회귀 모델인 `LogisticRegression` 클래스가 준비되어 있습니다. 
- 간단한 이진 분류를 연습해 봅니다. 
- 이진 분류일 경우 시그모이드 함수의 출력이 0.5보다 크면 양성 클래스, 0.5보다 작으면 음성 클래스로 판단합니다. 
- 정확히 0.5일 때 라이브러리마다 다를 수 있습니다. 사이킷런은 0.5일 때 음성 클래스로 판단합니다.

```python
char_arr = np.array(['A', 'B', 'C', 'D', 'E'])
print(char_arr[[True, False, True, False, False]])
```

- 넘파이 배열은 True, False 값을 전달하여 행을 선택할 수 있습니다. 이를 **불리언 인덱싱**(boolean indexing)이라고 합니다. 

```
['A' 'C']
```

```python
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]
```

- 위와 같은 방식을 사용해 훈련 세트에서 도미(Bream)와 빙어(Smelt)의 행만 골라냅니다. 비교 연산자를 사용하면 도미와 빙어의 행을 모두 True로 만들 수 있습니다.
- 도미인 행만 골라내려면 `train_target == 'Bream'` 과 같이 씁니다. 
- 도미와 빙어에 대한 비교 결과를 비트 OR 연산자(|)를 사용해 합치면 도미와 빙어에 대한 행만 골라낼 수 있습니다.

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)
```

- 이 데이터로 로지스틱 회귀 모델을 훈련합니다.
- `LogisticRegression` 클래스는 선형 모델이므로 `sklearn.linear_model` 패키지 아래 있습니다.


```python
print(lr.predict(train_bream_smelt[:5]))
```

- 훈련한 모델을 사용해 train_bream_smelt에 있는 처음 5개의 샘플을 예측해 봅니다.

```
['Bream', 'Smelt', 'Bream', 'Bream', 'Bream']
```

```python
print(lr.predict_proba(train_bream_smelt[:5]))
```

- `KNeighborsClassifier` 와 마찬가지로 예측 확률은 `predict_proba()` 메서드에서 제공합니다. 
- `train_bream_smelt` 에서 처음 5개 샘플 예측 확률을 출력해 보겠습니다.

```
[[0.99760007 0.00239993]
 [0.02737325 0.97262675]
 [0.99486386 0.00513614]
 [0.98585047 0.01414953]
 [0.99767419 0.00232581]]
```

- 샴풀마다 2개의 확률이 출력되었습니다. 
- 첫 번째 열이 음성 클래스(0)에 대한 확률이고 두 번째 열이 양성 클래스(1)에 대한 확률입니다. 

```python
print(lr.classes_)
```

- k-최근접 이웃 분류기에서 처럼 사이킷런은 타깃값을 알파벳순으로 정렬하여 사용합니다. 
- `classes_` 속성을 확인합니다. 

```
['Bream' 'Smelt']
```
- 빙어(Smelt)가 양성 클래스입니다. 
- `predict_proba()` 메서드가 반환한 배열 값을 보면 두 번째 샘플만 양성 클래스인 빙어의 확률이 높습니다. 
- 나머지는 모두 도미(Bream)로 예측합니다.

```python
print(lr.coef_, lr.intercept_)
```

- 선형 회귀에서처럼 로지스틱 회귀가 학습한 계수를 확인합니다.

```
[[-0.40451732 -0.57582787 -0.66248158 -1.01329614 -0.73123131]] [-2.16172774]
```

- 따라서 이 로지스틱 회귀 모델이 학습한 방정식은 다음과 같습니다.
  
![스크린샷 2024-11-03 오후 8 03 57](https://github.com/user-attachments/assets/52c5e3f5-2567-4d92-a556-be86ae11ca69)


- 로지스틱 회귀는 선형 회귀와 매우 비슷합니다. 
- `LogisticRegression` 모델로 z값을 계산해 봅시다. `LogisticRegression` 클래스는 `decision_function()` 메서드로 z값을 출력할 수 있습니다. 

```python
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)
```

- `train_bream_smelt`의 처음 5개 샘플 z 값을 출력해 봅시다.

```
[-6.02991358  3.57043428 -5.26630496 -4.24382314 -6.06135688]
```

```python
from scipy.special import expit

print(expit(decisions))
```

- 이 z 값을 시그모이드 함수에 통과시키면 확률을 얻을 수 있습니다. 
- 파이썬의 사이파이(scipy) 라이브러리에도 시그모이드 함수가 있습니다. - `expit()` 
- `decisions` 배열의 값을 확률로 변환해 봅시다.

```
[0.00239993 0.97262675 0.00513614 0.01414953 0.00232581]
```

- 출력된 값을 보면 `predict_proba()` 메서드 출력의 두 번째 열의 값과 동일, 즉 `decision_function()` 메서드는 양성 클래스에 대한 z 값을 반환합니다.
- 이진 분류를 위해 2개의 생선 샘플을 골라냈고 이를 사용해 로지스틱 회귀 모델을 훈련했습니다. 
- 이진 분류일 경우 `predict_proba()` 메서드는 양성 클래스에 대한 z값을 계산합니다. 
- `coef_` 속성과 `intercept_` 속성에는 로지스틱 모델이 학습한 선형 방정식의 계수가 들어 있습니다. 

## 로지스틱 회귀로 다중 분류 수행하기

- 다중 분류도 이진 분류와 크게 다르지 않습니다.
- `LogisticRegression` 클래스는 기본적으로 반복적인 알고리즘을 사용합니다. `max_iter` 매개변수에서 반복 횟수를 지정하며 기본값은 100입니다(여기에 준비한 데이터셋을 사용해 모델을 훈련하면 반복 횟수가 부족하다는 경고가 발생합니다.). 
- 충분하게 훈련시키기 위해 반복 횟수를 1,000으로 늘리겠습니다.
- `LogisticRegression`은 기본적으로 릿지 회귀와 같이 계수의 제곱을 규제합니다. 이런 규제를 `L2` 규제라고 부릅니다. 
- 릿지 회귀에서는 alpha 매개변수로 규제의 양을 조절했습니다. alpha가 커지면 규제도 커집니다. 
- `LogisticRegression`에서 규제를 제어하는 매개변수는 `C` 입니다. 하지만 `C`는 alpha와 반대로 작을수록 규제가 커집니다. `C`의 기본값은 1입니다. 여기에서는 규제를 조금 완하하기 위해 20으로 늘리겠습니다. 
- 다음 코드는 `LogisticRegression` 클래스로 다중 분류 모델을 훈련하는 코드입니다. 
- train_scaled, train_target은 7개의 생선 데이터가 모두 들어 있습니다.

```python
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)

print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
```

```
0.9327731092436975
0.925
```

- 훈련 세트와 테스트 세트에 대한 점수가 높고 과대적합이나 과소적합으로 치우친 것 같지 않습니다.

```python
print(lr.predict(test_scaled[:5]))
```

- 테스트 세트의 처음 5개 샘플에 대한 예측을 출력해 보면

```
['Perch' 'Smelt' 'Pike' 'Roach' 'Perch']
```

```python
proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))
```

- 테스트 세트의 처음 5개 샘플에 대한 예측 확률을 출력해 봅니다(출력을 간소하게 하기 위해 소수점 네 번째 자리에서 반올림)

```
[[0.    0.014 0.842 0.    0.135 0.007 0.003]
 [0.    0.003 0.044 0.    0.007 0.946 0.   ]
 [0.    0.    0.034 0.934 0.015 0.016 0.   ]
 [0.011 0.034 0.305 0.006 0.567 0.    0.076]
 [0.    0.    0.904 0.002 0.089 0.002 0.001]]
```

- 5개 샘플에 대한 예측이므로 5개의 행이 출력되었으며, 7개 생선에 대한 확률을 계산했으므로 7개의 열이 출력되었습니다. 

```python
print(lr.classes_)
```

```
['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']
```

- 첫 번째 샘플은 Perch를 가장 높은 확률로 예측했습니다.
- 두 번째 샘플은 여섯 번째 열인 Smelt를 가장 높은 확률(94.6%)로 예측했습니다.
- 이진 분류는 샘플마다 2개의 확률을 출력하고 다중 분류는 샘플마다 클래스 개수만큼 확률을 출력합니다.
- 여기에서는 7개입니다. 이 중에서 가장 높은 확률이 예측 클래스가 됩니다. 

```python
print(lr.coef_.shape, lr.intercept_.shape)
```

- 다중 분류인 경우 선형 방정식을 알아보려면 `coef_`와 `intercept_` 의 크기를 출력해 봅니다. 

```
(7, 5) (7,)
```
- 이 데이터는 5개의 특성을 사용하므로 `coef_` 배열의 열은 5개 입니다. 그런데 행이 7입니다. `intercept_`도 7개나 있습니다.
- 이 말은 이진 분류에서 보았던 z를 7개나 계산한다는 의미 즉, 다중 분류는 클래스마다 z값을 하나씩 계산합니다. 여기에서 가장 높은 z값을 출력하는 클래스가 예측 클래스가 됩니다. 
- 확률 계산시 이진 분류에서는 시그모이드 함수를 사용해 z를 0과 1사이의 값으로 변환했습니다. 다중 분류는 이와 달리 **소프트맥스**(softmax)함수를 사용하여 7개의 z값을 확률로 변환합니다. 

> 소프트맥스 함수
>
> 시그모이드 함수는 하나의 선형 방정식의 출력값을 0\~1 사이로 압축합니다. 이와 달리 소프트맥스 함수는 여러 개의 선형 방정식의 출력값을 0\~1 사이로 압축하고 전체 합이 1이 되도록 만듭니다. 이를 위해 지수 함수를 사용하기 때문에 **정규화된 지수 함수**라고도 부릅니다.

#### 소프트맥스 함수 계산 방식
- 먼저 7개의 z 값의 이름을 z1에서 z7이라고 붙이겠습니다. 
- z1\~z7까지 값을 사용해 지수 함수 e^z1\~e^z7을 계산해 모두 더합니다. 이를 e_sum이라고 하겠습니다.

![스크린샷 2024-11-03 오후 8 39 46](https://github.com/user-attachments/assets/935e0526-d694-44b2-8c8f-77bb764072d8)


- 그 다음 e^z1\~e^z7을 각각 e_sum으로 나누어 주면 됩니다.

![스크린샷 2024-11-03 오후 8 39 55](https://github.com/user-attachments/assets/66254150-6b7e-416d-895a-60ce18b8b7d3)


- s1에서 s7까지 모두 더하면 분자와 분모가 같아지므로 1이 됩니다. 7개의 생선에 대한 확률의 합은 1이 되어야 하므로 잘 맞습니다.

```python
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))
```

- 이진 분류에서 처럼 `decision_function()` 메서드로 z1\~z7까지의 합을 구한 다음 소프트맥스 함수를 사용해 확률로 바꾸어 보겠습니다. 
- 먼저 테스트 세트의 5개 샘플에 대한 z1\~z7의 값을 구해봅시다.

```
[[ -6.51   1.04   5.17  -2.76   3.34   0.35  -0.63]
 [-10.88   1.94   4.78  -2.42   2.99   7.84  -4.25]
 [ -4.34  -6.24   3.17   6.48   2.36   2.43  -3.87]
 [ -0.69   0.45   2.64  -1.21   3.26  -5.7    1.26]
 [ -6.4   -1.99   5.82  -0.13   3.5   -0.09  -0.7 ]]
```

```python
from scipy.special import softmax

proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))
```

- 사이파이 라이브러리는 소프트 맥스 함수도 제공합니다. `scipy.special` 아래에 `softmax()` 함수를 임포트해 사용합니다.
- 앞서 구한 `decision` 배열을 `softmax()` 함수에 전달했습니다. `softmax()` 의 axis 매개변수는 소프트맥스를 계산할 축을 지정합니다. 
- 여기에서는 `axis=1`로 지정하여 각 행, 즉 각 샘플에 대해 소프트맥스를 계산합니다. 
- 만약 axis 매개변수를 지정하지 않으면 배열 전체에 대해 소프트맥스를 계산합니다. 

```
[[0.    0.014 0.842 0.    0.135 0.007 0.003]
 [0.    0.003 0.044 0.    0.007 0.946 0.   ]
 [0.    0.    0.034 0.934 0.015 0.016 0.   ]
 [0.011 0.034 0.305 0.006 0.567 0.    0.076]
 [0.    0.    0.904 0.002 0.089 0.002 0.001]]
```
