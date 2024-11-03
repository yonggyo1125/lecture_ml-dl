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

## 로지스틱 회귀로 이진 분류 수행하기

```python
char_arr = np.array(['A', 'B', 'C', 'D', 'E'])
print(char_arr[[True, False, True, False, False]])
```

```
['A' 'C']
```

```python
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]
```

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)
```

```python
print(lr.predict(train_bream_smelt[:5]))
```

```python
print(lr.predict_proba(train_bream_smelt[:5]))
```

```python
print(lr.classes_)
```

```
['Bream' 'Smelt']
```

```python
print(lr.coef_, lr.intercept_)
```

```
[[-0.40451732 -0.57582787 -0.66248158 -1.01329614 -0.73123131]] [-2.16172774]
```

```python
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)
```

```
[-6.02991358  3.57043428 -5.26630496 -4.24382314 -6.06135688]
```

```python
from scipy.special import expit

print(expit(decisions))
```

```
[0.00239993 0.97262675 0.00513614 0.01414953 0.00232581]
```

## 로지스틱 회귀로 다중 분류 수행하기

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

```python
print(lr.predict(test_scaled[:5]))
```

```
['Perch' 'Smelt' 'Pike' 'Roach' 'Perch']
```

```python
proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))
```

```
[[0.    0.014 0.842 0.    0.135 0.007 0.003]
 [0.    0.003 0.044 0.    0.007 0.946 0.   ]
 [0.    0.    0.034 0.934 0.015 0.016 0.   ]
 [0.011 0.034 0.305 0.006 0.567 0.    0.076]
 [0.    0.    0.904 0.002 0.089 0.002 0.001]]
```

```python
print(lr.classes_)
```

```
['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']
```

```python
print(lr.coef_.shape, lr.intercept_.shape)
```

```
(7, 5) (7,)
```

```python
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))
```

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

```
[[0.    0.014 0.842 0.    0.135 0.007 0.003]
 [0.    0.003 0.044 0.    0.007 0.946 0.   ]
 [0.    0.    0.034 0.934 0.015 0.016 0.   ]
 [0.011 0.034 0.305 0.006 0.567 0.    0.076]
 [0.    0.    0.904 0.002 0.089 0.002 0.001]]
```
