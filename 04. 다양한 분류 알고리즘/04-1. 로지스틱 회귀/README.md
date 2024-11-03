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



```python
print(pd.unique(fish['Species']))
```

```
['Bream' 'Roach' 'Whitefish' 'Parkki' 'Perch' 'Pike' 'Smelt']
```

```python
fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
```

```python
print(fish_input[:5])
```

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

## k-최근접 이웃 분류기의 확률 예측

```python
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)

print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))
```

```
0.8907563025210085
0.85
```

```python
print(kn.classes_)
```

```
['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']
```

```python
print(kn.predict(test_scaled[:5]))
```

```
['Perch' 'Smelt' 'Pike' 'Perch' 'Perch']
```

```python
import numpy as np

proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4))
```

```
[[0.     0.     1.     0.     0.     0.     0.    ]
 [0.     0.     0.     0.     0.     1.     0.    ]
 [0.     0.     0.     1.     0.     0.     0.    ]
 [0.     0.     0.6667 0.     0.3333 0.     0.    ]
 [0.     0.     0.6667 0.     0.3333 0.     0.    ]]
```

```python
distances, indexes = kn.kneighbors(test_scaled[3:4])
print(train_target[indexes])
```

```
[['Roach' 'Perch' 'Perch']]
```

## 로지스틱 회귀

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
