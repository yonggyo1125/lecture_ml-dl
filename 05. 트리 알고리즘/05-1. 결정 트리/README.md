# 결정 트리

## 키워드 정리 
- **결정 트리**
    - 예 / 아니오에 대한 질문을 이어나가면서 정답을 찾아 학습하는 알고리즘입니다.
    - 비교적 예측 과정을 이해하기 쉽고 성능도 뛰어납니다. 
- **불순도**
    - 결정 트리가 최적의 질문을 찾기 위한 기준입니다. 
    - 사이킷런은 지니 불순도와 엔트로피 불순도를 제공합니다.
- **정보 이득**
    - 부모 노드와 자식 노드의 불순도 차이입니다.
    - 결정 트리 알고리즘은 정보 이득이 최대화되도록 학습합니다.
- **가지치기**
    - 결정 트는 제한 없이 성장하면 훈련 세트에 과대적합되기 쉽습니다. 
    - **가지치기**는 결정트리의 성장을 제한하는 방법입니다.
    - 사이킷런의 결정 트리 알고리즘은 여러가지 가지치기 매개변수를 제공합니다.
- **특성 중요도**
    - 결정 트리에 사용된 특성이 불순도를 감소하는데 기여한 정도를 나타내는 값입니다. 
    - 특성 중요도를 계산할 수 있는 것이 결정 트리의 또다른 큰 장점입니다.

## 핵심 패키지와 함수

### pandas
- **info()**
    - 데이터프레임의 요약된 정보를 출력합니다. 
    - 인덱스와 컬럼 타입을 출력하고 널(null)이 아닌 값의 개수, 메모리 사용량을 제공합니다.
    - verbose 매개변수의 기본값 True를 False로 바꾸면 각 열에 대한 정보를 출력하지 않습니다.
- **describe()**
    - 데이터프레임 열의 통계 값을 제공합니다. 
    - 수치형일 경우 최소, 최대, 평균, 표준편차와 사분위값 등이 출력됩니다.
    - 문자열 같은 객체 타입의 열은 가장 자주 등장하는 값과 횟수 등이 출력됩니다.
    - `percentiles` 매개변수에 백분위수를 지정합니다. 기본값은 \[0.25, 0.5, 0.75\]입니다.


### scikit-learn
- **DecisionTreeClassifier** : 결정 트리 분류 클래스
    - `criterion` 매개변수 : 불순도를 지정하며 기본값은 지니 불순도를 의미하는 `gini` 이고 `entropy` 를 선택하여 엔트로피 불순도를 사용할 수 있습니다.
    - `splitter` 매개변수 : 노드를 분할하는 전략을 선택합니다. 기본값은 `best`로 정보 이득이 최대가 되도록 분할합니다. `random`이면 임의로 노드를 분할합니다. 
    - `max_depth` : 트리가 성장할 최대 깊이를 지정합니다. 기본값은 `None`으로 리프 노드가 순수하거나 `min_samples_split`보다 샘플 개수가 적을 때까지 성장합니다.
    - `min_samples_split` : 노드를 나누기 위한 최소 샘플 개수입니다. 기본값은 2입니다.
    - `max_features` 매개변수 : 최적의 분할을 위해 탐색할 특성의 개수를 지정합니다. 기본값은 `None`으로 모든 특성을 사용합니다.
- **plot_tree()** 
    - 결정 트리 모델을 시각화합니다. 첫 번째 매개변수로 결정 트리 모델 객체를 전달합니다.
    - `max_depth` 매개변수: 나타낼 트리의 깊이를 지정합니다. 기본값은 `None`으로 모든 노드를 출력합니다.
    - `feature_names` 매개변수: 특성의 이름을 지정할 수 있습니다.
    - `filled` 매개변수: True로 지정하면 타깃값에 따라 노드 안에 색을 채웁니다.

## 로지스틱 회귀로 와인 분류하기

```python
import pandas as pd

wine = pd.read_csv('https://bit.ly/wine_csv_data')
```

- 판다스를 사용해 인터넷에서 직접 불러오겠습니다. 
- 다운로드할 주소는 https://bit.ly/wine_csv_data 입니다.
 
```python
wine.head()
```

- 와인 데이터셋을 판다스 데이터프레임으로 제대로 읽어 들였는지 `head()` 메서드로 처음 5개의 샘플을 확인해 보겠습니다.

![스크린샷 2024-11-05 오전 6 33 45](https://github.com/user-attachments/assets/46e8cf34-a196-4aec-b69c-3740c468f471)

- 처음 3개의 열(alcohol, sugar, pH)은 각각 알코올 도수, 당도, pH 값을 나타냅니다.
- 네 번째의 열(class)은 타깃값으로 0이면 레드 와인, 1이면 화이트 와인이라고 합니다. 레드 와인과 화이트 와인을 구분하는 이진 분류 문제이고, 화이트 와인이 양성 클래스입니다. 
- 즉 전체 와인 데이터에서 화이트 와인을 골라내는 문제

```python
wine.info()
```

- 이 메서드는 데이터프레임의 각 열의 데이터 타입과 누락된 데이터가 있는지 확인하는 데 유용합니다. 


![스크린샷 2024-11-05 오전 6 37 50](https://github.com/user-attachments/assets/5c15ae9c-919f-4924-8249-86f9585bf574)

- 출력 결과를 보면 총 6,497개의 샘플이 있고 4개의 열은 모두 실숫값입니다.
- Non-Null Count가 모두 6497이므로 누락된 값은 없습니다.


> 누락된 값이 있다면 그 데이터를 버리거나 평균값으로 채운 후 사용할 수 있습니다. 어떤 방식이 최선인지는 미리 알기 어렵습니다. 두 가지 모두 시도해 보세요. 여기에서도 항상 훈련 세트의 통계 값으로 테스트 세트를 변환해야 합니다. 즉, 훈련 세트의 평균값으로 테스트 세트의 누락된 값을 채워야 합니다.

```python
wine.describe()
```

- 이 메서드는 열에 대한 간략한 통계를 출력해 줍니다. 
- 최소, 최대, 평균값 등을 볼 수 있습니다.

![스크린샷 2024-11-05 오전 6 45 42](https://github.com/user-attachments/assets/1832b862-5d5c-4f07-91b7-f4554910f38f)

- 평균(mean), 표준편차(std), 최소(min), 최대(max)값, 중간값(50%), 1사분위수(25%), 3사분위수(75%)를 볼 수 있습니다. 
- 여기에서 알수 있는 것은 알코올 도수와 당도, pH 값의 스케일이 다르다는 것
- 사이킷런의 `StandardScaler` 클래스를 사용해 특성을 표준화합니다.

> 사분위수는 데이터를 순서대로 4등분 한 값입니다. 예를 들어 2사분위수(중간값)는 데이터를 일렬로 늘어놓았을 때 정중앙의 값입니다. 만약 데이터 개수가 짝수개라 중앙값을 선택할 수 없다면 가운데 2개 값의 평균을 사용합니다.

```python
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
```

- 판다스 데이터프레임을 넘파이 배열로 바꾸고 훈련 세트와 테스트 세트로 나누겠습니다. 
- wine 데이터프레임에서 처음 3개의 열을 넘파이 배열로 바꿔서 data 배열에 저장하고 마지막 class 열을 넘파이 배열로 바꿔서 target 배열에 저장합니다.

```python
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)
```

- 훈련 세트와 테스트 세트로 나눕니다.
- `train_test_split()` 함수는 설정값을 지정하지 않으면 25%를 테스트 세트로 지정합니다. 
- 샘플 개수가 충분히 많으므로 20% 정도만 테스트 세트로 나눴습니다. 코드의 `test_size=0.2`가 이런 의미 입니다. 


```python
print(train_input.shape, test_input.shape)
```

- 만들어진 훈련 세트와 테스트 세트의 크기를 확인합니다.

```
(5197, 3) (1300, 3)
```

- 훈련 세트는 5.197개이고 테스트 세트는 1,300개 입니다.

```python
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)

train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
```

- `StandardScaler` 클래스를 사용해 훈련 세트를 전처리합니다. 
- 그 다음 같은 객체를 그대로 사용해 테스트 세트를 변환하겠습니다.


```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train_scaled, train_target)

print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
```

- 표준점수로 변환된 `train_scaled` 와 `test_scaled` 를 사용해 로지스틱 회귀 모델을 훈련합니다.

```
0.7808350971714451
0.7776923076923077
```

- 훈련 세트와 테스트 세트의 점수가 모두 낮으니 모델이 다소 과소적합된 것 같습니다. 

### 설명하기 쉬운 모델과 어려운 모델

```python
print(lr.coef_, lr.intercept_)
```

```
[[ 0.51270274,  1.6733911,   -0.68767781]] [1.81777902]
```

## 결정 트리
- **결정 트리**(Decision Tree)모델은 이유를 설명하기 쉽습니다.
- 결정 트리 모델은 스무고개와 같이 질문을 하나씩 던져서 정답을 맞춰가는 것
- 데이터를 잘 나눌 수 있는 질문을 찾는다면 계속 질문을 추가해서 분류 정확도를 높일 수 있습니다.
- 사이킷런은 결정 트리 알고리즘을 제공합니다. **DecisionTreeClassifier** 클래스
- `fit()` 메서드를 호출해서 모델을 훈련한 다음 `score()` 메서드로 정확도를 평가합니다. 

![스크린샷 2024-11-05 오전 7 04 18](https://github.com/user-attachments/assets/4efae792-be2f-47d9-bb13-b94d1221e9da)

```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)

print(dt.score(train_scaled, train_target)) # 훈련 세트
print(dt.score(test_scaled, test_target))   # 테스트 세트
```

```
0.996921300750433
0.8592307692307692
```

- 훈련 세트에 대한 점수가 매우 높으나 테스트 세트의 성능은 그에 비해 조금 낮습니다. 과대적합된 모델이라고 볼 수 있습니다. 
- 이 모델을 그림으로 표현하려면 사이킷런의 `plot_tree()`함수를 사용해 결정 트리를 이해하기 쉬운 트리 그림으로 출력해 줍니다.




```python
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(10,7))
plot_tree(dt)
plt.show()
```

- 위에서 만든 결정 트리 모델 객체를 `plot_tree()` 함수에 전달해서 어떤 트리가 만들어졌는지 그려봅니다.




```python
plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()
```

## 가지치기

```python
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)

print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))
```

```
0.8454877814123533
0.8415384615384616
```

```python
plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()
```

```python
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)

print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))
```

```
0.8454877814123533
0.8415384615384616
```

```python
plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()
```

```python
print(dt.feature_importances_)
```

```
[0.12345626 0.86862934 0.0079144 ]
```
