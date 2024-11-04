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

- 와인 데이터셋을 판다스 데이터프레임으로 제대로 읽어 들였는지 `head()` 메서드로 처음 5개의 샘플을 확인해 보겠습니다..

```python
wine.info()
```

```python
wine.describe()
```

```python
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
```

```python
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)
```

```python
print(train_input.shape, test_input.shape)
```

```
(5197, 3) (1300, 3)
```

```python
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)

train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
```

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train_scaled, train_target)

print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
```

```
0.7808350971714451
0.7776923076923077
```

```python
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(10,7))
plot_tree(dt)
plt.show()
```

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
