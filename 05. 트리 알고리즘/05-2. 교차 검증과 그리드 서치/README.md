# 교차 검증과 그리드 서치

## 키워드 정리 

- **검증 세트** : 하이퍼파라미터 튜닝을 위해 모델을 평가할 때, 테스트 세트를 하용하지 않기 위해 훈련 세트에서 다시 떼어 낸 데이터 세트입니다.
- **교차 검증**
    - 훈련 세트를 여러 폴드로 나눈 다음 한 폴드가 검증 세트의 역할을 하고 나머지 폴드에서는 모델을 훈련합니다.
    - 교차 점증은 이런 식으로 모든 폴드에 대해 검증 점수를 얻어 평균하는 방법입니다.
- **그리드 서치**
    - 하이퍼파라미터 탐색을 자동화해 주는 도구입니다.
    - 탐색할 매개변수를 나열하면 교차 검증을 수행하여 가장 좋은 검증 점수의 매개변수 조합을 선택합니다. 
    - 마지막으로 이 매개변수 조합으로 최종 모델을 훈련합니다.
- **랜덤 서치**
    - 연속된 매개변수 값을 탐색할 때 유용합니다. 
    - 탐색할 값을 직접 나열하는 것이 아니고 탐색 값을 샘플링할 수 있는 확률 분포 객체를 전달합니다. 
    - 지정된 횟수만큼 샘플링하여 교차 검증을 수행하기 때문에 시스템 자원이 허락하는 만큼 탐색량을 조절할 수 있습니다.

# 검증 세트

```python
import pandas as pd

wine = pd.read_csv('https://bit.ly/wine_csv_data')
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
sub_input, val_input, sub_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)
```

```python
print(sub_input.shape, val_input.shape)
```

```
(4157, 3) (1040, 3)
```

```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input, sub_target)

print(dt.score(sub_input, sub_target))
print(dt.score(val_input, val_target))
```

```
0.9971133028626413
0.864423076923077
```

## 교차 검증

```python
from sklearn.model_selection import cross_validate

scores = cross_validate(dt, train_input, train_target)
print(scores)
```

```
{'fit_time': array([0.01341891, 0.02167416, 0.02525187, 0.04882073, 0.03598666]), 'score_time': array([0.0027864 , 0.0019815 , 0.00886154, 0.01437068, 0.02624893]), 'test_score': array([0.86923077, 0.84615385, 0.87680462, 0.84889317, 0.83541867])}
```

```python
import numpy as np

print(np.mean(scores['test_score']))
```

```
0.855300214703487
```

```python
from sklearn.model_selection import StratifiedKFold

scores = cross_validate(dt, train_input, train_target, cv=StratifiedKFold())
print(np.mean(scores['test_score']))
```

```
0.855300214703487
```

```python
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splitter)
print(np.mean(scores['test_score']))
```

```
0.8574181117533719
```

## 하이퍼파라미터 튜닝

```python
from sklearn.model_selection import GridSearchCV

params = {'min_impurity_decrease': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
```

```python
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
```

```python
gs.fit(train_input, train_target)
```

```python
dt = gs.best_estimator_
print(dt.score(train_input, train_target))
```

```
0.9615162593804117
```

```python
print(gs.best_params_)
```

```
{'min_impurity_decrease': 0.0001}
```

```python
print(gs.cv_results_['mean_test_score'])
```

```
[0.86819297 0.86453617 0.86492226 0.86780891 0.86761605]
```

```python
best_index = np.argmax(gs.cv_results_['mean_test_score'])
print(gs.cv_results_['params'][best_index])
```

```
{'min_impurity_decrease': 0.0001}
```

```python
params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001),
          'max_depth': range(5, 20, 1),
          'min_samples_split': range(2, 100, 10)
          }
```

```python
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
```

```python
print(gs.best_params_)
```

```
{'max_depth': 14, 'min_impurity_decrease': 0.0004, 'min_samples_split': 12}
```

```python
print(np.max(gs.cv_results_['mean_test_score']))
```

```
0.8683865773302731
```

## 랜덤 서치
