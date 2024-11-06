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

## 핵심 패키지와 함수

### scikit-learn

- **cross_validate()**
  - 교차 검증을 수행하는 함수입니다.
  - 첫 번째 매개변수에 교차 검증을 수행할 모델 객체를 전달합니다.
  - 두 번째와 세 번째 매개변수에 특성과 타깃 데이터를 전달합니다.
  - `scoring` 매개변수에 검증에 사용할 평가 지표를 지정할 수 있습니다. 기본적으로 분류 모델은 정확도를 의미하는 `accuracy`, 회귀 모델은 결정 계수를 의미하는 `r2` 가 됩니다.
  - `cv` 매개변수에 교차 검증 폴드 수나 스플리터 객체를 지정할 수 있습니다. 기본값은 5입니다.
  - 회귀일 때는 `KFold` 클래스를 사용하고 분류일 떄는 `StratifiedKFold` 클래스를 사용하여 `5-폴드 교차 검증`을 수행합니다.
  - `n_jobs` 매개변수는 교차 검증을 수행할 때 사용할 CPU 코어 수를 지정합니다. 기본값은 1로 하나의 코어를 사용합니다. -1로 지정하면 시스템에 있는 모든 코어를 사용합니다.
- **GridSearchCV**
  - 교차 검증으로 하이퍼파라미터 탐색을 수행합니다.
  - 최상의 모델을 찾은 후 훈련 세트 전체를 사용해 최종 모델을 훈련합니다.
  - 첫 번째 매개변수로 그리드 서치를 수행할 모델 객체를 전달합니다. 두 번째 매개변수에는 탐색할 모델의 매개변수의 값을 전달합니다.
  - `scoring`, `n_jobs`, `return_train_score` 매개변수는 `cross_validate()` 함수와 동일합니다.
- **RandomizedSearchCV**
  - 교차 검증으로 랜덤한 하이퍼파라미터 탐색을 수행합니다.
  - 최상의 모델을 찾은 후 훈련 세트 전체를 사용해 최종 모델을 훈련합니다.
  - 첫 번째 매개변수로 그리드 서치를 수행할 모델 객체를 전달합니다. 두 번째 매개변수에는 탐색할 모델의 매개변수와 확률 분포 객체를 전달합니다.
  - `scoring`, `cv`, `n_jobs`, `return_train_score` 매개변수는 `cross_validate()` 함수와 동일합니다.

# 검증 세트

- 테스트 세트를 사용하지 않으면 모델이 과대적합인지 과소적합인지 판단하기 어렵습니다. 
- 테스트 세트를 사용하지 않고 이를 측정하는 간단한 방법은 훈련세트를 또 나눈 것 입니다. 이 데이터를 **검증 세트**(validation set)라고 부릅니다. 
- 전체 데이터 중 20%를 테스트 세트로 만들고 나머지 80%를 훈련 세트로 만들었습니다. 이 훈련 세트 중에서 다시 20%를 떼어 내어 검증 세트로 만듭니다. 

![스크린샷 2024-11-07 오전 6 26 51](https://github.com/user-attachments/assets/214c58eb-29d2-40db-86a0-e5704531193f)


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
