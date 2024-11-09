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

> 보통 20\~30%를 테스트 세트와 검증 세트로 떼어 놓습니다. 하지만 문제에 따라 다릅니다. 훈련 데이터가 아주 많다면 단 몇 %만 떼어 놓아도 전체 데이터를 대표하는 데 문제가 없습니다.

- 훈련 세트에서 모델을 훈련하고 검증 세트로 모델을 평가합니다. 이런 식으로 테스트 하고 싶은 매개변수를 바꿔가며 가장 좋은 모델을 고릅니다. 
- 미 매개변수를 사용해 훈련 세트와 검증 세트를 합쳐 전체 훈련 데이터에서 모델을 다시 훈련합니다. 
- 마지막에 테스트 세트에서 최종 점수를 평가합니다. 실전에 투입했을 때 테스트 세트의 점수와 비슷한 성능을 기대할 수 있을 것입니다. 

```python
import pandas as pd

wine = pd.read_csv('https://bit.ly/wine_csv_data')
```

- 데이터를 다시 불러와서 검증 세트를 만들어 봅니다.
- 판다스로 CSV 데이터를 읽습니다.


```python
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
```

- `class` 열을 타깃으로 사용하고 나머지 열은 특성 배열에 저장합니다. 


```python
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)
```

- 훈련 세트와 테스트 세트를 나눕니다. 
- 훈련 세트의 입력 데이터와 타깃 데이터를 `train_input`과 `train_target` 배열에 저장

```python
sub_input, val_input, sub_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)
```

- `train_input` 과 `train_target` 을 다시 `train_test_split()` 함수에 넣어 훈련 세트 `sub_input`, `sub_target`과 검증 세트 `val_input`, `val_target`을 만듭ㄴ다. 
- test_size 매개변수를 0.2로 지정하여 `train_input` 의 약 20%를 `val_input`으로 만듭니다.


```python
print(sub_input.shape, val_input.shape)
```

- 단순히 `train_test_split()` 함수를 2번 적용해서 훈련 세트와 검증 세트로 나눠준 것
- 훈련 세트와 검증 세트의 크기를 확인

```
(4157, 3) (1040, 3)
```

- 원래 5,197개 였던 훈련 세트가 4,157개로 줄고, 검증 세트는 1,040개가 되었습니다. 
- `sub_input`, `sub_target`과 `val_input`, `val_target`을 사용해 검증 모델을 만들고 평가합니다.


```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input, sub_target)

print(dt.score(sub_input, sub_target))
print(dt.score(val_input, val_target))
```

- `sub_input`, `sub_target`과 `val_input`, `val_target`을 사용해 모델을 만들고 평가합니다.  

```
0.9971133028626413
0.864423076923077
```

- 이 모델은 훈련 세트에 과대적합되어 있습니다.
- 매개변수를 바꿔서 더 좋은 모델을 찾아야 합니다. 


## 교차 검증

- 검증 세트를 만드느라 훈련 세트가 줄었습니다. 보통 많은 데이터를 훈련에 사용할수록 좋은 모델이 만들어집니다. 그렇다고 검증 세트를 너무 조금 떼어 놓으면 검증 점수가 들쭉날쭉하고 불안정할 것입니다. 이럴 때 **교차 검증**(cross validation)을 이용하면 안정적인 검증 점수를 얻고 훈련에 더 많은 데이터를 사용할 수 있습니다. 
- 교차 검증은 검증 세트를 떼어 내어 평가하는 과정을 여러 번 반복합니다. 그 다음 이 점수를 평균하여 최종 검증 점수를 얻습니다. 
- 3-폴드 교차 검증

![스크린샷 2024-11-07 오전 6 43 11](https://github.com/user-attachments/assets/1f90224b-406a-48ef-8112-b2b1093e01a6)

> 3-폴드 교차 검증
>
> 훈련 세트를 세 부분으로 나눠서 교차 검증을 수행하는 것을 3-폴드 교차 검증이라고 합니다. 통칭 k-폴드 교차 검증(k-fold cross validation)이라고 하며, 훈련 세트를 몇 부분으로 나누냐에 따라 다르게 부릅니다. k-겹 교차 검증이라고도 부릅니다.

- 보통 5-폴드 교차 검증이나 10-폴드 교차 검증을 많이 사용합니다. 
- 이렇게 하면 데이터가 80\~90% 까지 훈련에 사용할 수 있습니다. 
- 검증 세트가 줄어들지만 각 폴드에서 계산한 검증 점수를 평균하기 때문에 안정된 점수로 생각할 수 있습니다.

```python
from sklearn.model_selection import cross_validate

scores = cross_validate(dt, train_input, train_target)
print(scores)
```

- 사이킷런에는 `cross_validate()`라는 교차 검증 함수가 있습니다. 
- 먼저 평가할 모델 객체를 첫 번째 매개변수로 전달합니다. 그 다음 앞에서처럼 직접 검증 세트를 떼어 내지 않고 훈련 세트 전체를 `cross_validate()` 함수에 전달합니다. 

```
{'fit_time': array([0.01341891, 0.02167416, 0.02525187, 0.04882073, 0.03598666]), 'score_time': array([0.0027864 , 0.0019815 , 0.00886154, 0.01437068, 0.02624893]), 'test_score': array([0.86923077, 0.84615385, 0.87680462, 0.84889317, 0.83541867])}
```

- `fit_time`: 모델을 훈련하는 시간
- `score_time`: 검증하는 시간
- 각 키마다 5개의 숫자가 담겨 있습니다. `cross_validate()` 함수는 기본적으로 **5-폴드 교차 검증**을 수행합니다.
- cv 매개변수에서 폴드 수를 바꿀 수도 있습니다. 

```python
import numpy as np

print(np.mean(scores['test_score']))
```

- 교차 검증의 최종 점수는 `test_score` 키에 담긴 5개의 점수를 평균하여 얻을 수 있습니다. 
- 이름은 `test_score` 지만 검증 폴드의 점수입니다. 

```
0.855300214703487
```
- 교차 검증을 수행하면 입력한 모델에서 얻을 수 있는 최상의 검증 점수를 예상해 볼 수 있습니다.
- 주의할 점은 `cross_validate()` 는 훈련 세트를 섞어 폴드를 나누지 않습니다. 앞서 `train_test_split()` 함수로 전체 데이터를 섞은 후 훈련 세트를 준비했기 때문에 따로 섞을 필요가 없습니다. 
- 만약 교차 검증을 할 때 훈련 세트를 섞으려면 분할기(splitter)를 지정해야 합니다.


```python
from sklearn.model_selection import StratifiedKFold

scores = cross_validate(dt, train_input, train_target, cv=StratifiedKFold())
print(np.mean(scores['test_score']))
```

- 사이킷런의 분할기는 교차 검증에서 폴드를 어떻게 나눌지 결정해 줍니다. 
- `cross_validate()` 함수는 기본적으로 회귀 모델일 경우 **KFold** 분할기를 사용하고 분류 모델일 경우 타깃 클래스를 골고루 나누기 위해 **StratifiedKFold**를 사용합니다. 


```
0.855300214703487
```


```python
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splitter)
print(np.mean(scores['test_score']))
```
- 10-폴드 교차 검증 수행은 위 코드와 같습니다.
- `n_splits` 매개변수는 몇(k) 폴드 교차 검증을 할지 정합니다.


```
0.8574181117533719
```

## 하이퍼파라미터 튜닝

- **모델 파라미터** : 머신러닝 모델이 학습하는 파라미터
- **하이퍼파라미터** : 모델이 학습할 수 없어서 사용자가 지정해야만 하는 파라미터
- 사이킷런과 같은 머신러닝 라이브러리를 사용할 때 이런 하이퍼파라미터는 모두 클래스나 메서드의 매개변수로 표현된다.
- 하이퍼파라미터를 튜닝하는 작업은 먼저 라이브러리가 제공하는 기본값을 그대로 사용해 모델을 훈련합니다. 그 다음 검증 세트의 점수나 교차 검증을 통해서 매개변수를 조금씩 바꿔 봅니다. 
- 모델마다 적게는 1\~2개에서, 많게는 5\~6개의 매개변수를 제공합니다. 이 매개변수를 바꿔가면서 모델을 훈련하고 교차검증을 수행합니다.

> 사람의 개입 없이 하이퍼파라미터 튜닝을 자동으로 수행하는 기술을 `AutoML` 이라고 부릅니다. 

- 결정 트리 모델에서 `max_depth` 의 최적값은 `min_samples_split` 매개변수의 값이 바뀌면 함께 달라집니다. 
- 즉, 이 두 매개변수를 동시에 바꿔가며 최적의 값을 찾아야 합니다. 
- 매개변수가 많아지만 문제는 복잡해 집니다. 그래서 이미 만들어진 도구인 사이킷런에서 제공하는 **그리드 서치**(Grid Search)를 사용합니다. 


```python
from sklearn.model_selection import GridSearchCV

params = {'min_impurity_decrease': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
```

- 사이킷런의 GridSearchCV 클래스는 친절하게도 하이퍼파라미터 탐색과 교차 검증을 한 번에 수행합니다. 별도로 `cross_validate()` 함수를 호출할 필요가 없습니다. 
- 기본 매개변수를 사용한 결정 트리 모델에서 `min_impurity_decrease` 매개변수의 최적값을 찾아봅니다. 
- 먼저 **GridSearchCV** 클래스를 임포트하고 탐색할 매개변수와 탐색할 값의 리스트를 딕셔너리로 만듭니다. 

```python
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
```
- 0.0001 부터 0.0005까지 0.0001씩 증가하는 5개의 값을 시도합니다. 
- **GridSearchCV** 클래스에 탐색 대상 모델과 `params` 변수를 전달하여 그리드 서치 객체를 만듭니다. 

```python
gs.fit(train_input, train_target)
```

- 일반 모델을 훈련하는 것처럼 `gs` 객체에 `fit()` 메서드를 호출합니다. 
- 이 메서드를 호출하면 그리드 서치 객체는 결정 트리 모델 `min_impurity_decrease` 값을 바꿔가며 총 5번 실행합니다.
- **GridSearchCV**의 cv 매개변수 기본값은 5입니다. 따라서 `min_impurity_decrease` 값마다 5-폴드 교차 검증을 수행합니다. 결국 5 X 5 = 25개의 모델을 훈련합니다. 
- 많은 모델을 훈련하기 때문에 **GridSearchCV** 클래스의 `n_jobs` 매개변수에서 병렬 실해에 사용할 CPU 코어 수를 지정하는 것이 좋습니다. 
- 이 매개변수의 기본값은 1입니다. -1로 지정하면 시스템에 있는 모든 코어를 사용합니다.

```python
dt = gs.best_estimator_
print(dt.score(train_input, train_target))
```

- 교차 검증에서 최적의 하이퍼파라미터를 찾으면 전체 훈련 세트로 모델을 다시 만들어야 합니다. 
- 편리하게도 사이킷런의 그리드 서치는 훈련이 끝나면 25개의 모델 중에서 검증 점수가 가장 높은 모델의 매개변수 조합으로 전체 훈련 세트에서 자동으로 다시 모델을 훈련합니다. 
- 이 모델은 `gs` 객체의 `best_estimator_` 속성에 저장되어 있습니다. 이 모델을 일반 결정 트리처럼 똑같이 사용할 수 있습니다.

```
0.9615162593804117
```

```python
print(gs.best_params_)
```
- 그리드 서치로 찾은 최적의 매개변수는 `best_params_` 속성에 저장되어 있습니다. 

```
{'min_impurity_decrease': 0.0001}
```

- 여기에서는 0.0001이 가장 좋은 값으로 선택되었습니다. 

```python
print(gs.cv_results_['mean_test_score'])
```

- 각 매개변수에서 수행한 교차 검증의 평균 점수는 `cv_results_` 속성의 `mean_test_score` 키에 저장되어 있습니다.
- 5번의 교차 검증으로 얻은 점수를 출력해 봅시다.

```
[0.86819297 0.86453617 0.86492226 0.86780891 0.86761605]
```
- 첫 번쨰 값이 가장 큽니다. 

```python
best_index = np.argmax(gs.cv_results_['mean_test_score'])
print(gs.cv_results_['params'][best_index])
```

- 넘파이 `argmax()`함수를 사용하면 가장 큰 값의 인덱스를 추출할 수 있습니다. 
- 그다음 이 인덱스를 사용해 `params` 키에 저장된 매개변수를 출력할 수 있습니다. 이 값이 최상의 검증 점수를 만든 매개변수 조합입니다. 


```
{'min_impurity_decrease': 0.0001}
```

- 앞서 출력한 `gs.best_params`와 동일한지 확인해 보세요.

- 이 과정을 정리해 보면
- 1 먼저 탐색할 매개변수를 지정합니다.
- 2 훈련 세트에서 그리드 서치를 수행하여 최상의 평균 검증 점수가 나오는 매개변수 조합을 찾습니다. 이 조합은 그리드 서치 객체에 저장됩니다.
- 3 그리드 서치는 최상의 매개변수에서 (교차 검증에 사용한 훈련 세트가 아니라) 전체 훈련 세트를 사용해 최종 모델을 훈련합니다. 이 모델도 그리드 서치 객체에 저장됩니다. 


```python
params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001),
          'max_depth': range(5, 20, 1),
          'min_samples_split': range(2, 100, 10)
          }
```

- 노드를 분할하기 위한 불순도 감소 최소량을 지정합니다. 여기에서 max_depth로 트리의 깊이를 제한하고 min_samples_split로 노드를 나누기 위한 최소 샘플 수도 골라봅니다.


```python
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
```

- 이 매개변수로 수행할 교차 검증 횟수는 9 X 15 X 10 = 1,350개 입니다. 기본 5-폴드 교차 검증을 수행하므로 만들어지는 모델의 수는 6,750개나 됩니다. 
- n_jobs 매개변수를 -1로 설정하고 그리드 서치를 수행합니다.

```python
print(gs.best_params_)
```

- 최상의 매개변수 조합을 확인해 보면 

```
{'max_depth': 14, 'min_impurity_decrease': 0.0004, 'min_samples_split': 12}
```

```python
print(np.max(gs.cv_results_['mean_test_score']))
```

- 최상의 교차 검증 점수도 확인합니다.

```
0.8683865773302731
```

- **GridSearchCV** 클래스를 사용하면 매개변수를 일일이 바꿔가며 교차검증을 수행하지 않고 원하는 매개변수 값을 나열하면 자동으로 교차 검증을 수행해서 최상의 매개변수를 찾을 수 있습니다, 
- 조금 아쉬운 점은 탐색할 매개변수의 간격을 0.0001 혹은 1로 설정했는데, 이렇게 간격을 둔 것에 특별한 근거는 없습니다. 이보다 더 좁거나 넓은 간격으로 시도해 볼수도 있습니다.

## 랜덤 서치

- 매개변수의 값이 수치일 떄 값의 범위나 간격을 미리 정하기 어려울 수 있습니다. 또 너무 많은 매개변수 조건이 있어 그리드 서치 수행 시간이 오래 걸릴 수 있습니다. 이럴 떄 **랜덤 서치**(Random Search)를 사용할 수 있습니다.
- 랜덤 서치에는 매개변수 값의 목록을 전달하는 것이 아니라 매개변수를 샘플링할 수 있는 확률 분포 객체를 전달합니다. 
- 사이 파이에서 2개의 확률 분포 클래스를 임포트 합니다.


```python
from scipy.stats import uniform, randint
```

```python
rgen = randint(0, 10)
rgen.rvs(10)
```

```
array([6, 4, 3, 6, 3, 1, 1, 6, 1, 6])
```

```python
np.unique(rgen.rvs(1000), return_counts=True)
```

```
(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
 array([121,  99,  81,  98, 106,  85,  93,  94, 109, 114]))
```

```python
ugen = uniform(0, 1)
ugen.rvs(10)
```

```
array([0.60857829, 0.2795936 , 0.4059522 , 0.47695652, 0.7427586 ,
       0.9801252 , 0.05012329, 0.79357074, 0.16195204, 0.33820475])
```

```python
params = {'min_impurity_decrease': uniform(0.0001, 0.001),
          'max_depth': randint(20, 50),
          'min_samples_split': randint(2, 25),
          'min_samples_leaf': randint(1, 25),
          }
```


```python
from sklearn.model_selection import RandomizedSearchCV

gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params,
                        n_iter=100, n_jobs=-1, random_state=42)
gs.fit(train_input, train_target)
```

```python
print(gs.best_params_)
```

```
{'max_depth': 39, 'min_impurity_decrease': 0.00034102546602601173, 'min_samples_leaf': 7, 'min_samples_split': 13}
```

```python
print(np.max(gs.cv_results_['mean_test_score']))
```

```
0.8695428296438884
```

```python
dt = gs.best_estimator_

print(dt.score(test_input, test_target))
```

```
0.86
```