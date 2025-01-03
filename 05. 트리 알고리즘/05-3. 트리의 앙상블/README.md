# 트리의 앙상블

## 키워드로 정리

- **앙상블 학습**
    - 더 좋은 예측 결과를 만들기 위해 여러 개의 모델을 훈련하는 머신러닝 알고리즘을 말합니다.
- **랜덤 포레스트**
    - 대표적인 결정 트리 기반의 앙상블 학습 방법입니다.
    - 부트스트랩 샘플을 사용하고 랜덤하게 일부 특성을 선택하여 트리를 만드는 것이 특징 입니다.
- **엑스트라 트리**
    - 랜덤 포레스트와 비슷하게 결정 트리를 사용하여 앙상블 모델을 만들지만 부트스트랩 샘플을 사용하지 않습니다. 
    - 대신 랜덤하게 노드를 분할해 과대적합을 감소시킵니다.
- **그레디언트 부스팅**
    - 랜덤 포레스트나 엑스트라 트리와 달리 결정 트리를 연속적으로 추가하여 손실 함수를 최소화하는 앙상블 방법입니다. 
    - 훈련 속도가 조금 느리지만 더 좋은 성능을 기대할 수 있습니다. 
    - 그레이디언트 부스팅의 속도를 개선한 것이 **히스토그램 기반 그레디언트 부스팅**이며 안정적인 결과와 높은 성능으로 매우 인기가 높습니다.

## 핵심 패키지와 함수
### scikit-learn
- **RandomForestClassifier**
    - 랜덤 포레스트 분류 클래스입니다.
    - `n_estimiators` 매개변수는 앙상블을 구성할 트리의 개수를 지정합니다. 기본값은 100입니다.
    - `criterion` 매개변수는 불순도를 지정하며 기본값은 지니 불순도를 의미하는 `gini` 이고 `entropy`를 선택하여 엔트로피 불순도를 사용할 수 있습니다.
    - `max_depth`는 트리가 성장할 최대 깊이를 지정합니다. 기본값은 `None`으로 지정하면 리프노드가 순수하거나 `min_samples_split`보다 샘플 개수가 적을 때까지 성장합니다.
    - `min_samples_split`는 노드를 나누기 위한 최소 샘플 개수 입니다. 기본값은 2입니다. 
    - `max_feature` 매개변수는 최적의 분할을 위해 탐색할 특성의 개수를 지정합니다. 기본값은 `auto`로 특성 개수의 제곱근입니다.
    - `bootstrap` 매개변수는 부트스트랩 샘플을 사용할지 지정합니다. 기본값은 `True` 입니다.
    - `oob_score`는 `OOB` 샘플을 사용하여 훈련한 모델을 평가할지 지정합니다. 기본값은 `False` 입니다. 
    - `n_jobs` 매개변수는 병렬 실행에 사용할 `CPU` 코어 수를 지정합니다. 기본값은 1로 하나의 코어를 사용합니다. -1로 지정하면 시스템에 있는 모든 코어를 사용합니다.
- **ExtraTreesClassifier**
    - 엑스트라 트리 분류 클래스입니다.
    - n_estimators, criterion, max_depth, min_samples_split, max_features 매개변수는 랜덤 포레스트와 동일합니다.
    - `bootstrap` 매개변수는 부트스트랩 샘플을 사용할지 지정합니다. 기본값은 `False`입니다. 
    - `oob_score`는 `OOB` 샘플을 사용하여 훈련한 모델을 평가할지 지정합니다. 기본값은 `False` 입니다. 
    - `n_jobs` 매개변수는 병렬 실행에 사용할 CPU 코어 수를 지정합니다. 기본값은 1로 하나의 코어를 사용합니다. -1로 지정하면 시스템에 있는 모든 코어를 사용합니다.
- **GradientBoostingClassifier**
    - 그레이디언트 부스팅 분류 클래스입니다.
    - `loss` 매개변수는 손실 함수를 지정합니다. 기본값은 로지스틱 손실 함수를 의미하는 `deviance`입니다. 
    - `learning_rate` 매개변수는 트리가 앙상블에 기여하는 정도를 조절합니다. 기본값은 0.1 입니다.
    - `n_estimators` 매개변수는 부스팅 단계를 수행하는 트리의 개수입니다. 기본값은 100입니다.
    - `subsample` 매개변수는 사용할 훈련 세트의 샘플 비율을 지정합니다. 기본값은 1.0입니다.
    - `max_depth` 매개변수는 개별 회귀 트리의 최대 깊이입니다. 기본값은 3입니다.
- **HistGradientBoostingClassifier**
    - 히스토그램 기반 그레디언트 부스팅 분류 클래스 입니다.
    - `learning_rate` 매개변수는 학습률 또는 감쇠율이라고 합니다. 기본값은 0.1이며 1.0이면 감쇠가 전혀 없습니다.
    - `max_iter`는 부스팅 단계를 수행하는 트리의 개수입니다. 기본값은 100입니다.
    - `max_bins`는 입력 데이터를 나눌 구간의 개수입니다. 기본값은 255이며 이보다 크게 지정할 수 없습니다. 여기에 1개의 구간이 누락된 값을 위해 추가됩니다.


## 정형 데이터와 비정형 데이터
- **정형 데이터** 
    - 어떤 구조로 되어 있는 데이터
    - 이런 데이터는 CSV나 데이터베이스(Database), 혹은 엑셀(Excel)에 저장하기 쉽습니다.
    - 프로그래머가 다루는 대부분의 데이터가 정형 데이터 입니다.
    - 지금까지 배운 머신러닝 알고리즘은 정형 데이터에 잘 맞습니다.
    - 그 중 정형 데이터를 다루는데 가장 뛰어난 성과를 내는 알고리즘이 **앙상블 학습**(ensemble learning)입니다. 
    - 이 알고리즘은 대부분 결정 트리를 기반으로 만들어져 있습니다. 

- **비정형 데이터**
    - 데이터베이스나 엑셀로 표현하기 어려운 것들
    - 책의 글과 같은 텍스트 데이터, 디지털카메라로 찍은 사진, 핸드폰으로 듣는 디지털 음악 등
    - 비정형 데이터는 규칙성을 찾기 어려워 전통적인 머신러닝 방법으로는 모델을 만들기 까다롭습니다.
    - 신경망 알고리즘의 놀라운 발전 덕분에 사진을 인식하고 텍스트를 이해하는 모델을 만들 수 있습니다.
 
## 랜덤 포레스트
- 앙상플 학습의 대표 주자, 안정적인 성능으로 널리 사용
- 결정 트리를 랜덤하게 만들어 결정 트리(나무)의 **숲**을 만듭니다. 그리고 각 결정 트리의 예측을 사용해 최종 예측을 만듭니다.

<img width="288" alt="스크린샷 2024-11-09 오후 8 00 21" src="https://github.com/user-attachments/assets/8dd56eaf-fa6c-4077-94c1-a8017ffdcb90">

- 랜덤 포레스트는 각 트리를 훈련하기 위한 데이터를 독특한 방법으로 랜덤하게 만듭니다. 
- 입력한 훈련 데이터에서 랜덤하게 샘플을 추출하여 훈련데이터를 만들며 이 때 한 샘플이 중복되어 추출될 수도 있습니다.
- 예를 들어 1,000개의 샘플이 들어 있는 가방에서 100개의 샘플을 뽑는다면 먼저 1개를 뽑고, 뽑았던 1개를 다시 가방에 넣습니다. 이런식으로 계속해서 100개를 가방에서 뽑으면 중복된 샘플을 뽑을 수 있습니다. 
- 이렇게 만들어진 샘플을 **부트스트랩 샘플**(bootstrap sample)이라고 부릅니다. 
- 기본적으로 부트스트랩 샘플은 훈련 세트의 크기와 같게 만듭니다. 1,000개의 샘플이 들어있는 가방에서 중복하여 1,000개의 샘플을 뽑습니다.


<img width="496" alt="스크린샷 2024-11-09 오후 8 10 08" src="https://github.com/user-attachments/assets/7c421256-8a1a-4152-8e4f-d895dba3e1d1">

> 부트스트랩이란?
>
> 보통 부트스트랩 방식이라고 하는데, 데이터 세트에서 중복을 허용하여 데이터를 샘플링하는 방식을 의미합니다. 본문에서 설명한 것처럼 가방에 1,000개의 샘플이 있을 때 먼저 1개를 뽑고, 다시 가방에 넣어 그다음 샘플을 뽑는 방식을 뜻합니다. 부트스트랩 샘플이란 결국 부트스트랩 방식으로 샘플링하여 분류한 데이터라는 의미입니다.

- 각 노드를 분할할 때 전체 특성 중에서 일부 특성을 무작위로 고른 다음 이 중에서 최선의 분할을 찾습니다. 분류 모델인 **RandomForestClassifier**는 기본적으로 전체 특성 개수의 제곱근만큼의 특성을 선택합니다. 즉 4개의 특성이 있다면 노드마다 2개를 랜덤하게 선택하여 사용합니다. 다만 회귀 모델인 **RandomForestRegressor**는 전체 특성을 사용합니다.

<img width="430" alt="스크린샷 2024-11-09 오후 8 09 46" src="https://github.com/user-attachments/assets/5dce5b95-a317-4f77-ab01-2308f384634a">


- 사이킷런의 렌덤 포레스트는 기본적으로 100개의 결정 트리를 이런 방식으로 훈련합니다. 그다음 분류일 때는 각 트리의 클래스별 확률을 평균하여 가장 높은 확률을 가진 클래스를 예측으로 삼습니다. 
- 회귀일 때는 단순히 각 트리의 예측을 평균합니다.
- 랜덤 포레스트는 랜덤하게 선택한 샘플과 특성을 사용하기 때문에 훈련 세트에 과대적합되는 것을 막아주고 검증 세트와 테스트 세트에서 안정적인 성능을 얻을 수 있습니다.
- 기본 매개변수 설정만으로도 아주 좋은 결과를 냅니다.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)
```

- **RandomForestClassifer** 클래스를 화이트 와인을 분류하는 문제에 적용해 봅니다. 
- 와인 데이터셋을 판다스로 불러오고 훈련 세트와 테스트 세트로 나눕니다.

```python
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```

- `cross_validate()` 함수를 사용해 교차 검증을 수행하비다.
- **RandomForestClassifier**는 기본적으로 100개의 결정 트리를 사용하므로 `n_jobs` 매개변수를 -1로 지정하여 모든 CPU 코어를 사용하는 것이 좋습니다. 
- `cross_validate()` 함수의 n_jobs 매개변수도 -1로 지정하여 최대한 병렬로 교차 검증을 수행합니다. 
- `return_train_score` 매개변수의 기본값은 False입니다.

```
0.9973541965122431 0.8905151032797809
```

- 출력된 결과를 보면 훈련 세트에 다소 과대적합된 것 같습니다. 

```python
rf.fit(train_input, train_target)
print(rf.feature_importances_)
```

- 랜덤 포레스트는 결정 트리의 앙상블이기 때문에 **DecisionTreeClassifier** 가 제공하는 중요한 매개변수를 모두 제공
- criterion, max_depth, max_features, min_sample_split, min_impurity_decrease, min_samples_leaf 등입니다. 
- 또한 결정 트리의 큰 장점 중 하나인 특성 중요도를 계산합니다. 랜덤 포레스트의 중요도는 각 결정 트리의 특성 중요도를 취한한 것입니다.
- 랜덤 포레스트 모델을 훈련 세트에 훈련한 후 특성 중요도를 출력합니다.

```
[0.23167441 0.50039841 0.26792718]
```

- 각각 \[알코올 도수, 당도, pH\]였는데, 두 번째 특성인 당도의 중요도가 감소하고 알코올 도수와 pH 특성의 중요도가 조금 상승했습니다. 이런 이유는 랜덤 포레스트가 특성의 일부를 랜덤하게 선택하여 결정트리를 훈련하기 때문입니다. 그 결과 하나의 특성이 훈련에 기여할 기회를 얻습니다. 이는 과대적합을 줄이고 일반화 성능을 높이는데 도움이 됩니다.



```python
rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)

rf.fit(train_input, train_target)
print(rf.oob_score_)
```

- **RandomForestClassifier**에는 자체적으로 모델을 평가하는 점수를 얻을 수 있습니다. 랜덤 포레스트는 훈련 세트에서 중복을 허용하여 부트스트랩 샘플을 만들어 결정 트리를 훈련한다고 했습니다. 이때 부트스트랩 샘플에 포함되지 않고 남는 샘플이 있습니다. 이런 샘플을 **OOB**(out of bag) 샘플이라고 합니다. 이 남은 샘플을 사용하여 부트스트랩 샘플로 훈련한 결정 트리를 평가할 수 있습니다. 마치 검증 세트의 역할을 하는 것
- 이 점수를 얻으려면 **RandomForestClassifier** 클래스의 `oob_score` 매개변수를 True로 지정해야 합니다(이 매개변수의 기본값은 False 입니다). 이렇게 하면 랜덤 포레스트는 각 결정 트리의 OOB 점수를 평균하여 출력합니다. `oob_score=True`로 지정하고 모델을 훈련하여 OOB 점수를 출려합니다.


```
0.8934000384837406
```

- 교차 검증에서 얻은 점수와 매우 비슷한 결고를 얻었습니다. 
- **OOB** 점수르 사용하면 교차 검증을 대신할 수 있어서 결과적으로 훈련 세트에 더 많은 샘플을 사용할 수 있습니다. 

## 엑스트라트리

- 랜덤 포레스트와 매우 비슷하게 동작합니다. 기본적으로 100개의 결정 트리를 훈련합니다. 
- 랜덤 포레스트와 동일하게 결정 트리가 제공하는 대부분의 매개변수를 지원합니다. 
- 전체 특성 중에 일부 특성을 랜덤하게 선택하여 노드를 분할하는 데 사용합니다. 
- 랜덤 포레스트와 엑스트 트리의 차이점은 부트스트랩 샘플을 사용하지 않는다는 점, 즉 결정 트리를 만들 때 전체 훈련 세트를 사용합니다. 대신 노드를 분할할 때 가장 좋은 분할을 찾는 것이 어나러 무작위로 분할합니다. 
- 결정트리는 2절에서 **DecisionTreeClassifier**의 splitter 매개변수를 `random`으로 지정했는데, 엑스트라 트리가 사용하는 결정트리가 바로 `splitter='random'`인 결정 트리입니다.
- 하나의 결정 트리에서 특성을 무작위로 분할한다면 성능은 낮아지겠지만 많은 트리를 앙상블하기 때문에 과대적합을 막고 검증 세트의 점수를 높이는 효과가 있습니다.


```python
from sklearn.ensemble import ExtraTreesClassifier

et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(et, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```

- 사이킷런에서 제공하는 엑스트라 트리는 **ExtraTreesClassifier** 입니다. 
- 이 모델의 교차 검증 점수를 확인해 봅니다.

```
0.9974503966084433 0.8887848893166506
```

- 랜덤 포레스트와 비슷한 결과를 얻었습니다. 
- 위 예제는 특성이 많지 않아 두 모델의 차이가 크지 않습니다. 
- 보통 엑스트라 트리의 무작위성이 좀 더 크기 때문에 랜더 포레스트보다 더 많은 결정 트리를 훈련해야 합니다. 하지만 랜덤하게 노드를 분할하기 때문에 빠른 계산 속도가 엑스트라 트리의 장점


> 결정 트리는 최적의 분할을 찾는 데 시간을 많이 소모합니다. 특히 고려해야 할 특성의 개수가 많을 때 더 그렇습니다. 만약 무작위로 나눈다면 훨씬 빨리 트리를 구성할 수 있습니다.

```python
et.fit(train_input, train_target)
print(et.feature_importances_)
```

- 엑스트라 트리도 랜덤 포레스트와 마찬가지로 특성 중요도를 제공합니다. 
- 순서는 \[알코올 도수, 당도, pH\]인데, 결과를 보면 엑스트라 트리도 결정 트리보다 당도에 대한 의존성이 작습니다.

```
[0.20183568 0.52242907 0.27573525]
```

- 엑스트라 트리의 회귀 버전은 **ExtraTreeRegressor** 클래스입니다.

## 그레디언트 부스팅

- **그레이디언트 부스팅**(gradient boosting)은 깊이가 얖은 결정 트리를 사용하여 이전 트리의 오차를 보완하는 방식으로 앙상블하는 방법입니다. 
- 사이킷런의 **GradientBoostingClassifier**는 기본적으로 깊이가 3인 결정 트리를 100개 사용합니다.
- 깊이가 얖은 결정트리를 사용하기 때문에 과대적합에 강하고 일반적으로 높은 일반화 성능을 기대할 수 있습니다.
- 그레이디언트란 이름과 같이 **경사 하강법**을 사용하여 트리를 앙상블에 추가합니다. 분류에서는 로지스틱 손실 함수를 사용하고 회귀에서는 평균 제곱 오차 함수를 사용합니다.
- 경사하강법은 손실함수를 산으로 정의하고 가장 낮은 곳을 찾아 내려오는 과정으로 설명 했습니다. 이때 가장 낮은 곳을 찾아 내려오는 방법은 모델의 가중치와 절편을 조금씩 바꾸는 것입니다.
- 그레이디언트 부스팅은 결정 트리를 계속 추가하면서 가장 낮은 곳을 찾아 이동합니다. 손실 함수의 낮은 곳으로 천천히 조금씩 이동해야 합니다. 그레이디언트 부스팅도 마찬가지 입니다. 그래서 깊이가 얕은 트리를 사용합니다. 또 학습률 매개변수로 속도를 조절합니다.

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(random_state=42)
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```

- 사이킷런에서 제공하는 **GradientBoostingClassifier**를 사용해 와인 데이터셋의 교차 검증 점수를 확인합니다.

```
0.8881086892152563 0.8720430147331015
```

- 과대적합이 많이 개선 되었습니다. 그레이디언트 부스팅은 결정 트리의 개수를 늘려도 과대적합에 매우 강합니다. 

```python
gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.2, random_state=42)
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```

- 학습률을 증가시키고 트리의 개수를 늘리면 조금 더 성능이 향상될 수 있습니다.

```
0.9464595437171814 0.8780082549788999
```
- 결정 트리의 개수를 500개로 늘렸지만 과대적합을 잘 억제하고 있습니다. 학습률 `learning_rate`의 기본값은 `0.1`입니다.


```python
gb.fit(train_input, train_target)
print(gb.feature_importances_)
```

- 그레이디언트 부스팅도 특성 중요도를 제공합니다.

```
[0.15887763 0.6799705  0.16115187]
```

- 그레이디언트 부스틸이 랜덤 포레스트보다 일부 특성(당도)에 더 집중합니다.

- 트리 훈련에 사용할 훈련 세트의 비율을 정하는 `subsample`입니다. 이 매개변수의 기본값은 `1.0`으로 전체 훈련 세트를 사용합니다. 
- 하지만 `subsample`이 1보다 작으면 훈련 세트의 일부를 사용합니다. 이는 마치 경사 하강법 단계마다 일부 샘플을 랜덤하게 선택하여 진행하는 확률적 경사하강법이나 미니배치 경사 하강법과 비슷
- 일반적으로 그레이디언트 부스팅이 랜덤 포레스트보다 조금 더 높은 성능을 얻을 수 있습니다. 
- 하지만 순서대로 트리를 추가하기 떄문에 훈련 속도가 느립니다. 즉 **GradientBoostingClassifier**에는 `n_jobs` 매개변수가 없습니다. 
- 그레이디언트 부스팅의 회귀 버전은 **GradientBoostingRegressor**입니다. 그레이디언트 부스팅의 속도와 성능을 더욱 개선한 것이 **히스토그램 기반 그레이디언트 부스팅** 입니다.

## 히스토그램 기반 부스팅

- **히스토그램 기반 그레이디언트 부스팅**(Histogram-based Gradient Boosting)은 정형 데이터를 다루는 머신러닝 알고리즘 중에 가장 인기가 높은 알고리즘입니다. 
- 히스토그램 기반 그레디언트 부스팅은 먼저 입력 특성을 256개의 구간으로 나눕니다. 따라서 노드를 분할할 때 최적의 분할을 매우 빠르게 찾을 수 있습니다.
- 히스토그램 기반 그레이디언트 부스팅은 256개의 구간 중에서 하나를 떼어 노고 누락된 값을 위해서 사용합니다. 따라서 입력에 누락된 특성이 있더라도 이를 따로 전처리할 필요가 없습니다.


```python
from sklearn.ensemble import HistGradientBoostingClassifier

hgb = HistGradientBoostingClassifier(random_state=42)
scores = cross_validate(hgb, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```

- 사이킷런의 히스토그램 기반 그레이디언트 부스팅 클래스는 **HistGradientBoostingClassifier**입니다. 
- 일반적으로 **HistGradientBoostingClassifier**는 기본 매개변수에서 안정적인 성능을 얻을 수 있습니다.
- **HistGradientBoostingClassifier**에는 트리의 개수를 지정하는데 `n_estimators` 대신에 부스팅 반복 횟수를 지정하는 `max_iter`를 사용합니다. 성능을 높이려면 `max_iter` 매개변수를 테스트해 보세요.


```
0.9321723946453317 0.8801241948619236
```

- 과대적합을 잘 억제하면서 그레이디언트 부스팅보다 조금 더 높은 성능을 제공합니다.


```python
from sklearn.inspection import permutation_importance

hgb.fit(train_input, train_target)
result = permutation_importance(hgb, train_input, train_target, n_repeats=10,
                                random_state=42, n_jobs=-1)
print(result.importances_mean)
```

- 특성 중요도를 확인합니다.
- 히스토그램 기반 그레이디언트 부스팅의 특성 중요도를 계산하기 위해 `permutation_importance()` 함수를 사용합니다. 이 함수는 특성을 하나씩 랜덤하게 섞어서 모델의 성능이 변화하는지 관찰하여 어떤 특성이 중요한지 계산합니다. 
- 훈련 세트뿐 아니라 테스트 세트에서도 적용할 수 있고 사이킷런에서 제공하는 추정기 모델에서 모두 사용할 수 있습니다.
- 히스토그램 기반 그레디언트 부스팅 모델을 훈련하고 훈련 세트에서 특성 중요도를 계산해 봅니다. `n_repeats` 매개변수는 랜덤하게 섞을 횟수를 지정합니다. 여기에서는 10으로 지정하겠습니다. 기본값은 5입니다.

```
[0.08876275 0.23438522 0.08027708]
```

- `permutation_importance()` 함수가 반환하는 객체는 반복하여 얻은 특성 중요도(importances), 평균(importances_mean), 표준 편차(importances_std)를 담고 있습니다.
- 평균을 출력해 보면 랜덤 포레스트와 비슷한 비율임을 알 수 있습니다.

```python
result = permutation_importance(hgb, test_input, test_target, n_repeats=10,
                                random_state=42, n_jobs=-1)
print(result.importances_mean)
```

- 테스트 세트에서 특성 중요도를 계산해 봅니다.

```
[0.05969231 0.20238462 0.049     ]
```

- 테스트 세트의 결과를 보면 그레이디언트 부스팅과 비슷하게 조금 더 당도에 집중하고 있다는 것을 알 수 있습니다.
- 이런 분석을 통해 모델을 실전에 투입했을 때 어떤 특성에 관심을 둘지 예상할 수 있습니다.

```python
hgb.score(test_input, test_target)
```

- **HistGradientBoostingClassifier**를 사용해 테스트 세트에서 성능을 최종적으로 확인해 봅시다.

```
0.8723076923076923
```

- 테스트 세트에서는 약 87% 정확도를 얻었습니다.
- 앙상블 모델은 확실히 단일 결정 트리보다 좋은 결과를 얻을 수 있습니다.
- 히스토그램 기반 그레이디언트 부스팅의 회귀 버전은 **HistGradientBoostingRegressor** 클래스에 구현되어 있습니다. 

## XGBoost

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(tree_method='hist', random_state=42)
scores = cross_validate(xgb, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```

```
0.9558403027491312 0.8782000074035686
```

## LightGBM

- 마이크로소프트에서 만들었습니다.
- 사이킷런의 히스토그램 기반 그레디언트 부스팅이 **LightGBM**에서 영향을 많이 받았습니다.

```python
from lightgbm import LGBMClassifier

lgb = LGBMClassifier(random_state=42)
scores = cross_validate(lgb, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```

```
0.935828414851749 0.8801251203079884
```