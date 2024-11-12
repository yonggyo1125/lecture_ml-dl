# 주성분 분석

- k-평균 알고리즘으로 업로드된 사진을 클러스터로 분류하여 폴더별로 저장했습니다. 그런데 이벤트가 진행되면서 문제가 생겼습니다. 너무 많은 사진이 등록되어 저장 공간이 부족합니다.
- 군집이나 분류에 영향을 끼치지 않으면서 업로드된 사진의 용량을 줄일 수 있을까요?

## 키워드 정리

- **차원 축소**
    - 원본 데이터의 특성을 적은 수의 새로운 특성으로 변환하는 비지도 학습의 한 종류입니다.
    - 차원 축소는 저장 공간을 줄이고 시각화하기 쉽습니다. 
    - 다른 알고리즘의 성능을 높일 수도 있습니다.
- **주성분 분석**
    - 차원 축소 알고리즘의 하나로 데이터에서 가장 분산이 큰 방향을 찾는 방법입니다. 이런 방향을 주성분이라고 부릅니다. 
    - 원본 데이터를 주성분에 투영하여 새로운 특성을 만들 수 있습니다. 
    - 일반적으로 주성분은 원본 데이터에 있는 특성 개수보다 작습니다. 
- **설명된 분산**
    - 주성분 분석에서 주성분이 얼마나 원본 데이터의 분산을 잘 나타내는지 기록한 것입니다. 
    - 사이킷런의 PCA 클래스는 주성분 개수나 설명된 분산의 비율을 지정하여 주성분 분석을 수행할 수 있습니다.

## 핵심 패키지와 함수
### scikit-learn

- **PCA** 
    - 주성분 분석을 수행하는 클래스입니다.
    - `n_components`는 주성분의 개수를 지정합니다. 기본값은 `None`으로 샘플 개수와 특성 개수 중에 작은 것의 값을 사용합니다. 
    - `random_state`에는 넘파이 난수 시드 값을 지정할 수 있습니다.
    - `components_` 속성에는 훈련 세트에서 찾은 주성분이 저장됩니다.
    - `explained_variance_` 속성에는 설명된 분산이 저장되고, `explained_variance_ratio_`에는 설명된 분산의 비율이 저장됩니다. 
    - `inverse_transform()` 메서드는 `transform()` 메서드로 차원을 축소시킨 데이터를 다시 원본 차원으로 복원합니다.


## 차원과 차원 축소
- 지금까지 우리는 데이터가 가진 속성을 특성이라 불렀습니다. 과일 사진의 경우 10,000개의 픽셀이 있기 때문에 10,000개의 특성이 있는 셈입니다. 
- 머신러닝에서는 이런 특성을 **차원**(dimension)이라고 부릅니다. 
- 10,000개의 특성은 결국 10,000개의 차원이라는 건데 이 차원을 줄일 수 있다면 저장 공간을 크게 절약할 수 있습니다. 

- 이를 위해 비지도 학습 작업 중 하나인 **차원 축소**(dimensionality reduction) 알고리즘을 다루어 봅니다.
- 특성이 많으면 선형 모델의 성능이 높아지고 훈련 데이터에 쉽게 과대적합된다는 것을 배웠습니다. 차원 축소는 데이터를 가장 잘 나타내는 일부 특성을 선택하여 데이터 크기를 줄이고 지도학습 모델의 성능을 향상시킬 수 있는 방법입니다.
- 또한 줄어든 차원에서 다시 원본 차원 (예를 들어 과일 사진의 경우 10,000개의 차원)으로 손실을 최대한 줄이면서 복원할 수도 있습니다. 이 절에서는 대표적인 차원 축소 알고리즘인 **주성분 분석**(principal component analysus)을 배우겠습니다.
- 주성분 분석을 간단히 **PCA**라고도 부릅니다.

## 주성분 분석 소개 
- 주성분 분석(PCA)은 데이터에 있는 분산이 큰 방향을 찾는 것으로 이해할 수 있습니다. 
- 분산은 데이터가 널리 퍼져있는 정도를 말합니다. 분산이 큰 방향이란 데이터를 잘 표현하는 어떤 벡터라고 생각할 수 있습니다. 


![스크린샷 2024-11-10 오후 10 38 12](https://github.com/user-attachments/assets/dde6aec7-1cce-4f2a-84ec-e101a9a65d85)

- 이 데이터는 x1, x2 2개의 특성이 있습니다. 대각선 방향으로 길게 늘어진 형태를 가지고 있습니다. 이 데이터에서 가장 분산이 큰 방향, 즉, 데이터의 분포를 가장 잘 표현하는 방향을 찾아봅시다.

![스크린샷 2024-11-10 오후 10 39 58](https://github.com/user-attachments/assets/d113ac3b-03ac-46d7-a477-d9fa0b17d5a0)


- 직관적으로 길게 늘어진 대각선 방향이 분산이 가장 크다고 알 수 있습니다. 위의 그림에서 화살표의 위치는 큰 의미가 없습니다. 오른쪽 위로 향하거나 왼쪽 아래로 향할 수도 있습니다. 중요한 것은 분산이 큰 방향을 찾는 것이 중요합니다.
- 앞에서 찾은 직선이 원점에서 출발한다면 두 원소로 이루어진 벡터로 쓸 수 있습니다. 예를 들어 다음 그림의 (2,1) 처럼 나타낼 수 있습니다.


![스크린샷 2024-11-10 오후 10 45 04](https://github.com/user-attachments/assets/01ce9305-fe1e-4b74-9085-c3a2625fdb2f)

 > 실제로 사이킷런의 PCA 모델을 훈련하면 자동으로 특성마다 평균값을 빼서 원점에 맞춰 줍니다. 따라서 우리가 수동으로 데이터를 원점에 맞출 필요가 없습니다.

- 이 벡터를 **주성분**(principle component)이라고 부릅니다. 이 주성분 벡터는 원본 데이터에 있는 어떤 방향입니다. 
- 따라서 주성분 벡터의 원소 개수는 원본 데이터셋에 있는 특성 개수와 같습니다. 하지만 원본 데이터는 주성분을 사용해 차원을 줄일 수 있습니다. 
- 예를 들면 다음과 같이 샘플 데이터 s(4, 2)를 주성분에 직각으로 투영하면 1차원 데이터 p(4.5)를 반들 수 있습니다.


![스크린샷 2024-11-12 오후 10 54 03](https://github.com/user-attachments/assets/4ff43a50-8a55-4508-ae85-91141305ff54)

- 주성분은 원본 차원과 같고 주성분으로 바꾼 데이터는 차원이 줄어듭니다.
- 주성분이 가장 분산이 큰 방향이기 때문에 주성분에 투영하여 바꾼 데이터는 원본이 가지고 있는 특성을 가장 잘 나타내고 있을 것입니다.
- 첫 번째 주성분을 찾은 다음 이 벡터에 수직이고 분산이 가장 큰 다음 방향을 찾습니다. 이 벡터가 두 번째 주성분입니다. 
- 여기에서는 2차원이기 때문에 두 번째 주성분의 방향은 다음처럼 하나입니다.

![스크린샷 2024-11-12 오후 11 00 38](https://github.com/user-attachments/assets/ef496c40-5b4d-41f4-8a4b-bdfea6ebe03a)

- 일반적으로 주성분은 원본 특성의 개수만큼 찾을 수 있습니다. 

> 기술적인 이유로 주성분은 원본 특성의 개수와 샘플 개수 중 작은 값만큼 찾을 수 있습니다. 일반적으로 비지도 학습은 대량의 데이터에서 수행하기 때문에 원본 특성의 개수만큼 찾을 수 있다고 말합니다. 

## PCA 클래스

```python
!wget https://bit.ly/fruits_300_data -O fruits_300.npy

import numpy as np

fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)
```

- 과일 사진 데이터를 다운로드하여 넘파이 배열로 적재합니다. 

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=50)
pca.fit(fruits_2d)
```

- 사이킷런은 `sklearn.decomposition` 모듈 아래 PCA 클래스로 주성분 분석 알고리즘을 제공합니다.
- **PCA** 클래스의 객체를 만들 떼 `n_components` 매개변수에 주성분의 개수를 지정해야 합니다. 
- k-평균과 마찬가지로 비지도 학습이기 때문에 `fit()` 메서드에 타깃값을 제공하지 않습니다. 

```python
print(pca.components_.shape)
```

- **PCA** 클래스가 찾은 주성분은 `components_` 속성에 저장되어 있습니다. 이 배열의 크기를 확인합니다.

```
(50, 10000)
```

- `n_components=50`으로 지정했기 때문에 `pca.components_` 배열의 첫 번째 차원이 50입니다. 즉 50개의 주성분을 찾았습니다. 두 번째 차원은 항상 원본 데이터의 특성 개수와 같은 10,000입니다. 

 

```python 
import matplotlib.pyplot as plt

def draw_fruits(arr, ratio=1):
    n = len(arr)    # n은 샘플 개수입니다
    # 한 줄에 10개씩 이미지를 그립니다. 샘플 개수를 10으로 나누어 전체 행 개수를 계산합니다.
    rows = int(np.ceil(n/10))
    # 행이 1개 이면 열 개수는 샘플 개수입니다. 그렇지 않으면 10개입니다.
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols,
                            figsize=(cols*ratio, rows*ratio), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n:    # n 개까지만 그립니다.
                axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
            axs[i, j].axis('off')
    plt.show()
```

```python
draw_fruits(pca.components_.reshape(-1, 100, 100))
```

- 원본 데이터와 차원이 같으므로 주성분을 100 X 100 크기의 이미지처럼 출력해 볼 수 있습니다. `draw_fruits()` 함수를 사용해서 이 주성분을 그림으로 그려보면 다음과 같습니다. 

![스크린샷 2024-11-12 오후 11 23 36](https://github.com/user-attachments/assets/92ca9a27-a773-4705-a63c-6acf3dd585f8)

- 이 주성분은 원본 데이터에서 가장 분산이 큰 방향을 순서대로 나타낸 것입니다. 한편으로는 데이터셋에 있는 어떤 특징을 잡아낸 것처럼 생각할 수도 있습니다.
- 주성분을 찾았으므로 원본 데이터를 주성분에 투영하여 특성의 개수를 10,000개에서 50개로 줄일 수 있습니다. 이는 마치 원본 데이터를 각 주성분으로 분해하는 것으로 생각할 수 있습니다.
- **PCA**의 `transform()` 메서드를 사용해 원본 데이터의 차원을 50으로 줄일 수 있습니다.

```python
print(fruits_2d.shape)
```

```
(300, 10000)
```

```python
fruits_pca = pca.transform(fruits_2d)
```

```python 
print(fruits_pca.shape)
```

```
(300, 50)
```

- fruit_2d는 (300, 10000) 크기의 배열이었습니다. 10,000개의 픽셀(특성)을 가진 300개의 이미지 입니다. 50개의 주성분을 찾은 **PCA** 모델을 사용해 이를 (300, 50) 크기의 배열로 변환했습니다. 
- 이제 `fruits_pca` 배열은 50개의 특성을 가진 데이터입니다.  

## 원본 데이터 재구성

- 앞에서 10,000개의 특성을 50개로 줄였습니다. 이로 인해 어느 정도 손실이 발생할 수 밖에 없습니다. 하지만 최대한 분산이 큰 방향으로 데이터를 투영했기 때문에 원본 데이터를 상당 부분 재구성할 수 있습니다. 

```python
fruits_inverse = pca.inverse_transform(fruits_pca)
print(fruits_inverse.shape)
```

- **PCA** 클래스는 이를 위해 `inverse_transform()` 메서드를 제공합니다. 
- 50개의 차원으로 축소한 `fruits_pca` 데이터를 전달해 10,000의 특성을 복원합니다. 

```
(300, 10000)
```

- 예상대로 10,000개의 특성이 복원되었습니다. 

```python
fruits_reconstruct = fruits_inverse.reshape(-1, 100, 100)
for start in [0, 100, 200]:
    draw_fruits(fruits_reconstruct[start:start+100])
    print("\n")
```

- 이 데이터를 100 X 100 크기로 바꾸어 100개씩 나누어 출력합니다. 
- 이 데이터는 순서대로 사과, 파인애플, 바나나를 100개씩 담고 있습니다.

![스크린샷 2024-11-12 오후 11 36 36](https://github.com/user-attachments/assets/7dad9b1a-b4eb-43c8-b33e-337c72d7eb9e)

- 일부 흐리고 번진 부분이 있지만 불과 50개의 특성을 10,000개로 늘린 것을 감안한다면 놀라운 일입니다. 이 50개의 특성이 분산을 가장 잘 보존하도록 변환된 것이기 때문입니다.
- 만약 주성분을 최대로 사용했다면 완벽하게 원본 데이터를 재구성할 수 있을 것입니다.

## 설명된 분산 

- 주성분이 원본 데이터의 분산을 얼마나 잘 나타내는지 기록한 값을 **설명된 분산**(explained variance)이라고 합니다. 
- **PCA** 클래스의 `explained_variance_ratio_` 에 각 주성분의 설명된 분산 비율이 기록되어 있습니다. 
- 당연히 첫 번째 주성분의 설명된 분산이 가장 큽니다. 이 분산 비율을 모두 더하면 50개의 주성분으로 표현하고 있는 총 분산 비율을 얻을 수 있습니다.

```python
print(np.sum(pca.explained_variance_ratio_))
```

```
0.9215275787736402
```

- 92%가 넘는 분산을 유지하고 있습니다.
- 앞에서 50개의 특성에서 원본 데이터를 복원했을 때 원본 이미지의 품질이 높았던 이유를 여기에서 찾을 수 있습니다. 
 

```python
plt.plot(pca.explained_variance_ratio_)
plt.show()
```

- 설명된 분산의 비율을 그래프로 그려보면 적절한 주성분의 개수를 찾는 데 도움이 됩니다.
- 맷플롯립 `plot()` 함수로 설명된 분산을 그래프로 출력하겠습니다.


![스크린샷 2024-11-12 오후 11 49 29](https://github.com/user-attachments/assets/a8e9ab89-c6de-4990-865f-16c9b777ac2b)


## 다른 알고리즘과 함께 사용하기

```python 
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
```

```python
target = np.array([0] * 100 + [1] * 100 + [2] * 100)
```

```python
from sklearn.model_selection import cross_validate

scores = cross_validate(lr, fruits_2d, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))
```

```
0.9966666666666667
0.9981564998626709
```

```python
scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))
```

```
0.9966666666666667
0.024461746215820312
```

```python
pca = PCA(n_components=0.5)
pca.fit(fruits_2d)
```

```python
print(pca.n_components_)
```

```python
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)
```

```python
scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))
```

```
0.9933333333333334
0.02928957939147949
```

```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_pca)
```

```python
print(np.unique(km.labels_, return_counts=True))
```

```
(array([0, 1, 2], dtype=int32), array([110,  99,  91]))
```

```python
for label in range(0, 3):
    draw_fruits(fruits[km.labels_ == label])
    print("\n")
```

```python
for label in range(0, 3):
    data = fruits_pca[km.labels_ == label]
    plt.scatter(data[:,0], data[:,1])
plt.legend(['pineapple', 'banana', 'apple'])
plt.show()
```
