# 주성분 분석

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

## PCA 클래스

```python
!wget https://bit.ly/fruits_300_data -O fruits_300.npy
```

```python
import numpy as np

fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)
```

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=50)
pca.fit(fruits_2d)
```

```python
print(pca.components_.shape)
```

```
(50, 10000)
```

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

## 원본 데이터 재구성

```python
fruits_inverse = pca.inverse_transform(fruits_pca)
print(fruits_inverse.shape)
```

```
(300, 10000)
```

```python
fruits_reconstruct = fruits_inverse.reshape(-1, 100, 100)
```

```python
for start in [0, 100, 200]:
    draw_fruits(fruits_reconstruct[start:start+100])
    print("\n")
```

## 설명된 분산 

```python
print(np.sum(pca.explained_variance_ratio_))
```

```
0.9215275787736402
```

```python
plt.plot(pca.explained_variance_ratio_)
```

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