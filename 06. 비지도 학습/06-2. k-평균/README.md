# k-평균

## 키워드 정리

- **k-평균**
  - 처음에 랜덤하게 클러스터의 중심을 정하고 클러스터를 만듭니다. 그 다음 클러스터의 중심을 이동하고 다시 클러스터를 만드는 식으로 반복해서 최적의 클러스터를 구성하는 알고리즘입니다.
- **클러스터 중심**
  - k-평균 알고리즘이 만든 클러스터에 속한 샘플의 특성 평균값니다.
  - 센트로이드(centroid)라고도 부릅니다. 가장 가까운 클러스터 중심을 샘플의 또 다른 특성으로 사용하거나 새로운 새믈에 대한 예측으로 활용할 수 있습니다.
- **엘보우 방법**
  - 최적의 클러스터 개수를 정하는 방법 중 하나입니다.
  - 이너셔는 클러스터 중심과 샘플 사이 거리의 제곱 합입니다. 클러스터 개수에 따라 이니셔 감소가 꺾이는 지점이 적절한 클러스터 개수 k가 될 수 있습니다.
  - 이 그래프의 모양을 따서 엘보우 방법이라고 부릅니다.

## 핵심 패키지와 함수

### scikit-learn

- **KMeans**
  - k-평균 알고리즘 클래스입니다.
  - `n_clusters` 에는 클러스터 개수를 지정합니다. 기본값은 8입니다.
  - 처음에는 랜덤하게 센트로이드를 초기화하기 때문에 여러 번 반복하여 이너셔를 기준으로 가장 좋은 결과를 선택합니다. `n_init`는 이 반복 횟수를 지정합니다. 기본값은 10이었으나 사이킷런 버전 1.4에서는 `auto`로 변경될 예정입니다.
  - `max_iter`는 k-평균 알고리즘의 한 번 실행에서 최적의 센트로이드를 찾기 위해 반복할 수 있는 최대 횟수입니다. 기본값은 200입니다.


## 비지도 학습

- 앞서 사과, 파인애플, 바나나에 있는 각 픽셀의 평균값을 구해서 가장 가까운 사진을 골랐습니다. 이 경우에는 사과, 파인애플, 바나나 사진임을 미리 알고 있었기 때문에 각 과일의 평균을 구할 수 있었습니다. 하지만 진짜 비지도 학습에서는 사진에 어떤 과일이 들어 있는지 알지 못합니다.
- 이러 경우 **k-평균**(k-means) 군집 알고리즘을 사용하면 평균값을 자동으로 찾아줍니다.
- 이 평균값이 클러스터의 중심에 위치하기 때문에 **클러스터 중심**(cluster center) 또는 **센트로이드**(centroid)라고 부릅니다.

## k-평균 알고리즘 소개

k-평균 알고리즘의 작동 방식은 다음과 같습니다.

- 무작위로 k개의 클러스터 중심을 정합니다.
- 각 샘플에서 가장 가까운 클러스터 중심을 찾아 해당 클러스터의 샘플로 지정합니다.
- 클러스터에 속한 샘플의 평균값으로 클러스터 중심을 변경합니다.
- 클러스터 중심에 변화가 없을 때까지 2번 돌아가 반복합니다, 



## KMeans 클래스

```python
!wget https://bit.ly/fruits_300_data -O fruits_300.npy
```

```python
import numpy as np

fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)
```

```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)
```

```python
print(km.labels_)
```

```
[2 2 2 2 2 0 2 2 2 2 2 2 2 2 2 2 2 2 0 2 2 2 2 2 2 2 0 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 0 2 0 2 2 2 2 2 2 2 0 2 2 2 2 2 2 2 2 2 0 0 2 2 2 2 2 2 2 2 0 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1]
```

```python
print(np.unique(km.labels_, return_counts=True))
```

```
(array([0, 1, 2], dtype=int32), array([112,  98,  90]))
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
draw_fruits(fruits[km.labels_==0])
```

```python
draw_fruits(fruits[km.labels_==1])
```

```python
draw_fruits(fruits[km.labels_==2])
```

## 클러스터 중심

```python
draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=3)
```

```python
print(km.transform(fruits_2d[100:101]))
```

```
[[3400.24197319 8837.37750892 5279.33763699]]
```

```python
print(km.predict(fruits_2d[100:101]))
```

```
[0]
```

```python
draw_fruits(fruits[100:101])
```

```python
print(km.n_iter_)
```

```
4
```

## 최적의 k 찾기

```python
inertia = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    km.fit(fruits_2d)
    inertia.append(km.inertia_)

plt.plot(range(2, 7), inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()
```