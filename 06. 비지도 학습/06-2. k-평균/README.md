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

![스크린샷 2024-11-10 오후 3 11 29](https://github.com/user-attachments/assets/95a9a157-dbd6-4a60-91cc-bf50b7189ff9)

- 먼저 3개의 클러스터 중심(빨간 점)을 랜덤하게 지정합니다(1). 그리고 클러스터 중심에서 가장 가까운 샘플을 하나의 클러스터로 묶습니다. 왼쪽 위부터 시계 방향으로 바나나 2개와 사과 1개 클러스터, 바나나 1개와 파인애플 2개 클러스터, 사과 2개와 파인애플 1개 클러스터가 만들어졌습니다. 클러스터에는 순서나 번호는 의미가 없습니다.
- 그 다음 클러스터의 중심을 다시 계산하여 이동시킵니다. 맨 아래 클러스터는 사과 쪽으로 중심이 조금 더 이동하고 왼쪽 위의 클러스터는 바나나 쪽으로 중심이 더 이동하는 식입니다.
- 그 다음 클러스터의 중심을 다시 계산하여 이동시킵니다. 맨 아래 클러스터는 사과 쪽으로 중심이 조금 더 이동하고 클러스터는 바나나 쪽으로 중심이 더 이동하는 식입니다. 
- 클러스터 중심을 다시 계산한 다음 가장 가까운 샘플을 다시 클러스터로 묶습니다(2). 이제 3개의 클러스터에는 바나나와 파인애플, 사과가 3개씩 올바르게 묶여 있습니다. 다시 한번 클러스터의 중심을 계산합니다. 그 다음 빨간 점을 클러스터의 가운데 부분으로 이동시킵니다.
- 이동된 클러스터의 중심에서 다시 한번 가장 가까운 샘플을 클러스터로 묶습니다(3). 중심에서 가장 가까운 샘플은 이전 클러스터(2)와 동일 합니다. 따라서 만들어진 클러스터에 변동이 없으므로 k-평균 알고리즘을 종료합니다.
- k-평균 알고리즘은 처음에는 랜덤하게 클러스터의 중심을 선택하고 점차 가장 가까운 샘플의 중심으로 이동하는 비교적 간단한 알고리즘입니다. 


## KMeans 클래스

- 사이킷런으로 k-평균 모델을 직접 만들 수 있습니다. 

```python
!wget https://bit.ly/fruits_300_data -O fruits_300.npy
```

- wget 명령으로 데이터를 다운로드 합니다.

```python
import numpy as np

fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)
```

- 넘파이 `np.load()` 함수를 사용해 npy 파일을 읽어 넘파이 배열을 준비합니다. k-평균 모델을 훈련하기 위해 (샘플 개수, 너비, 높이) 크기의 3차원 배열을 (샘플 개수, 너비 X 높이) 크기를 가진 2차원 배열로 변경합니다.

```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)
```

- 사이킷런의 k-평균 알고리즘은 `sklearn.cluster` 모듈 아래 **KMeans** 클래스에 구현되어 있습니다. 
- 이 클래스에서 설정할 매개변수는 클러스터 개수를 지정하는 `n_cluster` 입니다. 여기에서는 클러스터 개수를 3으로 지정하겠습니다.
- 이 클래스를 사용하는 방법도 다른 클래스들과 비슷합니다. 다만 비지도 학습이므로 `fit()` 메서드에서 타깃 데이터를 사용하지 않습니다. 

```python
print(km.labels_)
```

- 군집된 결과는 **KMeans** 클래스 객체의 `labels_` 속성에 저장됩니다. **labels_** 배열의 길이는 샘플 개수와 같습니다. 
- 이 배열은 각 샘플이 어떤 레이블에 해당되는지 나타냅니다. `n_clusters=3` 으로 지정했기 때문에 `labels_` 배열의 값은 0, 1, 2 중 하나입니다.

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

- 레이블값 0, 1, 2와 레이블 순서에는 어떤 의미도 없습니다. 실제 레이블 0, 1, 2가 어떤 과일 사진을 주로 모았는지 알아보려면 직접 이미지를 출력하는 것이 최선입니다. 그 전에 레이블 0, 1, 2로 모은 샘플의 개수를 확인하겠습니다.

```python
print(np.unique(km.labels_, return_counts=True))
```

```
(array([0, 1, 2], dtype=int32), array([112,  98,  90]))
```

- 첫 번째 클러스터(레이블 0)가 111개의 샘플을 모았고, 두 번째 클러스터(레이블 1)가 98개의 샘플을 모았습니다.
- 세 번째 클러스터(레이블 2)는 91개의 샘플을 모았습니다. 그럼 각 클러스터가 어떤 이미지를 나타냈는지 그림으로 출력하기 위해 유틸리티 함수 `draw_fruits()` 를 만들어 봅니다.

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

- `draw_fruits()` 함수는 (샘플 개수, 너비, 높이)의 3차원 배열을 입력받아 가로로 10개씩 이미지를 출력합니다. 샘플 개수에 따라 행과 열의 개수를 계산하고 `figsize`를 지정합니다. `figsize`는 `ratio` 매개변수에 비례하여 커집니다. `ratio`의 기본값은 1입니다.

- 그 다음 2중 for 반복문을 사용하여 먼저 첫 번째 행을 따라 이미지를 그립니다. 그리고 두 번째 행의 이미지를 그리는 식으로 계속됩니다.

```python
draw_fruits(fruits[km.labels_==0])
```

- 이 함수를 사용해 레이블이 0인 과일 사진을 모두 그려 보겠습니다. `km.labels_ == 0`과 같이 쓰면 `km.labels_` 배열에서 값이 0인 위치는 True,그 외에는 모두 False가 됩니다. 넘파이는 이런 불리언 배열을 사용해 원소를 선택할 수 있습니다. 이를 **불리언 인덱싱**이라고 합니다. 
- 넘파이 배열에 불리언 인덱싱을 적용하면 True인 위치의 원소만 모두 추출합니다.  


![스크린샷 2024-11-10 오후 4 35 03](https://github.com/user-attachments/assets/0406270d-430e-4864-96c5-8e92ee5a4c5d)

- 레이블 0으로 클러스터링된 91개의 이미지를 모두 출력했습니다. 이 클러스터는 대부분 파인애플이고 사과와 바나나가 간간히 섞여 있습니다. 


```python
draw_fruits(fruits[km.labels_==1])
```


![스크린샷 2024-11-10 오후 4 35 12](https://github.com/user-attachments/assets/574b1a6f-d768-4100-9092-445b1aaf98d5)


```python
draw_fruits(fruits[km.labels_==2])
```

![스크린샷 2024-11-10 오후 4 35 20](https://github.com/user-attachments/assets/bc2350ea-2bcb-484f-a0e7-6353c87fcef9)


- 레이블이 1인 클러스터는 바나나로만 이루어져 있고, 레이블이 2인 클러스터는 사과로만 이루어져 있습니다. 
- 하지만 레이블이 0인 클러스터는 파인애플에 사과 9개와 바나나 2개가 섞여 있네요. k-평균 알고리즘이 이 샘플들을 완벽하게 구별해내지는 못했습니다.
- 하지만 훈련 데이터에 타깃 레이블을 전혀 제공하지 않았음에도 스스로 비슷한 샘플들을 아주 잘 모은 것 같습니다.

## 클러스터 중심

- **KMeans** 클래스가 최종적으로 찾은 클러스터 중심은 `cluster_centers_` 속성에 저장되어 있습니다. 
- 이 배열은 `fruits_2d` 샘플의 클러스터 중심이기 때문에 각 중심을 이미지로 출력하려면 100 X 100 크기의 2차원 배열로 바꿔야 합니다. 

```python
draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=3)
```

![스크린샷 2024-11-10 오후 4 42 07](https://github.com/user-attachments/assets/471ce37b-00e6-4053-be81-5c81d228b830)

- 이전 절에서 사과, 바나나, 파인애플의 픽셀 평균값을 출력했던 것과 매우 비슷합니다.
- **KMeans** 클래스는 훈련 데이터 샘플에서 클러스터 중심까지 거리로 반환해 주는 `transform()` 메서드를 가지고 있습니다. `transform()` 메서드가 있다는 것은 마치 **StandardScaler** 클래스 처럼 특성값을 변환하는 도구로 사용할 수 있다는 의미이빈다. 

```python
print(km.transform(fruits_2d[100:101]))
```

- 인덱스가 100인 샘플에 `transform()` 메서드를 적용해 봅니다. `fit()` 메서드와 마찬가지로 2차원 배열을 기대합니다.
- `fruits_2d\[100\]` 처럼 쓰면 (10000,) 크기의 배열이 되므로 에러가 발생합니다. 슬라이싱 연산자를 사용해서 (1, 10000) 크기의 배열을 전달합니다.

```
[[3400.24197319 8837.37750892 5279.33763699]]
```

- 하나의 샘플을 전달했기 때문에 반환된 배열의 크기가 (1, 클러스터 개수)인 2차원 배열입니다. 첫 번째 클러스터(레이블 0), 두 번째 클러스터(레이블 1)가 각각 첫 번째 원소, 두 번째 원소의 값입니다. 
- 첫 번째 클러스터까지의 거리가 3393.8로 가장 작습니다. 이 샘플은 레이블 0에 속한 것 같습니다. 


```python
print(km.predict(fruits_2d[100:101]))
```

- **KMeans** 클래스는 가장 가까운 클러스터 중심으로 예측 클래스로 출력하는 `predict()` 메서드를 제공합니다. 

```
[0]
```

- `transform()`의 결과에서 짐작할 수 있듯이 레이블 0으로 예측했습니다. 클러스터 중심을 그려보았을때 레이블 0은 파인애플이었으므로 이 샘플은 파인애플입니다. 



```python
draw_fruits(fruits[100:101])
```

![스크린샷 2024-11-10 오후 5 56 40](https://github.com/user-attachments/assets/5f4f7c38-4f15-4c65-b54e-921e68e38dd4)


```python
print(km.n_iter_)
```

- k-평균 알고리즘은 앞에서 설명했듯이 반족적으로 클러스터 중심을 옮기면서 최적의 클러스터를 찾습니다. 
- 알고리즘이 반복한 횟수는 **KMeans** 클래스의 `n_iter_` 속성에 저장됩니다.

```
4
```

- 클러스터 중심을 특성 공학처럼 사용해 데이터셋을 저차원(이 경우에는 10,000에서 3으로 줄입니다)으로 변환할 수 있습니다. 또는 가장 가까운 거리에 있는 클러스터 중심을 샘플 예측 값으로 사용할 수 있다는 것을 배웠습니다.
- 타깃값을 사용하지 않았디만 `n_clusters`를 3으로 지정한 것은 타깃에 대한 정보를 활용한 셈입니다. 실전에서는 클러스터 개수조차 알 수 없습니다. 


## 최적의 k 찾기

- k-평균 알고리즘의 단점 중 하나는 클러스터 개수를 사전에 지정해야 한다는 것입니다. 실전에서는 몇개의 클러스터가 있는지 알 수 없습니다. 
- 사실 군집 알고리즘에서 적절한 k값을 찾기 위한 완벽한 방법은 없습니다. 여기서는 적절한 클러스터 개수를 찾기 위한 대표적인 방법인 **엘보우**(elbow)방법에 대해 알아봅니다.
- 앞서 본 것처럼 k-평균 알고리즘은 클러스터 중심과 클러스터에 속한 샘플 사이의 거리를 잴 수 있습니다. 이 거리의 제곱 합을 **이너셔**(inertia)라고 부릅니다. 
- **이너셔**는 클러스터에 속한 샘플이 얼마나 가깝게 모여 있는지를 나타내는 값으로 생각할 수 있습니다. 일반적으로 클러스터 개수가 늘어나면 클러스터 개개의 크기는 줄어들기 때문에 이너셔도 줄어듭니다. 
- 엘보우 방법은 클러스터 개수를 늘려가면서 이너셔의 변화를 관찰하여 최적의 클러스터 개수를 찾는 방법입니다.
- 클러스터 개수를 증가시키면서 이너셔를 그래프로 그리면 감소하는 속도가 꺾이는 지점이 있습니다. 이 지점부터는 클러스터 개수를 늘려도 클러스터에 잘 밀집된 정도가 크게 개선되지 않습니다. 즉 이너셔가 크게 줄어들지 않습니다. 이 지점이 마치 팔꿈치 모양이어서 엘보우 방법이라 부릅니다.


![스크린샷 2024-11-10 오후 6 07 32](https://github.com/user-attachments/assets/7bb74ad5-6c32-40d7-b850-41b1d6f5c5e8)


- 과일 데이터셋을 사용해 이너셔를 계산해 봅시다. 
- **KMeans** 클래스는 자동으로 이너셔를 계산해서 `inertia_` 속성으로 제공합니다. 다음 코드에서 클러스터 개수를 k를 2\~6까지 바꿔가며 **KMeans** 클래스를 5번 훈련합니다. `fit()` 메서드로 모델을 훈련한 후 `inertia_` 속성에 저장된 이너셔 값을 `inertia` 리스트에 추가합니다. 
- 마지막으로 inertia 리스트에 저장된 값을 그래프로 출력합니다. 

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

![스크린샷 2024-11-10 오후 6 11 42](https://github.com/user-attachments/assets/80baa882-396e-4ff7-85c7-8b93da989b05)

- 이 그래프에서는 꺾이는 지점이 두드러지지는 않지만, k = 3에서 그래프의 기울기가 조금 바뀐 것을 볼 수 있습니다. 
- 엘보우 지점보다 클러스터 개수가 많아지면 이너셔의 변화가 줄어들면서 군집 효과도 줄어듭니다. 
- 하지만 이 그래프에서는 이런 지점이 명확하지는 않습니다.