# 군집 알고리즘

## 키워드 정리

- **비지도 학습**
  - 머신러닝의 한 종류로 훈련 데이터에 타깃이 없습니다.
  - 타깃이 없기 떄문에 외부의 도움 없이 스스로 유용한 무언가를 학습해야 합니다.
  - 대표적인 비지도 학습 작업은 군집, 차원 축소 등입니다.
- **히스토그램**
  - 구간별로 값이 발생한 빈도를 그래프로 표시한 것입니다.
  - 보통 x축이 값의 구간(계급)이고 y축은 발생 빈도(도수)입니다.
- **군집**
  - 비슷한 샘플끼리 하나의 그룹으로 모으는 대표적인 비지도 학습 작업입니다.
  - 군집 알고리즘으로 모은 샘플 그룹을 클러스터라고 부릅니다.


## 타깃을 모르는 비지도 학습
- 타깃이 없을 때 사용하는 머신러닝 알고리즘이 바로 **비지도 학습**(unsupervised learning)입니다. 
- 사람이 가르쳐 주지 않아도 데이터에 있는 무서인가를 학습하는 ㄱㅅ




## 과일 사진 데이터 준비하기

```python
!wget https://bit.ly/fruits_300_data -O fruits_300.npy
```

- 과일 데이터는 사과, 바나나, 파인애플을 담고 있는 흑백 사진입니다.
- 이 데이터는 넘파이 배열의 기본 저장 포맷인 npy 파일로 저장되어 있습니다.
- 넘파이에서 이 파일을 읽으려면 먼저 코랩으로 다운로드해야 합니다. 

```python
import numpy as np
import matplotlib.pyplot as plt
```

- 넘파이와 맷플롯립을 임포트합니다.


```python
fruits = np.load('fruits_300.npy')
```

- 넘파이에서 npy파일을 로드합니다. 로드할 때는 `load()` 메서드를 사용합니다.


```python
print(fruits.shape)
```

- fruits는 넘파이이 배열이고 fruits_300.npy 파일에 들어있는 모든 데이터를 담고 있습니다. 
- fruits 배열의 크기를 확인해보면

```
(300, 100, 100)
```

- 이 배열의 첫 번째 차원(300)은 샘플의 갯수를 나타냅니다.
- 두 번째 차원(100)은 이미지 높이
- 세 번째 차원(100)은 이미지 너비
- 이미지 크기는 100 X 100 입니다. 
- 각 픽셀은 넘파이 배열의 원소 하나에 대응합니다. 즉 각 배열의 크기가 100 X 100 입니다.

```python
print(fruits[0, 0, :])
```

- 이미지의 첫 번째 행을 출력합니다. 
- 3차원 배열이기 때문에 처음 2개의 인덱스를 0으로 지정하고 마지막 인덱스는 지정하지 않거나 슬라이싱 연산자를 쓰면 첫 번째 이미지의 첫 번쨰 행을 모두 선택할 수 있습니다.

```python
[  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   2   1
   2   2   2   2   2   2   1   1   1   1   1   1   1   1   2   3   2   1
   2   1   1   1   1   2   1   3   2   1   3   1   4   1   2   5   5   5
  19 148 192 117  28   1   1   2   1   4   1   1   3   1   1   1   1   1
   2   2   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
   1   1   1   1   1   1   1   1   1   1]
```

- 첫 번째 행에 있는 픽셀 100개에 들어 있는 값을 출력했습니다.
- 이 넘파이 배열은 흑백 사진을 담고 있으므로 0\~255까지의 정수값을 가집니다. 


```python
plt.imshow(fruits[0], cmap='gray')
plt.show()
```

- 정수값에 대해 더 설명하기 전에 먼저 첫 번째 이미지를 그림으로 그려서 이 숫자와 비교합니다.
- 맷 플롯립의 `imshow()` 함수를 사용하면 넘파이 배열로 지정된 이미지를 쉽게 그릴 수 있습니다. 흑백 이미지이므로 `cmap` 매개변수를 `gray`로 지정합니다.
- 0에 가까울수록 검게 나타나고 높은 값은 밝게 표시됩니다.

- 첫 번째 행에 있는 필셀 100개에 들어 있는 값을 출력했습니다. 이 넘파이 배열은 흑백 사진을 담고 있으므로 0\~255까지의 정수값을 가집니다. 
- 맷플롯립의 `imshow()` 함수를 사용하면 넘파이 배열로 저장된 이미지를 쉽게 그릴 수 있습니다. 흑백 이미지이므로 cmap 매개변수를 `gray`로 지정합니다.



```python
plt.imshow(fruits[0], cmap='gray_r')
plt.show()
```



```python
fig, axs = plt.subplots(1, 2)
axs[0].imshow(fruits[100], cmap='gray_r')
axs[1].imshow(fruits[200], cmap='gray_r')
plt.show()
```

## 픽셀 값 분석하기

```python
apple = fruits[0:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1, 100*100)
```

```python
print(apple.shape)
```

```
(100, 10000)
```

```python
print(apple.mean(axis=1))
```

```
[ 88.3346  97.9249  87.3709  98.3703  92.8705  82.6439  94.4244  95.5999
  90.681   81.6226  87.0578  95.0745  93.8416  87.017   97.5078  87.2019
  88.9827 100.9158  92.7823 100.9184 104.9854  88.674   99.5643  97.2495
  94.1179  92.1935  95.1671  93.3322 102.8967  94.6695  90.5285  89.0744
  97.7641  97.2938 100.7564  90.5236 100.2542  85.8452  96.4615  97.1492
  90.711  102.3193  87.1629  89.8751  86.7327  86.3991  95.2865  89.1709
  96.8163  91.6604  96.1065  99.6829  94.9718  87.4812  89.2596  89.5268
  93.799   97.3983  87.151   97.825  103.22    94.4239  83.6657  83.5159
 102.8453  87.0379  91.2742 100.4848  93.8388  90.8568  97.4616  97.5022
  82.446   87.1789  96.9206  90.3135  90.565   97.6538  98.0919  93.6252
  87.3867  84.7073  89.1135  86.7646  88.7301  86.643   96.7323  97.2604
  81.9424  87.1687  97.2066  83.4712  95.9781  91.8096  98.4086 100.7823
 101.556  100.7027  91.6098  88.8976]
```

```python
plt.hist(np.mean(apple, axis=1), alpha=0.8)
plt.hist(np.mean(pineapple, axis=1), alpha=0.8)
plt.hist(np.mean(banana, axis=1), alpha=0.8)
plt.legend(['apple', 'pineapple', 'banana'])
plt.show()
```

```python
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].bar(range(10000), np.mean(apple, axis=0))
axs[1].bar(range(10000), np.mean(pineapple, axis=0))
axs[2].bar(range(10000), np.mean(banana, axis=0))
plt.show()
```

```python
apple_mean = np.mean(apple, axis=0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis=0).reshape(100, 100)
banana_mean = np.mean(banana, axis=0).reshape(100, 100)

fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].imshow(apple_mean, cmap='gray_r')
axs[1].imshow(pineapple_mean, cmap='gray_r')
axs[2].imshow(banana_mean, cmap='gray_r')
plt.show()
```

## 평균값과 가까운 사진 고르기

```python
abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis=(1,2))
print(abs_mean.shape)
```

```
(300,)
```

```python
pple_index = np.argsort(abs_mean)[:100]
fig, axs = plt.subplots(10, 10, figsize=(10,10))
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[apple_index[i*10 + j]], cmap='gray_r')
        axs[i, j].axis('off')
plt.show()
```