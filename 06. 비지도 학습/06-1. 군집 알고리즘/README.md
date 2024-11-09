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

## 과일 사진 데이터 준비하기

```python
!wget https://bit.ly/fruits_300_data -O fruits_300.npy
```

```python
import numpy as np
import matplotlib.pyplot as plt
```

```python
fruits = np.load('fruits_300.npy')
```

```python
print(fruits.shape)
```

```
(300, 100, 100)
```

```python
print(fruits[0, 0, :])
```

```python
[  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   2   1
   2   2   2   2   2   2   1   1   1   1   1   1   1   1   2   3   2   1
   2   1   1   1   1   2   1   3   2   1   3   1   4   1   2   5   5   5
  19 148 192 117  28   1   1   2   1   4   1   1   3   1   1   1   1   1
   2   2   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
   1   1   1   1   1   1   1   1   1   1]
```

```python
plt.imshow(fruits[0], cmap='gray')
plt.show()
```

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
