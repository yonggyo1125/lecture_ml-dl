# 훈련 세트와 테스트 세트

## 키워드 정리

### 지도 학습

- 입력과 타킷을 전달하여 모델을 훈련한 다음 새로운 데이터를 예측하는 데 활용
- 예) k-최근접 이웃

### 비지도 학습

- 타킷 데이터가 없음
- 무엇을 예측하는 것이 아니라 입력 데이터에서 어떤 특징을 찾는 데 주로 활용

### 훈련 세트

- 모델을 훈련할 때 사용하는 데이터입니다.
- 보통 훈련세트가 클수록 좋습니다. 따라서 테스트 세트를 제외한 모든 데이터를 사용합니다.

### 테스트 세트

- 전체 데이터에서 20% ~ 30%를 테스트 세트로 사용하는 경우가 많습니다.
- 전체 데이터가 아주 크다면 1%만 덜어내도 충분할 수 있습니다.

## 핵심 패키지와 함수

### numpy

#### seed()

- 넘파이에서 난수를 생성하기 위한 정수 초깃값을 지정합니다.
- 초기값이 같으면 동일한 난수를 뽑을 수 있습니다.
- 랜덤 함수의 결과를 동일하게 재현하고 싶을 때 사용합니다.

#### arange()

- 일정한 간격의 정수 또는 실수 배열을 만듭니다.
- 기본 간격은 1입니다.
- 매개변수가 하나이면 종료 숫자를 의미합니다.
- 0에서 종료 숫자까지 배열을 만듭니다. 종료 숫자는 배열에 포함되지 않습니다.

```python
print(np.arange(3))
```

```
[0, 1, 2]
```

- 매개변수가 2개면 시작 숫자, 종료 숫자를 의미합니다.

```python
print(np.arange(1, 3))
```

```
[1, 2]
```

- 매개변수가 3개이면 마지막 매개변수가 간격을 나타냅니다.

```python
print(np.arange(1, 3, 0.2))
```

```
[1., 1.2, 1.4, 1.6, 1.8, 2., 2.2, 2.4, 2.6, 2.8]
```

#### shuffle()

- 주어진 배열을 랜덤하게 섞습니다.
- 다차원 배열일 경우 첫 번쨰 축(행)에 대해서만 섞습니다.

```python
arr = np.array([[1, 2], [3, 4], [5, 6]])
np.random.shuffle(arr)
print(arr)
```

```
[[3,4], [5,6], [1,2]]
```

## 훈련 세트와 테스트 세트

- 같은 데이터로 테스트하면 모두 맞힐 수 밖에 없음
- 즉, 연습문제와 시험문제가 달라야 올바르게 평가할 수 있음
- 머신러닝 알고리즘의 성능을 제대로 평가하려면 훈련 데이터와 평가에 사용할 데이터가 각각 달라야 한다.
- **테스트 세트**(test set) : 평가에 사용되는 데이터
- **훈련 세트**(train set) : 훈련에 사용되는 데이터

```python
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
```

```python
fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]
fish_target = [1]*35 + [0]*14
```

- 도미와 방어 데이터를 합치고, 각 생선의 길이와 무게를 하나의 리스트로 담은 2차원 리스트로 만듭니다.
- **샘플**(sample): 하나의 생선 데이터, 전체 49개의 샘플이 있음, 이 중에서 **35개를 훈련 세트**로 사용하고 **나머지 14개를 테스트 세트**로 사용

```python
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()
```

```python
print(fish_data[4])
```

- 인덱싱 테스트

```
[29.0, 430.0]
```

```python
print(fish_data[0:5])
```

- 슬라이싱 테스트
- 마지막 인덱스는 포함되지 않습니다.

```
[[25.4, 242.0], [26.3, 290.0], [26.5, 340.0], [29.0, 363.0], [29.0, 430.0]]
```

```python
print(fish_data[:5])
```

- 처음부터 시작하는 것이면 앞의 0은 생략 가능

```
[[25.4, 242.0], [26.3, 290.0], [26.5, 340.0], [29.0, 363.0], [29.0, 430.0]]
```

```python
print(fish_data[44:])
```

- 마지막 원소까지 포함한다면 두번째 인덱스 생략 가능

```
[[12.2, 12.2], [12.4, 13.4], [13.0, 12.2], [14.3, 19.7], [15.0, 19.9]]
```

```python
# 훈련 세트로 입력값 중 0부터 34번째 인덱스까지 사용
train_input = fish_data[:35]

# 훈련 세트로 타깃값 중 0부터 34번째 인덱스까지 사용
train_target = fish_target[:35]

# 테스트 세트로 입력값 중 35번째부터 마지막 인덱스까지 사용
test_input = fish_data[35:]

# 테스트 세트로 타깃값 중 35번째부터 마지막 인덱스까지 사용
test_target = fish_target[35:]
```

- 처음 35개는 훈련 데이터, 나머지 14개는 테스트 데이터 처리

```python
kn.fit(train_input, train_target)
kn.score(test_input, test_target)
```

- 훈련 세트로 `fit()`메서드를 호출해 모델을 훈련
- 테스트 세트로 `score()` 메서드를 호출해 평가

```
0.0
```

> 정확도가 0.0으로 나왔는데, 이는 훈련 세트는 도미로, 테스트 세트는 빙어로 샘플이 편향되어 있어 훈련은 도미로, 평가를 빙어로 하기 진행되었기 때문

## 샘플링 편향

- 훈련 세트와 테스트 세트에 샘플이 골고루 섞여 있지 않으면 샘플링이 한쪽으로 치우치게 됩니다. 이를 **샘플링 편향**(sampling bias)이라고 부릅니다.
- 훈련 세트와 테스트 세트를 나누기 전에 데이터를 섞거나 골고르 샘플을 뽑아서 훈련세트와 테스트 세트를 만들어야 합니다.
- 특정 종류의 샘플이 과도하게 많은 샘플링 편향을 가지고 있다면 제대로된 지도 학습 모델을 만들 수 없습니다.

## 넘파이

- 파이썬의 대표적인 배열(array) 라이브러리
- 고차원의 배열을 손쉽게 만들고 조작할 수 있는 간편한 도구를 많이 제공

```python
import numpy as np
```

- 넘파이 라이브러리 임포트

```python
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)
```

- 파이썬 리스트를 전달하면 끝

```python
print(input_arr)
```

```
[[  25.4  242. ]
 [  26.3  290. ]
 [  26.5  340. ]
 [  29.   363. ]
 [  29.   430. ]
 [  29.7  450. ]
 [  29.7  500. ]
 [  30.   390. ]
 [  30.   450. ]
 [  30.7  500. ]
 [  31.   475. ]
 [  31.   500. ]
 [  31.5  500. ]
 [  32.   340. ]
 [  32.   600. ]
 [  32.   600. ]
 [  33.   700. ]
 [  33.   700. ]
 [  33.5  610. ]
 [  33.5  650. ]
 [  34.   575. ]
 [  34.   685. ]
 [  34.5  620. ]
 [  35.   680. ]
 [  35.   700. ]
 [  35.   725. ]
 [  35.   720. ]
 [  36.   714. ]
 [  36.   850. ]
 [  37.  1000. ]
 [  38.5  920. ]
 [  38.5  955. ]
 [  39.5  925. ]
 [  41.   975. ]
 [  41.   950. ]
 [   9.8    6.7]
 [  10.5    7.5]
 [  10.6    7. ]
 [  11.     9.7]
 [  11.2    9.8]
 [  11.3    8.7]
 [  11.8   10. ]
 [  11.8    9.9]
 [  12.     9.8]
 [  12.2   12.2]
 [  12.4   13.4]
 [  13.    12.2]
 [  14.3   19.7]
 [  15.    19.9]]
```

```python
print(input_arr.shape)
```

- 넘파이 배열 객체에는 배열의 크기를 알려주는 shape 속성 제공

```
(49, 2)  # (샘플 수, 특성 수)
```

- 49개의 샘플과 2개의 특성이 있음을 확인할 수 있다.

```python
np.random.seed(42)
index = np.arange(49)
np.random.shuffle(index)
```

- 생선 데이터 배열에서 랜덤하게 샘플을 선택해 훈련 세트와 테스트 세트로 만듭니다.
- 넘파이 `arange()`함수를 사용하여 0에서 부터 48까지 1씩 증가하는 인덱스를 만듭니다(0에서 부터 N-1까지 1씩 증가).
- random 패키지의 `shuffle()`함수는 주어진 배열을 무작위로 섞습니다.

> 넘파이에서 무작위 결과를 만드는 함수들은 실행할 때마다 다른 결과를 만듭니다. 일정한 결과를 얻으려면 초기에 랜덤 시드(random seed)를 지정하면 됩니다. 여기에서는 42로 지정했습니다.

```python
print(index)
```

- 만들어진 인덱스 출력

```
[13 45 47 44 17 27 26 25 31 19 12  4 34  8  3  6 40 41 46 15  9 16 24 33
 30  0 43 32  5 29 11 36  1 21  2 37 35 23 39 10 22 18 48 20  7 42 14 28
 38]
```

```python
print(input_arr[[1,3]])
```

- 잘 섞여 있는지 확인, 배열 인덱스 1,2로 조회가 되는지 확인한다.

```
[[ 26.3 290. ]
 [ 29.  363. ]]
```

```python
train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]
```

- 리스트 대신 넘파이 배열을 인덱스로 전달
- index 배열의 처음 35개를 `input_arr`과 `target_arr`에 전달하여 랜덤하게 35개의 샘플 훈련 세트를 만듭니다.

```python
print(input_arr[13], train_input[0])
```

- 테스트

```
[ 32. 340.] [ 32. 340.]
```

```python
test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]
```

- 나머지 14개를 테스트 세트로 만듭니다.

```python
import matplotlib.pyplot as plt

plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(test_input[:, 0], test_input[:, 1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

- 훈련 세트와 테스트 세트에 도미와 방버가 잘 섞여 있는지 산점도로 확인
- 파란색: 훈련 세트
- 주황색: 테스트 세트

## 두 번째 머신러닝 프로그램

```python
kn.fit(train_input, train_target)
```

- `fit()` 메서드를 실행할 때마다 `KNeighborsClassifier` 클래스의 객체는 이전에 학습한 모든 것을 잃어버린다.
- 이전 모델을 그대로 두고 싶다면 `KNeighborsClassifier` 클래스 객체를 새로 만들어야 한다.
- 여기에서는 이미 만든 kn객체를 그대로 사용
- 인덱스를 섞어 만든 train_input과 train_target으로 훈련 시킴

```python
kn.score(test_input, test_target)
```

- test_input, test_target으로 이 모델을 테스트 함

```
1.0
```

- 100%의 정확도로 테스트 세트에 있는 모든 생선을 맞힘

```python
kn.predict(test_input)
```

- `predict()`메서드로 테스트 세트의 예측과 실제 타킷을 확인

```
array([0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0])
```

```python
test_target
```

```
array([0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0])
```

- 테스트 세트에 대한 예측 결과와 정답이 일치합니다.
- `predict()`메서드가 반환하는 값은 단순한 파이썬 리스트가 아닌 넘파이 배열을 의미
- 사이킷런 모델의 입력과 출력은 모두 넘파이 배열
