# 순환 신경망으로 IMDB 리뷰 분류하기

## 키워드 정리

- **말뭉치** : 자연어 처리에서 사용하는 텍스트 데이터의 모음, 즉 훈련 데이터셋을 일컫습니다.
- **토큰** : 텍스트에서 공백으로 구분되는 문자열을 말합니다. 종종 소문자로 변환하고 구둣점은 삭제합니다.
- **원-핫 인코딩** : 어떤 클래스에 해당하는 원소만 1이고 나머지는 모두 0인 벡터입니다. 정수로 변환된 토큰을 원-핫 인코딩으로 변환하려면 어휘 사전의 크기의 벡터가 만들어집니다.
- **단어 임베딩** : 정수로 변환된 토큰을 비교적 작은 크기의 실수 밀집 벡터로 변환합니다. 이런 밀집 벡터는 단어 사이의 관계를 표현할 수 있기 때문에 자연어 처리에서 좋은 성능을 발휘합니다.

### Tensorflow

- **pad_sequences()** : 시퀀스 길이를 맞추기 위해 패딩을 추가합니다. 이 함수는 (샘플 개수, 타임스텝 개수) 크기의 2차원 배열을 기대합니다.
  - `maxlen` 매개변수로 원하는 시퀀스 길이를 지정할 수 있습니다. 이 값보다 긴 시퀀스는 잘리고 짧은 시퀀스는 패딩 됩니다. 이 매개변수를 지정하지 않으면 가장 긴 시퀀스의 길이가 됩니다.
  - `padding` 매개변수는 패딩을 추가할 위치를 지정합니다. 기본값인 `pre`는 시퀀스 앞에 패딩을 추가하고 `post`는 시퀀스 뒤에 패딩을 추가합니다.
  - `truncating` 매개변수는 긴 시퀀에서 잘라버릴 위치를 지정합니다. 기본값인 `pre`는 시퀀스 앞부분을 잘라내고 `post`는 시퀀스 뒷부분을 잘라냅니다.
- **to_categorical()** : 정수 시퀀스를 원-핫 인코딩으로 변환합니다. 토큰을 원-핫 인코딩하거나 타깃값을 원-핫 인코딩할 때 사용합니다.
  - `num_classes` 매개변수에서 클래스 개수를 지정할 수 있습니다. 지정하지 않으면 데이터에서 자동으로 찾습니다.
- **SimpleRNN** : 케라스의 기본 순환층 클래스입니다.
  - 첫 번째 매개변수에 뉴런의 개수를 지정합니다.
  - `activation` 매개변수에서 활성화 함수를 지정합니다. 기본값은 하이퍼볼릭 탄젠트인 `tanh`입니다.
  - `dropout` 매개변수에서 입력에 대한 드롭아웃 비율을 지정할 수 있습니다.
  - `return_sequences` 매개변수에서 모든 타임스텝의 은닉 상태를 출력할지 결정합니다. 기본값은 `False`입니다.
- **Embedding** : 단어 임베딩을 위한 클래스입니다.
  - 첫 번째 매개변수에서 어휘 사전의 크기를 지정합니다.
  - 두 번째 매개변수에서 `Embedding` 층이 출력할 밀집 벡터의 크기를 지정합니다.
  - `input_length` 매개변수에서 입력 시퀀스의 길이를 지정합니다. 이 매개변수는 `Embedding`층 바로 뒤에 `Flatten`이나 `Dense` 클래스가 올 때 꼭 필요합니다.

## 시작하기 전에

- 1절에서 순환 신경망의 작동 원리를 살펴보았습니다. 이번 절에서는 대표적인 순환 신경망 문제인 IMDB 리뷰 데이터셋을 사용해 가장 간단한 순환 신경망 모델을 훈련해 보겠습니다.
- 이 데이터셋을 두 가지 방법으로 변형하여 순환 신경망에 주입해 보겠습니다. 하나는 원-핫 인코딩이고 또 하나는 단어 임베딩입니다. 이 두 가지 방법의 차이점에 대해 설명하고 순환 신경망을 만들 때 고려해야 할 점을 알아보겠습니다.
- 그럼 먼저 이 절에서 사용할 IMDB 리뷰 데이터셋을 적재해 보겠습니다.

![스크린샷 2025-03-18 오후 9 50 02](https://github.com/user-attachments/assets/2af6c399-1cce-4048-8568-04f8295c13d9)

## IMDB 리뷰 데이터셋

- IMDB 리뷰 데이터셋은 유명한 인터넷 영화 데이터베이스인 `imdb.com`에서 수집한 리뷰를 감상평에 따라 긍정과 부정으로 분류해 놓은 데이터셋입니다. 총 50,000개의 샘플로 이루어져 있고 훈련 데이터와 테스트 데이터에 각각 25,000개씩 나누어져 있습니다.

> **자연어 처리와 말뭉치란 무엇인가요?**<br>**자연어 처리**(natual language processing, NLP)는 컴퓨터를 사용해 인간의 언어를 처리하는 분야입니다. 대표적인 세부 분야로는 음성 인식, 기계 번역, 감성 분석 등이 있습니다. IMDB 리뷰를 감상평에 따라 분류하는 작업은 감성 분석에 해당합니다. 자연어 처리 분야에서는 훈련 데이터를 종종 **말뭉치**(corpus)라고 부릅니다. 예를 들어 IMDB 리뷰 데이터셋이 하나의 말뭉치입니다.

- 사실 텍스트 자체를 신경망에 전달하지는 않습니다. 컴퓨터에서 처리하는 모든 것은 어떤 숫자 데이터입니다. 앞서 합성곱 신경망에서 이미지를 다룰 떄는 특별한 변환을 하지 않았습니다. 이미지가 정수 픽셀값으로 이루어져 있기 때문이죠. 텍스트 데이터의 경우 단어를 숫자 데이터로 바꾸는 일반적인 방법은 데이터에 등장하는 단어마다 고유한 정수를 부여하는 것입니다. 예를 들면 다음과 같습니다.

![스크린샷 2025-03-18 오후 9 55 13](https://github.com/user-attachments/assets/5ea4096f-c938-429d-9b4f-be8704b74f03)

- 앞의 두 문장에 등장하는 각 단어를 하나의 정수에 매핑했고, 동일한 단어는 동일한 정수에 매핑됩니다. 단어에 매핑되는 정수는 단어의 의미나 크기에 관련이 없습니다. 예를 들어 'He'를 10으로 매핑하고 'cat'을 13에 매핑하더라도 'cat'이 'He'보다 좋거나 크다는 뜻은 아닙니다. 이 정숫값 사이에는 어떤 관계도 없습니다. 일반적으로 영어 문장은 모두 소문자로 바꾸고 구둣점을 삭제한 다음 공백을 기준으로 분리합니다. 이렇게 분리된 단어를 **토큰**<sup>token</sup>이라고 부릅니다. 하나의 샘플은 여러 개의 토큰으로 이루어져 있고 1개의 토큰이 하나의 타임스텝에 해당합니다.

> 간단한 문제라면 영어 말뭉치에서 토큰을 단어와 같게 봐도 좋습니다. 한국어는 조금 다릅니다.

> 한글 문장은 어떻게 토큰을 분리하나요?<br>한글은 조사가 발달되어 있기 때문에 공백으로 나누는 것만으로는 부족합니다. 일반적으로 한글은 형태소 분석을 통해 토큰을 만듭니다.

- 토큰에 할당하는 정수 중에 몇 개는 특정한 용도로 예약되어 있는 경우가 많습니다. 예를 들어 0은 패딩(잠시 후에 설명합니다.), 1은 문장의 시작, 2는 어휘 사전에 없는 토큰을 나타냅니다.

> 어휘사전이란?<br>훈련 세트에서 고유한 단어를 뽑아 만든 목록을 어휘 사전이라고 말합니다. 예를 들어 텍스트 세트 안에 어휘 사전에 없는 단어가 있다면 2로 변환하여 신경망 모델에 주입합니다.

- 실제 IMDB 리뷰 데이터셋은 영어로 된 문장이지만 편리하게도 텐서플로에는 이미 정수로 바꾼 데이터가 포함되어 있습니다. `tensorflow.keras.datasets` 패키지 아래 imdb모듈을 임포트하여 이 데이터를 적재해 보겠습니다. 여기에서는 전체 데이터셋에서 가장 자주 등장하는 단어 300개만 사용하겠습니다. 이렇게 하기 위해 `load_data()` 함수의 `num_words` 매개변수를 300으로 지정합니다.

```python
from tensorflow.keras.datasets import imdb

(train_input, train_target), (test_input, test_target) = imdb.load_data(
    num_words=300)
```

- 먼저 훈련세트와 테스트 세트의 크기를 확인해 보겠습니다.

```python
print(train_input.shape, test_input.shape)
```

```
(25000,) (25000,)
```

- 앞서 말했듯이 이 데이터셋은 훈련 세트와 테스트 세트가 각각 25,000개의 샘플로 이루어져 있습니다. 그런데 배열이 1차원인 게 이상하게 보이지요? IMDB 리뷰 텍스트는 길이가 제각각입니다. 따라서 고정 크기의 2차원 배열에 담기 보다는 리뷰마다 별도의 파이썬 리스트로 담아야 메모리를 효율적으로 사용할 수 있습니다.

![스크린샷 2025-03-18 오후 10 05 58](https://github.com/user-attachments/assets/ee90d905-9729-4590-bbfc-facdb0f63a1d)

- 즉 앞의 그림처럼 이 데이터는 개별 리뷰를 담은 파이썬 리스트 객체로 이루어진 넘파이 배열입니다. 넘파이 배열은 정수나 실수 외에도 파이썬 객체를 담을 수 있습니다. 그럼 다음과 같이 첫 번째 리뷰의 길이를 출력해 보겠습니다.

```python
print(len(train_input[0]))
```

```
218
```

- 첫 번째 리뷰의 길이는 218개의 토큰으로 이루어져 있습니다. 두 번째 리뷰의 길이를 확인해 보겠습니다.

```python
print(len(train_input[1]))
```

```
189
```

- 몇 개 더 해볼 수도 있겠지만 리뷰마다 각각 길이가 다릅니다. 여기서 하나의 리뷰가 하나의 샘플이 됩니다. 서로 다른 길이의 샘플을 어떻게 신경망에 전달하는지 잠시 후에 살펴보겠습니다. 이제 첫 번째 리뷰에 담긴 내용을 출력해 보죠.

```python
print(train_input[0])
```

```
[1, 14, 22, 16, 43, 2, 2, 2, 2, 65, 2, 2, 66, 2, 4, 173, 36, 2, 5, 25, 100, 43, 2, 112, 50, 2, 2, 9, 35, 2, 2, 5, 150, 4, 172, 112, 167, 2, 2, 2, 39, 4, 172, 2, 2, 17, 2, 38, 13, 2, 4, 192, 50, 16, 6, 147, 2, 19, 14, 22, 4, 2, 2, 2, 4, 22, 71, 87, 12, 16, 43, 2, 38, 76, 15, 13, 2, 4, 22, 17, 2, 17, 12, 16, 2, 18, 2, 5, 62, 2, 12, 8, 2, 8, 106, 5, 4, 2, 2, 16, 2, 66, 2, 33, 4, 130, 12, 16, 38, 2, 5, 25, 124, 51, 36, 135, 48, 25, 2, 33, 6, 22, 12, 2, 28, 77, 52, 5, 14, 2, 16, 82, 2, 8, 4, 107, 117, 2, 15, 2, 4, 2, 7, 2, 5, 2, 36, 71, 43, 2, 2, 26, 2, 2, 46, 7, 4, 2, 2, 13, 104, 88, 4, 2, 15, 2, 98, 32, 2, 56, 26, 141, 6, 194, 2, 18, 4, 2, 22, 21, 134, 2, 26, 2, 5, 144, 30, 2, 18, 51, 36, 28, 2, 92, 25, 104, 4, 2, 65, 16, 38, 2, 88, 12, 16, 2, 5, 16, 2, 113, 103, 32, 15, 16, 2, 19, 178, 32]
```

- 앞서 설명했듯이 텐서플로에 있는 IMDB 리뷰 데이터는 이미 정수로 변환되어 있습니다. 앞서 `num_words=300`으로 지정했기 때문에 어휘 사전에는 300개의 단어만 들어가 있습니다. 따라서 어휘 사전에 없는 단어는 모두 2로 표시되어 나타납니다.

> 어떤 기준으로 300개의 단어를 고른 것인가요?<br>`imdb.load_data()` 함수는 전체 어휘 사전에 있는 단어에 등장 횟수 순서대로 나열한 다음 가장 많이 등장한 300개의 단어를 선택합니다.

- 이번에는 타깃 데이터를 출력해 보겠습니다.

```python
print(train_target[:20])
```

```
[1 0 0 1 0 0 1 0 1 0 1 0 0 0 0 0 1 1 0 1]
```

- 해결할 문제는 리뷰가 긍정인지 부정인지를 판단하는 겁니다. 그러면 이진 분류 문제로 볼 수 있으므로 타깃값이 0(부정)과 1(긍정)로 나누어집니다.
- 데이터를 더 살펴보기 전에 훈련세트에서 검증 세트를 떼어 놓도록 하죠. 원래 훈련 세트의 크기가 25,000개였으므로 20%를 검증 세트로 떼어 놓으면 훈련 세트의 크기는 20,000개로 줄어들 것입니다.

```python
from sklearn.model_selection import train_test_split

train_input, val_input, train_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)
```

- 이제 훈련 세트에 대해 몇 가지 조사를 해 보겠습니다. 먼저 각 리뷰의 길이를 계산해 넘파이 배열에 담겠습니다. 이렇게 하는 이유는 평균적인 리뷰의 길이와 가장 짧은 리뷰의 길이 그리고 가장 긴 리뷰의 길이를 확인하고 싶기 때문입니다. 이를 위해 넘파이 리스트 내포를 사용해 `train_input`의 원소를 순회하면서 길이를 재도록 하겠습니다.

```python
import numpy as np

lengths = np.array([len(x) for x in train_input])
```

- `lengths` 배열이 준비되었으므로 넘파이 `mean()` 함수와 `median()` 함수를 사용해 리뷰 길이의 평균과 중간값을 구해 보겠습니다.

```python
print(np.mean(lengths), np.median(lengths))
```

```
239.00925 178.0
```

- 리뷰의 평균 단어 개수는 239개이고 중간값이 178인 것으로 보아 이 리뷰의 길이 데이터는 한쪽에 치우친 분포를 보일 것 같습니다. `lengths` 배열을 히스토그램으로 표현해 보겠습니다.

```python
import matplotlib.pyplot as plt

plt.hist(lengths)
plt.xlabel('length')
plt.ylabel('frequency')
plt.show()
```

![스크린샷 2025-03-18 오후 10 15 41](https://github.com/user-attachments/assets/36511312-2f25-4408-a3e5-b52e1486ccef)

- 역시 한쪽으로 치우쳤군요. 대부분의 리뷰 길이는 300 미만입니다. 평균이 중간값보다 높은 이유는 오른쪽 끝에 아주 큰 데이터가 있기 떄문입니다. 어떤 리뷰는 1,000개의 단어가 넘기도 합니다.
- 리뷰는 대부분 짧아서 이 예제에서는 중간값보다 훨씬 짧은 100개의 단어만 사용하겠습니다. 하지만 여전히 100개의 단어보다 작은 리뷰가 있습니다. 이런 리뷰들의 길이를 100에 맞추기 위해 패딩이 필요합니다. 보통 패딩을 나타내는 토큰으로 0을 사용합니다.
- 물론 수동으로 훈련 세트에 있는 20,000개의 리뷰를 순회하면서 길이가 100이 되도록 잘라내거나 0으로 패딩할 수 있습니다. 하지만 자주 있는 번거로운 작업에는 항상 편리한 도구가 준비되어 있죠. 케라스는 시퀀스 데이터의 길이를 맞추는 `pad_sequences()` 함수를 제공합니다. 이 함수를 사용해 `train_input`의 길이를 100으로 맞추어 보겠습니다.

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_seq = pad_sequences(train_input, maxlen=100)
```

- 사용법은 간단합니다. `maxlen`에 원하는 길이를 지정하면 이보다 긴 경우는 잘라내고 짧은 경우는 0으로 패딩합니다. 패딩 된 결과가 어떻게 나타나는지 확인해 보겠습니다. 먼저 `train_seq`의 크기를 확인해 보죠.

```python
print(train_seq.shape)
```

```
(20000, 100)
```

- `train_input`은 파이썬 리스트의 배열이었지만 길이를 100으로 맞춘 `train_seq`는 이제 (20000, 100) 크기의 2차원 배열이 되었습니다.

![스크린샷 2025-03-18 오후 10 21 54](https://github.com/user-attachments/assets/8b5016a0-150f-4374-b9d7-50b2aae095e4)

- `train_seq`에 있는 첫 번쨰 샘플을 출력해 보겠습니다.

```python
print(train_seq[0])
```

```
[ 10   4  20   9   2   2   2   5  45   6   2   2  33   2   8   2 142   2
   5   2  17  73  17   2   5   2  19  55   2   2  92  66 104  14  20  93
  76   2 151  33   4  58  12 188   2 151  12   2  69   2 142  73   2   6
   2   7   2   2 188   2 103  14  31  10  10   2   7   2   5   2  80  91
   2  30   2  34  14  20 151  50  26 131  49   2  84  46  50  37  80  79
   6   2  46   7  14  20  10  10   2 158]
```

- 이 샘플의 앞뒤에 패딩값 0이 없는 것으로 보아 100보다는 길었을 것 같습니다. 그럼 원래 샘플의 앞부분이 잘렸을까요? 뒷부분이 잘렸을까요? `train_input`에 있는 원본 샘플의 끝을 확인해 보죠.

```python
print(train_input[0][-10:])
```

```
[6, 2, 46, 7, 14, 20, 10, 10, 2, 158]
```

- 음수 인덱스와 슬라이싱을 사용해 `train_input[0]`에 있는 마지막 10개의 토큰을 출력했습니다. `train_seq[0]`의 출력값과 비교하면 정확히 일치합니다. 그렇다면 샘플의 앞부분이 잘렸다는 것을 짐작할 수 있겠네요.
- `pad_sequences()` 함수는 기본적으로 `maxlen`보다 긴 시퀀스의 앞부분을 자릅니다. 이렇게 하는 이유는 일반적으로 시퀀스의 뒷부분의 정보가 더 유용하리라 기대하기 때문입니다. 영화 리뷰 데이터를 생각해 보면 리뷰 끝에 뭔가 결정적인 소감을 말할 가능성이 높다고 볼 수 있습니다. 만약 시퀀스의 뒷부분을 잘라내고 싶다면 `pad_sequences()` 함수의 `truncating` 매개변수의 값을 기본값 `pre`가 아닌 `post`로 바꾸면 됩니다.
- 이번에는 `train_seq`에 있는 여섯 번째 샘플을 출력해 보겠습니다.

```python
print(train_seq[5])
```

```
[  0   0   0   0   1   2 195  19  49   2   2 190   4   2   2   2 183  10
  10  13  82  79   4   2  36  71   2   8   2  25  19  49   7   4   2   2
   2   2   2  10  10  48  25  40   2  11   2   2  40   2   2   5   4   2
   2  95  14   2  56 129   2  10  10  21   2  94   2   2   2   2  11 190
  24   2   2   7  94   2   2  10  10  87   2  34  49   2   7   2   2   2
   2   2   2   2  46  48  64  18   4   2]
```

- 앞부분에 0이 있는 것으로 보아 이 샘플의 길이는 100이 안 되겠군요. 역시 같은 이유로 패딩 토큰은 시퀀스의 뒷부분이 아니라 앞부분에 추가됩니다. 시퀀스의 마지막에 있는 단어가 셀의 은닉 상태에 가장 큰 영향을 미치게 되므로 마지막에 패딩을 추가하는 것은 일반적으로 선호하지 않습니다. 하지만 원한다면 `pad_sequences()` 함수의 `padding` 매개변수의 기본값인 `pre`를 `post`로 바꾸면 샘플의 귓부분에 패딩을 추가할 수 있습니다.
- 그럼 이런 방식대로 검증 세트의 길이도 100으로 맟추어 보죠.

```python
val_seq = pad_sequences(val_input, maxlen=100)
```

- 이제 훈련 세트와 검증 세트 준비를 마쳤습니다. 이제 본격적으로 순환 신경망 모델을 만들어 보겠습니다.

## 순환 신경망 만들기

- 케라스는 여러 종류의 순환층 클래스를 제공합니다. 그중에 가장 간단한 것은 `SimpleRNN` 클래스입니다. 이 클래스는 7장 1절에서 설명한 것과 거의 비슷한 기능을 수행합니다. IMDB 리뷰 분류 문제는 이진 분류이므로 마지막 출력층은 1개의 뉴런을 가지고 시그모이드 활성화 함수를 사용해야 합니다. 먼저 케라스의 `Sequential` 클래스로 만든 신경망 코드를 살펴보죠.

```python
from tensorflow import keras

model = keras.Sequential()

model.aadd(keras.layers.Input(shape=(100,300)))
model.add(keras.layers.SimpleRNN(8))
model.add(keras.layers.Dense(1, activation='sigmoid'))
```

- 이 코드는 지금까지 보았던 구성과 매우 비슷합니다. 달라진 것은 Dense나 Conv2D 클래스 대신 `SimpleRNN` 클래스를 사용했습니다. 첫 번째 매개변수에는 사용할 뉴런의 개수를 지정하고, `Input`의 `shape`에 입력 차원을 (100, 300)으로 지정했습니다. 첫 번째 차원이 100인 것은 앞에서 샘플의 길이를 100으로 지정했기 때문입니다. 그럼 300은 어디서 온 숫자일까요? 이에 대해서는 잠시 후에 설명하겠습니다.
- 순환층도 당연히 활성화 함수를 사용해야 합니다. `SimpleRNN` 클래스의 `activation` 매개변수의 기본값은 `tanh`로 하이퍼볼릭 탄젠트 함수를 사용합니다. 여기서는 기본값을 그대로 사용합니다.
- 그럼 Input의 shape의 두번째 차원인 300은 어디에서 온 숫자일까요? 이전 섹션에 만든 `train_seq`와 `val_seq`에는 한 가지 큰 문제가 있습니다. 토큰을 정수로 변환한 이 데이터를 신경망에 주입하면 큰 정수가 큰 활성화 출력을 만들기 때문입니다.
- 분명히 이 정수 사이에는 어떤 관련이 없습니다. 20번 토큰을 10번 토큰보다 더 중요시해야 할 이유가 없습니다. 따라서 단순한 정수값을 신경망에 입력하기 위해서는 다른 방식을 찾아야 합니다.
- 정수값에 있는 크기 속성을 없애고 각 정수를 고유하게 표현하는 방법은 7장에서 잠깐 보았던 원-핫 인코딩입니다. 예를 들어 `train_seq[0]`의 첫 번째 토큰인 10을 원-핫 인코딩으로 바꾸면 다음과 같습니다.

![스크린샷 2025-03-18 오후 10 38 43](https://github.com/user-attachments/assets/debda8a6-e4de-43df-9073-e8c9b913e6fe)

- 열한 번째 원소만 1이고 나머지는 모두 0인 배열입니다. 이 배열의 길이는 얼마일까요?
- `imdb.load_data()` 함수에서 300개의 단어만 사용하도록 지정했기 때문에 고유한 단어는 모두 300개입니다. 즉 훈련 데이터에 포함될 수 있는 정수값의 범위는 0(패딩 토큰)에서 299까지입니다. 따라서 이 범위를 원-핫 인코딩으로 표현하려면 배열의 길이가 300이어야 합니다.
- 7장 1절에서 "I am a boy"에 있는 각 단어를 숫자 3개를 사용해 표현한다고 예를 들었던 것을 기억하나요? 여기에서도 개념은 동일합니다. 토큰마다 300개의 숫자를 사용해 표현하는 것이죠. 다만 300개 중에 하나만 1이고 나머지는 모두 0으로 만들어 정수 사이에 있던 크기 속성을 없애는 원-핫 인코딩을 사용합니다.
- 혹시 예상했을 수 있겠지만 케라스에는 이미 원-핫 인코딩을 위한 유틸리티를 제공합니다. 따라서 수동으로 위와 같은 배열을 만들 필요가 없죠. 이 유틸리티는 바로 `keras.utils` 패키지 아래에 있는 `to_categorical()` 함수입니다. 정수 배열을 입력하면 자동으로 원-핫 인코딩된 배열을 반환해 줍니다.

```python
train_oh = keras.utils.to_categorical(train_seq)
```

- 먼저 `train_seq`를 원-핫 인코딩으로 변환하여 `train_oh` 배열을 만들었습니다. 이 배열의 크기를 출력해 보겠습니다.

```python
print(train_oh.shape)
```

```
(20000, 100, 300)
```

- 정수 하나마다 모두 300차원의 배열로 변경되었기 때문에 (20000, 100) 크기의 `train_seq`가 (20000, 100, 300) 크기의 `train_oh`로 바뀌었습니다. 이렇게 샘플 데이터의 크기가 1차원 정수 배열 (100,) 에서 2차원 배열 (100, 300)로 바뀌어야 하므로 `Input`의 `shape` 매개변수의 값을 (100, 300)으로 지정한 것입니다.
- `train_oh`의 첫 번째 샘플의 첫 번째 토큰 10이 잘 인코딩되었는지 출력해 보죠.

```python
print(train_oh[0][0][:12])
```

```
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
```

- 처음 12개 원소를 출력해 보면 열한 번째 원소가 1인 것을 확인할 수 있습니다. 나머지 원소는 모두 0일까요? 넘파이 `sum()` 함수로 모든 원소의 값을 더해서 1이 되는지 확인해 보죠.

```python
print(np.sum(train_oh[0][0]))
```

```
1.0
```

- 네, 토큰 10이 잘 인코딩된 것 같습니다. 열한 번째 원소만 1이고 나머지는 모두 0이어서 원-핫 인코딩된 배열의 값을 모두 더한 결과가 1이 되었습니다. 같은 방식으로 val_seq도 원-핫 인코딩으로 바꾸어 놓겠습니다.

```python
val_oh = keras.utils.to_categorical(val_seq)
```

- 이제 훈련에 사용할 훈련 세트와 검증 세트가 모두 준비되었습니다. 앞서 만든 모델의 구조를 출력해 보죠.

```python
model.summary()
```

![스크린샷 2025-03-18 오후 10 49 42](https://github.com/user-attachments/assets/eae7f766-aa15-49c8-9346-64d4de2b056c)

![스크린샷 2025-03-18 오후 10 49 53](https://github.com/user-attachments/assets/0e05d45f-e58b-4386-9ad3-c158c03376c1)

- `SimpleRNN`에 전달할 샘플의 크기는 (100, 300)이지만 이 순환층은 마지막 타임스텝의 은닉 상태만 출력합니다. 이 때문에 출력 크기가 순환층의 뉴런 개수와 동일한 8임을 확인할 수 있습니다.
- 순환층에 사용된 모델 파라미터의 개수를 계산해 보죠. 입력 토큰은 300차원의 원-핫 인코딩 배열입니다. 이 배열이 순환층의 뉴런 8개와 완전히 연결되기 때문에 총 300 \* 8 = 2,400개의 가중치가 있습니다. 순환층의 은닉 상태는 다시 다음 타임스텝에 사용되기 위해 또 다른 가중치와 곱해집니다. 이 은닉 상태도 순환층의 뉴런과 완전히 연결되기 때문에 8(은닉 상태 크기) X 8(뉴런 개수) = 64개의 가중치가 필요합니다. 마지막으로 뉴런마다 하나의 절편이 있습니다. 따라서 모두 2,400 + 64 + 8 = 2,472개의 모델 파라미터가 필요합니다.
- 케라스 API를 사용해 순환 신경망 모델을 손쉽게 만들었습니다. 이전에 만들었던 완전 연결 신경망에 비해 크게 바뀐 것은 없습니다. `Dense` 층 대신에 `SimpleRNN` 층을 사용했고 입력 데이터의 차원을 원-핫 인코딩으로 바꾸어 주었습니다. 다음 섹션에서 이 순환 신경망 모델을 훈련해 보겠습니다.

## 순환 신경망 훈련하기

- 순환 신경망의 훈련은 완전 연결 신경망이나 합성곱 신경망과 크게 다르지 않습니다. 모델을 만드는 것은 달라도 훈련하는 방법은 모두 같습니다. 이것이 케라스 API를 사용하는 장점이죠. 다음 코드처럼 모델을 컴파일하고 훈련하는 전체 구조가 동일합니다.
- 이 예에서는 기본 `RMSprop`의 학습률 0.001을 사용하지 않기 위해 별도 `RMSprop` 객체를 만들어 학습률을 0.0001로 지정하였습니다. 그 다음 에포크 횟수를 100으로 늘리고 배치 크기는 64개로 설정했습니다. 그 밖에 체크포인트와 조기 종료를 구성하는 코드는 거의 동일합니다.

```python
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rmsprop, loss='binary_crossentropy',
              metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-simplernn-model.keras',
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

history = model.fit(train_oh, train_target, epochs=100, batch_size=64,
                    validation_data=(val_oh, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])
```

```
Epoch 1/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 15s 39ms/step - accuracy: 0.4945 - loss: 0.7135 - val_accuracy: 0.4924 - val_loss: 0.7053
Epoch 2/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 12s 18ms/step - accuracy: 0.5020 - loss: 0.7038 - val_accuracy: 0.4960 - val_loss: 0.7010
Epoch 3/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.5059 - loss: 0.6993 - val_accuracy: 0.4962 - val_loss: 0.6984
Epoch 4/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 11s 21ms/step - accuracy: 0.5124 - loss: 0.6964 - val_accuracy: 0.5038 - val_loss: 0.6962
Epoch 5/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.5154 - loss: 0.6943 - val_accuracy: 0.5110 - val_loss: 0.6939
Epoch 6/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.5195 - loss: 0.6925 - val_accuracy: 0.5120 - val_loss: 0.6925
Epoch 7/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 7s 23ms/step - accuracy: 0.5228 - loss: 0.6915 - val_accuracy: 0.5162 - val_loss: 0.6917
Epoch 8/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 9s 20ms/step - accuracy: 0.5255 - loss: 0.6905 - val_accuracy: 0.5198 - val_loss: 0.6910
Epoch 9/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 9s 16ms/step - accuracy: 0.5305 - loss: 0.6897 - val_accuracy: 0.5232 - val_loss: 0.6904
Epoch 10/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.5355 - loss: 0.6888 - val_accuracy: 0.5256 - val_loss: 0.6899
Epoch 11/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 18ms/step - accuracy: 0.5392 - loss: 0.6881 - val_accuracy: 0.5256 - val_loss: 0.6893
Epoch 12/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.5413 - loss: 0.6873 - val_accuracy: 0.5272 - val_loss: 0.6888
Epoch 13/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.5443 - loss: 0.6866 - val_accuracy: 0.5328 - val_loss: 0.6883
Epoch 14/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.5470 - loss: 0.6859 - val_accuracy: 0.5344 - val_loss: 0.6878
Epoch 15/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.5505 - loss: 0.6851 - val_accuracy: 0.5372 - val_loss: 0.6873
Epoch 16/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.5537 - loss: 0.6844 - val_accuracy: 0.5432 - val_loss: 0.6867
Epoch 17/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 16ms/step - accuracy: 0.5544 - loss: 0.6837 - val_accuracy: 0.5466 - val_loss: 0.6861
Epoch 18/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.5559 - loss: 0.6829 - val_accuracy: 0.5478 - val_loss: 0.6856
Epoch 19/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 18ms/step - accuracy: 0.5578 - loss: 0.6821 - val_accuracy: 0.5490 - val_loss: 0.6850
Epoch 20/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 17ms/step - accuracy: 0.5599 - loss: 0.6813 - val_accuracy: 0.5522 - val_loss: 0.6845
Epoch 21/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 9s 29ms/step - accuracy: 0.5630 - loss: 0.6805 - val_accuracy: 0.5524 - val_loss: 0.6839
Epoch 22/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 9s 29ms/step - accuracy: 0.5656 - loss: 0.6797 - val_accuracy: 0.5530 - val_loss: 0.6832
Epoch 23/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.5670 - loss: 0.6789 - val_accuracy: 0.5538 - val_loss: 0.6825
Epoch 24/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.5691 - loss: 0.6780 - val_accuracy: 0.5578 - val_loss: 0.6819
Epoch 25/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 17ms/step - accuracy: 0.5706 - loss: 0.6771 - val_accuracy: 0.5604 - val_loss: 0.6812
Epoch 26/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 17ms/step - accuracy: 0.5730 - loss: 0.6761 - val_accuracy: 0.5622 - val_loss: 0.6804
Epoch 27/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 11s 18ms/step - accuracy: 0.5747 - loss: 0.6750 - val_accuracy: 0.5632 - val_loss: 0.6797
Epoch 28/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.5747 - loss: 0.6740 - val_accuracy: 0.5634 - val_loss: 0.6787
Epoch 29/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 17ms/step - accuracy: 0.5768 - loss: 0.6728 - val_accuracy: 0.5662 - val_loss: 0.6774
Epoch 30/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 17ms/step - accuracy: 0.5796 - loss: 0.6713 - val_accuracy: 0.5716 - val_loss: 0.6759
Epoch 31/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 11s 18ms/step - accuracy: 0.5801 - loss: 0.6695 - val_accuracy: 0.5790 - val_loss: 0.6737
Epoch 32/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 11s 20ms/step - accuracy: 0.5860 - loss: 0.6671 - val_accuracy: 0.5836 - val_loss: 0.6705
Epoch 33/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.5962 - loss: 0.6634 - val_accuracy: 0.5974 - val_loss: 0.6647
Epoch 34/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.6079 - loss: 0.6564 - val_accuracy: 0.6116 - val_loss: 0.6511
Epoch 35/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 16ms/step - accuracy: 0.6356 - loss: 0.6393 - val_accuracy: 0.6616 - val_loss: 0.6188
Epoch 36/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 20ms/step - accuracy: 0.6708 - loss: 0.6103 - val_accuracy: 0.6684 - val_loss: 0.6126
Epoch 37/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 19ms/step - accuracy: 0.6774 - loss: 0.6029 - val_accuracy: 0.6758 - val_loss: 0.6062
Epoch 38/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.6811 - loss: 0.5968 - val_accuracy: 0.6796 - val_loss: 0.6015
Epoch 39/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.6876 - loss: 0.5911 - val_accuracy: 0.6864 - val_loss: 0.5966
Epoch 40/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.6914 - loss: 0.5860 - val_accuracy: 0.6854 - val_loss: 0.5926
Epoch 41/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 19ms/step - accuracy: 0.6976 - loss: 0.5811 - val_accuracy: 0.6904 - val_loss: 0.5889
Epoch 42/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 17ms/step - accuracy: 0.7006 - loss: 0.5762 - val_accuracy: 0.6974 - val_loss: 0.5852
Epoch 43/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 17ms/step - accuracy: 0.7060 - loss: 0.5720 - val_accuracy: 0.6988 - val_loss: 0.5817
Epoch 44/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 16ms/step - accuracy: 0.7100 - loss: 0.5681 - val_accuracy: 0.6990 - val_loss: 0.5787
Epoch 45/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 16ms/step - accuracy: 0.7115 - loss: 0.5654 - val_accuracy: 0.7022 - val_loss: 0.5757
Epoch 46/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 17ms/step - accuracy: 0.7131 - loss: 0.5625 - val_accuracy: 0.7028 - val_loss: 0.5734
Epoch 47/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 17ms/step - accuracy: 0.7149 - loss: 0.5601 - val_accuracy: 0.7058 - val_loss: 0.5717
Epoch 48/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 11s 18ms/step - accuracy: 0.7164 - loss: 0.5574 - val_accuracy: 0.7084 - val_loss: 0.5693
Epoch 49/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 18ms/step - accuracy: 0.7152 - loss: 0.5561 - val_accuracy: 0.7106 - val_loss: 0.5675
Epoch 50/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 19ms/step - accuracy: 0.7186 - loss: 0.5541 - val_accuracy: 0.7120 - val_loss: 0.5662
Epoch 51/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 11s 22ms/step - accuracy: 0.7204 - loss: 0.5524 - val_accuracy: 0.7126 - val_loss: 0.5653
Epoch 52/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.7213 - loss: 0.5511 - val_accuracy: 0.7152 - val_loss: 0.5647
Epoch 53/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.7232 - loss: 0.5497 - val_accuracy: 0.7156 - val_loss: 0.5632
Epoch 54/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 18ms/step - accuracy: 0.7241 - loss: 0.5483 - val_accuracy: 0.7166 - val_loss: 0.5624
Epoch 55/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 16ms/step - accuracy: 0.7257 - loss: 0.5472 - val_accuracy: 0.7204 - val_loss: 0.5615
Epoch 56/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.7263 - loss: 0.5463 - val_accuracy: 0.7210 - val_loss: 0.5605
Epoch 57/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.7265 - loss: 0.5451 - val_accuracy: 0.7202 - val_loss: 0.5601
Epoch 58/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 19ms/step - accuracy: 0.7279 - loss: 0.5440 - val_accuracy: 0.7214 - val_loss: 0.5590
Epoch 59/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 9s 16ms/step - accuracy: 0.7287 - loss: 0.5431 - val_accuracy: 0.7224 - val_loss: 0.5579
Epoch 60/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 16ms/step - accuracy: 0.7301 - loss: 0.5422 - val_accuracy: 0.7210 - val_loss: 0.5584
Epoch 61/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.7279 - loss: 0.5410 - val_accuracy: 0.7238 - val_loss: 0.5574
Epoch 62/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.7286 - loss: 0.5400 - val_accuracy: 0.7270 - val_loss: 0.5558
Epoch 63/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 19ms/step - accuracy: 0.7319 - loss: 0.5391 - val_accuracy: 0.7256 - val_loss: 0.5556
Epoch 64/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 9s 16ms/step - accuracy: 0.7310 - loss: 0.5384 - val_accuracy: 0.7262 - val_loss: 0.5544
Epoch 65/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.7310 - loss: 0.5375 - val_accuracy: 0.7292 - val_loss: 0.5532
Epoch 66/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 18ms/step - accuracy: 0.7332 - loss: 0.5369 - val_accuracy: 0.7298 - val_loss: 0.5530
Epoch 67/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.7335 - loss: 0.5361 - val_accuracy: 0.7324 - val_loss: 0.5524
Epoch 68/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.7333 - loss: 0.5355 - val_accuracy: 0.7312 - val_loss: 0.5516
Epoch 69/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.7334 - loss: 0.5351 - val_accuracy: 0.7324 - val_loss: 0.5515
Epoch 70/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.7343 - loss: 0.5345 - val_accuracy: 0.7330 - val_loss: 0.5510
Epoch 71/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.7349 - loss: 0.5336 - val_accuracy: 0.7322 - val_loss: 0.5504
Epoch 72/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.7360 - loss: 0.5329 - val_accuracy: 0.7300 - val_loss: 0.5501
Epoch 73/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 17ms/step - accuracy: 0.7365 - loss: 0.5324 - val_accuracy: 0.7330 - val_loss: 0.5501
Epoch 74/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 16ms/step - accuracy: 0.7364 - loss: 0.5318 - val_accuracy: 0.7326 - val_loss: 0.5500
Epoch 75/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 19ms/step - accuracy: 0.7363 - loss: 0.5314 - val_accuracy: 0.7298 - val_loss: 0.5494
Epoch 76/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 11s 20ms/step - accuracy: 0.7373 - loss: 0.5307 - val_accuracy: 0.7310 - val_loss: 0.5496
Epoch 77/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 9s 17ms/step - accuracy: 0.7378 - loss: 0.5303 - val_accuracy: 0.7286 - val_loss: 0.5494
Epoch 78/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 17ms/step - accuracy: 0.7385 - loss: 0.5298 - val_accuracy: 0.7316 - val_loss: 0.5491
Epoch 79/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.7382 - loss: 0.5294 - val_accuracy: 0.7308 - val_loss: 0.5493
Epoch 80/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 17ms/step - accuracy: 0.7386 - loss: 0.5291 - val_accuracy: 0.7318 - val_loss: 0.5492
Epoch 81/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 17ms/step - accuracy: 0.7386 - loss: 0.5285 - val_accuracy: 0.7320 - val_loss: 0.5491
Epoch 82/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.7387 - loss: 0.5281 - val_accuracy: 0.7324 - val_loss: 0.5491
Epoch 83/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.7383 - loss: 0.5275 - val_accuracy: 0.7324 - val_loss: 0.5495
Epoch 84/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 11s 19ms/step - accuracy: 0.7389 - loss: 0.5270 - val_accuracy: 0.7330 - val_loss: 0.5497
```

- 이 훈련은 서른다섯 번째 에포크에서 조기 종료 되었습니다. 검증 세트에 대한 정확도는 약 80% 정도 입니다. 매우 뛰어난 성능은 아니지만 감상평을 분류하는 데 어느 정도 성과를 내고 있다고 판단할 수 있습니다.
- 그럼 이전 장에서처럼 훈련 손실과 검증 손실을 그래프로 그려서 훈련 과정을 살펴보겠습니다.

```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```

![스크린샷 2025-03-18 오후 11 00 09](https://github.com/user-attachments/assets/ae6ea004-6145-4b3f-8067-3fb7f562e5ba)

- 훈련 손실은 꾸준히 감소하고 있지만 검증 손실은 대략 스무 번째 에포크에서 감소가 둔해지고 있습니다. 적절한 에포크에서 훈련을 멈춘 것 같네요. 네, 성공입니다. 1절에서 배운 순환 신경망을 성공적으로 훈련시켜서 IMDB 리뷰 데이터를 긍정과 부정으로 분류하는 작업을 수행했습니다.
- 여기서 한 가지 생각할 점이 있습니다. 이 작업을 하기 위해서 입력 데이터를 원-핫 인코딩으로 변환했습니다. 원-핫 인코딩의 단점은 입력 데이터가 엄청 커진다는 것입니다. 실제로 `train_seq`배열과 `train_oh`배열의 `nbytes` 속성을 출력하여 크기를 확인해 보세요.

```python
print(train_seq.nbytes, train_oh.nbytes)
```

- 토큰 1개를 300차원으로 늘렸기 때문에 대략 300배가 커집니다! 이는 썩 좋은 방법은 아닌 것 같습니다. 훈련 데이터가 커질수록 더 문제가 될 것입니다. 다음 섹션에서 순환 신경망에 사용하는 더 좋은 단어 표현 방법을 알아보도록 하겠습니다.

## 단어 임베딩을 사용하기

- 순환 신경망에서 텍스트를 처리할 때 즐겨 사용하는 방법은 **단어 임베딩**<sup>word embedding</sup>입니다. 단어 임베딩은 각 단어를 고정된 크기의 실수 벡터로 바꾸어 줍니다. 예를 들면 다음 그림과 같습니다.

![스크린샷 2025-03-19 오전 10 48 14](https://github.com/user-attachments/assets/dc123d6a-cb4c-483f-b9a7-b948d63800e7)

- 이런 단어 임베딩으로 만들어진 벡터는 원-핫 인코딩된 벡터보다 훨씬 의미 있는 값으로 채워져 있기 때문에 자연어 처리에서 더 좋은 성능을 내는 경우가 많습니다. 물론 이런 단어 임베딩 벡터를 만드는 층은 이미 준비되어 있습니다. 케라스에서는 `keras.layers` 패키지 아래 `Embedding` 클래스로 임베딩 기능을 제공합니다. 이 클래스를 다른 층처럼 모델에 추가하면 처음에는 모든 벡터가 랜덤하게 초기화되지만 훈련을 통해 데이터에서 좋은 단어 임베딩을 학습합니다.
- 단어 임베딩의 장점은 입력으로 정수 데이터를 받는다는 것입니다. 즉 원-핫 인코딩으로 변경된 `train_oh` 배열이 아니라 `train_seq`를 사용할 수 있습니다. 이 때문에 메모리를 훨씬 효율적으로 사용할 수 있습니다.
- 앞서 원-핫 인코딩은 샘플 하나를 300차원으로 늘렸기 떄문에 (100,) 크기의 샘플이 (100, 300)으로 커졌습니다. 이와 비슷하게 임베딩도 (100,) 크기의 샘플을 (100, 20)과 같이 2차원 배열로 늘립니다. 하지만 원-핫 인코딩과는 달리 훨씬 작은 크기로도 단어를 잘 표현할 수 있습니다.

- `Embedding` 클래스를 `SimpleRNN` 층 앞에 추가한 두 번째 순환 신경망을 만들어 보겠습니다.

```python
model2 = keras.Sequential()

model2.add(keras.layers.Input(shape=(100,)))
model2.add(keras.layers.Embedding(300, 16))
model2.add(keras.layers.SimpleRNN(8))
model2.add(keras.layers.Dense(1, activation='sigmoid'))
```

- `Embedding` 클래스의 첫 번째 매개변수 (300)는 어휘 사전의 크기입니다. 앞서 IMDB 리뷰 데이터셋에서 300개의 단어만 사용하도록 `imdb.load_data(num_words=300)`과 같이 설정했기 때문에 이 매개변수의 값을 300으로 지정합니다.
- 두 번째 매개변수(16)는 임베딩 벡터의 크기입니다. 여기에서는 원-핫 인코딩보다 훨씬 작은 크기(16)의 벡터를 사용했습니다.
- 세 번째 `Input`의 `shape` 매개변수는 입력 시퀀스의 길이입니다. 앞서 샘플의 길이를 100으로 맞추거 `train_seq`를 만들었습니다. 따라서 이 값을 100으로 지정합니다.
- 그다음 `SimpleRNN` 층과 `Dense` 층은 이전과 동일합니다. 이 모델의 구조를 출력해 보죠.

```python
model2.summary()
```

![스크린샷 2025-03-19 오전 11 23 08](https://github.com/user-attachments/assets/5253fdae-0e7e-4d9f-99bf-ae9ef031607b)

- `summary()` 메서드의 출력에서 알 수 있듯이 임베딩 층은 (100,) 크기의 입력을 받아 (100,16) 크기의 출력을 만듭니다. 이 모델에서 사용되는 모델 파라미터 개수를 계산해 보죠.
- `Embedding` 클래스는 300개의 각 토큰을 크기가 16인 벡터로 변경하기 때문에 총 300 X 16 = 4,800개의 모델 파라미터를 가집니다. 그다음 `SimpleRNN` 층은 임베딩 벡터 크기가 16이므로 8개의 뉴런과 곱하기 위해 필요한 가중치 16 X 8 = 128개를 가집니다. 또한 은닉 상태에 곱해지는 가중치 8 X 8 = 64개가 있습니다. 마지막으로 8개의 절편이 있으므로 이 순환층에 있는 전체 모델 파라미터의 개수는 128 + 64 + 8 = 200개 입니다.
- 마지막 `Dense`층의 가중치 개수는 이전과 동일하게 9개입니다. 원-핫 인코딩보다 `SimpleRNN`에 주입되는 입력의 크기가 크게 줄었지만 임베딩 벡터는 단어를 잘 표현하는 능력이 있기 떄문에 훈련 결과는 이전에 못지않을 것입니다. 모델 훈련 과정은 이전과 동일합니다.

```python
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model2.compile(optimizer=rmsprop, loss='binary_crossentropy',
               metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-embedding-model.keras',
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

history = model2.fit(train_seq, train_target, epochs=100, batch_size=64,
                     validation_data=(val_seq, val_target),
                     callbacks=[checkpoint_cb, early_stopping_cb])
```

```
Epoch 1/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 22ms/step - accuracy: 0.5096 - loss: 0.6937 - val_accuracy: 0.6108 - val_loss: 0.6764
Epoch 2/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 19ms/step - accuracy: 0.6264 - loss: 0.6719 - val_accuracy: 0.6544 - val_loss: 0.6605
Epoch 3/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 17ms/step - accuracy: 0.6600 - loss: 0.6577 - val_accuracy: 0.6786 - val_loss: 0.6466
Epoch 4/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 12s 22ms/step - accuracy: 0.6802 - loss: 0.6438 - val_accuracy: 0.6916 - val_loss: 0.6347
Epoch 5/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 17ms/step - accuracy: 0.6897 - loss: 0.6316 - val_accuracy: 0.6596 - val_loss: 0.6366
Epoch 6/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 11s 18ms/step - accuracy: 0.6973 - loss: 0.6199 - val_accuracy: 0.7086 - val_loss: 0.6108
Epoch 7/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 19ms/step - accuracy: 0.7087 - loss: 0.6071 - val_accuracy: 0.7092 - val_loss: 0.6007
Epoch 8/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 18ms/step - accuracy: 0.7112 - loss: 0.5974 - val_accuracy: 0.7030 - val_loss: 0.5963
Epoch 9/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.7133 - loss: 0.5890 - val_accuracy: 0.7072 - val_loss: 0.5892
Epoch 10/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 11s 19ms/step - accuracy: 0.7199 - loss: 0.5801 - val_accuracy: 0.7044 - val_loss: 0.5841
Epoch 11/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 18ms/step - accuracy: 0.7235 - loss: 0.5722 - val_accuracy: 0.7128 - val_loss: 0.5747
Epoch 12/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.7259 - loss: 0.5665 - val_accuracy: 0.7192 - val_loss: 0.5670
Epoch 13/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 17ms/step - accuracy: 0.7283 - loss: 0.5611 - val_accuracy: 0.7186 - val_loss: 0.5631
Epoch 14/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 17ms/step - accuracy: 0.7310 - loss: 0.5568 - val_accuracy: 0.7182 - val_loss: 0.5598
Epoch 15/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.7324 - loss: 0.5531 - val_accuracy: 0.7210 - val_loss: 0.5565
Epoch 16/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 17ms/step - accuracy: 0.7318 - loss: 0.5498 - val_accuracy: 0.7224 - val_loss: 0.5535
Epoch 17/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.7342 - loss: 0.5469 - val_accuracy: 0.7242 - val_loss: 0.5511
Epoch 18/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.7338 - loss: 0.5445 - val_accuracy: 0.7260 - val_loss: 0.5494
Epoch 19/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 11s 19ms/step - accuracy: 0.7347 - loss: 0.5426 - val_accuracy: 0.7268 - val_loss: 0.5480
Epoch 20/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 11s 20ms/step - accuracy: 0.7345 - loss: 0.5410 - val_accuracy: 0.7284 - val_loss: 0.5466
Epoch 21/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 9s 18ms/step - accuracy: 0.7343 - loss: 0.5395 - val_accuracy: 0.7282 - val_loss: 0.5455
Epoch 22/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 19ms/step - accuracy: 0.7355 - loss: 0.5381 - val_accuracy: 0.7280 - val_loss: 0.5448
Epoch 23/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 19ms/step - accuracy: 0.7356 - loss: 0.5368 - val_accuracy: 0.7290 - val_loss: 0.5442
Epoch 24/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 17ms/step - accuracy: 0.7364 - loss: 0.5356 - val_accuracy: 0.7294 - val_loss: 0.5437
Epoch 25/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 19ms/step - accuracy: 0.7362 - loss: 0.5345 - val_accuracy: 0.7294 - val_loss: 0.5432
Epoch 26/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 17ms/step - accuracy: 0.7363 - loss: 0.5334 - val_accuracy: 0.7296 - val_loss: 0.5427
Epoch 27/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 19ms/step - accuracy: 0.7350 - loss: 0.5323 - val_accuracy: 0.7308 - val_loss: 0.5423
Epoch 28/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 18ms/step - accuracy: 0.7352 - loss: 0.5313 - val_accuracy: 0.7296 - val_loss: 0.5418
Epoch 29/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.7367 - loss: 0.5303 - val_accuracy: 0.7294 - val_loss: 0.5415
Epoch 30/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 17ms/step - accuracy: 0.7369 - loss: 0.5294 - val_accuracy: 0.7290 - val_loss: 0.5415
Epoch 31/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.7387 - loss: 0.5287 - val_accuracy: 0.7306 - val_loss: 0.5413
Epoch 32/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 19ms/step - accuracy: 0.7385 - loss: 0.5279 - val_accuracy: 0.7310 - val_loss: 0.5415
Epoch 33/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 17ms/step - accuracy: 0.7394 - loss: 0.5273 - val_accuracy: 0.7316 - val_loss: 0.5416
Epoch 34/100
313/313 ━━━━━━━━━━━━━━━━━━━━ 6s 18ms/step - accuracy: 0.7395 - loss: 0.5268 - val_accuracy: 0.7322 - val_loss: 0.5417
```

- 출력 결과를 보면 원-핫 인코딩을 사용한 모델과 비슷한 성능을 냈습니다. 반면에 순환층의 가중치 개수는 훨씬 작고 훈련 세트 크기도 훨씬 줄어들었습니다. 마지막으로 훈련 손실과 검증 손실을 그래프로 출력해 보겠습니다.

```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```

![스크린샷 2025-03-19 오전 11 28 36](https://github.com/user-attachments/assets/12ed1c63-13cf-41b4-85d8-a566e6076371)

- 검증 손실이 더 감소되지 않아 훈련이 적절히 조기 종료된 것 같습니다. 이에 비해 훈련 손실은 계속 감소합니다. 이를 더 개선할 방법이 있는지 다음 절에서 알아보겠습니다.
