# Transformer를 이용한 포르투칼어-영어번역

- Transformer를 이용해서 포르투칼어를 영어로 번역하는 딥러닝 모델을 만들어봅시다.

## 라이브러리 설치

```python
!pip install -q tensorflow_datasets
!pip install -U tensorflow-text==2.7.3
```

## 라이브러리 import

```python
import collections
import logging
import os
import pathlib
import re
import string
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf
```

## 로거 설정

```python
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings
```

## 포르투칼어-영어 번역 데이터셋을 다운로드합니다.

### Refernece

- https://github.com/neulab/word-embeddings-for-nmt
- https://www.ted.com/participate/translate

### 50000개의 training 데이터와 1100개의 validation 데이터, 2000개의 테스트 데이터를 가지고 있습니다.

```python
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']
```

## Sanity check를 위해 trianing data에 있는 3개의 문장을 살펴봅니다.

```python
for pt_examples, en_examples in train_examples.batch(3).take(1):
  for pt in pt_examples.numpy():
    print(pt.decode('utf-8'))

  print()

  for en in en_examples.numpy():
    print(en.decode('utf-8'))
```

```
e quando melhoramos a procura , tiramos a única vantagem da impressão , que é a serendipidade .
mas e se estes fatores fossem ativos ?
mas eles não tinham a curiosidade de me testar .

and when you improve searchability , you actually take away the one advantage of print , which is serendipity .
but what if it were active ?
but they did n't test for curiosity .
```

## text 데이터를 numeric 데이터로 변경하기 위한 Tokenizer를 다운로드받고 불러옵니다.

```python
model_name = "ted_hrlr_translate_pt_en_converter"
tf.keras.utils.get_file(
    f"{model_name}.zip",
    f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
    cache_dir='.', cache_subdir='', extract=True
)
```

```
Downloading data from https://storage.googleapis.com/download.tensorflow.org/models/ted_hrlr_translate_pt_en_converter.zip
188416/184801 [==============================] - 0s 0us/step
./ted_hrlr_translate_pt_en_converter.zip
```

```python
tokenizers = tf.saved_model.load(model_name)
```

```python
[item for item in dir(tokenizers.en) if not item.startswith('_')]
```

```python
['detokenize',
 'get_reserved_tokens',
 'get_vocab_path',
 'get_vocab_size',
 'lookup',
 'tokenize',
 'tokenizer',
 'vocab']
```

## 예제 데이터를 통한 Tokenizer 사용법 익히기

```python
for en in en_examples.numpy():
  print(en.decode('utf-8'))
```

```
and when you improve searchability , you actually take away the one advantage of print , which is serendipity .
but what if it were active ?
but they did n't test for curiosity .
```

```python
encoded = tokenizers.en.tokenize(en_examples)

for row in encoded.to_list():
  print(row)
```

```
[2, 72, 117, 79, 1259, 1491, 2362, 13, 79, 150, 184, 311, 71, 103, 2308, 74, 2679, 13, 148, 80, 55, 4840, 1434, 2423, 540, 15, 3]
[2, 87, 90, 107, 76, 129, 1852, 30, 3]
[2, 87, 83, 149, 50, 9, 56, 664, 85, 2512, 15, 3]
```

```python
round_trip = tokenizers.en.detokenize(encoded)
for line in round_trip.numpy():
  print(line.decode('utf-8'))
```

```
and when you improve searchability , you actually take away the one advantage of print , which is serendipity .
but what if it were active ?
but they did n ' t test for curiosity .
```

```python
tokens = tokenizers.en.lookup(encoded)
tokens
```

```
<tf.RaggedTensor [[b'[START]', b'and', b'when', b'you', b'improve', b'search', b'##ability', b',', b'you', b'actually', b'take', b'away', b'the', b'one', b'advantage', b'of', b'print', b',', b'which', b'is', b's', b'##ere', b'##nd', b'##ip', b'##ity', b'.', b'[END]'], [b'[START]', b'but', b'what', b'if', b'it', b'were', b'active', b'?', b'[END]'], [b'[START]', b'but', b'they', b'did', b'n', b"'", b't', b'test', b'for', b'curiosity', b'.', b'[END]']]>
```

## Training에 사용할 pipeline 구축을 위한 tokenizing 함수 정의

```python
def tokenize_pairs(pt, en):
    pt = tokenizers.pt.tokenize(pt)
    # Convert from ragged to dense, padding with zeros.
    pt = pt.to_tensor()

    en = tokenizers.en.tokenize(en)
    # Convert from ragged to dense, padding with zeros.
    en = en.to_tensor()
    return pt, en
```

## Batch Size 및 batch 생성 함수 정의

```python
BUFFER_SIZE = 20000
BATCH_SIZE = 64
```

```python
def make_batches(ds):
  return (
      ds
      .cache()
      .shuffle(BUFFER_SIZE)
      .batch(BATCH_SIZE)
      .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
      .prefetch(tf.data.AUTOTUNE))


train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)
```

## 위치 인코딩(Positional Encoding)을 위한 함수 정의


![스크린샷 2024-12-12 오후 10 10 44](https://github.com/user-attachments/assets/93434795-5215-4e2d-9392-08272afe0721)



```python
def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates
```
