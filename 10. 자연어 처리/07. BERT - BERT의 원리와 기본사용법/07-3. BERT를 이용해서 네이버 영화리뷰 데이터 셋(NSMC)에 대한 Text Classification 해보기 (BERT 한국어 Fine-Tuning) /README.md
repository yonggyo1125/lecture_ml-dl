# BERT를 이용한 한국어 네이버 영화리뷰 Sentiment Classifcation

- NSMC Dataset : https://github.com/e9t/nsmc
- Reference : https://www.tensorflow.org/text/tutorials/classify_text_with_bert

## 라이브러리 설치

```python
# A dependency of the preprocessing for BERT inputs
!pip install -q -U tensorflow-text
```

```python 
!pip install -q tf-models-official
```

## 라이브러리 import

```python
import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

tf.get_logger().setLevel('ERROR')
```

## 데이터셋 다운로드 및 불러오기

## Kopora(Korean Corpora Archives) : https://github.com/ko-nlp/Korpora

```python
!pip install Korpora
```

```python 
rom Korpora import Korpora
Korpora.fetch("nsmc")
```

```
[Korpora] Corpus `nsmc` is already installed at /data/jinhoyang/Korpora/nsmc/ratings_train.txt
[Korpora] Corpus `nsmc` is already installed at /data/jinhoyang/Korpora/nsmc/ratings_test.txt
```

```python
corpus = Korpora.load("nsmc")
```

```
Korpora 는 다른 분들이 연구 목적으로 공유해주신 말뭉치들을
    손쉽게 다운로드, 사용할 수 있는 기능만을 제공합니다.

    말뭉치들을 공유해 주신 분들에게 감사드리며, 각 말뭉치 별 설명과 라이센스를 공유 드립니다.
    해당 말뭉치에 대해 자세히 알고 싶으신 분은 아래의 description 을 참고,
    해당 말뭉치를 연구/상용의 목적으로 이용하실 때에는 아래의 라이센스를 참고해 주시기 바랍니다.

    # Description
    Author : e9t@github
    Repository : https://github.com/e9t/nsmc
    References : www.lucypark.kr/docs/2015-pyconkr/#39

    Naver sentiment movie corpus v1.0
    This is a movie review dataset in the Korean language.
    Reviews were scraped from Naver Movies.

    The dataset construction is based on the method noted in
    [Large movie review dataset][^1] from Maas et al., 2011.

    [^1]: http://ai.stanford.edu/~amaas/data/sentiment/

    # License
    CC0 1.0 Universal (CC0 1.0) Public Domain Dedication
    Details in https://creativecommons.org/publicdomain/zero/1.0/

[Korpora] Corpus `nsmc` is already installed at /data/jinhoyang/Korpora/nsmc/ratings_train.txt
[Korpora] Corpus `nsmc` is already installed at /data/jinhoyang/Korpora/nsmc/ratings_test.txt
```

```python
df_train_text = pd.DataFrame(corpus.train.texts, columns=['text'])
df_train_labels = pd.DataFrame(corpus.train.labels, columns=['labels'])
```

```python
df_train = pd.concat([df_train_text, df_train_labels], axis=1)
df_train
```

![스크린샷 2024-12-15 오전 9 18 44](https://github.com/user-attachments/assets/faaf9c98-bef9-4960-b86f-1e2d602f4158)



![스크린샷 2024-12-15 오전 9 19 26](https://github.com/user-attachments/assets/ddb3f12b-f5dd-4635-99b7-8671e6d50739)





