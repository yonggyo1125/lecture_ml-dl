# 토크나이징과 원핫 인코딩 실습

- 토크나이징과 원핫 인코딩을 직접 구현해서 실습해봅시다.

## 자연어 처리 기초 - 토크나이징과 원핫인코딩

```python
!pip install nltk
```

```
Requirement already satisfied: nltk in /usr/local/lib/python3.7/dist-packages (3.2.5)
Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from nltk) (1.15.0)
```

```python
import nltk
```

```python
nltk.download("all", quiet=True)
```

```
True
```

## 단어 단위 Tokenizing

```python
from nltk.tokenize import word_tokenize
text = "Friends, Romans, Countrymen, lend me your ears;."
print(word_tokenize(text))
```

```python
['Friends', ',', 'Romans', ',', 'Countrymen', ',', 'lend', 'me', 'your', 'ears', ';', '.']
```

## 문장 단위 Tokenizing

```python
from nltk.tokenize import sent_tokenize
text = "Natural language processing (NLP) is a subfield of \
linguistics, computer science, and artificial intelligence \
concerned with the interactions between computers and human language, \
in particular how to program computers to process and analyze large \
amounts of natural language data. The goal is a computer capable of \
understanding the contents of documents, including the contextual \
nuances of the language within them. The technology can then accurately \
extract information and insights contained in the documents as well as categorize and organize the documents themselves."

print(sent_tokenize(text))
```

```python
['Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.', 'The goal is a computer capable of understanding the contents of documents, including the contextual nuances of the language within them.', 'The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.']
```

## One-hot Encoding 적용하기

### 관련 라이브러리 import

```python
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
```

```python
# 원핫 인코딩을 적용할 예제 데이터를 설정합니다
data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
print(set(data))
values = array(data)
print(values)
```

```python
{'warm', 'hot', 'cold'}
['cold' 'cold' 'warm' 'cold' 'hot' 'hot' 'warm' 'cold' 'warm' 'hot']
```

```python
# Integer Encoding을 진행합니다.
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
```

```python
[0 0 2 0 1 1 2 0 2 1]
```

```python
integer_encoded.shape
```

```python
(10,)
```

```python
# One-hot Encoding을 진행합니다.
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
print(integer_encoded.reshape(len(integer_encoded), 1))
print(integer_encoded.shape)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
```

```python
[[0]
 [0]
 [2]
 [0]
 [1]
 [1]
 [2]
 [0]
 [2]
 [1]]
(10, 1)
[[1. 0. 0.]
 [1. 0. 0.]
 [0. 0. 1.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [1. 0. 0.]
 [0. 0. 1.]
 [0. 1. 0.]]
```

```python
# One-hot Encoding을 적용한 예제를 다시 string 형태로 변경합니다.
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
print(inverted)
```

```python
['cold']
```

```python
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[2, :])])
print(inverted)
```

```python
['warm']
```
