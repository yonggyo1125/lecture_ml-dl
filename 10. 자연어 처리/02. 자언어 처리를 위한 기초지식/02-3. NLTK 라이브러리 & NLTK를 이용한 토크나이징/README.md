## NLTK 라이브러리

- NLTK 라이브러리는 자연어 처리를 위한 토크나이징 등 편리한 기능을 제공하는 자연어처리 파이썬 라이브러입니다.
- https://www.nltk.org/
- 아래 명렁어를 통해 NLTK 라이브러리를 설치할 수 있습니다.

```
pip install nltk
```

- NLTK 라이브러리를 설치한 이후 적절한 corpus를 다운로드해주어야만 합니다. “all”을 설정해서 전체 corpus를 다운로드 받아줍니다.

```python
nltk.download("all", quiet=True)
```

```
True
```

## NLTK 라이브러리를 이용한 단어 단위 토크나이징

- NLTK 라이브러리를 이용해서 간단하게 **단어 단위로 토크나이징**을 진행할수 있습니다.

```python
from nltk.tokenize import word_tokenize
text = "Friends, Romans, Countrymen, lend me your ears;."
print(word_tokenize(text))
```

```python
['Friends',',', 'Romans', ',', 'Countrymen', ',', 'lend', 'me', 'your', 'ears', ';', '.']
```

- NLTK 라이브러리를 이용해서 간단하게 문장 단위로 토크나이징을 진행할수 있습니다.

![스크린샷 2024-12-08 오후 3 26 11](https://github.com/user-attachments/assets/f3439c9d-0849-45d2-86e2-70e7c3c5bb25)
