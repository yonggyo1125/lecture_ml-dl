# N-Gram 예제

## NLTK 라이브러리 설치 및 import

```python
!pip install -U nltk
```

```python
from nltk.util import pad_sequence
from nltk.util import bigrams
from nltk.util import ngrams
from nltk.util import everygrams
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import flatten
```

```python
import nltk
nltk.download("all", quiet=True)
```

```python
True
```

```python
text = [['a', 'b', 'c'], ['a', 'c', 'd', 'c', 'e', 'f']]
```

```python
list(bigrams(text[0]))
```

```python
[('a', 'b'), ('b', 'c')]
```

```python
list(ngrams(text[1], n=3))
```

```python
[('a', 'c', 'd'), ('c', 'd', 'c'), ('d', 'c', 'e'), ('c', 'e', 'f')]
```

## 문장의 시작(\<s\>)과 끝(\</s\>)을 나타내는 Padding 추가

```python
from nltk.util import pad_sequence
list(pad_sequence(text[0],
                  pad_left=True, left_pad_symbol="<s>",
                  pad_right=True, right_pad_symbol="</s>",
                  n=2)) # The n order of n-grams, if it's 2-grams, you pad once, 3-grams pad twice, etc.
```

```python
['<s>', 'a', 'b', 'c', '</s>']
```

```python
padded_sent = list(pad_sequence(text[0],
                                pad_left=True, left_pad_symbol="<s>",
                                pad_right=True, right_pad_symbol="</s>",
                                n=2))
list(ngrams(padded_sent, n=2))
```

```python
[('<s>', 'a'), ('a', 'b'), ('b', 'c'), ('c', '</s>')]
```

## pad_both_ends 함수를 이용해서 이 과정을 좀더 쉽게 수행할 수 있습니다.

```python
from nltk.lm.preprocessing import pad_both_ends
list(pad_both_ends(text[0], n=2))
```

```python
['<s>', 'a', 'b', 'c', '</s>']
```

```python
list(bigrams(pad_both_ends(text[0], n=2)))
```

```python
[('<s>', 'a'), ('a', 'b'), ('b', 'c'), ('c', '</s>')]
```

## everygrams 함수를 이용해서 각 N-gram(e.g. 1-gram, 2-gram, 3-gram)의 시작과 끝에 padding을 적용할 수 있습니다.

```python
from nltk.util import everygrams
padded_bigrams = list(pad_both_ends(text[0], n=2))
print(padded_bigrams)
print(list(everygrams(padded_bigrams, max_len=3)))
```

```python
'<s>', 'a', 'b', 'c', '</s>']
[('<s>',), ('<s>', 'a'), ('<s>', 'a', 'b'), ('a',), ('a', 'b'), ('a', 'b', 'c'), ('b',), ('b', 'c'), ('b', 'c', '</s>'), ('c',), ('c', '</s>'), ('</s>',)]
```

## flatten 함수를 이용해서 모든 문자들을 펼칠 수 있습니다.

```python
from nltk.lm.preprocessing import flatten
list(flatten(pad_both_ends(sent, n=2) for sent in text))
```

```python
['<s>', 'a', 'b', 'c', '</s>', '<s>', 'a', 'c', 'd', 'c', 'e', 'f', '</s>']
```

```python
from nltk.lm.preprocessing import padded_everygram_pipeline
training_ngrams, padded_sentences = padded_everygram_pipeline(2, text)
for ngramlize_sent in training_ngrams:
    print(list(ngramlize_sent))
    print()
print('#############')
list(padded_sentences)
```

```python
[('<s>',), ('<s>', 'a'), ('a',), ('a', 'b'), ('b',), ('b', 'c'), ('c',), ('c', '</s>'), ('</s>',)]

[('<s>',), ('<s>', 'a'), ('a',), ('a', 'c'), ('c',), ('c', 'd'), ('d',), ('d', 'c'), ('c',), ('c', 'e'), ('e',), ('e', 'f'), ('f',), ('f', '</s>'), ('</s>',)]

#############
['<s>', 'a', 'b', 'c', '</s>', '<s>', 'a', 'c', 'd', 'c', 'e', 'f', '</s>']
```

## 테스트를 위한 텍스트 파일(language-never-random.txt)을 다운받습니다.

```python
import os
import requests
import io

# Text version of https://kilgarriff.co.uk/Publications/2005-K-lineer.pdf
if os.path.isfile('language-never-random.txt'):
    with io.open('language-never-random.txt', encoding='utf8') as fin:
        text = fin.read()
else:
    url = "https://gist.githubusercontent.com/alvations/53b01e4076573fea47c6057120bb017a/raw/b01ff96a5f76848450e648f35da6497ca9454e4a/language-never-random.txt"
    text = requests.get(url).content.decode('utf8')
    with io.open('language-never-random.txt', 'w', encoding='utf8') as fout:
        fout.write(text)
```

```python
text
```

```python
from nltk import word_tokenize, sent_tokenize 
# Tokenize the text.
tokenized_text = [list(map(str.lower, word_tokenize(sent))) 
                  for sent in sent_tokenize(text)]
```

```python
tokenized_text[0]
```

```
['language',
 'is',
 'never',
 ',',
 'ever',
 ',',
 'ever',
 ',',
 'random',
 'adam',
 'kilgarriff',
 'abstract',
 'language',
 'users',
 'never',
 'choose',
 'words',
 'randomly',
 ',',
 'and',
 'language',
 'is',
 'essentially',
 'non-random',
 '.']
```

```python
print(text[:500])
```

```
                       Language is never, ever, ever, random

                                                               ADAM KILGARRIFF




Abstract
Language users never choose words randomly, and language is essentially
non-random. Statistical hypothesis testing uses a null hypothesis, which
posits randomness. Hence, when we look at linguistic phenomena in cor-
pora, the null hypothesis will never be true. Moreover, where there is enough
data, we shall (almost) always be able to establish 
```

## 3-gram 모델 설정

```python
# Preprocess the tokenized text for 3-grams language modelling
n = 3
train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)
```

```python
train_data
```

```
<generator object padded_everygram_pipeline.<locals>.<genexpr> at 0x7f1dca291d20>
```

## 학습을 위해 MLE(Maximum Likelihood Estimation) 추정

```python
from nltk.lm import MLE
model = MLE(n) # Lets train a 3-grams model, previously we set n=3
```

```python
len(model.vocab)
```

```
0
```

```python
model.fit(train_data, padded_sents)
print(model.vocab)
```

```
<Vocabulary with cutoff=1 unk_label='<UNK>' and 1391 items>
```

```python
len(model.vocab)
```

```
1391
```

```python
print(model.vocab.lookup(tokenized_text[0]))
```

```
1391
```

```python
print(model.vocab.lookup(tokenized_text[0]))
```

```python
('language', 'is', 'never', ',', 'ever', ',', 'ever', ',', 'random', 'adam', 'kilgarriff', 'abstract', 'language', 'users', 'never', 'choose', 'words', 'randomly', ',', 'and', 'language', 'is', 'essentially', 'non-random', '.')
```

## 만약 Vocab 집합에 포함되지 않는 단어라면 라는 특수 토큰으로 처리됩니다.

```python
# If we lookup the vocab on unseen sentences not from the training data, 
# it automatically replace words not in the vocabulary with `<UNK>`.
print(model.vocab.lookup('language is never random lah .'.split()))
```

```python
('language', 'is', 'never', 'random', '<UNK>', '.')
```

```python
print(model.counts)
```

```
<NgramCounter with 3 ngram orders and 19611 ngrams>
```

```python
model.counts['language'] # i.e. Count('language')
```

```python
model.counts[['language']]['is'] # i.e. Count('is'|'language')
```

```python
model.counts[['language', 'is']]['never'] # i.e. Count('never'|'language is')
```

```python
model.score('language') # P('language')
```

```
0.003691671588895452
```

```python
model.score('is', 'language'.split())  # P('is'|'language')
```

```
0.44
```

```python
model.score('never', 'language is'.split())  # P('never'|'language is')
```

```
0.6363636363636364
```

## 등장하지 않는 단어들은 0의 확률값이 계산됩니다.

```python
print(model.score("<UNK>") == model.score("lah"))
print(model.score("<UNK>"))
print(model.score("lah"))
```

```python
True
0.0
0.0
```

```python
model.score("<UNK>") == model.score("leh")
```

```python
True
```

```python
model.score("<UNK>") == model.score("lor")
```

```python
True
```

## 편의를 위해 log를 씌운 확률로도 계산할 수 있습니다.

```python
model.logscore("never", "language is".split())
```

```
-0.6520766965796932
```

## N-gram 모델을 이용해서 랜덤한 새로운 텍스트를 생성합니다.

```python
print(model.generate(20, random_seed=7))
```

```
['and', 'carroll', 'used', 'hypothesis', 'testing', 'has', 'been', 'used', ',', 'and', 'a', 'half', '.', '</s>', '</s>', '</s>', '</s>', '</s>', '</s>', '</s>']
```

```python
from nltk.tokenize.treebank import TreebankWordDetokenizer

detokenize = TreebankWordDetokenizer().detokenize

def generate_sent(model, num_words, random_seed=42):
    """
    :param model: An ngram language model from `nltk.lm.model`.
    :param num_words: Max no. of words to generate.
    :param random_seed: Seed value for random.
    """
    content = []
    for token in model.generate(num_words, random_seed=random_seed):
        if token == '<s>':
            continue
        if token == '</s>':
            break
        content.append(token)
    return detokenize(content)
```

```python
generate_sent(model, 20, random_seed=7)
```

```
and carroll used hypothesis testing has been used, and a half.
```

```python
print(model.generate(28, random_seed=0))
```

```python
['the', 'scf-verb', 'link', 'is', 'motivated', '.', '</s>', '</s>', '</s>', '</s>', '</s>', '</s>', '</s>', '</s>', '</s>', '</s>', '</s>', '</s>', '</s>', '</s>', '</s>', '</s>', '</s>', '</s>', '</s>', '</s>', '</s>', '</s>']
```

```python
generate_sent(model, 28, random_seed=0)
```

```
the scf-verb link is motivated.
```

```python
generate_sent(model, 20, random_seed=1)
```

```
237⫺246.
```

```python
generate_sent(model, 20, random_seed=30)
```

```
hypothesis is ever a useful construct.
```

```python
generate_sent(model, 20, random_seed=42)
```

```
more (or cold) weather, or on saturday nights, or by people in (or poorer)
```