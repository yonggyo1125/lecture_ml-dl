# Word2Vec

![스크린샷 2024-12-08 오후 9 12 08](https://github.com/user-attachments/assets/6e1d7b0f-b64d-4419-ab86-3b8add81f674)

- Word2Vec은 구글 직원인 토마스 미콜로프 (Tomas Mikolov)가 2013년에 제안한 대표
  적이고 효율적인 임베딩을 위한 딥러닝 모델중에 하나입니다.
- Word2Vec은 CBOW와 Skip-Gram이라는 2가지 방식으로 구현될 수 있습니다.
  - 1 CBOW(Continuous Bag-Of-Words) model : 소스 컨텍스트에서 타겟 단어를 예측합니다.
    - 예를 들어, ‘the cat sits on the’라는 소스 컨텍스트로부터 ‘mat’이라는 타겟 단어를 예측합니다. CBOW는 작은 규모의(smaller) 데이터셋에 적합합니다.
  - 2 Skip-Gram model - 타겟 단어로부터 소스 컨텍스트를 예측합니다. 예를 들어, ‘mat’이라는 타겟 단어로부터 ‘the cat sits on the’라는 소스 컨텍스트를 예측합니다. Skip-Gram model은 큰 규모의(larger) 데이터셋에 적합합니다.
    - 따라서, 큰 규모의 데이터셋에 적합한 Skip-Gram model에 초점을 맞춰서 좀더 살펴봅시다.

## Skip-Gram 모델

- 예를 들어, 아래와 같은 데이터셋이 주어졌다고 가정해봅시다.

```
the quick brown fox jumped over the lazy dog
```

- 먼저 컨텍스트를 정의해야합니다. 컨텍스트는 어떤 형태로든 정의할 수 있지만, 사람들은 보통 구문론적 컨텍스트를 정의합니다. 이번 구현에서는 간단하게, **컨텍스트를 타겟 단어의 왼쪽과 오른쪽 단어들의 윈도우**로 정의합니다. 윈도우 사이즈를 1로하면 (context, target) 쌍으로 구성된 아래와 같은 데이터셋을 얻을 수 있습니다.

```
([the, brown], quick), ([quick fox], brown), ([brown, jumped], fox), …
```

- skip-gram 모델은 **타겟 단어로부터 컨텍스트를 예측**한다는 점을 상기합시다. 따라서 우리가 해야할 일은 ‘quick’이라는 타겟단어로부터 컨텍스트 ‘the’와 ‘brown’을 예측하는 것입니다. 따라서 우리 데이터셋은 아래와 같은 (input, output) 쌍으로 표현할 수 있습니다.

```
(quick, the), (quick, brown), (brown, quick), (brown, fox), …
```

![스크린샷 2024-12-09 오후 10 18 37](https://github.com/user-attachments/assets/a1480df2-3b37-4bf3-8810-be6706c3ede1)

## Noise-Contrastive 트레이닝을 통한 규모확장(Scaling Up)

- 전통적인 방식의 계산 구조를 나타내면 아래와 같습니다. 어휘집합에 존재하는 모든 단어 𝑤" 에대한 계산을 수행해야만 합니다.


![스크린샷 2024-12-09 오후 10 23 52](https://github.com/user-attachments/assets/1bb7d1c9-9d68-4a3e-b085-249027c6428a)

- 이에 반해, word2vec을 이용한 특징 학습(feature learning) 방법에서는 전체 확률 모델이 필요하지 않습니다.
- CBOW와 skip-gram <b>모델은 이진 분류(binary classification) 방법(로지스틱 회귀분석-logistic regression)</b>을 이용합니다.
- 같은 컨텍스트에서 <b>k개의 상상의 (noise) 단어 !𝑤와 진짜 타겟 단어 𝑤+를 식별</b>합니다.
- 아래의 그림은 CBOW 모델에 해당 방법을 적용한 예시를 나타냅니다. skip-gram 모델은 단순히 화살표의 방향이 반대로 바뀐 모양입니다.

![스크린샷 2024-12-09 오후 10 28 05](https://github.com/user-attachments/assets/a9441cd8-fd64-47ca-bddf-bd4e4d1af8fe)


