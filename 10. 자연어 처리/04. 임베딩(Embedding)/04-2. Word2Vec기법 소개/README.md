# Word2Vec

- Word2Vec은 구글 직원인 토마스 미콜로프 (Tomas Mikolov)가 2013년에 제안한 대표
  적이고 효율적인 임베딩을 위한 딥러닝 모델중에 하나입니다.
- Word2Vec은 CBOW와 Skip-Gram이라는 2가지 방식으로 구현될 수 있습니다.
  - 1 CBOW(Continuous Bag-Of-Words) model : 소스 컨텍스트에서 타겟 단어를 예측합니다.
    - 예를 들어, ‘the cat sits on the’라는 소스 컨텍스트로부터 ‘mat’이라는 타겟 단어를 예측합니다. CBOW는 작은 규모의(smaller) 데이터셋에 적합합니다.
  - 2 Skip-Gram model ‒ 타겟 단어로부터 소스 컨텍스트를 예측합니다. 예를 들어, ‘mat’이라는 타겟 단어로부터 ‘the cat sits on the’라는 소스 컨텍스트를 예측합니다. Skip-Gram model은 큰 규모의(larger) 데이터셋에 적합합니다.
    - 따라서, 큰 규모의 데이터셋에 적합한 Skip-Gram model에 초점을 맞춰서 좀더 살펴봅시다.

## Skip-Gram 모델
