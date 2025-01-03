## seq2seq(Sequence-to-sequence) 모델

- <b>기계번역(NMT)</b>을 구현하기 위한 seq2seq(Sequence-to-sequence) 모델의 개념을 먼저 살펴봅시다.
- seq2seq 모델은 번역할 언어로 작성된 <b>소스 문장들(source sentences)</b>을 <b>인코더(Encoder)</b>를 이용해서 “생각” 벡터(“thought” vector) 형태로 변환하고, 이 “생각” 벡터를 특징값으로 이용해서 <b>디코더(Decoder)가 번역할 언어로 작성된 타겟 문장(target sentences)</b>을 생성하는 기법입니다.
- 아래 그림은 seq2seq를 이용해서 영어 문장을 프랑스어 문장으로 번역하는 법을 보여줍니다.

![스크린샷 2024-12-10 오후 8 56 17](https://github.com/user-attachments/assets/a0ce8035-64ba-4a7d-b409-38c18d058edf)

- seq2seq 모델은 다양한 구조를 이용해서 구현할 수 있지만 **RNN을 이용해서 구현하는 것**이 가장 일반적입니다. RNN을 이용해서 seq2seq 모델을 구현할 경우 다음의 과정을 거칩니다.
  - 번역할 언어로 작성된 소스 문장을 단어 단위로 쪼개고, 소스 문장의 단어들을 RNN의 인풋으로 넣는다.
  - **소스 문장의 가장 마지막 단어를 인풋으로 넣고 구한 RNN의 상태값**을 **타겟 문장의 첫번째 단어의 예측을 시작할때 초기 상태값으로 지정**한다.
  - \<s\>는 문장의 시작을 의미하는 특수 키워드로써, 번역할 언어로 작성된 타겟 문장의 첫번째 단어로 사용한다.
  - \<s\>를 넣고 RNN이 다음에 올 수 있는 타겟 단어 후보군(Vocabulary)를 Softmax 행렬 형태(|V|)로 예측하면, 다음에 올 가장 그럴듯한 단어(Softmax 행렬 출력값 중 가장 큰값을 가지는 단어)를 선택해서 \<s\> <b>다음 단어(예를 들어, Je)로 확정</b>한다.
  - <b>\<s\>를 넣고 예측해서 구한 다음 단어(Je)를 RNN의 인풋으로 넣고</b>, 다시 다음에 올 수 있는 타겟 단어 후보군(Vocabulary)를 Softmax 행렬 형태(|V|)로 예측하면, 다음에 올 가장 그럴듯한 단어를 선택해서 <b>다음 단어(예를 들어, suis)로 확정</b>한다.
  - 위 과정을 <b>문장의 끝을 의미하는 특수 키워드 \</s\>가 다음 단어로 확정될 때까지 반복</b>한다.
  - <b>타겟 문장의 초기 상태값으로 소스 문장의 마지막 상태값을 사용하는 것</b>을 제외하면 앞서 배운 <b>Char-RNN과 비슷한 형태</b>임을 알 수 있습니다.
  - 아래 그림은 RNN을 이용한 seq2seq 모델 구조의 예시를 나타냅니다.

![스크린샷 2024-12-10 오후 9 24 49](https://github.com/user-attachments/assets/39839742-24a2-4bb1-81f6-7f7362e0a563)

## Attention 추가

- 기본 Seq2Seq 모델에 Attention을 추가해서 추가적인 성능 개선을 이루어 낼 수 있습니다.
- h<sub>s</sub> encoder의 시퀀스 출력값을 의미합니다.
- h<sub>t</sub> 는 decoder의 상태값을 의미합니다.
- a<sub>t</sub> 는 decoder의 최종 출력값을 의미합니다.

![스크린샷 2024-12-10 오후 9 34 34](https://github.com/user-attachments/assets/8eb82315-576d-4618-9219-7513994bfd13)

## 포르투칼어-영어 번역의 Attention 일부

- 포르투칼어-영어의 번역의 Attention 일부를 살펴보면 다음과 같습니다.

![스크린샷 2024-12-10 오후 9 37 34](https://github.com/user-attachments/assets/76ce91f8-2a6e-4617-ba36-bf6d0d03c943)

- **Input** : <start> hace mucho frio aqui . \<end\>
- **Predicted translation** : it s too cold here . \<end\>

- 포르투칼어-영어의 번역의 Attention 일부를 살펴보면 다음과 같습니다.

![스크린샷 2024-12-10 오후 9 39 45](https://github.com/user-attachments/assets/d712fdbb-96b3-4831-8166-4fa5fbd03f15)

- **Input** : <start> esta es mi vida . \<end\>
- **Predicted translation** : this is my life . \<end\>
