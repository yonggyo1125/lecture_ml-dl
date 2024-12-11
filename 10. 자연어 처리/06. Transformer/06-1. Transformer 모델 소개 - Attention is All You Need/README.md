## Transformer
- Transfomer는 “<b>Attention is all you need</b>”라는 제목의 논문으로 2017년도에 구글에서 발표한 모델입니다.
- 기존의 seq2seq 모델 구조에 기반하되, RNN 대신 <b>Attention 기법을 이용해서 더 높은 성능을 보여준 모델</b>입니다.
- Transformer 모델의 장점은 다음과 같습니다.
    - 특징들의 시간적, 공간적 연관관계에 대한 선행 가정을 하지 않습니다.
    - 레이어의 출력값이 RNN처럼 순차적인 형태가 아니라 **병렬적으로 계산**될 수 있습니다.
    - 많은 RNN 스텝이나 Convolution을 거치지 않고도 멀리 떨어진 정보들이 서로 영향을 주고 받을 수 있습니다.
    - **멀리 떨어진 정보들에 대한 연관관계를 학습**할 수 있습니다. 이는 많은 시계열 처리 문제에서 도전적인 문제입니다.

## seq2seq(Sequence-to-sequence) 모델
- Seq2Seq 모델구조를 다시 상기해보면 아래와 같습니다.
- **타겟 문장의 초기 상태값으로 소스 문장의 마지막 상태값을 사용하는 것**을 제외하면 앞서 배운 <b>Char-RNN과 비슷한 형태</b>임을 알 수 있습니다.
- 아래 그림은 RNN을 이용한 seq2seq 모델 구조의 예시를 나타냅니다.

![스크린샷 2024-12-11 오후 10 59 13](https://github.com/user-attachments/assets/6134e569-2529-48e3-aa9f-9d234ca33def)


## Transformer Architecture

- Transformer의 전체 Architecture를 살펴보면 다음과 같습니다.


![스크린샷 2024-12-11 오후 11 03 44](https://github.com/user-attachments/assets/35f5745a-defe-48b5-91b6-6f12c4a11c07)


## Positional Encoding

- Transformers는 RNN과 같이 시간축에 따른 반복이 없기 때문에 <b>위치 인코딩(Positional Encoding)</b>을 통해 **문장 내 해당 단어의 위치에 대한 정보**를 학습합니다.
- 일반적인 Embedding은 단어 들간의 유사 정도를 표현해주지만 **문장내 단어의 위치정보**는 알 수 없습니다. 따라서 **별도의 Positional Encoding으로 계산한 값과 Embedding 값을 더해서 Transformer 모델의 입력값**으로 사용합니다.
- Positional Encoding은 다음의 수식에 의해서 계산됩니다.
  

![스크린샷 2024-12-11 오후 11 17 16](https://github.com/user-attachments/assets/526ec083-e6b4-46e8-8fc6-3568b2f1d652)

## Scaled Dot-Product Attention

- Transformer의 Attention은 <b>Q(Query), K(Key), V(Value)</b>를 통해서 계산됩니다.
- 기본적인 Attention 종류인 <b>Scaled Dot-Product Attention</b>의 계산 수식은 다음과 같습니다.


![스크린샷 2024-12-11 오후 11 19 54](https://github.com/user-attachments/assets/ebb9dfef-4ac8-479c-9680-724e57847365)


## Multi-head Attention

- Transformer는 여러개의 head를 통한 Multi-head Attention을 계산합니다.
- <b>Multi-Head Attention</b>은 다음과 같은 연산을 수행합니다.
    - Linear 연산을 취하고 이를 각각의 Head로 나눔
    - 각각의 Head에서 Scaled dot-product attention 연산을 수행
    - 각각의 head에서 일어난 연산을 Concatenation
    - 마지막 linear layer
- **여러 개의 head가 다른 정보들을 학습**하고 이들을 활용할 수 있기 때문에 Multi-head Attention은 좋은 성능을 발휘할 수 있습니다.

![스크린샷 2024-12-11 오후 11 24 06](https://github.com/user-attachments/assets/09851610-530e-443a-85d7-621dba875efb)

## Point wise feed forward network

- Point wise feed forward network는 2개의 Fully Connected layer와 ReLU activation으로 구성된 Neural Networks입니다
- 원 논문에서는 은닉층의 개수를 2048개로 하고 출력층의 개수를 512개로 설정한 Feed Forward Network를 사용하였습니다.