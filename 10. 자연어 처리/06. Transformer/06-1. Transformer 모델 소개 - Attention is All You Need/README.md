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

## Transformer Architecture

- Transformer의 전체 Architecture를 살펴보면 다음과 같습니다.


