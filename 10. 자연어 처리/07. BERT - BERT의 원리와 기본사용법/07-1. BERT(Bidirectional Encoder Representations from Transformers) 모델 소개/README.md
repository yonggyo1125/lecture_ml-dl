# BERT(Bidirectional Encoder Representations from Transformers)

## 전이 학습(Transfer Learning)

- <b>전이 학습(Transfer Learning)</b> 또는 Fine-Tuning이라고 부르는 기법은 이미 학습된 Neural Networks의 파라미터를 <b>새로운 Task에 맞게 다시 미세조정(Fine-Tuning)</b>하는 것을 의미합니다.
- **컴퓨터 비전** 문제 영역에서는 **ImageNet 등의 데이터셋에 미리 Pre-Training** 시키고 이 파라미터들을 **내가 풀고자하는 문제에 맞게 Fine-Tuning**하는 과정이 광범위하게 사용되고 있었습니다.
- 최근에는 **BERT, GPT 같은 대규모 자연어 처리 모델**이 등장하면서 **자연어 처리 문제 영역에서도 전이 학습의 개념이 광범위하게 사용**되고 있습니다.

![스크린샷 2024-12-13 오후 10 16 21](https://github.com/user-attachments/assets/b9193574-28b2-401e-8e4a-78413f4d732d)

## BERT의 핵심 아이디어

- **BERT의 핵심 아이디어** : 대량의 단어 Corpus로 <b>양방향으로(Bidirectional)</b> 학습시킨 Pre-Trained 자연어 처리 모델을 제공하고, 마지막 레이어에 간단한 ANN 등의 추가만을 통한 Fine-Tuning을 이용해서 **다양한 자연어처리 Task에 대해서 state-of-the-art 성능**을 보여줄 수 있음

## 𝑩𝑬𝑹𝑻<sub>𝑩𝑨𝑺𝑬</sub>와 𝑩𝑬𝑹𝑻<sub>𝑳𝑨𝑹𝑮𝑬</sub>

- **L** : number of layers(i.e., Transformer blocks)
- **H** : the hidden size
- **A** : the number of self-attention heads

- 2가지 사이즈의 BERT 모델을 공개
  - 𝑩𝑬𝑹𝑻<sub>𝑩𝑨𝑺𝑬</sub> (L=12, H=768, A=12, Total Parameters=**110M**)
  - 𝑩𝑬𝑹𝑻<sub>𝑳𝑨𝑹𝑮𝑬</sub> (L=24, H=1024, A=16, Total Parameters=**340M**)

## BERT Overview

- 대량의 corpus 데이터셋으로 **Pre-training** -> 목적에 맞게 **Fine-Tuning**

