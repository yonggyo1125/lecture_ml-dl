# BERT Fine-Tuning SQuAD 데이터셋

## 전이 학습(Transfer Learning)

- 전이 학습(Transfer Learning) 또는 Fine-Tuning이라고 부르는 기법은 이미 학습된 Neural Networks의 파라미터를 새로운 Task에 맞게 다시 미세조정(Fine-Tuning)하는 것을 의미합니다.
- 컴퓨터 비전 문제 영역에서는 ImageNet 등의 데이터셋에 미리 Pre-Training 시키고 이 파라미터들을 내가 풀고자하는 문제에 맞게 Fine-Tuning하는 과정이 광범위하게 사용되고 있었습니다.
- 최근에는 BERT, GPT 같은 대규모 자연어 처리 모델이 등장하면서 자연어 처리 문제 영역에서도 전이 학습의 개념이 광범위하게 사용되고 있습니다.

![스크린샷 2024-12-15 오전 9 58 38](https://github.com/user-attachments/assets/4cc03795-ed59-4c76-8529-51952990557c)

## BERT Overview

- 대량의 corpus 데이터셋으로 Pre-training -> 목적에 맞게 Fine-Tuning


