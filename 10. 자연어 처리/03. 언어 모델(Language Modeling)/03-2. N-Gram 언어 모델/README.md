## N-gram 언어 모델
- **N-gram 언어 모델**은 이전에 존재하는 k개의 단어에 기반해서 다음에 올 단
어를 예측하는 언어 모델 방법론입니다.
- 이때 **N=k+1** 입니다.
- 이때 **N=2일 경우 이전에 존재하는 1개의 단어에 기반**해서 다음에 올 단어를 예측하고, N=3일 경우에는 이전에 존재하는 2개의 단어, N=4일 경우에는
이전에 존재하는 3개의 단어에 기반해서 다음에 올 경우를 예측합니다.
- N의 개수는 상황마다 다르기 때문에 내 문제에 맞게 적절한 값을 설정해주어야만 합니다.

## N-gram 예시 

![스크린샷 2024-12-08 오후 4 59 38](https://github.com/user-attachments/assets/3316eea7-6c24-43e0-8ad9-3f4211e7e740)


## N-gram 모델 ‒ Count에 기반한 방법

- N-gram 모델을 이용할 경우, 다음 단어가 올 확률은 해당 단어 조합이 등장한 횟수(Count-C-)에 기반해서 계산됩니다.
- 만약 “its water is so transparent that” 이라는 단어조합이 나온 횟수가 10,000번이고 ”its water is so transparent that the”라는 단어조합이 나온 횟수가 500번이라면 “its water is so transparent that” 이라는 단어조합뒤에 “its water is so transparent that the”라는 단어조합이 나올 확률은 <b>0.05(=5%)</b>가 되게 됩니다.

