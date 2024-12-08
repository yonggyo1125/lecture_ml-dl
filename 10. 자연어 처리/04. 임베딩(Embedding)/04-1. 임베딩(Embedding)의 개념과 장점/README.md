# 원핫 인코딩 표현의 문제점 
- <b>임베딩(Embedding)</b>은 머신러닝 알고리즘을 사용할 때, 그 중에서도 특히 자연어 처리 문제를 다룰 때 널리 사용되는 기법입니다.
- 머신러닝 알고리즘을 사용할 때 데이터를 표현하는 일반적인 방법은 **One-hot Encoding**입니다. 하지만 아래 그림에서 볼 수 있듯이 One-hot Encoding은 **데이터의 표현 형태가 Sparse**하다는 문제점이 있습니다.
- 예를 들어 10,000개의 단어사전에 있는 단어 하나를 One-hot Encoding으로 표현하면 10,000×1의 행렬에서 1개의 행에만 1이라는 값이 있고, 나머지 9999개의 행에는 0이라는 의미없는 값이 들어가 있을 것입니다. 결과적으로 표현에서 잉여 부분이 많아집니다. 또한 One-hot Encoding은 유사한 의미를 단어를 가진 단어간의 연관성도 표현할 수 없습니다.

![스크린샷 2024-12-08 오후 8 45 52](https://github.com/user-attachments/assets/dca152a8-128d-4b27-801e-1e919a239316)

## 임베딩(Embedding)의 개념

- <b>임베딩(Embedding)</b>은 이러한 문제점을 해결하기 위해서 Sparse한 One-hot Encoding의 데이터 표현을 Dense한 표현형태로 변환하는 기법입니다. 이를 위해서 원본 데이터에 Dense한 임베딩 행렬(Embedding Matrix)을 곱해서 데이터의 표현형태를 아래 수식처럼 변환합니다.

![스크린샷 2024-12-08 오후 8 52 14](https://github.com/user-attachments/assets/2603f70d-4d94-4616-b6fd-fe3fbf4db553)


- 아래 그림은 10,000개의 단어사전을 One-hot Encoding으로 표현한 데이터에 10,000×250 크기의 임베딩 행렬을 곱해서 임베딩을 수행한 예시를 보여줍니다.

![스크린샷 2024-12-08 오후 8 52 27](https://github.com/user-attachments/assets/40f924eb-5dd0-41f4-8405-d48641e2a413)


## 임베딩(Embedding)의 장점

- <b>임베딩(Embedding)</b>을 이용한 표현은 다음과 같은 장점이 있습니다.
- 차원 축소
- 유사한 의미를 가진 단어를 유사한 벡터구조로 표현 가능
- 희박(Sparse)한 데이터 표현 형태를 빽빽한(Dense) 데이터 표현 형태로 변경가능
![스크린샷 2024-12-08 오후 8 55 19](https://github.com/user-attachments/assets/cde29443-09a3-4745-86fa-0d551d83cd0c)


## 임베딩(Embedding)을 이용한 사칙연산

- 임베딩(Embedding)을 이용해서 유사한 의미 단어 벡터들간의 덧셈과 뺄셈을 수행할 수 있습니다.
- 예를 들어 king을 표현하는 임베딩 벡터에서 man을 나타내는 임베딩 벡터를 빼고 woman을 나타내는 임베딕 벡터를 더하면 이 벡터의 값이 queen을 나타내는 임베딩 벡터와 유사해집니다.

![스크린샷 2024-12-08 오후 8 55 26](https://github.com/user-attachments/assets/5f145fdc-74fa-43bc-a224-cb0cf1aec043)



- 아래 사이트에서 임베딩 사칙연산을 실습해볼 수 있습니다.
- http://w.elnn.kr/search/

![스크린샷 2024-12-08 오후 8 57 10](https://github.com/user-attachments/assets/b46bdb19-5d9c-44b1-aea7-e97dd63f0445)



- 임베딩(Embedding) 벡터를 tSNE를 이용해서 좌표평면에 그려보면 유사한 의미를 단어를 가진 단어가 유사한 위치에 있는 모습을 확인할 수 있습니다.


![스크린샷 2024-12-08 오후 8 57 58](https://github.com/user-attachments/assets/94c7213a-a5bb-4d54-b3a8-054455b527fe)


