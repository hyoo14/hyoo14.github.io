---
layout: post
title:  "2021-11-08-Attention Is All You Need"
date:   2021-11-08 12:55:10 +0900
categories: study
---




{% highlight ruby %}
짧은 요약 :

어텐션만으로된 트랜스포머네트워크 제안
성능이 기존 RNN, CNN 보다 좋음 SOTA

(QA, LI 등)

{% endhighlight %}


*RNN, LSTM, GRU가 LM이나 MT에서 이전의 SOTA였음
**하지만 적용 범위를 넓히는 노력이 있었음에도, 이 적용범위에 한계가 있었음
***문장이나 텍스트가 너무 길어질 경우 성능이 떨어지는 한계

**또한 sequence 계산이 병렬적으로 안 되는 문제도 있음

*여기서 제안하는 어텐션은 이러한 길이제한 문제와 병렬 계산 문제를 모두 풀어줄 해법이 됨

*트랜스포머도 이전의 sequence용 모델들 처럼 encoder-decoder 구조. autoregressive함
**self-attention, poitwise FCL로 아키택처 구현

*어텐션은 query를 입력받으면 key-value 쌍으로 매핑해주는 것
**이 때, key, query, value 모두 백터임. output도
**여기서는 scaled dot-product attention 사용함
***query와 모든 key들을 dot-product(외적)하고 차원크기로 나눠주는 것
****참고로 dot-product할 때마다 차원 수가 너무 커져서 나눠주는 것임. 

*병렬 가능케하는 것은 다음과 같음
**어텐션 함수들을 병렬적으로 수행하고 후에 concat해줌

*멀티헤드는 각각 다른 관점을 대표하는 것.

*인코더의 self-attention은 q,k,v가 같은 곳에서 옴.
**output은 이전 layer encoder서 옴

*인코더-디코더 어텐션은 이전 디코더의 쿼리가 in. 인코더에서 나온 key/value가 나옴.
**모든 디코더 위치서 input으로 가능. seq2seq의 인코더 디코더 구조 유사하게 구성

*디코더의 self-attention은 인코더의 것과 같으나 masking을 해주는 차이(답 숨기는 masking)

*FCFN은 potition에 적용하는데 ReLU 2개로 구성됨
**kernel size 1짜리 두 Convolution이라고 표현할 수도 있음

*학습된 임베딩을 사용하고 학습도니 linear-transformation과 softmax를 사용함

*위치정보는 포지션인코딩 사용.

*왜 self-attention사용하는가 하면,,
**컴퓨터 복잡도 줄여주니깐.. 병렬성 가능케하고.
**긴 거리 의존성도 가장 좋음.

*실제 테스트에서도 결과가 좋았음


