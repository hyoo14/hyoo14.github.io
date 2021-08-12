---
layout: post
title: AI CONCEPTS NOTE
date: '2021-08-04 11:52:10 +0900'
categories: _NLP study
published: true
---


한번 정리하고 넘어가고 싶어서 개념들 서술해봅니다.



-2-

랭기지모델 학습 - 단어 분포 모사. 마코브 어썸션 도입하여 더 낮은 엔그램으로 근사

좋은 모델 : 언어분포 잘 모사. 잘 정의된 테스트셋에 높은 확률 반환

펄플렉시티: 몇개중에 헷갈리고 있냐. 작을 수록 좋은 것

뉴럴랭기지 모델은 제너럴라이즈가 잘 된다는 것이 장점, 등장하지 않은 단어들도 처리 가능

엔트로피 불확실성 나타냄. 자주 일어나는 사건은 낮은 정보량. 드물게 발생하면 높은 정보량 가짐.
불확실성은 1/등장확률 에 비례하고, 정보량에 반비례
확률 낮은 것일 수록 정보량 많은 것, 불확실성 큰 것
정보량은 마이너스 로그 확률. 확률 높아질수록 정보량 낮아짐. 0가까워질수록 높아짐

엔트로피 높을수록 플랫 디스트리뷰션, 낮을수록 샤프한 디스트리뷰션

크로스엔트로피는 퍼플렉시티의 로그취한 값.
퍼플렉시티 작게하는 것이 목적으로 이는 크로스엔트로피 익스퍼넨셜 미니마이즈.
즉, 크로스엔트로피 미니마이즈로 볼 수 있음

클래시피케이션이니까 크로스엔트로피 쓴다고 봐도 되고
두 분포 차이 미니마이즈니까 크로스엔트로피 미니마이즈.
아키르릴후드 맥시마이즈 할꺼니까 네거티브 라이클리후드 마니마이즈 하는 건데 이게 크로스엔트로피 미니마이즈와 같음.

크로스엔트로피 로그 취하면 퍼플렉시티.
퍼플렉시티 익스퍼넨셜 취하면 크로스엔트로피임.

seq2seq의 many to many는 many to one 과 one to many 로 이해하는 것이 좋음

non-auto-regessive-현재 상태가 앞 뒤 문맥에 의해 정해짐
auto-regressive- 과거상태 의존

티처포싱 안 하면 many2many 개수 안 맞을 수도. eos먼저 뜨면
티처포싱 안 하면 이전 출력에 따라 현재 스테이트 달라짐 mle 이상해짐
mle다르게 수식 적용
그래서 실제 정답을 넣으줌. 그래서 학습이랑 테스트(추론) 다로 2개 짜야함
티처포싱 성능 저하될 수 있으나 기본적 성능 좋아서 걍 쓰면 되지만.. 왜 저하되냐면 순서까지도 기억해주는 오버피팅이 되기때문. 그래서 보통 반반티처포싱 등 사용

fluent한 문장 골라내는 일이나 다음단어 뽑아내는 일은 언어모델로 사실 같음.


-4-
seq to seq 도 auto encoder와 비슷
autoencoder - 특징 추출 하는 것.(차원축소(latent space:잠재 feature의 공간?) 및 복원을 통해)

decoder는 conditional language model이라고 볼 수 있음-인코더로부터 문장을 압축한 context vector를 바탕으로 문장생성
(conditional? 조건부 확률 느낌?)

classification(단어선택), discrete value 예측하는 거니까 크로스엔트로피 쓰면 됨. (소프트맥스 결과값에)
MLE중이니까 negative log likelihood 미니마이즈해야히니까 크로스엔트로피.
디코더가 컨디셔널랭기지 모델이니깐 퍼플렉시티 미니마이즈 해야하는 것. 그니깐 크로스엔트로피 익스포넨셜
이 코르스엔트로피니깐. 크로스엔트로피 쓰면 
결국 ground truth 분포와 모델 분포 삳이의 차이를 최소화하기 위함
(Ground-truth는 학습하고자 하는 데이터의 원본 혹은 실제 값)

어텐션?
키-벨류 펑션으로 미분가능함
파이선의 딕셔너리= {K:v1, K2:v2} ,,, d[K] 벨류리턴 - (딕셔너리설명)
기존 딕셔너리 key-value 펑션과 달리 query와 key의 "유사도"에 따라 value 반환!(weighted sum으로)
->lstm hidden state한계인 부족정보를 직접 encoder에서 조회해서 예측에 필요한 정보 얻어오는 것.
정보를 잘 얻어오기(이 과정이 attention) 위해 query 잘 만들어내는 과정을 학습 

attention은 QKV.
Q:현재 time-step의 decoder의 output
K: 각 time-step 별 encoder의 output
V: 각 time-step 별 encoder의 output
q와 k의 유사도 계산(encoder output token들)
유사도를 각각 encoder output toekn들에 곱해주고 더해서 현재 context vector 만들어줌.
쿼리 날린 decoder 히든스테이트와 context vector를 컨캣해줘서 새로운 히든스테이트를 얻음(이것이 반영 된 것)
근데 유사도 구할 때 유사도 팍팍 안 구해짐. 그래서 linear transform(linear layer 통과시킴) 후에 유사도 얻어옴.
그래서 우리는 linear transform을 학습해서 유사도를 잘 받아오게 학습해야함. 이것도 잘 학습해야함

비유해보면 "오리역에서 가장 편하게 밥먹는 집 어디야?" -> "오리역 가정식 백반집"(쿼리 바꾸는 거 학습한 것)
마음속의 상태(state)를 잘 반영하면서 좋은 쿼리를 만들기 위함임.
어텐션은 "쿼리를 잘 만들어내는(변환) 과정" 배우는 것이다. 여기서 batch matrix multiplication(BMM) 사용(행렬들 곱)
닷프로덕트가 코싸인시뮬러리티와 비슷. 소프트맥스까지 쒸우면 유사도라고 볼 수 있음

<PAD> 위치에 weight가 가지 않도록 하는 안전장치 추가->이것이 마스킹


input feeding? 샘플링과정서 손실되는 정보 최소화, 티쳐포싱으로 인한 학습/추론 사이의 괴리 최소화
인풋피딩? 아웃풋 히든스테이트를을 다음 인풋 히든스테이트에 컨캣

auto-regressive : 과거 자신의 상태를 참조하여 현재 자신의 상태를 업데이트. 시퀀스에서 이전 단어 보고
다음 단어 예측하는 것.

teacher forcing 통해 auto-regressive task에 대한 sequential modeling 가능. 하지만 training mode 와 inference mode의 괴리
(discrepancy) 생김

(dilde : ~ (물결표) )

pytorch ignite로 procedure 짜놓을 수 있음(lightening도 가능)


{% highlight ruby %}

{% endhighlight %}
