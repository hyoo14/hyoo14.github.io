 ---
layout: post
title:  "[2014]Sequence to Sequence Learning with Neural Networks"
date:   2023-03-23 16:49:33 +0900
categories: study
---


{% highlight ruby %}
짧은 요약(Abstract) :    
* DNN powerful but seq2seq 'X'  
** multi layer LSTM으로 seq2seq 'O'  
** input->deepLSTM enocder->vector->deepLSTM decoder->target 
** eng2french -data: WMT14 dataset   
** result: BLEU 34.8 for test  
** OOV 때문에 BLEU 저하  
** long sent 잘 다룸  
** 대조군 : phrase 기반 SMT BLEU33.3  
** LSTM rerank 1000hypothesis after SMT: BLEU 36.5 - SOTA근접  
** word order에 민감, 상대적으로 active%passive voice에 invarient  
** 역순 입력으로 LSTM 성능 많이 올림(short term dependency 줌-> 소스와 타겟 사이->optimal easier)  


   
{% endhighlight %}


[Paper with my notes](https://drive.google.com/drive/folders/1aCKj7Q_oCMt4V51PrkqUbX2XBBblW5UC?usp=sharing)  


[~~Lecture link~~]()  


# 단어정리  
* invarient: 변함없는, 변치 않는  
* quadratic: 이차의(이차방정식할 때)  
* pose a challenge: 도전하다  
* time lag: 시간상의 차이  
* non-monotonic: 비단조  
* negligible: 무시해도 될 정도의, 무시해도 좋은   

   

# 1 Introduction  
* DNN 성능 굿  
** speech recog, visual object recog  
** 병렬이라 강력  
** N bit 숫자 2 hidden layer로 정열->강력  
** 통계모델과 연관  
** 큰 DNN & 충분 정보->좋은 결과  
* input과 output이 벡터나 고정차원으로 잘 인코드될 때 성능 좋음  
** 이건 제약이 됨  
** seq( 길이 모르는/ 어디가 더 중요한지 모르는) 처리 어렵  
** speech recog, MT 가 seq  
** QA도  
* seq2seq가 유용함  
** DNN엔 in/out 차원 fit해야되서 어렵  
** LSTM으로 seq2seq 구현하여 해결  
** input->LSTM->hidden state->LSTM->out  
* NN이용하는 관련연구 많음  
** 본 논문은 Kalchbenmer, Blunsom과 연관  
** 전체 input+seq vector와 매핑  
** Cho, Graves의 attention과 비슷  
** Bahdanan는 NN focus 다른 부분의 input & MT에 적합     
** seq 분류도 유명 seq2seq 매핑 NN(모노토닉이지만  
* 본 논문 거꾸로 읽게해서 이득봄  
* 결과  
** WMT14 영프 번역결과 BLEU34.81(5 deep LSTM 앙상블함)  
** 큰 NN 중 최고 성능  
** vocab 80k개로 얻은 결과  
** 성능 개선 여지 많음에도 이미 phrase 기반 SMT 압도  
** SMT 1000list 중 LSTM으로 점수 조정할 경우 BLEU 36.5로 sota 37.0에 근접(base보다 3.2 향상)  
* LSTM 긴문장에 취약하다고 하지만 거꾸로 입력하여(출력은 올바르게) 좋은 성능 얻음  
** 적은 의존성->최적화 더 쉬워짐  
** SGD->LSTM문제 사용 안된다고 하지만, 거꾸로 입력할 경우 긴문장이어도 SGD 가능  
* LSTM 유용 특성:  
** map 학습시, 다양한 길이 input sent를 고정차원벡터로 표현 가능  
** 주어진 번역 소스 sent 파라프레이징하는 경향 보임  
** LSTM이 유사한 의미도 잘 찾음  
** 질적평가서 증명됨  
** active/passive voice에 구애받지 않음  

<br/>

# 2 The model   
* RNN은 seq대한 FFNN의 자연생성  
** seq input(X1,..,XT) wndjwlaus (Y1,...,YT) 출력  
** 다음을 순환 반복하여.. 
*** ht = sigm(W^hx xt + W^hh ht-1)  
*** yt = W^yh ht  
** RNN이 길이 같게 fit하여 in/out 쉽게 함  
** 하지만 길이 다를 때 non-monotonic 일때 불확실  
** seq 학습 매핑 RNN: in->RNN - RNN->out  
** 긴 문장 처리 이슈->LSTM은 해결  
** LSTM 목표: 조건부 확률 측정  
*** P(y1...yT't|x1...XT) = ㅠ t=1...T' p(yt|v, y1,...,yt-1)  
*** (X1...XT) input, (y1...yT') output, T!=T'   
*** X(1...T) -> LSTM(v fixed dim) -> hidden state -> LSTM -> Y(1...T')  
*** P(yt|v, y1,...yt-1) 분포는 softmax(모든 vocab 단어에 대한)  
*** Graves의 LSTM식 사용, <EOS>는 문장 끝 나타냄->유용  
** 본 모델은 위에 소개한 것과 3가지가 특히 다름  
*** 1. 2개 다른 LSTM 사용: 하나는 input, 다른건 output -> 파라미터수 많고 병렬 가능(초기 학습)  
*** 2. deep LSTM(4개 레이어) 사용(shallow 압도)  
*** 3. input 거꾸로 입력 -> SGD가 연관 더 잘 짓게함 -> 성능 향상  

<br/>

# 3 Experiments  
* WMT'14 영프 MT task 2가지로 적용  
** 1. 번역 바로, SMT refer 없이  
** 2. SMT 베이스에 점수 재정의  
** 정확도, 예시, 시각화 보여줌  


## 3.1 Dataset details  
* 12M 문장 subset으로 학습  
** 348M 프랑스어 단어 & 304M개의 영어 단어(clean, selected)  
** 사용 이유: public, tokenized, 1000 best lists SMT가 제공됨  
* NL 모델 단어벡터에 종속  
** 160,000 다빈도 소스 단어와 80,000 다빈도 타겟 언어 단어 사용  
** OOV는 UNK로 대체  


## 3.2 Decoding and Rescoring  
** maximize log확률 -> 올바른 번역T, 주어진 소스문장 S일때  
** S가 학습셋일때, 학습 후 LSTM으로 가장 유사한 번역 찾음  
** left-to-right 빔서치 디코더 사용  
*** 작은 수 B의 부분 가정 유지?  
**** 가능 단어 다 따지면 계산 너무 커지니까 B만 남김(가장 유사한 B개의 단어만 계산에 반영)  
** <EOS>는 빔서치서 제외  
** 1000 best -> 빔사이즈1, 2에서 성능 괜찮게 나옴  
** 리스코어링 때 LSTM 이용하여 log 확률 계산, 평균 메김  


## 3.3 Reversing the Source Sentences   
* 거꾸로 효과  
** PP5.8 -> 4.7로 감소(성능 향상)  
** BLEU 25.9 -> 30.6 으로 향상  
** 정확한 이유 모르나 short term dependency 이유로 추정  
** minimal time lag 감소  
** Backprofagation 쉬워짐  
** 긴문장서 특히 거꾸로 입력 효과가 큼  


## 3.4 Training details  
*구조 디테일  
**Deep LSTM 4layers 1000 cells 각 레이어 1000차원 Word Embedding  
***input vocab 160,000개, output vocab 80,000개  
** Deep이 shallow 압도, 추가 Layer가 PP 10% 낮춤, hidden state 더 많아서..  
** 기본 소프트맥스 80,000 단어 output으로 씀  
** LSTM은 38M 파라미터 가짐  
*** 64M은 recurrent connection용(32M for 인코더 LSTM, 32M for 디코더 LSTM)  
* 트레이닝 디테일  
** 파라미터 초기화 with uniform distribution -0.08~0.08  
** SGD 모멘텀 없이 사용, l.r 0.7인데 초기 5에폭까지만 적용하고 이후 반에폭마다 반감시킴, total 7.5epoch   
** vanishing gradient문제는 없을지라도 exploding gradient 문제는 있을 수 있음  
*** 제약을 줌-> norm of gradient [10, 25]로  
*** s = ||g||2, g는 gradient/128  
*** if s>5 then g=5g / s 로 세팅  
** 다른 문장은 다른 길이 가짐, 대부분 20-30 길이로 짧지만 일부 100이상의 길이 가짐  
** 그래서 미니배치 128 랜덤픽으로 학습문장 고름  
*** 짧은 문장이 많아서 배치 낭비임  
*** 위 해결책으로 미니배치 같게하면 속도 2배 높일 수 있음  


## 3.5 Parallerlization  
* 병렬화  
** c++ 구현체, 1개 GPU 속도는 1,700words/s -> 너무 느림  
** 8GPU 병렬화, 각 LSTM 레이어 다른 GPU서 처리하고 communicate함 또 다른 GPU에서  
** 남은 4GPU는 병렬 softmax에 사용, 각 GPU는 1000 X 20000 mat 곱함  
** 결과: 6,300words/s with minibatch size128로 속도향상, 학습에 총 10일 걸림  


## 3.6 Experimental Results  
* 실험 결과  
** cased BLEU 점수 사용(번역질 test)  
** multi-bleu.pl로 계산  
*** 33.3점 [29]번 모델  
*** SOTA 재보니 기존에 알려진 35.8보다 높은 37.0점 나옴  
** 결과는 테이블1,2에 있음  
*** best 결과(앙상블) 랜덤 초기화 & 랜덤 minibatch, 디코더 번역 LSTM 앙상블로 sota는 달성 못 함  
*** phase 기반 SMT는 뛰어넘음 NN으로  


## 3.7 Performance on long sentences  
* 긴문장 성능 좋음  


## 3.8 Model Analysis  
* 강점 : sequence -> vector(fixed dimemtion)  
** 단어 순서에 민감(긍정적)  
** 능동/수동 차이엔 둔감(긍정적)  

<br/>

# 4 Related work  
* NN->MT 연구 많음  
** RNNLM 간단하고 효과적  
** NNLM(MT용) 리스코어링, 강한 MT의 baseline  
** 최근 연구자들 소스언어정보 NNLM에 포함시키는 연구시작  
*** NNLM + topic(input) ->rescore 성능 향상  (인코더에서)  
*** 디코더는 같이 안 함(이게 중요한데 안 함)  
*** 무튼 성능 많이 올림  
** 본 논문은 Kolchbremer&Blunsom 연구와 연관  
*** map input sentence -> vector -> sentence  
**** 단 이 때, input to vector 때 CNN을 써서 단어순서 잃음  
** Cho도 유사 LSTM같은 RNN 씀, sent->vector->sent 구조, NN을 SMT에 통합하는 방향이었음  
** Bahdanau는 직번역 NN에 어텐션 사용, 긴문장 질 떨어지는 문제 해결, 결과 굿  
** Pouget-Abadie도 메모리 문제 해결 시도, 문장 부분해석, smooth 번역사용, phase based SMT와 비슷  
*** 리버스한 거랑 성과 비슷   
** Hermann도 End-to-End에 초점 맞춤 -> input -FFNN-> output  
*** 하지만 번역을 바로 하지는 못함.. 번역하려면 리스코어링 할 수 있는 사전 계산 DB가 잇어야함(가까운 백터 룩업용)  

<br/>  

# 5 Conclulsion  
** 라지 딥 LSTM 적은 vocab으로 SMT압도(무제한 vocab인SMT)   
** Seq에서 성능 좋음  
** 리버스 input으로 RNN 문제 어느정도 해결  
** LSTM만으로 긴문장 처리 굿  
** Seq 문제 더 잘 적용 굿  
 











