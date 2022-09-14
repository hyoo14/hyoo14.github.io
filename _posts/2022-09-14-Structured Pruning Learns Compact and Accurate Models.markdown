---
layout: post
title:  "Structured Pruning Learns Compact and Accurate Models"
date:   2022-09-14 00:02:19 +0900
categories: study
---





{% highlight ruby %}
짧은 요약 :  

모델 압축이 요지(아래 방법론 사용)  
-가지치기  
-정제  


가지치기만 사용할 경우 사이즈는 작아지나 속도는 느림  
정제 사용 경우 속도는 빨라지지만 데이터가 많이 필요하고 학습능력도 필요  

CoFi제안 (Coarse-graind and fined-grained prunning)  
-coarse : 많은 양의 데이터, 적게 보냄(레이어 전체 등 큰 단위)  
-fined : 적은 양의 데이터, 많이 보냄(어텐션의 헤드나 히든디멘션 처럼 위보다 작은 단위)  


CoFi는 병렬네트워크이고 정제메서드와 매칭되서 정확도와 레이턴시가 좋고, 언라벨 재분류가 필요 없음  


세분화 mask로 가치치기 디시전하고, layerwise distilation 하기 때문에 최적화됨  


GLUE, SQuAD 실험에서 속도 10배 빠르고, 성능은 90%로 방어함  
(이전 모델들 압도)  


{% endhighlight %}


링크 (https://drive.google.com/drive/folders/1ZLbPbKFeexbTR-VJ6mvw6w40i3wGMiGS?usp=sharing)


# 단어정리  
*granularity : 세분성/얼마나 자세하게 분할되는지, account for : ~에 대해/설명하다, interleaving : 끼워넣다,   
task-agnostic : 작업에 구애받지 않는, quantization : 양자화  

# 1 Introduction  
*PLM이 NLP의 주류인데 용량-메모리-연산시간이 매우 큼  
**압축이 필요하고  
***압축방법으로 가지치기와 정제가 있음  


*가지치기는 정확한(성능이 유지되는) sub네트워크 찾는 것  
**layer or head or dimension or blocks(가중행렬) 줄여보는 것  
**위와 같은 fine-grained unit 줄여보는 것이 트렌드  
**성능 향상은 적음 (2,3배 속도 향상)  


*정제모델은 고정 구조로 정제  
**라벨 없는 코퍼스 사용  
**잘 설계된 구조의 학생 모델은 속도-성능 트레이드 오프에서 좋은 균형 잘 찾음  
**근데 학습이 느림  


*특정 작업 위주의 구조화 가지치기인 CoFi 제안  
**거친(self어텐션 전체, FF layer) + 다듬어진(헤드, 히든디멘션) 동시에 가지치기함  
**가지치기 졀정은 매우 세분화해서 함  
**유연성 좋고 최적화 잘 됨  


*가지치기+정제: 성능 올라감  
**고정 학생 구조 대신 레이어와이즈 정제 기법 사용  
***두 구조 사이 역동적 학습  
****성능 잘 나옴  


*CoFi가 정확하고 속도 좋음(GLUE and SQuAD)  
**작고 빠름  



# 2 Background  
# 2.1 Transformer  
*트랜스포머 네트워크 구성  
*L개의 블럭  
**블록당 MHA와 FFN으로 구성  
***MHA에는 쿼리, 키, 벨류, 아웃풋을 인풋으로 하는 어텐션 펑션으로 구성(d:히든레이어)  


*FF레이어  
**up projection(Wu)과 down projection(Wd)으로 구성  
**FFN(x) = gelu(XW0) . Wd  
(MHA와 FFN 이후에 레이어 정규화 있음)  


*MHA, FFN이 각각 모델 파라미터의 1/3, 2/3 비중  
*둘의 GPU타임은 비슷하지만 FFN의 경우 CPU 정체 걸림  


# 2.2 Distillation  
*지식 정제: 전이 지식 압축 모델  
**라벨 없는 데이터로 학습(지식 전이 데이터)  
**2개 합치면 성능 좋음 대신 학습이 비쌈  


*새로운 정제 모델  
기존: 획정된 구조  
신규: 다이나믹(변화) 구조  



# 2.3 Pruning  
*가지치기-반복적 파라미터 제거  
**기존 : 트렌스포머 모델, 다른 컴포넌트에 focus  


*레이어 가지치기  
**트렌스포머 블락들 다 제거  
**경험적으로 50% 드랍 가능  
**스피드 2배 향상  


*헤드 가지치기  
**헤드 제거  
**1.4 배 스피드 향상  


*FFN 가지치기  
**fine-grained 레벨  


*블럭&비구조화 가지치기  
**최적화 어렵고  
**스피드 향상 어려움  


*정제와 같이 가지치기  
**예측 레이어 정제와 동시에 이뤄짐  
**레이어 와이즈에서 어떻게 정제될지 불명확함  



# 3 Method  
*CoFi : 가지치기(거친+다듬어진) + 정제(레이어와이즈-지식전이이용) = 속도 향상, 압축 잘됨  


# 3.1 Coarse- and Fine-Grained Pruning  
*최근 구조적 가지치기: 적은 유닛 가지치기로 유연성 향상  
**파인 그레인 가지치기는 거친 가지치기 수반함  
***head 가지치기 = MHA 가지치기에 수렴  
****최적화가 어려움  


*해법: MHA, FFN 파인그레인 가지치기되게끔 마스킹 추가  
**Z_MHA, Z_FFN 마스킹 추가  


*레이어마스킹 - 전체레이어 외적 가지치기(헤드(MHA) 전체 가지쳐버리는 방식 방지)  
**MHA, FFN 따로 따로  


*output 차원도 가지치기(히든디멘션 차원) - 조금 가지치기되지만 성능은 향상됨  

*data granularity : 데이터가 얼마나 자세히 분할되는지  