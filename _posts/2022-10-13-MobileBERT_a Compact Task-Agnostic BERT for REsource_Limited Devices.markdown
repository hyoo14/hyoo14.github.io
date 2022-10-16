---
layout: post
title:  "MobileBERT_a Compact Task-Agnostic BERT for REsource_Limited Devices"
date:   2022-10-13 23:00:19 +0900
categories: study
---





{% highlight ruby %}
짧은 요약 :  

pre-train model이 nlp에서 매우 성공  
그러나 헤비한 모델 사이즈와 높은 latency는 아쉬운 점  
->해결책으로 모바일 버트 제안  
-버트라지 thin버전  
-보틀넥 구조 사용  
-지식 전이 기법 이용한 티처 구조 만들어서 사용  
-버트베이스보다 4.3배 작아지고 5.5배 빨라짐  
-GLUE 77.7점으로 버트베이스보다 0.6만 낮음  
-SQuAD F1 90/79.2로 버트보다 1.5/1.2 높음  

{% endhighlight %}


[링크](https://drive.google.com/drive/folders/1ztQKESAp8HbBBCRgYTgs_7VbM21V-m6W?usp=sharing)


# 단어정리  
*discrepancy: 차이/불일치, intermediate: 중간의, generic: 일반적인    


# 1 Introduction  
*프리트레인 self supervised러닝 성능 폭발적  
**천만 파라미터 가짐  
**헤비한 모델 & 높은 레이턴시  
**리소스 제한된 모바일 기기서 못 씀  
*해결책으로 distillBERT 나옴  
**compact한 버트 만드는 것  
**task-agnostic에서 쓰기 어렵  
*모바일 버트 실험  
**largeBERT 파인튠해서 teacher모델 만들고 distill하는 것  
*컴팩트 버트 쉬워보이지만 그렇지 않음  
**narrow, shallow한 버트 만들면 끝일 것 같지만  
**convex combination에 수렵하게 하고 prediction loss & convex combination 해주면  
***정작 정확도 많이 떨어짐  
***얕은 네트워크는 표현이 불충분함  
***좁은 네트워크는 학습이 힘듬  


*제안하는 모바일버트는  
**narrow but bottleneck구조로 self어텐션과 FFNN 벨런싱 해줌  
**깊고 얕은 모델 위해 티처 학습(특별 제작한)  
**지식전이 기법 사용  


*결과  
**4.3배 축소, 5.5배 속도 증가  
**NLP벤치마크 중 GLUE의 경우 약간 미흡하나 성능 비슷, SQuAD는 성능 오히려 향상시킴  



# 2 Related Work  
*버트 압축 시도들  
**프리트레인 더 작은 버트  
***task specific 지식 정제 위한  
**버트 정제  
***매우 작은 lstm으로 시퀀스 레이블링 위한  
**싱글테스크 버트로  
***멀티태스크 버트 학습  
**앙상블 버트를  
***싱글 버트로 정제  
**본 논문과 동시에 distill BERT 나옴  
***student 얕게, 추가 지식 전이-히든레이어  
**TinyBERT  
***layer-wise distill사용, pre/finetune에 모두 사용  
**DistillBERT  
***depth만 조정한 BERT  
**본 논문 제안  
***지식전이만 사용, 프리트레인에서만 모디파이  
***depth말고 width줄임->더 효과적  



# 3 Mobile BERT  
*모바일 버트  
**세부구조, 학습전략 소개  


# 3.1 Bottleneck and Inverted-bottleneck  
*모바일버트 버트라지만큼 deep  
**블록은 더 작음-히든디멘션128  
**두 선형변환으로 in/out 512로 조절해줌(보틀넥이라 명명)  
*deep and thin model 학습의 어려움  
**티처 네트워크 구현 -> 지식전이 -> 모바일버트 성능 향상  
**티처넷 아키텍처-버트라지+인버티드 보틀넥=IB-BERT_LARGE = 모바일버트와 같은 구조, 512 feature map size  
***IB버트 모바일버트 바로 비교 가능  
*보틀넥-> 모바일버트 , 인버티드 보틀넥-> IB버트  
**둘 다 쓰면 버트까지 보존, 모바일버트 compact 보존  


# 3.2 Stacked Feed-Forward Networks  
*MHA(Multi Head Attention)과 FFB 비교 벨런스 복구 문제 있음  
**MHA: 다른 공간 info 묶어줌  
**FFN: non-linearlity 증대  
**오리지널버트: MHA,FFN=1:2  
**보틀넥구조: MHA 더 넓은 피처맵, input FFN 좁음, MHA가 파라미터 더 많은 문제   
*Stacked FFN-해결책으로 제안  
**벨런스 맞춰줌  
***어텐션 마다 4개 stacked FFN 사용  



# 3.3 Operational Optimization  
*연산 최적화  
**layer norm 대신 선형변환 NoNorm(h) = r o(Hadanard Product) h + b 사용  
**gelu 대신 relu 사용  


# 3.4 Embedding Factorization  
*임베딩 table 축소 512-> (128 + 3raw token)  


# 3.5 Training Objectives  
*2개 목적함수  
**faster map trasform  
**attention trasfer  
최종은 위 둘 linear combination 사용  


*Feature Map Transfer(FMT)  
**feature map  
**teacher와 가깝게 된  
**MSE사용 -  모바일 버트 피처맵과 IB버트 사이 차이를  
**용어 l은 layer index, T는 seq 크기, Ni는 feature map size, 학습 안정화 도움(descrepency와 loss term이)  


 
*Attention Transfer(AT)  
**어텐션이 NLP 성능 올리고 버트 핵심인데  
**셀프어텐션맵 사용(teacher로부터 학습)  
**KL divergence 낮추는 것 목적함수로  
**A는 어텐션 헤드 수  


*Pre-training Distillation(PD)  
**그 전에 knowledge distillation loss 사용  
**linear combination of MLM loss and NSP loss  
**MLM KD loss  
**alpha는 hyperparameter in (0,1)  



# 3.6 Training Strategies  
*학습전략 소개  
*보조자 지식 전이 Auxiliary Knowldege Transfer  
**지식 정제의 보조자로 여기는 전략  
**선형 합성 (loos들을) pre-train 정제 loss처럼  
*JointKnowledgeTransfer  
**layer wise 지식 전이 loss 같이 train하는 경우  
*Progressive Knowlege Transfer  
**IB버트 성능 못 따라온 경우, 각 layer별 progressive knowledge transfer  
*Diagram of three strategeis  
**낮은 layer에 적은 latency rate, 전체 freezing 대신  


# 4 Experiments   
*test진행   


# 4.1 Model Settings  
*세팅 찾기  
**SQuAD v1.1, F1 score 메트릭으로 125k step 2048 배치  
*IB버트(리턴모델)  
**작은 inter-block hidden size  
**inter-block batch size 줄임  
***performance에 영향 없었음  512까지  
**그래서 IB버트 인터블럭 512로 함  
**intra block은 줄이면 성능 많이 떨어져서 건드리지 않음  


*아키텍처 최적 탐색  
**4배 압축 25M파라미터 in MHA, FFN for 모바일버트(학생)  
**최적 벨런스 MHA, FFN 의 비율은 비율 MHA가 FFN 0.4~0.6배  
***오리지널 버트가 0.5인 이유   
**128인트라블록, 히든사ㅣ즈 4 stacked FFN(정확하고 효율적이었음)  
**티처 어텐션 헤드 4개  
**어텐션헤드 16->4로 줄여도 성능에 지장은 없었음  


# 4.2 Implementation Details  
*구현 상세  
**BOOK CORPUS + ENGLISH WIKI data for pretraining  
**버트라지와 비슷한 정확도  
**256 TPU v3 500K step 배치 4096 LAMB optimizer  
***trick은 사용 안 함(성능 높이는 트릭..추가적인 휴리스틱 학습 같은?)  
**같은 training 스케쥴(프리트레인 정제 과정 사용, 공정 위해서)  
**지식 전이 추가 240K step 24 layer  
**공정 위해 joint knowledge transfer & auxiliary knowledge transfer 다 240K step  
**파인튜닝 오리지널 버트처럼 진행  
**batch 16/32/48  
**l.r (1-10)*e-5  
**epoch 2-10  
**서치스페이스 다르므로 모바일 버트는 l.r 높이는 것이 권장됨 epoch도 파인튜닝시 높은 것 권장  



# 4.3 Results on GLUE  
*결과  
**GLUE 9task 리더보드 모델들과 비교  
**모바일버트 경쟁력 있음, 버트베이스보다 단지 0.6 점수 낮음  
**4.3배 작고, 5.5배나 빠름  
**GPT보다 0.8점 높고, 사이즈는 4.3배 작음  


# 4.4 Results on SQuAD  
*SQuAD 결과  
**참고로 v1.1은 모두 답이 있고, v2.0은 답 없는 것도 있음  
**single model은 리더보드에 없어서 특정 버트들과 따로 비교함  
**결과는 모바일 버트가 압도  


# 4.5 Quantization  
*수량화? 양자화?  
**해도 성능저하 없었음  
**아직 압축 가능성 많이 남아 있다고 판단  


# 4.6 Ablation Studies  
# 4.6.1 Operational Optimizations  
*NoNorm, relu opertation optimization 잘 되지만 FLOPS는 optimization 안 되서 real world와 성능 gap 생김  


# 4.6.2 Training Strategies  
*학습전략 달리해서 test  
**auxiliary knowledge transfer가 우위  
**이유는 student에 추가 학습 필요하기 때문  


# 4.6.3 Training Objectives  
*학습 목적함수  
**경감 스터디에서 Attention Transfer, Feature Map Transfer, Pretraining Distillation, Operational Optimization 제거 test  
**AT, PD성능 향상 잘 됨  
**IB 버트 라지 긍정 역할  
**향상여지 많음.. 왜냐하면 teacher에 비해 성능이 떨어지므로  


# 5 Conclusion  
*결론  
**모바일 버트-테스크 무관하게 컴팩트한 버트  
***버트보다 작고 빠르며 NLPtask에서 경쟁력이 입증됨  
**모바일버트 deep, thin을 소개함  
**bottleneck, inverted bottleneck 소개  
**지식 전이 소개  
**일반화되고 응용 잘 되길 기대  