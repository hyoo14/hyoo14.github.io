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
*consecutive: 연속적인(연이은), quantifies: 양화(증량화), auxiliary: 보조자, amenable: 받을 수 있는,  
disentangle: 풀다  


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
*.  


 
