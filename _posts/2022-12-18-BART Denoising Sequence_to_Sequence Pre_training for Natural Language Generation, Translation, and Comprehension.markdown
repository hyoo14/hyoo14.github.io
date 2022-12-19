---
layout: post
title:  "BART Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension"
date:   2022-12-18 20:00:19 +0900
categories: study
---





{% highlight ruby %}
짧은 요약 :  

BART -> 노이즈 제거 오토인코더  
Seq2Seq PLM 이용  
학습 (1) 노이즈 주고   
      (2) 복구(corrupted text -> original)  
트랜스포머 기반 버트, gpt 등 사용  
-노이즈 주는 메서드들 평가->랜덤셔플이 best  
                                -> infilling scheme  ->single noise token 교체  
-TEXT GENERATION에 효과적, TEXT UNDERSTANDING에도 괜찮은 성능  
(RoBERTaW정도의 성능 IN GLUE, SQuAD)  
(SOTA in 대화, QA, 요약-3.5ROGUE, 번역-1.1BLEU)  
-다른 pre-Train 메서드들 적용해봄(효과 검증 위해)  
    
{% endhighlight %}


[링크](https://drive.google.com/drive/folders/1ejDoh5Iyh49gi0zI6Z0Q0h3LFGYxbnTL?usp=sharing)


# 단어정리  
*o: ㅇ  


# 1 Introduction  
*self supervised learning은 nlp서 매우 성공적  
**MLM이 대표적  
***노이즈 제거 오토인코더  
***복원 목적  
**최근 분포 개선에 중점  
***특정 task에 국한되는 한계에 봉착  
*BART 제안  
**디노이징 오토인코더 빌트인  
***즉 pre train + Bidirectional Autoregressive Transformer  
(1) text에 노이즈 추가  
(2) Seq2seqㄹㅎ 복원(노이즈 제거) 학습  
**버트, GPT 등 사용  
*노이즈 유연성 이점  
**랜덤셔플링 + 스킴 채우기  
***버트 mask, NSP 일반화 버전  
***더 긴 문장 커버  
*BART 성능  
**text generation 에서 좋은 성능  
**GLUE, SQuAD 에서 RoBERTa만큼  
**SOTA in 추상대화, QA, 요약-3.5ROGUE/XSum도 능가  
*BART -> 파인튜닝 새로운 방식 제시  
**추가 트랜스포머 레이어 쌓음  
**레이어는 외국어->노이즈 영어 BART 통해 번역  
**Back translation MT 1.1BLeU 달성 in WMT romance  
*경감 study for 이유 확인  
**Data 와 parameter 최적화 중요  
**BART 성능 full range task서 성능 최고  



# 2. Model  
*ㅇ





