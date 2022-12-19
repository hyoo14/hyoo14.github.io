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
*BART는 노이즈 제거 오토인코더  
**문서에 노이즈 추가->원본 문서로 복구  
**Seq2seq로 bidirectional 인코더와 left to right 디코더로 구성  
**프리트레인 시 negative log likelihood 최적화  
*BART는 일반 seq2seq 트랜스포머 구조 GPT 따름, 몇몇 제외  
**ReLU->GeLU  
**이니셜파라미터 N(0, 0.02) 
**일반모델 6layer, 라지 12layer  
**버트와 다른점  
(1) 각 디코더 레이어에 cross attention over final hidden layer가 인코더에 추가됨  
(2) 추가 FFNN 없고 10퍼센트 파라미터 더 씀  


# 2.2 Pre-training BART  
*바트는 노이즈된 문서 복원  
**cross entropy 최적화  
(디코더 output과 원본 사이)  
**다른 디노이징과 달리 특정 노이징 스킴 따름  
**노이징 소스 없을시 그냥 LM  
**아래 스킴을 test  
*token masking  
**버트 따름, 랜덤 토큰을 [mask]로 바꿈  
*토큰 삭제  
**랜덤 토큰 삭제  
*Text Infilling(텍스트 채우기)  
**채울 길이를 샘플링-포아송분포 이용  
**SpanBERT에서 영감  
but 본 모델은 분포 다르고 seq token 길이 다름  
*문장 순서 바꾸기  
**다큐먼트의 문장을 랜덤 셔플링  
*문서 회전  
**토큰이 유니크 랜덤 픽, 다큐먼트 로테이트,  
시작 다큐먼트 식별?  



# 3. Fine-tuning BART  
BART 다양하게 사용(응용)  


# 3.1 Sequence Classification Tasks  
*Seq 분류  
**인코더, 디코더에 같은 인풋  
**cls토큰이 역할하듯이 final hidden state가 분류  
**디코더가 가짐  


*토큰 분류  
**SQuAD의 답 end point 분류같은 것  
**full 문서를 input으로 인코딩, 디코딩  
**디코더의 top hidden state를 분류용으로 사용  


*Seq 생성  
**BART는 autoregressive 디코더에서 QA나 요약같은 sequence generatio에 직결  
**정보는 카피되고 조정됨  
**디노이징 프리트레인 연관  
**인코더 input 과 디코더 output이 오토리그레시브  





