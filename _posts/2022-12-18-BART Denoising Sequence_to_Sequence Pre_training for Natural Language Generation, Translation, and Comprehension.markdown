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



# 3.4 Machine Translation  
*BART로 MT 향상, 디코더 영번역서  
**이전엔 프리트레인 인코더 향상, 여기선 pre train LM으로 향상   
**BART서 가능, 프리트레인 디코더에서 인코더 파라미터 추가로 bitext 학습  
**바트 인코더에 새 랜덤초기화 인코더  
end to end 학습  
새 인코더가 외국어->노이즈 영어로  
**새 인코더 단어(오리지널 바트) 사용  
**소스 인코더 두단계-역전파 cross entropy loss  
(1) BART 파라미터 freeze, 랜덤이니셜라이징된 것 +self어텐션 input+포지션임베딩만 업데이트  
(2) 모델 파라미터 학습을 적은 반복으로  


# 4 Comparing Pre-training Objectives  
BART는 더 넓은 노이징 스킴을 프리트레인일 때 지원  
*base모델로 비교들 함  


# 4.1 Comparison Objectives  
*비교 목적  
**학습 부분, 자원, 구조 비교  
*모델, 파인튠 과정도 비교  
**discrimination&generation tasks 프리트레인으로 재구현  
***다른 unrelated 프리트레인 제어 목적  
**minor change도 만듬(learning rate, layer normalilze)  
**버트와 비교 1M step으로 book+wiki 학습  


**LM: GPT처럼 L2R 트랜스포머 LM(BART인코더 cross attention 뺀 것과 비슷)  


**퍼뮤티드 LM: XLNet처럼 1/6 token 샘플하여 랜덤 생성, 포지션임베딩 사용x, attention cross x  


**MLM: 버트처럼 15% mask 토큰, original 예측  


**멀티마스크 LM: 1/6 L2R, 1/6 R2L, 1/3 Mask, 1/3*05 unmask, L2R mask  


**Mask Seq2Seq : MASS처럼 50% mask, predict mask  


*퍼뮤티드 LM, MLM, MMLM위해 two stream attention 사용, likelihood 효율적 계산  
*diagonal self-attention L2R 사용  
**실험  
***(1) Seq2Seq 문제로 소스 in, target out  
***(2) decorder에 소스 prefix로 target에 tartget part seq의 loss 추가
***BART에는 (1)이 나음   
***나머지는 (2)가 나음  


# 4.2 Tasks  
*SQuAD  
**QA-wiki, 버트처럼, q&context concat and input to 디코더, 인코더  
**모델이 분류자 갖고 시, 종 예측  
*MNLI  
**문장이 수반되나 체크  
**2문장 train, EOS token, both go to 인코더/디코더  
**버트와 달리 sent 관계 분류에 EOS 사용  
*ELI5 : 추상 QA  
**Q+지원 다큐먼트 concat  
*XSum : news 요약  
*ConvAI2 : 대화 대답 생성, context, persona 따름  
*CNN/DM : 뉴스 요약, source 문장과 밀접  



# 4.3 Results  
결과는 TB1에 있음  
*pre Train 메서드는 task따라 성능 달라져.  
ex) simpleLM ELI5서 베스트이지만 SQuAD에서는 worst  
*토큰 마스킹 매우 중요  
rotate, permute poor  
deletion, masking, self attention masking 성능 good  
deletion outperform mask in 생성task  
*L2R pre Train가 generation 성능 향상  
MLM, PermutedLM generation서 성능 별로(L2R LM없는)  
*SQuAD에서 Bidirectional encoder 기본  
L2R 디코더 poor in SQuAD  
future context 중요한데 반영 못 하기 때문  
BART는 half bidirectional이어서 좋은 유사한 성능 보임  
*pre Train 목적함수 안 중요  
permutedLM이 XLNet 보다 약간 못함  
relatively position 임베딩 또는 seg-level recurrence 없기 때문  
*기본 LM이 ELI5 BEST  
ELI5는 outliar  
PP높아서 다른 모델들이 BART 보다 나음  
losely constrained는 BART 효과 적음  
*BART는 consistently서 강한 성능  
ELI5 빼고 text infilling 사용시 성능 상위권임  


# 5 Large-scale Pre-training Experiments  
요즘 large batch & corpora pre Train으로 down stream 성능 많이 향상  
그래서 관련 실험 함  


# 5.1 Experimental Setup  
*세팅  
12layer encoder, decoder  
hidden size 1024(follow RoBERTa)  
batch size 8000  
train 500,000 steps  
GPT-2 byte pari encoding + tokenize  
text infilling & sentence permute combination  
30% mask  
permute가 CNN/DM 요약서 성능 좋지만 라지가 더 나을 것이라 판단  
dropout 10% 없앰  
Pre Train data Liu꺼 사용  
160Gb news, books, sotries, and web text  


# 5.2 Discriminative Tasks  
BART 성능 비교(테이블 2,3 참고)  
SQuAD, GLUE서 RoBERTa와 비슷  
generation 성능 많이 올려주지만 classification의 성능이 떨어진 것은 아님  


# 5.3 Generation Tasks  
*생성 task  
**BART 파인튠 Seq2Seq smothed(0.1) 크로스엔트로피 사용, beam size-5, remove 중복 grigrams in 빔서치  
**min-len, max-len, length penalty valid로 튜닝  
*요약: CNN/DM, XSum test  
**CNN/DM은 소스문장 재구성 영향, 추출모델 성능 굿, BART가 성능 압도  
**XSum은 추상성이 높아서 추출모델 성능 안 좋지만 BART는 성능 압도하고 ROGUE3.5찍음  
*인간평가  
**BART가 나머지 압도, but 인간엔 못 미침  
*대화 응답생성 ConvAI2로 테스트, BART가 압도  
**context 이전과 textual persona 사용하는 task임  
*추상 QA, ELI5 data, BART가 1.2ROGUE-L로 압도  
**그러나 dataset 부족, 응답이 질문과 특징이 약하게 됨  


# 5.4 Translation  
*번역  
**WMT16 루마니아-영어 번역 data 사용  
**6layer 트랜스포머 BART로 map 디노이즈 to ENGLISH  
**BART & 튜닝된 BART 성능 비교  
beam 5 width, length alpha=1  
BART가 덜 효과적, Back Translation Data 없이 overfit 경향  
**추가 regularization이 future work임  


# 6 Qualitative Analysis  
*BART 요약서 향상, 35point가 sota보다 더 향상  
**이해 위해 generation 질적 분석  
**테이블9가 보여줌, BART 장/단점  
**ex는 wiki뉴스 pre train용. event 요약 지문.  첫 문장은 지운 데이터(요약이라 추정되어서 지움)  
**놀랍게도 Model output fleuent & 문법 good  
**하지만 추상적, 약간의 COpy와 사실기반 + 지지문장(input doesn't support background)  
**즉, BART는 NLU, NLG combination으로 배움  


# 7 Related Work  
oo





