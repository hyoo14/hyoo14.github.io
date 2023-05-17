---
layout: post
title:  "[2022]An Improved Baseline for Sentence-level Relation Extraction"
date:   2023-05-10 21:29:33 +0900
categories: study
---






{% highlight ruby %}


짧은 요약(Abstract) :    
* RE에 typed marker 도입  
** SOTA 성능  




{% endhighlight %}  

<br/>


[Paper with my notes](https://drive.google.com/drive/folders/1VGH3pSvVShKOByUX4mGmI_IemB-1EpXq?usp=sharing)  


[~~Lecture link~~]()  

<br/>

# 단어정리  
* preliminary: 예비의  









<br/>

# 1 Introduction  
* RE는 IE(Information Extraction)의 주요 task  
** 두 entity의 관계 표시  
** 최근 추가지식 + PLM (ERNIE, K-BERT)  
** LUKE: MLM + entity인지  
** PLM link entity(BERT-MTB)  
** 완벽 x  
* 두가지 문제가 있음  
** raw text + side info(entity)  
** human RE noise & wrong labels  
* 향상된 RE 베이스라인 제시  
** typed entity marker(sent level RE)  
*** 높은 서능 보임  
** TACRED, TACREV, RE-TACRED로 평가  
** RoBERTa 사용(백본 모델로)  
** F1 74.6% TACRED, 83.2 TACREV 달성 -> SOTA  
</br>

# 2 Method  
* 문장 내 entity간 RE  
** sent x1, pair(es, eo), pari entity 사이 관계 r, R: 사전정의 relation set, rel 없을 시 NA로 표기  


## 2.2 Model Architecture  
* 이전 PLM 기반  
** 입력 sent x  
** entity type & span 주어짐  
** PLM 거침(sent가)  
*** context embedding  
*** 위 hidden layer사용  
** softmax classifier로 확률 얻음&학습  


## 2.3 Entity Representation  
* 엔티티 포현  
** 문장 내에 엔티티 정보 표현  
### Entity mask    
* 엔티티 마스크, 새 special token 제안  
** PA-LSTM서 영감, SpanBERT서 사용  
** overfit 방지  
### Entity marker  
* 엔티티 마커  
** [EI]같은 토큰  
### Entity marker (punct)  
* 엔티티마커(구두점:@, #)  
** 모델의 vocab 사용하면 됨  
** 새롭게 소개 필요 없음  
### Typed entity marker  
* 타입 엔티티 마커  
** <S:TYPE>, <O:TYPE> 과 같음  
### Typed entity marker (punct)  
* 타입 엔티티마커 구두점  
** 구두점과 type 함께 사용  
** 본 논문이 제안하는 것  


** 새 토큰 임베딩은 랜덤 초기화 후 파인튜닝 때 update  
</br>

# 3 Experiments  
* RE 벤치마크로 실험  


## 3.1 Preliminaries  
* 예비작업들  


### Datasets.  
* 데이터셋  
** TACRED, TextAnalysisConferenceRelationExtraction Dataset  
** original TACRED, Re-TACRED, TACREV  
** TACRED noise 6.62%  
** 관련 통계 Appx A 참고  


### Copared methods.  
* 비교 방법론들  
** PA-LSTM(BiLSTM + POS Attention + SOFTMAX)  
** C-GCN(graph-base model, pruned depency tree of sent, graph CNN)  
** SpanBERT(PLM, span 예측 추가)  
** KnowBERT(LM과 entity linker 함꼐 학습, subtoken이 entity임베딩에 가도록 허락(KB서 학습된)  
** LUKE(LM을 큰 text 코포라와 Knowledge graph로 학습, Frequent entity추가, entity aware-attention 제안)  
* 모델 configure  
** re-run official 추천 파라미터로  
** 허깅페이스 기반 + Adam, 5e-5 learning rate in BERT base, 3e-5 in BERT large & RoBERTa Large, linear warm-up 10% first, decay 0, batch 64, fine tuning epoch=5, f1 5 training average  


## 3.2 Analysis on Entity Representation  
* entity 표현 분석  
** base & large BERT, large RoBERTA as encoder  
** PLM별 성는ㅇ 비교  
** 통찰  
*** typed 엔티티 마커가 여러 변형 버전들 능가  
*** RoBERTa F1 74.6(구두점 typed entity 마커)로 SOTA  
*** 모든 카테고리 정보에서 도움 됨을 확인(RE서)  
*** 특수 구두점 효과 좋음(새 마크 학습은 별로)  


## 3.3 Comparison with Prior Methods  
* 이전 메서드들 비교  
** unseen 일반화 성능 & 어노테이션 에러시 성능 Appx B, C 참고  
</br>

# 4 Conclusion  
* 결론  
** 간단, 강력한 엔티티 펑츄에이션 타입 마커로 RE SOTA  
** entity representation & noisey label 문제 접근  
** 기록 압도 & SOTA 달성  


