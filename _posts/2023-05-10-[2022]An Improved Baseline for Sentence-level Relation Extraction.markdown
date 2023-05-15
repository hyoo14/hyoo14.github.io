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
* .  









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


