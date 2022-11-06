---
layout: post
title:  "What Makes a Good Podcast Summary?"
date:   2022-11-06 20:00:19 +0900
categories: study
---





{% highlight ruby %}
짧은 요약 :  

팟캐스트 요약의 평가를 위한 연구  

뉴스나 다른 아티클과는 다른 도메인이기 때문에  

다양한 자동평가척도와 인판의 판단 사이의 연관성을 연구  

{% endhighlight %}


[링크](https://drive.google.com/drive/folders/1iYupYlk9vt4UZWW8rY_AKKOZCP6v-LBa?usp=sharing)


# 단어정리  
*o: ㅇ  


# 1 Introduction  
*팟캐스트 요약이 팟캐스트 시청에 영향을 줌  
*자동평가 메트릭과 언어 특성들의 사람평가 사이의 연관성 연구  


# 2 TREC PODCAST SUMMARIZATION TASK  
*TREC 팟캐스트 요약 task data 사용, NIST 제공 사람 평가 사용  


# 2.1 Podcast Corpus  
*스포티파이 팟캐스트 데이터  
**105,306 팟캐스트 에피소드로 구성  
**training data 요약 task용  
**구성  
***자동생성 원고(받아 쓴)  
***오디오  
***RSS헤더  
***제작자의 요약  
***필터버전엔 광고 등을 제거  
***테스트용 1,027 에피소드로 구성  


# 2.2 Summarization Systems  
*요약시스템  
**짧고 정확, 문법적 올바른 요약 생성(팟캐스트 원고를 입력으로 받아서)  
**22개의 모델 제출됨  
**abstractive기술 사용  
***바트 트랜스포머와 함께 뉴스 요약 data로 학습, 파인튜닝->제작자 요약 사용, T5,GAN 사용  
*경쟁모델과 세팅  
**대본은 길이가 긴 다큐먼트  
***BART 입력길이 넘음(1024)  
**일부는 문장 선택해서 사용  
**앙상블 모델 사용  
**BART에 Longformer(4096까지 가능) 사용  
**베이스라인에는 TextRanking과 BART 포함  
**단, 오디오 사용 팀은 없었음  


# 2.3 Manual Evaluation  
*평가는 4가지로  
**Excellent 4(EGFB스코어)  
**Good 2  
**Fair  2  
**Bad 0  
*양자택일 평가 진행  
**Q1: 메인케릭터 이름 포함 여부  
**Q2: 일대기 포함 여부  
**Q3: 주제 포함 여부  
**Q4: 포맷과 스타일  
**Q5: 문맥(제목)  
**Q6: 반복 없는 지  
**Q7: 영어 잘 사용했는지  
**Q8: 시작과 끝 적절한지  


*인간과 시스템 차이  
**EGFB 계산  
**성능 차이 큼  
**BART기본이 성능 좋음(시스템 중에서는)  
**반복 줄인게 성능 좋음  


*제작자가 직접 요약한 것  
**사람들의 평가가 안 좋음  
***제작자 요약 기반 메트릭이 완벽하지 않음을 의미  


# 3 FEATURES AND EVALUATION METRICS  
*자동평가 매트릭 선정  
**어법 특성  
***POS 다른 비율  
**가독성  
***가독, 복잡 통계 사용  
**의미 유사도  
***cosim: w2v  
***TFIDF score  
***extractiveness 점수  
***n-gram 없는거 나올 확률  
***n-gram 반복 확률(대본 안)  
**참고문헌 비교 특성  
***ROGUE, ROGUE-WE, BLEU, METEOR, CIDFr, BertScore, chrF  


# 4 RESULTS  
*자동평가척도로 사람평가 예측 가능?  
**멀티노미널 로지스틱회귀(5-fod cross validation) 사용  
**EGFB 점수 예측  
**ROUGE-L메트릭 사용  
**버트 SCORE, ROUGE-1이 예측 성능 좋음  
**POS 추가도 도움 됨  
**피쳐 전부 사용할 경우 성능 좋음(68.61 F1 vs 59.4(base))  
*분류기의 사람 판정 예측 가능  
**noise가 있어 힘든 점 있음  
**같은 에피소드에 같은 평가자 임에도 11.97%의 경우 다른 평가를 해서 문제가 됨  
*어떤 feature가 좋은 요약과 관련?  
**Kendall's Tau 계산  
**피처와 사람의 판단 사이의 연관도 측정함  
**ROUGE와 바트가 인간의 판단과 연관도 높음  
**적절한 명사 체크도 연관도 높음  
**EGFB와 Q1(주연, 출연 이름 언급)도 연관도 높음  
**동사는 연관도 낮음  
**정답명사와 결정자  
***정보 밀집과 특징자료로 쓰임  
**외부밀도와 NGRAM 반복  
***요약 퀄리티와 연관도 높음  
**요약길이는 인간 평가와 연관도 낮음  
**NGRAM 새로운 것과 가독성은 반복피하기, 좋은 영어 체크와 연관, 퀄리티와는 연관성 낮음  


# 5 RELATIONSHIP TO PREVIOUS WORK  
*요약퀄리티를 특징으로 예측하는 것 연구중  
**CNN 가독성 연구  
***적합 메트릭 발견 못 했음  
***ROUGE와 사람 평가 간의 연관도 있음 발견  
**INPUT과 요약 사이의 분포 유사성  
***ROUGE가 적당한 메트릭  
***의미적 유사성은 사람의 평가의 예측에 연관  
**extractiveness 정확  
***약간 지도학습 요약 사실성 체크  
****사실 측정 어렵, 자동 메트릭 연관성 낮음  
***ROUGE SCORE가 그래도 연관성 있음  
**사실성 예측은 여전히 열린 문제  
**요약 퀄리티 연구는 연구 안 하는 중  
***듣는이의 몰두 예측이 연구가 이어지는 중  


# 6 CONCLUSION  
*팟캐스트 요약 평가  
**적합명사, 결정사, ADVERB, 동사 중요성 찾음  
**인간 평가 예측 어렵(메트릭이 딱 없어서)  
**인간 평가 자체로 NOISE 있음  
**EGFB 늬앙스 다 파악 못함  
**더 FINE-GRAINED 평가 필요  
**ROUGE 완벽하진 않지만 포텐셜 있음  
