---
layout: post
title:  "[2020-2022]Papers related to NLP topics"
date:   2023-03-09 01:55:23 +0900
categories: study
---



[Paper with my notes](https://drive.google.com/drive/folders/1--eHt7Hc7Y86kIvo4Ps2G0KxRjWNgmAC?usp=sharing)  

<br/>


# [2021]Knowledge-based XAI through CBR  


{% highlight ruby %}
짧은 요약 :  

*KB x AI  
**데이터 중심 에이전트  
***diverse x data  
*도메인지식 중요  
**분류 성능 증대  

{% endhighlight %}


[~~Paper with my notes~~]()  


[~~Lecture link~~]()  


# 단어정리  
* .  


# [2021]Explainable artificial intelligence and social science  Further insights for qualitative
* XAI&사회과학: 질적탐구 인사이트   

{% highlight ruby %}
짧은 요약 :  


*질적분석수반 XAI 연구 리뷰  
a)이론&프레임워크  
b)방법론  
c)데이터 수집법  
d)분석과정(데이터)  
  

{% endhighlight %}


[~~Paper with my notes~~]()  


[~~Lecture link~~]()  


# 단어정리  
* .  



<br/>


# [2021]Explainable Case-Based Reasoning: A Survey  
* 설명 가능한 케이스 기반 reasoning(추론): 설문  



{% highlight ruby %}
짧은 요약 :  


-XAI 연구 증가  
**법적규제와 ML신뢰성 저하가 원인  
**Expert System의 경험기반 검증이 ML에는 적용이 어렵(경험기반 아니니)  
*XCBR(eXplainable Case Based Reasoning) 접근 제안  
**카테고리 기반 taxonomy 설명      
**AI/CBR의 설명 생성 연구자들에 도움  


{% endhighlight %}


[~~Paper with my notes~~]()  


[~~Lecture link~~]()  


# 단어정리  
* opaque: 불투명한, 이해하기 힘든  



<br/>


# [2022]Data-Efficient Autoregressive Document Retrieval for Fact Verification    



{% highlight ruby %}
짧은 요약 :  


-적은 레이블로 잘 찾는 검색 모델 제안(autoregressive fact verification)  
**슈퍼비전 제안  
**F-T 에서 성능 많이 향상, 레이블은 1/4(훨씬 효율적)  


{% endhighlight %}


[~~Paper with my notes~~]()  


[~~Lecture link~~]()  


# 단어정리  
* supervision: 감독, 지도, 관리  

<br/>


# [2022]Are People Located in the Places They Mention in Their Tweets? A Multimodal Approach      



{% highlight ruby %}
짧은 요약 :  

*언급한 위치에 실제로 있는지 체크  
*텍스트와 이미지 멀티모달  
*뉴럴 아키텍처 text & image 이용이 인간 annotator 보다 좋은 성능 보임  
*에러 분석으로 왜/언제 이점 있는지 탐구  


{% endhighlight %}


[~~Paper with my notes~~]()  


[~~Lecture link~~]()  


# 단어정리  
* .  


<br/>



# [2022]Leveraging Affirmative Interpretations from Negation Improves Natural Language Understanding        



{% highlight ruby %}
짧은 요약 :  

*부정문 NLU서 난제  
*부정문 이해가 정확한 이해에 중요  
*자동으로 긍정, 부정문 pair & 올바른 해석 제공(번역)하는 것 제안  
*결과->성능 향상  
(a)T5 해석력 상승  
(b)RoBERTa base 분류기 NLI 성능 향상  
*plugin-and-play 뉴럴 생성기 만듬  
**부정문->잘 해석  
*RoBERTa 감성분석 성능 향상  


{% endhighlight %}


[~~Paper with my notes~~]()  


[~~Lecture link~~]()  


# 단어정리  
* .  



# [2022]Pinpointing Fine-Grained Relationship between Hateful Tweets and Replies        



{% highlight ruby %}
짧은 요약 :  

*detect hate & counter-hate using replies(without annotatoes)  
*reply 종류  
(a)hate  
(b)justification
(c)attacks the author    
(d)additional hate  
*성능 향상  
(a)use replies  
(b)pre-train with related tasks  


{% endhighlight %}


[~~Paper with my notes~~]()  


[~~Lecture link~~]()  


# 단어정리  
* .  


<br/>


# [2021]Spotify at TREC 2020: Genre-Aware Abstractive Podcast Summarization        



{% highlight ruby %}
짧은 요약 :  

*팟캐스트 요약 from 에피소드 대본(듣고 쓴)  
**짧고, 정보 충분 - key info 포함  
***장르와 개체명 사용  
**지도학습(제작자 요약 데이터 사용)  
**베이스라인 1.49대비 9% 향상 1.58점  


{% endhighlight %}


[~~Paper with my notes~~]()  


[~~Lecture link~~]()  


# 단어정리  
* .  


<br/>


# [2020]100,000 Podcasts: A Spoken English Document Corpus        



{% highlight ruby %}
짧은 요약 :  

*팟캐스트는 크고 성장하는 말하는 라디오  
**때문에 더 다양한 스타일(뉴스보다, 비디오 보다도 장르 다양, 대화 코퍼스 보다도)  
*STT로 대본 얻으면 유용/매력적 코퍼스임(NLP, IR&Linguistics)  
*오디오와 pair로 대화 처리, 파라링귀스틱, 시회적 언어학, 음성학 등 매우 유용  
*스포티파이 팟캐스트 데이터셋 소개(100,000개로 구성)  
**task test  
(1)단락 찾기  
(2)요약   
(검색/요약 연구용으로 좋음)  


{% endhighlight %}


[~~Paper with my notes~~]()  


[~~Lecture link~~]()  


# 단어정리  
* .  


<br/>


# [2022]An expert-in-the-loop method for domain-specific document categorization based on small training data      



{% highlight ruby %}
짧은 요약 :  

*도메인 지식이 도메인=특징 문서 분류에 긍정 영향 줌  
*일반론 적용 어려움 있을 경우에 특히 도움  


{% endhighlight %}


[~~Paper with my notes~~]()  


[~~Lecture link~~]()  


# 단어정리  
* .  


<br/>


# [2020]An Empirical Methodology for Detecting and Prioritizing Needs during Crisis Events        



{% highlight ruby %}
짧은 요약 :  

*위기 때 필요한 것 감지  
**소셜 미디어 감지   
**감지 때 어려움 있음  
***sparsity, noise -> 인식 어려움  
*새 기법 사용  
(1)필요 리소스 뽑기(mask, ventilator)  
**0.64 precision  
(2)문장 탐지: 누가 뭘 필요로 하는지(covid19 case)  
**1000 태깅 트위터서 0.68 F1score 성능 보임  



{% endhighlight %}


[~~Paper with my notes~~]()  


[~~Lecture link~~]()  


# 단어정리  
* ventilator:산소 호흡기, 환풍기    


<br/>


# [2021]Detecting Extraneous Content in Podcasts        



{% highlight ruby %}
짧은 요약 :  

*팟캐스트 에피소드들에 외적 요인 포함(광고 등...)  
**탐지(설명과 대본서)  
*이 task 통해 요약 향상, ROUGE점수 향상, 요약서 외부 요인 제거  


{% endhighlight %}


[~~Paper with my notes~~]()  


[~~Lecture link~~]()  


# 단어정리  
* .  


<br/>


# [2021]Incorporating the Measurement of Moral Foundations Theory into Analyzing Stances on Controversial Topics        



{% highlight ruby %}
짧은 요약 :  

*도덕적 기반과 이슈 의견 중 스텐스와 관련성 연관관계  
**Moral Foundation Theory 이용(Dictionary)  
**여러 토픽별 스텐스 분석  
**토픽별 moral & lexical profile 다름  
**스텐스 study 때 morality 고려  


{% endhighlight %}


[~~Paper with my notes~~]()   


[~~Lecture link~~]()  


# 단어정리  
* .  


<br/>
















   


