---
layout: post
title:  "September Papers"
date:   2023-09-25 17:10:11 -0400
categories: study
---


# []

{% highlight ruby %}


짧은 요약(Abstract) :    
* 

{% endhighlight %}  

<br/>


[~~Lecture link~~]()  

<br/>

# 단어정리  


<br/>




# [Papers with my notes]()  


# [2020]Keeping up with the genomics: efficient learning of our increasing knowledge of the tree of life  

{% highlight ruby %}


짧은 요약(Abstract) :    
* 제목? 유전자학과 계속 연락 지속해라? 증가하는 계통분류학 지식을 반영하는 효과적인 학습법  
* 배경(요약): NCBI RefSeq 박테리아 유전체 DB가 기하급수적으로 증가하지만 처음부터 다시 학습하는 건 비효율적임, 업데이트 방안 필요(acc 안 떨어뜨리고 전체 재학습보다 나은)  
* 결과: 유전체 정보 늘어날 수록 정확도 상승하는 것 확인, 나이브베이즈 구현으로 매년 업데이트 때 이전보다 4배 빨라짐 확인  
* 결론: 최신기술 사용해서 성능 향상되었음, 데이터가 많아지는 상황에서 점진 학습 분류기는 매우 효율적으로 업데이트됨(reprocess cost 없고 기존 DB 접곤 읎으므로 공간 save하고 컴퓨팅 리소스 아낌)  

{% endhighlight %}  

<br/>

# 키워드: tree of likfe(계통 유전학), genomics(유전자학), naive Bayes, incremental learning(called online learning, 1steady, 2storage efficeincy, 3dynamic environment, 데이터 순차적, 작은 배치로 학습)  


[~~Lecture link~~]()  

<br/>

# 단어정리  
* keep up with: 알게 되다, ~와 계속 연락하고 지내다, 유형을 따르다, 뒤지지 않다  
* rich literature: 핵심, 영감을 주는  
* at its disposal: 사용할 수 있는  
* disposal: 처리  
* deluge: 대홍수, 매우 많아지는 상황을 비유할 때 사용  
* tractable: 추적 가능한  
* National Center for Biotechnology Information(NCBI): 미국 국립 생물 공학 정보 센터  
* RefSeq: Reference Sequence  


<br/>


# [2022]How Scalable Are Clade-Specific Marker K-Mer Based Hash Methods for Metagenomic Taxonomic Classification?  

{% highlight ruby %}


짧은 요약(Abstract) :    
* bio sample서 미생물 인식 중요(의학, 생물학 모두-> 더 나은 진단에 기여)  
* 메타유전체적 2가지 본질적 질문이 있음  
** 샘플의 미생물 인식  
** taxonomi classifier를 효율적으로 update하는 것(새 sequence가 들어올 때)  
* 지식 더 학습시 변화를 보기 위해 sub db 만들었음(snapshot, NCBI에 대한 sub 디비, 실제 사람 장기 data로 이루어짐)  
** DB data 증가함에 따른 성능 측정  
** Btray-Curtis 거리로 시간이 지남에 따라 데이터 쌓여감에 따른 향상 봄  
*** Kraken 2 data에 대해서는 유전체 증가 시 더 많은 미생물 잘 분류  

*** CLARK도 비슷했지만 마지막해(마지막 data 묶음, 2022년 데이터)에 분류가 더 안됨을 확인  

{% endhighlight %}  

<br/>


[~~Lecture link~~]()  

<br/>

# 단어정리  
* microbes: 미생물   
* concordance: 일치  

<br/>



# [2022]An Interpretable Deep Learning Model for Predicting the Risk of Severe COVID-19 from Spike Protein Sequence  

{% highlight ruby %}


짧은 요약(Abstract) :    
* gene 변화에 따른 risk prediction model 만듬  
* with SARS-CoV-2 data  
** omicron이 delta보다 40-50% risk 적음 예측  
** 미래에 사용 용이  
** model: tokenizer embedding -> CNN -> Transformer -> attention -> FCN -> FCN  


{% endhighlight %}  

<br/>


[~~Lecture link~~]()  

<br/>

# 단어정리  


<br/>



# [2020]Visualizing and Annotating Protein Sequence using A Deep Neural Network  

{% highlight ruby %}


짧은 요약(Abstract) :    
* 딥러닝 + 어텐션 기반 단백질 마커(tagger/annotator) + visualizer  


{% endhighlight %}  

<br/>


[~~Lecture link~~]()  

<br/>

# 단어정리  


<br/>


# [2020]Learning, visualizing and exploring 16 sRNA structure using an attention-base deep neural network  

{% highlight ruby %}


짧은 요약(Abstract) :    
* 미생물 DNA seq 데이터를 이용하여 RNN + attention 기반의 taxonomy 분류 구현    
** microbiome의 read(핵산 조각)로부터 phenotype(표현형) 얻는 framework 제안  
** attention -> visualization 되고 inflammatory bowel disease(IBD) 비교적 잘 캐치?  
** 전처리, 후처리 없고 복잡/에러 없어서 이득   

{% endhighlight %}  

<br/>


[~~Lecture link~~]()  

<br/>

# 단어정리  
* amplicon: PCR로 얻은 DNA 증폭체?  
* pheonotype: 표현형  
* nucleotide: 핵산 구성 단위 분자  
* microbiomes: 미생물들 총합(특정 환경에 서식하는)  
* inflammatory bowel disease(IBD): 염증성 장 질환  
* throughput: 처리량(예를 들어, 시간당 tx 수)    
* DNA reads: 차세대 염기서열 분석법을 통해 분석한 하나의 핵산 조각   


<br/>

