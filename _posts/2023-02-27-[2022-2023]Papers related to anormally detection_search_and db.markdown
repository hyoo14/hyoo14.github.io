---
layout: post
title:  "[2022-2023]Papers related to anormally detection, search, and db"
date:   2023-02-22 20:14:33 +0900
categories: study
---

[Paper with my notes](https://drive.google.com/drive/folders/1NM6rlsoX3YLvY_fi0bihqvIGHyEsOZCg?usp=sharing)  
<br/>


# [2022]Volume Under the Surface: A New Accuracy Evaluation Measure for Time-Series Anomaly Detection  




{% highlight ruby %}
짧은 요약 :  


-이상탐지는 시계열 뷴석서 핵심 다운스트림 테스크
-일반적 이상탐지인 포인트기반과 달리 시계열 이상탐지에서는 범위 기반이어야 함  
-하지만 여전히 포인트기반 전통 척도인 precision, recall, f1score가 쓰임  
-이 경우 이산레이블을 연속 데이터에 매핑해야하기 때문에 치명적 약점 야기  
-측정법 달리할 때마다 vias문제 심각  
-60년의 역사동안 또한 대용량 양적/질적 척도 부재함  
-본 논문에서 새 척도 제시  
-잡음, 잘못된 배치, 유니크의 개수가 다른 경우 모두를 강건하게 평가하는 척도로 VUS 제안
-Volume Under the Surface는 퀄리티
밸류에대한 독립적 측정이어서 시계열 이상팀지에 적합함  
-테스트 통해 더 나음이 입증  


{% endhighlight %}


[~Paper with my notes~]()  


[~~Lecture link~~]()  


# 단어정리  
* cardinality: 유니크한 것 개수   






# [2023]AMIR: Active Multimodal Interaction Recognition from Video and Network Traffic in Connected Environments  



{% highlight ruby %}
짧은 요약 :  

* Activity recognition(for elder care etc..)  
** brittle(인터넷 환경 등의 영향으로)  
* 보완책 제시  
** video + network traffic trace 로 학습 및 탐지  
** 17.76% 정확도 향상  




{% endhighlight %}


[~Paper with my notes~]()  


[~~Lecture link~~]()  


# 단어정리  
* brittle: 잘 부러지는, 불안정한, 부서쥐기 쉬운, 취약한  




# [2022]TSB-UAD: An End-to-End Benchmark Suite for Univariate Time-Series Anomaly Detection  


{% highlight ruby %}
짧은 요약 :  


* 이상탐지 인기지만 완전한 벤치마크가 없음  
** 이유:  
(i) 독점 또는 합성 데이터 이용  
(ii) 공개 데이터 부족  
** 어떤 데이터서 성능이 좋은 모델이라도 다른 데이터에서는 안 좋은 경우도 많음  
* TSB-UAP로 그간의 연구 벤치마크들, 데이터 종합시킴  
** 일변량(변수 1개?) 시계열 이상탐지 평가용  
** 13766 시계열데이터 레이블링됨(type/ratio/size)  
** 18개의 이전 데이터셋 포함(1980 시계열, 989 생성, 126 시계열 분류 데이터셋 변환)  
** 12개의 메서드로 데이터셋 평가 & 공개  




{% endhighlight %}


[~Paper with my notes~]()  


[~~Lecture link~~]()  


# 단어정리  
* proprietary: 독점  
* univariate: 일변량  




# [2022]Teseus: Navigating the Labyrinth of Time-Series Anomaly Detection  


{% highlight ruby %}
짧은 요약 :  

* 이상탐지 주목받지만 확살힌 벤치마크 없음  
** 일부 모델들은 어떤 data서 성능 좋지만 어떤 data에선 안 좋음  
* TSB-UAD에 벤치마크 정리/종합  
* Theseus 정의  
** 모듈화된 확장된 웹어플리케이션  
** 벤치마크 탐색, 이점/약점, 정확도 측정, 조건별 정리됨  
**12개 시계열 이상탐지 비교, 13개 정확도 척도 사용, 가장 적합한 거 사용케 해줌  





{% endhighlight %}


[~Paper with my notes~]()  


[~~Lecture link~~]()  


# 단어정리  
* labyrinth: 미로, 미궁, 혼란, 착잡  
* verdict: 결정, 판단  




# [2022]Fast Adaptive Similarity Search through Variance-Aware Quantization  


{% highlight ruby %}
짧은 요약 :  

* 연구동기: 고차원데이터 증가와 여기에 대한 양자화기법 이용(양자화: 연속데이터->디지털(이진)데이터)  
** 빠른 쿼리 반응  
** 인코딩 연산 & 스토리지 코스트 적음  
** 기존 양자화 컨셉: data 차원 안겹치는 subspace로 분해, subspace 마다 dictionary 확장(사이즈 동일하게, 그리고 밸런싱)    
** 하지만 이러한 기존 방법은 벨런싱이 항상 가능하지는 않다는 단점이 있고 때문에 성능도 떨어질 수 있음  
* Variance-Aware Quantization 제안  
** 내부 차원 감소 사용(subspace 다룸)  
** 부분적으로만 밸런싱  
** dictionary size 중요도에 따라 비례적으로 할당하여 최적화 문제 해결  
* 평가  
** 양자화, 해싱, 인덱싱 기법을 5개의 거대한 벤치마크로 평가  
* 결과  
** 제안 모델이 hash와 양자와에 비해 성능서 압도했고 속도는 5배 빨랐음  
** 하드웨어 가속 기법 보다 성능은 조금 떨어졌지만 속도는 14배 빠름  
** 비용은 반으로 줄어듦  
** VAQ 간단데이터 스킵솔루션은 인덱싱에 비해 경쟁력과 나은 성능을 보임  
** 새로운 인덱싱 기법으로의 전환 기대  



{% endhighlight %}


[~Paper with my notes~]()  


[~~Lecture link~~]()  


# 단어정리  
* quantization: 양자화  
* proportionally: 비례적으로  
* oblivious: 잊기 쉬운  
* evaluation against: ~에 대한 평가   




# [2022]VergeDB: A Database for IoT Analytics on Edge Devices


{% highlight ruby %}
짧은 요약 :  


* IoT 번영-대용량의 시계열 수집, 저장, 분석 시스템 필요  
* VergeDB 제안  
** 유연, task기반 압축, 복잡 분석 task, ml가능  
** 가벼운 스토리지 엔진  
*** 연산리소스, 수용량, 네트워크 최적화 통해 처리량, 압축, task처리 정확도 극대화  



{% endhighlight %}


[~Paper with my notes~]()  


[~~Lecture link~~]()  


# 단어정리  
* verge: 가장자리, 분산    
* throughput: 처리량  


















   


