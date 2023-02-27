---
layout: post
title:  "[2022-2023]Papers related to anormally detection, search, and db"
date:   2023-02-22 20:14:33 +0900
categories: study
---




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




# [2022]


{% highlight ruby %}
짧은 요약 :  






{% endhighlight %}


[~Paper with my notes~]()  


[~~Lecture link~~]()  


# 단어정리  
* .




# [2022]


{% highlight ruby %}
짧은 요약 :  






{% endhighlight %}


[~Paper with my notes~]()  


[~~Lecture link~~]()  


# 단어정리  
* .






   


