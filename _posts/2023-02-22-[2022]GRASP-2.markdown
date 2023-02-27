---
layout: post
title:  "[2022-2023]Papers related to anormally detection, search, and db"
date:   2023-02-22 20:14:33 +0900
categories: study
---




# Volume Under the Surface: A New Accuracy Evaluation Measure for Time-Series Anomaly Detection  




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


[Paper with my notes]()  


[~~Lecture link~~]()  


# 단어정리  
* cardinality: 유니크한 것 개수



   


