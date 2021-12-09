---
layout: post
title:  "Buy It Again: Modeling Repeat Purchase Recommendations"
date:   2021-12-09 14:20:10 +0900
categories: study
---




{% highlight ruby %}
짧은 요약 :

구매 내역(history)을 보고 상품추천  
특히 반복된 구매에 대한 시기적절한 추천  

{% endhighlight %}


*모델 = 구매 회수 기반 랭킹 + 시간에 따른 반감기 + 반복구매 주기적 현상 반영  
**이전에만 많이 구매한 경우를 반영  
**반복구매 현상 반영  

*최종 모델 도출 과정까지의 모델들 정의  
**RCP(Repeat Customer Probability) MODEL  
***물건 구매한 사람들 중 반복구매(한번 초과)한 사람 비율  

**ATD(Aggregate Time Distribution) MODEL  
***반복 구매 시간 분포 이용(log-normal 분포 사용)  

**PG(Poisson-Gamma) MODEL  
***고객의 과거 구매 이력을 사전 분포 합산하여 고객의 반복구매비율을 베이시안 추정-포아송과정  
***포아송 분포 사용, 포아송분포의 매개변수 람다는 사전감마분포 따름  
***하나의 고객이 반복구매할 확률(불특정 하나의 고객임)  

**MPG(Modified Poisson-Gamma) MODEL  
***PG에서 반영하지 못 하는 시간 변수를 반영하기 위한 모델-예를 들어, 구매 직후 구매욕구 떨어지는 것을 반영  
***modified된 포아송 과정을 사용-고객이 마지막에 반복구매한 시점에 의존적인 람다 사용  

*평가  
**분석적 평가-시나리오 사용  
***페이퍼타울 2달마다, 세탁세정제 3달마다 구매 시나리오  
****RCP는 항상 페이퍼타울 우위, ATD는 세탁세정제 항상 우위  
****PG의 경우 페이퍼타울이 항상 우위(고객이 1주전에 산 것 반영 못 함), MPG의 시기에 따라 세탁세정제가 우위. (고객이 1주전에 산 것 반영)  

**오프라인 실험-아마존 고객 구매내역 사용  
***metric은 precision(TP/(TP+FP)), recall(TP/(TP+FN)), NDCG   
***nDCG는 normalized Discounted Cumulative Gain  
****CG는 랭크 상위 p개가 연관성 있는지의 합. 모두 연관성 있다면 p  
****DCG는 랭킹 순서에 따라 가중치 준 것. 랭킹이 i만큼 멀어질 수록 log2(i+1)배 됨(discount됨)  
****nDCG는 p길이가 길어지면 성능이 좋게 보이는 DCG를 보완한 것으로 IdealDCG-연관성 큰걸로 배열한 이상적 경우로 DCG값을 나눠줌(nomalize한 것)  
***MPG가 가장 좋은 성능 보임  

**온라인 실험-A/B테스트(실험/대조군)  
***metric은 CTR(클릭 비율)  
****ATD는 7.1% lift CTR증진  
****MPG는 여기에 1.3% 추가적으로 CTR 증진  





