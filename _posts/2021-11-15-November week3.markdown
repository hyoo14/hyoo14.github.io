---
layout: post
title:  "November week3"
date:   2021-11-15 14:55:10 +0900
categories: study
---




{% highlight ruby %}

11월 셋째 주:  
*논문 복습 및 보강 with nlp 자료  
*모델링  
*REC-SYS  
*CODE  
*기타  -eng, app  



{% endhighlight %}

2021년 11월 15일 월요일  

*CODE  
**binary search 문제 recap.  
***더 간단한 풀이가 있었음  
****거의 근접했으나 못 푼 것은 나의 실력의 문제ㅠㅠ  
*논문 복습 및 보강 with nlp 자료  
**컨셉노트랑 transformer 자료 보는 중  
*모델링  
**CL이용, seq(we) -> title & seq+POS -> title & seq+NE -> title  
***seq -> title acc will improved.  
*REC-SYS  
**Bayesian Personal Rank  
*기타  -eng, app  
o  


2021년 11월 16일 화요일  
*논문 복습 및 보강 with nlp 자료  
**bert note etc  
*모델링  
**soy nlp 전처리용으로 사용  
***word extractor 학습 및 save
***이걸로 전처리한 코퍼스로 학습 진행 및 저장  
*REC-SYS  
**bpr 및 mf trial  
o  
*CODE  
*기타  -eng, app  





REC  
*이웃기반 (유저-아이템) CF  
**구현 간단 model base CF보다 계산 적고 새유저-아이템에 대해 안정적  
**대신 콜드스타트문제와 데이터 많아질수록 계산량 문제, 롱테일 이코노미문제(소수 아이템 쏠림 현상)의 단점이 있음  

*자카드 유사도  
**교집합/합집합 ->얼마나 많은 아이템이 겹치는지 판단  
*피어슨 유사도  
**코사인 유사도와 유사. 정규화가 추가됨. 1은 양의상관관계, -1은 음의상관관계, 0은 상관관계 없음  











