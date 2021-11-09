---
layout: post
title:  "November week2"
date:   2021-11-08 13:15:10 +0900
categories: study
---




{% highlight ruby %}

11월 둘째 주:  
*논문 복습 및 보강 with nlp 자료  
*모델링  
*REC-SYS  
*CODE  
*기타  -eng, app  



{% endhighlight %}

2021년 11월 08일 월요일  
*논문  
**Attention Is All You Need 복습  
*CODE  
**h-r array01  
*기타  
**app  
*모델링  
**CLNER 발굴?  
*REC-SYS  
*이웃기반 CF  
o  

2021년 11월 09일 화요일  
*논문  
**CLNER 읽기 시작  
*모델링  
**kifrs 자료 & kgaap 자료 수집  
***http://www.kasb.or.kr/fe/accstd/NR_list.do;jsessionid=C4765A34282A020F46705B8B70A2C789?sortCd=K-IFRS&divCd=01 여기가 삼일 보다 다운받기 쉬움  
**fasttext 돌려봐야지.. gensim?  
***pdf file일단 text로 refine  
***돌려봤는데 뭔가 vocab에 문제가 있음.. unseen 처리 못 하는 것이야 생각했던 거지만 뭔가 token 이상하게 짤린 듯? 내일 체크해봐야..
*CODE  
**h-r array02  
*기타  -eng, app  
o  



*REC-SYS  






REC  
*이웃기반 (유저-아이템) CF  
**구현 간단 model base CF보다 계산 적고 새유저-아이템에 대해 안정적  
**대신 콜드스타트문제와 데이터 많아질수록 계산량 문제, 롱테일 이코노미문제(소수 아이템 쏠림 현상)의 단점이 있음  

*자카드 유사도  
**교집합/합집합 ->얼마나 많은 아이템이 겹치는지 판단  
*피어슨 유사도  
**코사인 유사도와 유사. 정규화가 추가됨. 1은 양의상관관계, -1은 음의상관관계, 0은 상관관계 없음  











