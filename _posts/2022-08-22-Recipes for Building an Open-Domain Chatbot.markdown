---
layout: post
title:  "Recipes for Building an Open-Domain Chatbot"
date:   2022-08-22 13:51:19 +0900
categories: study
---




{% highlight ruby %}
짧은 요약 :  

오픈도메인 챗봇 여전히 도전적이  

기존->스케일링(데이터의 파라미터, 서비스)  

여기선->blended skilss 사용  
(-providing engaging talking point, 지식,공간,개인적 적절성 보여주도록 함)  


[https://drive.google.com/drive/folders/1enQBlhrvU9l1Rpyvropi-LVmXXPfncGr?usp=sharing]


{% endhighlight %}


#단어정리  
*takeaways: 시사점, engagingness: 몰입도, utterance: 발언, bland: 부드러운, spicy: 매운, interrogate: 정보를 얻다, startify: 계층화하다,   
gauge: 판단하다,측정하다, whittle: 깎다,줄이다, nowhere near as close: 멀었다, elucidate: 설명하다, snippeet: 토막,정보, conducted: 수행되다,  
albeit: 비록 ~일지라도, further notes: 추가참고사항, annotator: 주석자, avenue: 수단, hallucinate: 환각, seed: 유포, mitigate: 완화, 

# 1. Introduction  
서론  
*오픈 도메인 챗봇 만드는 레시피(방법론) 제공  
*어떤 방법 조합이 좋은 성능 낼지 보여줌  
(NLP & 대화에이전트는 큰 코퍼스(코퍼라)가 중요)  
*여기선  
**파인튜닝 중 개인성, 몰두성, 지식, 공간에 포커스  
***BlendedSkillTask(BST) set-up으로 트레이닝 데이터와 초기 대화컨텍스트에 이러한 특성 제공  
**PP비슷한 두 모델이라도 어떤 decoding 알고리즘 쓰느냐에 따라 성능은 달라짐  
***빔서치 성능 별로였고 샘플링이 더 좋았음->파라미터(빔 길이같은 것) 조정해서 영향 끼침->bland vs spicy 스펙트럼 영향  
*본 모델이 DialogGPT, Meena챗봇 압도함  
**75%, 25% engaging  
**65%, 35% humanness  
***더 나음 본 모델이  
*성능 좋지만 한계는 있음  
**깊은 지식 부족  
**간단한 언어 선호  
**자주 쓰는 말 반복  
*unlikelihood학습과 검색, 정제로 극복시도했으나 성공은 못 함  
**미래에 이 문제 경감 위해 다루어 보겠음  
*릴리즈된 모델이 잘 활용되게 잘 공개함  
*큰규모 SOTA대화 에이전트(코드-파인튜닝, 모델웨이트, 평가코드) 다 제공해줌  


# 2. Related Work  
*관련 연구  
  
