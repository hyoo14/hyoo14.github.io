---
layout: post
title:  "2021-11-12-Improving Named Entity Recognition by External Context Retrieving and Cooperative Learning"
date:   2021-11-12 16:30:10 +0900
categories: study
---




{% highlight ruby %}
짧은 요약 :

labeling을 위한 sequence 뿐만 아니라  
검색을 통한 external text를 같이 input으로 사용  
CL(Collaborative Learning)으로 성능을 향상  
->external text가 없을 때의 성능도 향상  

{% endhighlight %}

*문장 포함 문서가 있으면 당연히 NER 성능이 향상  
**하지만 문장 찾기가 쉽진 않음  
**트위터 등 소셜미디어나 이커머스 등서 찾기 힘듬  
**그래서 서치엔진으로 찾음  

*검색 결과를 Re-Rank 해줌  
**의미적 유사도로 Re-Rank!  
***BERTScore 씀  

*모델의 흐름?  
** (input sentence + external contexts) -> pretrained embedding -> CRF -> output  

*CL?  
**input 2개로 학습  
**L2 distance 거리 작게 해서 2 인풋에 대한 각각의 아웃풋 차이가 적게끔 해줌  
**KL divergence 를 통해 아웃풋 분포 차이 적게 해줌  
***궁극적으로 external texts가 없을 때의 모델의 성능도 향상  
** 참고로, CL 안 쓸 경우 외부 context 길고 잠재의미 사전에 몰라서 매우 느림.  

*negative log-likelihood 줄이게 학습 진행  
*CRF서 loss function 어려워서 KL씀  
*backpropagate 안 하므로 KL이 cross-entropy와 같음  

*Re-Rankg 때 Roberta-Large 씀  
*바이오는 Bio-BERT, 다른 도메인은 XLM-RoBERTa  

*AdanW 옵티마이저 씀  

**여러 비교군들과 실험해본 결과 8개 중 5개에서 SOTA 찍음  



