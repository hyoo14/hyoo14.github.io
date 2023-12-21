---
layout: post
title:  "[2022]Generative Biomedical Entity Linking Knowledge Base-Guided Pre-training and Synonyms-Aware Fine-tuning"
date:   2023-12-21 07:33:11 -0400
categories: study
---

{% highlight ruby %}


짧은 요약(Abstract) :    
* implement generative biomedical entity linking with KB and pre-training/fine-tuning  
* 동의어 잘 다루게 처리함  


Useful sentences :  
* Generative methods achieve remarkable performances in general domain EL with less memory usage while requiring expensive pre-training.  

{% endhighlight %}  

<br/>

[Paper link](https://drive.google.com/drive/folders/13bnmQ3MRGQIM8emOmVTYbsxankEB7-3U?usp=sharing)  
[~~Lecture link~~]()  

<br/>

# 단어정리  

<br/>

# 1 Introduction  
* 현재 한계:  
** There are no such large-scale human-labeled biomedical EL datasets for pre-training.  
** Biomedical concepts may have multiple synonyms.  
* 제안:  
** BioEL위해 KB활용 generative 접근법제안(프리트레인,파인튜닝 포함됨), sota 달성  

# 2 Approach  
* 지식 베이스에서 추출한 동의어를 활용, 파인튜닝+프리트레인  

## 2.1 Seq2Seq EL  
* EL위한 Seq2Seq모델을 구현, context encoding위해 신경망 사용    
* EL을 텍스트 생성 문제로 변환하여 모델이 엔티티 이름 출력하도록함   

## 2.2 KB-Guided Pre-training  
* KB를 이용, 실제 데이터를 모방한 합성 샘플을 생성하여 모델이 다양한 엔티티와 관계를 효과적으로 학습  

## 2.3 Synonyms-Aware Fine-tuning  
* 모델이 특정 개체에 대한 다양한 동의어를 학습(파인튜닝)  

# 3 Experiments
## 3.1 Datasets and KBs  
### Pre-training  
* 개념과 동의어로 input/output 만들어서 프리트레인    



### Fine-tuning  
* BC5CDR, NCBI, COMETA, and AAP로 파인튜닝 및 성능 평가  


## 3.2 Implementation Details  
* BART-large를 백본으로 사용, 티쳐포싱/레이블스무딩으로 프리트레인 및 파인튜닝  




## 3.3 Main Results  
SOTA 달성  

# 4 Discussion   
### Does pre-training help?  
성능향상됐으므로 ㅇㅇ  

### Selection of Names  
tf-idf, shortest name, random sample 사용해봤는데 tf-idf가 가장 좋음  


### Decoder Prompting  
디코더 프롬프팅(모델의 디코더 부분에 특정 입력이나 지시를 제공하여 원하는 출력을 유도하는 방법, 특정 스타일, 형식, 또는 주제의 텍스트를 생성하도록 유도)하니 성능향상


### Sub-population Analysis  
* 여러 벤치마크 실험들 분석한 것.. 성능 본 논문이 좋음  
** 파인튜닝 없이도 더 좋음  
** Unseen concepts! 도 성능 압도(제로샷 퍼포먼스)  
** KB 가이드 사전학습이 대부분의 서브셋에서 성능향상->효과성 입증  
** mention이 긴 것은 어려움...(챌린징 과제)    


# 5 Related Work  
### Biomedical EL  
* 생물의학 EL 연구는 텍스트에서 특정한 생물의학적 개체를 식별   
* 이를 표준화된 데이터베이스나 온톨로지 내의 관련 개체와 연결하는 작업을 포함  

### Generative EL   
* 생성형 모델 사용해서 연결관계를 생성하도록하는 모델  


# 6 Conclusion  
* 본 논문 모델이 짱이다  







# 2 