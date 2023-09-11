---
layout: post
title:  "[2021]Can Language Models be Biomedical Knowledge Bases?"
date:   2023-07-31 22:17:24 +0900
categories: study
---






{% highlight ruby %}


짧은 요약(Abstract) :    
* PLM으로 NLP다양 task 처리 중  
** 일종의 KB로 다뤄지는 PLM으로부터 어떤 정보 포함여부와 정보추출하는 것에대한 관심 증대 
* 도메인 특화적인 것은 다소 부족한데, BIO LAMA 라는 BIO 특화  PLM 제안  
** 18.51% ACC@5를 달성하여 희망적  


{% endhighlight %}  

<br/>


[Paper with my notes](https://drive.google.com/drive/folders/1F7NGnhzQX6Y3MxBsXf-k_EyADSBtTphZ?usp=sharing)  


[~~Lecture link~~]()  

<br/>

# 단어정리  
* genomics: 유전체학  
* metathesaurus: 메타시소러스는 UMLS의 기본 형태를 구성하고 백만개 이상의 생의학 개념들과 5백만개의 명칭으로 구성  







<br/>

# 1 Introduction  
* LM 흥함 -> KB로 활용 가능성 모색(probing, 검증느낌)  
* BioLAMA 만들어서 효용성 실험 test  
** Comparative Toxicongenomics Database(CTD), Unified Medical Language System(UMLS), Wiki data  
** 성능 굿  
* LM과 실허(검증) 방법 효용성 확인도 함  
<br/>  

# 2 BIO LAMA   
## 2.1 Knowledge Sources  
* 지식 기반 소개  
### CTD  
* public biomedical DB(rel and interactions between biomedical entities(disease, chemicals, genes)  


### UMLS  
* 메타시소러스는 대용량(라지스케일) DB  


### Wikidata  
* 위키피디아  


## 2.2 Data Pre0processing  
* 전처리 과정  


### Evaluation Metric  
* 평가척도: top-k acc, ACC@1, ACC@5    
<br/>

# 3 Experiment  
## 3.1 Models  
### Information Extraction  
* IE task->relative entity/articles (given query)   
* BEST(Biomedical Entity Search Tool) 함께 쓰임      
* PubMed 인덱싱해서 사용   
* LM 으로 BioBERT 사용(BioLM도 사용)  
** BERT랑 같은 vocab  
* BioLM   
** 둘 다 PubMed PreTrain이지만 BioLM은 custom vocab씀  


## 3.2 Probing Methods  
* 실험 입증 도구  
### Prompts  
* 프롬프팅 - 빈칸채우기  
* 도구 2가지  
** 1. 매뉴얼(수동)  
** 2. OptiPrompt   


### Multi-token Object Decoding  
* 멀티토큰 object 디코딩  
** Jiang 따름(confidence 기반), greedy하게 디코딩  


## 3.3 Main Results  
* 결과  
** BioBERT, BioLM both better than BERT  
** IE system is worse than both BioBERT, BioLM in IE  
<br/>

# 4 LMs are Not Biomedical KBs, Yet  
* LM 아직 Bio KB는 아님  
## 4.1 Predictions  
* 예측에서 높은 정확도 달성    
## 4.2 How Biomedical LMs Predict  
### Prompt Bias  
* KB로 보려면 올바른 object entity prediction이 되야함  


### Synonym Variance  
* 동의어 변화폭에서 성능 좋음  


### Results  
* 더 좋기는 함  
<br/>  

# 5 Conclusion  
* 특정 프롬프트에 의존적임이 드러남  




