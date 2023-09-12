---
layout: post
title:  "[2022]Biomedical NER for the Enterprise with Distillated BERN2 and the Kazu Framework"
date:   2023-07-31 22:21:24 +0900
categories: study
---






{% highlight ruby %}


짧은 요약(Abstract) :    
*  제약회사들의 bio NER과 internal, public corpora사이의 연결 기술 사용 노력 있음  
* 하지만 불충분  
* 그래서 Kazu 제안    
** Kazu는 오픈소스 프레임워크로 BIO NLP 여러 분야 지원함    
** BERN2 NER 모델을 기반으로함(Tiny BERN2로 좀 더 효율적)  


{% endhighlight %}  

<br/>


[Paper with my notes](https://drive.google.com/drive/folders/1Rt3yFa1WCPVXty2XgHmI1sSzP4LfQLDh?usp=sharing)  


[~~Lecture link~~]()  

<br/>

# 단어정리  
* plethora: 과다, 과잉  
* posit: 제안하다, 가정하다  
* intricacies: 복잡한 부분, 세부 사항
* forthcoming: 곧 나올 것인, 임박한  
* analogous: 유사한   
* marginal: 미미한? 약간?   
* off-the-shelves: 바로 응용 가능한, 사용 가능한  
* throughput: 프로세스당 처리랑(예: 트랜섹션 한번당 data 처치량)  




<br/>

# 1 Introduction  
* search for new drugs -> involves NER, EL(Entity Linking, grounding, normalization)    
* recent work -> overfit  
* important -> balance between error rate and other performance metrics  
* Kazu framework including TinyBERN2 is released as open-source framework  
<br/>

# 2 Challenges of BioNLP in the Pharmaceutical Sector and the Kazu Framework  
* 유지 관리 측면 중요, 어렵(MLOps 흥한것이 그 이유)  
## 2.1 Language/technology agnostic and scalability  
* 확장성(스케일 업)측면 Ray framework 사용  


## 2.2 Flexibility of datasource ingestion  
* 유연성 측면 parsing system 구축  


## 2.3 Robustness of data model  
* biomedical 기술 포함시킴  


## 2.4 Extensibility of pipeline desing  
* 대체 및 추가가 쉽게 파이프라인화(새로운 모델 추가 용이하게)  


## 2.5 Stability in execution  
* for stability -> emmory monitoring and automatic worker restarting  
<br/>   

# 3 Methods  
## 3.1 Model Architecture  
* focus on nested entity handling  
** use BIO tagging  
* input -> BERT -> dense layer(standard BCE train) -> output(BIO tag)  


## 3.2 Weakly supervised learning  
* labeling with BERN2  


## 3.3 Distillation  
* test with F1 score  
* two stages of distillation -> distill transformer Language Model     
** 1. use PubMedBERT as a teacher of BERN2 model  
** 2. use BERN2 as a teacher of TinyBERN2 model  
<br/>

# 4. Experiments and Results  
## 4.1 Benchmark Datasets  
* 8 benchmark datasets of 6 entity classes: Gene/Protein, Disease, Chemical, Species, Cellline, and Cell type   
** object: to examine generalizability(predict unseen entity)  
** datasets from MTL-Bioinformatics-2016 github repo  
** CoNLL-X format  


## 4.2 Results  
### 4.2.1 Evaluation metrics(accuracy)    
* WS-BERN2 is analogous to BERN2  
* TinyBERN2 is lower than WS-BERN2 but marginal     


### 4.2.2 Evaluation metrics(Computational costs)  
* is memory and speed(probably)  
<br/>  

# 5 Discussions  
## 5.1 Effect of traiing by soft-labeling  
* meaningful in fewer ratio labels  


## 5.2 Tagging Schema  
* BIO tagging vs IO tagging(No B tag, B to I)  
** use BPE tokenizing(unknown to byte)   
** IO is economical in speed  
** IO is not recommended in enterprise because this not guarnatee the result  


## 5.3 Enterprise usage of Kazu  
* use in AstraZeneca for biological knowledge graph(BIKG) construction and clinical trial design   
<br/>

# Limitations  
* not tested in large scale condition(like many cpus and gpus)  
** not know how throughput will change depends on scales  



















