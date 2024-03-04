---
layout: post
title:  "[2022]Medical Coding with Biomedical Transformer Ensembles and Zero/Few-shot Learning"  
date:   2024-03-03 18:55:11 -0400
categories: study
---

{% highlight ruby %}


짧은 요약(Abstract) :    

* 의료 코딩은 신뢰할 수 있는 데이터 검색 및 보고를 위한 필수 전제 조건임
* 주어진 자유 텍스트 보고 용어(예: "오른쪽 허벅지의 무릎까지의 통증")에 대해, 매우 크고 지속적으로 성장하는 표준화된 의료 용어 저장소에서 해당하는 가장 낮은 수준의 용어(LLT)를 식별하는 작업임 –이 경우에는 "일측성 다리 통증"
* 그러나 이 작업을 자동화하는 것은 LLT 코드의 대규모 수(작성 시 80,000개 이상), 긴 꼬리/신생 클래스에 대한 훈련 데이터의 제한된 가용성, 그리고 의료 분야의 일반적으로 높은 정확도 요구 사항 때문에 도전적임
* 이 논문에서는 MC 작업을 소개하고, 그 과제를 논의하며, 전통적인 BERT 기반 분류와 최근의 제로/퓨샷 학습 접근 방식(TARS)을 결합한 새로운 접근 방식인 XTARS를 제시함
* XTARS를 사용한 광범위한 실험을 통해, 특히 퓨샷 체제에서 강력한 기준선을 능가하는 것을 보여줌


* Medical coding is a prerequisite for reliable data retrieval and reporting in the medical field
* Identifying the lowest level term (LLT) in a vast and growing standardized medical terminology repository for a given free text report term  
** For example "pain from knee to right thigh" to "unilateral leg pain"  
* Automating this task is challenging due to the large number of LLT codes (over 80,000 at the time of writing), limited availability of training data for long-tail/emerging classes, and the typically high accuracy requirements in the medical field
This paper introduces the MC task, discusses its challenges, and presents a new approach, XTARS, combining traditional BERT-based classification with recent zero/few-shot learning methods (TARS)
* Extensive experiments with XTARS, especially in a few-shot regime, demonstrate its superiority over strong baselines



Useful sentences :  


{% endhighlight %}  

<br/>

[Paper link](https://drive.google.com/drive/folders/1870mgj6jrTdl2H-UWHWk6CmV8tVj6EiA?usp=sharing)  
[~~Lecture link~~]()  

<br/>

# 단어정리  
* 
<br/>

# 1 Introduction  
