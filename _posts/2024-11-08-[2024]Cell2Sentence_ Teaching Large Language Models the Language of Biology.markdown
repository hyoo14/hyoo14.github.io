---
layout: post
title:  "[2024]Cell2Sentence: Teaching Large Language Models the Language of Biology"  
date:   2024-11-08 15:27:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 



짧은 요약(Abstract) :    


이 논문은 "Cell2Sentence(C2S)"라는 새로운 방법을 통해 대형 언어 모델을 생물학적 데이터 분석에 적용하는 연구를 소개하고 있습니다. 이 방법은 유전자 발현 데이터를 "셀 문장"으로 변환하여, 단일 세포 전사체 데이터를 언어 모델이 이해할 수 있도록 합니다. 실험 결과에 따르면, C2S를 사용하여 GPT-2를 미세 조정하면 생물학적으로 유효한 세포를 생성하고, 특정 세포 유형을 예측할 수 있으며, 데이터 기반 텍스트 생성이 가능함을 보여줍니다. C2S는 언어 모델이 단일 세포 생물학을 이해하면서도 텍스트 생성 능력을 유지할 수 있게 해주며, 이는 자연어 처리와 생물학 데이터를 통합하는 유연하고 접근 가능한 프레임워크를 제공합니다.  

This paper introduces "Cell2Sentence (C2S)," a novel method to adapt large language models for biological data analysis, particularly single-cell transcriptomics. By transforming gene expression data into "cell sentences," C2S enables language models to interpret single-cell data. Experimental results show that fine-tuning GPT-2 with C2S allows it to generate biologically valid cells, predict specific cell types, and perform data-driven text generation. C2S provides a flexible and accessible framework that integrates natural language processing with biological data, allowing language models to acquire a significant understanding of single-cell biology while maintaining text generation capabilities  


* Useful sentences :  


{% endhighlight %}  

<br/>

[Paper link]()  
[~~Lecture link~~]()   

<br/>

# 단어정리  
*  







 
<br/>
# Methodology    

**Cell2Sentence(C2S)**는 단일 세포 전사체 데이터(single-cell transcriptomics data)를 대형 언어 모델이 이해할 수 있는 텍스트 형식으로 변환하는 방법입니다. 생물학 데이터를 언어 모델에 적용하려는 목표로, C2S는 유전자 발현 데이터를 "셀 문장(cell sentences)"이라는 텍스트로 표현하여, 언어 모델이 이를 자연어처럼 처리하고 학습할 수 있도록 합니다. 

### C2S의 구체적인 작동 원리
1. **유전자 발현 데이터 변환**: 
   - 단일 세포 전사체 데이터에서 각 유전자(gene)의 발현 수준을 측정하고, 그 순위를 매겨 발현량이 높은 순서대로 정렬합니다. 
   - 정렬된 유전자 목록을 텍스트 형식으로 변환하여 "셀 문장"을 만듭니다. 예를 들어, 발현이 높은 유전자 목록을 텍스트로 나열하여 GPT-2가 이를 마치 문장처럼 이해할 수 있도록 하는 것입니다.

2. **기존 텍스트 데이터와 결합**: 
   - 생물학적 메타데이터(예: 세포 유형, 조직 정보)를 추가하여 각 셀 문장이 더욱 풍부한 정보를 담을 수 있게 만듭니다.
   - 이러한 텍스트 데이터가 언어 모델에 입력되면, 모델은 이를 통해 유전자 발현 패턴과 생물학적 의미를 학습하게 됩니다.

### GPT-2 미세 조정(fine-tuning) 방법
- 이 연구에서는 **GPT-2 모델의 기존 가중치를 업데이트하는 방식**으로 미세 조정을 수행했습니다. 즉, 새로운 레이어나 구조적 변경 없이 **기존 GPT-2의 가중치만 조정**하여 모델이 셀 문장을 처리하고 학습할 수 있게 만들었습니다.
- **LoRA(저차원 적응)나 추가 레이어는 사용되지 않았습니다.** 대신, "셀 문장" 형식의 데이터를 통해 자연어처럼 모델이 학습하도록 하여 생물학적 데이터를 이해하고 예측할 수 있게 했습니다.

### C2S의 결과
C2S 방식으로 미세 조정된 GPT-2 모델은 생물학적 데이터를 기반으로 세포 유형을 예측하거나, 유전자 발현 데이터를 텍스트로 생성하는 작업에서 높은 성능을 보였습니다. 이는 기존의 생물학 데이터 생성 모델과 비교했을 때 더 나은 결과를 보여줬습니다.



**Cell2Sentence (C2S)** is a method that converts single-cell transcriptomics data into a text format interpretable by large language models. Aiming to apply biological data to language models, C2S represents gene expression data as "cell sentences," enabling language models to process and learn from it as if it were natural language.

### How C2S Works
1. **Gene Expression Data Transformation**:
   - From single-cell transcriptomics data, the expression level of each gene is measured and ranked in descending order of expression.
   - The ordered list of genes is then converted into a text format to create a "cell sentence." For example, a list of highly expressed genes is formatted as text, allowing GPT-2 to interpret it as if it were a sentence.

2. **Combining with Existing Text Data**:
   - Biological metadata (e.g., cell type, tissue information) is added to make each cell sentence more information-rich.
   - When this textual data is fed into the language model, it allows the model to learn patterns in gene expression and interpret biological meanings.

### Fine-Tuning GPT-2
In this study, GPT-2 was fine-tuned by updating the model's existing weights without adding new layers or making structural changes. Thus, only GPT-2's original weights were adjusted to enable the model to process and learn from cell sentences. LoRA (Low-Rank Adaptation) or additional layers were not used; instead, the model learned as if processing natural language through the cell sentence format, allowing it to interpret and predict biological data.

### Results of C2S
The GPT-2 model fine-tuned with C2S demonstrated high performance in tasks like predicting cell types and generating text based on gene expression data. This approach yielded better results compared to traditional biological data generation models.

<br/>
# Results  


연구 결과, **Cell2Sentence(C2S)** 방식으로 미세 조정한 GPT-2 모델이 기존에 튜닝되지 않은 GPT-2 모델에 비해 생물학적 작업에서 훨씬 높은 성능을 보였습니다. C2S로 미세 조정한 GPT-2는 특정 세포 유형 예측, 유전자 발현 데이터 기반 텍스트 생성에서 탁월한 정확도를 보였으며, 이를 통해 생물학적 의미와 패턴을 더 잘 학습할 수 있었습니다. 

예를 들어, k-NN (K-최근접 이웃) 정확도와 Gromov-Wasserstein 거리를 기준으로 평가했을 때, C2S를 적용한 GPT-2 모델이 튜닝되지 않은 GPT-2를 포함한 기존 모델들보다 일관되게 높은 성능을 기록했습니다. 특히, 세포 유형 예측 정확도에서 큰 차이가 나타났으며, C2S가 단일 세포 데이터에서 얻을 수 있는 생물학적 통찰을 더 잘 반영하고 있음을 확인할 수 있었습니다.

---

In the study, the **Cell2Sentence (C2S)**-fine-tuned GPT-2 model demonstrated significantly improved performance in biological tasks compared to the untuned GPT-2 model. The C2S-fine-tuned GPT-2 showed superior accuracy in predicting specific cell types and generating text based on gene expression data, better capturing biological meanings and patterns.

For instance, when evaluated using metrics like k-NN (K-Nearest Neighbors) accuracy and Gromov-Wasserstein distance, the C2S-enhanced GPT-2 model consistently outperformed not only the untuned GPT-2 but also other existing models. This difference was particularly notable in cell type prediction accuracy, indicating that C2S effectively enhances the model's capacity to derive biological insights from single-cell data.

<br/>
# 예제  



**예제 셀 문장**:
```
"MT-V1 RPS9 RPS9 RPL8 ..."
```

이러한 셀 문장은 유전자 발현 순위에 따라 생성되며, 각 유전자는 해당 세포에서 발현 수준이 높은 순서대로 나열됩니다. 예를 들어, "MT-V1"은 해당 세포에서 가장 많이 발현된 유전자를 의미하며, "RPS9"와 같은 유전자들이 뒤따라 발현 순위에 따라 나열됩니다. 이 형식은 GPT-2와 같은 언어 모델이 자연어처럼 처리할 수 있도록 돕고, 유전자 발현의 상대적 순위와 관련된 정보를 모델이 학습하게 합니다.

### 의미
이러한 형식은 유전자 발현 패턴을 텍스트로 변환하여, 언어 모델이 이를 학습함으로써 특정 세포 유형을 예측하거나, 유전자 발현 데이터를 기반으로 의미 있는 텍스트를 생성할 수 있도록 합니다. 각 유전자의 위치가 발현 수준의 상대적 중요도를 나타내기 때문에 모델은 유전자 간의 관계와 세포 유형 간의 차이를 자연어 처리 방식으로 이해할 수 있게 됩니다.

---

**Example Cell Sentence**:
```
"MT-V1 RPS9 RPS9 RPL8 ..."
```

These cell sentences are generated based on the rank of gene expression, with each gene listed in descending order of expression within the cell. For example, "MT-V1" represents the most highly expressed gene in that cell, followed by "RPS9" and other genes ranked by expression level. This format allows language models like GPT-2 to process the information similarly to natural language, enabling the model to learn about gene expression patterns through relative ranking.

### Meaning
This format translates gene expression patterns into text, allowing language models to predict specific cell types or generate meaningful text based on gene expression data. Each gene's position indicates its relative significance in expression, helping the model understand inter-gene relationships and cell-type distinctions through a natural language processing approach.


<br/>  
# 요약   




Cell2Sentence(C2S)는 단일 세포 유전자 발현 데이터를 "셀 문장" 형식의 텍스트로 변환하여, 대형 언어 모델이 생물학적 데이터를 자연어처럼 처리할 수 있게 하는 방법입니다. 연구진은 C2S 방식을 통해 GPT-2를 미세 조정하여 세포 유형 예측과 같은 생물학적 작업에서 기존 모델보다 높은 성능을 달성했습니다. 예를 들어, "MT-V1 RPS9 RPS9 RPL8 ..."와 같은 셀 문장을 활용해 GPT-2가 유전자 발현 패턴을 학습하도록 했습니다.

---


Cell2Sentence (C2S) is a method that converts single-cell gene expression data into a "cell sentence" text format, enabling large language models to process biological data similarly to natural language. Using C2S, the researchers fine-tuned GPT-2 to achieve superior performance in biological tasks, such as cell type prediction, compared to existing models. For example, GPT-2 was trained on cell sentences like "MT-V1 RPS9 RPS9 RPL8 ..." to learn gene expression patterns.


<br/>  
# 기타  




<br/>
# refer format:     


@article{Levine2024Cell2Sentence,
  title = {Cell2Sentence: Teaching Large Language Models the Language of Biology},
  author = {Daniel Levine and Sacha Lévy and Syed Asad Rizvi and Nazreen Pallikkavaliyaveetil and Xingyu Chen and David Zhang and Sina Ghadermarzi and Ruiming Wu and Zihe Zheng and Ivan Vrkic and Anna Zhong and Daphne Raskin and Insu Han and Antonio Henrique de Oliveira Fonseca and Josue Ortega Caro and Amin Karbasi and Rahul M. Dhodapkar and David van Dijk},
  year = {2024},
  journal = {bioRxiv},
  doi = {10.1101/2023.09.11.557287},
  url = {https://doi.org/10.1101/2023.09.11.557287},
  note = {preprint}
}  



Levine, Daniel, Sacha Lévy, Syed Asad Rizvi, Nazreen Pallikkavaliyaveetil, Xingyu Chen, David Zhang, Sina Ghadermarzi, Ruiming Wu, Zihe Zheng, Ivan Vrkic, Anna Zhong, Daphne Raskin, Insu Han, Antonio Henrique de Oliveira Fonseca, Josue Ortega Caro, Amin Karbasi, Rahul M. Dhodapkar, and David van Dijk. "Cell2Sentence: Teaching Large Language Models the Language of Biology." bioRxiv (2024)   


