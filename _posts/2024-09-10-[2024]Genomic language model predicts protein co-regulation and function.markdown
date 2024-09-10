---
layout: post
title:  "[2024]Genomic language model predicts protein co-regulation and function"  
date:   2024-09-10 15:53:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

짧은 요약(Abstract) :    


이 논문은 유전자와 그 유전체 맥락 간의 관계를 이해하기 위해 개발된 유전체 언어 모델(gLM)에 대해 설명합니다.  
이 모델은 수백만 개의 메타게놈 서열을 학습하여 잠재적인 기능적 및 규제적 관계를 포착합니다.  
gLM은 유전자의 기능적 의미와 규제 구문을 성공적으로 인코딩하며, 주의 패턴 분석을 통해 공동 규제 기능 모듈을 학습합니다.  
연구 결과는 gLM이 유전체 영역 내에서 복잡한 유전자 관계를 발견하는 데 유망한 접근법임을 시사합니다.  



This paper presents the development of a genomic language model (gLM) designed to understand the relationships between genes and their genomic contexts.
The model is trained on millions of metagenomic sequences to capture latent functional and regulatory relationships.  
It successfully encodes functional semantics and regulatory syntax of genes, learning co-regulated functional modules through attention pattern analysis.   
The findings suggest that gLM is a promising approach for discovering complex gene relationships within genomic regions.  

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


아키텍처: gLM은 Hugging Face의 RoBERTa 트랜스포머 아키텍처를 기반으로 합니다. 이 모델은 19개의 레이어로 구성되어 있으며, 각 레이어는 1280의 히든 사이즈와 10개의 어텐션 헤드를 가지고 있습니다. 또한 상대적 위치 임베딩을 사용합니다.  

훈련 방법: 훈련 중에 각 서열의 15%의 유전자가 무작위로 마스킹되며, 모델은 마스킹된 유전자를 예측하는 작업을 수행합니다. 마스킹된 유전자에 대한 라벨은 PCA를 사용해 차원을 축소한 100개의 피처 벡터로 구성되며, 해당 유전자의 단백질 임베딩과 방향성을 포함합니다. 훈련은 4개의 NVIDIA A100 GPU에서 분산 병렬화를 통해 1,296,960 단계(560 에포크) 동안 수행되었습니다. AdamW 옵티마이저가 사용되었으며, 학습률은 1e-4로 설정되었습니다. 총 손실은 MSE와 CrossEntropyLoss를 포함하여 계산되었습니다.    

특수 기법: 훈련 안정성을 높이기 위해 PCA 차원 축소 기법을 사용했습니다. 또한, 어텐션 패턴 분석을 통해 gLM이 오페론과 같은 공동 규제 기능 모듈을 학습하는 것을 확인했습니다.(마스킹된 서열 예측하는 테스크를 분석해보니 오페론 규제 기능을 인식한 것으로 보인다..정도)    



Architecture: The gLM is based on the Hugging Face RoBERTa transformer architecture. The model consists of 19 layers, each with a hidden size of 1280 and 10 attention heads. It also uses relative position embeddings.  

Training Method: During training, 15% of the genes in each sequence were randomly masked, and the model was tasked with predicting the masked genes. The labels for the masked genes were composed of 100 feature vectors, reduced in dimensionality using PCA, and included the protein embedding and orientation of the corresponding gene. The training was performed over 1,296,960 steps (560 epochs) on four NVIDIA A100 GPUs using distributed data parallelization. The AdamW optimizer was used with a learning rate of 1e-4. The total loss was computed by combining MSE and CrossEntropyLoss.  

Special Techniques: PCA dimensionality reduction was used to improve training stability. Additionally, attention pattern analysis revealed that gLM learned co-regulated functional modules, such as operons. (The task of predicting masked sequences appears to have led the model to recognize regulatory functions like operons.)  


<br/>
# Results  
결과 부분에서 gLM은 효소 기능 예측, 오페론 예측, 유전자 기능 분류 등 여러 작업에서 뛰어난 성능을 보였습니다. 기존의 단백질 언어 모델(pLM)과 비교했을 때, gLM은 효소 기능 예측에서 5.5% 더 높은 정확도(51.6% vs. 47%)를 기록했습니다. 또한, 다양한 EC(Enzyme Commission) 클래스에서 F1 점수가 유의미하게 향상된 것을 확인하였으며, 이는 유전자 간의 맥락에 따른 관계를 잘 학습했음을 보여줍니다. 


gLM demonstrated strong performance in several tasks, including enzyme function prediction, operon prediction, and gene function classification. Compared to traditional protein language models (pLM), gLM achieved 5.5% higher accuracy in enzyme function prediction (51.6% vs. 47%). Additionally, significant improvements in F1 scores across various EC (Enzyme Commission) classes were observed, indicating that the model effectively learned context-dependent relationships between genes.  




<br/>
# 예시  

gLM의 성능을 보여주는 예시로 ModA와 ModC 단백질의 상동 유전자(paralog) 매칭 작업이 있습니다. 이 두 단백질은 ABC 수송체 복합체를 형성하는데, gLM은 이 작업에 대해 명시적으로 훈련되지 않았음에도 불구하고 추가적인 파인튜닝 없이 상호작용하는 상동 유전자를 성공적으로 예측했습니다. 2700개의 상호작용하는 쌍 중에서, gLM은 398개의 쌍을 정확히 예측했으며, 이는 무작위 예측에 비해 훨씬 높은 성능입니다(무작위로는 약 1.6개의 예측이 정확할 것으로 예상됨). 또한, 예측 신뢰도가 높은 경우(>0.9)의 경우 상동 유전자 매칭에서 25.1%의 정확도를 달성하여, 서열 맥락을 넘어 유전자 간의 관계를 학습하는 gLM의 능력을 보여주었습니다.   



An example that demonstrates gLM’s performance is the paralog matching task involving the ModA and ModC proteins, which form an ABC transporter complex. Even though gLM was not explicitly trained for this task, it successfully predicted the embedding of interacting paralogs without any additional fine-tuning. Out of 2700 interacting pairs, gLM correctly predicted 398 pairs, significantly outperforming random chance (expected to predict around 1.6 pairs). Additionally, for predictions with a high confidence score (>0.9), gLM achieved an accuracy of 25.1% in matching paralogs, showcasing its ability to learn relationships between genes beyond their immediate sequence context​.  



<br/>  
# 요약 

gLM은 Hugging Face의 RoBERTa 트랜스포머 아키텍처를 기반으로 하여 훈련되었습니다. 훈련 중 각 서열의 15%가 무작위로 마스킹되며, 모델은 유전체 맥락을 활용해 마스킹된 유전자를 예측합니다. gLM은 효소 기능 예측에서 기존 pLM보다 5.5% 더 높은 정확도를 기록하였으며, 다양한 EC 클래스에서 F1 점수도 유의미하게 향상되었습니다. ModA와 ModC 단백질의 상동 유전자 매칭 작업에서, gLM은 2700개의 상호작용 쌍 중 398개의 쌍을 정확히 예측하여 무작위 예측보다 훨씬 뛰어난 성과를 보였습니다. 이는 gLM이 서열 이상의 유전자 간 관계를 학습했음을 보여줍니다.



The gLM is based on the Hugging Face RoBERTa transformer architecture. During training, 15% of the sequences were randomly masked, and the model predicted the masked genes using genomic context. gLM achieved 5.5% higher accuracy than traditional pLM in enzyme function prediction and significantly improved F1 scores across various EC classes. In the paralog matching task involving ModA and ModC proteins, gLM accurately predicted 398 out of 2700 interacting pairs, outperforming random chance predictions. This demonstrates gLM's ability to learn relationships between genes beyond simple sequence prediction.



# 기타  


<br/>
# refer format:     

@article{hwang2024genomic,
  title={A Genomic Language Model Learns Functional and Regulatory Relationships from Metagenomic Contexts},
  author={Yunha Hwang and Andre L. Cornman and Elizabeth H. Kellogg and Sergey Ovchinnikov and Peter R. Girguis},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={2880},
  year={2024},
  doi={10.1038/s41467-024-46947-9},
  publisher={Springer Nature}
}




Hwang, Yunha, Andre L. Cornman, Elizabeth H. Kellogg, Sergey Ovchinnikov, and Peter R. Girguis. 2024. "A Genomic Language Model Learns Functional and Regulatory Relationships from Metagenomic Contexts." Nature Communications 15 (1): 2880. https://doi.org/10.1038/s41467-024-46947-9.







