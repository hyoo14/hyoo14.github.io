---
layout: post
title:  "[2024]The OMG dataset: An Open MetaGenomic corpus for mixed-modality genomic language modeling"  
date:   2024-09-10 16:48:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

짧은 요약(Abstract) :    

이 논문은 **Open MetaGenomic(OMG)** 데이터세트를 소개합니다. OMG는 두 개의 대규모 메타게놈 저장소(JGI의 IMG와 EMBL의 MGnify)를 결합하여 약 3.1조 염기쌍과 33억 개의 단백질 코딩 서열을 포함하고 있으며, **뉴클레오타이드(핵산)**와 **아미노산(단백질 코딩 서열)**을 함께 사용하는 **혼합 모드 데이터세트**입니다. 단백질 코딩 유전자 부분은 아미노산으로, 비유전자 부분은 뉴클레오타이드로 표현되어 있습니다. 이를 활용해 **gLM2**라는 최초의 혼합 모드 유전체 언어 모델을 훈련시켰으며, 이 모델은 단백질-단백질 상호작용 인터페이스에서 기능적 표현과 공진화 신호를 학습합니다.

gLM2는 **RoBERTa**와 유사한 **트랜스포머 기반 아키텍처**를 사용하지만, 혼합 모드 서열에 맞게 특화되었습니다. 아미노산 서열은 개별 아미노산으로 토큰화되고, 뉴클레오타이드 서열은 **BPE(Byte-Pair Encoding)** 방식으로 토큰화됩니다. gLM2는 약 2048개의 토큰 길이를 가지며, 150M 및 650M 파라미터 크기로 훈련되었으며, 중복 제거를 통해 데이터 균형을 맞추어 성능을 향상시켰습니다.  



This paper introduces the Open MetaGenomic (OMG) dataset. OMG combines two large-scale metagenomic repositories (JGI’s IMG and EMBL’s MGnify), containing approximately 3.1 trillion base pairs and 3.3 billion protein-coding sequences. It is a mixed-modality dataset that includes both nucleotide (nucleic acid) and amino acid (protein-coding sequences). The protein-coding regions are represented by amino acids, while non-coding regions are represented by nucleotides. Using this dataset, the first mixed-modality genomic language model, gLM2, was trained. This model effectively learns functional representations and coevolutionary signals in protein-protein interaction interfaces.

gLM2 utilizes a transformer-based architecture, similar to RoBERTa, but specialized for mixed-modality sequences. Amino acid sequences are tokenized at the individual amino acid level, while nucleotide sequences are tokenized using Byte-Pair Encoding (BPE). gLM2 supports a sequence length of approximately 2048 tokens and has been trained with 150M and 650M parameter versions. Additionally, semantic deduplication was applied to balance the dataset and improve performance.






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


gLM2는 RoBERTa 아키텍처를 기반으로 하지만, **뉴클레오타이드(핵산)**와 아미노산 서열을 함께 학습시킨다는 점에서 차별화됩니다. 이 모델은 두 종류의 서열 데이터를 혼합 모드로 처리하여 학습하는 것이 주요 특징입니다. 구체적으로는, 단백질 코딩 서열은 아미노산으로, 비유전자 서열은 뉴클레오타이드로 변환된 후, 두 서열이 **연결(컨케티네이트)**되어 하나의 시퀀스로 학습됩니다.

또한, 마스킹은 각각의 뉴클레오타이드와 아미노산 서열에 대해 따로 적용되어 학습됩니다. 이를 통해 두 종류의 생물학적 데이터를 효과적으로 이해할 수 있도록 설계되었습니다.

토크나이저는 BPE(Byte-Pair Encoding) 방식을 사용하며, 이는 RoBERTa와 동일한 방식입니다. 두 모델 모두 BPE를 사용해 서열을 서브워드 단위로 나누어 처리합니다.

결론적으로, gLM2는 RoBERTa 아키텍처를 기반으로 뉴클레오타이드와 아미노산 서열을 혼합해 학습시키는 모델이며, 두 서열을 연결한 후 각각의 서열에 대해 마스킹을 적용하고, BPE 토크나이저를 통해 데이터를 처리합니다.



gLM2 is based on the RoBERTa architecture but differs in that it jointly trains on both nucleotide (nucleic acid) and amino acid sequences. The main feature of this model is that it processes and learns from both types of sequences in a mixed modality. Specifically, protein-coding sequences are represented as amino acids, and non-coding sequences are represented as nucleotides. These two sequences are then concatenated and trained as a single sequence.

Additionally, masking is applied separately to both the nucleotide and amino acid sequences. This design allows the model to effectively understand and learn from both types of biological data.

The tokenizer uses BPE (Byte-Pair Encoding), which is the same method used by RoBERTa. Both models tokenize sequences into subword units using BPE.

In conclusion, gLM2 is a model based on the RoBERTa architecture that learns from both nucleotide and amino acid sequences in a mixed modality. The sequences are concatenated, masked separately, and processed using a BPE tokenizer.


<br/>
# Results  


단백질 및 유전체 서열에서의 성능:

gLM2는 ESM2 모델보다 단백질 관련 다운스트림 작업에서 성능이 더 우수했습니다. 특히 단백질-단백질 상호작용에서 공진화 신호를 학습하는 능력이 뛰어났습니다. 이는 gLM2가 다중 단백질 맥락 정보를 활용하여 상호작용 신호를 더 잘 학습할 수 있기 때문입니다.
유전체 데이터에서의 성능:

gLM2는 DNA 서열 처리에서도 좋은 성능을 보였습니다. gLM2는 주로 단백질 작업에 중점을 두었지만, DNA 작업에서도 Nucleotide Transformer 시리즈와 비슷한 수준의 성능을 보여주었습니다.
데이터 균형:

gLM2는 데이터 중복 제거(semantic deduplication)를 통해 데이터의 편향성을 줄이고, 특히 과소대표된 생물종에 대한 성능을 개선하는 데 기여했습니다. 이를 통해 특정 생물종에 대한 편향 없이 더 균형 잡힌 성능을 보여주었습니다.



Performance on protein and genomic sequences:

gLM2 outperformed the ESM2 model in downstream tasks related to proteins. In particular, gLM2 excelled at learning coevolutionary signals in protein-protein interactions. This is because gLM2 can better capture interaction signals by utilizing multi-protein contextual information.

Performance on genomic data:

gLM2 also demonstrated strong performance in processing DNA sequences. While gLM2 primarily focuses on protein tasks, it showed performance comparable to the Nucleotide Transformer series when handling DNA tasks.

Data balancing:

gLM2 utilized semantic deduplication to reduce biases in the dataset, particularly improving performance for underrepresented taxa. This resulted in more balanced performance without over-reliance on certain species.






<br/>
# 예시  

gLM2는 단백질-단백질 상호작용 작업에서 특히 뛰어난 성능을 보였으며, 단백질 잔기 간의 공진화 신호를 학습하는 데 매우 효과적이었습니다. 예를 들어, 2ONK 복합체(ModA와 ModC 단백질)의 상호작용 잔기를 정확하게 예측했으며, 이는 다른 모델들, 특히 ESM2가 잘 포착하지 못했던 부분입니다. gLM2는 카테고리컬 제이콥(Jacobian) 계산을 사용하여 단백질 간 상호작용을 예측했으며, 이를 위해 다중 서열 정렬(MSA) 없이도 뛰어난 성능을 발휘했습니다. 또한, gLM2는 아미노산 작업에서 우수한 성능을 보였으며, **뉴클레오타이드 변환기(Nucleotide Transformer)**와 비교했을 때 핵산 작업에서도 유사한 성능을 보였습니다. 이 모델의 데이터 중복 제거 과정은 특히 과소 대표된 생물종에 대한 성능을 향상시켜 다양한 유전체 환경에서 더 균형 잡힌 기능 예측을 가능하게 했습니다.



gLM2 demonstrated outstanding performance in protein-protein interaction tasks, particularly in effectively learning coevolutionary signals between protein residues. For instance, it accurately predicted the interacting residues of the 2ONK complex (ModA and ModC proteins), a challenge that other models, especially ESM2, struggled to capture. gLM2 used categorical Jacobian calculations to predict protein-protein interactions, and it achieved excellent results without the need for multiple sequence alignment (MSA). Additionally, gLM2 showed superior performance in amino acid tasks and performed similarly to the Nucleotide Transformer in nucleic acid tasks. The model's semantic deduplication process particularly improved its performance on underrepresented taxa, enabling more balanced functional prediction across various genomic contexts.




<br/>  
# 요약 

gLM2는 RoBERTa 아키텍처를 기반으로 하여, 뉴클레오타이드와 아미노산 서열을 함께 학습하는 혼합 모드 모델입니다. 이 모델은 각각의 서열에 대해 마스킹을 적용하고, BPE(Byte-Pair Encoding) 토크나이저를 통해 데이터를 처리합니다. gLM2는 단백질-단백질 상호작용에서 뛰어난 성능을 보였으며, 공진화 신호를 효과적으로 학습하여, ESM2보다 2ONK 복합체의 상호작용 잔기를 더 잘 예측했습니다. 또한, 유전체 작업에서도 Nucleotide Transformer와 비슷한 성능을 보였고, 데이터 중복 제거를 통해 과소 대표된 생물종에 대한 성능을 개선했습니다. 결론적으로, gLM2는 다양한 생물학적 서열에서 균형 잡힌 성능을 발휘하는 모델로 평가됩니다.



gLM2 is a mixed modality model based on the RoBERTa architecture, designed to jointly learn from nucleotide and amino acid sequences. The model applies masking to each sequence type and processes the data using a BPE (Byte-Pair Encoding) tokenizer. gLM2 excelled in protein-protein interaction tasks, effectively learning coevolutionary signals and outperforming ESM2 in predicting the interacting residues of the 2ONK complex. It also performed comparably to the Nucleotide Transformer in genomic tasks and improved performance on underrepresented taxa through semantic deduplication. Overall, gLM2 demonstrates balanced performance across various biological sequences.




# 기타  


ESM2는 단백질 서열 관련 작업을 위해 설계된 단백질 언어 모델입니다. Meta AI에서 개발한 이 모델은 트랜스포머 아키텍처를 기반으로 하지만, 단백질의 구조 및 기능을 예측하는 데 특화되어 있습니다. ESM2는 대규모 **자기 지도 학습(self-supervised learning)**을 통해 단백질 서열의 패턴을 학습하고, 이를 통해 단백질 접힘(folding)이나 기능 예측과 같은 생물학적 작업에 사용됩니다.

ESM2는 **로버타(RoBERTa)**와 같은 일반적인 자연어 처리 모델과 유사한 점이 있지만, 생물학적 서열을 처리하는 데 최적화되어 있습니다. 특히, 단백질의 구조적 정보를 잘 학습하고, 단백질 서열의 기능적 의미를 추론하는 데 뛰어난 성능을 보입니다.

따라서 ESM2는 자연어 처리보다는 단백질 서열 분석에 특화된 트랜스포머 모델로, 단백질의 기능과 상호작용을 예측하는 생물학적 연구에서 많이 사용됩니다.



ESM2 is a protein language model designed for tasks related to protein sequences. It is based on transformer architecture, but it is not specifically built on RoBERTa. ESM2 is a large-scale self-supervised model developed by Meta AI that predicts the structure and function of proteins. While it shares some similarities with RoBERTa in using transformer layers, ESM2 is particularly optimized for biological sequences, focusing on protein folding and functional prediction rather than natural language tasks.

In summary, ESM2 is a transformer-based model but tailored for protein sequence analysis, rather than a direct adaptation of RoBERTa like gLM2.


<br/>
# refer format:     

@article{cornman2024omg,
  title={The OMG dataset: An Open MetaGenomic corpus for mixed-modality genomic language modeling},
  author={Cornman, Andre and West-Roberts, Jacob and Camargo, Antonio Pedro and Roux, Simon and Beracochea, Martin and Mirdita, Milot and Ovchinnikov, Sergey and Hwang, Yunha},
  journal={bioRxiv},
  year={2024},
  doi={10.1101/2024.08.14.607850},
  url={https://doi.org/10.1101/2024.08.14.607850}
}



Cornman, Andre, Jacob West-Roberts, Antonio Pedro Camargo, Simon Roux, Martin Beracochea, Milot Mirdita, Sergey Ovchinnikov, and Yunha Hwang. 2024. "The OMG Dataset: An Open MetaGenomic Corpus for Mixed-Modality Genomic Language Modeling." bioRxiv. August 17, 2024. https://doi.org/10.1101/2024.08.14.607850.




