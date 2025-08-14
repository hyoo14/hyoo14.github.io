---
layout: post
title:  "[2025]SetBERT: the deep learning platform for contextualized embeddings and explainable predictions from high-throughput sequencing"
date:   2025-08-14 20:23:47 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

Set Transformer(ST) 프레임워크를 기반으로 하여 수천 개의 짧은 읽기 시퀀스를 순열 불변 방식으로 학습(BERT 식으로 학습)  


짧은 요약(Abstract) :

이 논문은 고처리량 시퀀싱(HTS) 데이터를 처리하기 위한 SetBERT라는 새로운 딥러닝 플랫폼을 소개합니다. HTS는 미생물 군집을 프로파일링하기 위해 수천 개의 짧은 유전자 조각을 시퀀싱하는 기술입니다. 그러나 HTS 데이터의 비구조적 특성 때문에 대부분의 기존 모델은 DNA 시퀀스를 개별적으로 처리하는 데 제한이 있습니다. 이는 미생물 간의 중요한 상호작용을 놓치게 하여 미생물 군집에 대한 이해를 방해합니다. SetBERT는 이러한 문제를 해결하기 위해 설계되었으며, HTS 데이터를 처리하여 문맥화된 임베딩을 생성하고 설명 가능한 예측을 가능하게 합니다. SetBERT는 시퀀스 간의 상호작용을 활용하여 다른 모델보다 우수한 성능을 보이며, 속(genus) 수준의 분류 정확도가 95%에 달합니다. 또한, SetBERT는 모델이 식별한 생물학적으로 관련 있는 분류군을 확인함으로써 예측을 정확하게 설명할 수 있습니다.


The paper introduces SetBERT, a novel deep learning platform designed for processing high-throughput sequencing (HTS) data. HTS is a technology used to profile microbial communities by sequencing thousands of short genomic fragments. However, due to the unstructured nature of HTS data, most existing models are limited to processing DNA sequences individually, missing out on key interactions between microorganisms, which hinders our understanding of microbial communities. SetBERT addresses these issues by creating contextualized embeddings and enabling explainable predictions. By leveraging sequence interactions, SetBERT significantly outperforms other models, achieving a genus-level classification accuracy of 95%. Furthermore, SetBERT can autonomously explain its predictions by confirming the biological relevance of the taxa identified by the model.


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


SetBERT는 고처리량 시퀀싱(HTS) 데이터를 처리하기 위해 설계된 심층 학습 아키텍처로, 특히 앰플리콘 시퀀싱에 중점을 두고 있습니다. 이 모델은 Set Transformer(ST) 프레임워크를 기반으로 하여 수천 개의 짧은 읽기 시퀀스를 순열 불변 방식으로 처리할 수 있는 아키텍처를 만듭니다. SetBERT는 입력의 풍부도/분포 정보를 캡처하여 맥락화된 예측을 수행할 수 있으며, 생물학적 서식지에 대한 명시적인 표시 없이도 작동할 수 있습니다.

SetBERT의 핵심은 BERT(Bi-directional Encoder Representations from Transformers) 모델의 아이디어를 차용한 것입니다. BERT는 자연어 처리(NLP)에서 문맥화된 단어 임베딩을 학습하기 위해 설계된 강력한 심층 학습 모델입니다. BERT는 트랜스포머 아키텍처를 기반으로 하며, 사전 훈련/미세 조정 패러다임을 사용합니다. SetBERT는 이러한 BERT의 개념을 DNA 시퀀스 처리에 적용하여, 시퀀스 간의 상호작용을 활용하여 더 정확한 예측을 수행합니다.

SetBERT의 아키텍처는 8개의 세트 주의 블록(SAB)으로 구성되어 있으며, 각 블록은 8개의 주의 헤드를 포함합니다. 이 모델은 DNA 시퀀스 집합을 입력으로 받아, 각 시퀀스를 문맥화된 벡터 표현으로 변환합니다. 또한, SetBERT는 DNABERT를 사용하여 DNA 시퀀스를 임베딩하며, 이는 SILVA 데이터셋에서 150bp 시퀀스로 사전 훈련된 모델입니다.

SetBERT의 사전 훈련은 합성 샘플을 사용하여 수행됩니다. 이러한 합성 샘플은 실제 샘플의 분포를 기반으로 생성되며, 모델은 이러한 샘플을 통해 풍부도 분포를 예측하도록 훈련됩니다. 사전 훈련 후, SetBERT는 다양한 다운스트림 작업에 대해 빠르게 미세 조정될 수 있습니다.





SetBERT is a deep learning architecture designed for processing high-throughput sequencing (HTS) data, with a particular focus on amplicon sequencing. The model builds upon the Set Transformer (ST) framework to create an architecture capable of processing thousands of short-read sequences in a permutation-invariant manner. SetBERT captures the input abundance/distribution information to make contextualized predictions and can operate without explicit indication of the biological habitat.

The core of SetBERT borrows the idea from BERT (Bi-directional Encoder Representations from Transformers), a powerful deep learning model designed for learning contextualized word embeddings in natural language processing (NLP). BERT is based on the transformer architecture and employs a pre-training/fine-tuning paradigm. SetBERT applies these BERT concepts to DNA sequence processing, leveraging interactions between sequences to make more accurate predictions.

The architecture of SetBERT consists of 8 set attention blocks (SABs), each with 8 attention heads. The model takes a set of DNA sequences as input and transforms each sequence into a contextualized vector representation. Additionally, SetBERT employs DNABERT to embed DNA sequences, which is a model pre-trained on 150bp sequences from the SILVA dataset.

Pre-training of SetBERT is performed using synthetic samples. These synthetic samples are generated based on the distribution of real samples, and the model is trained to predict the abundance distribution from these samples. After pre-training, SetBERT can be quickly fine-tuned for various downstream tasks.


<br/>
# Results


SetBERT는 고처리량 시퀀싱(HTS) 데이터를 처리하기 위한 강력한 딥러닝 모델로, 다른 모델들에 비해 뛰어난 성능을 보였습니다. 이 모델은 DNA 시퀀스 간의 상호작용을 활용하여 더 정확한 예측을 가능하게 합니다. 다양한 생물군집에서 사전 학습을 통해, SetBERT는 유사한 환경에 대해 일반화할 수 있는 능력을 갖추고 있습니다. 또한, 모델의 예측에 사용된 관련된 분류군을 주의 점수(attention scores)를 통해 분석할 수 있으며, 이는 기존 문헌에서 생물학적 중요성을 확인할 수 있습니다.

#### 경쟁 모델과의 비교

SetBERT는 DNABERT, DNABERT 2, MetaBERTa, HyenaDNA, SetQuence/SetOmic 등과 같은 기존의 모델들과 비교되었습니다. 이들 모델은 주로 단일 시퀀스 처리에 중점을 두고 있으며, 샘플 수준의 처리나 원시 샘플 처리에는 한계가 있었습니다. 반면, SetBERT는 샘플 수준의 처리와 원시 샘플 처리 모두를 지원하며, 사전 학습이 가능하여 다양한 다운스트림 작업에 빠르게 적응할 수 있습니다.

#### 테스트 데이터

SetBERT는 Hopland, Nachusa, Snake Fungal Disease (SFD), Wetland 등 네 가지 실험적 앰플리콘 시퀀싱 데이터셋을 사용하여 평가되었습니다. 각 데이터셋은 서로 다른 환경에서 수집된 샘플로 구성되어 있으며, 모델의 일반화 능력을 테스트하는 데 사용되었습니다.

#### 메트릭 및 비교

SetBERT의 성능은 주로 분류 정확도와 설명 가능성 측면에서 평가되었습니다. 분류 정확도는 속(genus) 수준에서 측정되었으며, SetBERT는 다른 모델들에 비해 높은 정확도를 기록했습니다. 특히, SetBERT는 샘플 내의 시퀀스 상호작용을 활용하여 더 높은 정확도를 달성할 수 있었습니다. 설명 가능성 측면에서는, 주의 점수를 통해 모델의 예측에 가장 큰 영향을 미치는 시퀀스나 분류군을 식별할 수 있었습니다.



SetBERT is a powerful deep learning model for processing high-throughput sequencing (HTS) data, demonstrating superior performance compared to other models. It leverages interactions between DNA sequences to enable more accurate predictions. Through pre-training across various biomes, SetBERT is capable of generalizing to similar environments. Additionally, attention scores can be analyzed to identify relevant taxa used in the model's predictions, confirming their biological significance in existing literature.

#### Comparison with Competing Models

SetBERT was compared with existing models such as DNABERT, DNABERT 2, MetaBERTa, HyenaDNA, and SetQuence/SetOmic. These models primarily focus on single-sequence processing and have limitations in sample-level or raw sample processing. In contrast, SetBERT supports both sample-level and raw sample processing and is pre-trainable, allowing it to quickly adapt to various downstream tasks.

#### Test Data

SetBERT was evaluated using four experimental amplicon sequencing datasets: Hopland, Nachusa, Snake Fungal Disease (SFD), and Wetland. Each dataset consists of samples collected from different environments and was used to test the model's generalization capabilities.

#### Metrics and Comparison

SetBERT's performance was primarily evaluated in terms of classification accuracy and explainability. Classification accuracy was measured at the genus level, with SetBERT achieving higher accuracy compared to other models. Notably, SetBERT was able to achieve higher accuracy by leveraging sequence interactions within samples. In terms of explainability, attention scores were used to identify sequences or taxa that had the most significant impact on the model's predictions.


<br/>
# 예제

#### 예시: SetBERT의 트레이닝 및 테스트 데이터

1. **트레이닝 데이터**
   - **입력 데이터**: SetBERT는 다양한 생태계에서 수집된 고처리량 시퀀싱(HTS) 데이터를 사용하여 훈련됩니다. 이 데이터는 주로 토양 기반 생태계(예: Hopland, Nachusa, Wetland)와 뱀 피부 질병(SFD) 데이터셋으로 구성됩니다. 각 샘플은 수천 개의 짧은 DNA 시퀀스로 구성되어 있으며, 각 시퀀스는 해당 샘플의 미생물 군집을 대표합니다.
   - **출력 데이터**: 모델은 각 샘플에 대해 해당 샘플의 미생물 군집을 나타내는 벡터 표현을 생성합니다. 이 벡터는 샘플 내의 시퀀스 간 상호작용을 반영하여 샘플의 전체적인 특성을 나타냅니다.

2. **테스트 데이터**
   - **입력 데이터**: 테스트 데이터는 훈련 데이터와 유사한 형식으로 구성되며, 모델의 일반화 능력을 평가하기 위해 훈련에 사용되지 않은 샘플로 구성됩니다. 예를 들어, SFD 데이터셋을 제외한 나머지 데이터셋으로 훈련한 모델을 SFD 데이터셋으로 테스트합니다.
   - **출력 데이터**: 모델은 테스트 샘플에 대해 벡터 표현을 생성하고, 이를 통해 샘플 간의 유사성을 평가하거나 특정 분류 작업을 수행합니다.

#### 구체적인 테스크

1. **분류 작업**
   - **Hopland 데이터셋**: 이 데이터셋은 두 가지 토양 영역(벌크 및 근권)에서 수집된 샘플로 구성됩니다. 모델은 각 샘플이 벌크인지 근권인지 예측하는 이진 분류기를 학습합니다.
   - **SFD 데이터셋**: 이 데이터셋은 Ophidiomyces ophidiicola의 존재 여부에 따라 양성/음성으로 구분됩니다. 모델은 각 샘플이 양성인지 음성인지 예측하는 이진 분류기를 학습합니다.

2. **설명 가능성**
   - 모델은 주의(attention) 점수를 분석하여 예측에 가장 중요한 시퀀스를 식별합니다. 이를 통해 모델의 예측 이유를 설명하고, 생물학적으로 중요한 미생물 군집을 확인할 수 있습니다.

3. **계통 분류 할당**
   - 모델은 샘플 내의 각 DNA 시퀀스에 대해 계통 분류를 할당합니다. SetBERT는 시퀀스 간 상호작용을 활용하여 더 정확한 계통 분류를 수행합니다.



#### Example: Training and Testing Data for SetBERT

1. **Training Data**
   - **Input Data**: SetBERT is trained using high-throughput sequencing (HTS) data collected from various ecosystems. This data primarily consists of soil-based ecosystems (e.g., Hopland, Nachusa, Wetland) and snake fungal disease (SFD) datasets. Each sample comprises thousands of short DNA sequences, each representing the microbial community of that sample.
   - **Output Data**: For each sample, the model generates a vector representation that reflects the overall characteristics of the sample by capturing the interactions between sequences within the sample.

2. **Testing Data**
   - **Input Data**: The testing data is structured similarly to the training data and consists of samples not used during training to evaluate the model's generalization ability. For instance, a model trained on datasets excluding the SFD dataset is tested on the SFD dataset.
   - **Output Data**: The model generates vector representations for the test samples, which are used to assess sample similarity or perform specific classification tasks.

#### Specific Tasks

1. **Classification Tasks**
   - **Hopland Dataset**: This dataset consists of samples collected from two soil regions: bulk and rhizosphere. The model learns a binary classifier to predict whether each sample is from the bulk or rhizosphere region.
   - **SFD Dataset**: This dataset is categorized into positive/negative based on the presence of Ophidiomyces ophidiicola. The model learns a binary classifier to predict whether each sample is positive or negative.

2. **Explainability**
   - The model analyzes attention scores to identify the most critical sequences for its predictions. This process explains the model's reasoning and confirms biologically significant microbial communities.

3. **Taxonomic Assignment**
   - The model assigns taxonomic labels to each DNA sequence within a sample. SetBERT leverages sequence interactions to perform more accurate taxonomic classification.

<br/>
# 요약

SetBERT는 고처리량 시퀀싱 데이터를 처리하기 위해 설계된 딥러닝 아키텍처로, 시퀀스 간의 상호작용을 활용하여 문맥화된 예측을 수행합니다. 이 모델은 다양한 생물군집에서 사전 학습되어, 유사한 환경에 대해 일반화된 예측을 제공하며, 주목할 만한 생물학적 관련성을 가진 택사를 식별할 수 있습니다. 실험 결과, SetBERT는 다른 모델들보다 높은 정확도로 택사 분류를 수행하며, 특히 대량의 시퀀스를 동시에 처리할 때 성능이 향상됩니다.



SetBERT is a deep learning architecture designed to process high-throughput sequencing data, leveraging interactions between sequences to make contextualized predictions. Pre-trained across various biomes, the model provides generalized predictions for similar environments and can identify taxa with notable biological relevance. Experimental results show that SetBERT outperforms other models in taxonomic classification accuracy, especially when processing large sets of sequences simultaneously.

<br/>
# 기타


#### Figure 1: SetBERT 모델 개요
이 다이어그램은 SetBERT 모델의 구조와 작동 방식을 시각적으로 설명합니다. SetBERT는 DNA 시퀀스를 입력으로 받아, 이를 문맥화된 임베딩으로 변환합니다. 이 과정에서 [CLS]와 [MASK] 같은 특별한 토큰을 사용하여 시퀀스의 상대적 분포를 예측합니다. 이 다이어그램은 SetBERT가 어떻게 시퀀스 간의 상호작용을 활용하여 예측을 수행하는지를 보여줍니다.

#### Figure 2: SetBERT 샘플 임베딩의 MDS 플롯
이 플롯은 SetBERT 모델이 생성한 샘플 임베딩을 2차원으로 시각화한 것입니다. 샘플 간의 유사성은 점들 간의 거리로 표현되며, 가까운 점들은 유사성이 높음을 나타냅니다. 이 플롯은 토양 샘플과 뱀 피부 미생물군집 간의 명확한 구분을 보여주며, SetBERT가 샘플 간의 유사성을 잘 포착하고 있음을 시사합니다.

#### Figure 3: SetBERT의 MDS 및 정밀도/재현율 곡선
이 그림은 SetBERT의 샘플 임베딩을 MDS로 시각화한 것과 정밀도/재현율 곡선을 보여줍니다. MDS 플롯은 샘플 간의 명확한 구분을 보여주며, 정밀도/재현율 곡선은 모델의 분류 성능을 나타냅니다. 높은 AUC 점수는 SetBERT가 높은 정확도로 샘플을 분류할 수 있음을 나타냅니다.

#### Figure 4: Hopland 및 SFD 이진 분류기의 상위 10개 분류군
이 그림은 Hopland와 SFD 데이터셋에서 모델이 가장 중요하게 여기는 상위 10개 분류군을 보여줍니다. 상대적 풍부도와 상대적 긍정적 기여도를 비교하여, 모델이 단순한 풍부도 이상의 정보를 활용하여 예측을 수행함을 시사합니다.

#### Figure 5: 속(genus) 수준의 분류 정확도
이 바이올린 플롯은 다양한 모델의 속 수준 분류 정확도를 보여줍니다. SetBERT는 다른 모델들에 비해 높은 분류 정확도를 보이며, 이는 시퀀스 간의 상호작용을 활용한 결과임을 나타냅니다.

#### Figure 6: 훈련에서 제외된 데이터셋의 분류 정확도
이 그림은 훈련에서 제외된 데이터셋에 대한 SetBERT의 일반화 성능을 보여줍니다. SetBERT는 유사한 환경에 대해 잘 일반화할 수 있지만, 훈련에서 제외된 SFD 데이터셋에 대해서는 성능이 저하됩니다. 이는 SFD가 다른 데이터셋과 생태계가 다르기 때문입니다.

### English Version



#### Figure 1: Overview of the SetBERT Model
This diagram visually explains the structure and operation of the SetBERT model. SetBERT takes DNA sequences as input and transforms them into contextualized embeddings. It uses special tokens like [CLS] and [MASK] to predict the relative distribution of sequences. The diagram illustrates how SetBERT leverages interactions between sequences to make predictions.

#### Figure 2: MDS Plot of SetBERT Sample Embeddings
This plot visualizes the sample embeddings generated by the SetBERT model in two dimensions. The similarity between samples is represented by the distance between points, with closer points indicating higher similarity. The plot shows a clear distinction between soil samples and snake skin microbiomes, suggesting that SetBERT effectively captures sample similarities.

#### Figure 3: MDS and Precision/Recall Curves of SetBERT
This figure shows the MDS visualization of SetBERT's sample embeddings and the precision/recall curves. The MDS plot demonstrates clear separation between samples, and the precision/recall curves indicate the model's classification performance. High AUC scores suggest that SetBERT can classify samples with high accuracy.

#### Figure 4: Top 10 Taxa from Hopland and SFD Binary Classifiers
This figure displays the top 10 taxa that the model considers most important in the Hopland and SFD datasets. By comparing relative abundance and relative positive attribution, it suggests that the model uses more than just abundance information to make predictions.

#### Figure 5: Genus-level Taxonomy Classification Accuracy
This violin plot shows the genus-level classification accuracy of various models. SetBERT demonstrates higher classification accuracy compared to other models, indicating its ability to leverage interactions between sequences.

#### Figure 6: Classification Accuracy of Datasets Left Out of Training
This figure shows SetBERT's generalization performance on datasets excluded from training. SetBERT generalizes well to similar environments but shows reduced performance on the SFD dataset, which differs ecologically from the others.

<br/>
# refer format:

**BibTeX 형식:**
```bibtex
@article{Ludwig2025,
  author = {David W. Ludwig II and Christopher Guptil and Nicholas R. Alexander and Kateryna Zhalnina and Edi M.-L. Wipf and Albina Khasanova and Nicholas A. Barber and Wesley Swingley and Donald M. Walker and Joshua L. Phillips},
  title = {SetBERT: the deep learning platform for contextualized embeddings and explainable predictions from high-throughput sequencing},
  journal = {Bioinformatics},
  volume = {41},
  number = {7},
  pages = {btaf370},
  year = {2025},
  publisher = {Oxford University Press},
  doi = {10.1093/bioinformatics/btaf370},
  url = {https://doi.org/10.1093/bioinformatics/btaf370}
}
```

**시카고 스타일:**
Ludwig, David W., II, Christopher Guptil, Nicholas R. Alexander, Kateryna Zhalnina, Edi M.-L. Wipf, Albina Khasanova, Nicholas A. Barber, Wesley Swingley, Donald M. Walker, and Joshua L. Phillips. 2025. "SetBERT: The Deep Learning Platform for Contextualized Embeddings and Explainable Predictions from High-Throughput Sequencing." *Bioinformatics* 41 (7): btaf370. https://doi.org/10.1093/bioinformatics/btaf370.
