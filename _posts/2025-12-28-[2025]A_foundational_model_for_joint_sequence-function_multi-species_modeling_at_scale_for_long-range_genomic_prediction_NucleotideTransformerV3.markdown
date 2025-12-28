---
layout: post
title:  "[2025]A foundational model for joint sequence-function multi-species modeling at scale for long-range genomic prediction_NucleotideTransformer_v3"
date:   2025-12-28 16:37:16 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 Nucleotide Transformer v3 (NTv3)라는 다중 종 유전체 모델을 소개하며, 이 모델은 1Mb의 긴 서열을 처리하고 기능적 예측 및 유전체 주석을 동시에 수행할 수 있도록 설계되었습니다.   
요즘 흥한다는 마스크 디퓨전 모델  


짧은 요약(Abstract) :


이 논문에서는 Nucleotide Transformer v3 (NTv3)라는 다중 종 유전자 모델링을 위한 기초 모델을 소개합니다. NTv3는 지역 서열 특성과 수백 킬로베이스에서 메가베이스에 걸친 장거리 조절 의존성을 통합하여 유전자 예측을 수행합니다. 기존의 접근 방식들은 일반적으로 서로 다른 모델 클래스와 아키텍처에서 독립적으로 발전해왔으며, 이로 인해 장거리, 염기 해상도 예측, 기능 모델링 및 제어 가능한 생성 기능을 단일 효율적인 프레임워크로 통합하는 데 한계가 있었습니다. NTv3는 이러한 기능을 통합하여 1Mb까지의 컨텍스트를 효율적으로 모델링할 수 있도록 설계되었습니다. NTv3는 9조(9 trillion base pairs) 개의 염기쌍으로 구성된 OpenGenome2 데이터셋에서 사전 훈련되었으며, 24종의 동물 및 식물에서 약 16,000개의 기능 트랙과 주석 레이블에 대한 감독 학습을 통해 후속 훈련을 진행했습니다. NTv3는 기능 트랙 예측 및 유전자 주석에서 최첨단 정확도를 달성하며, 다양한 종에서의 성능을 평가하여 기존의 모델들을 초월하는 결과를 보여줍니다. 마지막으로, NTv3는 마스크 확산 언어 모델링을 통해 제어 가능한 생성 모델로 조정되어, 실험적으로 검증된 특정 활성 수준과 프로모터 선택성을 가진 증강기 서열을 설계할 수 있습니다.


This paper introduces the Nucleotide Transformer v3 (NTv3), a foundational model for multi-species genomic modeling. NTv3 integrates local sequence features with long-range regulatory dependencies spanning hundreds of kilobases to megabases for genomic prediction. Existing approaches have typically progressed independently across distinct model classes and architectures, limiting the ability to combine long-context, base-resolution prediction, functional modeling, and controllable generation within a single efficient framework. NTv3 is designed to unify these capabilities while efficiently modeling contexts up to 1 Mb. It is pre-trained on 9 trillion base pairs from the OpenGenome2 dataset and subsequently fine-tuned with supervised learning on approximately 16,000 functional tracks and annotation labels from 24 animal and plant species. NTv3 achieves state-of-the-art accuracy for functional-track prediction and genome annotation across species, outperforming leading models. Finally, NTv3 is fine-tuned into a controllable generative model via masked diffusion language modeling, enabling the design of enhancer sequences with specified activity levels and promoter selectivity, validated experimentally.


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


#### 1. 모델 아키텍처
Nucleotide Transformer v3 (NTv3)는 유전자 서열을 처리하고 다양한 기능적 출력을 예측하기 위해 설계된 대규모 유전체 기초 모델입니다. NTv3는 U-Net 스타일의 인코더-디코더 아키텍처를 따르며, 이는 단일 뉴클레오타이드 해상도로 모델링할 수 있도록 구성되어 있습니다. 이 아키텍처는 다음과 같은 주요 구성 요소로 이루어져 있습니다:

- **시퀀스 인코더**: 입력 DNA 서열을 점진적으로 다운샘플링하여 128bp 해상도의 임베딩으로 변환합니다. 이 과정은 여러 개의 컨볼루션 블록을 통해 이루어지며, 각 블록은 레이어 정규화, GELU 활성화, 잔차 연결 및 평균 풀링을 포함합니다.
  
- **트랜스포머 타워**: 인코더의 출력을 받아서 1Mb의 입력 컨텍스트에서 장거리 상호작용을 모델링합니다. 이 타워는 여러 개의 멀티헤드 어텐션 레이어로 구성되어 있으며, 로터리 위치 임베딩(RoPE)을 사용하여 위치 정보를 처리합니다.

- **시퀀스 디코더**: 인코더의 구조를 반대로 따라가며, 128bp 해상도의 임베딩을 원래의 1bp 해상도로 복원합니다. 이 과정에서도 U-Net 스타일의 스킵 연결을 사용하여 세부 정보를 복원합니다.

- **출력 헤드**: NTv3는 세 가지 출력 헤드를 가지고 있습니다: 언어 모델 헤드, 기능 트랙 예측 헤드, 유전체 주석 헤드. 각 헤드는 특정한 예측 작업을 수행합니다.

#### 2. 트레이닝 데이터
NTv3는 OpenGenome2 데이터셋을 사용하여 사전 훈련되었습니다. 이 데이터셋은 약 128,000종의 생물에서 수집된 8조 개 이상의 뉴클레오타이드로 구성되어 있습니다. NTv3는 두 단계의 훈련 과정을 거칩니다:

- **사전 훈련**: 마스크된 언어 모델링(MLM) 기법을 사용하여 DNA 서열의 특정 부분을 마스킹하고, 모델이 원래의 뉴클레오타이드를 복원하도록 학습합니다. 이 과정에서 다양한 길이의 서열을 포함하여 모델이 다양한 컨텍스트를 학습할 수 있도록 합니다.

- **후속 훈련**: 기능적 트랙과 유전체 주석 데이터를 사용하여 모델을 추가로 훈련합니다. 이 단계에서는 24종의 동물 및 식물에서 수집된 데이터가 포함되어 있으며, 각 종에 대해 기능적 트랙과 주석을 예측하는 헤드를 추가합니다.

#### 3. 특별한 기법
NTv3는 다음과 같은 특별한 기법을 사용하여 성능을 극대화합니다:

- **혼합 길이 훈련**: 다양한 길이의 서열을 혼합하여 훈련함으로써 모델이 짧은 서열과 긴 서열 모두에서 잘 작동하도록 합니다. 이는 짧은 서열에 대한 성능 저하를 방지하는 데 도움을 줍니다.

- **다중 종 조건화**: 각 종에 대한 예측을 조정하기 위해 종별 임베딩을 사용하여 모델의 출력을 조정합니다. 이는 모델이 다양한 종에 대해 더 정확한 예측을 할 수 있도록 합니다.

- **조건부 생성**: NTv3는 생성 모델로서도 활용될 수 있으며, 특정 활동 수준이나 프로모터에 대한 선택적 활성화를 조건으로 하여 서열을 생성할 수 있습니다. 이를 통해 실험적으로 검증된 활성화 수준을 가진 새로운 유전자 조절 요소를 설계할 수 있습니다.




#### 1. Model Architecture
Nucleotide Transformer v3 (NTv3) is a large-scale genomic foundation model designed to process DNA sequences and predict various functional outputs. NTv3 follows a U-Net style encoder-decoder architecture, which is structured to model at single-nucleotide resolution. The architecture consists of the following key components:

- **Sequence Encoder**: This component progressively downsamples the input DNA sequence to produce embeddings at a resolution of 128bp. This process is achieved through multiple convolutional blocks, each containing layer normalization, GELU activation, residual connections, and average pooling.

- **Transformer Tower**: Following the encoder, the transformer tower processes the 128bp resolution embeddings to model long-range interactions across the full 1Mb input context. This tower is composed of multiple multi-head attention layers that utilize rotary positional embeddings (RoPE) to handle positional information.

- **Sequence Decoder**: The sequence decoder mirrors the encoder's structure but operates in reverse, progressively upsampling the 128bp resolution embeddings back to the original 1bp resolution. U-Net style skip connections are used to recover fine-grained sequence information lost during downsampling.

- **Output Heads**: NTv3 has three output heads: a language model head, a functional track prediction head, and a genome annotation head. Each head performs specific prediction tasks.

#### 2. Training Data
NTv3 was pre-trained on the OpenGenome2 dataset, which comprises over 8 trillion nucleotides collected from approximately 128,000 species. NTv3 undergoes two phases of training:

- **Pre-training**: The model is trained using a masked language modeling (MLM) objective, where certain parts of the DNA sequence are masked, and the model learns to reconstruct the original nucleotides. This phase includes sequences of varying lengths to allow the model to learn from diverse contexts.

- **Post-training**: The model is further trained on functional track prediction and genome annotation data. This phase includes data from 24 species, allowing the model to predict functional tracks and annotations.

#### 3. Special Techniques
NTv3 employs several special techniques to maximize performance:

- **Mixed-Length Training**: By continuously sampling sequences of varying lengths during training, the model maintains exposure to both short and long sequences, preventing performance degradation on shorter contexts.

- **Multi-Species Conditioning**: A species-specific embedding is used to adjust the model's predictions for each species, enabling more accurate predictions across diverse taxa.

- **Conditional Generation**: NTv3 can also function as a generative model, allowing for the design of regulatory elements with specified functional properties. This capability enables the design of new enhancer sequences validated experimentally to achieve desired activity levels.


<br/>
# Results



이 논문에서는 Nucleotide Transformer v3 (NTv3) 모델의 성능을 여러 경쟁 모델과 비교하여 평가하였습니다. NTv3는 다양한 기능적 트랙과 유전자 주석을 예측하는 데 있어 뛰어난 성능을 보였으며, 특히 다음과 같은 주요 결과를 도출하였습니다.

1. **경쟁 모델 비교**: NTv3는 Borzoi, SegmentNT, SpliceAI 등 여러 최신 모델과 비교되었습니다. NTv3는 기능적 트랙 예측에서 Borzoi 모델보다 일관되게 높은 Pearson 상관 계수를 기록하였으며, 특히 ATAC-seq, CAGE, RNA-seq와 같은 실험에서 두드러진 성과를 보였습니다. 예를 들어, NTv3는 RNA-seq 예측에서 Borzoi보다 평균 0.5% 더 높은 성능을 보였습니다.

2. **테스트 데이터**: NTv3는 OpenGenome2 데이터셋을 기반으로 한 다양한 하위 데이터셋에서 평가되었습니다. 이 데이터셋은 128,000종 이상의 유전체로 구성되어 있으며, NTv3는 1Mb의 긴 서열을 처리할 수 있는 능력을 갖추고 있습니다. 각 하위 데이터셋에 대해 NTv3는 고유한 테스트 세트를 사용하여 성능을 평가하였습니다.

3. **메트릭**: NTv3의 성능은 주로 Pearson 상관 계수와 Matthews 상관 계수(MCC)를 사용하여 평가되었습니다. 기능적 트랙 예측의 경우, NTv3는 평균적으로 0.7 이상의 Pearson 상관 계수를 기록하였고, 유전자 주석 예측에서는 MCC가 0.6 이상으로 나타났습니다. 이러한 메트릭은 NTv3가 다양한 생물학적 신호를 정확하게 예측할 수 있음을 보여줍니다.

4. **비교 결과**: NTv3는 특히 긴 서열을 처리하는 데 있어 다른 모델들보다 우수한 성능을 보였습니다. 예를 들어, NTv3는 1Mb의 서열을 처리하면서도 높은 정확도를 유지하였고, 이는 기존의 모델들이 짧은 서열에 최적화되어 있는 것과 대조적입니다. NTv3는 다양한 생물학적 맥락에서의 예측 정확도를 높이기 위해 혼합 길이 훈련을 통해 짧은 서열에 대한 성능 저하를 방지하였습니다.

5. **결론**: NTv3는 기능적 트랙 예측과 유전자 주석 예측 모두에서 경쟁 모델들보다 뛰어난 성능을 보였으며, 특히 다양한 생물 종에 걸쳐 일반화 능력이 뛰어난 것으로 평가되었습니다. 이러한 결과는 NTv3가 유전체 예측 및 설계에 있어 강력한 기초 모델로 자리 잡을 수 있음을 시사합니다.

---




In this paper, the performance of the Nucleotide Transformer v3 (NTv3) model was evaluated by comparing it with several competitive models. NTv3 demonstrated superior performance in predicting various functional tracks and genome annotations, leading to the following key results:

1. **Comparison with Competitive Models**: NTv3 was compared against several state-of-the-art models, including Borzoi, SegmentNT, and SpliceAI. NTv3 consistently achieved higher Pearson correlation coefficients in functional track predictions compared to Borzoi, particularly excelling in experiments such as ATAC-seq, CAGE, and RNA-seq. For instance, NTv3 showed an average performance improvement of 0.5% over Borzoi in RNA-seq predictions.

2. **Test Data**: NTv3 was evaluated on various sub-datasets derived from the OpenGenome2 dataset, which comprises genomes from over 128,000 species. NTv3 is capable of processing long sequences of up to 1Mb. Each sub-dataset was assessed using a unique test set to evaluate performance.

3. **Metrics**: The performance of NTv3 was primarily assessed using Pearson correlation coefficients and Matthews correlation coefficients (MCC). In functional track predictions, NTv3 achieved an average Pearson correlation coefficient of over 0.7, while in genome annotation predictions, the MCC was above 0.6. These metrics indicate NTv3's ability to accurately predict diverse biological signals.

4. **Comparison Results**: NTv3 exhibited superior performance in handling long sequences compared to other models. For example, NTv3 maintained high accuracy while processing 1Mb sequences, contrasting with existing models that are optimized for shorter sequences. NTv3 employed mixed-length training to mitigate performance degradation on shorter sequences, ensuring robust performance across various biological contexts.

5. **Conclusion**: NTv3 outperformed competitive models in both functional track prediction and genome annotation prediction, demonstrating exceptional generalization capabilities across diverse species. These results suggest that NTv3 can establish itself as a powerful foundational model for genomic prediction and design.


<br/>
# 예제


이 논문에서는 Nucleotide Transformer v3 (NTv3)라는 모델을 소개하며, 이 모델의 훈련 및 평가 과정에서 사용된 데이터와 작업에 대해 구체적으로 설명합니다.

#### 1. 훈련 데이터와 테스트 데이터

NTv3 모델은 OpenGenome2 데이터셋을 사용하여 훈련되었습니다. 이 데이터셋은 약 128,000종의 유전체에서 수집된 9조 개의 염기쌍으로 구성되어 있습니다. 훈련 데이터는 다음과 같은 방식으로 구성됩니다:

- **훈련 데이터**: 
  - Eukaryotic Genic Windows: 10,485,760 토큰
  - GTDB v220: 18,874,368 토큰
  - IMGPR: 11,534,336 토큰
  - IMGVR: 17,825,792 토큰
  - Metagenomes: 22,020,096 토큰
  - mRNA: 29,360,128 토큰
  - ncRNA: 13,631,488 토큰
  - Organelles: 10,485,760 토큰
  - Transcripts: 23,068,672 토큰

이 데이터는 다양한 길이의 시퀀스를 포함하고 있으며, 각 시퀀스는 1,024bp에서 1,048,576bp까지 다양합니다. 훈련 과정에서 모델은 이 시퀀스들을 사용하여 염기 단위로 예측을 수행합니다.

- **테스트 데이터**: 
  - 테스트 데이터는 훈련 데이터와 동일한 출처에서 가져오지만, 모델이 훈련 중에 보지 못한 새로운 시퀀스들로 구성됩니다. 예를 들어, 인간 유전체의 경우, Borzoi 모델의 테스트 세트를 사용하여 NTv3의 성능을 평가합니다. 이 테스트 세트는 345.58M의 유전자 시퀀스를 포함하고 있습니다.

#### 2. 구체적인 작업(Task)

NTv3는 두 가지 주요 작업을 수행합니다:

- **기능적 추적 예측 (Functional Track Prediction)**: 
  - 이 작업에서는 모델이 RNA-seq, ATAC-seq, ChIP-seq 등의 다양한 기능적 데이터를 기반으로 유전자 발현 및 조절 신호를 예측합니다. 예를 들어, 모델은 특정 유전자에 대한 RNA-seq 데이터를 입력받아 해당 유전자의 발현 수준을 예측합니다. 이 작업은 평균 Pearson 상관계수를 사용하여 평가됩니다.

- **유전체 주석 예측 (Genome Annotation Prediction)**: 
  - 이 작업에서는 모델이 유전자 구조 요소(예: 엑손, 인트론, 스플라이스 자리 등)를 예측합니다. 각 염기 위치에 대해 해당 주석이 존재하는지를 이진 분류로 예측하며, Matthews 상관계수(MCC)를 사용하여 성능을 평가합니다.

이러한 작업들은 NTv3의 성능을 평가하는 데 중요한 역할을 하며, 모델이 다양한 유전체 데이터에서 얼마나 잘 일반화되는지를 보여줍니다.

---




This paper introduces a model called Nucleotide Transformer v3 (NTv3) and provides a detailed explanation of the data and tasks used in the training and evaluation process.

#### 1. Training Data and Test Data

The NTv3 model was trained using the OpenGenome2 dataset, which consists of approximately 128,000 species' genomes and 9 trillion base pairs. The training data is structured as follows:

- **Training Data**: 
  - Eukaryotic Genic Windows: 10,485,760 tokens
  - GTDB v220: 18,874,368 tokens
  - IMGPR: 11,534,336 tokens
  - IMGVR: 17,825,792 tokens
  - Metagenomes: 22,020,096 tokens
  - mRNA: 29,360,128 tokens
  - ncRNA: 13,631,488 tokens
  - Organelles: 10,485,760 tokens
  - Transcripts: 23,068,672 tokens

This data includes sequences of varying lengths, ranging from 1,024bp to 1,048,576bp. During the training process, the model uses these sequences to make predictions at the nucleotide level.

- **Test Data**: 
  - The test data is sourced from the same origins as the training data but consists of new sequences that the model has not seen during training. For example, in the case of the human genome, the test set from the Borzoi model is used to evaluate NTv3's performance. This test set includes 345.58M nucleotide sequences.

#### 2. Specific Tasks

NTv3 performs two main tasks:

- **Functional Track Prediction**: 
  - In this task, the model predicts gene expression and regulatory signals based on various functional data such as RNA-seq, ATAC-seq, and ChIP-seq. For instance, the model takes RNA-seq data for a specific gene as input and predicts the expression level of that gene. This task is evaluated using the mean Pearson correlation coefficient.

- **Genome Annotation Prediction**: 
  - In this task, the model predicts genomic structural elements (e.g., exons, introns, splice sites). It predicts whether annotation labels are present at each nucleotide position using a binary classification approach, and the performance is evaluated using the Matthews correlation coefficient (MCC).

These tasks play a crucial role in assessing NTv3's performance and demonstrate how well the model generalizes across various genomic data.

<br/>
# 요약


이 논문에서는 Nucleotide Transformer v3 (NTv3)라는 다중 종 유전체 모델을 소개하며, 이 모델은 1Mb의 긴 서열을 처리하고 기능적 예측 및 유전체 주석을 동시에 수행할 수 있도록 설계되었습니다. NTv3는 9조 개의 염기쌍으로 사전 훈련되었으며, 24종의 동물 및 식물에서의 기능적 트랙과 주석 데이터로 후속 훈련을 통해 최첨단 성능을 달성했습니다. 실험적으로 생성된 강화자 서열은 의도한 활성 수준을 재현하며, NTv3의 설계 능력을 입증했습니다.

---

This paper introduces the Nucleotide Transformer v3 (NTv3), a multi-species genomic model designed to process long sequences of up to 1Mb while simultaneously performing functional predictions and genome annotations. NTv3 is pre-trained on 9 trillion base pairs and achieves state-of-the-art performance through post-training on functional tracks and annotation data from 24 species of animals and plants. Experimentally validated enhancer sequences generated by NTv3 demonstrate its design capabilities by recapitulating intended activity levels.

<br/>
# 기타
### 기타(다이어그램, 피규어, 테이블, 어펜딕스 등) 결과와 인사이트

#### 1. 다이어그램 및 피규어
- **NTv3 모델 아키텍처 (Supplementary Figure A.1)**: U-Net 스타일의 아키텍처를 채택하여 DNA 시퀀스를 1bp 해상도로 모델링합니다. 이 구조는 입력 시퀀스를 점진적으로 다운샘플링하고, 중앙의 Transformer 타워에서 장거리 상호작용을 모델링한 후, 다시 업샘플링하여 최종 예측을 생성합니다. 이 아키텍처는 효율적인 장기 시퀀스 처리를 가능하게 하여, 다양한 유전자 기능 예측 및 주석 작업에 적합합니다.

- **성능 평가 (Supplementary Figures A.2, A.3, A.4)**: NTv3의 성능은 다양한 시퀀스 길이에서 평가되었으며, 각 서브 데이터셋에 대한 마스킹된 토큰 정확도가 시각화되었습니다. 이 결과는 모델이 다양한 길이의 시퀀스에서 일관된 성능을 유지함을 보여줍니다.

- **포스트 트레이닝 동역학 (Supplementary Figure A.5)**: 포스트 트레이닝 동안 NTv3의 학습 곡선이 제시되며, MLM 손실, 기능 트랙에 대한 평균 Pearson 상관관계, 유전체 주석에 대한 평균 MCC가 시간에 따라 어떻게 변화하는지를 보여줍니다. 이는 모델이 포스트 트레이닝을 통해 성능을 향상시키는 과정을 시각적으로 나타냅니다.

- **유전체 주석 데이터 및 성능 (Supplementary Figure A.7)**: 다양한 종에 대한 주석 데이터의 통계와 NTv3의 성능을 보여줍니다. 이 결과는 NTv3가 다양한 유전체 주석 작업에서 높은 정확도를 달성함을 나타냅니다.

#### 2. 테이블
- **OpenGenome2 서브 데이터셋 (Table 12)**: 각 서브 데이터셋의 토큰 수와 최대 시퀀스 길이를 요약합니다. 이 표는 NTv3의 훈련 및 평가에 사용된 데이터의 다양성과 범위를 보여줍니다.

- **기능 트랙 및 유전체 주석 (Supplementary Tables C.1, C.2)**: 포스트 트레이닝에 사용된 기능 트랙과 유전체 주석 요소의 세부 정보를 제공합니다. 이 표는 각 종에 대한 데이터의 구성과 다양성을 강조합니다.

- **Ntv3 Benchmark (Supplementary Table C.3)**: Ntv3 Benchmark의 작업 세부 정보를 요약합니다. 이 표는 다양한 종에서의 모델 성능을 평가하기 위한 기준을 제공합니다.

#### 3. 어펜딕스
- **모델 최적화 및 스케일링 (B.1, B.2)**: NTv3의 훈련 및 최적화 과정에서의 주요 결정 사항과 실험 결과를 설명합니다. 이 섹션은 모델의 성능을 향상시키기 위한 다양한 최적화 전략을 제시합니다.

- **시퀀스 길이 혼합 (B.3)**: 시퀀스 길이 혼합의 중요성과 그 효과를 설명합니다. 이 섹션은 모델이 다양한 시퀀스 길이에 대해 어떻게 일반화할 수 있는지를 보여줍니다.

- **포스트 트레이닝 데이터 (B.6)**: 포스트 트레이닝에 사용된 데이터의 구성과 그 중요성을 설명합니다. 이 섹션은 모델이 다양한 기능 트랙과 유전체 주석을 학습하는 데 필요한 데이터의 다양성을 강조합니다.



#### 1. Diagrams and Figures
- **NTv3 Model Architecture (Supplementary Figure A.1)**: The architecture adopts a U-Net style for modeling DNA sequences at 1bp resolution. This structure progressively downsamples the input sequence, models long-range interactions in a central Transformer tower, and then upsamples to generate final predictions. This architecture enables efficient long-sequence processing, making it suitable for various gene function prediction and annotation tasks.

- **Performance Evaluation (Supplementary Figures A.2, A.3, A.4)**: The performance of NTv3 is evaluated across different sequence lengths, with masked token accuracy visualized for each sub-dataset. These results demonstrate that the model maintains consistent performance across various sequence lengths.

- **Post-training Dynamics (Supplementary Figure A.5)**: The learning curves of NTv3 during post-training are presented, showing how MLM loss, mean Pearson correlation on functional tracks, and mean MCC on genome annotations evolve over time. This visually represents the model's performance improvement through post-training.

- **Genome Annotation Data and Performance (Supplementary Figure A.7)**: This figure shows statistics of annotation data across species and the performance of NTv3. The results indicate that NTv3 achieves high accuracy across various genome annotation tasks.

#### 2. Tables
- **OpenGenome2 Sub-datasets (Table 12)**: This table summarizes the number of tokens and maximum sequence lengths for each sub-dataset. It highlights the diversity and range of data used for training and evaluation of NTv3.

- **Functional Tracks and Genome Annotations (Supplementary Tables C.1, C.2)**: These tables provide details on the functional tracks and genome annotation elements used in post-training. They emphasize the composition and diversity of data for each species.

- **Ntv3 Benchmark (Supplementary Table C.3)**: This table summarizes the tasks included in the Ntv3 Benchmark. It provides a basis for evaluating model performance across various species.

#### 3. Appendix
- **Model Optimization and Scaling (B.1, B.2)**: This section describes key decisions and experimental results related to the training and optimization of NTv3. It presents various optimization strategies aimed at improving model performance.

- **Sequence Length Mixing (B.3)**: The importance and effects of sequence length mixing are explained. This section shows how the model can maintain robustness across a range of sequence lengths.

- **Post-training Data (B.6)**: This section describes the composition of data used for post-training and its significance. It emphasizes the diversity of data necessary for the model to learn various functional tracks and genome annotations.

<br/>
# refer format:


### BibTeX 
```bibtex
@article{Boshar2025,
  author = {Sam Boshar and Benjamin Evans and Ziqi Tang and Armand Picard and Yanis Adel and Franziska K. Lorbeer and Chandana Rajesh and Tristan Karch and Shawn Sidbon and David Emms and Javier Mendoza-Revilla and Fatimah Al-Ani and Evan Seitz and Yair Schiff and Yohan Bornachot and Ariana Hernandez and Marie Lopez and Alexandre Laterre and Karim Beguir and Peter Koo and Volodymyr Kuleshov and Alexander Stark and Bernardo P. de Almeida and Thomas Pierrot},
  title = {A foundational model for joint sequence-function multi-species modeling at scale for long-range genomic prediction},
  journal = {bioRxiv},
  year = {2025},
  month = {December},
  doi = {10.64898/2025.12.22.695963},
  url = {https://doi.org/10.64898/2025.12.22.695963}
}
```

### 시카고 스타일
Boshar, Sam, Benjamin Evans, Ziqi Tang, Armand Picard, Yanis Adel, Franziska K. Lorbeer, Chandana Rajesh, Tristan Karch, Shawn Sidbon, David Emms, Javier Mendoza-Revilla, Fatimah Al-Ani, Evan Seitz, Yair Schiff, Yohan Bornachot, Ariana Hernandez, Marie Lopez, Alexandre Laterre, Karim Beguir, Peter Koo, Volodymyr Kuleshov, Alexander Stark, Bernardo P. de Almeida, and Thomas Pierrot. 2025. "A Foundational Model for Joint Sequence-Function Multi-Species Modeling at Scale for Long-Range Genomic Prediction." bioRxiv. https://doi.org/10.64898/2025.12.22.695963.
