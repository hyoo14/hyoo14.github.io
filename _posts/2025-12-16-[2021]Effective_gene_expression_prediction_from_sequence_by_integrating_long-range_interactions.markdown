---
layout: post
title:  "[2021]Effective gene expression prediction from sequence by integrating long-range interactions"
date:   2025-12-16 21:48:57 -0000
categories: study
---

{% highlight ruby %}

한줄 요약:  DNA 서열로부터 유전자 발현을 예측하는 새로운 딥러닝 아키텍처인 Enformer를 소개  
(장거리 상호작용 정보를 통합할 수 있어, 유전자 발현 예측의 정확성을 크게 향상)  


Enformer는 7개의 합성곱 블록을 사용하여 입력 DNA 서열의 공간 차원을 줄입  
Enformer는 11개의 트랜스포머 블록을 사용하여 장거리 상호작용을 캡처  
마지막으로, Enformer는 두 개의 유기체별 네트워크 헤드로 분기하여 인간과 마우스의 유전자 발현 예측을 수행  


짧은 요약(Abstract) :



이 연구에서는 DNA 서열로부터 유전자 발현을 예측하는 새로운 딥러닝 아키텍처인 Enformer를 소개합니다. Enformer는 최대 100kb 떨어진 장거리 상호작용 정보를 통합할 수 있어, 유전자 발현 예측의 정확성을 크게 향상시킵니다. 이 개선은 자연 유전 변이와 포화 돌연변이의 유전자 발현에 대한 효과 예측을 더 정확하게 만들어 주며, Enformer는 DNA 서열로부터 직접적으로 증강자-프로모터 상호작용을 예측할 수 있습니다. 이러한 발전은 인간 질병 연관성을 더 효과적으로 세밀하게 매핑하고, cis-조절 진화 해석을 위한 프레임워크를 제공할 것으로 기대됩니다.




In this study, we introduce a new deep learning architecture called Enformer that predicts gene expression from DNA sequences. Enformer is capable of integrating long-range interaction information from up to 100 kb away, significantly improving the accuracy of gene expression predictions. This improvement yields more accurate predictions of the effects of natural genetic variants and saturation mutagenesis on gene expression, and Enformer learns to predict enhancer-promoter interactions directly from the DNA sequence. We expect that these advances will enable more effective fine-mapping of human disease associations and provide a framework for interpreting cis-regulatory evolution.


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



**모델 아키텍처: Enformer**

Enformer는 유전자 발현과 크로마틴 상태를 DNA 서열로부터 예측하기 위해 설계된 신경망 아키텍처입니다. 이 모델은 크게 세 가지 부분으로 구성됩니다: 

1. **합성곱 블록**: Enformer는 7개의 합성곱 블록을 사용하여 입력 DNA 서열의 공간 차원을 줄입니다. 이 과정에서 입력 서열의 길이는 196,608 bp에서 1,536으로 축소됩니다. 각 서열 위치 벡터는 128 bp를 나타내며, 이 블록은 인접한 뉴클레오타이드가 함께 작용하는 경향을 반영합니다.

2. **트랜스포머 블록**: Enformer는 11개의 트랜스포머 블록을 사용하여 장거리 상호작용을 캡처합니다. 트랜스포머 블록은 다중 헤드 주의 메커니즘을 통해 입력 서열의 모든 위치 간의 정보를 공유하고, 이를 통해 프로모터와 인핸서 간의 상호작용을 모델링합니다. 이 블록은 상대 위치 인코딩을 사용하여 각 위치 간의 거리에 따라 주의 가중치를 조정합니다.

3. **출력 헤드**: 마지막으로, Enformer는 두 개의 유기체별 네트워크 헤드로 분기하여 인간과 마우스의 유전자 발현 예측을 수행합니다.

**특별한 기법**

Enformer는 기존의 모델인 Basenji2와 비교하여 몇 가지 중요한 개선 사항을 포함하고 있습니다. 첫째, Enformer는 합성곱 대신 트랜스포머 블록을 사용하여 장거리 상호작용을 더 잘 캡처할 수 있습니다. 둘째, 주의 풀링(attention pooling)을 사용하여 입력 서열의 연속적인 청크를 요약합니다. 셋째, 상대 위치 인코딩을 통해 각 위치 간의 상호작용을 더 잘 모델링할 수 있습니다. 이러한 개선 사항 덕분에 Enformer는 100 kb까지의 장거리 인핸서를 효과적으로 통합할 수 있으며, 이는 유전자 발현 예측의 정확성을 크게 향상시킵니다.

**트레이닝 데이터**

Enformer는 인간과 마우스의 유전체를 포함한 대규모 데이터셋에서 훈련되었습니다. 훈련 데이터는 1 Mb 영역으로 나누어져 있으며, 각 영역은 100 kb 이상의 정렬된 서열을 가진 두 영역 간의 연결을 기반으로 구성됩니다. 이 데이터셋은 34,021개의 훈련 샘플, 2,213개의 검증 샘플, 1,937개의 테스트 샘플로 구성되어 있습니다. 각 샘플은 5,313개의 유전자 발현 및 크로마틴 상태 관련 트랙을 포함하고 있습니다.

**훈련 과정**

모델은 Adam 옵티마이저를 사용하여 훈련되며, 150,000 스텝 동안 TPU v3 코어에서 훈련됩니다. 훈련 과정에서 데이터 증강을 통해 입력 서열을 무작위로 이동시키고, 역상보적인 서열로 변환하여 모델의 일반화 능력을 향상시킵니다. 최적의 학습률은 그리드 검색을 통해 결정되며, 훈련 데이터의 평균과 표준편차를 기준으로 피처를 스케일링하여 모델의 성능을 극대화합니다.




**Model Architecture: Enformer**

Enformer is a neural network architecture designed to predict gene expression and chromatin states from DNA sequences. The model consists of three main parts:

1. **Convolutional Blocks**: Enformer uses 7 convolutional blocks to reduce the spatial dimension of the input DNA sequence. This process reduces the length of the input sequence from 196,608 bp to 1,536. Each position vector in the sequence represents 128 bp, and these blocks reflect the tendency of adjacent nucleotides to act together.

2. **Transformer Blocks**: Enformer employs 11 transformer blocks to capture long-range interactions. The transformer blocks utilize a multi-head attention mechanism to share information across all positions in the input sequence, allowing the model to capture interactions between promoters and enhancers. These blocks use relative positional encodings to adjust attention weights based on the distance between positions.

3. **Output Heads**: Finally, Enformer branches into two organism-specific network heads to predict gene expression for humans and mice.

**Special Techniques**

Enformer includes several key improvements compared to the previous model, Basenji2. First, Enformer uses transformer blocks instead of convolutions, allowing it to better capture long-range interactions. Second, it employs attention pooling to summarize contiguous chunks of the input sequence. Third, relative positional encodings enable better modeling of interactions between positions. These enhancements allow Enformer to effectively integrate long-range enhancers up to 100 kb away, significantly improving the accuracy of gene expression predictions.

**Training Data**

Enformer was trained on a large dataset that includes the genomes of humans and mice. The training data is partitioned into 1 Mb regions, with connections established between regions that have more than 100 kb of aligning sequence. This dataset consists of 34,021 training samples, 2,213 validation samples, and 1,937 test samples, each containing 5,313 genomic tracks related to gene expression and chromatin states.

**Training Process**

The model is trained using the Adam optimizer over 150,000 steps on TPU v3 cores. During training, data augmentation is applied by randomly shifting the input sequence and reverse-complementing it to enhance the model's generalization ability. The optimal learning rate is determined through grid search, and features are scaled based on the mean and standard deviation of the training data to maximize model performance.


<br/>
# Results



Enformer 모델은 Basenji2와 비교하여 여러 가지 측면에서 우수한 성능을 보였습니다. 특히, Enformer는 유전자 발현 예측에서 평균 상관계수를 0.81에서 0.85로 증가시켰으며, 이는 실험적 수준의 정확도인 0.94에 비해 약 1/3에 해당하는 성능 향상입니다. 이 성능 향상은 다양한 유전자 발현 데이터셋에서 일관되게 나타났으며, 특히 CAGE(캡 분석 유전자 발현) 데이터에서 두드러졌습니다.

Enformer는 5,313개의 유전자 발현 트랙을 예측하는 데 있어 Basenji2보다 더 높은 Pearson 상관계수를 기록했습니다. 예를 들어, Enformer는 CAGE 실험에서 유전자 발현을 예측할 때 Basenji2보다 평균적으로 더 높은 성능을 보였으며, 이는 다양한 세포 유형과 조직에서의 유전자 발현 특성을 더 잘 포착했습니다.

또한, Enformer는 CRISPRi(유전자 발현 억제 기술) 실험을 통해 검증된 활성화 및 억제 유전자 변형의 효과를 예측하는 데 있어서도 Basenji2보다 더 높은 정확도를 보였습니다. Enformer의 기여 점수는 유전자 발현에 대한 예측을 개선하는 데 중요한 역할을 하였으며, 이는 세포 유형에 따라 다르게 나타나는 enhancer(강화자)와 promoter(촉진자) 간의 상호작용을 더 잘 반영했습니다.

Enformer는 또한 eQTL(유전자 발현 정량적 형질좌위) 데이터에 대한 변이 효과 예측에서도 우수한 성능을 보였습니다. GTEx 프로젝트에서 발견된 eQTL을 기반으로 한 분석에서, Enformer의 예측은 Basenji2보다 더 높은 SLDP(유전자형-표현형 연관성) Z-점수를 기록했습니다. 이는 Enformer가 세포 유형에 맞는 유전자 발현 예측을 더 잘 수행하고 있음을 나타냅니다.

마지막으로, Enformer는 MPRAs(대량 병렬 리포터 분석) 데이터셋을 사용한 변이 효과 예측에서도 뛰어난 성능을 보였으며, 여러 경쟁 모델과 비교했을 때 가장 높은 Pearson 상관계수를 기록했습니다. 이러한 결과들은 Enformer가 유전자 발현 예측 및 변이 효과 예측에서 기존 모델들보다 더 나은 성능을 발휘하고 있음을 보여줍니다.




The Enformer model demonstrated superior performance compared to Basenji2 across various aspects. Notably, Enformer increased the average correlation in gene expression prediction from 0.81 to 0.85, which is approximately one-third of the way toward the experimental-level accuracy of 0.94. This performance improvement was consistently observed across various gene expression datasets, particularly in CAGE (Cap Analysis Gene Expression) data.

Enformer achieved higher Pearson correlation coefficients in predicting 5,313 gene expression tracks compared to Basenji2. For instance, Enformer showed better average performance in predicting gene expression in CAGE experiments, effectively capturing tissue- or cell-type-specific expression characteristics.

Additionally, Enformer outperformed Basenji2 in predicting the effects of genetic variants on gene expression, as validated by CRISPRi (CRISPR interference) experiments. The contribution scores from Enformer played a crucial role in improving predictions of gene expression, reflecting the interactions between enhancers and promoters more accurately, depending on the cell type.

Enformer also excelled in variant effect prediction on eQTL (expression quantitative trait loci) data. In analyses based on eQTLs discovered by the GTEx project, Enformer predictions yielded higher SLDP (signed linkage disequilibrium profile) Z-scores compared to Basenji2. This indicates that Enformer is better at performing cell-type-specific gene expression predictions.

Finally, Enformer showed outstanding performance in predicting variant effects using datasets from massively parallel reporter assays (MPRAs), achieving the highest Pearson correlation coefficients among various competing models. These results demonstrate that Enformer outperforms existing models in both gene expression prediction and variant effect prediction.


<br/>
# 예제



이 논문에서는 Enformer라는 새로운 딥러닝 모델을 개발하여 DNA 서열로부터 유전자 발현과 크로마틴 상태를 예측하는 방법을 제시합니다. Enformer는 특히 장거리 상호작용을 통합할 수 있는 능력을 가지고 있어, 100kb까지의 거리에서 유전자 발현에 영향을 미치는 요소들을 고려할 수 있습니다. 

#### 트레이닝 데이터와 테스트 데이터

1. **트레이닝 데이터**:
   - **인풋**: Enformer는 196,608 bp 길이의 DNA 서열을 입력으로 받습니다. 이 서열은 one-hot 인코딩 방식으로 변환되어 모델에 제공됩니다. 
   - **아웃풋**: 모델은 5,313개의 유전자 발현 및 크로마틴 상태 관련 트랙을 예측합니다. 이 트랙들은 CAGE (Cap Analysis of Gene Expression) 데이터, 히스톤 변형, 전사 인자 결합, DNase 접근성 등 다양한 유전자 발현 관련 정보를 포함합니다.

2. **테스트 데이터**:
   - **인풋**: 테스트 데이터는 트레이닝 데이터와 동일한 형식으로, 196,608 bp 길이의 DNA 서열을 포함합니다.
   - **아웃풋**: 테스트 데이터에 대한 예측 결과는 CAGE 데이터와의 Pearson 상관계수를 통해 평가됩니다. 예를 들어, Enformer는 특정 유전자의 TSS(전사 시작 지점)에서의 CAGE 발현을 예측하며, 이 예측 결과는 실제 측정된 CAGE 발현과 비교됩니다.

#### 구체적인 테스크

- **유전자 발현 예측**: Enformer는 주어진 DNA 서열로부터 유전자 발현 수준을 예측하는 작업을 수행합니다. 이 작업은 다양한 세포 유형에서의 유전자 발현 패턴을 이해하는 데 중요한 역할을 합니다.
- **변이 효과 예측**: Enformer는 유전자 발현에 미치는 유전적 변이의 영향을 예측하는 데도 사용됩니다. 예를 들어, 특정 SNP(단일 염기 다형성)가 유전자 발현에 미치는 영향을 예측하여, 이 변이가 질병과 관련이 있는지를 평가할 수 있습니다.

이러한 방식으로 Enformer는 유전자 발현 예측의 정확성을 크게 향상시키며, 다양한 생물학적 질문에 대한 답을 제공할 수 있는 강력한 도구로 자리잡고 있습니다.

---





This paper presents a new deep learning model called Enformer, which predicts gene expression and chromatin states from DNA sequences. Enformer has the capability to integrate long-range interactions, allowing it to consider elements influencing gene expression from distances of up to 100 kb.

#### Training Data and Test Data

1. **Training Data**:
   - **Input**: Enformer takes as input a DNA sequence of length 196,608 bp, which is converted into a one-hot encoded format before being fed into the model.
   - **Output**: The model predicts 5,313 genomic tracks related to gene expression and chromatin states. These tracks include various gene expression-related information such as CAGE (Cap Analysis of Gene Expression) data, histone modifications, transcription factor binding, and DNase accessibility.

2. **Test Data**:
   - **Input**: The test data is in the same format as the training data, consisting of DNA sequences of length 196,608 bp.
   - **Output**: The predictions for the test data are evaluated using Pearson correlation with CAGE data. For example, Enformer predicts CAGE expression at the TSS (transcription start site) of specific genes, and these predictions are compared to the actual measured CAGE expression.

#### Specific Tasks

- **Gene Expression Prediction**: Enformer performs the task of predicting gene expression levels from given DNA sequences. This task is crucial for understanding gene expression patterns across different cell types.
- **Variant Effect Prediction**: Enformer is also used to predict the effects of genetic variants on gene expression. For instance, it can assess how a specific SNP (single nucleotide polymorphism) influences gene expression and whether this variant is associated with diseases.

In this way, Enformer significantly enhances the accuracy of gene expression predictions and serves as a powerful tool for addressing various biological questions.

<br/>
# 요약



Enformer는 DNA 서열로부터 유전자 발현과 크로마틴 상태를 예측하기 위해 자기 주의(attention) 기반의 신경망 아키텍처를 사용하여, 100kb까지의 장거리 상호작용 정보를 통합하여 예측 정확도를 향상시켰다. 이 모델은 CRISPRi 실험을 통해 검증된 강화제-유전자 쌍을 우선적으로 식별하는 데 있어 Basenji2보다 더 높은 정확도를 보였다. Enformer는 또한 유전자 발현에 대한 변이 효과 예측에서 Basenji2보다 더 나은 성능을 나타내었다.




Enformer utilizes a self-attention-based neural network architecture to predict gene expression and chromatin states from DNA sequences, enhancing prediction accuracy by integrating long-range interactions up to 100kb. The model demonstrated higher accuracy in prioritizing enhancer-gene pairs validated by CRISPRi experiments compared to Basenji2. Additionally, Enformer outperformed Basenji2 in predicting the effects of variants on gene expression.

<br/>
# 기타



#### 1. 다이어그램 및 피규어
- **Enformer 모델 아키텍처 (Extended Data Fig. 1)**: Enformer는 7개의 합성곱 블록과 11개의 트랜스포머 블록으로 구성되어 있으며, Basenji2와 비교하여 더 많은 채널과 긴 입력 시퀀스를 사용하여 성능을 향상시킴. 이 구조는 유전자 발현 예측에서 더 나은 성능을 발휘함.
  
- **유전자 발현 예측 성능 (Fig. 1)**: Enformer는 CAGE 실험에서 유전자 발현 예측의 평균 상관계수를 0.81에서 0.85로 증가시킴. 이는 Basenji2와 비교하여 두 배의 성능 향상을 보여줌.

- **예측 정확도 비교 (Extended Data Fig. 4)**: Enformer는 ExPecto 모델보다 RNA-seq 기반 유전자 발현 예측에서 더 높은 정확도를 보임. 이는 Enformer가 다양한 세포 유형에서의 발현 변화를 더 잘 포착함을 나타냄.

- **CRISPRi 실험 결과 (Fig. 2)**: Enformer의 기여 점수는 CRISPRi 검증된 인핸서와 유전자 쌍을 더 정확하게 우선순위화함. 이는 Enformer가 생물학적으로 관련된 영역을 고려하고 있음을 시사함.

#### 2. 테이블
- **변이 효과 예측 (Fig. 3)**: Enformer는 GTEx eQTL 데이터에서 변이 효과 예측의 정확도를 향상시킴. 이는 Enformer가 세포 유형에 따라 유전자 발현에 미치는 변이의 영향을 더 잘 예측할 수 있음을 나타냄.

- **MPRAs 데이터 (Fig. 4)**: Enformer는 MPRAs를 통한 변이 효과 예측에서 다른 방법들보다 더 높은 상관관계를 보임. 이는 Enformer가 변이의 기능적 영향을 더 잘 포착하고 있음을 시사함.

#### 3. 어펜딕스
- **모델 훈련 및 평가 방법 (Methods)**: Enformer는 인간 및 마우스 유전체의 다양한 데이터셋을 사용하여 훈련됨. 이 과정에서 다양한 하이퍼파라미터 조정과 데이터 증강 기법이 사용되어 모델의 성능을 극대화함.

- **기여 점수 계산 방법 (Methods)**: Enformer는 기여 점수를 계산하기 위해 gradient × input, attention, in silico mutagenesis 방법을 사용하여 인핸서-유전자 쌍의 우선순위를 정함.




#### Summary of Results and Insights

#### 1. Diagrams and Figures
- **Enformer Model Architecture (Extended Data Fig. 1)**: The Enformer consists of 7 convolutional blocks and 11 transformer blocks, utilizing more channels and longer input sequences compared to Basenji2, leading to improved performance in gene expression prediction.

- **Gene Expression Prediction Performance (Fig. 1)**: Enformer increased the average correlation of gene expression predictions in CAGE experiments from 0.81 to 0.85, demonstrating a two-fold performance improvement over Basenji2.

- **Accuracy Comparison (Extended Data Fig. 4)**: Enformer outperformed the ExPecto model in RNA-seq based gene expression predictions, indicating its ability to better capture expression changes across various cell types.

- **CRISPRi Experimental Results (Fig. 2)**: Enformer's contribution scores prioritized validated enhancer-gene pairs more accurately, suggesting that it considers biologically relevant regions in its predictions.

#### 2. Tables
- **Variant Effect Prediction (Fig. 3)**: Enformer improved the accuracy of variant effect predictions on GTEx eQTL data, indicating its capability to better predict the impact of variants on gene expression based on cell type.

- **MPRAs Data (Fig. 4)**: Enformer showed higher correlation in predicting variant effects from saturation mutagenesis compared to other methods, suggesting it captures the functional impact of variants more effectively.

#### 3. Appendix
- **Model Training and Evaluation Methods (Methods)**: Enformer was trained using diverse datasets from human and mouse genomes, employing various hyperparameter tuning and data augmentation techniques to maximize performance.

- **Contribution Score Calculation Methods (Methods)**: Enformer utilized gradient × input, attention, and in silico mutagenesis methods to compute contribution scores for prioritizing enhancer-gene pairs.

<br/>
# refer format:
### BibTeX 

```bibtex
@article{Avsec2021,
  author = {Žiga Avsec and Vikram Agarwal and Daniel Visentin and Joseph R. Ledsam and Agnieszka Grabska-Barwinska and Kyle R. Taylor and Yannis Assael and John Jumper and Pushmeet Kohli and David R. Kelley},
  title = {Effective gene expression prediction from sequence by integrating long-range interactions},
  journal = {Nature Methods},
  volume = {18},
  number = {10},
  pages = {1196--1203},
  year = {2021},
  doi = {10.1038/s41592-021-01252-x},
  publisher = {Nature Publishing Group}
}
```

### 시카고 스타일

Avsec, Žiga, Vikram Agarwal, Daniel Visentin, Joseph R. Ledsam, Agnieszka Grabska-Barwinska, Kyle R. Taylor, Yannis Assael, John Jumper, Pushmeet Kohli, and David R. Kelley. "Effective Gene Expression Prediction from Sequence by Integrating Long-Range Interactions." *Nature Methods* 18, no. 10 (2021): 1196-1203. https://doi.org/10.1038/s41592-021-01252-x.
