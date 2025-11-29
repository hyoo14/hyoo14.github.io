---
layout: post
title:  "[2024]Semi-supervised learning with pseudo-labeling compares favorably with large language models for regulatory sequence prediction"
date:   2025-11-29 01:27:50 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 크로스-스피시스 의사 라벨링을 기반으로 한 반지도 학습(SSL) 방법을 제안하여, 제한된 라벨 데이터로부터 모델을 효과적으로 훈련할 수 있음을 보여주었다. 
기존 경쟁 모델과 유사하거나 더 나은 성능 보임..  


짧은 요약(Abstract) :



이 논문에서는 비코딩 단일 뉴클레오타이드 다형성(SNP)을 예측하기 위해 심층 학습을 활용하는 새로운 접근 방식을 제안합니다. 기존의 심층 학습 방법은 기능적 데이터와 연관된 DNA 서열이 필요하지만, 이러한 데이터는 인간 유전체의 한정된 크기로 인해 매우 부족합니다. 반면, 포유류 DNA 서열은 대규모 시퀀싱 프로젝트 덕분에 기하급수적으로 증가하고 있지만, 대부분의 경우 기능적 데이터가 없습니다. 이를 해결하기 위해 저자들은 라벨이 없는 DNA 서열을 활용할 수 있는 새로운 반지도 학습(Semi-supervised Learning, SSL) 방법을 제안합니다. 이 방법은 노이즈 학생(Noisy Student) 알고리즘의 원리를 통합하여, 매우 적은 훈련 데이터로도 전사 인자(transcription factor)의 예측 성능을 향상시킵니다. 이 접근 방식은 유연성이 뛰어나며, 최신 모델을 포함한 모든 신경망 아키텍처를 훈련하는 데 사용할 수 있으며, 표준 감독 학습에 비해 강력한 예측 성능 향상을 보여줍니다. 또한, SSL로 훈련된 소형 모델은 대형 언어 모델인 DNABERT2와 유사하거나 더 나은 성능을 보였습니다.




This paper proposes a novel approach to predicting non-coding single nucleotide polymorphisms (SNPs) using deep learning. Most existing deep learning methods rely on supervised learning, which requires DNA sequences associated with functional data, but such data is severely limited by the finite size of the human genome. In contrast, the amount of mammalian DNA sequences is growing exponentially due to large-scale sequencing projects, but in most cases, without functional data. To address this limitation, the authors propose a new semi-supervised learning (SSL) method that exploits unlabeled DNA sequences. This method incorporates principles from the Noisy Student algorithm to enhance predictions for transcription factors with very few binding sites (very small training data). The approach is highly flexible and can be used to train any neural architecture, including state-of-the-art models, and shows strong predictive performance improvements compared to standard supervised learning. Moreover, small models trained by SSL demonstrated similar or better performance than the large language model DNABERT2.


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



이 논문에서는 비지도 학습의 한 형태인 반지도 학습(Semi-Supervised Learning, SSL)을 통해 DNA 서열의 예측 성능을 향상시키기 위한 새로운 방법을 제안합니다. 이 방법은 주로 두 가지 단계로 구성됩니다: **사전 훈련(pre-training)**과 **미세 조정(fine-tuning)**입니다.

1. **사전 훈련**: 
   - 연구자들은 먼저 라벨이 있는 데이터(예: 인간 유전체의 ChIP-seq 피크)를 사용하여 모델을 훈련합니다. 그러나 라벨이 있는 데이터는 제한적이기 때문에, 연구팀은 **교차 종 유사성(cross-species homology)**을 활용하여 라벨이 없는 데이터에서 유사한 서열을 찾아냅니다. 이를 통해, 인간 유전체에서 라벨이 있는 서열과 유사한 서열을 다른 포유류 유전체에서 찾아내어 **의사 라벨링(pseudo-labeling)**을 수행합니다. 
   - 이 과정에서 UCSC liftover 프로그램을 사용하여 인간 유전체의 피크 좌표를 다른 유전체로 변환하고, 유사한 서열을 추출하여 라벨이 없는 데이터의 양을 크게 증가시킵니다. 이로 인해 모델은 수량적으로 훨씬 더 많은 데이터로 사전 훈련을 받을 수 있습니다.

2. **미세 조정**: 
   - 사전 훈련이 완료된 후, 모델은 원래의 라벨이 있는 데이터로 미세 조정됩니다. 이 단계에서는 라벨이 있는 데이터와 라벨이 없는 데이터에서 얻은 의사 라벨링 데이터를 혼합하여 모델을 훈련합니다. 
   - 이 과정에서 **Noisy Student 알고리즘**의 원리를 도입하여 의사 라벨링 데이터의 신뢰도를 예측하고, 이를 통해 모델의 성능을 더욱 향상시킵니다. 이 알고리즘은 신뢰도가 높은 의사 라벨링 데이터에 더 높은 가중치를 부여하고, 가우시안 노이즈를 추가하여 모델의 결정 경계를 부드럽게 만듭니다.

3. **모델 아키텍처**: 
   - 연구팀은 다양한 모델 아키텍처를 사용하여 SSL을 적용했습니다. 여기에는 **DeepBind**(얕은 CNN), **DeepSea**(깊은 CNN), 그리고 **DNABERT2**(대형 언어 모델)가 포함됩니다. 각 모델은 SSL을 통해 훈련된 후, 다양한 실험 데이터에 대해 성능을 비교하였습니다.

이러한 방법론을 통해, 연구팀은 특정 전사 인자(TF)와 같은 데이터가 적은 경우에도 모델의 예측 성능을 크게 향상시킬 수 있음을 입증하였습니다. 특히, SSL을 통해 훈련된 간단한 CNN 모델이 대형 언어 모델인 DNABERT2와 유사하거나 더 나은 성능을 보이는 경우도 있었습니다.





In this paper, the authors propose a novel method to enhance the predictive performance of DNA sequence analysis using a form of semi-supervised learning (SSL). This method primarily consists of two stages: **pre-training** and **fine-tuning**.

1. **Pre-training**:
   - The researchers first train the model using labeled data (e.g., ChIP-seq peaks from the human genome). However, since labeled data is limited, they utilize **cross-species homology** to identify similar sequences from unlabeled data. This involves performing **pseudo-labeling** by finding homologous sequences in other mammalian genomes that correspond to the labeled sequences in the human genome.
   - The UCSC liftover program is used to convert peak coordinates from the labeled genome to other genomes, extracting homologous sequences and significantly increasing the amount of unlabeled data. This allows the model to be pre-trained on a much larger dataset.

2. **Fine-tuning**:
   - After pre-training, the model is fine-tuned using the original labeled data. In this stage, a mix of labeled data and the pseudo-labeled data obtained from the unlabeled sequences is used to train the model.
   - The authors incorporate principles from the **Noisy Student algorithm** to predict the confidence in the pseudo-labeled data, further enhancing the model's performance. This algorithm assigns higher weights to highly confident pseudo-labeled data and adds Gaussian noise to smooth the model's decision boundary.

3. **Model Architecture**:
   - The research team applied SSL to various model architectures, including **DeepBind** (shallow CNN), **DeepSea** (deep CNN), and **DNABERT2** (large language model). Each model was trained using SSL and compared for performance across different experimental datasets.

Through this methodology, the authors demonstrated that even in cases with limited data, such as for specific transcription factors (TFs), the predictive performance of the models could be significantly improved. Notably, simple CNN models trained with SSL outperformed or matched the performance of the large language model DNABERT2 in certain scenarios.


<br/>
# Results




이 연구에서는 제안된 반지도 학습(Semi-Supervised Learning, SSL) 방법이 여러 경쟁 모델과 비교하여 DNA 서열 예측에서 어떻게 성능을 향상시키는지를 평가했습니다. 연구에 사용된 주요 모델은 DeepBind, DeepSea, 그리고 DNABERT2로, 각각의 모델은 다양한 실험 데이터에 대해 성능을 비교했습니다.

#### 1. 테스트 데이터
테스트 데이터는 여러 전사 인자(Transcription Factors, TFs)의 ChIP-seq 데이터로 구성되었습니다. 사용된 TFs는 다음과 같습니다:
- ATF3
- ETS1
- ANDR
- REST
- MAX
- P300
- RAD21
- CTCF
- H3K4me3
- POL2
- ATAC-seq

각 TF에 대해 양성 시퀀스의 수와 pseudo-labeling을 통해 얻은 양성 시퀀스의 수가 기록되었습니다. 예를 들어, ATF3의 경우 원래 양성 시퀀스는 1,306개였으나, pseudo-labeling을 통해 23,555개로 증가했습니다.

#### 2. 메트릭
모델 성능은 주로 두 가지 메트릭으로 평가되었습니다:
- **AUROC (Area Under the Receiver Operating Characteristic Curve)**: 이 메트릭은 모델의 분류 성능을 평가하는 데 사용됩니다.
- **AUPR (Area Under the Precision-Recall Curve)**: 이 메트릭은 클래스 불균형이 있는 데이터에서 모델의 성능을 더 잘 반영합니다.

#### 3. 성능 비교
- **DeepBind**: SSL을 적용한 DeepBind 모델은 ATF3, REST, P300에 대해 각각 +11.3%, +7.4%, +24.4%의 AUROC 향상을 보였습니다. AUPR에서는 RAD21, CTCF, POL2에서 각각 +48%, +36%, +29.7%의 향상이 있었습니다.
  
- **DeepSea**: DeepSea 모델에서도 SSL을 적용했을 때 REST와 P300에서 각각 +23.8%, +17.5%의 AUROC 향상이 있었고, AUPR에서는 ATAC에서 +42.7%의 향상이 있었습니다.

- **DNABERT2**: DNABERT2는 SSL을 통해 fine-tuning을 진행했으며, ATF3, ETS1, REST와 같은 특정 TF에 대해 AUROC와 AUPR 모두에서 성능이 향상되었습니다. 그러나 다른 데이터에서는 성능이 감소하는 경향을 보였습니다.

#### 4. 결론
결과적으로, SSL 방법은 특히 데이터가 적은 경우(예: 특정 TF의 경우)에서 성능을 크게 향상시켰습니다. 간단한 CNN 모델(27K 파라미터)이 특정 상황에서 DNABERT2(117M 파라미터)와 유사하거나 더 나은 성능을 보였다는 점이 주목할 만합니다. SSL과 Noisy Student 알고리즘을 결합한 SSL-NS는 ATF3와 ETS1에 대한 예측 성능을 더욱 향상시켰습니다.

---




In this study, the proposed Semi-Supervised Learning (SSL) method was evaluated against several competitive models to assess how it enhances performance in DNA sequence prediction. The primary models used in the study were DeepBind, DeepSea, and DNABERT2, and their performance was compared across various experimental datasets.

#### 1. Test Data
The test data consisted of ChIP-seq data for several transcription factors (TFs). The TFs used included:
- ATF3
- ETS1
- ANDR
- REST
- MAX
- P300
- RAD21
- CTCF
- H3K4me3
- POL2
- ATAC-seq

For each TF, the number of positive sequences and the number of positive sequences obtained through pseudo-labeling were recorded. For instance, for ATF3, the original positive sequences were 1,306, but this increased to 23,555 through pseudo-labeling.

#### 2. Metrics
Model performance was primarily evaluated using two metrics:
- **AUROC (Area Under the Receiver Operating Characteristic Curve)**: This metric is used to assess the classification performance of the model.
- **AUPR (Area Under the Precision-Recall Curve)**: This metric better reflects model performance on imbalanced datasets.

#### 3. Performance Comparison
- **DeepBind**: The DeepBind model with SSL showed AUROC improvements of +11.3%, +7.4%, and +24.4% for ATF3, REST, and P300, respectively. In terms of AUPR, improvements of +48%, +36%, and +29.7% were observed for RAD21, CTCF, and POL2.

- **DeepSea**: The DeepSea model also showed AUROC improvements of +23.8% and +17.5% for REST and P300 with SSL, and an AUPR improvement of +42.7% for ATAC.

- **DNABERT2**: DNABERT2 was fine-tuned with SSL, showing performance improvements in both AUROC and AUPR for specific TFs like ATF3, ETS1, and REST. However, for other datasets, a trend of performance decrease was noted.

#### 4. Conclusion
Overall, the SSL method significantly improved performance, especially in cases with limited data (e.g., for specific TFs). Notably, a simple CNN model with 27K parameters outperformed or matched the performance of DNABERT2 with 117M parameters in certain situations. The combination of SSL with the Noisy Student algorithm (SSL-NS) further enhanced predictions for ATF3 and ETS1.


<br/>
# 예제



이 논문에서는 반지도 학습(Semi-Supervised Learning, SSL)과 가짜 레이블링(pseudo-labeling)을 활용하여 DNA 서열의 규제 요소를 예측하는 방법을 제안합니다. 이 방법은 특히 데이터가 부족한 경우에 효과적입니다. 다음은 트레이닝 데이터와 테스트 데이터의 구체적인 예시입니다.

#### 트레이닝 데이터
1. **입력 데이터**: 
   - DNA 서열: 예를 들어, 특정 전사 인자(Transcription Factor, TF)인 ATF3가 결합하는 200bp 길이의 DNA 서열.
   - 레이블: 해당 서열이 ATF3에 의해 결합되는지 여부 (결합됨: 1, 결합되지 않음: 0).

2. **가짜 레이블링**:
   - 인간 유전체에서 ATF3가 결합하는 서열을 기반으로, 유사한 서열을 가진 다른 포유류 유전체(예: 침팬지, 개)에서 가짜 레이블을 생성합니다. 이 과정에서, 인간 유전체의 레이블이 있는 서열을 다른 유전체에 "전이"하여 해당 서열에 대한 레이블을 부여합니다.

3. **데이터 증강**:
   - 예를 들어, ATF3가 결합하는 1306개의 서열이 있을 때, 가짜 레이블링을 통해 23,555개의 서열로 데이터가 증가합니다. 이는 약 18배의 데이터 증강을 의미합니다.

#### 테스트 데이터
1. **입력 데이터**: 
   - 테스트를 위해, 모델이 학습한 DNA 서열과 동일한 형식의 새로운 DNA 서열을 사용합니다. 이 서열은 ATF3가 결합하는지 여부를 예측하기 위해 사용됩니다.

2. **출력 데이터**:
   - 모델의 예측 결과: 각 서열에 대해 ATF3가 결합할 확률을 출력합니다. 예를 들어, 특정 서열에 대해 0.85의 확률을 출력하면, 이는 해당 서열이 ATF3에 의해 결합될 가능성이 85%임을 의미합니다.

3. **성능 평가**:
   - 모델의 성능은 AUPR(Precision-Recall Curve의 면적)과 AUROC(Receiver Operating Characteristic Curve의 면적)를 통해 평가됩니다. 예를 들어, ATF3에 대한 AUPR이 0.149로 나타나면, 이는 모델이 ATF3의 결합을 예측하는 데 있어 상당한 성능을 보였음을 나타냅니다.

이러한 방식으로, SSL과 가짜 레이블링을 통해 모델은 적은 양의 레이블이 있는 데이터로도 높은 성능을 발휘할 수 있습니다.

---




This paper proposes a method for predicting regulatory elements of DNA sequences using Semi-Supervised Learning (SSL) and pseudo-labeling. This method is particularly effective when data is scarce. Below is a detailed example of training and testing data.

#### Training Data
1. **Input Data**: 
   - DNA Sequence: For example, a 200bp DNA sequence where a specific transcription factor (TF) called ATF3 binds.
   - Labels: Whether the sequence is bound by ATF3 (bound: 1, not bound: 0).

2. **Pseudo-Labeling**:
   - Based on the sequences where ATF3 binds in the human genome, homologous sequences in other mammalian genomes (e.g., chimpanzee, dog) are used to generate pseudo-labels. In this process, labels from human genome sequences are "transferred" to other genomes.

3. **Data Augmentation**:
   - For instance, if there are 1306 sequences bound by ATF3, pseudo-labeling can increase this to 23,555 sequences. This represents an approximately 18-fold data augmentation.

#### Testing Data
1. **Input Data**: 
   - For testing, new DNA sequences in the same format as those used for training are employed. These sequences are used to predict whether ATF3 binds.

2. **Output Data**:
   - Model predictions: The model outputs the probability that ATF3 binds to each sequence. For example, if a specific sequence outputs a probability of 0.85, it indicates an 85% likelihood that ATF3 will bind to that sequence.

3. **Performance Evaluation**:
   - The model's performance is evaluated using AUPR (Area Under the Precision-Recall Curve) and AUROC (Area Under the Receiver Operating Characteristic Curve). For example, if the AUPR for ATF3 is 0.149, it indicates that the model has demonstrated significant performance in predicting ATF3 binding.

In this way, through SSL and pseudo-labeling, the model can achieve high performance even with a limited amount of labeled data.

<br/>
# 요약


이 논문에서는 크로스-스피시스 의사 라벨링을 기반으로 한 반지도 학습(SSL) 방법을 제안하여, 제한된 라벨 데이터로부터 모델을 효과적으로 훈련할 수 있음을 보여주었다. 실험 결과, SSL을 적용한 모델이 전통적인 감독 학습(SL)보다 특정 전사 인자(TF)에 대해 성능이 크게 향상되었으며, 특히 데이터가 적은 경우에 효과적이었다. 예를 들어, ATF3와 REST의 경우 SSL을 통해 AUPR이 각각 2323.8%와 794.2% 증가하였다.




This paper proposes a semi-supervised learning (SSL) method based on cross-species pseudo-labeling, demonstrating effective model training from limited labeled data. Experimental results show that models using SSL significantly outperform traditional supervised learning (SL), especially for specific transcription factors (TFs) when data is scarce. For instance, the AUPR increased by 2323.8% for ATF3 and 794.2% for REST with SSL.

<br/>
# 기타

### 결과 및 인사이트 요약

#### 다이어그램 및 피규어
1. **Figure 1**: 데이터 의사 라벨링 및 SSL 프로세스
   - 이 다이어그램은 크로스-스피시스 의사 라벨링을 통해 어떻게 원래의 라벨이 있는 데이터의 양을 증가시킬 수 있는지를 보여줍니다. 의사 라벨링된 데이터는 모델의 사전 훈련에 사용되며, 이후 원래의 라벨이 있는 데이터로 미세 조정됩니다. 이 과정은 모델의 초기 매개변수를 더 잘 설정하는 데 기여합니다.

2. **Figure 2**: AUPR 증가 비율
   - 이 피규어는 SSL을 사용한 경우와 사용하지 않은 경우의 AUPR(정밀-재현율 곡선 아래 면적) 증가를 비교합니다. 특정 전사 인자(TF)에서 SSL이 큰 성능 향상을 가져왔음을 보여줍니다. 예를 들어, ATF3의 경우 AUPR이 2323.8% 증가했습니다.

3. **Figure 3**: SNP 효과 예측
   - 이 피규어는 CTCF와 ANDR의 SNP 효과 예측 결과를 보여줍니다. SSL을 사용한 모델이 SL을 사용한 모델보다 더 높은 상관관계를 보였으며, 특히 ANDR의 경우 SL에서는 부정적인 상관관계를 보였으나 SSL에서는 긍정적인 상관관계를 보였습니다.

#### 테이블
1. **Table 1**: 의사 라벨링된 긍정 시퀀스 수
   - 이 테이블은 각 데이터셋에 대해 의사 라벨링을 통해 얻은 긍정 시퀀스의 수를 보여줍니다. 예를 들어, ATF3의 경우 원래 1306개의 긍정 시퀀스가 23,555개로 증가했습니다. 이는 데이터의 양을 크게 증가시켜 모델 훈련에 유리한 환경을 제공합니다.

2. **Table 2**: CNN과 DNABERT2의 AUPR 비교
   - 이 테이블은 SSL을 사용한 간단한 CNN 모델과 대형 언어 모델인 DNABERT2의 AUPR을 비교합니다. CNN-27K-SSL 모델이 특정 데이터셋에서 DNABERT2와 유사하거나 더 나은 성능을 보였음을 나타냅니다. 특히 ATF3의 경우 AUPR이 0.176으로, CNN-27K-SSL이 SL을 사용한 CNN보다 8배 높은 성능을 보였습니다.

#### 어펜딕스
- 어펜딕스에는 추가적인 데이터와 실험 결과가 포함되어 있으며, SSL의 효과를 뒷받침하는 다양한 실험 결과와 통계적 분석이 제공됩니다. 이 데이터는 SSL이 특정 TF에 대해 어떻게 성능을 향상시키는지를 보여주는 중요한 증거로 작용합니다.

---

### Summary of Results and Insights

#### Diagrams and Figures
1. **Figure 1**: Data Pseudo-Labeling and SSL Process
   - This diagram illustrates how cross-species pseudo-labeling can increase the amount of available labeled data. The pseudo-labeled data is used for model pre-training, followed by fine-tuning on the original labeled data. This process contributes to better initialization of the model's parameters.

2. **Figure 2**: AUPR Increase Percentage
   - This figure compares the increase in AUPR (Area Under the Precision-Recall Curve) between cases with and without SSL. It shows significant performance improvements with SSL, particularly for specific transcription factors (TFs). For instance, ATF3 saw a 2323.8% increase in AUPR.

3. **Figure 3**: SNP Effect Prediction
   - This figure presents the results of SNP effect predictions for CTCF and ANDR. Models using SSL showed higher correlations than those using SL, with ANDR showing a positive correlation with SSL, whereas SL resulted in a negative correlation.

#### Tables
1. **Table 1**: Number of Pseudo-Labeled Positive Sequences
   - This table shows the number of positive sequences obtained through pseudo-labeling for each dataset. For example, ATF3 increased from 1306 positive sequences to 23,555, significantly enhancing the data available for model training.

2. **Table 2**: AUPR Comparison of CNN and DNABERT2
   - This table compares the AUPR of simple CNN models trained with SSL against the large language model DNABERT2. The CNN-27K-SSL model performed similarly or better than DNABERT2 on certain datasets, with ATF3 achieving an AUPR of 0.176, which is 8 times higher than the CNN-27K with SL.

#### Appendix
- The appendix includes additional data and experimental results, providing statistical analyses that support the effectiveness of SSL. This data serves as crucial evidence for how SSL enhances performance, particularly for specific TFs.

<br/>
# refer format:
### BibTeX 

```bibtex
@article{Phan2024,
  author = {Han Phan and Céline Brouard and Raphaël Mourad},
  title = {Semi-supervised learning with pseudo-labeling compares favorably with large language models for regulatory sequence prediction},
  journal = {Briefings in Bioinformatics},
  year = {2024},
  volume = {25},
  number = {6},
  pages = {bbae560},
  doi = {10.1093/bib/bbae560},
  publisher = {Oxford University Press},
  note = {Open Access article distributed under the terms of the Creative Commons Attribution License (https://creativecommons.org/licenses/by/4.0/)}
}
```

### 시카고 스타일

Phan, Han, Céline Brouard, and Raphaël Mourad. "Semi-supervised Learning with Pseudo-labeling Compares Favorably with Large Language Models for Regulatory Sequence Prediction." *Briefings in Bioinformatics* 25, no. 6 (2024): bbae560. https://doi.org/10.1093/bib/bbae560. Open Access article distributed under the terms of the Creative Commons Attribution License (https://creativecommons.org/licenses/by/4.0/).
