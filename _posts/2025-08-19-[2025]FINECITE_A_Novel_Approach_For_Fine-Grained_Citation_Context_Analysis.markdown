---
layout: post
title:  "[2025]FINECITE: A Novel Approach For Fine-Grained Citation Context Analysis"
date:   2025-08-19 21:57:05 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 FINECITE라는 새로운 데이터셋을 구축하여 세 가지 차원(정보, 인식, 배경)에 기반한 세분화된 인용 맥락 정의를 제안(인용 문맥은 인용된 문서에서 사용된 정보, 저자가 인용된 정보를 어떻게 사용했는지, 그리고 인용이 포함된 이유를 설명하는 내용을 포함)    


짧은 요약(Abstract) :

이 논문은 인용 맥락 분석(Citation Context Analysis, CCA)에 대한 새로운 접근법인 FINECITE를 제안합니다. 기존의 CCA 연구는 개별 인용에 기능이나 의도 레이블을 할당하는 데 집중해왔지만, 인용 맥락 자체에 대한 연구는 상대적으로 부족했습니다. 이로 인해 인용 맥락의 정의가 모호하고 제한적이었습니다. 본 연구에서는 기존 연구의 맥락 개념화를 분석하고, 인용 텍스트의 의미적 특성을 기반으로 한 포괄적인 맥락 정의를 제시합니다. 이를 평가하기 위해 1,056개의 수동으로 주석이 달린 인용 맥락을 포함하는 FINECITE 코퍼스를 구축하고 공개했습니다. 실험 결과, 제안한 세밀한 맥락 정의가 기존의 최첨단 접근법에 비해 최대 25%의 성능 향상을 보여주었습니다. 연구 결과와 데이터는 공개적으로 제공됩니다.


This paper presents a novel approach to citation context analysis (CCA) called FINECITE. While previous CCA research has focused on assigning function or intent labels to individual citations, the citation context itself has received relatively little attention. This neglect has led to vague definitions and restrictive assumptions regarding citation contexts. In this study, we analyze the context conceptualizations from prior works and propose a comprehensive context definition based on the semantic properties of the citing text. To evaluate this definition, we construct and publish the FINECITE corpus, which contains 1,056 manually annotated citation contexts. Our experiments demonstrate that the proposed fine-grained context definition shows improvements of up to 25% compared to state-of-the-art approaches. We make our findings and data publicly available.


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



이 논문에서는 인용 맥락 분석(Citation Context Analysis, CCA)을 위한 새로운 접근 방식인 FINECITE를 제안합니다. FINECITE는 인용 문헌의 맥락을 보다 세밀하게 분석하기 위해 세 가지 주요 차원(정보 차원, 인식 차원, 배경 차원)을 정의합니다. 이러한 차원은 인용된 문헌에서 어떤 정보가 인용되었는지, 인용된 정보를 어떻게 인식하고 사용하는지, 그리고 인용의 이유를 설명하는 데 도움을 줍니다.

#### 모델 및 아키텍처
FINECITE의 핵심은 SCIBERT라는 BERT 기반의 모델을 사용하는 것입니다. SCIBERT는 과학 문헌에 특화된 사전 훈련된 언어 모델로, 인용 맥락을 추출하고 분류하는 데 사용됩니다. 이 모델은 인용 문헌의 맥락을 추출하기 위해 다양한 분류 헤드를 사용하여 훈련됩니다. 여기에는 선형 분류기, Bi-LSTM, 조건부 랜덤 필드(CRF) 분류기가 포함됩니다.

#### 데이터셋
FINECITE는 ACL Anthology Network Corpus에서 추출한 1,056개의 수동 주석이 달린 인용 맥락으로 구성된 데이터셋입니다. 이 데이터셋은 인용 문헌의 맥락을 세밀하게 분석하기 위해 설계되었으며, 각 인용 문헌에 대해 세 가지 차원(정보, 인식, 배경)을 기반으로 주석이 달려 있습니다.

#### 훈련 기법
모델 훈련은 AdamW 옵티마이저를 사용하여 진행되며, 조기 중단과 가중치 교차 엔트로피 손실을 적용하여 클래스 불균형 문제를 해결합니다. 훈련 과정에서 다양한 하이퍼파라미터(배치 크기, 학습률, 드롭아웃 비율)를 조정하여 최적의 성능을 달성합니다.

#### 평가
FINECITE의 성능은 기존의 CCA 벤치마크와 비교하여 평가되며, 인용 맥락의 세밀한 정의가 기존 방법들보다 최대 25%의 성능 향상을 가져온 것으로 나타났습니다. 이 연구는 인용 맥락의 중요성을 강조하며, 향후 연구에서 더 많은 과학적 텍스트와 도메인으로 데이터셋을 확장할 계획입니다.

---




This paper proposes a novel approach for Citation Context Analysis (CCA) called FINECITE. FINECITE aims to analyze the context of citations in a more fine-grained manner by defining three main dimensions: Information Dimension, Perception Dimension, and Background Dimension. These dimensions help to clarify what information is cited from the referenced work, how the cited information is perceived and used, and the reasons for including the citation.

#### Model and Architecture
The core of FINECITE is the use of SCIBERT, a BERT-based model specifically designed for scientific literature. SCIBERT is employed to extract and classify citation contexts. Various classification heads are utilized during training, including a linear classifier, Bi-LSTM, and Conditional Random Field (CRF) classifier.

#### Dataset
FINECITE comprises a dataset of 1,056 manually annotated citation contexts extracted from the ACL Anthology Network Corpus. This dataset is designed to facilitate a fine-grained analysis of citation contexts, with annotations based on the three dimensions (Information, Perception, Background) for each citation.

#### Training Techniques
Model training is conducted using the AdamW optimizer, with early stopping and weighted cross-entropy loss applied to address class imbalance issues. Various hyperparameters (batch size, learning rate, dropout rate) are adjusted during the training process to achieve optimal performance.

#### Evaluation
The performance of FINECITE is evaluated against existing CCA benchmarks, demonstrating improvements of up to 25% compared to state-of-the-art methods due to the fine-grained definition of citation contexts. This research emphasizes the importance of citation context and plans to expand the dataset to include a wider range of scientific texts and domains in future work.


<br/>
# Results



이 논문에서는 FINECITE라는 새로운 데이터셋과 함께 세밀한 인용 맥락 정의를 제안하고, 이를 통해 인용 맥락 분석(Citation Context Analysis, CCA)에서의 성능 향상을 입증하였다. 연구자들은 SCIBERT 모델을 사용하여 인용 맥락 추출 및 분류 작업을 수행하였으며, 여러 경쟁 모델과 비교하여 성능을 평가하였다.

#### 경쟁 모델
1. **SCAFFOLDS**: 인용 의도 분류를 위한 구조적 접근법으로, 평균 매크로 F1 점수는 0.453이었다.
2. **SCIBERT**: 인용 문맥 분류를 위한 BERT 기반 모델로, 평균 매크로 F1 점수는 0.546을 기록하였다.
3. **GPT-4o**: 제로샷 설정에서 평가된 대형 언어 모델로, 평균 매크로 F1 점수는 0.430이었다.
4. **SCITULU 70B**: 과학 문헌을 위한 지침 조정된 LLM으로, 평균 매크로 F1 점수는 0.405였다.

#### 테스트 데이터
FINECITE 모델은 다음의 네 가지 CCA 벤치마크 데이터셋에서 평가되었다:
- **ACL-ARC**: 1,933개의 레이블이 있는 인용으로, 6개의 레이블 분류 체계를 따름.
- **ACT2**: 4,000개의 레이블이 있는 혼합 도메인 데이터셋.
- **SCICITE**: 11,000개의 샘플로 구성된 다중 도메인 데이터셋으로, 간단한 3개 클래스 레이블 체계를 사용.
- **MULTI CITE**: 12,653개의 레이블이 있는 다중 문장, 다중 레이블 데이터셋.

#### 메트릭
모델의 성능은 매크로 F1 점수로 평가되었으며, 이는 각 클래스의 F1 점수를 평균한 값이다. FINECITE 모델은 다음과 같은 성과를 보였다:
- **SCIBERT (Linear)**: 평균 매크로 F1 점수 0.579.
- **SCIBERT (BiLSTM)**: 평균 매크로 F1 점수 0.578.
- **SCIBERT (CRF)**: 평균 매크로 F1 점수 0.571.

FINECITE 모델은 모든 데이터셋에서 경쟁 모델보다 우수한 성능을 보였으며, 특히 ACT2 데이터셋에서는 25%의 성능 향상을 기록하였다. ACL-ARC 데이터셋에서는 13%의 성능 향상이 있었다. MULTI CITE 데이터셋에서는 이미 동적 맥락을 제공하기 때문에 성능 향상이 제한적이었다.

#### 비교
FINECITE 모델은 기존의 단순한 인용 문장 모델링을 넘어, 세밀한 인용 맥락을 정의함으로써 인용 문맥의 풍부한 정보를 포착할 수 있었다. 이로 인해 인용 분류 작업에서 더 나은 성능을 발휘할 수 있었으며, 이는 과학적 논증 탐색의 상호작용적 개선에 기여할 것으로 기대된다.

---




In this paper, the authors introduced a novel dataset called FINECITE and proposed a fine-grained definition of citation contexts, demonstrating performance improvements in Citation Context Analysis (CCA). The researchers utilized the SCIBERT model to perform citation context extraction and classification tasks, evaluating performance against several competitive models.

#### Competitive Models
1. **SCAFFOLDS**: A structural approach for citation intent classification, achieving an average macro F1 score of 0.453.
2. **SCIBERT**: A BERT-based model for citation context classification, recording an average macro F1 score of 0.546.
3. **GPT-4o**: A large language model evaluated in a zero-shot setting, with an average macro F1 score of 0.430.
4. **SCITULU 70B**: An instruction-tuned LLM for scientific literature, achieving an average macro F1 score of 0.405.

#### Test Data
The FINECITE model was evaluated on four CCA benchmark datasets:
- **ACL-ARC**: Comprising 1,933 labeled citations following a six-label classification schema.
- **ACT2**: A mixed-domain dataset with 4,000 labeled citations.
- **SCICITE**: A multi-domain dataset containing 11,000 samples annotated with a simple three-class schema.
- **MULTI CITE**: The largest dataset with 12,653 annotated citations, designed for multi-sentence, multi-label classification.

#### Metrics
Model performance was evaluated using the macro F1 score, which is the average of the F1 scores for each class. The FINECITE model achieved the following results:
- **SCIBERT (Linear)**: Average macro F1 score of 0.579.
- **SCIBERT (BiLSTM)**: Average macro F1 score of 0.578.
- **SCIBERT (CRF)**: Average macro F1 score of 0.571.

The FINECITE model outperformed all competitive models across all datasets, with a notable 25% performance increase on the ACT2 dataset and a 13% increase on the ACL-ARC dataset. The performance increase on the MULTI CITE dataset was limited due to its already dynamic context.

#### Comparison
The FINECITE model's ability to define fine-grained citation contexts allowed it to capture the rich information of citation contexts, moving beyond the simplistic modeling of individual citation sentences. This capability led to improved performance in citation classification tasks, contributing to enhanced interactive exploration of scientific argumentation.


<br/>
# 예제



**FINECITE 데이터셋의 예시 및 작업 설명**

FINECITE 데이터셋은 인용 문맥을 세밀하게 분석하기 위해 구축된 데이터셋으로, 각 인용에 대한 문맥을 수집하고 주석을 달아 구성됩니다. 이 데이터셋은 1,056개의 문단으로 구성되어 있으며, 각 문단에는 하나의 인용 마커가 포함되어 있습니다. 주석 작업은 다음과 같은 단계로 진행됩니다.

1. **데이터 수집**: FINECITE 데이터셋은 ACL Anthology Network Corpus에서 추출된 문서들로 구성됩니다. 이 문서들은 컴퓨터 언어학 분야의 연구 논문들로, 인용이 포함된 문단을 샘플링하여 수집합니다.

2. **주석 가이드라인 작성**: 주석 작업을 위한 가이드라인이 작성됩니다. 이 가이드라인은 인용 문맥의 정의와 주석 달기 위한 규칙을 포함하고 있습니다. 주석자는 각 문단을 읽고, 인용 마커와 관련된 문맥을 주석으로 달아야 합니다.

3. **주석 작업**: 주석자는 각 문단을 읽고, 인용 마커와 관련된 문맥을 주석으로 달아야 합니다. 이때, 인용 문맥은 인용된 문서에서 사용된 정보, 저자가 인용된 정보를 어떻게 사용했는지, 그리고 인용이 포함된 이유를 설명하는 내용을 포함합니다.

4. **주석 검증**: 주석의 품질을 보장하기 위해, 여러 주석자 간의 일치도를 측정합니다. F1 점수와 Cohen의 카파 통계량을 사용하여 주석의 일관성을 평가합니다.

**예시**:
- **입력**: "BERT는 대규모 언어 모델로, [TREF]에서 제안되었습니다. 이 모델은 다양한 자연어 처리 작업에서 우수한 성능을 보여주었습니다."
- **출력**: 
  - **정보 범위 (INF)**: "BERT는 대규모 언어 모델로, [TREF]에서 제안되었습니다."
  - **인식 범위 (PERC)**: "이 모델은 다양한 자연어 처리 작업에서 우수한 성능을 보여주었습니다."
  - **배경 범위 (BACK)**: "이 모델은 다양한 자연어 처리 작업에서 우수한 성능을 보여주었습니다." (이 문장은 인용의 이유를 설명하는 배경 정보로 해석될 수 있습니다.)

이러한 방식으로, FINECITE 데이터셋은 인용 문맥을 세밀하게 분석하고, 각 인용의 기능과 목적을 이해하는 데 도움을 줍니다.

---




**Example and Task Description of the FINECITE Dataset**

The FINECITE dataset is constructed to analyze citation contexts in a fine-grained manner, consisting of 1,056 paragraphs, each containing a single citation marker. The annotation process is carried out in the following steps:

1. **Data Collection**: The FINECITE dataset is built from the ACL Anthology Network Corpus, which includes research papers in the field of computational linguistics. Paragraphs containing citations are sampled for inclusion in the dataset.

2. **Annotation Guidelines Creation**: Guidelines for the annotation task are developed. These guidelines include definitions of citation contexts and rules for annotating them. Annotators are required to read each paragraph and annotate the context related to the citation marker.

3. **Annotation Task**: Annotators read each paragraph and annotate the context related to the citation marker. The citation context includes information used from the cited document, how the author perceives or uses that information, and the reason for including the citation.

4. **Annotation Validation**: To ensure the quality of the annotations, inter-annotator agreement is measured using F1 scores and Cohen's kappa statistics to evaluate the consistency of the annotations.

**Example**:
- **Input**: "BERT is a large language model proposed in [TREF]. This model has shown excellent performance across various natural language processing tasks."
- **Output**: 
  - **Information Scope (INF)**: "BERT is a large language model proposed in [TREF]."
  - **Perception Scope (PERC)**: "This model has shown excellent performance across various natural language processing tasks."
  - **Background Scope (BACK)**: "This model has shown excellent performance across various natural language processing tasks." (This sentence can be interpreted as background information explaining the reason for the citation.)

In this way, the FINECITE dataset aids in the fine-grained analysis of citation contexts, helping to understand the function and purpose of each citation.

<br/>
# 요약

이 논문에서는 FINECITE라는 새로운 데이터셋을 구축하여 세 가지 차원(정보, 인식, 배경)에 기반한 세분화된 인용 맥락 정의를 제안하였다. 실험 결과, 이 정의를 사용한 모델이 기존의 최첨단 방법들보다 최대 25% 향상된 성능을 보였으며, 다양한 인용 분류 벤치마크에서 우수한 결과를 기록하였다. 이 연구는 인용 맥락 분석 분야에서 새로운 연구 방향을 제시하고, 향후 다양한 과학적 텍스트에 대한 적용 가능성을 탐색할 계획이다.

---

In this paper, a new dataset called FINECITE is constructed to propose a fine-grained citation context definition based on three dimensions (Information, Perception, Background). Experimental results show that models using this definition achieved up to 25% improved performance compared to state-of-the-art methods, excelling in various citation classification benchmarks. This research presents a new direction for citation context analysis and plans to explore its applicability to a wider range of scientific texts in the future.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: 다양한 인용 맥락 개념화의 시각적 비교를 제공하며, 기존 연구에서의 인용 맥락 정의의 한계를 강조합니다. 이는 연구자들이 인용 맥락을 보다 포괄적으로 이해하고 정의할 필요성을 보여줍니다.
   - **Figure 2**: FINECITE 데이터셋의 통계적 분석 결과를 시각화합니다. 이 피규어는 맥락 길이의 분포와 레이블 분포를 보여주며, 특히 BACK 차원이 긴 맥락을 차지하고 있음을 나타냅니다. 이는 BACK 차원이 인용의 배경 정보를 제공하는 데 중요한 역할을 한다는 것을 시사합니다.

2. **테이블**
   - **Table 1**: 이전 연구와의 구조적 비교를 통해 FINECITE의 기여를 강조합니다. 이 테이블은 다양한 연구에서 사용된 맥락의 제약 조건과 개념적 제한을 정리하여, FINECITE가 어떻게 더 유연하고 포괄적인 맥락 정의를 제공하는지를 보여줍니다.
   - **Table 2**: 고정 크기 윈도우를 사용한 맥락 제한 실험 결과를 보여줍니다. 이 결과는 고정된 문장 수로 제한된 맥락이 F1 점수와 %-Match에서 상당한 오류를 초래함을 나타내며, 이는 FINECITE의 동적 맥락 정의의 필요성을 강조합니다.
   - **Table 3**: FINECITE 데이터셋에서의 인용 맥락 추출 결과를 보여줍니다. SCIBERT 모델을 사용한 다양한 분류 헤드의 성능을 비교하며, CRF 분류기가 가장 높은 F1 점수를 기록했습니다. 이는 FINECITE의 맥락 정의가 효과적임을 나타냅니다.
   - **Table 4**: 인용 분류 작업에서 FINECITE 접근 방식의 성능을 기존 방법과 비교합니다. FINECITE 모델이 모든 데이터셋에서 기존 방법보다 우수한 성능을 보였으며, 특히 ACT2 데이터셋에서 25%의 성능 향상을 보여줍니다.

3. **어펜딕스**
   - **Appendix A**: 주석 플랫폼에 대한 설명을 제공하며, 주석 작업의 효율성을 높이기 위한 다양한 기능을 설명합니다. 이는 연구자들이 인용 맥락을 보다 정확하게 주석할 수 있도록 돕습니다.
   - **Appendix B**: 주석자 간 합의(IAA) 측정 방법을 설명합니다. F1 점수와 Cohen의 카파를 사용하여 주석의 신뢰성을 평가하며, 이는 데이터셋의 품질을 보장하는 데 중요한 역할을 합니다.
   - **Appendix E**: 주석 가이드라인을 제공하여 주석 작업의 일관성을 높입니다. 이는 주석자들이 인용 맥락을 정의하고 주석하는 데 필요한 명확한 기준을 제공합니다.

---




1. **Diagrams and Figures**
   - **Figure 1**: Provides a visual comparison of different conceptualizations of citation contexts, highlighting the limitations of existing definitions in previous research. This emphasizes the need for researchers to understand and define citation contexts more comprehensively.
   - **Figure 2**: Visualizes the statistical analysis results of the FINECITE dataset, showing the distribution of context lengths and label distributions. It particularly indicates that the BACK dimension occupies longer contexts, suggesting its critical role in providing background information for citations.

2. **Tables**
   - **Table 1**: Highlights the contributions of FINECITE through a structural comparison with previous works. This table summarizes the constraints and conceptual limitations of contexts used in various studies, demonstrating how FINECITE offers a more flexible and comprehensive context definition.
   - **Table 2**: Shows the results of experiments using fixed-size windows for context restrictions. The results indicate that limiting contexts to a fixed number of sentences leads to significant errors in F1 scores and %-Match, underscoring the necessity of FINECITE's dynamic context definition.
   - **Table 3**: Presents the results of citation context extraction on the FINECITE dataset. It compares the performance of various classification heads using the SCIBERT model, with the CRF classifier achieving the highest F1 score, indicating the effectiveness of FINECITE's context definition.
   - **Table 4**: Compares the performance of FINECITE approaches in citation classification tasks against existing methods. The FINECITE models outperformed all baselines across datasets, with a notable 25% performance increase on the ACT2 dataset.

3. **Appendices**
   - **Appendix A**: Describes the annotation platform, detailing various features that enhance the efficiency of the annotation process. This aids researchers in accurately annotating citation contexts.
   - **Appendix B**: Explains the methods for measuring inter-annotator agreement (IAA). It uses F1 scores and Cohen's kappa to assess the reliability of annotations, which is crucial for ensuring the quality of the dataset.
   - **Appendix E**: Provides annotation guidelines to enhance consistency in the annotation process. This offers clear criteria for annotators to define and annotate citation contexts effectively.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{jantsch2025finecite,
  author = {Lasse Jantsch and Dong-Jae Koh and Seonghwan Yoon and Jisu Lee and Anne Lauscher and Young-Kyoon Suh},
  title = {FINECITE: A Novel Approach For Fine-Grained Citation Context Analysis},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2025},
  pages = {24525--24542},
  year = {2025},
  month = {July},
  publisher = {Association for Computational Linguistics},
  address = {South Korea}
}
```

### 시카고 스타일

Lasse Jantsch, Dong-Jae Koh, Seonghwan Yoon, Jisu Lee, Anne Lauscher, and Young-Kyoon Suh. "FINECITE: A Novel Approach For Fine-Grained Citation Context Analysis." In *Findings of the Association for Computational Linguistics: ACL 2025*, 24525–24542. South Korea: Association for Computational Linguistics, July 2025.
