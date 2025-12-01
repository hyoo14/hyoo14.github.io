---
layout: post
title:  "[2025]COLMATE: Contrastive Late Interaction and Masked Text for Multimodal Document Retrieval"
date:   2025-12-01 02:02:06 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: COLMATE는 시각적 문서 검색을 위한 새로운 모델로, 세 가지 주요 구성 요소인 Masked OCR Language Modeling, TopKSim, Self-supervised Masked Contrastive Learning을 통합하여 성능을 향상시킵니다.


짧은 요약(Abstract) :


COLMATE는 멀티모달 문서 검색을 위한 모델로, 기존의 텍스트 전용 검색 기법의 한계를 극복하기 위해 설계되었습니다. 이 모델은 OCR 기반의 새로운 사전 훈련 목표, 자기 지도형 마스킹 대조 학습 목표, 그리고 멀티모달 문서 구조와 시각적 특성에 더 적합한 지연 상호작용 점수 계산 메커니즘을 활용합니다. COLMATE는 ViDoRe V2 벤치마크에서 기존 검색 모델보다 3.61% 향상된 성능을 보여주며, 도메인 외 벤치마크에 대한 일반화 능력이 더 뛰어난 것으로 나타났습니다.



COLMATE is a model designed for multimodal document retrieval that addresses the limitations of existing text-only retrieval techniques. This model utilizes a novel OCR-based pretraining objective, a self-supervised masked contrastive learning objective, and a late interaction scoring mechanism that is more relevant to multimodal document structures and visual characteristics. COLMATE demonstrates a 3.61% improvement over existing retrieval models on the ViDoRe V2 benchmark, showcasing stronger generalization to out-of-domain benchmarks.


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



COLMATE는 멀티모달 문서 검색을 위한 모델로, 기존의 방법들이 가진 한계를 극복하기 위해 세 가지 주요 구성 요소를 통합하여 설계되었습니다. 이 모델은 다음과 같은 방법론을 사용합니다:

1. **Masked OCR Language Modeling (MOLM)**: 이 구성 요소는 시각적 토큰 표현을 최적화하기 위해 OCR 기반의 마스킹 언어 모델링을 사용합니다. 입력 문서 이미지에서 OCR 단어 토큰을 추출하고, 이 중 30%를 무작위로 마스킹하여 모델이 시각적 컨텍스트를 기반으로 마스킹된 단어를 예측하도록 훈련합니다. 이를 통해 시각적 표현이 풍부해지고, 멀티모달 검색 성능이 향상됩니다.

2. **TopKSim**: 기존의 MaxSim 메커니즘을 개선한 이 방법은 훈련 중 각 쿼리 토큰에 대해 상위 K개의 유사도 점수를 평균화하여 훈련 노이즈를 줄입니다. 이는 이미지 패치에서 발생할 수 있는 단어의 분할이나 결합 문제를 완화하여, 더 견고한 검색 및 쿼리-이미지 쌍의 매칭을 가능하게 합니다.

3. **Self-supervised Masked Contrastive Learning (MaskedCL)**: 이 구성 요소는 레이블이 없는 데이터에서 효과적으로 학습할 수 있도록 설계된 자기 지도 대조 학습 방법입니다. PDF 문서에서 무작위로 텍스트의 일부를 마스킹하고, 해당 문서 이미지의 일부를 흰색 마스크로 가립니다. 그런 다음, 마스킹된 텍스트 표현과 마스킹된 시각적 표현 간의 대조적 정렬을 수행하여, 레이블이 없는 상황에서도 강력한 크로스 모달 표현 학습을 촉진합니다.

COLMATE는 이러한 세 가지 구성 요소를 통해 기존의 멀티모달 검색 방법들과의 간극을 메우고, 더 강력하고 일반화된 멀티모달 문서 검색 성능을 달성합니다. 실험 결과, COLMATE는 ViDoRe V1 및 V2 벤치마크에서 기존 모델들보다 일관되게 우수한 성능을 보였으며, 특히 보지 못한 도메인에 대한 일반화 능력이 두드러졌습니다.




COLMATE is a model designed for multimodal document retrieval, integrating three key components to overcome the limitations of existing methods. The methodology includes:

1. **Masked OCR Language Modeling (MOLM)**: This component utilizes an OCR-based masked language modeling approach to optimize visual token representations. It extracts OCR word tokens from input document images and randomly masks 30% of them, training the model to predict the masked words based on visual context. This significantly enriches the visual representations and enhances multimodal retrieval performance.

2. **TopKSim**: An improvement over the existing MaxSim mechanism, this method averages the top K similarity scores during training for each query token, reducing training noise. This alleviates issues related to the fragmentation or merging of words that can occur with image patches, enabling more robust retrieval and matching of query-image pairs.

3. **Self-supervised Masked Contrastive Learning (MaskedCL)**: This component is a self-supervised contrastive learning method designed to effectively learn from unlabeled data. It randomly masks portions of text extracted from PDF documents and overlays white masks on corresponding patches of the document images. Then, it performs contrastive alignment between the masked textual representations and masked visual representations, promoting robust cross-modal representation learning even in the absence of labeled data.

Through these three components, COLMATE bridges the gap between existing multimodal retrieval approaches and achieves more robust and generalizable multimodal document retrieval performance. Experimental results demonstrate that COLMATE consistently outperforms existing models on the ViDoRe V1 and V2 benchmarks, with particularly notable generalization capabilities to unseen domains.


<br/>
# Results



COLMATE 모델은 ViDoRe V1 및 V2 벤치마크에서 기존 모델들과 비교하여 성능을 평가하였습니다. ViDoRe V1은 인도메인 데이터셋으로, 10개의 학술 및 실제 데이터셋을 포함하고 있으며, ViDoRe V2는 아웃오브도메인 데이터셋으로, 9개의 다양한 실제 도메인에서 수집된 문서들로 구성되어 있습니다. 두 벤치마크 모두 다국어를 지원하지만, 훈련 데이터는 영어로만 제공됩니다.

#### 성능 비교
1. **ViDoRe V1 (인도메인)**:
   - COLMATE-Pali-3B 모델은 평균 nDCG@5 점수 85.14를 기록하여, 기존의 ColPali-3B 모델(84.93)과 재현 모델(84.68)을 초과했습니다. 이는 COLMATE가 인도메인에서 더 나은 성능을 발휘함을 보여줍니다.

2. **ViDoRe V2 (아웃오브도메인)**:
   - COLMATE는 ViDoRe V2에서 57.61의 nDCG@5 점수를 기록하여, ColPali-3B의 54.60 및 재현 모델의 54.00을 초과했습니다. 이는 COLMATE가 보지 못한 도메인에 대한 일반화 능력이 뛰어남을 나타냅니다.

#### 자가 지도 학습 성능
COLMATE는 자가 지도 학습 방식인 Masked Contrastive Learning (MaskedCL)을 통해도 성능을 평가하였습니다. MaskedCL은 nDCG@5 점수에서 ViDoRe V1에서 74.52, ViDoRe V2에서 41.50을 기록하였으며, 이는 레이블이 없는 데이터셋에서도 경쟁력 있는 성능을 보여줍니다. 특히 AI, 에너지, 정부, 건강과 같은 실제 문서의 하위 집합에서 성능 차이가 좁아지는 경향을 보였습니다.

#### 구성 요소의 기여도
COLMATE의 각 구성 요소에 대한 기여도를 평가하기 위해 세부적인 ablation 연구를 수행하였습니다. TopKSim, Masked OCR Language Modeling (MOLM), 그리고 MaskedCL의 조합이 성능 향상에 기여한 것으로 나타났습니다. 특히, TopKSim은 아웃오브도메인 ViDoRe V2 벤치마크에서 56.41의 점수를 기록하여 MaxSim(54.00)보다 우수한 일반화 성능을 보였습니다.

이러한 결과들은 COLMATE가 기존의 멀티모달 문서 검색 방법의 한계를 극복하고, 다양한 도메인에서 강력한 성능을 발휘할 수 있음을 보여줍니다.

---




The COLMATE model was evaluated against existing models on the ViDoRe V1 and V2 benchmarks. ViDoRe V1 is an in-domain dataset that includes 10 academic and real-world datasets, while ViDoRe V2 is an out-of-domain dataset composed of documents collected from 9 diverse real-world domains. Both benchmarks support multilingual data, but the training data is exclusively in English.

#### Performance Comparison
1. **ViDoRe V1 (In-domain)**:
   - The COLMATE-Pali-3B model achieved an average nDCG@5 score of 85.14, surpassing the existing ColPali-3B model (84.93) and the reproduced model (84.68). This indicates that COLMATE performs better in the in-domain setting.

2. **ViDoRe V2 (Out-of-domain)**:
   - COLMATE recorded a score of 57.61 on ViDoRe V2, exceeding ColPali-3B's score of 54.60 and the reproduced model's score of 54.00. This demonstrates COLMATE's strong generalization ability to unseen domains.

#### Self-supervised Learning Performance
COLMATE was also evaluated using the self-supervised learning approach called Masked Contrastive Learning (MaskedCL). MaskedCL achieved nDCG@5 scores of 74.52 on ViDoRe V1 and 41.50 on ViDoRe V2, showcasing competitive performance even in the absence of labeled datasets. Notably, the performance gap narrowed in subsets representing real-world documents, such as AI, Energy, Government, and Health.

#### Contribution of Components
To assess the contribution of each component of COLMATE, detailed ablation studies were conducted. The combination of TopKSim, Masked OCR Language Modeling (MOLM), and MaskedCL was found to enhance performance. Specifically, TopKSim achieved a score of 56.41 on the out-of-domain ViDoRe V2 benchmark, outperforming MaxSim (54.00) and demonstrating superior generalization performance.

These results indicate that COLMATE overcomes the limitations of existing multimodal document retrieval methods and exhibits strong performance across various domains.


<br/>
# 예제



COLMATE 모델은 멀티모달 문서 검색을 위한 새로운 접근 방식을 제안합니다. 이 모델은 세 가지 주요 구성 요소를 통해 성능을 향상시킵니다: Masked OCR Language Modeling (MOLM), TopKSim, 그리고 Masked Contrastive Learning (MaskedCL)입니다. 이 모델의 훈련 및 평가 과정은 다음과 같이 진행됩니다.

1. **훈련 데이터 준비**:
   - **입력**: 4M 개의 디지털 PDF 문서가 이미지로 변환되어 사용됩니다. 각 문서에는 텍스트와 해당 텍스트의 경계 상자가 포함되어 있습니다.
   - **출력**: 모델은 OCR(Optical Character Recognition) 기술을 사용하여 문서에서 텍스트를 추출하고, 이 텍스트와 관련된 시각적 토큰을 생성합니다.

2. **Masked OCR Language Modeling (MOLM)**:
   - **입력**: 문서 이미지와 해당 이미지에서 추출된 텍스트 토큰이 주어집니다. 예를 들어, "건강 교육"이라는 텍스트가 포함된 문서 이미지가 있을 수 있습니다.
   - **출력**: 모델은 주어진 이미지에서 30%의 텍스트 토큰을 마스킹하고, 마스킹된 토큰을 예측합니다. 예를 들어, "건강 교육"에서 "건강"을 마스킹하면, 모델은 "교육"을 기반으로 "건강"을 예측해야 합니다.

3. **TopKSim**:
   - **입력**: 쿼리와 문서의 인코딩된 표현이 주어집니다. 예를 들어, 쿼리 "건강 교육 자료"와 문서의 인코딩된 표현이 있을 수 있습니다.
   - **출력**: 모델은 쿼리와 문서 간의 유사성을 계산하여 가장 유사한 K개의 문서 토큰을 선택합니다. 이 과정에서 K=5로 설정하여 상위 5개의 유사성을 평균화합니다.

4. **Masked Contrastive Learning (MaskedCL)**:
   - **입력**: 마스킹된 쿼리와 해당 쿼리와 관련된 문서 이미지가 주어집니다. 예를 들어, "건강 교육"의 일부를 마스킹한 쿼리와 그에 해당하는 문서 이미지가 있을 수 있습니다.
   - **출력**: 모델은 마스킹된 텍스트 표현과 마스킹된 시각적 표현 간의 대조적 정렬을 수행하여, 두 표현 간의 유사성을 극대화합니다.

5. **테스트 데이터**:
   - **입력**: ViDoRe V1 및 V2 벤치마크에서 제공되는 쿼리-문서 쌍이 사용됩니다. 예를 들어, "비만 예방을 위한 교육 자료"라는 쿼리와 관련된 문서가 있을 수 있습니다.
   - **출력**: 모델은 주어진 쿼리에 대해 가장 관련성이 높은 문서를 검색하여 nDCG@5 점수를 계산합니다.

이러한 과정을 통해 COLMATE 모델은 멀티모달 문서 검색에서 기존 방법들보다 더 나은 성능을 발휘합니다.

---




The COLMATE model proposes a new approach for multimodal document retrieval, enhancing performance through three main components: Masked OCR Language Modeling (MOLM), TopKSim, and Masked Contrastive Learning (MaskedCL). The training and evaluation process of this model is structured as follows:

1. **Training Data Preparation**:
   - **Input**: 4 million digital PDF documents are converted into images. Each document includes text and corresponding bounding boxes for the text.
   - **Output**: The model uses Optical Character Recognition (OCR) technology to extract text from the documents and generate visual tokens associated with this text.

2. **Masked OCR Language Modeling (MOLM)**:
   - **Input**: A document image and the text tokens extracted from that image are provided. For example, a document image containing the text "Health Education."
   - **Output**: The model masks 30% of the text tokens and predicts the masked tokens. For instance, if "Health Education" masks "Health," the model must predict "Health" based on the context of "Education."

3. **TopKSim**:
   - **Input**: Encoded representations of the query and document are given. For example, a query "Health Education Materials" and the encoded representation of a document.
   - **Output**: The model calculates the similarity between the query and document, selecting the top K most similar document tokens. This process averages the top 5 similarities when K=5.

4. **Masked Contrastive Learning (MaskedCL)**:
   - **Input**: A masked query and the corresponding document image related to that query are provided. For example, a query with part of "Health Education" masked and its corresponding document image.
   - **Output**: The model performs contrastive alignment between the masked textual representation and the masked visual representation, maximizing the similarity between the two.

5. **Test Data**:
   - **Input**: Query-document pairs provided by the ViDoRe V1 and V2 benchmarks are used. For example, a query "Educational Materials for Obesity Prevention" and a related document.
   - **Output**: The model retrieves the most relevant documents for the given query, calculating the nDCG@5 score.

Through this process, the COLMATE model demonstrates superior performance in multimodal document retrieval compared to existing methods.

<br/>
# 요약


COLMATE는 시각적 문서 검색을 위한 새로운 모델로, 세 가지 주요 구성 요소인 Masked OCR Language Modeling, TopKSim, Self-supervised Masked Contrastive Learning을 통합하여 성능을 향상시킵니다. 실험 결과, COLMATE는 ViDoRe V1 및 V2 벤치마크에서 기존 모델보다 평균 3.61% 향상된 성능을 보이며, 특히 도메인 외 일반화에서 두드러진 성과를 나타냅니다. 이 모델은 주석이 없는 데이터에서도 효과적으로 작동하여, 실제 검색 시나리오에서 유용성을 입증합니다.

---

COLMATE is a novel model for visual document retrieval that integrates three key components: Masked OCR Language Modeling, TopKSim, and Self-supervised Masked Contrastive Learning to enhance performance. Experimental results show that COLMATE achieves an average improvement of 3.61% over existing models on the ViDoRe V1 and V2 benchmarks, particularly excelling in out-of-domain generalization. This model demonstrates effectiveness even in scenarios with unlabeled data, proving its utility in practical retrieval contexts.

<br/>
# 기타



1. **다이어그램 및 피규어**:
   - **Figure 1**: COLMATE의 주요 구성 요소를 보여줍니다. Masked OCR Language Modeling (MOLM), TopKSim, Masked Contrastive Learning (MaskedCL) 세 가지 구성 요소가 어떻게 상호작용하여 멀티모달 문서 검색 성능을 향상시키는지를 시각적으로 설명합니다. 이 다이어그램은 각 구성 요소의 역할과 흐름을 명확히 하여 COLMATE의 혁신적인 접근 방식을 강조합니다.

2. **테이블**:
   - **Table 1**: ViDoRe V1 벤치마크에서 COLMATE와 기존 모델의 nDCG@5 점수를 비교합니다. COLMATE는 평균 85.14로 가장 높은 성능을 기록하며, 특히 MaskedCL을 사용한 경우 성능이 향상되었습니다. 이는 자가 지도 학습이 데이터가 부족한 상황에서도 효과적임을 보여줍니다.
   - **Table 2**: ViDoRe V2 벤치마크에서의 성능을 나타냅니다. COLMATE는 57.61의 점수로 가장 높은 평균을 기록하며, 이는 새로운 도메인에 대한 일반화 능력이 뛰어남을 시사합니다.
   - **Table 3**: COLMATE의 각 구성 요소가 ViDoRe V1 및 V2 벤치마크에서의 성능에 미치는 영향을 보여주는 세부적인 ablation 연구 결과입니다. 각 구성 요소가 성능에 미치는 기여도를 정량적으로 분석하여, MOLM과 TopKSim이 특히 중요한 역할을 한다는 것을 확인할 수 있습니다.
   - **Table 4**: Qwen2.5-VL-3B 모델에 COLMATE를 적용한 결과를 보여줍니다. COLMATE는 이 강력한 백본 모델에서도 성능 향상을 가져오며, 특히 out-of-domain 성능에서 유의미한 개선을 나타냅니다.

3. **어펜딕스**:
   - 어펜딕스에서는 실험 설정, 데이터셋, 하이퍼파라미터 및 추가적인 실험 결과를 제공합니다. 이는 연구의 재현성을 높이고, COLMATE의 성능을 뒷받침하는 데이터와 방법론을 명확히 합니다.

### Insights and Results from Other Sections (Diagrams, Figures, Tables, Appendices)

1. **Diagrams and Figures**:
   - **Figure 1**: This figure illustrates the key components of COLMATE. It visually explains how the three components—Masked OCR Language Modeling (MOLM), TopKSim, and Masked Contrastive Learning (MaskedCL)—interact to enhance multimodal document retrieval performance. The diagram emphasizes the innovative approach of COLMATE by clarifying the role and flow of each component.

2. **Tables**:
   - **Table 1**: This table compares the nDCG@5 scores of COLMATE and existing models on the ViDoRe V1 benchmark. COLMATE achieves the highest average score of 85.14, with notable improvements when using MaskedCL. This indicates the effectiveness of self-supervised learning, especially in low-data scenarios.
   - **Table 2**: It presents performance results on the ViDoRe V2 benchmark. COLMATE scores 57.61, the highest average, demonstrating strong generalization capabilities to unseen domains.
   - **Table 3**: This table shows detailed ablation study results, highlighting the impact of each component of COLMATE on the ViDoRe V1 and V2 benchmarks. It quantifies the contributions of MOLM and TopKSim, confirming their critical roles in enhancing performance.
   - **Table 4**: It reports results from applying COLMATE to the Qwen2.5-VL-3B model. COLMATE improves the out-of-domain performance of this powerful backbone model, indicating that the inductive bias introduced by COLMATE is beneficial even with stronger models.

3. **Appendices**:
   - The appendices provide experimental setups, datasets, hyperparameters, and additional experimental results. This enhances the reproducibility of the research and clarifies the data and methodologies supporting COLMATE's performance.

<br/>
# refer format:


### BibTeX 

```bibtex
@inproceedings{Masry2025,
  author    = {Ahmed Masry and Megh Thakkar and Patrice Bechard and Sathwik Tejaswi Madhusudhan and Rabiul Awal and Shambhavi Mishra and Akshay Kalkunte Suresh and Srivatsava Daruru and Enamul Hoque and Spandana Gella and Torsten Scholak and Sai Rajeswar},
  title     = {COLMATE: Contrastive Late Interaction and Masked Text for Multimodal Document Retrieval},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track},
  pages     = {2071--2080},
  year      = {2025},
  publisher = {Association for Computational Linguistics},
  address   = {November 4-9, 2025}
}
```

### 시카고 스타일

Masry, Ahmed, Megh Thakkar, Patrice Bechard, Sathwik Tejaswi Madhusudhan, Rabiul Awal, Shambhavi Mishra, Akshay Kalkunte Suresh, Srivatsava Daruru, Enamul Hoque, Spandana Gella, Torsten Scholak, and Sai Rajeswar. "COLMATE: Contrastive Late Interaction and Masked Text for Multimodal Document Retrieval." In *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track*, 2071–2080. November 4-9, 2025. Association for Computational Linguistics.
