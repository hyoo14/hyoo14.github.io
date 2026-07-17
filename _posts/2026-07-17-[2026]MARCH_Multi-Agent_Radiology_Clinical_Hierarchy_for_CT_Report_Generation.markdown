---
layout: post
title:  "[2026]MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation"
date:   2026-07-17 02:54:05 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: MARCH는 다중 에이전트 시스템을 활용하여 CT 보고서를 생성하는 프레임워크로, 초기 초안 작성, 검색 기반 수정, 합의 기반 최종화의 세 가지 단계로 구성된다.  
에이전트에 전문화된 역할을 부여해서 보고서를 에이전트들로 작성  


짧은 요약(Abstract) :


이 논문의 초록에서는 자동화된 3D 방사선 보고서 생성이 임상적 환각과 인간의 검토 과정에서 발견되는 반복적인 검증 부족으로 어려움을 겪고 있다는 점을 언급합니다. 최근의 비전-언어 모델(VLMs)이 이 분야에서 발전을 이루었지만, 이들은 일반적으로 임상 작업 흐름의 협력적 감독 없이 단일한 "블랙 박스" 시스템으로 작동합니다. 이러한 문제를 해결하기 위해, 저자들은 MARCH(다중 에이전트 방사선 임상 계층)라는 다중 에이전트 프레임워크를 제안합니다. 이 프레임워크는 방사선과 관련된 전문 계층을 모방하고 각 에이전트에 전문화된 역할을 부여합니다. MARCH는 초기 초안을 작성하는 레지던트 에이전트, 수정 작업을 수행하는 여러 펠로우 에이전트, 진단 불일치를 해결하기 위한 반복적 합의 담화를 조정하는 참석 에이전트로 구성됩니다. RadGenome-ChestCT 데이터셋에서 MARCH는 임상적 충실도와 언어적 정확성 모두에서 최신 기법들을 크게 초월하는 성과를 보였습니다. 이 연구는 인간과 유사한 조직 구조를 모델링함으로써 고위험 의료 분야에서 AI의 신뢰성을 향상시킬 수 있음을 보여줍니다.



The abstract of this paper discusses how automated 3D radiology report generation often struggles with clinical hallucinations and a lack of iterative verification found in human practice. While recent Vision-Language Models (VLMs) have made advancements in the field, they typically operate as monolithic "black-box" systems without the collaborative oversight characteristic of clinical workflows. To address these challenges, the authors propose MARCH (Multi-Agent Radiology Clinical Hierarchy), a multi-agent framework that emulates the professional hierarchy of radiology departments and assigns specialized roles to distinct agents. MARCH utilizes a Resident Agent for initial drafting, multiple Fellow Agents for retrieval-augmented revision, and an Attending Agent that orchestrates an iterative, stance-based consensus discourse to resolve diagnostic discrepancies. On the RadGenome-ChestCT dataset, MARCH significantly outperforms state-of-the-art baselines in both clinical fidelity and linguistic accuracy. This work demonstrates that modeling human-like organizational structures enhances the reliability of AI in high-stakes medical domains.


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


MARCH(다중 에이전트 방사선 임상 계층)는 CT 보고서 생성을 위한 다중 에이전트 프레임워크로, 방사선과 부서의 전문적인 계층 구조를 모방하여 각 에이전트에 특화된 역할을 부여합니다. 이 프레임워크는 세 가지 주요 단계로 구성됩니다: 초기 보고서 초안 작성, 검색 기반 보고서 수정, 그리고 합의 기반 최종화입니다.

1. **초기 보고서 초안 작성**: 이 단계에서 '레지던트 에이전트'가 CT 스캔 이미지를 기반으로 초기 보고서를 생성합니다. 이 에이전트는 대규모의 쌍으로 된 CT 스캔과 보고서 데이터셋을 학습하여 시각적 병리와 텍스트 설명 간의 교차 모달 정렬을 학습합니다. 또한, 다중 지역 분할 모듈을 사용하여 CT 스캔을 10개의 해부학적 하위 영역으로 나누어 각 영역의 병리적 특성을 보다 세밀하게 분석합니다.

2. **검색 기반 보고서 수정**: 초기 보고서가 작성된 후, '검색 에이전트'가 임상 데이터베이스에서 관련된 임상 맥락을 식별합니다. 이 단계에서는 이미지-이미지 및 이미지-텍스트 검색을 통해 시각적으로 유사한 CT 볼륨과 해당 보고서를 검색합니다. 검색된 증거는 '펠로우 에이전트'에게 제공되어 초기 초안을 검토하고 수정하여 일관성을 높이고 누락된 정보를 보완합니다.

3. **합의 기반 최종화**: 마지막 단계에서는 '어텐딩 에이전트'가 여러 펠로우 에이전트의 수정된 보고서를 통합하여 최종 보고서를 작성합니다. 이 과정은 다수의 에이전트가 서로의 의견을 검토하고, 동의하거나 수정 제안을 하며, 최종적으로 임상적으로 일관된 합의에 도달할 때까지 반복됩니다.

MARCH는 이러한 다중 에이전트 협업을 통해 CT 보고서의 임상적 정확성과 언어적 품질을 크게 향상시키며, 기존의 단일 에이전트 시스템에서 발생할 수 있는 해리적 오류를 줄이는 데 기여합니다. 이 프레임워크는 방사선과의 임상적 요구를 충족시키기 위해 설계되었으며, AI의 신뢰성을 높이는 데 중요한 역할을 합니다.



MARCH (Multi-Agent Radiology Clinical Hierarchy) is a multi-agent framework for CT report generation that emulates the professional hierarchy of radiology departments and assigns specialized roles to distinct agents. This framework consists of three main stages: Initial Report Drafting, Retrieval-Augmented Report Revision, and Consensus-Driven Finalization.

1. **Initial Report Drafting**: In this stage, the 'Resident Agent' generates an initial report based on CT scan images. This agent is trained on a large-scale corpus of paired volumetric CT scans and reports to learn the cross-modal alignment between visual pathology and textual descriptions. Additionally, a multi-region segmentation module is employed to partition the CT scan into ten anatomical subregions, allowing for a more detailed analysis of localized anatomical and pathological features.

2. **Retrieval-Augmented Report Revision**: After the initial report is drafted, the 'Retrieval Agent' identifies relevant clinical context from a clinical database. This stage involves image-to-image and image-to-text retrieval to find visually similar CT volumes and their corresponding reports. The retrieved evidence is then provided to 'Fellow Agents', who review and refine the initial draft to enhance coherence and address any omissions.

3. **Consensus-Driven Finalization**: The final stage employs an 'Attending Agent' to consolidate the revised reports from multiple Fellow Agents into a single final report. This process involves a multi-round collaborative protocol where agents review each other's opinions, agree, propose corrections, and iteratively refine the report until a clinically coherent consensus is reached.

MARCH significantly enhances the clinical fidelity and linguistic quality of CT reports through this multi-agent collaboration, reducing the risk of interpretive errors that can occur in single-agent systems. The framework is designed to meet the clinical demands of radiology and plays a crucial role in improving the reliability of AI in high-stakes medical domains.


<br/>
# Results


MARCH 모델은 RadGenome-ChestCT 데이터셋을 사용하여 여러 경쟁 모델과 비교 평가되었습니다. 이 데이터셋은 25,692개의 흉부 CT 스캔과 21,304명의 환자에 대한 상세한 방사선 보고서를 포함하고 있습니다. MARCH는 다음과 같은 주요 메트릭을 사용하여 성능을 평가했습니다: BLEU, ROUGE-L, METEOR, 그리고 Clinical Efficacy (CE) 점수. 

1. **경쟁 모델 비교**: MARCH는 여러 최신 모델과 비교하여 모든 평가 메트릭에서 우수한 성능을 보였습니다. 예를 들어, MARCH는 BLEU-4 점수에서 0.257을 기록하여, R2GenPT, MedVInT, CT2Rep, M3D, RadFM 등과 같은 다른 모델들보다 높은 점수를 기록했습니다. 특히, MARCH는 Clinical Efficacy 점수에서 0.399를 달성하여, 진단 정확성을 높이는 데 기여했습니다.

2. **테스트 데이터**: MARCH는 1,564개의 CT 스캔을 포함하는 테스트 세트에서 평가되었습니다. 이 테스트 세트는 훈련 세트와는 별도로 구성되어 있어, 모델의 일반화 능력을 평가하는 데 중요한 역할을 합니다.

3. **메트릭**: MARCH는 BLEU, ROUGE-L, METEOR와 같은 자연어 생성 메트릭을 사용하여 언어적 품질을 평가했습니다. 또한, Clinical Efficacy 점수는 18개의 정의된 임상 이상에 대한 정확성을 측정하여, 모델이 실제 임상 환경에서 얼마나 신뢰할 수 있는지를 평가하는 데 사용되었습니다.

4. **비교 결과**: MARCH는 모든 평가 메트릭에서 경쟁 모델들보다 우수한 성능을 보였으며, 특히 임상적 정확성과 언어적 일관성에서 두드러진 성과를 나타냈습니다. 예를 들어, MARCH는 18개의 임상 이상을 식별하는 데 있어 높은 재현율을 기록했으며, 특히 "hiatal hernia"와 "pericardial effusion"과 같은 미세한 이상을 감지하는 데 뛰어난 능력을 보였습니다.

이러한 결과는 MARCH가 다중 에이전트 협업을 통해 방사선 보고서 생성의 신뢰성을 높이고, 임상적 오류를 줄이는 데 효과적임을 보여줍니다.

---



The MARCH model was evaluated against several competing models using the RadGenome-ChestCT dataset, which contains 25,692 chest CT scans and detailed radiology reports for 21,304 unique patients. MARCH was assessed using key metrics such as BLEU, ROUGE-L, METEOR, and Clinical Efficacy (CE) scores.

1. **Comparison with Competing Models**: MARCH demonstrated superior performance across all evaluation metrics compared to several state-of-the-art models. For instance, MARCH achieved a BLEU-4 score of 0.257, outperforming other models like R2GenPT, MedVInT, CT2Rep, M3D, and RadFM. Notably, MARCH also achieved a Clinical Efficacy score of 0.399, contributing to improved diagnostic accuracy.

2. **Test Data**: MARCH was evaluated on a test set comprising 1,564 CT scans, which is separate from the training set, playing a crucial role in assessing the model's generalization capabilities.

3. **Metrics**: MARCH utilized natural language generation metrics such as BLEU, ROUGE-L, and METEOR to evaluate linguistic quality. Additionally, the Clinical Efficacy score measured the accuracy of 18 predefined clinical abnormalities, assessing how reliable the model is in real clinical settings.

4. **Comparison Results**: MARCH consistently outperformed competing models across all evaluation metrics, particularly excelling in clinical accuracy and linguistic coherence. For example, MARCH recorded high recall rates in identifying clinical abnormalities, especially for subtle findings like "hiatal hernia" and "pericardial effusion."

These results demonstrate that MARCH effectively enhances the reliability of radiology report generation through multi-agent collaboration, reducing cognitive errors in clinical interpretation.


<br/>
# 예제


MARCH (Multi-Agent Radiology Clinical Hierarchy) 프레임워크는 CT 보고서 생성을 위한 다중 에이전트 시스템으로, 세 가지 주요 단계로 구성됩니다: 초기 보고서 초안 작성, 검색 기반 보고서 수정, 그리고 합의 기반 최종화입니다. 이 시스템은 CT 스캔 데이터를 입력으로 받아, 각 단계에서 다양한 에이전트가 협력하여 최종 보고서를 생성합니다.

1. **초기 보고서 초안 작성 (Initial Report Drafting)**:
   - **입력**: 3D CT 스캔 이미지 (예: 환자의 흉부 CT 스캔).
   - **처리**: Resident Agent가 CT 스캔 이미지를 분석하여 각 해부학적 영역에 대한 초기 보고서를 작성합니다. 이 에이전트는 대규모의 CT 스캔과 보고서 쌍으로 훈련되어, 시각적 병리와 텍스트 설명 간의 관계를 학습합니다.
   - **출력**: 초기 보고서 초안 (예: "복부: 양측 부신은 정상이며, 공간 차지 병변이 발견되지 않았습니다.").

2. **검색 기반 보고서 수정 (Retrieval-Augmented Report Revision)**:
   - **입력**: 초기 보고서 초안과 관련된 임상 데이터베이스.
   - **처리**: Retrieval Agent가 관련된 임상 정보를 검색하여 Fellow Agent에게 전달합니다. Fellow Agent는 이 정보를 바탕으로 초기 보고서를 수정하고, 발견된 불일치 사항을 해결합니다.
   - **출력**: 수정된 보고서 (예: "복부: 양측 부신은 정상이며, 공간 차지 병변이 발견되지 않았습니다. 추가적으로, 간에 공간 차지 병변이 발견되지 않았습니다.").

3. **합의 기반 최종화 (Consensus-Driven Finalization)**:
   - **입력**: 수정된 보고서들.
   - **처리**: Attending Agent가 여러 Fellow Agent의 수정된 보고서를 종합하여 최종 보고서를 작성합니다. 이 과정에서 각 Fellow Agent는 자신의 의견을 제시하고, Attending Agent는 이 의견을 바탕으로 최종 결정을 내립니다.
   - **출력**: 최종 보고서 (예: "복부: 양측 부신은 정상이며, 공간 차지 병변이 발견되지 않았습니다. 간에 공간 차지 병변이 발견되지 않았습니다. 추가적으로, 갑상선 초음파 검사가 권장됩니다.").

이러한 단계들은 MARCH 프레임워크가 CT 보고서 생성을 위해 어떻게 다중 에이전트 협업을 활용하는지를 보여줍니다. 각 에이전트는 특정 역할을 수행하며, 최종적으로는 더 높은 임상 정확성과 언어적 품질을 가진 보고서를 생성하는 데 기여합니다.

---



The MARCH (Multi-Agent Radiology Clinical Hierarchy) framework is a multi-agent system designed for CT report generation, consisting of three main stages: Initial Report Drafting, Retrieval-Augmented Report Revision, and Consensus-Driven Finalization. This system takes CT scan data as input and utilizes various agents to collaboratively generate the final report at each stage.

1. **Initial Report Drafting**:
   - **Input**: 3D CT scan images (e.g., a patient's chest CT scan).
   - **Processing**: The Resident Agent analyzes the CT scan images to create an initial report draft for each anatomical region. This agent is trained on a large corpus of paired CT scans and reports to learn the relationship between visual pathology and textual descriptions.
   - **Output**: Initial report draft (e.g., "Abdomen: Bilateral adrenal glands are normal, and no space-occupying lesions were detected.").

2. **Retrieval-Augmented Report Revision**:
   - **Input**: Initial report draft and relevant clinical database.
   - **Processing**: The Retrieval Agent searches for relevant clinical information and provides it to the Fellow Agent. The Fellow Agent refines the initial report based on this information, addressing any discrepancies found.
   - **Output**: Revised report (e.g., "Abdomen: Bilateral adrenal glands are normal, and no space-occupying lesions were detected. Additionally, no lesions were found in the liver.").

3. **Consensus-Driven Finalization**:
   - **Input**: Revised reports from multiple Fellow Agents.
   - **Processing**: The Attending Agent consolidates the revised reports into a final report. During this process, each Fellow Agent presents their opinions, and the Attending Agent makes the final decision based on these inputs.
   - **Output**: Final report (e.g., "Abdomen: Bilateral adrenal glands are normal, and no space-occupying lesions were detected. No lesions were found in the liver. Additionally, a thyroid ultrasound is recommended.").

These stages illustrate how the MARCH framework leverages multi-agent collaboration for CT report generation. Each agent performs a specific role, ultimately contributing to the creation of reports with higher clinical accuracy and linguistic quality.

<br/>
# 요약
MARCH는 다중 에이전트 시스템을 활용하여 CT 보고서를 생성하는 프레임워크로, 초기 초안 작성, 검색 기반 수정, 합의 기반 최종화의 세 가지 단계로 구성된다. 실험 결과, MARCH는 기존의 최첨단 방법들보다 임상적 정확성과 언어적 품질에서 유의미한 성과를 보였다. 사례 연구를 통해 MARCH의 다중 에이전트 협업이 진단의 모호성을 해결하고 더 신뢰할 수 있는 보고서를 생성하는 데 기여함을 보여주었다.

---

MARCH is a framework that utilizes a multi-agent system for generating CT reports, consisting of three stages: initial draft creation, retrieval-augmented revision, and consensus-driven finalization. Experimental results demonstrate that MARCH significantly outperforms existing state-of-the-art methods in clinical accuracy and linguistic quality. A case study illustrates how MARCH's multi-agent collaboration helps resolve diagnostic ambiguities and produce more reliable reports.

<br/>
# 기타


#### 1. 다이어그램 및 피규어
- **MARCH 프레임워크 개요 (Figure 1)**: MARCH는 세 가지 주요 단계로 구성되어 있습니다: 초기 보고서 초안 작성, 검색 기반 보고서 수정, 합의 기반 최종화. 각 단계에서 다양한 에이전트가 협력하여 최종 보고서를 생성합니다. 이 구조는 의료 영상 해석의 복잡성을 효과적으로 관리하고, 각 에이전트의 전문성을 활용하여 진단의 정확성을 높입니다.

- **임상 효능 분석 (Figure 2)**: MARCH는 18가지 임상 이상에 대한 F1 점수를 비교하여, 특히 "hiatal hernia"와 "pericardial effusion"과 같은 미세한 이상을 감지하는 데 뛰어난 성능을 보였습니다. 이는 MARCH가 세밀한 임상 발견을 식별하는 능력이 향상되었음을 나타냅니다.

- **생성된 보고서 예시 (Figure 5)**: MARCH가 생성한 보고서는 Resident Agent가 생성한 초기 초안보다 더 정제되고 임상적으로 신뢰할 수 있는 정보를 제공합니다. MARCH는 불확실한 관찰을 필터링하고, 임상적으로 중요한 발견을 강조하여 보고서의 질을 향상시킵니다.

#### 2. 테이블
- **성능 비교 (Table 1)**: MARCH는 BLEU, ROUGE-L, METEOR와 같은 다양한 자연어 생성 메트릭에서 기존의 최첨단 방법들을 초월하는 성능을 보였습니다. 이는 MARCH가 고품질의 임상적으로 정확한 방사선 보고서를 생성하는 데 효과적임을 보여줍니다.

- **구성 요소의 기여도 (Table 2)**: MARCH의 각 구성 요소가 전체 성능에 미치는 영향을 평가한 결과, 합의 기반 최종화 단계의 제거가 가장 큰 성능 저하를 초래했습니다. 이는 다수의 에이전트 간의 협력이 보고서의 질을 높이는 데 필수적임을 시사합니다.

- **에이전트 수의 효과 (Table 5)**: Fellow Agent의 수를 조정한 결과, 5명의 에이전트가 가장 높은 언어적 품질을 달성했습니다. 그러나 20명의 에이전트로 증가시키면 성능이 약간 감소하는 경향이 있어, 적절한 에이전트 수의 균형이 중요함을 나타냅니다.

#### 3. 어펜딕스
- **데이터셋 통계 (Appendix B)**: RadGenome-ChestCT 데이터셋은 25,692개의 3D 흉부 CT 스캔을 포함하고 있으며, 각 스캔은 경험이 풍부한 방사선 전문의가 작성한 상세한 보고서와 함께 제공됩니다. 이 데이터셋은 다양한 임상 이상을 포함하여, MARCH의 성능 평가에 중요한 역할을 합니다.

- **사례 연구 (Appendix C)**: MARCH의 다단계 에이전트 워크플로우를 통해 생성된 보고서의 예시를 제공하며, 각 단계에서의 에이전트의 역할과 기여를 설명합니다. 이는 MARCH의 임상적 유용성을 강조합니다.



#### Summary of Results and Insights

#### 1. Diagrams and Figures
- **Overview of the MARCH Framework (Figure 1)**: MARCH consists of three main stages: Initial Report Drafting, Retrieval-Augmented Report Revision, and Consensus-Driven Finalization. Various agents collaborate at each stage to generate the final report. This structure effectively manages the complexity of medical image interpretation and enhances diagnostic accuracy by leveraging the expertise of each agent.

- **Clinical Efficacy Analysis (Figure 2)**: MARCH demonstrated superior performance in detecting subtle abnormalities, particularly "hiatal hernia" and "pericardial effusion," as indicated by the F1 scores across 18 clinical abnormalities. This suggests that MARCH has improved its ability to identify fine clinical findings.

- **Examples of Generated Reports (Figure 5)**: Reports generated by MARCH are more refined and clinically reliable compared to the initial drafts produced by the Resident Agent. MARCH filters out uncertain observations and emphasizes clinically relevant findings, thereby enhancing the quality of the reports.

#### 2. Tables
- **Performance Comparison (Table 1)**: MARCH outperformed existing state-of-the-art methods across various natural language generation metrics, including BLEU, ROUGE-L, and METEOR. This demonstrates MARCH's effectiveness in generating high-quality, clinically accurate radiology reports.

- **Contribution of Components (Table 2)**: An ablation study revealed that removing the consensus-driven finalization stage resulted in the most significant drop in performance. This indicates that collaboration among multiple agents is essential for improving report quality.

- **Effect of Number of Agents (Table 5)**: Adjusting the number of Fellow Agents showed that an ensemble of five agents achieved the highest linguistic quality. However, increasing the number to twenty led to a slight decline in performance, suggesting the importance of balancing the number of agents.

#### 3. Appendix
- **Dataset Statistics (Appendix B)**: The RadGenome-ChestCT dataset contains 25,692 3D chest CT scans, each accompanied by detailed reports authored by experienced radiologists. This dataset plays a crucial role in evaluating MARCH's performance.

- **Case Study (Appendix C)**: A case study illustrates the hierarchical, multi-agent workflow intrinsic to MARCH, detailing the contributions of agents at each stage. This emphasizes the clinical utility of MARCH.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{Lin2026,
  author    = {Yi Lin and Yihao Ding and Yonghui Wu and Yifan Peng},
  title     = {MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
  pages     = {273--285},
  year      = {2026},
  month     = {July},
  publisher = {Association for Computational Linguistics},

}
```

### 시카고 스타일

Lin, Yi, Yihao Ding, Yonghui Wu, and Yifan Peng. "MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation." In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, 273–285. Association for Computational Linguistics, July 2026.  