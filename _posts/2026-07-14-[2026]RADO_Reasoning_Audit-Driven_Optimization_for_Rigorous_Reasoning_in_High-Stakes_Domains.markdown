---
layout: post
title:  "[2026]RADO: Reasoning Audit-Driven Optimization for Rigorous Reasoning in High-Stakes Domains"
date:   2026-07-14 18:58:44 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: RADO는 고위험 도메인에서의 논리적 엄격성을 향상시키기 위해 감사 모델을 활용하는 프레임워크입니다.


짧은 요약(Abstract) :


이 논문의 초록에서는 고위험 분야(예: 금융, 법률, 생물 의학)에서 정확한 결과와 엄격한 추론이 필요하다는 점을 강조합니다. 현재의 강화 학습 패러다임은 주로 결과 기반 보상에 의존하고 있으며, 중간 단계에서의 잠재적인 논리적 오류를 간과하는 경향이 있습니다. 저자들은 RADO(Reasoning Audit-Driven Optimization)라는 새로운 접근 방식을 제안합니다. RADO는 외부 도구를 활용하여 지역적 논리적 결함을 식별하고 보상 신호를 조정하는 전문 감사 모델을 도입합니다. Direct Preference Optimization(DPO)와 Group Relative Policy Optimization(GRPO)을 통합하여, 이 프레임워크는 추론 경로에 대한 명시적인 감독을 가능하게 합니다. 실험 결과는 RADO가 최종 정확성을 일관되게 향상시키고 고위험 분야에서 논리적 엄격성을 크게 개선함을 보여줍니다.



The abstract of this paper emphasizes the need for accurate results and rigorous reasoning in high-stakes domains such as finance, law, and biomedicine. Current reinforcement learning paradigms primarily rely on outcome-based rewards, often overlooking potential logical errors in intermediate steps. The authors propose a new approach called RADO (Reasoning Audit-Driven Optimization). RADO introduces a specialized audit model that leverages external tools to identify local logical flaws and calibrate reward signals. By integrating Direct Preference Optimization (DPO) and Group Relative Policy Optimization (GRPO), this framework enables explicit supervision over reasoning paths. Experimental results demonstrate that RADO consistently improves final accuracy while significantly enhancing logical rigor in high-stakes domains.


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



RADO(Reasoning Audit-Driven Optimization)는 고위험 도메인에서의 엄격한 추론을 향상시키기 위해 설계된 프레임워크입니다. 이 프레임워크는 세 가지 주요 단계로 구성되어 있습니다.

1. **모델 초기화**: RADO는 정책 모델(Policy Model, PM), 보상 모델(Reward Model, RM), 감사 모델(Audit Model, AM)로 구성됩니다. 정책 모델은 오픈 소스 지침 모델에서 초기화되며, 보상 모델은 금융, 법률, 의학 분야의 질문-답변 데이터셋에서 수집된 데이터를 사용하여 훈련됩니다. 감사 모델은 보상 모델의 평가 결과를 검토하고 오류를 식별하기 위해 DeepSeek-v3.2와 같은 강력한 일반 모델을 호출하여 초기화됩니다.

2. **반복 최적화**: 두 번째 단계에서는 감사 모델이 다양한 추론 경로를 감사하여 보상 모델의 정확성을 향상시키기 위해 반복적으로 최적화됩니다. 감사 모델은 외부 도구를 호출하여 추론 단계를 검증하고, 이를 통해 생성된 데이터로 보상 모델을 개선합니다. 이 과정에서 Direct Preference Optimization(DPO) 기법을 사용하여 보상 모델의 정밀도를 점진적으로 향상시킵니다.

3. **정책 모델 최적화**: 마지막 단계에서는 최적화된 보상 모델을 사용하여 정책 모델을 훈련합니다. Group Relative Policy Optimization(GRPO) 기법을 통해 정책 모델은 보상 모델이 제공하는 세밀한 보상 신호를 기반으로 최적화됩니다. 이 과정에서 모델은 명시적으로 보상 함수를 인식하고, 의도적으로 추론을 극대화합니다.

RADO는 금융, 법률, 의학의 세 가지 고위험 도메인에서 실험을 통해 최종 답변의 정확성과 추론 경로의 논리적 엄격성을 모두 향상시키는 데 성공했습니다. 이 프레임워크는 고위험 도메인에서의 안전성과 규정 준수를 보장하기 위해 설계되었습니다.




RADO (Reasoning Audit-Driven Optimization) is a framework designed to enhance rigorous reasoning in high-stakes domains. This framework consists of three main stages.

1. **Model Initialization**: RADO is composed of a Policy Model (PM), a Reward Model (RM), and an Audit Model (AM). The Policy Model is initialized from open-source instruction models, while the Reward Model is trained using data collected from question-answering datasets in finance, law, and biomedicine. The Audit Model is initialized by reviewing the evaluation results of the Reward Model and identifying errors, using a powerful general model like DeepSeek-v3.2.

2. **Iterative Optimization**: In the second stage, the Audit Model audits various reasoning paths to iteratively optimize the Reward Model's accuracy. The Audit Model calls external tools to verify reasoning steps, and the data generated from this process is used to improve the Reward Model. During this process, Direct Preference Optimization (DPO) techniques are employed to gradually enhance the precision of the Reward Model.

3. **Policy Model Optimization**: The final stage involves training the Policy Model using the optimized Reward Model. Group Relative Policy Optimization (GRPO) techniques are used to optimize the Policy Model based on the fine-grained reward signals provided by the Reward Model. In this process, the model explicitly perceives its reward function and maximizes reasoning intentionally.

RADO has successfully improved both the accuracy of final answers and the logical rigor of reasoning paths through experiments in three high-stakes domains: finance, law, and biomedicine. This framework is designed to ensure safety and compliance in high-stakes domains.


<br/>
# Results



이 논문에서는 RADO(Reasoning Audit-Driven Optimization)라는 새로운 프레임워크를 제안하고, 이를 통해 고위험 도메인(재무, 법률, 생물의학)에서의 논리적 엄격성을 향상시키는 방법을 설명합니다. RADO는 외부 도구를 활용한 감사 모델을 통해 중간 단계에서의 논리적 오류를 식별하고 보상 신호를 조정하여, 최종 결과의 정확성과 논리적 rigor를 동시에 개선합니다.

#### 실험 결과

1. **경쟁 모델**: RADO는 여러 경쟁 모델과 비교되었습니다. 예를 들어, DeepSeek-R1-Llama, GPT-OSS, HIPO, MiMo-RL, Qwen-2.5-Instruct, Llama-3-Instruct 등 다양한 모델이 포함되었습니다. 이들 모델은 각각 고유한 아키텍처와 훈련 방법을 가지고 있으며, RADO는 이들 모델과의 성능 비교를 통해 그 우수성을 입증하고자 했습니다.

2. **테스트 데이터**: RADO는 LegalBench, MMLU, PubMedQA, FinQA, Stock Prediction 등 다양한 데이터셋에서 평가되었습니다. 각 데이터셋은 특정 도메인에 맞춰 설계되었으며, RADO는 이러한 데이터셋에서의 성능을 통해 그 효과를 검증했습니다.

3. **메트릭**: 성능 평가는 정확도(accuracy)와 PROOF-Score와 같은 다양한 메트릭을 사용하여 이루어졌습니다. PROOF-Score는 논리적 완전성, 도메인 안전성, 사실 정확성을 평가하는 지표로, RADO의 논리적 경로의 질을 정량적으로 평가하는 데 사용되었습니다.

4. **비교 결과**: RADO는 모든 테스트 데이터셋에서 경쟁 모델들보다 높은 성능을 보였습니다. 예를 들어, LegalBench에서 RADO는 81.11%의 정확도를 기록하여, 가장 가까운 경쟁 모델인 Qwen-2.5-Instruct-7B(73.86%)보다 7.25% 높은 성과를 보였습니다. MMLU와 PubMedQA에서도 RADO는 각각 80.54%와 80.39%의 정확도를 기록하며, 다른 모델들보다 우수한 성능을 입증했습니다. 특히, FinQA에서는 RADO가 81.52%의 정확도로, 이전 모델들보다 5.38% 높은 성과를 달성했습니다.

이러한 결과는 RADO가 고위험 도메인에서의 논리적 rigor와 최종 결과의 정확성을 동시에 향상시킬 수 있는 강력한 도구임을 보여줍니다.

---




This paper introduces a new framework called RADO (Reasoning Audit-Driven Optimization) and explains how it enhances logical rigor in high-stakes domains such as finance, law, and biomedicine. RADO utilizes an audit model augmented with external tools to identify logical errors in intermediate steps and calibrate reward signals, thereby improving both the accuracy of final results and logical rigor.

#### Experimental Results

1. **Competing Models**: RADO was compared against several competing models, including DeepSeek-R1-Llama, GPT-OSS, HIPO, MiMo-RL, Qwen-2.5-Instruct, and Llama-3-Instruct. Each of these models has its unique architecture and training methods, and RADO aimed to demonstrate its superiority through performance comparisons.

2. **Test Data**: RADO was evaluated on various datasets such as LegalBench, MMLU, PubMedQA, FinQA, and Stock Prediction. Each dataset was designed to cater to specific domains, and RADO's performance on these datasets was used to validate its effectiveness.

3. **Metrics**: Performance evaluation was conducted using various metrics, including accuracy and PROOF-Score. PROOF-Score is a metric that assesses reasoning completeness, domain safety, and factual accuracy, and it was used to quantitatively evaluate the quality of RADO's reasoning paths.

4. **Comparison Results**: RADO consistently outperformed competing models across all test datasets. For instance, in LegalBench, RADO achieved an accuracy of 81.11%, surpassing the nearest competitor, Qwen-2.5-Instruct-7B (73.86%), by 7.25%. In MMLU and PubMedQA, RADO recorded accuracies of 80.54% and 80.39%, respectively, demonstrating superior performance compared to other models. Notably, in FinQA, RADO achieved an accuracy of 81.52%, which is 5.38% higher than previous models.

These results indicate that RADO is a powerful tool capable of enhancing both logical rigor and the accuracy of final outcomes in high-stakes domains.


<br/>
# 예제



이 논문에서는 RADO(Reasoning Audit-Driven Optimization)라는 프레임워크를 제안하고, 이를 통해 고위험 도메인에서의 논리적 추론의 엄격성을 향상시키는 방법을 설명합니다. RADO는 세 가지 주요 단계로 구성됩니다: 보상 모델(Reward Model, RM) 초기화, 감사 모델(Audit Model, AM) 최적화, 그리고 정책 모델(Policy Model, PM) 최적화입니다.

1. **보상 모델 초기화**: 
   - RADO는 금융, 법률, 생물의학 분야의 질문-답변 데이터셋에서 2000개의 질문을 수집합니다. 
   - 정책 모델(예: Qwen-2.5-Instruct-7B)을 사용하여 각 질문에 대한 여러 샘플을 생성합니다.
   - DeepSeek-v3.2라는 모델을 호출하여 각 샘플의 추론 과정과 최종 답변의 정확성을 평가하고, 이를 바탕으로 보상 모델을 초기화합니다.

2. **감사 모델 최적화**: 
   - 감사 모델은 보상 모델의 평가 결과를 검토하여 오류를 식별합니다. 
   - 이 과정에서 외부 도구(예: 웹 검색, 수치 계산)를 사용하여 더 정밀한 평가를 수행합니다. 
   - 감사 모델은 보상 모델의 정확성을 높이기 위해 반복적으로 최적화됩니다.

3. **정책 모델 최적화**: 
   - 최적화된 보상 모델을 사용하여 정책 모델을 훈련합니다. 
   - 이 단계에서는 그룹 상대 정책 최적화(Group Relative Policy Optimization, GRPO)를 통해 정책 모델의 성능을 향상시킵니다.

### 예시
- **트레이닝 데이터**: 
  - 질문: "이 조항은 당사자의 의무 위반 시 책임의 한도를 명시하고 있습니까?"
  - 정답: "아니오"
  - 모델의 응답: "<think> 이 조항을 분석해야 합니다. </think><answer> 아니오 </answer>"

- **테스트 데이터**: 
  - 질문: "자산 퇴직 의무가 2008년에서 2009년 사이에 몇 퍼센트 증가했습니까?"
  - 정답: "14.197%"
  - 모델의 응답: "<think> 자산 퇴직 의무의 증가율을 계산해야 합니다. </think><answer> 14.23% </answer>"

이러한 방식으로 RADO는 고위험 도메인에서의 추론의 정확성과 논리적 엄격성을 높이는 데 기여합니다.

---




This paper proposes a framework called RADO (Reasoning Audit-Driven Optimization) and explains how it enhances the rigor of logical reasoning in high-stakes domains. RADO consists of three main stages: initialization of the Reward Model (RM), optimization of the Audit Model (AM), and optimization of the Policy Model (PM).

1. **Reward Model Initialization**: 
   - RADO collects 2,000 questions from question-answer datasets in finance, law, and biomedicine.
   - A policy model (e.g., Qwen-2.5-Instruct-7B) is used to generate multiple samples for each question.
   - The DeepSeek-v3.2 model is invoked to evaluate the correctness of each sample's reasoning process and final answer, which is used to initialize the Reward Model.

2. **Audit Model Optimization**: 
   - The Audit Model reviews the evaluations from the Reward Model to identify errors.
   - During this process, external tools (e.g., web search, numerical computation) are used to perform more precise evaluations.
   - The Audit Model is iteratively optimized to enhance the accuracy of the Reward Model.

3. **Policy Model Optimization**: 
   - The optimized Reward Model is used to train the Policy Model.
   - In this stage, Group Relative Policy Optimization (GRPO) is employed to improve the performance of the Policy Model.

### Example
- **Training Data**: 
  - Question: "Does this clause specify a cap on liability upon the breach of a party’s obligation?"
  - Correct Answer: "No"
  - Model Response: "<think> I need to analyze this clause. </think><answer> No </answer>"

- **Test Data**: 
  - Question: "By what percentage did asset retirement obligations increase from 2008 to 2009?"
  - Correct Answer: "14.197%"
  - Model Response: "<think> I need to calculate the percentage increase in asset retirement obligations. </think><answer> 14.23% </answer>"

In this way, RADO contributes to improving the accuracy and logical rigor of reasoning in high-stakes domains.

<br/>
# 요약


RADO는 고위험 도메인에서의 논리적 엄격성을 향상시키기 위해 감사 모델을 활용하는 프레임워크입니다. 실험 결과, RADO는 최종 정확도를 개선할 뿐만 아니라 논리적 경로의 엄격성을 크게 향상시켰습니다. 이 방법은 금융, 법률 및 생물 의학 분야에서 효과적으로 적용되었습니다.

---

RADO is a framework that enhances logical rigor in high-stakes domains by leveraging an audit model. Experimental results demonstrate that RADO not only improves final accuracy but also significantly enhances the rigor of reasoning paths. This method has been effectively applied in finance, law, and biomedicine.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: RADO의 접근 방식과 기존 모델의 차이를 시각적으로 비교합니다. RADO는 명확한 감사 과정을 통해 논리적 경로의 엄격함을 보장하며, 기존 모델들이 자주 발생하는 "잘못된 이유로 올바른 답변" 문제를 해결합니다. 이 다이어그램은 RADO의 효과적인 감사 메커니즘을 강조합니다.
   - **Figure 2**: RADO 프레임워크의 전체 아키텍처를 보여줍니다. 세 가지 단계(보상 모델 초기화, 감사 모델 최적화, 정책 모델 최적화)를 통해 RADO가 어떻게 작동하는지를 설명합니다. 이 구조는 RADO의 복잡한 최적화 과정을 명확하게 나타냅니다.

2. **테이블**
   - **Table 1, 2, 3**: RADO의 성능을 다양한 데이터셋(법률, 생물의학, 금융)에서 다른 모델들과 비교한 결과를 보여줍니다. RADO는 모든 도메인에서 SOTA(최첨단) 성능을 달성하며, 특히 법률 및 금융 도메인에서 기존 모델들보다 유의미한 성능 향상을 보였습니다. 이는 RADO의 감사 모델이 논리적 엄격성을 높이는 데 기여했음을 나타냅니다.
   - **Table 4**: PROOF-Score를 통해 RADO의 추론 경로 품질을 평가한 결과를 보여줍니다. RADO는 모든 도메인에서 가장 높은 PROOF-Score를 기록하여, 감사 기반 최적화가 추론 경로의 구조적 무결성과 논리적 투명성을 크게 향상시켰음을 나타냅니다.
   - **Table 5**: RADO의 구성 요소에 대한 Ablation Study 결과를 보여줍니다. 감사 모델이나 외부 도구 호출 기능이 없는 경우 성능이 저하되는 것을 확인할 수 있습니다. 이는 RADO의 감사 모델이 필수적임을 강조합니다.

3. **어펜딕스**
   - 어펜딕스에서는 RADO의 구현 세부사항, 실험 설정, 데이터셋 설명, 그리고 추가적인 실험 결과를 제공합니다. 특히, RADO의 반복 최적화 과정과 감사 모델의 훈련 방법에 대한 자세한 설명이 포함되어 있어, 연구자들이 RADO의 작동 방식을 이해하는 데 도움을 줍니다.

### Insights
- RADO는 고위험 도메인에서의 논리적 엄격성을 보장하기 위해 설계된 혁신적인 프레임워크입니다. 
- 감사 모델의 도입은 RADO의 성능을 크게 향상시키며, 이는 고위험 결정에서의 안전성과 신뢰성을 높이는 데 기여합니다.
- RADO의 구조적 접근 방식은 기존의 결과 기반 보상 모델의 한계를 극복하고, 중간 단계에서의 논리적 오류를 식별하는 데 효과적입니다.

---




1. **Diagrams and Figures**
   - **Figure 1**: Visually compares RADO's approach with existing models. RADO ensures the rigor of logical paths through a clear auditing process, addressing the common issue of "getting the right answer for the wrong reasons" that existing models often face. This diagram emphasizes the effectiveness of RADO's auditing mechanism.
   - **Figure 2**: Shows the overall architecture of the RADO framework. It explains how RADO operates through three stages (initialization of the reward model, optimization of the audit model, and optimization of the policy model). This structure clearly illustrates the complex optimization process of RADO.

2. **Tables**
   - **Table 1, 2, 3**: Present the performance of RADO compared to other models across various datasets (legal, biomedical, finance). RADO achieves state-of-the-art (SOTA) performance in all domains, showing significant performance improvements over existing models, particularly in legal and financial domains. This indicates that RADO's audit model contributes to enhancing logical rigor.
   - **Table 4**: Displays the results of evaluating the quality of reasoning paths through PROOF-Score. RADO records the highest PROOF-Score across all domains, indicating that audit-driven optimization significantly enhances the structural integrity and logical transparency of reasoning paths.
   - **Table 5**: Shows the results of an ablation study on the components of RADO. It confirms that the performance degrades without the audit model or external tool-calling capabilities, emphasizing the necessity of RADO's audit model.

3. **Appendix**
   - The appendix provides detailed information on RADO's implementation, experimental settings, dataset descriptions, and additional experimental results. It includes a thorough explanation of RADO's iterative optimization process and the training methods for the audit model, aiding researchers in understanding how RADO operates.

### Insights
- RADO is an innovative framework designed to ensure logical rigor in high-stakes domains.
- The introduction of the audit model significantly enhances RADO's performance, contributing to increased safety and reliability in high-stakes decision-making.
- RADO's structural approach effectively overcomes the limitations of existing outcome-based reward models, identifying logical errors at intermediate steps.

<br/>
# refer format:
### BibTeX 


```bibtex
@inproceedings{Tan2026,
  author    = {Zhijie Tan and Xu Chu and Guanyu Wang and Ziyu Li and Weiping Li and Tong Mo},
  title     = {RADO: Reasoning Audit-Driven Optimization for Rigorous Reasoning in High-Stakes Domains},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {4659--4683},
  year      = {2026},
  month     = {July},
  publisher = {Association for Computational Linguistics},
  
  
}
```

### 시카고 스타일

Tan, Zhijie, Xu Chu, Guanyu Wang, Ziyu Li, Weiping Li, and Tong Mo. "RADO: Reasoning Audit-Driven Optimization for Rigorous Reasoning in High-Stakes Domains." In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 4659–4683. Association for Computational Linguistics, July 2026.
