---
layout: post
title:  "[2026]Investigating Stigmatizing Language in Clinical Documentation with Open-Source Large Language Models"
date:   2026-07-14 00:40:54 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 본 연구에서는 StigMAD 프레임워크를 통해 오픈소스 대형 언어 모델을 활용하여 임상 문서에서 낙인 언어를 탐지하는 방법을 제안하였다.


짧은 요약(Abstract) :

이 논문의 초록에서는 임상 문서에서 낙인 언어를 탐지하기 위한 새로운 프레임워크인 StigMAD를 소개하고 평가합니다. 임상 문서는 환자 치료, 청구 및 의학 연구에 필수적이지만, 편견이 내재되어 있을 수 있습니다. 수작업 차트 검토는 이러한 편견을 식별할 수 있지만, 노동 집약적이고 전문가 의존적입니다. StigMAD는 오픈 소스 대형 언어 모델(LLMs)을 활용하여 낙인 언어를 탐지하는 다중 에이전트 토론 프레임워크를 제안합니다. 이 프레임워크는 추론, 자기 반성 및 자기 일관성을 조사하며, 임상 노트와 환자 요약에 대한 실험을 통해 규칙 기반 및 감독 학습 기준선보다 유의미한 이점을 제공함을 보여줍니다. 특정 도메인 LLM인 MedGemma는 StigMAD 추론 프레임워크를 사용하여 최고의 성능을 달성했으며, 일반 목적의 LLM인 Llama는 자기 일관성 프레임워크에서 우수한 결과를 보였습니다. 이러한 발견은 구조화된 프롬프트와 반사적 추론에 의해 조정된 오픈 소스 LLM이 낙인 언어 감사에 효과적으로 기여할 수 있음을 시사합니다.



The abstract of this paper introduces and evaluates a new framework called StigMAD for detecting stigmatizing language in clinical documentation. Clinical documentation is essential for patient care, billing, and medical research, but it can be subject to entrenched bias. Manual chart reviews can identify such bias, but they are labor-intensive and expert-dependent. StigMAD proposes a multi-agent debate framework leveraging open-source large language models (LLMs) to detect stigmatizing language. This framework investigates reasoning, self-reflection, and self-consistency, demonstrating significant advantages over rule-based and supervised baselines through experiments on clinical notes and patient summaries. A domain-specific LLM, MedGemma, achieved its highest performance using the StigMAD reasoning framework, while a general-purpose LLM, Llama, showed superior results with the self-consistency framework. These findings suggest that open-source LLMs, guided by structured prompting and reflective reasoning, can effectively support the auditing of stigmatizing language.


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



이 논문에서는 임상 문서에서 낙인 언어를 탐지하기 위해 STIGMAD라는 다중 에이전트 토론 프레임워크를 도입하고 평가합니다. STIGMAD는 오픈 소스 대형 언어 모델(LLM)을 활용하여 낙인 언어를 식별하는 데 중점을 두고 있으며, 이 과정에서 자기 일관성, 자기 반성, 그리고 다중 에이전트 토론을 활용합니다.

1. **모델 및 아키텍처**: 
   - STIGMAD 프레임워크는 두 가지 주요 LLM을 사용합니다: MedGemma(의료 도메인 특화 모델)와 Llama(일반 목적 모델). 이 두 모델은 각각의 프레임워크에서 낙인 언어를 탐지하는 데 사용됩니다.
   - 프레임워크는 네 가지 주요 에이전트로 구성됩니다: 긍정 에이전트, 부정 에이전트, 중재자 에이전트, 그리고 판사 에이전트입니다. 긍정 에이전트는 주어진 주제에 대해 찬성하는 주장을 하고, 부정 에이전트는 이를 반박합니다. 중재자 에이전트는 두 에이전트의 주장을 조정하고, 판사 에이전트는 최종 결정을 내립니다.

2. **트레이닝 데이터**: 
   - 연구에서는 MIMIC-IV 데이터셋과 PMC-Patients 요약 데이터셋을 사용합니다. MIMIC-IV 데이터셋은 실제 임상 문서로 구성되어 있으며, PMC-Patients 데이터셋은 환자 요약 정보를 포함합니다. 이 데이터셋들은 낙인 언어의 존재를 평가하기 위해 사용됩니다.

3. **특별한 기법**: 
   - STIGMAD는 자기 반성(self-reflection)과 자기 일관성(self-consistency) 기법을 사용하여 모델의 출력을 검증하고, 낙인 언어의 탐지 정확성을 높입니다. 자기 반성 기법은 모델이 이전에 예측한 결과를 재평가하게 하여 정확성을 높이고, 자기 일관성 기법은 동일한 프롬프트를 여러 번 실행하여 일관된 출력을 도출합니다.
   - 또한, 다중 에이전트 토론(MAD) 프레임워크를 통해 모델이 다양한 관점을 고려하고, 더 깊이 있는 분석을 수행할 수 있도록 합니다.

이러한 방법론을 통해 STIGMAD는 임상 문서에서 낙인 언어를 효과적으로 탐지할 수 있는 가능성을 보여주며, 향후 공정한 임상 NLP 시스템 개발에 기여할 수 있는 기반을 마련합니다.

---




This paper introduces and evaluates a multi-agent debate framework called STIGMAD for detecting stigmatizing language in clinical documentation. STIGMAD leverages open-source large language models (LLMs) to identify stigmatizing language, focusing on self-consistency, self-reflection, and multi-agent debate in the process.

1. **Model and Architecture**: 
   - The STIGMAD framework utilizes two primary LLMs: MedGemma (a domain-specific model) and Llama (a general-purpose model). These models are employed in their respective frameworks to detect stigmatizing language.
   - The framework consists of four main agents: the Affirmative Agent, Negative Agent, Moderator Agent, and Judge Agent. The Affirmative Agent presents arguments supporting the topic, while the Negative Agent challenges these arguments. The Moderator Agent oversees the debate, and the Judge Agent makes the final decision.

2. **Training Data**: 
   - The study uses the MIMIC-IV dataset and the PMC-Patients summary dataset. The MIMIC-IV dataset consists of real-world clinical documents, while the PMC-Patients dataset includes patient summary information. These datasets are utilized to evaluate the presence of stigmatizing language.

3. **Special Techniques**: 
   - STIGMAD employs self-reflection and self-consistency techniques to validate the model's outputs and enhance the accuracy of stigmatizing language detection. The self-reflection technique allows the model to reassess its previous predictions, improving accuracy, while the self-consistency technique runs the same prompt multiple times to derive consistent outputs.
   - Additionally, the multi-agent debate (MAD) framework enables the model to consider diverse perspectives and conduct deeper analyses.

Through these methodologies, STIGMAD demonstrates the potential to effectively detect stigmatizing language in clinical documentation, laying the groundwork for the development of more equitable clinical NLP systems in the future.


<br/>
# Results



이 연구에서는 STIGMAD 프레임워크를 사용하여 임상 문서에서 낙인 언어를 탐지하는 성능을 평가했습니다. 실험은 두 가지 데이터셋, 즉 MIMIC-IV와 PMC-Patients를 사용하여 진행되었습니다. 각 데이터셋에서 다양한 모델을 비교하여 낙인 언어 탐지의 정확성과 효율성을 평가했습니다.

#### 경쟁 모델
연구에서는 두 가지 주요 모델인 MedGemma와 Llama를 사용했습니다. MedGemma는 의료 분야에 특화된 대형 언어 모델로, Llama는 일반 목적의 대형 언어 모델입니다. 이 두 모델은 STIGMAD 프레임워크의 다양한 구성 요소와 함께 사용되어 낙인 언어 탐지 성능을 비교했습니다.

#### 테스트 데이터
MIMIC-IV 데이터셋은 4,710개의 비식별화된 임상 노트를 포함하고 있으며, PMC-Patients 데이터셋은 167,000개의 환자 요약을 포함합니다. 각 데이터셋에서 낙인 언어가 포함된 문서의 비율과 그 분포를 분석했습니다.

#### 메트릭
성능 평가는 F1 점수와 정확도(Accuracy)로 측정되었습니다. F1 점수는 모델의 정밀도와 재현율의 조화 평균으로, 낙인 언어 탐지의 성능을 종합적으로 평가하는 데 유용합니다. 정확도는 모델이 올바르게 예측한 비율을 나타냅니다.

#### 비교
STIGMAD 프레임워크는 기존의 규칙 기반 모델과 감독 학습 모델과 비교하여 우수한 성능을 보였습니다. MedGemma는 MIMIC-IV 데이터셋에서 STIGMAD 프레임워크를 사용할 때 가장 높은 성능을 기록했으며, Llama는 자기 일관성 프레임워크에서 더 나은 결과를 보였습니다. 연구 결과에 따르면, STIGMAD 프레임워크는 낙인 언어 탐지에서 더 높은 정확성과 신뢰성을 제공하며, 이는 임상 NLP 시스템의 공정성을 높이는 데 기여할 수 있습니다.




In this study, the performance of the STIGMAD framework for detecting stigmatizing language in clinical documentation was evaluated. Experiments were conducted using two datasets: MIMIC-IV and PMC-Patients. Various models were compared to assess the accuracy and efficiency of stigmatizing language detection.

#### Competing Models
The study utilized two primary models, MedGemma and Llama. MedGemma is a large language model specialized for the medical domain, while Llama is a general-purpose large language model. These two models were used in conjunction with different components of the STIGMAD framework to compare their performance in detecting stigmatizing language.

#### Test Data
The MIMIC-IV dataset includes 4,710 de-identified clinical notes, while the PMC-Patients dataset contains 167,000 patient summaries. The analysis focused on the proportion of documents containing stigmatizing language and its distribution within each dataset.

#### Metrics
Performance evaluation was measured using F1 scores and accuracy. The F1 score is the harmonic mean of precision and recall, making it useful for comprehensively assessing the performance of stigmatizing language detection. Accuracy indicates the proportion of correct predictions made by the model.

#### Comparison
The STIGMAD framework demonstrated superior performance compared to existing rule-based and supervised learning models. MedGemma achieved the highest performance when using the STIGMAD framework on the MIMIC-IV dataset, while Llama showed better results with the self-consistency framework. The findings suggest that the STIGMAD framework provides higher accuracy and reliability in detecting stigmatizing language, contributing to the fairness of clinical NLP systems.


<br/>
# 예제



이 연구에서는 임상 문서에서 낙인 언어를 탐지하기 위해 STIGMAD라는 프레임워크를 도입하고 평가합니다. STIGMAD는 여러 개의 에이전트가 참여하는 토론 방식(multi-agent debate), 자기 반성(self-reflection), 자기 일관성(self-consistency) 등을 활용하여 낙인 언어를 식별합니다. 이 프레임워크는 오픈 소스 대형 언어 모델(LLM)을 사용하여 임상 문서에서의 편향을 탐지하는 데 도움을 줍니다.

#### 예시: 트레이닝 데이터와 테스트 데이터

1. **트레이닝 데이터**:
   - **입력**: 임상 문서의 특정 구문(예: "환자는 치료에 비협조적이다.").
   - **출력**: 
     - **바이어스 유형**: "Stigmatizing Language"
     - **인용문**: "비협조적이다."

   이 데이터는 모델이 특정 문장에서 낙인 언어를 식별하도록 훈련하는 데 사용됩니다.

2. **테스트 데이터**:
   - **입력**: 새로운 임상 문서의 구문(예: "환자는 약물 복용에 대해 비협조적이었다.").
   - **출력**: 
     - **바이어스 유형**: "Stigmatizing Language"
     - **인용문**: "비협조적이었다."

이러한 방식으로 모델은 훈련 데이터에서 학습한 내용을 바탕으로 새로운 데이터에서 낙인 언어를 탐지할 수 있습니다.

#### 구체적인 테스크
- **바이어스 식별**: 주어진 임상 문서에서 낙인 언어를 식별하고, 해당 언어가 어떤 유형의 편향을 나타내는지 분류합니다.
- **자기 반성**: 모델이 이전에 예측한 바이어스 유형이 올바른지 재평가합니다.
- **자기 일관성**: 동일한 입력에 대해 여러 번 모델을 실행하여 일관된 출력을 확인합니다.
- **다중 에이전트 토론**: 여러 에이전트가 주어진 바이어스 유형에 대해 토론하고, 최종 결정을 내립니다.

이러한 프로세스를 통해 STIGMAD 프레임워크는 임상 문서에서의 낙인 언어 탐지의 정확성을 높이고, 편향을 줄이는 데 기여할 수 있습니다.

---




This study introduces and evaluates a framework called STIGMAD for detecting stigmatizing language in clinical documentation. STIGMAD leverages multi-agent debate, self-reflection, and self-consistency to identify stigmatizing language, utilizing open-source large language models (LLMs) to assist in uncovering bias in clinical texts.

#### Example: Training Data and Test Data

1. **Training Data**:
   - **Input**: A specific phrase from a clinical document (e.g., "The patient is non-compliant with treatment.").
   - **Output**: 
     - **Bias Type**: "Stigmatizing Language"
     - **Quote**: "non-compliant."

   This data is used to train the model to identify stigmatizing language in specific sentences.

2. **Test Data**:
   - **Input**: A new phrase from a clinical document (e.g., "The patient was non-compliant with medication.").
   - **Output**: 
     - **Bias Type**: "Stigmatizing Language"
     - **Quote**: "non-compliant."

In this way, the model can detect stigmatizing language in new data based on what it learned from the training data.

#### Specific Tasks
- **Bias Identification**: Identify stigmatizing language in a given clinical document and classify the type of bias it represents.
- **Self-Reflection**: Re-evaluate whether the previously predicted bias type is correct.
- **Self-Consistency**: Run the model multiple times on the same input to check for consistent outputs.
- **Multi-Agent Debate**: Engage multiple agents in a discussion about the given bias type and reach a final decision.

Through these processes, the STIGMAD framework aims to enhance the accuracy of detecting stigmatizing language in clinical documentation and contribute to reducing bias.

    <br/>
    # 요약
    **한국어 요약:** 본 연구에서는 StigMAD 프레임워크를 통해 오픈소스 대형 언어 모델을 활용하여 임상 문서에서 낙인 언어를 탐지하는 방법을 제안하였다. 실험 결과, StigMAD는 규칙 기반 및 감독 학습 모델에 비해 유의미한 성능 향상을 보였으며, 특히 MedGemma 모델이 가장 높은 성능을 기록하였다. 이 연구는 임상 NLP 시스템에서 공정성을 높이기 위한 중요한 단계를 제시한다.

**English Summary:** This study introduces the StigMAD framework, which utilizes open-source large language models to detect stigmatizing language in clinical documentation. Experimental results demonstrate that StigMAD provides significant performance improvements over rule-based and supervised models, with the MedGemma model achieving the highest performance. This research marks a critical step toward enhancing equity in clinical NLP systems.

<br/>
# 기타



1. **다이어그램 및 피규어**:
   - **Figure 1**: Bias Detection Methodological Framework
     - 이 다이어그램은 연구에서 제안한 STIGMAD 프레임워크의 구조를 시각적으로 나타냅니다. STIGMAD는 여러 에이전트를 활용하여 임상 문서에서 낙인 언어를 탐지하는 과정을 설명합니다. 이 프레임워크는 각 에이전트가 특정 역할을 수행하며, 이를 통해 더 정교한 논의와 결정을 이끌어냅니다.
   
   - **Figure 2**: Proposed STIGMAD Framework
     - STIGMAD 프레임워크의 구성 요소와 각 에이전트의 역할을 보여줍니다. 이 프레임워크는 긍정적 에이전트, 부정적 에이전트, 중재자 에이전트, 판사 에이전트로 구성되어 있으며, 각 에이전트는 논의의 특정 측면을 담당합니다. 이 구조는 낙인 언어 탐지의 정확성을 높이는 데 기여합니다.

2. **테이블**:
   - **Table 1**: Performance across models
     - 이 표는 다양한 모델과 프레임워크에서의 성능을 비교합니다. MedGemma와 Llama 모델의 F1 점수와 정확도를 보여주며, 각 프레임워크의 효과를 평가합니다. MedGemma는 STIGMAD 프레임워크를 사용할 때 더 높은 성능을 보였고, Llama는 자기 일관성 프레임워크에서 더 나은 결과를 나타냈습니다. 이는 특정 모델이 특정 프레임워크에서 더 잘 작동할 수 있음을 시사합니다.

3. **어펜딕스**:
   - 어펜딕스에는 다양한 프롬프트 템플릿이 포함되어 있습니다. 이 프롬프트들은 LLM이 임상 문서에서 낙인 언어를 탐지하고 평가하는 데 사용됩니다. 각 프롬프트는 특정 작업을 수행하도록 설계되어 있으며, LLM의 출력을 개선하는 데 기여합니다.




1. **Diagrams and Figures**:
   - **Figure 1**: Bias Detection Methodological Framework
     - This diagram visually represents the structure of the proposed STIGMAD framework in the study. STIGMAD utilizes multiple agents to detect stigmatizing language in clinical documentation. The framework illustrates how each agent performs specific roles, leading to more refined discussions and decisions.

   - **Figure 2**: Proposed STIGMAD Framework
     - This figure shows the components of the STIGMAD framework and the roles of each agent. The framework consists of an Affirmative Agent, Negative Agent, Moderator Agent, and Judge Agent, with each agent responsible for a specific aspect of the debate. This structure contributes to enhancing the accuracy of stigmatizing language detection.

2. **Tables**:
   - **Table 1**: Performance across models
     - This table compares the performance of various models and frameworks. It displays the F1 scores and accuracies of the MedGemma and Llama models, evaluating the effectiveness of each framework. MedGemma showed higher performance when using the STIGMAD framework, while Llama demonstrated better results with the self-consistency framework. This suggests that certain models may perform better with specific frameworks.

3. **Appendix**:
   - The appendix includes various prompt templates. These prompts are used to guide LLMs in detecting and evaluating stigmatizing language in clinical documents. Each prompt is designed to perform a specific task, contributing to the improvement of LLM outputs.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{Dahal2026,
  author    = {Rajashree Dahal and Pardis Hossein Pour and Pranathi Kamisetty and Satwik Pamulaparthy and Saeid Tizpaz-Niari and Natalie Parde},
  title     = {Investigating Stigmatizing Language in Clinical Documentation with Open-Source Large Language Models},
  booktitle = {Proceedings of the 25th Workshop on Biomedical Language Processing (BioNLP 2026)},
  pages     = {490--501},
  year      = {2026},
  month     = {July},
  publisher = {Association for Computational Linguistics},
}
```

### 시카고 스타일

Rajashree Dahal, Pardis Hossein Pour, Pranathi Kamisetty, Satwik Pamulaparthy, Saeid Tizpaz-Niari, and Natalie Parde. "Investigating Stigmatizing Language in Clinical Documentation with Open-Source Large Language Models." In *Proceedings of the 25th Workshop on Biomedical Language Processing (BioNLP 2026)*, 490–501. July 3-4, 2026. Association for Computational Linguistics. 
    