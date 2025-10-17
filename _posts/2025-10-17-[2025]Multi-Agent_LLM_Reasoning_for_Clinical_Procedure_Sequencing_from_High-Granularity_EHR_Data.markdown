---
layout: post
title:  "[2025]Multi-Agent LLM Reasoning for Clinical Procedure Sequencing from High-Granularity EHR Data"
date:   2025-10-17 16:34:27 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 다중 전문 분야의 LLM 에이전트를 활용하여 환자의 전자 건강 기록(EHR) 데이터를 기반으로 의료 절차의 순서를 추천하는 다중 에이전트 프레임워크를 제안합니다. (RAG 사용, MIMIC-III 데이터 사용


짧은 요약(Abstract) :


이 논문에서는 실제 임상 결정이 협업을 통해 이루어진다는 점을 반영하기 위해 다중 에이전트 프레임워크를 도입합니다. 각 대형 언어 모델(LLM) 에이전트는 특정 전문 분야(예: 심장학, 내분비학)를 대표하며, 환자 절차의 순서를 협력적으로 추천합니다. 각 에이전트는 도메인별 PubMed 문헌의 개인 지식 기반, 환자의 전자 건강 기록(EHR)을 포함한 공유 메모리, 그리고 진행 중인 상호 에이전트 논의의 이력을 사용하여 추론합니다. 우리는 만장일치 합의가 필요한 조직 모델과 지정된 팀 리더가 최종 결정을 내리는 모델 두 가지를 시뮬레이션합니다. MIMIC-III 데이터셋을 사용하여 절차 순서를 예측한 결과, 리더 기반 모델이 만장일치 및 단일 에이전트 구성보다 일관되게 더 높은 정확도를 달성했습니다. 이 연구는 다중 에이전트 임상 의사 결정 지원을 위한 강력하고 해석 가능한 패러다임을 제시하며, AI가 임상 실습의 협력적 특성과 더 밀접하게 일치하도록 합니다.


This paper introduces a multi-agent framework to better reflect the collaborative nature of real-world clinical decisions. Each large language model (LLM) agent embodies a distinct specialty (e.g., cardiology, endocrinology) and collaboratively recommends a sequence of patient procedures. Each agent reasons using a private knowledge base of domain-specific PubMed literature, a shared memory containing the patient's electronic health records (EHRs), and the history of the ongoing inter-agent discussion. We simulate two organizational models: one requiring unanimous consensus and another where a designated team leader makes the final call. Using the MIMIC-III dataset to predict procedural sequences, we show that the leader-based model consistently outperforms both the consensus and single-agent configurations, achieving higher accuracy. Our work presents a robust and interpretable paradigm for multi-agent clinical decision support, aligning AI more closely with the collaborative nature of clinical practice.


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



이 연구에서는 전자 건강 기록(EHR) 데이터를 기반으로 한 임상 절차 추천을 위한 다중 에이전트 시스템을 제안합니다. 이 시스템은 각기 다른 전문 분야를 대표하는 대형 언어 모델(LLM) 에이전트들이 협력하여 환자에게 필요한 절차의 순서를 추천하는 구조로 설계되었습니다. 

#### 1. 모델 아키텍처
다중 에이전트 시스템은 각 에이전트가 특정 의료 분야(예: 심장학, 내분비학)에 대한 전문 지식을 갖추고 있으며, 이들은 환자의 EHR과 PubMed 문헌을 기반으로 독립적으로 추론합니다. 각 에이전트는 다음과 같은 세 가지 주요 구성 요소를 포함하는 개인 메모리 모듈을 유지합니다:
- **과학 문헌 해석**: 에이전트는 최신 연구 결과를 바탕으로 환자에게 적합한 절차를 추천하기 위해 PubMed에서 검색한 정보를 요약합니다.
- **상호작용 기록**: 다른 에이전트와의 논의 이력을 기록하여, 이전의 논의 내용을 바탕으로 더 나은 결정을 내릴 수 있도록 합니다.
- **EHR 정보**: 환자의 임상 상태에 대한 세부 정보를 포함하여, 에이전트가 환자의 현재 상황을 이해하고 적절한 절차를 추천할 수 있도록 합니다.

#### 2. 훈련 데이터
이 연구는 MIMIC-III 데이터셋을 사용하여 1,000명의 환자에 대한 데이터를 분석합니다. 이 데이터셋은 약 42,000명의 고유 환자로부터 수집된 EHR 정보를 포함하고 있으며, 각 환자의 입원 기록에 따라 절차 코드가 순차적으로 기록되어 있습니다. 연구팀은 이 데이터셋을 기반으로 환자의 임상 상태를 평가하고, 각 절차의 적절성을 판단하기 위해 ICD-9 절차 코드를 사용합니다.

#### 3. 특별한 기법
- **Retrieval-Augmented Generation (RAG)**: 각 에이전트는 RAG 기법을 사용하여, 환자의 EHR과 논의 맥락에 따라 가장 관련성이 높은 문헌 조각을 검색하고 이를 요약하여 추천에 활용합니다. 이 과정은 에이전트가 최신 연구 결과를 기반으로 결정을 내릴 수 있도록 지원합니다.
- **의사 결정 구조**: 두 가지 의사 결정 모델을 실험합니다. 하나는 모든 에이전트가 만장일치로 동의해야 하는 합의 기반 모델이고, 다른 하나는 지정된 팀 리더가 최종 결정을 내리는 리더 기반 모델입니다. 연구 결과, 리더 기반 모델이 더 높은 정확도를 보였습니다.

이러한 구조는 실제 임상 환경에서의 협업을 모사하며, 다중 전문 분야의 의견을 통합하여 더 나은 임상 결정을 내릴 수 있도록 합니다.

---




This study proposes a multi-agent system for clinical procedure recommendation based on electronic health record (EHR) data. The system is designed such that large language model (LLM) agents, each representing a different specialty (e.g., cardiology, endocrinology), collaboratively recommend a sequence of procedures needed for the patient.

#### 1. Model Architecture
The multi-agent system consists of agents that possess expertise in specific medical fields, and they independently reason based on the patient's EHR and PubMed literature. Each agent maintains a private memory module that includes three main components:
- **Interpretation of Scientific Literature**: Agents summarize information retrieved from PubMed to recommend appropriate procedures based on the latest research findings.
- **Interaction History**: They keep a record of discussions with other agents, allowing them to make better decisions based on previous dialogues.
- **EHR Information**: This includes detailed information about the patient's clinical status, enabling agents to understand the current situation and recommend suitable procedures.

#### 2. Training Data
The study utilizes the MIMIC-III dataset, analyzing data from 1,000 patients. This dataset contains EHR information collected from approximately 42,000 unique patients, with procedure codes recorded sequentially according to each patient's admission history. The research team uses this dataset to assess the clinical status of patients and determine the appropriateness of procedures using ICD-9 procedure codes.

#### 3. Special Techniques
- **Retrieval-Augmented Generation (RAG)**: Each agent employs the RAG technique to search for the most relevant pieces of literature based on the patient's EHR and discussion context, summarizing this information to inform their recommendations. This process supports agents in making decisions grounded in the latest research.
- **Decision-Making Structure**: Two decision-making models are experimented with: one requiring unanimous agreement among all agents (consensus-based model) and another where a designated team leader makes the final decision (leader-based model). The results indicate that the leader-based model consistently achieves higher accuracy.

This structure mimics collaboration in real clinical environments, integrating diverse expert opinions to facilitate better clinical decision-making.


<br/>
# Results



이 연구에서는 다중 에이전트 시스템을 활용하여 임상 절차의 순서를 예측하는 성능을 평가하였습니다. 실험은 MIMIC-III 데이터셋을 기반으로 하였으며, 다양한 언어 모델을 사용하여 세 가지 결정 구조(독립형, 합의 기반, 리더 결정)를 비교했습니다.

1. **경쟁 모델**: 
   - **GPT-4.1**: OpenAI의 최신 모델로, 높은 토큰 수를 지원합니다.
   - **DeepSeek-R1**: Qwen2.5에서 증류된 모델로, 긴 컨텍스트를 효율적으로 처리할 수 있도록 설계되었습니다.
   - **OpenBioLLM**: 여러 의료 QA 데이터셋으로 훈련된 LLaMA 3.2의 도메인 적응 버전입니다.
   - **MedGemma**: 의료 작업에 적합하도록 조정된 일반 목적의 Gemma 모델입니다.

2. **테스트 데이터**: 
   - MIMIC-III 데이터셋에서 무작위로 선택된 1,000명의 환자 데이터를 사용하였습니다. 각 환자의 입원 기록에 따라 절차 코드가 순차적으로 기록되어 있습니다.

3. **메트릭**: 
   - **성공률(Success Rate)**: 에이전트가 유효한 절차 코드를 생성한 입원 기록의 비율.
   - **평균 라운드 수(Average Number of Rounds)**: 결정을 내리기 위해 필요한 평균 토론 라운드 수.
   - **전문가 신뢰도(Confidence)**: 각 라운드에서 에이전트가 보고한 신뢰도 점수의 평균.
   - **정확도(Mean Reciprocal Rank, MRR)**: 예측된 절차 코드의 순위와 실제 코드 간의 일치 정도를 평가합니다.

4. **비교 결과**:
   - **독립형 구성**: GPT-4.1 모델이 98.3%의 성공률을 기록하였고, 평균 1회의 라운드로 결정을 내렸습니다. 그러나 신뢰도는 상대적으로 낮았습니다.
   - **합의 기반 구성**: GPT-4.1의 성공률은 90.7%였으며, 평균 2.4회의 라운드가 필요했습니다. 신뢰도는 높았지만, 결정이 느렸습니다.
   - **리더 결정 구성**: GPT-4.1은 99.4%의 성공률을 기록하며, 평균 1회의 라운드로 결정을 내렸습니다. 이 구성은 가장 높은 신뢰도를 보였습니다.

결과적으로, 리더 기반 결정 구조가 합의 기반 및 독립형 구성보다 우수한 성능을 보였으며, 이는 임상 팀의 협업 구조를 잘 반영한 결과입니다. 또한, 일반 목적의 LLM이 의료 도메인에 특화된 모델보다 더 나은 성능을 보였다는 점이 주목할 만합니다.

---




In this study, the performance of a multi-agent system for predicting the sequence of clinical procedures was evaluated. The experiments were based on the MIMIC-III dataset and compared various language models across three decision structures (independent, consensus-based, and leader decision).

1. **Competing Models**:
   - **GPT-4.1**: The latest model from OpenAI, capable of handling a high number of tokens.
   - **DeepSeek-R1**: A model distilled from Qwen2.5, designed for efficient long-context reasoning.
   - **OpenBioLLM**: A domain-adapted variant of LLaMA 3.2 trained on multiple medical QA datasets.
   - **MedGemma**: A fine-tuned variant of the general-purpose Gemma model for medical tasks.

2. **Test Data**:
   - A random subset of 1,000 patients from the MIMIC-III dataset was used. Procedure codes were recorded sequentially based on each patient's admission records.

3. **Metrics**:
   - **Success Rate**: The percentage of admissions for which the agents successfully generated at least one valid procedure code.
   - **Average Number of Rounds**: The average number of discussion rounds required to reach a decision.
   - **Expert Confidence**: The average self-reported confidence score across all rounds.
   - **Mean Reciprocal Rank (MRR)**: Evaluates the degree of alignment between predicted procedure codes and actual codes.

4. **Comparison Results**:
   - **Independent Configuration**: The GPT-4.1 model achieved a success rate of 98.3% with an average of 1 round to make a decision, but the confidence level was relatively low.
   - **Consensus-Based Configuration**: GPT-4.1 had a success rate of 90.7%, requiring an average of 2.4 rounds. While confidence was high, the decision-making was slower.
   - **Leader Decision Configuration**: GPT-4.1 recorded a success rate of 99.4% with an average of 1 round to make a decision, showing the highest confidence levels.

Overall, the leader-based decision structure consistently outperformed both consensus-based and independent configurations, reflecting the collaborative nature of clinical teams. Additionally, general-purpose LLMs outperformed models fine-tuned for the medical domain, highlighting the importance of architectural flexibility in handling heterogeneous data.


<br/>
# 예제



이 논문에서는 전자 건강 기록(EHR) 데이터를 기반으로 한 임상 절차 순서를 추천하기 위해 다중 에이전트 시스템을 제안합니다. 이 시스템은 각기 다른 전문 분야를 가진 대형 언어 모델(LLM) 에이전트들이 협력하여 환자에게 필요한 절차를 추천하는 방식으로 작동합니다. 

#### 트레이닝 데이터와 테스트 데이터

1. **트레이닝 데이터**: 
   - **데이터셋**: MIMIC-III 데이터셋을 사용합니다. 이 데이터셋은 약 42,000명의 고유 환자에 대한 EHR 정보를 포함하고 있습니다.
   - **샘플링**: 이 연구에서는 1,000명의 환자를 무작위로 선택하여 분석의 기초로 삼습니다. 이 샘플은 전체 데이터셋의 분포와 유사한 분포를 유지합니다.
   - **입력 데이터**: 각 환자의 입원 기록에 대해 ICD-9 절차 코드를 사용하여 예측할 절차의 순서를 정의합니다. 이 절차 코드는 순차적으로 기록되며, 각 절차는 고유한 순서 위치를 가집니다.
   - **출력 데이터**: 각 환자에 대해 실제로 수행된 절차의 순서가 ground truth로 사용됩니다.

2. **테스트 데이터**: 
   - **입력 데이터**: 테스트 데이터는 동일한 MIMIC-III 데이터셋에서 무작위로 선택된 환자들로 구성됩니다. 각 환자의 EHR에서 수집된 다양한 정보(예: 차트 이벤트, 입력 이벤트, 출력 이벤트, 실험실 결과 등)를 포함합니다.
   - **출력 데이터**: 에이전트들이 추천한 절차의 순서가 실제 수행된 절차의 순서와 비교됩니다. 이 과정에서 에이전트는 각 시간 창에서 다음 절차를 추천합니다.

#### 구체적인 작업(Task)

- **작업 목표**: 각 환자에 대해 적절한 절차의 순서를 추천하는 것입니다. 이를 위해 에이전트들은 환자의 EHR을 분석하고, 각 전문 분야에 대한 지식을 바탕으로 논의합니다.
- **결정 구조**: 두 가지 결정 구조를 사용합니다. 하나는 모든 에이전트가 만장일치로 동의해야 하는 합의 기반 모델이고, 다른 하나는 지정된 팀 리더가 최종 결정을 내리는 리더 기반 모델입니다.
- **성능 평가**: 에이전트의 추천 정확도를 평가하기 위해 Mean Reciprocal Rank(MRR)와 같은 메트릭을 사용합니다. 이 메트릭은 추천된 절차의 순서가 실제 수행된 절차와 얼마나 잘 일치하는지를 측정합니다.




This paper proposes a multi-agent system for recommending clinical procedure sequences based on electronic health record (EHR) data. The system operates by having large language model (LLM) agents, each with distinct specialties, collaborate to recommend necessary procedures for patients.

#### Training Data and Test Data

1. **Training Data**:
   - **Dataset**: The MIMIC-III dataset is utilized, which contains EHR information for approximately 42,000 unique patients.
   - **Sampling**: For this study, a random subset of 1,000 patients is selected to serve as the basis for analysis. This subset maintains a distribution similar to that of the full dataset.
   - **Input Data**: For each patient admission, ICD-9 procedure codes are used to define the target sequences for prediction. These procedure codes are recorded in sequential order, with each procedure assigned a unique ordinal position.
   - **Output Data**: The actual sequence of procedures performed for each patient serves as the ground truth.

2. **Test Data**:
   - **Input Data**: The test data consists of patients randomly selected from the same MIMIC-III dataset. It includes various information collected from each patient's EHR (e.g., chart events, input events, output events, laboratory results, etc.).
   - **Output Data**: The sequence of procedures recommended by the agents is compared to the actual procedures performed. In this process, agents sequentially recommend the next procedure for each time window.

#### Specific Task

- **Task Objective**: The goal is to recommend an appropriate sequence of procedures for each patient. To achieve this, agents analyze the patient's EHR and engage in discussions based on their knowledge of specific specialties.
- **Decision Structure**: Two decision structures are employed. One requires unanimous agreement among all agents (consensus-based model), while the other has a designated team leader making the final decision (leader-based model).
- **Performance Evaluation**: Metrics such as Mean Reciprocal Rank (MRR) are used to assess the accuracy of the agents' recommendations. This metric measures how well the recommended sequence of procedures aligns with the actual procedures performed.

<br/>
# 요약


이 논문에서는 다중 전문 분야의 LLM 에이전트를 활용하여 환자의 전자 건강 기록(EHR) 데이터를 기반으로 의료 절차의 순서를 추천하는 다중 에이전트 프레임워크를 제안합니다. MIMIC-III 데이터셋을 사용하여 리더 기반 모델이 합의 기반 및 단일 에이전트 구성보다 더 높은 정확도를 달성함을 보여주었으며, 이는 임상 의사 결정의 협업적 본질을 반영합니다. 각 에이전트는 특정 전문 분야에 대한 지식을 바탕으로 환자의 EHR을 분석하고, 관련 문헌을 검색하여 추천 절차를 제안합니다.

---

This paper proposes a multi-agent framework utilizing LLM agents from various specialties to recommend sequences of medical procedures based on patient electronic health records (EHR). Using the MIMIC-III dataset, it demonstrates that the leader-based model consistently outperforms both consensus-based and single-agent configurations in terms of accuracy, reflecting the collaborative nature of clinical decision-making. Each agent analyzes the patient's EHR and retrieves relevant literature to suggest recommended procedures based on their domain-specific knowledge.

<br/>
# 기타


1. **다이어그램 및 피규어**
   - **Multi-Agent Interaction Environment (Figure 1)**: 이 다이어그램은 독립적인 결정, 합의 기반 결정, 팀 리더 결정의 세 가지 상호작용 모델을 비교합니다. 각 모델의 장단점을 시각적으로 나타내어, 팀 리더 모델이 더 나은 성과를 내는 이유를 설명합니다.
   - **Internal Memory Analysis (Figure 3)**: 각 전문가 에이전트의 개인 메모리 구조를 보여줍니다. 이 메모리는 과거의 논의 이력, 환자의 EHR 정보, 그리고 문헌에서의 해석을 포함하여 에이전트가 독립적으로 추론할 수 있도록 돕습니다. 이는 에이전트가 환자 상황에 맞는 질문을 생성하고, 관련 문헌을 검색하여 더 나은 결정을 내릴 수 있도록 합니다.
   - **Interaction Analysis (Figure 6)**: 여러 에이전트 간의 상호작용을 보여주는 예시로, 각 에이전트가 다른 에이전트의 의견에 어떻게 반응하는지를 강조합니다. 이는 다중 전문 분야의 협업이 어떻게 이루어지는지를 잘 보여줍니다.

2. **테이블**
   - **Completion and Correctness Performance (Table 2)**: 다양한 결정 구성 및 모델 유형에 따른 성공률, 평균 라운드 수, 전문가의 평균 신뢰도, 그리고 MRR(Mean Reciprocal Rank) 점수를 비교합니다. 팀 리더 기반 결정이 다른 구성보다 높은 성공률과 정확도를 보이며, 이는 팀 리더 모델이 더 효과적인 의사결정 구조임을 나타냅니다.
   - **Error Analysis (Table 3)**: 에이전트의 오류 유형을 분석하여, 잘못된 출력 형식, 알파벳 문자가 포함된 코드, 존재하지 않는 숫자 코드의 비율을 보여줍니다. MedGemma 모델이 가장 높은 비율의 오류를 보이며, 이는 도메인 특정 미세 조정 후의 기억 상실을 나타냅니다.

3. **어펜딕스**
   - 어펜딕스에는 연구에 사용된 데이터셋, 모델, 실험 설정 및 평가 방법에 대한 자세한 설명이 포함되어 있습니다. 이는 연구의 재현성을 높이고, 다른 연구자들이 유사한 접근 방식을 사용할 수 있도록 돕습니다.

### Insights
- **협업의 중요성**: 다중 에이전트 시스템이 단일 전문가보다 더 나은 성과를 내는 이유는 다양한 전문 지식이 결합되어 더 포괄적이고 정보에 기반한 결정을 내릴 수 있기 때문입니다.
- **리더 기반 모델의 우수성**: 팀 리더 모델이 합의 기반 모델보다 더 높은 정확도와 신뢰도를 보이는 것은 실제 임상 환경에서의 의사결정 구조를 잘 반영하고 있음을 나타냅니다.
- **오류 분석의 필요성**: 모델의 오류 유형을 분석함으로써, 향후 개선 방향을 제시하고, 특정 모델의 한계를 이해하는 데 도움이 됩니다.

---




1. **Diagrams and Figures**
   - **Multi-Agent Interaction Environment (Figure 1)**: This diagram compares three interaction models: independent decision-making, consensus-based decision-making, and team leader decision-making. It visually represents the strengths and weaknesses of each model, explaining why the team leader model yields better performance.
   - **Internal Memory Analysis (Figure 3)**: This shows the private memory structure of each expert agent. This memory includes past discussion history, patient EHR information, and interpretations from literature, enabling agents to reason independently. It helps agents generate context-specific questions and search relevant literature for better decision-making.
   - **Interaction Analysis (Figure 6)**: An example of interaction among multiple agents, highlighting how each agent responds to the opinions of others. This effectively illustrates how interdisciplinary collaboration occurs.

2. **Tables**
   - **Completion and Correctness Performance (Table 2)**: Compares success rates, average rounds, average expert confidence, and Mean Reciprocal Rank (MRR) scores across different decision configurations and model types. The leader-based decision consistently shows higher success rates and accuracy, indicating that it is a more effective decision-making structure.
   - **Error Analysis (Table 3)**: Analyzes error types among agents, showing the proportion of invalid output formats, codes containing alphabetic characters, and numeric but non-existent codes. The MedGemma model shows the highest error rates, indicating signs of catastrophic forgetting after domain-specific fine-tuning.

3. **Appendix**
   - The appendix includes detailed descriptions of the datasets, models, experimental setups, and evaluation methods used in the study. This enhances the reproducibility of the research and helps other researchers apply similar approaches.

### Insights
- **Importance of Collaboration**: The reason multi-agent systems outperform single experts is that the combination of diverse expertise leads to more comprehensive and informed decisions.
- **Superiority of Leader-Based Models**: The higher accuracy and confidence of the team leader model compared to consensus-based models reflect a decision-making structure that closely aligns with real clinical environments.
- **Need for Error Analysis**: Analyzing the types of errors made by models helps suggest future improvements and understand the limitations of specific models.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{zhong2025multi,
  author = {Yishan Zhong and Wenqi Shi and Ben Tamo Jr. and Micky Nnamdi and Yining Yuan and May D Wang},
  title = {Multi-Agent LLM Reasoning for Clinical Procedure Sequencing from High-Granularity EHR Data},
  booktitle = {Proceedings of the 16th ACM International Conference on Bioinformatics, Computational Biology, and Health Informatics (BCB '25)},
  year = {2025},
  month = {October},
  location = {Philadelphia, PA, USA},
  publisher = {ACM},
  pages = {1--12},
  doi = {10.1145/3765612.3767238},
  url = {https://doi.org/10.1145/3765612.3767238}
}
```

### 시카고 스타일

Yishan Zhong, Wenqi Shi, Ben Tamo Jr., Micky Nnamdi, Yining Yuan, and May D Wang. 2025. "Multi-Agent LLM Reasoning for Clinical Procedure Sequencing from High-Granularity EHR Data." In *Proceedings of the 16th ACM International Conference on Bioinformatics, Computational Biology, and Health Informatics (BCB '25)*, 1-12. Philadelphia, PA, USA: ACM. https://doi.org/10.1145/3765612.3767238.
