---
layout: post
title:  "[2025]An LLM chatbot to facilitate primary-to-specialist care transitions: a randomized controlled trial"
date:   2026-01-20 17:50:16 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 연구에서는 PreA라는 LLM 챗봇을 개발하여 2,069명의 환자를 대상으로 3개의 그룹(PreA-only, PreA-human, No-PreA)으로 무작위 배정하여 1차 진료에서 전문의로의 전환을 평가하였다.


진단가능목록, 필요 검사 목록, 히스토리 요약으로 평가    


짧은 요약(Abstract) :



이 연구에서는 1차 진료에서 전문 진료로의 전환을 원활하게 하기 위해 설계된 대형 언어 모델(LLM) 챗봇인 PreA의 효과를 평가했습니다. PreA는 환자와 의료 제공자 간의 의사소통을 개선하고, 진료 시간을 단축시키며, 환자 중심의 치료를 강화하는 데 기여할 수 있는 잠재력을 가지고 있습니다. 이 연구는 24개 의료 분야의 111명의 전문의와 2,069명의 환자를 대상으로 한 무작위 대조 시험을 통해 진행되었습니다. 결과적으로, PreA를 독립적으로 사용한 그룹은 전문의와의 상담 시간이 28.7% 단축되었고, 의사-환자 간의 의사소통 용이성 및 치료 조정에 대한 인식이 유의미하게 향상되었습니다. PreA는 지역 이해관계자와의 공동 설계를 통해 개발되었으며, 이는 LLM을 효과적으로 배포하는 전략으로 확인되었습니다.



This study evaluated the effectiveness of PreA, a large language model (LLM) chatbot designed to facilitate transitions from primary to specialist care. PreA has the potential to improve communication between patients and healthcare providers, reduce consultation times, and enhance patient-centered care. The research was conducted through a randomized controlled trial involving 111 specialists across 24 medical disciplines and 2,069 patients. The results showed that the group using PreA independently experienced a 28.7% reduction in consultation time with specialists, along with significant improvements in perceived ease of communication and care coordination. PreA was developed through co-design with local stakeholders, confirming it as an effective strategy for deploying LLMs.


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



이 연구에서 개발된 PreA(Pre-Assessment) 챗봇은 지역 이해관계자와의 공동 설계를 통해 만들어진 대규모 언어 모델(LLM) 기반의 시스템입니다. PreA는 환자와 의사 간의 소통을 개선하고, 전문의 상담을 위한 사전 평가 및 추천 보고서를 생성하는 데 중점을 두고 있습니다. 이 챗봇의 아키텍처는 두 가지 주요 구성 요소로 나뉩니다: 환자-facing 챗봇과 임상 인터페이스입니다.

1. **모델 아키텍처**:
   - **환자-facing 챗봇**: 이 챗봇은 두 단계의 임상 추론 모델을 사용합니다. 첫 번째 단계는 환자의 건강 관련 정보를 수집하기 위한 질문을 하는 '질의' 단계이며, 두 번째 단계는 수집된 정보를 바탕으로 1~3개의 관련 진단 가능성을 생성하는 '결론' 단계입니다. 이 과정에서 챗봇은 표준 의료 상담 가이드라인을 준수합니다.
   - **임상 인터페이스**: PreA는 환자 정보를 바탕으로 전문의에게 제공할 추천 보고서를 생성합니다. 이 보고서에는 환자의 인구 통계, 병력, 주요 불만 사항, 증상, 가족력, 제안된 검사, 초기 진단 및 치료 권장 사항이 포함됩니다.

2. **훈련 데이터 및 기법**:
   - PreA는 지역 사회와 임상 이해관계자와의 공동 설계를 통해 수집된 데이터로 훈련되었습니다. 이 과정에서 120명의 환자, 간병인, 지역 보건 근무자, 의사 및 간호사와의 적대적 테스트를 통해 모델을 개선했습니다. 또한, 가상의 환자 시뮬레이션을 통해 저건강 문해력 사용자의 상호작용을 최적화했습니다.
   - 데이터 수집은 중국의 11개 성에서 이루어졌으며, 수집된 대화 데이터는 엄격하게 비식별화되어 훈련에 사용되었습니다. 이 데이터는 지역의 의료 상담 상호작용을 반영하며, 다양한 사회경제적 배경을 가진 환자들의 요구를 충족시키기 위해 설계되었습니다.

3. **성능 평가**:
   - PreA의 성능은 효율성, 요구 사항 식별, 명확성, 포괄성 및 친근함의 다섯 가지 기준으로 평가되었습니다. 전문가 패널이 이 기준에 따라 모델의 성능을 평가하였으며, 각 기준에 대한 점수가 3 미만일 경우 추가적인 개선이 이루어졌습니다.

4. **무작위 대조 시험(RCT)**:
   - PreA는 다기관 무작위 대조 시험을 통해 평가되었습니다. 환자들은 PreA를 독립적으로 사용할 그룹(PreA-only), 직원 지원을 받는 그룹(PreA-human), 또는 PreA를 사용하지 않는 대조군(No-PreA)으로 무작위 배정되었습니다. 이 연구의 주요 목표는 PreA가 상담 시간 단축, 의사-환자 소통 개선 및 진료 조정에 미치는 영향을 평가하는 것이었습니다.

이러한 방법론은 PreA가 고부담의 의료 환경에서 환자 중심의 진료를 개선하고, 의료 시스템의 효율성을 높이는 데 기여할 수 있음을 보여줍니다.

---




The PreA (Pre-Assessment) chatbot developed in this study is a large language model (LLM)-based system created through co-design with local stakeholders. PreA focuses on improving communication between patients and physicians, as well as generating pre-consultation assessments and referral reports for specialists. The architecture of this chatbot is divided into two main components: a patient-facing chatbot and a clinical interface.

1. **Model Architecture**:
   - **Patient-facing Chatbot**: This chatbot employs a two-stage clinical reasoning model: the 'Inquiry' stage, where it asks questions to gather comprehensive health-related information, and the 'Conclusion' stage, where it generates 1-3 relevant diagnostic possibilities based on the collected information. This process adheres to standard medical consultation guidelines.
   - **Clinical Interface**: PreA is configured to generate referral reports for specialists based on patient information. These reports include patient demographics, medical history, chief complaints, symptoms, family history, suggested investigations, preliminary diagnoses, and treatment recommendations.

2. **Training Data and Techniques**:
   - PreA was trained on data collected through co-design with community and clinical stakeholders. This process involved adversarial testing with 120 patients, caregivers, community health workers, physicians, and nurses to refine the model. Additionally, virtual patient simulations were used to optimize interactions for low-health-literacy users.
   - Data collection occurred across 11 provinces in China, and the collected dialogue data was rigorously de-identified before being used for training. This data reflects local medical consultation interactions and was designed to meet the needs of patients from diverse socioeconomic backgrounds.

3. **Performance Evaluation**:
   - PreA's performance was evaluated across five criteria: efficiency, needs identification, clarity, comprehensiveness, and friendliness. An expert panel assessed the model's performance based on these criteria, and any scores below 3 triggered further iterative refinement.

4. **Randomized Controlled Trial (RCT)**:
   - PreA was evaluated through a multicenter randomized controlled trial. Patients were randomly assigned to one of three groups: the PreA-only group (independent use of PreA), the PreA-human group (staff-supported use of PreA), or the No-PreA control group (standard care). The primary aim of this study was to assess the impact of PreA on consultation duration, physician-patient communication, and care coordination.

These methodologies demonstrate that PreA can improve patient-centered care and enhance the efficiency of healthcare systems in high-demand medical environments.


<br/>
# Results



이 연구에서는 PreA라는 LLM(대형 언어 모델) 챗봇을 사용하여 2,069명의 환자를 대상으로 한 무작위 대조 시험(RCT)의 결과를 분석했습니다. 연구의 주요 목표는 PreA가 전문의와의 상담에서 환자 경험과 의료 효율성을 어떻게 향상시키는지를 평가하는 것이었습니다.

#### 1. 경쟁 모델
PreA는 두 가지 모델과 비교되었습니다. 첫 번째는 PreA 모델을 지역 대화 데이터로 미세 조정한 데이터 조정 모델이었고, 두 번째는 PreA의 공동 설계 모델이었습니다. 연구 결과, 공동 설계된 PreA 모델이 데이터 조정 모델보다 모든 임상 평가 영역에서 더 높은 품질 점수를 기록했습니다. 예를 들어, 역사 수집, 진단 및 검사 주문의 경우, 공동 설계된 모델은 각각 4.56, 4.67, 4.23의 점수를 기록한 반면, 데이터 조정 모델은 각각 3.86, 2.47, 2.21의 점수를 기록했습니다.

#### 2. 테스트 데이터
연구에 사용된 테스트 데이터는 515개의 환자-의사 시나리오로 구성된 오디오 코퍼스에서 수집되었습니다. 이 데이터는 중국의 11개 성에서 수집되었으며, 저소득 및 고소득 지역으로 분류되었습니다. 이 데이터는 PreA의 성능을 평가하기 위해 사용되었습니다.

#### 3. 메트릭
주요 메트릭은 다음과 같습니다:
- **상담 지속 시간**: PreA-only 그룹은 No-PreA 그룹에 비해 상담 시간이 28.7% 단축되었습니다 (3.14분 대 4.41분, P < 0.001).
- **의료 조정**: PreA-only 그룹의 의사들은 PreA가 생성한 추천 보고서의 유용성을 높게 평가했습니다 (3.69 대 1.73, P < 0.001).
- **환자 경험**: 환자들은 PreA-only 그룹에서 의사와의 소통 용이성, 의사의 주의 깊음, 대인 관계 존중, 환자 만족도 및 향후 수용 가능성에서 유의미한 향상을 보고했습니다.

#### 4. 비교
PreA-only 그룹과 PreA-human 그룹 간의 결과는 유사했으며, 이는 PreA의 자율적 운영 능력을 확인하는 데 중요한 결과였습니다. 또한, PreA는 환자-의사 간의 소통을 개선하고, 의사들의 업무 부담을 줄이는 데 기여했습니다. 연구 결과는 PreA가 고부담 의료 시스템에서 환자 중심의 치료를 강화할 수 있는 잠재력을 가지고 있음을 보여줍니다.




This study analyzed the results of a randomized controlled trial (RCT) involving 2,069 patients using a large language model (LLM) chatbot called PreA. The primary goal of the study was to evaluate how PreA enhances patient experience and healthcare efficiency during consultations with specialists.

#### 1. Competing Models
PreA was compared with two models. The first was a data-tuned model, which fine-tuned the PreA model on local dialogue data, and the second was the co-designed version of PreA. The results showed that the co-designed PreA model outperformed the data-tuned model across all clinical evaluation domains. For instance, in history-taking, diagnosis, and test ordering, the co-designed model scored 4.56, 4.67, and 4.23, respectively, while the data-tuned model scored 3.86, 2.47, and 2.21.

#### 2. Test Data
The test data used in the study consisted of an audio corpus of 515 patient-physician scenarios collected from 11 provinces in China, categorized into low-income and high-income regions. This data was utilized to evaluate the performance of PreA.

#### 3. Metrics
The primary metrics included:
- **Consultation Duration**: The PreA-only group showed a 28.7% reduction in consultation time compared to the No-PreA group (3.14 minutes vs. 4.41 minutes, P < 0.001).
- **Care Coordination**: Physicians in the PreA-only group rated the usefulness of the PreA-generated referral reports significantly higher (3.69 vs. 1.73, P < 0.001).
- **Patient Experience**: Patients reported significant improvements in communication ease, physician attentiveness, interpersonal regard, patient satisfaction, and future acceptability in the PreA-only group.

#### 4. Comparisons
The outcomes between the PreA-only and PreA-human groups were similar, confirming the autonomous operation capability of PreA. Additionally, PreA contributed to improving patient-physician communication and reducing physician workload. The findings indicate that PreA has the potential to enhance patient-centered care in high-demand healthcare systems.


<br/>
# 예제



이 연구에서는 PreA라는 LLM(대형 언어 모델) 챗봇을 개발하여 1차 진료에서 전문 진료로의 전환을 원활하게 하는 데 초점을 맞추었습니다. PreA는 환자와의 대화를 통해 의료 정보를 수집하고, 이를 바탕으로 전문의에게 전달할 수 있는 추천 보고서를 생성하는 기능을 가지고 있습니다. 이 챗봇의 훈련 데이터와 테스트 데이터는 다음과 같은 방식으로 구성되었습니다.

#### 트레이닝 데이터
1. **입력(Input)**: 환자의 기본 정보(예: 나이, 성별), 주 증상, 병력, 가족력, 현재 복용 중인 약물 등.
2. **출력(Output)**: 
   - **진단 가능성**: 환자의 증상에 기반하여 가능한 진단 목록을 제시.
   - **추천 검사**: 진단을 확인하기 위해 필요한 검사 목록.
   - **의료 역사 요약**: 환자의 병력과 증상을 요약한 보고서.

예를 들어, 환자가 "최근에 기침이 나고 열이 나요"라고 입력하면, 챗봇은 다음과 같은 출력을 생성할 수 있습니다:
- **진단 가능성**: "감기, 독감, 폐렴"
- **추천 검사**: "흉부 X선, 혈액 검사"
- **의료 역사 요약**: "환자는 30세 남성으로, 최근 3일간 기침과 열이 발생하였으며, 과거 병력은 없음."

#### 테스트 데이터
1. **입력(Input)**: 실제 환자와의 대화에서 수집된 데이터로, 다양한 증상과 병력을 포함.
2. **출력(Output)**: 챗봇이 생성한 추천 보고서와 전문의가 작성한 임상 노트를 비교하여 일치 여부를 평가.

테스트 데이터의 예시로는, 환자가 "복통이 심하고 구토가 있어요"라고 입력했을 때, 챗봇이 생성한 보고서와 전문의의 노트를 비교하여 다음과 같은 평가를 진행합니다:
- **일치 여부**: 챗봇의 추천 진단과 전문의의 진단이 일치하는지 확인.
- **품질 평가**: 챗봇이 생성한 보고서의 완전성, 적절성, 임상 관련성을 평가.

이러한 방식으로 PreA는 환자와의 상호작용을 통해 수집된 데이터를 기반으로 전문의에게 유용한 정보를 제공하고, 진료의 효율성을 높이는 데 기여합니다.

---




In this study, a large language model (LLM) chatbot named PreA was developed to facilitate transitions from primary care to specialist care. PreA interacts with patients to gather medical information and generates referral reports that can be sent to specialists. The training and testing data for this chatbot were structured as follows:

#### Training Data
1. **Input**: Basic patient information (e.g., age, gender), chief complaints, medical history, family history, current medications, etc.
2. **Output**: 
   - **Diagnostic Possibilities**: A list of potential diagnoses based on the patient's symptoms.
   - **Recommended Tests**: A list of tests needed to confirm the diagnosis.
   - **Medical History Summary**: A summary report of the patient's history and symptoms.

For example, if a patient inputs "I've had a cough and fever recently," the chatbot might generate the following output:
- **Diagnostic Possibilities**: "Cold, Flu, Pneumonia"
- **Recommended Tests**: "Chest X-ray, Blood test"
- **Medical History Summary**: "The patient is a 30-year-old male with a cough and fever for the past 3 days, with no significant past medical history."

#### Testing Data
1. **Input**: Data collected from actual patient interactions, encompassing a variety of symptoms and medical histories.
2. **Output**: A comparison of the referral reports generated by the chatbot and the clinical notes written by specialists to assess agreement.

An example of the testing data could involve a patient stating, "I have severe abdominal pain and vomiting." The chatbot's generated report would be compared to the specialist's notes to evaluate:
- **Agreement**: Whether the chatbot's suggested diagnoses align with the specialist's diagnosis.
- **Quality Assessment**: Evaluating the completeness, appropriateness, and clinical relevance of the chatbot-generated report.

Through this structured approach, PreA aims to provide valuable information to specialists based on interactions with patients, thereby enhancing the efficiency of medical consultations.

<br/>
# 요약
이 연구에서는 PreA라는 LLM 챗봇을 개발하여 2,069명의 환자를 대상으로 3개의 그룹(PreA-only, PreA-human, No-PreA)으로 무작위 배정하여 1차 진료에서 전문의로의 전환을 평가하였다. 결과적으로 PreA-only 그룹은 No-PreA 그룹에 비해 의사 상담 시간이 28.7% 단축되었고, 환자와 의사 간의 의사소통 용이성이 16% 향상되었다. 이 연구는 지역 이해관계자와의 공동 설계를 통해 LLM의 효과적인 임상 적용 가능성을 보여주었다.

---

In this study, the PreA LLM chatbot was developed and evaluated with 2,069 patients randomly assigned to three groups (PreA-only, PreA-human, No-PreA) to assess transitions from primary to specialist care. The results showed that the PreA-only group had a 28.7% reduction in physician consultation time compared to the No-PreA group, along with a 16% improvement in ease of communication between patients and physicians. This research demonstrates the potential for effective clinical application of LLMs through co-design with local stakeholders.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **CONSORT Flow Diagram**: 연구 참여자 등록, 무작위 배정 및 분석 흐름을 보여줍니다. 총 2,332명의 환자가 평가되었고, 최종적으로 2,069명이 분석에 포함되었습니다. 이는 연구의 투명성과 신뢰성을 높입니다.
   - **Consultation Duration and Experience**: 환자 상담 시간과 경험을 비교한 박스 플롯은 PreA-only 그룹이 No-PreA 그룹에 비해 상담 시간이 유의미하게 짧았음을 보여줍니다. 이는 PreA의 효율성을 강조합니다.
   - **Patient-Centeredness and Care Coordination Metrics**: 레이더 플롯은 환자 보고 지표(의사 주의, 만족도 등)와 의사 평가 지표(케어 조정)를 비교합니다. PreA-only 그룹이 No-PreA 그룹에 비해 모든 지표에서 유의미한 개선을 보였습니다.

2. **테이블**
   - **Baseline Covariates**: 세 그룹 간의 인구 통계학적 특성을 비교한 테이블로, 각 그룹 간의 균형을 보여줍니다. 이는 연구의 신뢰성을 높이는 데 기여합니다.
   - **Consultation Workflow, Physician Ratings, and Patient Experience**: 이 테이블은 각 그룹의 상담 흐름, 의사 평가 및 환자 경험을 요약합니다. PreA 사용이 환자 경험을 개선했음을 나타냅니다.

3. **어펜딕스**
   - **Model Development and Evaluation**: PreA의 개발 과정과 평가 방법을 상세히 설명합니다. 이는 연구의 방법론적 rigor를 강조합니다.
   - **Statistical Analysis**: 통계 분석 방법을 설명하여 결과의 신뢰성을 높입니다. 다양한 통계적 방법이 사용되어 결과의 유의성을 검증합니다.

### Insights

- **효율성 향상**: PreA-only 그룹은 No-PreA 그룹에 비해 상담 시간이 28.7% 단축되었습니다. 이는 PreA가 의사와 환자 간의 상담을 더 효율적으로 만들어 주는 도구임을 나타냅니다.
- **환자 경험 개선**: 환자들은 PreA를 사용한 후 의사와의 소통이 더 원활하다고 보고했으며, 이는 환자 중심의 의료 제공을 강화하는 데 기여합니다.
- **의사 평가**: 의사들은 PreA가 제공하는 추천 보고서의 질이 기존의 보고서보다 높다고 평가했습니다. 이는 PreA가 의사들의 임상 결정을 지원하는 데 효과적임을 시사합니다.

---




1. **Diagrams and Figures**
   - **CONSORT Flow Diagram**: This shows the registration, random assignment, and analysis flow of study participants. A total of 2,332 patients were assessed, with 2,069 included in the final analysis. This enhances the transparency and reliability of the study.
   - **Consultation Duration and Experience**: The box plots comparing patient consultation times and experiences indicate that the PreA-only group had significantly shorter consultation times compared to the No-PreA group, emphasizing the efficiency of PreA.
   - **Patient-Centeredness and Care Coordination Metrics**: The radar plots compare patient-reported metrics (physician attentiveness, satisfaction, etc.) and physician-rated metrics (care coordination). The PreA-only group showed significant improvements across all metrics compared to the No-PreA group.

2. **Tables**
   - **Baseline Covariates**: This table compares demographic characteristics across the three groups, demonstrating balance among the groups, which contributes to the study's reliability.
   - **Consultation Workflow, Physician Ratings, and Patient Experience**: This table summarizes the consultation flow, physician ratings, and patient experiences for each group, indicating that the use of PreA improved patient experiences.

3. **Appendices**
   - **Model Development and Evaluation**: This section details the development process and evaluation methods of PreA, emphasizing the methodological rigor of the study.
   - **Statistical Analysis**: This explains the statistical methods used, enhancing the reliability of the results. Various statistical techniques were employed to validate the significance of the findings.

### Insights

- **Improved Efficiency**: The PreA-only group experienced a 28.7% reduction in consultation time compared to the No-PreA group, indicating that PreA serves as an effective tool for streamlining consultations between physicians and patients.
- **Enhanced Patient Experience**: Patients reported that communication with physicians was smoother after using PreA, contributing to a more patient-centered approach to healthcare delivery.
- **Physician Evaluation**: Physicians rated the quality of referral reports generated by PreA higher than traditional reports, suggesting that PreA effectively supports clinical decision-making.

<br/>
# refer format:
### BibTeX Citation

```bibtex
@article{Tao2025,
  author = {Xinge Tao and Shuya Zhou and Kai Ding and Sairan Li and Yanzeng Li and Boyou Wu and Qirui Huang and Wangyue Chen and Muzi Shen and En Meng and Xiaowang Chen and Hong Hu and Jinchao Zhang and Jie Zhou and Lei Zou and Libing Ma and Shasha Han},
  title = {An LLM chatbot to facilitate primary-to-specialist care transitions: a randomized controlled trial},
  journal = {Nature Medicine},
  year = {2025},
  volume = {31},
  number = {1},
  pages = {1-12},
  doi = {10.1038/s41591-025-04176-7},
  url = {https://doi.org/10.1038/s41591-025-04176-7}
}
```

### Chicago Style Citation

Tao, Xinge, Shuya Zhou, Kai Ding, Sairan Li, Yanzeng Li, Boyou Wu, Qirui Huang, Wangyue Chen, Muzi Shen, En Meng, Xiaowang Chen, Hong Hu, Jinchao Zhang, Jie Zhou, Lei Zou, Libing Ma, and Shasha Han. "An LLM Chatbot to Facilitate Primary-to-Specialist Care Transitions: A Randomized Controlled Trial." *Nature Medicine* 31, no. 1 (2025): 1-12. https://doi.org/10.1038/s41591-025-04176-7.
