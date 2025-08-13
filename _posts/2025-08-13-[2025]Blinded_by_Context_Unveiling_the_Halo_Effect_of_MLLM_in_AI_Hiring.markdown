---
layout: post
title:  "[2025]Blinded by Context: Unveiling the Halo Effect of MLLM in AI Hiring"
date:   2025-08-13 03:20:25 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 연구는 AI 기반 채용 평가에서 대규모 언어 모델(LLM)과 다중모달 대규모 언어 모델(MLLM)이 비직무 관련 정보에 의해 어떻게 영향을 받는지를 조사합니다.


짧은 요약(Abstract) :

이 연구는 대규모 언어 모델(LLMs)과 다중모달 대규모 언어 모델(MLLMs)을 활용한 AI 기반 채용 평가에서 발생하는 후광 효과를 조사합니다. 가상의 직무 지원서를 사용한 실험을 통해, 이러한 모델들이 직무와 관련 없는 정보(예: 과외 활동, 소셜 미디어 이미지)에 의해 어떻게 평가가 영향을 받는지를 분석했습니다. 다양한 역량 차원에서 리커트 척도 질문에 대한 모델의 응답을 분석한 결과, AI 모델은 특히 이미지 기반 평가에서 상당한 후광 효과를 보였으며, 텍스트 기반 평가에서는 편향에 대한 저항력이 더 강했습니다. 이러한 결과는 보조적인 다중모달 정보가 AI 채용 결정에 상당한 영향을 미칠 수 있음을 보여주며, AI 기반 채용 시스템에서의 잠재적 위험을 강조합니다.


This study investigates the halo effect in AI-driven hiring evaluations using Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs). Through experiments with hypothetical job applications, we examined how these models’ evaluations are influenced by non-job-related information, including extracurricular activities and social media images. By analyzing models’ responses to Likert-scale questions across different competency dimensions, we found that AI models exhibit significant halo effects, particularly in image-based evaluations, while text-based assessments showed more resistance to bias. The findings demonstrate that supplementary multimodal information can substantially influence AI hiring decisions, highlighting potential risks in AI-based recruitment systems.


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
논문 "Blinded by Context: Unveiling the Halo Effect of MLLM in AI Hiring"에서는 대규모 언어 모델(LLM)과 다중모달 대규모 언어 모델(MLLM)이 채용 평가에서 나타내는 후광 효과를 조사합니다. 이 연구는 AI 기반 채용 시스템에서 비직무 관련 정보가 어떻게 평가에 영향을 미치는지를 실험적으로 분석합니다. 연구의 주요 방법론은 다음과 같습니다.

1. **모델 선택 및 설정**: 연구에서는 다양한 LLM과 MLLM을 사용하여 채용 평가를 수행합니다. 사용된 모델에는 Llama-3.1-Instruct, Qwen2.5-Instruct, Falcon3-Instruct와 같은 오픈소스 모델과 GPT-4o와 같은 폐쇄형 모델이 포함됩니다. MLLM의 경우, InternVL2.5, Qwen2-VL-Instruct, LLaVA-OneVision과 같은 모델이 사용됩니다. 이러한 모델들은 다양한 입력 모달리티(텍스트, 이미지, 비디오)를 처리할 수 있도록 설계되었습니다.

2. **데이터셋 구성**: 연구는 가상의 채용 시나리오를 기반으로 한 데이터셋을 구축합니다. 이 데이터셋은 텍스트 기반의 이력서, 소셜 미디어 스타일의 이미지, 모의 인터뷰 비디오 클립을 포함합니다. 이러한 데이터는 비직무 관련 정보를 제공하여 모델이 후광 효과를 나타내는지를 평가하는 데 사용됩니다.

3. **평가 프로토콜**: 각 모델은 주어진 직무 설명과 이력서를 기반으로 지원자의 역량을 평가합니다. 평가 항목은 교육, 기술 및 역량, 경험 및 과거 성과, 개인적 특성과 문화적 적합성의 네 가지 주요 차원으로 구성됩니다. 각 항목은 리커트 척도를 사용하여 점수화되며, 모델의 평가 이유를 분석하기 위한 개방형 질문도 포함됩니다.

4. **통계적 분석**: 연구는 후광 효과를 측정하기 위해 통계적 분석을 수행합니다. 특히, 보조적인 다중모달 정보가 평가 점수에 미치는 영향을 분석하고, 이러한 정보가 비직무 관련 속성을 통해 간접적으로 평가에 영향을 미치는지를 확인합니다. 이를 위해 매개변수 분석을 사용하여 비직무 관련 정보가 평가 점수에 미치는 간접 효과를 측정합니다.

5. **결과 해석**: 연구 결과, 이미지 기반의 보조 정보가 텍스트 기반 정보보다 더 강한 후광 효과를 유발하는 것으로 나타났습니다. 또한, 특정 모델은 비직무 관련 정보에 더 민감하게 반응하여 평가에 편향을 나타내기도 했습니다. 이러한 결과는 AI 기반 채용 시스템에서 편향을 완화하기 위한 조치가 필요함을 시사합니다.



The paper "Blinded by Context: Unveiling the Halo Effect of MLLM in AI Hiring" investigates the halo effect exhibited by Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) in hiring evaluations. The study experimentally analyzes how non-job-related information influences AI-based hiring systems. The main methodology of the study is as follows:

1. **Model Selection and Configuration**: The study employs various LLMs and MLLMs to conduct hiring evaluations. The models used include open-source models like Llama-3.1-Instruct, Qwen2.5-Instruct, Falcon3-Instruct, and closed-source models like GPT-4o. For MLLMs, models such as InternVL2.5, Qwen2-VL-Instruct, and LLaVA-OneVision are used. These models are designed to handle various input modalities (text, image, video).

2. **Dataset Construction**: The study constructs a dataset based on hypothetical hiring scenarios. This dataset includes text-based resumes, social media-style images, and mock interview video clips. These data provide non-job-related information to evaluate whether models exhibit the halo effect.

3. **Evaluation Protocol**: Each model evaluates candidates based on given job descriptions and resumes. The evaluation items are categorized into four main dimensions: education, skills and competencies, experience and past performance, and personal characteristics and cultural fit. Each item is scored using a Likert scale, and open-ended questions are included to analyze the model's reasoning.

4. **Statistical Analysis**: The study conducts statistical analysis to measure the halo effect. Specifically, it analyzes the impact of supplementary multimodal information on evaluation scores and examines whether this information indirectly influences evaluations through non-job-related attributes. Mediation analysis is used to measure the indirect effects of non-job-related information on evaluation scores.

5. **Interpretation of Results**: The study finds that image-based supplementary information induces stronger halo effects than text-based information. Additionally, certain models are more sensitive to non-job-related information, exhibiting biases in evaluations. These findings highlight the need for measures to mitigate bias in AI-based hiring systems.


<br/>
# Results


#### 연구 결과 요약

이 연구는 다양한 멀티모달 대형 언어 모델(MLLM)이 채용 평가에서 직무와 무관한 정보에 어떻게 반응하는지를 조사하여, 이로 인해 발생하는 후광 효과를 분석했습니다. 연구는 텍스트, 이미지, 비디오와 같은 다양한 입력 유형을 통해 후광 효과가 어떻게 나타나는지를 다각도로 분석했습니다. 또한, 인구통계학적 변동과 특정 평가 항목에 미치는 영향을 조사하여 AI 기반 평가 시스템의 편향성과 취약성을 종합적으로 이해하고자 했습니다.

#### 경쟁 모델

연구에서는 여러 최신의 오픈 소스 및 폐쇄 소스 모델을 평가했습니다. 텍스트 모델로는 Llama-3.1-Instruct, Qwen2.5-Instruct, Falcon3-Instruct, GPT-4o, GPT-4o-mini가 포함되었습니다. 이미지 기반 평가를 위한 MLLM으로는 InternVL2.5, Qwen2-VL-Instruct, LLaVA-OneVision, GPT-4o, GPT-4o-mini가 사용되었습니다. 비디오 입력을 위한 모델로는 MiniCPM-o 2.6, MiniCPM-V 2.6, GPT-4o, GPT-4o-mini, gemini-1.5-flash, gemini-1.5-flash-8b, gemini-2.0-flash-exp가 포함되었습니다.

#### 테스트 데이터

각 모델은 다양한 직무 역할과 수준을 아우르는 가상의 채용 시나리오에서 평가되었습니다. 텍스트 기반 평가에서는 추가적인 텍스트 정보(예: 과외 활동 설명)가 포함되었고, 이미지 기반 평가에서는 소셜 미디어 이미지가 사용되었습니다. 비디오 기반 평가에서는 다양한 환경적 배경을 가진 짧은 인터뷰 클립이 사용되었습니다.

#### 메트릭

모델의 평가 결과는 10개의 리커트 척도 질문을 통해 측정되었으며, 이는 교육, 기술 및 역량, 경험 및 과거 성과, 개인적 특성과 문화적 적합성의 네 가지 차원으로 분류되었습니다. 또한, 모델의 평가가 직무와 무관한 정보에 얼마나 의존하는지를 측정하기 위해 '직무 무관성 점수'가 도입되었습니다.

#### 비교 및 분석

텍스트 기반 모델의 경우, 대부분의 모델이 텍스트로 인한 후광 효과에 대해 강건성을 보였으나, Llama-3.1-Instruct (8B)는 실내 취미 시나리오에서 역후광 효과를 보였습니다. 이미지 기반 모델에서는 후광 효과가 더 강하게 나타났으며, 특히 LLaVA-OneVision 시리즈가 전문적인 이미지 설정에 민감하게 반응했습니다. 비디오 기반 평가에서는 비공식적인 환경이 AI 판단에 특히 큰 영향을 미치는 것으로 나타났습니다.

#### 결론

이 연구는 AI 기반 평가 시스템에서 후광 효과가 어떻게 나타나는지를 다각도로 분석하여, 공정성과 신뢰성을 보장하기 위한 지속적인 검토와 개선의 필요성을 강조합니다.

---



#### Summary of Findings

This study investigated how various multimodal large language models (MLLMs) respond to job-irrelevant information in hiring evaluations, analyzing the resulting halo effects. The study examined how halo effects manifest across different input types, including text, images, and video, from multiple perspectives. Additionally, the study explored the influence of demographic variations and the impact on specific score categories, providing a comprehensive understanding of bias and susceptibility in AI-driven evaluation systems.

#### Competing Models

The study evaluated several state-of-the-art open-source and closed-source models. Text models included Llama-3.1-Instruct, Qwen2.5-Instruct, Falcon3-Instruct, GPT-4o, and GPT-4o-mini. For image-based evaluations, MLLMs such as InternVL2.5, Qwen2-VL-Instruct, LLaVA-OneVision, GPT-4o, and GPT-4o-mini were used. Video input models included MiniCPM-o 2.6, MiniCPM-V 2.6, GPT-4o, GPT-4o-mini, gemini-1.5-flash, gemini-1.5-flash-8b, and gemini-2.0-flash-exp.

#### Test Data

Each model was evaluated in a fictional hiring scenario covering various job roles and levels. Text-based evaluations included supplementary text information (e.g., descriptions of extracurricular activities), while image-based evaluations used social media images. Video-based evaluations used short interview clips with different environmental backgrounds.

#### Metrics

The evaluation results were measured using ten Likert-scale questions, categorized into four dimensions: Education, Skills and Competencies, Experience and Past Performance, and Personal Characteristics and Cultural Fit. Additionally, a "job irrelevance score" was introduced to measure how much the model's evaluation relied on non-job-related information.

#### Comparison and Analysis

For text-based models, most demonstrated robustness against text-induced halo effects, but Llama-3.1-Instruct (8B) showed reverse halo effects in indoor hobby scenarios. Image-based models exhibited stronger halo effects, with the LLaVA-OneVision series particularly sensitive to professional image settings. In video-based evaluations, informal environments were especially potent in biasing AI judgment.

#### Conclusion

This study highlights the need for ongoing scrutiny and refinement of AI evaluation systems to ensure fairness and reliability in high-stakes decision-making contexts.


<br/>
# 예제
논문 "Blinded by Context: Unveiling the Halo Effect of MLLM in AI Hiring"에서는 AI 기반 채용 평가에서 대규모 언어 모델(LLM)과 다중모달 대규모 언어 모델(MLLM)이 어떻게 비직무 관련 정보에 의해 평가가 영향을 받는지를 조사합니다. 연구는 가상의 직무 지원서를 사용하여 실험을 진행하였으며, 여기에는 텍스트 설명과 소셜 미디어 이미지가 포함되었습니다. 이 연구는 AI 모델이 비직무 관련 정보에 의해 평가가 어떻게 변하는지를 분석하기 위해 다양한 시나리오와 서브 시나리오를 설계했습니다.

### 예시 설명

#### 트레이닝 데이터
- **입력 데이터**: 가상의 직무 지원서, 텍스트 설명(예: 야외 취미, 실내 취미), 소셜 미디어 이미지(예: 전문 초상화, 작업 중, 캐주얼 환경, 야외 취미, 실내 취미)
- **출력 데이터**: AI 모델이 생성한 평가 점수(리커트 척도)와 평가 이유

#### 테스트 데이터
- **입력 데이터**: 새로운 가상의 직무 지원서와 추가적인 비직무 관련 정보(텍스트 설명 및 이미지)
- **출력 데이터**: AI 모델이 생성한 평가 점수와 평가 이유

#### 구체적인 테스크
1. **직무 지원서 평가**: AI 모델은 주어진 직무 지원서를 기반으로 지원자의 역량을 평가합니다.
2. **비직무 관련 정보의 영향 분석**: 텍스트 설명과 이미지가 추가된 경우, AI 모델의 평가가 어떻게 변하는지를 분석합니다.
3. **평가 점수의 변화 측정**: 리커트 척도를 사용하여 교육, 기술, 경험, 의사소통 등 다양한 평가 항목에 대한 점수 변화를 측정합니다.
4. **평가 이유 분석**: AI 모델이 제공한 평가 이유를 분석하여 비직무 관련 정보가 평가에 미치는 영향을 파악합니다.



The paper "Blinded by Context: Unveiling the Halo Effect of MLLM in AI Hiring" investigates how AI-driven hiring evaluations using Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) are influenced by non-job-related information. The study conducts experiments using hypothetical job applications, which include text descriptions and social media images. The research designs various scenarios and sub-scenarios to analyze how AI models' evaluations change due to non-job-related information.

### Example Description

#### Training Data
- **Input Data**: Hypothetical job applications, text descriptions (e.g., outdoor hobbies, indoor hobbies), social media images (e.g., professional portrait, working, casual setting, outdoor hobby, indoor hobby)
- **Output Data**: Evaluation scores (Likert scale) and reasoning generated by AI models

#### Test Data
- **Input Data**: New hypothetical job applications with additional non-job-related information (text descriptions and images)
- **Output Data**: Evaluation scores and reasoning generated by AI models

#### Specific Tasks
1. **Job Application Evaluation**: AI models evaluate the applicant's competency based on the given job application.
2. **Impact Analysis of Non-Job-Related Information**: Analyze how the addition of text descriptions and images affects AI models' evaluations.
3. **Measurement of Score Changes**: Use Likert scale to measure changes in scores across various evaluation dimensions such as education, skills, experience, and communication.
4. **Analysis of Evaluation Reasoning**: Analyze the reasoning provided by AI models to understand the impact of non-job-related information on evaluations.

<br/>
# 요약
이 연구는 AI 기반 채용 평가에서 대규모 언어 모델(LLM)과 다중모달 대규모 언어 모델(MLLM)이 비직무 관련 정보에 의해 어떻게 영향을 받는지를 조사합니다. 실험 결과, 이미지 기반 평가에서 특히 강한 후광 효과가 나타났으며, 이는 AI 채용 시스템에서 잠재적인 편향의 위험성을 강조합니다. 예를 들어, 지원자의 자신감 있는 외모나 전문적인 사진이 직무 능력 평가에 긍정적인 영향을 미치는 것으로 나타났습니다.

This study investigates how Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) in AI-driven hiring evaluations are influenced by non-job-related information. The experiments revealed significant halo effects, particularly in image-based evaluations, highlighting potential bias risks in AI recruitment systems. For instance, a candidate's confident appearance or professional-looking photo positively influenced competency assessments.

<br/>
# 기타


#### 1. 다이어그램 및 피규어
- **Figure 1: Overview of the Halo Effect in Candidate Evaluation**
  - 이 다이어그램은 후보자의 평가에서 발생하는 헤일로 효과를 시각적으로 설명합니다. 이 다이어그램은 이력서만으로 평가된 후보자와 멀티모달 정보를 추가하여 평가된 후보자의 점수 차이를 보여줍니다. 멀티모달 정보가 추가되면 후보자의 평가 점수가 상승하는 경향이 있으며, 이는 헤일로 효과의 존재를 시사합니다.

- **Figure 2: Halo Effect Induced by Supplementary Text Information**
  - 이 피규어는 다양한 모델과 시나리오에서 텍스트 정보가 유도하는 헤일로 효과를 보여줍니다. Llama-3.1-Instruct (8B) 모델은 특히 실내 취미 시나리오에서 역 헤일로 효과를 보였으며, Llama-3.1-Instruct (70B) 모델은 모든 시나리오에서 긍정적인 헤일로 효과를 보였습니다.

- **Figure 3: Halo Effect Induced by Supplementary Image**
  - 이미지 기반 정보가 유도하는 헤일로 효과를 보여줍니다. LLaVA-OneVision (7B) 모델은 대부분의 시나리오에서 일관된 긍정적 헤일로 효과를 보였으며, 특히 전문적인 이미지 설정에서 강한 영향을 받았습니다.

#### 2. 테이블
- **Table 1: Scenarios and Sub-Scenarios for Extra-Curricular Activity Descriptions and Social Media Images**
  - 이 테이블은 연구에서 사용된 다양한 시나리오와 하위 시나리오를 나열합니다. 각 시나리오는 후보자의 직무 관련 역량을 암시하지 않도록 설계되었습니다.

- **Table 3: Likert-scale Questions and Evaluation Dimensions**
  - 이 테이블은 평가에 사용된 10개의 리커트 척도 질문을 네 가지 차원으로 분류하여 보여줍니다. 각 질문은 교육, 기술 및 역량, 경험 및 과거 성과, 개인적 특성 및 문화 적합성에 대한 평가를 포함합니다.

#### 3. 어펜딕스
- **Appendix A: Mediation Analysis**
  - 매개 분석의 정의와 연구에서의 적용 방법을 설명합니다. 이 분석은 보조 정보가 평가에 미치는 영향을 중재하는지 여부를 평가하는 데 사용되었습니다.

- **Appendix B: Scenarios and Sub-Scenarios for Extra-Curricular Descriptions and Social Media Images**
  - 연구에서 사용된 시나리오와 하위 시나리오의 전체 목록을 제공합니다. 각 시나리오는 직무 관련 역량을 암시하지 않도록 설계되었습니다.

- **Appendix C: Job Application and Evaluation Setup**
  - 가상의 직무 설명과 생성된 이력서의 예시를 제공합니다. 이는 연구의 실험적 설정을 이해하는 데 도움이 됩니다.


### Results and Insights from Diagrams, Figures, Tables, and Appendices

#### 1. Diagrams and Figures
- **Figure 1: Overview of the Halo Effect in Candidate Evaluation**
  - This diagram visually explains the halo effect in candidate evaluation. It shows the score difference between candidates evaluated with only a resume and those evaluated with additional multimodal information. The addition of multimodal information tends to increase the candidate's evaluation score, indicating the presence of a halo effect.

- **Figure 2: Halo Effect Induced by Supplementary Text Information**
  - This figure shows the halo effect induced by text information across various models and scenarios. The Llama-3.1-Instruct (8B) model exhibited a reverse halo effect, especially in indoor hobby scenarios, while the Llama-3.1-Instruct (70B) model showed positive halo effects across all scenarios.

- **Figure 3: Halo Effect Induced by Supplementary Image**
  - This figure illustrates the halo effect induced by image-based information. The LLaVA-OneVision (7B) model showed consistent positive halo effects across most scenarios, with a strong influence from professional image settings.

#### 2. Tables
- **Table 1: Scenarios and Sub-Scenarios for Extra-Curricular Activity Descriptions and Social Media Images**
  - This table lists the various scenarios and sub-scenarios used in the study. Each scenario was designed to avoid implying job-related competencies.

- **Table 3: Likert-scale Questions and Evaluation Dimensions**
  - This table categorizes the 10 Likert-scale questions used for evaluation into four dimensions. Each question assesses aspects such as education, skills and competencies, experience and past performance, and personal characteristics and cultural fit.

#### 3. Appendices
- **Appendix A: Mediation Analysis**
  - This appendix explains the definition and application of mediation analysis in the study. The analysis was used to assess whether supplementary information mediates the impact on evaluations.

- **Appendix B: Scenarios and Sub-Scenarios for Extra-Curricular Descriptions and Social Media Images**
  - This appendix provides a full list of scenarios and sub-scenarios used in the study. Each scenario was designed to avoid implying job-related competencies.

- **Appendix C: Job Application and Evaluation Setup**
  - This appendix provides examples of fictional job descriptions and generated resumes. It helps in understanding the experimental setup of the study.

<br/>
# refer format:

**BibTeX:**
```bibtex
@inproceedings{kim2025blinded,
  title={Blinded by Context: Unveiling the Halo Effect of MLLM in AI Hiring},
  author={Kim, Kyusik and Ryu, Jeongwoo and Jeon, Hyeonseok and Suh, Bongwon},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2025},
  pages={26067--26113},
  year={2025},
  organization={Association for Computational Linguistics}
}
```


Kim, Kyusik, Jeongwoo Ryu, Hyeonseok Jeon, and Bongwon Suh. 2025. "Blinded by Context: Unveiling the Halo Effect of MLLM in AI Hiring." In *Findings of the Association for Computational Linguistics: ACL 2025*, 26067-26113. Association for Computational Linguistics.
