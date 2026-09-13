---
layout: post
title:  "[2026]AgentClinic: A Multimodal Benchmark for Tool-Using Clinical AI Agents"
date:   2026-09-13 21:41:06 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

AgentClinic에서는 AI가 의사 역할을 맡아 대화기반 증상파악, 검사요청, 텍스트/이미지 분석, 최종진단결정, 환자의 평가까지 포함하는 벤치마크   
(실제 진료와 유사하게 환자 에이전트와 의사 에이전트가 어케 행동하나 체크.. 의사 에이전트만 평가한다지만..)
(세 Patient agent- 증상과 병력을 알고 있지만 정답 진단은 모름. 의사의 질문에 답함, Measurement agent- 혈압, 심전도, 혈액검사, X-ray, MRI 등의 검사 결과를 제공함, Moderator agent- 의사의 최종 진단을 정답 진단과 비교해 맞고 틀림을 판정함. 와 상호작용... 그냥 LLM 백본 평가 아닌가.. 에이전트처럼 구성한 역할극..?)  
agentic clinical **simulation** benchmark 이걸ㄹ로 보는게 명확할듯...    
  


짧은 요약(Abstract) :


이 논문은 **AgentClinic**이라는 새로운 임상 AI 평가 벤치마크를 소개합니다. 기존의 의료 AI 평가는 주로 정답이 포함된 사례를 보고 답을 고르는 **정적인 객관식 문제**에 의존했지만, 실제 진료는 환자에게 질문하고, 필요한 검사를 선택하며, 불완전한 정보를 바탕으로 순차적으로 판단해야 합니다.

AgentClinic에서는 AI가 의사 역할을 맡아 다음과 같은 과정을 수행합니다.

- 환자 에이전트와 대화하며 증상과 병력을 파악
- 혈압, 심전도, 영상검사 등 필요한 검사 요청
- 텍스트와 의료 이미지를 함께 분석
- 제한된 질문 횟수 안에서 최종 진단 결정
- 환자의 신뢰도, 치료를 따를 의향, 재방문 의향까지 평가

이 벤치마크는 **9개 의료 전문 분야와 7개 언어**를 지원하며, USMLE 문제, 실제 전자의무기록인 MIMIC-IV, NEJM 임상 증례 등을 활용했습니다.

주요 결과는 다음과 같습니다.

1. 정적인 MedQA 문제를 잘 푸는 모델이라도, 실제 대화와 정보 수집이 필요한 AgentClinic에서는 성능이 크게 떨어졌습니다. 일부 모델은 진단 정확도가 기존 점수의 10분의 1 이하로 낮아지기도 했습니다.
2. 전반적으로 **Claude 3.5 계열 모델**이 대부분의 조건에서 가장 높은 성능을 보였습니다.
3. 도구 사용 능력은 모델마다 크게 달랐습니다. 특히 메모를 저장하고 다음 환자 사례에 활용하는 **Notebook 도구**를 사용했을 때, Llama-3는 상대적으로 최대 92%의 성능 향상을 보였습니다. 반면 일부 모델은 도구를 사용하면 오히려 성능이 낮아졌습니다.
4. 의사나 환자에게 인지적·사회적 편견을 부여하면 진단 정확도뿐 아니라 환자의 의사에 대한 신뢰, 치료 순응도, 재상담 의향도 낮아질 수 있었습니다.
5. 따라서 의료 AI는 단순히 의학 문제의 정답률만으로 평가해서는 안 되며, **대화 능력, 정보 수집, 도구 활용, 편견에 대한 강건성, 환자 중심성**까지 함께 평가해야 한다는 점을 보여줍니다.

---



This paper introduces **AgentClinic**, a new benchmark for evaluating clinical AI agents in more realistic medical settings. Most existing medical AI evaluations rely on static multiple-choice questions, where all relevant information is already provided. In real clinical practice, however, doctors must ask questions, decide which tests to order, interpret incomplete information, and make decisions sequentially.

In AgentClinic, an AI model acts as a doctor and must:

- Interview a simulated patient to obtain symptoms and medical history  
- Request medical measurements and imaging studies  
- Interpret both textual and visual information  
- Reach a diagnosis within a limited number of interactions  
- Consider patient-centered outcomes such as trust, treatment compliance, and willingness to return  

The benchmark covers **nine medical specialties and seven languages** and uses cases derived from USMLE questions, de-identified electronic health records from MIMIC-IV, and NEJM case challenges.

The main findings are:

1. Strong performance on static MedQA questions did not reliably translate into strong performance in the interactive AgentClinic setting. Diagnostic accuracy could fall to less than one-tenth of the original performance.
2. **Claude 3.5-based agents** generally achieved the best results across most evaluation settings.
3. Models differed substantially in their ability to use external tools. For example, Llama-3 showed up to a **92% relative improvement** when using a persistent notebook that stored and reused information across cases, while some other models performed worse with tools.
4. Cognitive and social biases introduced into doctor or patient agents could reduce diagnostic accuracy and negatively affect patient trust, treatment compliance, and willingness to seek follow-up care.
5. Overall, the study argues that clinical AI should be evaluated not only by medical question-answering accuracy, but also by its ability to conduct dialogue, gather information, use tools, resist bias, and provide patient-centered care.


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



### 1) 전체 평가 프레임워크: AgentClinic

AgentClinic은 정적인 객관식 문제를 푸는 방식이 아니라, **의사 AI가 환자와 대화하고 필요한 검사를 요청한 뒤 진단을 내리는 시뮬레이션 환경**이다.

시뮬레이션에는 네 종류의 언어 에이전트가 사용된다.

- **환자 에이전트**: 증상과 병력을 알고 있지만 정답 진단은 모름. 의사의 질문에 환자처럼 응답함.
- **의사 에이전트**: 평가 대상 모델. 환자에게 질문하고 검사 결과를 요청하여 진단함.
- **측정 에이전트(Measurement agent)**: 혈압, 심전도, 혈액검사, X-ray, MRI 등 요청된 검사 결과를 제공함.
- **중재자 에이전트(Moderator)**: 의사의 최종 진단을 정답 진단과 비교하여 정확도를 판정함.

의사 에이전트는 대부분의 실험에서 최대 **20회의 상호작용**을 사용할 수 있으며, 각 단계에서 환자에게 질문하거나 검사를 요청할 수 있다. 마지막에는 자유 형식으로 진단을 제시하고, 중재자 에이전트가 정답과 일치하는지를 판단한다.

---

### 2) 데이터와 임상 시나리오 구성

시나리오는 다음 자료에서 구성했다.

- **MedQA/USMLE 문제**
- **MIMIC-IV 비식별화 전자의무기록**
- **NEJM Case Challenges**
- 전문 분야 실험에는 **MedMCQA의 증례 기반 문제**

각 사례는 증상, 병력, 검사 결과, 정답 진단 등의 정보를 포함하는 구조화된 JSON 형식으로 변환했다.

초기 사례 정보의 구조화에는 **GPT-4**를 사용했으며, 이후 연구자가 각 사례를 수동으로 검토하고 수정했다. 정보는 에이전트별로 분리했다.

- 환자: 증상과 병력
- 측정 에이전트: 신체검사 및 검사 결과
- 의사: 환자를 평가하라는 목표와 제한된 초기 정보
- 중재자: 정답 진단

또한 다음과 같은 범위를 포함했다.

- 9개 의료 전문 분야
- 영어, 중국어, 프랑스어, 스페인어, 힌디어, 페르시아어, 한국어 등 7개 언어
- 텍스트뿐 아니라 X-ray, MRI, 병리 이미지 등 시각 정보가 포함된 다중모달 사례

다국어 사례는 GPT-4로 번역한 뒤 원어민이 검수했다.

---

### 3) 평가한 모델

AgentClinic에서는 다음과 같은 모델들을 의사 에이전트로 평가했다.

- Claude 3.5 Sonnet
- GPT-4
- GPT-4o
- GPT-3.5
- Mixtral-8×7B
- Llama 3 70B-Instruct
- Llama 2 70B-Chat
- OpenBioLLM-70B
- MedLLaMA3-8B
- PMC-LLaMA-7B
- Meditron-70B

#### 모델 구조와 학습 데이터의 주요 차이

- **GPT-4, GPT-4o, GPT-3.5, Claude 3.5**
  - 구체적인 아키텍처와 학습 데이터는 공개되지 않음.
  - GPT-4와 GPT-4o는 텍스트와 이미지 입력을 처리할 수 있는 다중모달 모델로 사용됨.

- **Mixtral-8×7B**
  - **Sparse Mixture-of-Experts, MoE** 구조.
  - 여러 전문가 네트워크 중 각 토큰마다 일부 전문가만 선택하여 계산함.
  - 총 약 47B 파라미터 중 토큰 처리 시 약 13B가 활성화되는 방식으로 설명됨.

- **Llama 2 및 Llama 3**
  - Meta의 공개 모델.
  - 대규모 공개 데이터 약 2조 토큰을 사용해 사전학습됨.
  - Llama 3 70B-Instruct는 지시문 수행 능력이 강화된 모델임.

- **PMC-LLaMA**
  - 의생명 논문 약 480만 편과 의학 교과서 3만 권을 활용해 의학 지식을 주입함.
  - 이후 의학 질의응답, 추론 설명, 대화 자료 약 2억 200만 토큰으로 instruction fine-tuning을 수행함.

- **OpenBioLLM-70B**
  - Llama 3 기반의 공개 생의학 모델.
  - 고품질 생의학 데이터로 추가 미세조정되었으며, DPO(Direct Preference Optimization)를 사용함.

- **MedLLaMA3**
  - Llama 3 기반의 의료 특화 모델.
  - 의료 관련 데이터로 추가 학습됨.

- **Meditron-70B**
  - Llama 2-70B를 기반으로 의료 코퍼스에서 계속 사전학습한 모델.
  - PubMed 논문, 초록, 의료 지침 등이 사용됨.

즉, 연구는 일반 목적의 상용 LLM, 공개 범용 모델, 의료 특화 모델을 동일한 상호작용 환경에서 비교했다.

---

### 4) 에이전트 도구와 추론 기법

연구에서는 단순히 모델의 내재적 지식만 평가하지 않고, 외부 도구를 사용하는 능력도 평가했다. 사용한 도구는 여섯 가지다.

1. **Zero-shot Chain-of-Thought**
   - 예시 없이 단계적으로 추론하도록 유도함.

2. **One-shot Chain-of-Thought**
   - 성공 사례 하나를 예시로 제공하여 유사한 문제를 풀도록 함.
   - 현재 문제와 의미적으로 유사한 과거 사례를 선택함.

3. **Reflection CoT**
   - 자신의 추론 과정을 다시 검토하고 오류나 개선점을 반영하도록 함.

4. **Adaptive RAG—Book**
   - 의사가 필요하다고 판단할 때 의학 교과서 데이터베이스에서 정보를 검색함.

5. **Adaptive RAG—Web**
   - 의사가 필요할 때 PubMed, StatPearls, Wikipedia 등의 자료에서 검색함.
   - 일반적인 RAG처럼 매번 자동 검색하는 것이 아니라, 모델이 검색 시점과 질의를 직접 결정함.

6. **Notebook**
   - 환자 사례가 바뀌어도 유지되는 메모리 도구.
   - 성공한 사례를 검토하여 진단 전략이나 유용한 원칙을 메모로 저장하고, 이후 유사 사례에서 검색해 활용함.
   - 모델의 가중치를 업데이트하는 학습은 아니며, 외부 메모리를 이용한 **in-context learning**에 해당함.

Notebook의 경험학습 과정은 다음과 같다.

1. 진단 결과와 실제 정답 비교  
2. 정답을 맞힌 경우 성공적인 추론을 요약  
3. 요약과 원래 사례를 벡터 데이터베이스에 저장  
4. 새 사례에서 유사한 과거 경험을 검색  
5. 검색 결과를 현재 프롬프트에 포함

---

### 5) 편향과 환자 중심 평가

의사와 환자 에이전트의 시스템 프롬프트에 총 **23가지 인지적·암묵적 편향**을 주입했다.

예시는 다음과 같다.

- 최신성 편향(recency bias)
- 고정관념 또는 초기 판단에 매달리는 앵커링 편향
- 인종, 성별, 교육 수준, 사회경제적 지위
- 종교, 문화, 성적 지향 등에 대한 암묵적 편향

평가는 진단 정확도뿐 아니라 환자 에이전트가 다음 항목을 1~10점으로 평가하도록 했다.

- 의사의 판단에 대한 **신뢰도**
- 치료를 따를 의향인 **순응도**
- 같은 의사에게 다시 진료받을 의향인 **재상담 의향**

이러한 지표를 통해 모델이 정답을 맞혔는지뿐 아니라, 대화 과정에서 환자의 신뢰와 치료 참여에 어떤 영향을 주는지도 분석했다.

---

### 6) 주요 평가 지표와 실험 조건

주요 지표는 다음과 같다.

- 진단 정확도
- 환자 정보 수집률(coverage)
- 환자 신뢰도, 순응도, 재상담 의향
- 대화의 현실성
- 측정 결과의 현실성
- 의사 에이전트의 공감 능력

또한 다음 조건을 변화시켜 실험했다.

- 상호작용 횟수: 10, 15, 20, 25, 30회
- 환자 에이전트의 기반 모델
- 언어
- 의료 전문 분야
- 이미지가 처음부터 제공되는지, 요청해야 제공되는지
- 도구 사용 여부
- 의사 또는 환자 에이전트의 편향 유무

핵심적으로 AgentClinic은 모델의 지식량만이 아니라, **불완전한 정보에서 질문을 선택하고, 검사를 요청하고, 도구를 활용하며, 여러 단계의 상호작용을 통해 진단하는 능력**을 평가하도록 설계되었다.

---




### 1) Overall framework: AgentClinic

AgentClinic evaluates clinical AI agents in an interactive simulation rather than through static multiple-choice questions. The doctor model must interview a patient, request diagnostic tests, interpret the returned information, and produce a final diagnosis.

The benchmark uses four language agents:

- **Patient agent**: Knows the patient’s symptoms and history but not the ground-truth diagnosis.
- **Doctor agent**: The model being evaluated. It asks questions, requests tests, and makes the diagnosis.
- **Measurement agent**: Returns simulated results for tests such as blood pressure, ECG, laboratory tests, X-ray, and MRI.
- **Moderator agent**: Compares the doctor’s final diagnosis with the ground-truth diagnosis and determines correctness.

In most experiments, the doctor agent is allowed up to **20 interaction turns**. Each turn may involve a patient question, a test request, or another tool interaction.

---

### 2) Clinical data and scenario construction

Cases were derived from:

- MedQA/USMLE questions
- De-identified MIMIC-IV electronic health records
- NEJM Case Challenges
- MedMCQA case reports for specialty-specific evaluations

The cases were converted into structured JSON scenarios containing symptoms, medical history, examination findings, test results, and the ground-truth diagnosis.

GPT-4 was initially used to populate the structured case templates, after which the scenarios were manually reviewed and validated. Information was separated by agent role:

- The patient agent received symptoms and history.
- The measurement agent received examination and test-result information.
- The doctor agent received a limited initial objective.
- The moderator agent received the correct diagnosis.

The benchmark included nine medical specialties, seven languages, and multimodal cases involving medical images. Multilingual cases were translated with GPT-4 and manually checked by native speakers.

---

### 3) Evaluated models

The evaluated doctor models included:

- Claude 3.5 Sonnet
- GPT-4
- GPT-4o
- GPT-3.5
- Mixtral-8×7B
- Llama 3 70B-Instruct
- Llama 2 70B-Chat
- OpenBioLLM-70B
- MedLLaMA3-8B
- PMC-LLaMA-7B
- Meditron-70B

#### Model architectures and training data

- **GPT-4, GPT-4o, GPT-3.5, and Claude 3.5**
  - Their detailed architectures and training datasets are not publicly disclosed.
  - GPT-4 and GPT-4o were used as multimodal models capable of processing text and images.

- **Mixtral-8×7B**
  - Uses a sparse Mixture-of-Experts architecture.
  - Only a subset of expert networks is activated for each token.
  - The model has approximately 47B total parameters, with around 13B active per token according to the paper.

- **Llama 2 and Llama 3**
  - Open-access Meta models.
  - Pretrained on approximately two trillion tokens from publicly available data.
  - Llama 3 70B-Instruct is instruction-tuned for following user and task instructions.

- **PMC-LLaMA**
  - Biomedical pretraining involved approximately 4.8 million biomedical papers and 30,000 medical textbooks.
  - It was subsequently fine-tuned on about 202 million tokens containing medical QA, reasoning rationales, and conversational data.

- **OpenBioLLM-70B**
  - A biomedical model based on the Llama 3 architecture.
  - Further trained on biomedical data and optimized using Direct Preference Optimization.

- **MedLLaMA3**
  - A medical model based on Llama 3 and further trained for medical applications.

- **Meditron-70B**
  - Based on Llama 2-70B and continually pretrained on medical resources such as PubMed articles, abstracts, and clinical guidelines.

Thus, the study compared proprietary general-purpose models, open-source general models, and medically specialized models within the same interactive clinical environment.

---

### 4) Agent tools and reasoning techniques

The study evaluated six tools or reasoning configurations:

1. **Zero-shot Chain-of-Thought**
   - Encourages step-by-step reasoning without providing an example.

2. **One-shot Chain-of-Thought**
   - Provides one solved example to guide reasoning.
   - The example is selected based on semantic similarity to the current case.

3. **Reflection CoT**
   - Requires the model to review and critique its own reasoning.

4. **Adaptive RAG—Book**
   - Allows the doctor agent to retrieve information from a medical textbook database when needed.

5. **Adaptive RAG—Web**
   - Allows retrieval from sources such as PubMed, StatPearls, and Wikipedia.
   - Unlike fixed RAG, the agent decides when to retrieve and what query to use.

6. **Notebook**
   - Provides persistent memory across patient cases.
   - The agent records useful diagnostic lessons from successful cases and retrieves them for similar future cases.
   - This does not update the model weights; it is an external-memory or in-context learning mechanism.

The notebook-based experiential learning pipeline consists of:

1. Comparing the agent’s diagnosis with the ground truth  
2. Reflecting on successful reasoning  
3. Storing the takeaway and original case in a vector database  
4. Retrieving similar past experiences for a new case  
5. Inserting the retrieved experience into the current prompt

---

### 5) Bias and patient-centered evaluation

The researchers introduced 23 cognitive and implicit biases into the system prompts of doctor and patient agents. Examples included recency bias, anchoring-like effects, and biases related to race, gender, education, socioeconomic status, religion, culture, and sexual orientation.

In addition to diagnostic accuracy, the patient agent rated:

- **Confidence** in the doctor’s assessment
- **Compliance**, or willingness to follow treatment
- **Consultation willingness**, or willingness to see the same doctor again

These metrics were intended to assess not only whether the model reached the correct diagnosis, but also how the interaction affected simulated patient trust and engagement.

---

### 6) Evaluation settings and metrics

The benchmark measured:

- Diagnostic accuracy
- Coverage of relevant patient information
- Patient confidence, compliance, and willingness to return
- Dialogue realism
- Realism of measurement results
- Doctor-agent empathy

The experiments varied:

- Number of interaction turns
- Patient-agent model
- Language
- Medical specialty
- Availability and timing of image input
- Tool usage
- Presence or absence of doctor and patient biases

Overall, AgentClinic evaluates whether a model can **actively gather incomplete information, select useful questions, request appropriate tests, use external tools, and make sequential diagnostic decisions**, rather than merely recall medical facts from a static question.


<br/>
# Results



## 1. 평가 목적과 기본 설정

AgentClinic은 의료 LLM을 단순한 객관식 문제풀이가 아니라, 실제 진료와 유사한 **다단계 대화·정보수집·검사 선택·진단 환경**에서 평가하는 벤치마크이다.

평가 과정에서 의사 에이전트는 다음을 수행한다.

- 환자 에이전트와 대화하며 증상과 병력 확인
- 혈압, 심전도, 혈액검사, X-ray, MRI 등 검사 요청
- 필요한 정보를 제한된 횟수 안에 수집
- 최종적으로 자유 형식의 진단 제시

기본 실험에서는 의사 에이전트에게 최대 **20회의 상호작용**이 허용되었다. 환자, 검사·측정, 평가를 담당하는 에이전트는 별도의 LLM으로 구성되었다.

---

## 2. 테스트 데이터와 평가 환경

### 주요 데이터셋

| 평가 세트 | 데이터 출처 및 특징 |
|---|---|
| **AgentClinic-MedQA** | USMLE 기반 의료 문항을 대화형 임상 상황으로 변환 |
| **AgentClinic-MIMIC-IV** | 비식별화된 실제 전자의무기록 기반 사례 |
| **AgentClinic-NEJM** | NEJM Case Challenge 120개 사례, 이미지와 개방형 진단 포함 |
| **AgentClinic-Spec** | MedMCQA의 전문 분야 사례 기반 |
| **AgentClinic-Lang** | 영어, 중국어, 프랑스어, 스페인어, 힌디어, 페르시아어, 한국어 |
| **Bias 평가** | 인지 편향과 인구사회학적 암묵적 편향 23종 적용 |

평가는 일반적인 진단 정확도뿐 아니라 다음과 같은 환자 중심 지표도 포함했다.

- 진단 정확도
- 환자 정보 수집 정도
- 의사·환자 대화의 현실성
- 공감성
- 환자의 의사에 대한 신뢰도
- 치료를 따를 의향(compliance)
- 같은 의사에게 다시 진료받을 의향(consultation)

---

## 3. 모델 경쟁 결과: AgentClinic-MedQA

### 주요 모델의 진단 정확도

GPT-4 환자·측정 에이전트를 사용한 AgentClinic-MedQA 결과는 다음과 같다.

| 모델 | 진단 정확도 |
|---|---:|
| **Claude 3.5 Sonnet** | **62.1%** |
| OpenBioLLM-70B | 58.3% |
| 인간 의사 3명 | 54.0% |
| GPT-4 | 51.6% |
| Mixtral-8×7B | 37.1% |
| GPT-3.5 | 36.6% |
| GPT-4o | 34.2% |
| MedLLaMA3-8B | 31.4% |
| Meditron-70B | 29.1% |
| PMC-LLaMA-7B | 23.6% |
| Llama 3 70B | 19.0% |
| Llama 2 70B-chat | 4.5% |

### 해석

- **Claude 3.5 Sonnet이 가장 높은 성능**을 보였다.
- OpenBioLLM-70B는 전문 의료 모델로서 높은 성능을 보였으며, 인간 의사 평균보다도 높았다.
- 인간 의사의 결과는 3명만을 대상으로 했고 변동성이 매우 커서 일반적인 인간 의사 수준으로 해석하기에는 제한이 있다.
- Llama 계열은 정적 의료 QA에서는 상대적으로 가능성을 보일 수 있지만, 대화형 정보수집 환경에서는 성능이 크게 떨어졌다.
- 특히 Llama 2 70B-chat의 정확도는 4.5%로 매우 낮았다.

---

## 4. MIMIC-IV 기반 실제 의료기록 사례 결과

AgentClinic-MIMIC-IV에서는 다음과 같은 결과가 나타났다.

| 모델 | 진단 정확도 |
|---|---:|
| **Claude 3.5 Sonnet** | **42.9%** |
| OpenBioLLM-70B | 38.1% |
| GPT-4 | 34.0% |
| PMC-LLaMA-7B | 34.3% |
| GPT-3.5 | 27.5% |
| Mixtral-8×7B | 29.5% |
| Meditron-70B | 25.5% |
| GPT-4o | 24.0% |
| Llama 2 70B-chat | 13.5% |
| Llama 3 70B | 8.5% |

MedQA 기반 사례보다 전체적으로 정확도가 낮아졌다. 이는 실제 전자의무기록 사례가 더 복잡하고, 정보가 불완전하거나 시간 순서에 따라 흩어져 있기 때문으로 해석된다.

---

## 5. 정적 MedQA와 대화형 AgentClinic의 비교

논문에서 가장 중요한 발견 중 하나는 **기존 MedQA 성적이 AgentClinic 성적을 잘 예측하지 못한다는 점**이다.

정적 MedQA에서는 모든 증상, 병력, 검사 결과가 한 번에 주어진다. 반면 AgentClinic에서는 모델이 직접 질문하고 검사도 선택해야 한다.

### 정보 수집 정도

- AgentClinic에서 의사 에이전트가 확보한 관련 정보의 평균 비율: **67%**
- 정확히 진단한 사례: **72%**
- 오진한 사례: **63%**

즉, AgentClinic의 성능은 의료 지식 자체뿐 아니라 다음 능력에 크게 좌우된다.

- 어떤 질문을 할지 선택하는 능력
- 중요한 정보를 놓치지 않는 능력
- 검사 결과를 적절히 요청하는 능력
- 수집한 정보를 순차적으로 통합하는 능력

따라서 단순한 객관식 의료 시험 성적만으로 실제 대화형 임상 수행능력을 판단하기 어렵다.

---

## 6. 도구 사용 능력 비교

AgentClinic은 다음 6가지 도구 또는 추론 방식을 비교했다.

1. Zero-shot Chain-of-Thought
2. One-shot Chain-of-Thought
3. Reflection CoT
4. Adaptive RAG—의학 서적
5. Adaptive RAG—인터넷·의학자료
6. Notebook—사례 간 지속되는 메모리

### 주요 결과

- Claude 3.5의 도구 사용 평균 정확도: **51.3%**
- Claude 3.5의 최고 성능: Notebook 사용 시 **56.1%**
- GPT-4는 Adaptive RAG(Web) 사용 시 최대 **43.9%**
- GPT-4는 Reflection CoT에서 **42.2%**로 특정 도구에서는 Claude 3.5보다 높았다.
- GPT-4o는 Notebook 사용 시 최대 **43.0%**
- Llama 3 70B는 도구 사용 시 평균적으로 약 **9.4%p 향상**되었고, Notebook과 Reflection CoT 사용 시 **41.1%**까지 상승했다.
- GPT-3.5는 대부분의 도구에서 오히려 성능이 감소했으며, Adaptive RAG(Book)에서는 약 **27.1% 감소**했다.

### 핵심 해석

도구는 모든 모델에 동일하게 도움이 되지 않았다. 같은 RAG나 메모리 도구를 사용해도 모델에 따라 성능이 향상되거나 감소했다.

이는 임상 에이전트 평가에서 단순한 모델 크기나 사전학습 지식뿐 아니라 다음 능력도 중요하다는 뜻이다.

- 언제 도구를 사용할지 판단하는 능력
- 적절한 검색어를 만드는 능력
- 검색 결과를 임상 맥락에 적용하는 능력
- 과거 사례에서 유용한 전략을 재사용하는 능력

---

## 7. 편향이 진단과 환자 경험에 미치는 영향

연구진은 의사 또는 환자 에이전트에 23가지 편향을 주입했다.

### 진단 정확도

#### GPT-4

- 편향이 없는 기본 정확도: **52%**
- 환자 인지 편향 적용: 48%, 약 4%p 감소
- 의사 인지 편향 적용: 50.3%
- 암묵적 편향 적용 시에도 평균 감소폭은 약 1~2%p 수준

#### Mixtral-8×7B

- 편향이 없는 기본 정확도: **37%**
- 의사 인지 편향 적용: 약 **29~32%**
- 환자 인지 편향 적용: 약 **33~35%**
- 의사 암묵적 편향 적용: 약 **32.7%**

Mixtral은 GPT-4보다 편향에 더 취약했다. 예를 들어 최근에 본 질환에 지나치게 고정되거나, 환자의 인종적 특성 때문에 필요한 검사를 덜 요청하는 문제가 관찰되었다.

### 환자 중심 평가

진단 정확도 변화가 작더라도 환자 에이전트의 인식은 크게 악화될 수 있었다.

평가 지표는 다음 세 가지였다.

- 의사에 대한 신뢰·확신
- 치료를 따를 의향
- 같은 의사에게 다시 진료받을 의향

특히 교육 수준, 성별, 사회경제적 지위, 문화적 배경과 관련된 암묵적 편향은 세 지표 모두에 부정적인 영향을 주었다.

다만 이 지표들은 실제 사람이 아니라 LLM 환자 에이전트가 부여한 점수이므로, 실제 환자 경험의 직접적인 대체 지표로 해석해서는 안 된다.

---

## 8. 전문 분야별 성능

9개 의료 전문 분야에서 평가한 결과, Claude 3.5 Sonnet이 평균적으로 가장 우수했다.

### Claude 3.5의 주요 결과

- 전체 평균: **66.7%**
- 내과: **78.3%**
- 이비인후과: **76.7%**
- 산부인과: **74.3%**

### GPT-4의 주요 결과

- 산부인과: **68.5%**
- 안과: **65.2%**
- 응급의학: **32.3%**
- 노인의학: **40.0%**

전문 분야별 차이는 단순한 의학 지식 차이뿐 아니라, 대화로 병력을 얻고 불완전한 정보에서 진단해야 하는 특성 때문일 수 있다.

---

## 9. 다국어 평가

평가 언어는 영어, 중국어, 프랑스어, 스페인어, 힌디어, 페르시아어, 한국어였다.

### 평균 정확도

| 모델 | 평균 정확도 |
|---|---:|
| **Claude 3.5 Sonnet** | **48.4%** |
| GPT-4 | 20.9% |

모든 모델은 영어에서 가장 높은 성능을 보였다. GPT-4의 경우 언어별 정확도가 다음과 같이 크게 달랐다.

- 영어: 40.18%
- 중국어: 11.21%

GPT-4o는 한국어에서 3.73%까지 낮아졌고, GPT-3.5는 페르시아어에서 1.86%를 기록했다. 중국어는 대부분 모델에게 특히 어려운 언어였다.

Claude 3.5는 다른 모델보다 언어별 성능 편차가 작고, 비영어권에서도 상대적으로 안정적이었다.

---

## 10. 이미지·멀티모달 진단 결과

NEJM 사례 120개를 이용해 환자와 대화하면서 이미지도 이해해야 하는 상황을 평가했다.

### 이미지가 처음부터 제공된 경우

| 모델 | 정확도 |
|---|---:|
| **Claude 3.5 Sonnet** | **37.2%** |
| GPT-4 | 27.7% |
| GPT-4o | 21.4% |
| GPT-4o-mini | 8.0% |

### 이미지가 요청 후 제공된 경우

| 모델 | 정확도 |
|---|---:|
| **Claude 3.5 Sonnet** | **35.4%** |
| GPT-4 | 25.4% |
| GPT-4o | 19.1% |
| GPT-4o-mini | 6.1% |

이미지를 처음부터 제공하는 것이 요청 후 제공하는 것보다 모든 모델에서 약간 더 높은 성능을 보였다. 그러나 전반적인 정확도는 여전히 낮았으며, 이미지 이해와 대화형 추론을 동시에 수행하는 것이 어렵다는 점을 보여준다.

---

## 11. 상호작용 횟수와 시간 제한의 영향

기본 상호작용 횟수는 20회였다.

| 상호작용 횟수 | 정확도 |
|---:|---:|
| 10회 | 25% |
| 15회 | 38% |
| 20회 | 52% |
| 25회 | 48% |
| 30회 | 43% |

상호작용이 10회로 줄면 필요한 정보를 충분히 얻지 못해 성능이 크게 떨어졌다. 그러나 20회를 넘어 25~30회로 늘려도 성능은 오히려 감소했다.

가능한 이유는 다음과 같다.

- 입력 문맥이 지나치게 길어짐
- 불필요한 정보가 누적됨
- 모델이 앞선 정보를 제대로 유지·통합하지 못함
- 질문이 많아지면서 진단 과정이 복잡해짐

즉, 임상 대화에서는 질문을 많이 하는 것보다 **적절한 질문을 효율적으로 선택하는 능력**이 중요하다.

---

## 12. 대화의 현실성 및 공감성 평가

MD 자격을 가진 임상의 3명이 영어 대화 20개를 1~10점으로 평가했다.

| 평가 항목 | 평균 점수 |
|---|---:|
| 의사 역할의 현실성 | 6.2 |
| 환자 역할의 현실성 | 6.7 |
| 검사·측정 결과의 현실성 | 6.3 |
| 의사의 공감성 | 5.8 |

주요 문제점은 다음과 같았다.

- 의사 에이전트가 대화를 자연스럽게 시작하지 못함
- 특정 진단에 지나치게 고정됨
- 기본적인 질문이나 검사를 놓침
- 환자 에이전트가 불필요하게 장황하거나 질문을 반복함
- 측정 에이전트가 일부 검사 결과를 누락함
- 공감적 표현보다 증상 확인에만 바로 들어감

단, 임상 평가자는 3명뿐이고 전문 분야도 통제되지 않았으므로 일반화에는 한계가 있다.

---

## 13. 논문의 핵심 결론

이 논문이 보여주는 가장 중요한 점은 다음과 같다.

1. **정적 의료 QA 성능과 실제 대화형 임상 수행능력은 다르다.**
2. 진단 정확도는 의료 지식뿐 아니라 정보수집, 검사 선택, 순차적 추론 능력에 좌우된다.
3. Claude 3.5 Sonnet이 대부분의 환경에서 가장 안정적이고 높은 성능을 보였다.
4. OpenBioLLM-70B는 의료 특화 모델 중 높은 성능을 보였다.
5. 도구 사용 효과는 모델마다 크게 달랐으며, Notebook과 RAG가 항상 유익한 것은 아니었다.
6. 편향은 진단 정확도뿐 아니라 환자의 신뢰, 치료 순응도, 재방문 의향에도 영향을 줄 수 있다.
7. 비영어권·멀티모달·전문 분야 환경에서는 모델 간 격차가 더 커졌다.
8. AgentClinic은 의료 AI를 단순 지식 시험이 아니라 실제 업무에 가까운 **상호작용형 에이전트 평가**로 확장했다.

---



## 1. Evaluation setting

AgentClinic evaluates medical LLMs in an interactive clinical environment rather than through static multiple-choice questions.

The doctor agent must:

- Ask the patient about symptoms and history
- Request laboratory tests, vital signs, ECGs, X-rays, or other measurements
- Gather incomplete information within a limited number of interaction turns
- Provide an open-ended final diagnosis

Most experiments allowed up to **20 interaction turns**.

---

## 2. Test data and evaluation tracks

| Track | Data source and characteristics |
|---|---|
| **AgentClinic-MedQA** | USMLE-style medical questions converted into interactive cases |
| **AgentClinic-MIMIC-IV** | De-identified real-world electronic health record cases |
| **AgentClinic-NEJM** | 120 NEJM Case Challenge cases with multimodal images |
| **AgentClinic-Spec** | Specialty cases derived from MedMCQA |
| **AgentClinic-Lang** | English, Chinese, French, Spanish, Hindi, Persian, and Korean |
| **Bias evaluation** | 23 cognitive and implicit bias conditions |

The benchmark measured not only diagnostic accuracy but also information coverage, dialogue realism, empathy, patient confidence, treatment compliance, and willingness to consult the same doctor again.

---

## 3. Model comparison on AgentClinic-MedQA

Using GPT-4 as the patient and measurement agent, the main diagnostic results were:

| Model | Diagnostic accuracy |
|---|---:|
| **Claude 3.5 Sonnet** | **62.1%** |
| OpenBioLLM-70B | 58.3% |
| Human physicians | 54.0% |
| GPT-4 | 51.6% |
| Mixtral-8×7B | 37.1% |
| GPT-3.5 | 36.6% |
| GPT-4o | 34.2% |
| MedLLaMA3-8B | 31.4% |
| Meditron-70B | 29.1% |
| PMC-LLaMA-7B | 23.6% |
| Llama 3 70B | 19.0% |
| Llama 2 70B-chat | 4.5% |

Claude 3.5 Sonnet achieved the best overall performance. OpenBioLLM-70B also performed strongly, while several Llama-based models struggled substantially in the interactive setting.

The human comparison was based on only three physicians and should therefore not be interpreted as a general estimate of human clinical performance.

---

## 4. Results on MIMIC-IV cases

| Model | Diagnostic accuracy |
|---|---:|
| **Claude 3.5 Sonnet** | **42.9%** |
| OpenBioLLM-70B | 38.1% |
| PMC-LLaMA-7B | 34.3% |
| GPT-4 | 34.0% |
| Mixtral-8×7B | 29.5% |
| GPT-3.5 | 27.5% |
| Meditron-70B | 25.5% |
| GPT-4o | 24.0% |
| Llama 2 70B-chat | 13.5% |
| Llama 3 70B | 8.5% |

Performance was generally lower than on MedQA-based cases, likely because real-world EHR cases are more complex and contain incomplete or dispersed information.

---

## 5. Static MedQA versus interactive AgentClinic

A central finding is that performance on static MedQA was only weakly predictive of performance on AgentClinic-MedQA.

In static MedQA, all relevant symptoms, history, and test results are presented upfront. In AgentClinic, the model must actively elicit the information.

### Information coverage

- Average information coverage: **67%**
- Correct-diagnosis cases: **72%**
- Incorrect-diagnosis cases: **63%**

This indicates that successful diagnosis depends heavily on asking the right questions, ordering appropriate tests, and integrating information over multiple turns.

---

## 6. Tool-use comparison

The study evaluated:

1. Zero-shot Chain-of-Thought
2. One-shot Chain-of-Thought
3. Reflection CoT
4. Adaptive RAG from medical books
5. Adaptive RAG from web-based medical sources
6. A persistent Notebook memory

### Main findings

- Claude 3.5 achieved an average tool-augmented accuracy of **51.3%**
- Its best result was **56.1% with the Notebook tool**
- GPT-4 reached **43.9%** with Adaptive RAG from the web
- GPT-4 achieved **42.2%** with Reflection CoT
- GPT-4o reached **43.0%** with the Notebook
- Llama 3 70B improved by about **9.4 percentage points on average** across tools and reached **41.1%** with Notebook and Reflection CoT
- GPT-3.5 generally declined with tools, including a **27.1% drop** with book-based Adaptive RAG

Tool use was therefore highly model-dependent. External tools did not automatically improve performance; the model also needed to decide when and how to use them effectively.

---

## 7. Bias evaluation

The researchers introduced 23 cognitive and implicit biases into either the doctor or patient agent.

### GPT-4

- Baseline accuracy: **52%**
- Patient cognitive bias: approximately **48%**
- Doctor cognitive bias: approximately **50.3%**
- Implicit bias caused only relatively small average accuracy reductions

### Mixtral-8×7B

- Baseline accuracy: **37%**
- Accuracy fell to approximately **29–33%** under several doctor-bias conditions
- The model was more vulnerable to both cognitive and implicit biases than GPT-4

Observed failure modes included fixation on recent diagnoses and reluctance to order necessary tests for patients affected by demographic bias.

### Patient-centered effects

Bias also affected:

- Confidence in the doctor
- Willingness to follow treatment
- Willingness to consult the doctor again

Implicit biases, particularly education, gender, socioeconomic, and cultural biases, had substantial effects on simulated patient perceptions. However, these ratings were generated by LLM-based patient agents and are not direct substitutes for real human patient experience.

---

## 8. Specialty performance

Claude 3.5 achieved the highest overall performance across nine specialties.

- Overall average: **66.7%**
- Internal medicine: **78.3%**
- Otolaryngology: **76.7%**
- Gynecology: **74.3%**

GPT-4 performed relatively well in:

- Gynecology: **68.5%**
- Ophthalmology: **65.2%**

But GPT-4 performed less well in:

- Emergency medicine: **32.3%**
- Geriatrics: **40.0%**

The specialty differences suggest that dialogue-based diagnosis presents a different difficulty profile from static multiple-choice medical QA.

---

## 9. Multilingual results

The seven evaluated languages were English, Chinese, French, Spanish, Hindi, Persian, and Korean.

| Model | Average accuracy |
|---|---:|
| **Claude 3.5 Sonnet** | **48.4%** |
| GPT-4 | 20.9% |

All models performed best in English. GPT-4, for example, ranged from **40.18% in English** to **11.21% in Chinese**. GPT-4o reached only **3.73% in Korean**, and GPT-3.5 reached **1.86% in Persian**.

Claude 3.5 showed both the highest average performance and the most consistent results across languages.

---

## 10. Multimodal diagnostic performance

Using 120 NEJM cases, the models had to combine patient dialogue with image understanding.

### Images provided initially

| Model | Accuracy |
|---|---:|
| **Claude 3.5 Sonnet** | **37.2%** |
| GPT-4 | 27.7% |
| GPT-4o | 21.4% |
| GPT-4o-mini | 8.0% |

### Images provided only after request

| Model | Accuracy |
|---|---:|
| **Claude 3.5 Sonnet** | **35.4%** |
| GPT-4 | 25.4% |
| GPT-4o | 19.1% |
| GPT-4o-mini | 6.1% |

Providing the image initially produced slightly better results for all models. Nevertheless, overall accuracy remained modest, showing that combining visual interpretation with sequential clinical reasoning is difficult.

---

## 11. Effect of interaction limits

| Number of interaction turns | Accuracy |
|---:|---:|
| 10 | 25% |
| 15 | 38% |
| 20 | 52% |
| 25 | 48% |
| 30 | 43% |

Reducing the number of turns substantially hurt performance because the agent could not gather enough information. Increasing the number beyond 20 also reduced accuracy, probably because of longer contexts, information overload, and difficulty maintaining a coherent diagnostic process.

The result suggests that efficient information elicitation is more important than simply asking more questions.

---

## 12. Human dialogue ratings

Three physicians rated 20 English-language dialogues on a 1–10 scale.

| Category | Mean score |
|---|---:|
| Doctor realism | 6.2 |
| Patient realism | 6.7 |
| Measurement realism | 6.3 |
| Empathy | 5.8 |

Common problems included poor opening statements, excessive focus on one diagnosis, missed questions or tests, repetitive patient responses, incomplete measurement results, and limited empathic language.

---

## 13. Overall conclusions

The main conclusions are:

1. Static medical QA performance does not reliably predict interactive clinical performance.
2. Diagnosis depends on information elicitation, test selection, sequential reasoning, and tool use.
3. Claude 3.5 Sonnet was the most consistently strong model across most settings.
4. OpenBioLLM-70B performed particularly well among biomedical models.
5. Tool benefits were highly model-specific; RAG and memory were not universally helpful.
6. Bias can affect not only diagnostic accuracy but also patient trust, treatment compliance, and willingness to return.
7. Performance gaps became larger in multilingual, multimodal, and specialty-specific settings.
8. AgentClinic expands medical LLM evaluation from static knowledge testing toward interactive, tool-using, sequential clinical decision-making.


<br/>
# 예제



### 1. AgentClinic은 무엇을 평가하나?

AgentClinic은 의학 객관식 문제를 한 번에 푸는 벤치마크가 아니라, **의사가 환자와 대화하고 필요한 검사를 선택한 뒤 진단을 내리는 과정**을 평가하는 시뮬레이션이다.

평가 대상은 **doctor agent**이며, 다음 세 에이전트와 상호작용한다.

- **Patient agent**: 증상과 병력을 알고 있지만 정답 진단은 모름. 의사의 질문에 답함.
- **Measurement agent**: 혈압, 심전도, 혈액검사, X-ray, MRI 등의 검사 결과를 제공함.
- **Moderator agent**: 의사의 최종 진단을 정답 진단과 비교해 맞고 틀림을 판정함.

---

### 2. 데이터 구성: 학습 데이터와 테스트 데이터의 의미

이 논문은 일반적인 지도학습처럼 모델을 별도의 **training set으로 학습시킨 뒤 test set에서 평가**한 연구는 아니다. 대부분의 모델은 이미 학습된 상태에서 AgentClinic의 새로운 시뮬레이션에 투입된다.

AgentClinic의 사례는 다음 자료에서 구축되었다.

- **MedQA/USMLE 문제**
- **MIMIC-IV 비식별화 전자의무기록**
- **NEJM Case Challenges**
- 전문 분야 사례에는 **MedMCQA**
- 다국어 사례는 MedQA 사례를 중국어, 힌디어, 한국어, 스페인어, 프랑스어, 페르시아어로 변환

논문은 각 사례를 다음과 같은 구조화된 정보로 만들었다.

| 에이전트 | 제공되는 정보 |
|---|---|
| Doctor agent | 환자의 최소한의 초기 정보와 진료 목표 |
| Patient agent | 증상, 병력, 생활습관 등 |
| Measurement agent | 신체검사·검사·영상 결과 |
| Moderator agent | 정답 진단 |

초기 사례 구조화에는 GPT-4가 사용되었고, 이후 사람이 사례를 검증했다. 따라서 여기서 “training data”라고 부를 수 있는 것은 모델 파라미터를 학습시키는 데이터라기보다, **시뮬레이션 사례를 만든 원천 데이터**에 가깝다.

---

### 3. 구체적인 테스트 입력과 출력 예시

#### 예시 A: 흉통 환자 진단

**Doctor agent의 초기 입력**

```text
환자가 흉통, 두근거림, 호흡곤란을 호소한다.
환자를 평가하고 진단하라.
```

의사는 처음부터 전체 병력이나 검사 결과를 받지 않는다.

**Doctor agent의 질문 또는 도구 사용**

```text
환자의 통증은 언제 시작되었습니까?
```

또는

```text
Perform EKG
```

**Patient agent의 출력**

```text
흉통은 약 30분 전에 시작되었고, 왼쪽 팔로 퍼집니다.
식은땀도 났습니다.
```

**Measurement agent의 출력**

```text
EKG: II, III, aVF 유도에서 ST분절 상승
Troponin I: 상승
CK-MB: 상승
Chest X-ray: 심장 크기 정상, 폐울혈 없음
```

**Doctor agent의 최종 출력**

```text
급성 심근경색, 특히 하벽 STEMI로 판단합니다.
```

**Moderator의 판정**

- 정답 진단: Acute Myocardial Infarction
- 의사 에이전트 진단: Acute Myocardial Infarction
- 출력: `Yes`
- 최종 평가: 정답

이 예시에서 핵심은 의사가 모든 정보를 처음부터 받는 것이 아니라, **질문과 검사 선택을 통해 필요한 정보를 수집해야 한다는 점**이다.

---

#### 예시 B: Hodgkin 림프종 사례

논문에서는 Hodgkin 림프종 환자의 경우 다음과 같은 검사 결과가 나올 수 있다고 설명한다.

**Doctor agent의 입력**

```text
환자의 전반적인 증상과 병력을 평가하라.
```

**Doctor agent의 검사 요청**

```text
CBC와 관련 혈액검사를 시행하라.
```

**Measurement agent의 출력 예시**

```text
Hemoglobin: 비정상
Platelet count: 비정상
White blood cell count: 비정상
기타 여러 혈액검사 수치 이상
```

**Doctor agent의 최종 출력**

```text
Hodgkin lymphoma가 가장 가능성이 높은 진단입니다.
```

이 사례에서도 검사 요청 자체가 평가 대상이다. 단순히 의학 지식을 알고 있는지뿐 아니라, **어떤 검사를 언제 요청할지**가 중요하다.

---

### 4. 실제 평가 과제

AgentClinic은 다음과 같은 과제를 포함한다.

1. **대화 기반 진단**
   - 환자에게 질문
   - 병력과 증상 파악
   - 제한된 질문 횟수 안에 진단

2. **의료검사 선택**
   - 혈압, CBC, EKG, X-ray, CT, MRI 등 요청
   - 검사 결과를 해석해 진단에 반영

3. **불완전한 정보 처리**
   - 필요한 정보가 자동으로 주어지지 않음
   - 의사가 직접 질문하거나 검사해야 함

4. **멀티모달 진단**
   - NEJM 사례의 의료 이미지를 입력으로 사용
   - 이미지가 처음부터 제공되거나, 의사가 요청할 때 제공됨

5. **전문 분야 진단**
   - 응급의학, 노인의학, 내과, 정신의학, 안과, 이비인후과, 소아과 등

6. **다국어 진단**
   - 영어, 중국어, 프랑스어, 스페인어, 힌디어, 페르시아어, 한국어

7. **도구 사용**
   - Zero-shot CoT
   - One-shot CoT
   - Reflection CoT
   - Adaptive RAG: 교과서 또는 인터넷 검색
   - Notebook: 이전 사례의 학습 내용을 저장하고 재사용

---

### 5. 도구 사용의 입력과 출력 예시

#### Adaptive RAG

**입력**

```text
Research textbooks "What are the symptoms of myasthenia gravis?"
```

**출력**

관련 의학 교과서 내용이 검색되어 의사의 프롬프트에 추가된다. 의사는 이 정보를 현재 환자의 증상 해석에 활용한다.

#### Notebook

**이전 사례 종료 후 저장되는 메모**

```text
[Note #17] 증상의 시작 시점과 진행 속도는 진단에 중요한 단서가 될 수 있다.
```

**새 사례에서의 입력**

```text
현재 사례와 관련된 이전 경험을 검색하라.
```

**출력**

과거에 성공적으로 해결한 사례와 메모가 검색되어 새로운 진단에 참고 정보로 제공된다.

---

### 6. 출력과 평가 지표

주요 출력은 의사의 **최종 진단**이며, 정답 여부는 정확도로 평가된다.

추가로 환자 에이전트가 다음을 1~10점으로 평가한다.

- **Confidence**: 의사의 판단을 얼마나 신뢰하는가
- **Compliance**: 치료를 따를 가능성
- **Consultation**: 같은 의사를 다시 찾을 가능성

또한 실제 임상의 3명이 대화의 다음 항목을 1~10점으로 평가했다.

- 의사 역할의 현실성: 6.2
- 환자 역할의 현실성: 6.7
- 검사 결과의 현실성: 6.3
- 공감 능력: 5.8

---

### 7. 논문의 핵심 결과

- AgentClinic-MedQA에서는 의사가 약 **20회의 상호작용**을 사용할 수 있다.
- 상호작용 횟수를 20회에서 10회로 줄이면 정확도가 **52%에서 25%**로 감소했다.
- AgentClinic에서는 필요한 정보의 평균 **67%**만 수집되었다.
  - 정답을 맞힌 경우: 72%
  - 오답인 경우: 63%
- 따라서 단순한 의학 지식보다 **질문을 잘하고, 필요한 검사를 선택하며, 정보를 충분히 수집하는 능력**이 중요하다.
- 정적 MedQA 점수는 AgentClinic 성능을 약하게만 예측했다.
- Claude 3.5가 대부분의 조건에서 가장 높은 성능을 보였다.
- Llama 3는 Notebook 도구 사용 시 성능이 크게 향상되었지만, 모든 모델이 도구 사용으로 좋아진 것은 아니다.

---




### 1. What does AgentClinic evaluate?

AgentClinic is not a conventional multiple-choice medical benchmark. It evaluates whether an AI agent can act like a doctor by:

1. talking with a simulated patient,
2. requesting appropriate medical tests,
3. interpreting the returned information, and
4. producing a final diagnosis.

The evaluated model is the **doctor agent**. It interacts with:

- **Patient agent**: knows the symptoms and medical history but not the true diagnosis.
- **Measurement agent**: returns results for tests such as blood pressure, EKG, laboratory tests, X-rays, CT, or MRI.
- **Moderator agent**: compares the doctor’s final diagnosis with the ground-truth diagnosis.

---

### 2. How are the data and cases constructed?

The study does not use a standard supervised-learning setup in which models are trained on a separate training set and then tested on a test set. Instead, pretrained models are evaluated on simulated clinical cases.

The cases are derived from:

- MedQA/USMLE questions
- de-identified MIMIC-IV electronic health records
- NEJM Case Challenges
- MedMCQA cases for specialist evaluations
- translated MedQA cases for multilingual evaluations

Each case is converted into a structured format. Information is distributed across agents:

| Agent | Information received |
|---|---|
| Doctor agent | Initial patient information and clinical objective |
| Patient agent | Symptoms, history, and lifestyle information |
| Measurement agent | Physical examination, laboratory, and imaging results |
| Moderator agent | Ground-truth diagnosis |

GPT-4 was initially used to populate the structured case files, and the cases were then manually validated. Thus, the source datasets function mainly as **case-generation data**, not as training data used to update the evaluated model’s parameters.

---

### 3. Example of a test input and output

#### Example: Patient with chest pain

**Initial input to the doctor agent**

```text
Evaluate a patient presenting with chest pain, palpitations, and shortness of breath.
```

The doctor does not initially receive the complete medical history or test results.

**Doctor agent’s interaction**

```text
When did the chest pain begin?
```

or

```text
Perform an EKG.
```

**Patient agent’s output**

```text
The chest pain began about 30 minutes ago and radiates to my left arm.
I also experienced sweating.
```

**Measurement agent’s output**

```text
EKG: ST-segment elevation in leads II, III, and aVF.
Troponin I: elevated.
CK-MB: elevated.
Chest X-ray: no pulmonary congestion; normal heart size.
```

**Doctor agent’s final output**

```text
The most likely diagnosis is acute myocardial infarction, specifically an inferior STEMI.
```

**Moderator evaluation**

```text
Ground truth: Acute Myocardial Infarction
Doctor diagnosis: Acute Myocardial Infarction
Output: Yes
```

The important feature is that the diagnosis cannot be made simply by reading a complete case vignette. The doctor must actively collect relevant information.

---

### 4. Main tasks in AgentClinic

AgentClinic evaluates:

1. **Dialogue-based diagnosis**
   - Asking questions
   - Gathering symptoms and history
   - Diagnosing within a limited number of turns

2. **Medical test selection**
   - Requesting tests such as CBC, EKG, blood pressure, X-ray, CT, or MRI
   - Interpreting the returned results

3. **Decision-making under incomplete information**
   - Relevant information is not automatically revealed
   - The agent must decide what to ask or measure

4. **Multimodal diagnosis**
   - Understanding medical images from NEJM cases
   - Images may be provided initially or only after the doctor requests them

5. **Specialist diagnosis**
   - Including emergency medicine, geriatrics, internal medicine, psychiatry, ophthalmology, otolaryngology, pediatrics, and other specialties

6. **Multilingual diagnosis**
   - English, Chinese, French, Spanish, Hindi, Persian, and Korean

7. **Tool use**
   - Zero-shot CoT
   - One-shot CoT
   - Reflection CoT
   - Adaptive RAG from textbooks or the web
   - A persistent notebook for experience-based learning

---

### 5. Examples of tool inputs and outputs

#### Adaptive RAG

**Input**

```text
Research textbooks "What are the symptoms of myasthenia gravis?"
```

**Output**

Relevant information retrieved from medical textbooks is added to the doctor agent’s context and can be used to interpret the current patient’s symptoms.

#### Notebook

**Stored note after a previous case**

```text
[Note #17] The timing and onset of symptoms can provide valuable diagnostic insights.
```

**Use in a new case**

```text
Retrieve relevant lessons from previous cases.
```

**Output**

Previously stored takeaways and similar successful cases are retrieved and provided as contextual examples for the new diagnosis.

---

### 6. Outputs and evaluation metrics

The primary output is the doctor agent’s **final diagnosis**, evaluated against the ground-truth diagnosis.

The simulated patient also provides three 1–10 ratings:

- **Confidence**: confidence in the doctor’s assessment
- **Compliance**: willingness to follow the recommended treatment
- **Consultation**: willingness to consult the same doctor again

Three human clinicians also rated dialogue quality on a 1–10 scale:

- Doctor realism: 6.2
- Patient realism: 6.7
- Measurement realism: 6.3
- Empathy: 5.8

---

### 7. Main findings

- Most experiments allowed approximately **20 interaction turns**.
- Reducing the number of turns from 20 to 10 reduced accuracy from **52% to 25%**.
- Doctors extracted, on average, **67%** of the relevant information available in the original case.
  - Correct-diagnosis cases: 72%
  - Incorrect-diagnosis cases: 63%
- Therefore, performance depends not only on medical knowledge but also on **asking effective questions, selecting useful tests, and gathering sufficient information**.
- Static MedQA performance was only weakly predictive of AgentClinic performance.
- Claude 3.5 generally performed best across the evaluated settings.
- Llama 3 showed substantial gains with the persistent notebook, but tool use did not improve every model.

<br/>
# 요약


AgentClinic은 환자·의사·검사·중재자 에이전트가 대화, 검사·영상 요청, 도구 사용을 수행하도록 설계하고, USMLE·MIMIC-IV·NEJM 사례를 바탕으로 9개 전문과, 7개 언어, 23개 편향 조건에서 임상 AI를 평가했다.  
정적 MedQA보다 순차적 상호작용에서 진단 정확도가 크게 낮아졌으며, MedQA-AgentClinic 간 성능 상관도 약했고, Claude 3.5가 대부분의 조건에서 가장 우수했지만 모델별 도구 활용 능력에는 큰 차이가 있었다.  
예를 들어 Llama 3는 노트북 도구 사용 시 정확도가 크게 향상된 반면 GPT-3.5는 일부 도구에서 성능이 하락했고, 편향은 진단 정확도뿐 아니라 환자의 의사 신뢰도·치료 순응도·재상담 의향도 낮췄다.  



AgentClinic evaluates clinical AI using patient, doctor, measurement, and moderator agents that interact through dialogue, test and image requests, and external tools across USMLE, MIMIC-IV, and NEJM cases, covering nine specialties, seven languages, and 23 bias conditions.  
Diagnostic accuracy dropped substantially compared with static MedQA, MedQA performance only weakly predicted AgentClinic performance, and Claude 3.5 generally performed best, while models differed markedly in their ability to benefit from tools.  
For example, Llama 3 improved substantially with the persistent notebook tool whereas GPT-3.5 often declined with tools, and bias reduced not only diagnostic accuracy but also patients’ confidence, treatment compliance, and willingness to consult again.

<br/>
# 기타



논문에 제시된 **다이어그램·피규어·테이블·어펜딕스 등 기타 자료**를 결과와 인사이트 중심으로 정리하면 다음과 같습니다.

## 1. Figure 1: AgentClinic의 전체 작동 구조

### 결과
AgentClinic은 네 종류의 언어 에이전트로 구성됩니다.

- **Doctor agent**: 진단을 수행하는 평가 대상 모델
- **Patient agent**: 증상과 병력을 제공하는 가상 환자
- **Measurement agent**: 혈압, 심전도, X-ray, MRI 등 검사 결과 제공
- **Moderator agent**: 의사의 최종 진단을 정답과 비교해 평가

의사 에이전트는 환자에게 질문하거나 검사를 요청할 수 있으며, 제한된 상호작용 횟수 안에 진단을 내려야 합니다. 대부분의 실험에서 의사 에이전트에게 **최대 20회의 상호작용**이 주어졌습니다.

### 핵심 인사이트
이 구조는 기존의 정적 객관식 문제와 달리, 모델이 다음 능력을 동시에 사용하도록 합니다.

- 어떤 질문을 할지 결정하는 능력
- 불완전한 정보에서 필요한 정보를 수집하는 능력
- 검사를 선택하고 결과를 해석하는 능력
- 대화 흐름을 유지하면서 최종 진단을 내리는 능력

즉, 단순한 의학 지식보다 **순차적 의사결정과 정보 수집 능력**을 평가합니다.

---

## 2. Figure 2: 모델별 AgentClinic 진단 정확도

### 결과
AgentClinic-MedQA에서 주요 결과는 다음과 같습니다.

| 모델 | 정확도 |
|---|---:|
| Claude 3.5 Sonnet | 62.1% |
| OpenBioLLM-70B | 58.3% |
| 인간 의사 3명 | 54.0% |
| GPT-4 | 51.6% |
| Mixtral-8×7B | 37.1% |
| GPT-3.5 | 36.6% |
| GPT-4o | 34.2% |
| MedLlama3-8B | 31.4% |
| Meditron-70B | 29.1% |
| PMC-Llama-7B | 23.6% |
| Llama 3 70B | 19.0% |
| Llama 2 70B-chat | 4.5% |

MIMIC-IV 기반 환경에서는 전반적으로 정확도가 더 낮았습니다.

- Claude 3.5: 42.9%
- OpenBioLLM-70B: 38.1%
- GPT-4: 34.0%
- Llama 3 70B: 8.5%

### 핵심 인사이트
- **Claude 3.5가 대부분의 환경에서 가장 우수**했습니다.
- 의학 특화 모델이라고 해서 반드시 높은 성능을 보이지는 않았습니다. 예를 들어 OpenBioLLM은 강했지만, 일부 의료 특화 모델은 GPT-4보다 낮았습니다.
- 인간 의사의 평균 정확도는 54%였지만 표준편차가 매우 컸습니다. 다만 인간 비교군은 3명뿐이므로 일반화에는 주의가 필요합니다.
- 정적 시험에서 강한 모델이 실제와 유사한 대화형 진단에서도 강하다고 보기는 어렵습니다.

---

## 3. Figure 3: MedQA와 AgentClinic-MedQA 비교

### 결과
논문은 기존의 정적 MedQA 정확도와 AgentClinic-MedQA 정확도 사이의 관계가 **약한 상관관계**만 보인다고 보고합니다.

AgentClinic에서는 모델이 모든 증상과 검사 결과를 처음부터 받지 않습니다. 필요한 정보를 직접 질문하거나 검사를 요청해야 합니다. 정보 수집률은 평균 **67%**였습니다.

- 정확한 진단을 한 경우: 정보 수집률 72%
- 오진한 경우: 정보 수집률 63%

### 핵심 인사이트
모델의 진단 실패는 의학 지식 부족만이 아니라, **중요한 정보를 충분히 끌어내지 못한 것**에서 비롯될 수 있습니다.

따라서 MedQA 점수만으로는 다음 능력을 평가하기 어렵습니다.

- 질문 우선순위 설정
- 환자 답변에서 핵심 정보 추출
- 추가 검사 선택
- 정보 부족 상태에서 판단하기

AgentClinic은 “무엇을 알고 있는가”보다 “무엇을 물어보고 어떻게 알아내는가”를 평가한다는 점에서 차별적입니다.

---

## 4. Figure 4: 인지 편향과 암묵적 편향의 영향

### 결과
GPT-4와 Mixtral-8×7B에 의사 또는 환자 편향을 주입했습니다.

#### GPT-4
- 편향이 없을 때 정확도: 52%
- 환자 인지 편향: 48%
- 의사 인지 편향: 50.3%
- 암묵적 편향 적용 시에도 대체로 50~51% 수준

GPT-4의 정확도 하락은 비교적 작았습니다.

#### Mixtral-8×7B
- 편향이 없을 때 정확도: 37%
- 의사 인지 편향: 약 29~32%
- 환자 편향: 약 31~35%
- 암묵적 편향: 약 32.7%

Mixtral은 GPT-4보다 편향에 더 취약했습니다. 특히 의사에게 인지 편향이 주어진 경우 정확도가 크게 낮아졌습니다.

### 환자 인식 결과
환자 에이전트는 진단 후 다음을 1~10점으로 평가했습니다.

- 의사에 대한 신뢰도
- 치료를 따를 의향
- 같은 의사를 다시 찾을 의향

인지 편향보다 **암묵적 편향**이 환자 인식에 더 큰 영향을 미쳤습니다. 특히 교육 수준, 성별, 문화적 배경 등에 관한 편향은 환자의 신뢰·순응도·재방문 의향을 낮췄습니다.

### 핵심 인사이트
진단 정확도만 보면 편향의 영향이 작아 보일 수 있습니다. 그러나 환자의 관점에서는 다음과 같은 부정적 효과가 나타날 수 있습니다.

- 의사를 신뢰하지 않음
- 치료를 따르지 않음
- 후속 진료를 받지 않음

따라서 임상 AI 평가는 정확도뿐 아니라 **환자 경험, 신뢰, 치료 순응도**도 함께 측정해야 합니다.

단, 이 환자 평가는 실제 사람이 아니라 LLM으로 시뮬레이션한 환자의 응답이므로 실제 환자 경험의 직접적인 대리 지표로 해석해서는 안 됩니다.

---

## 5. Figure 5: 전문과목·언어·도구 사용 결과

Figure 5는 세 영역을 함께 보여줍니다.

### 5-1. 전문과목별 성능

Claude 3.5의 평균 정확도는 66.7%였습니다.

상대적으로 높은 분야:

- 내과: 78.3%
- 이비인후과: 76.7%
- 산부인과: 74.3%

GPT-4는 산부인과와 안과에서 비교적 강했지만, 응급의학과와 노인의학에서는 성능이 낮았습니다.

### 인사이트
대화형 진단에서는 전문과목별 성능 차이가 컸습니다. 객관식 의료시험에서 상대적으로 쉬운 분야와 실제 대화형 환경에서 쉬운 분야가 반드시 일치하지 않았습니다.

이는 일부 전문과목이 단순 지식 회상보다 다음을 더 많이 요구하기 때문일 수 있습니다.

- 증상의 시간적 변화 파악
- 환자 표현 해석
- 감별진단을 위한 추가 질문
- 비정형적인 임상 맥락 처리

---

### 5-2. 다국어 성능

평가 언어는 영어, 중국어, 프랑스어, 스페인어, 힌디어, 페르시아어, 한국어였습니다.

- 모든 모델은 영어에서 가장 좋은 성능
- Claude 3.5 평균: 48.4%
- GPT-4 평균: 20.9%
- GPT-4의 정확도 범위: 중국어 11.21%~영어 40.18%
- GPT-4o는 한국어에서 3.73%까지 낮아짐
- 중국어는 대부분의 모델에서 특히 어려운 언어였음

### 인사이트
의료 AI의 다국어 능력은 단순한 일반 번역 능력과 다릅니다. 환자의 증상 표현, 문화적 맥락, 의료 용어를 동시에 처리해야 하기 때문입니다.

Claude 3.5는 다른 모델보다 언어 간 성능 편차가 작아, 다국어 임상 환경에서 상대적으로 안정적인 모습을 보였습니다.

---

### 5-3. Agent Toolbox의 효과

평가한 도구는 다음과 같습니다.

- Zero-shot Chain-of-Thought
- One-shot Chain-of-Thought
- Reflection CoT
- Adaptive RAG—교과서
- Adaptive RAG—웹
- Notebook

주요 결과:

- Claude 3.5 평균 정확도: 51.3%
- Claude 3.5의 Notebook 사용 시: 56.1%
- GPT-4는 Web RAG 사용 시 43.9%
- GPT-4는 Reflection CoT 사용 시 최고 42.2%
- GPT-4o는 Notebook 사용 시 43.0%
- Llama 3 70B는 도구 사용으로 평균 약 9.4% 향상
- Llama 3는 Notebook과 Reflection CoT 사용 시 41.1%
- GPT-3.5는 대부분의 도구에서 오히려 성능 하락
- GPT-3.5의 Book RAG 사용 시 성능이 27.1% 감소

### 핵심 인사이트
도구는 모든 모델에 동일하게 도움이 되지 않았습니다.

- 강한 모델은 검색·반성·메모리를 효과적으로 활용할 수 있음
- 일부 모델은 도구를 사용하면서 입력 복잡성이나 혼란이 증가함
- 따라서 “도구를 제공했는가”보다 **모델이 도구를 언제, 왜, 어떻게 사용하는가**가 중요함

특히 Notebook은 과거 사례에서 얻은 교훈을 다음 사례에 적용하게 하므로, 단순 검색보다 **경험 기반 학습**에 가깝습니다.

참고로 논문의 초록은 Llama 3의 Notebook 사용으로 최대 92%의 상대적 향상을 언급하지만, 본문에서는 평균 9.4% 향상 및 Notebook 사용 시 41.1% 등의 수치를 제시합니다. 따라서 92%는 특정 기준 또는 특정 조건에서의 최대 상대 향상으로 이해하는 것이 안전합니다.

---

## 6. Figure 6: 멀티모달 진단 성능

### 결과
NEJM 사례 120개를 사용해 이미지와 대화를 함께 처리하도록 했습니다.

#### 이미지가 처음부터 제공된 경우

- Claude 3.5: 37.2%
- GPT-4: 27.7%
- GPT-4o: 21.4%
- GPT-4o-mini: 8.0%

#### 의사가 요청해야 이미지를 받는 경우

- Claude 3.5: 35.4%
- GPT-4: 25.4%
- GPT-4o: 19.1%
- GPT-4o-mini: 6.1%

### 핵심 인사이트
- Claude 3.5가 이미지가 처음 제공되거나 요청 후 제공되는 두 조건 모두에서 가장 높았습니다.
- 이미지를 직접 요청해야 하는 조건에서는 모든 모델의 성능이 소폭 하락했습니다.
- 이는 시각적 이해뿐 아니라 “이미지가 필요하다는 사실을 판단하고 요청하는 능력”도 중요한 요소임을 보여줍니다.
- 이미지와 언어를 함께 처리하는 멀티모달 진단은 여전히 어려웠으며, 단순 이미지 인식 능력만으로 해결되지 않았습니다.

---

## 7. Human dialogue ratings: 대화 품질 평가

### 결과
의사 3명이 20개의 영어 대화를 1~10점으로 평가했습니다.

| 평가 항목 | 평균 점수 |
|---|---:|
| 의사 에이전트의 현실성 | 6.2 |
| 환자 에이전트의 현실성 | 6.7 |
| 검사 결과의 현실성 | 6.3 |
| 공감 능력 | 5.8 |

주요 지적 사항은 다음과 같습니다.

- 의사 에이전트가 대화를 너무 급하게 시작함
- 특정 진단에 과도하게 고정됨
- 기본적인 오류를 범함
- 환자 에이전트가 질문을 반복하거나 지나치게 장황함
- 검사 에이전트가 필요한 결과를 일부 누락함
- 의사 에이전트의 말투가 중립적이고 공감 표현이 부족함

### 핵심 인사이트
진단 정확도가 높아도 실제 임상 대화의 질이 자동으로 높아지는 것은 아닙니다. 임상 AI는 다음을 별도로 평가받아야 합니다.

- 대화의 자연스러움
- 공감과 설명 능력
- 검사 결과의 완전성
- 환자 중심성

다만 평가는 3명의 의사가 20개 대화만 검토한 소규모 연구이므로 일반화에는 한계가 있습니다.

---

## 8. Tables와 Appendix

제공된 본문에는 별도의 주요 표(table)나 상세 어펜딕스가 포함되어 있지 않습니다. 논문의 정량적 결과는 주로 Figures 1~6과 본문에 제시되어 있습니다.

다만 논문은 다음 자료를 공개한다고 설명합니다.

- AgentClinic 데이터
- 코드
- 프롬프트
- 평가 스크립트

GitHub 저장소:

`github.com/SamuelSchmidgall/AgentClinic`

또한 MIMIC-IV 기반 데이터는 PhysioNet 규정에 따라 별도 접근 승인이 필요합니다.

### 관련 인사이트
재현성 측면에서는 코드와 프롬프트가 공개되어 있다는 점이 강점입니다. 그러나 다음 요소들은 결과에 영향을 줄 수 있어 추가 검증이 필요합니다.

- Moderator agent의 모델과 프롬프트
- 환자 에이전트의 모델 선택
- LLM 기반 검사 결과 생성
- 시뮬레이션 환자 평가의 신뢰도
- 인간 평가자의 적은 표본 수

---

## 전체 요약

AgentClinic의 핵심 메시지는 다음과 같습니다.

1. **정적 의학 QA 성능은 실제와 유사한 대화형 진단 능력을 충분히 예측하지 못한다.**
2. 진단 정확도는 의학 지식뿐 아니라 **질문 전략과 정보 수집 능력**에 크게 좌우된다.
3. 모델별로 도구 사용 효과가 크게 다르며, 도구를 제공하는 것만으로 성능 향상이 보장되지 않는다.
4. 편향은 진단 정확도뿐 아니라 환자의 신뢰·치료 순응도·재방문 의향에도 영향을 줄 수 있다.
5. 멀티모달·다국어·전문과목 환경에서는 모델 간 성능 격차가 더욱 커진다.
6. 따라서 임상 AI 평가는 정확도 하나가 아니라 **순차적 의사결정, 도구 활용, 편향 강건성, 대화 품질, 환자 중심 지표**를 함께 봐야 한다.

---




## 1. Figure 1: Overall AgentClinic workflow

### Results
AgentClinic consists of four language agents:

- **Doctor agent**: the model being evaluated
- **Patient agent**: a simulated patient providing symptoms and history
- **Measurement agent**: returns examination and imaging results
- **Moderator agent**: compares the final diagnosis with the ground truth

The doctor agent must ask questions, request tests, and reach a diagnosis within a limited number of interactions. In most experiments, it was given **20 interaction turns**.

### Key insight
Unlike static multiple-choice questions, the benchmark evaluates whether a model can:

- Decide what to ask
- Collect missing information
- Select appropriate tests
- Interpret results
- Make a final diagnosis under uncertainty

Thus, it measures **sequential clinical decision-making**, not only medical knowledge.

---

## 2. Figure 2: Diagnostic accuracy across models

### Results
On AgentClinic-MedQA, the main results were:

| Model | Accuracy |
|---|---:|
| Claude 3.5 Sonnet | 62.1% |
| OpenBioLLM-70B | 58.3% |
| Human physicians | 54.0% |
| GPT-4 | 51.6% |
| Mixtral-8×7B | 37.1% |
| GPT-3.5 | 36.6% |
| GPT-4o | 34.2% |
| MedLlama3-8B | 31.4% |
| Meditron-70B | 29.1% |
| PMC-Llama-7B | 23.6% |
| Llama 3 70B | 19.0% |
| Llama 2 70B-chat | 4.5% |

Performance was generally lower on the MIMIC-IV-based environment. For example:

- Claude 3.5: 42.9%
- OpenBioLLM-70B: 38.1%
- GPT-4: 34.0%
- Llama 3 70B: 8.5%

### Key insight
- **Claude 3.5 performed best in most settings.**
- Medical specialization alone did not guarantee high performance.
- Human performance was highly variable, and the human baseline included only three physicians.
- Strong performance on static medical exams did not necessarily translate into strong interactive diagnostic performance.

---

## 3. Figure 3: MedQA versus AgentClinic-MedQA

### Results
The study found only a **weak relationship** between performance on static MedQA and interactive AgentClinic-MedQA.

In AgentClinic, the doctor does not receive all symptoms and test results upfront. It must actively obtain them. The average information coverage was **67%**.

- Correct diagnosis: 72% coverage
- Incorrect diagnosis: 63% coverage

### Key insight
Diagnostic failures may result not only from a lack of medical knowledge, but also from failure to obtain critical information.

AgentClinic therefore evaluates:

- Question prioritization
- Information elicitation
- Test selection
- Reasoning with incomplete information

This makes it substantially different from a conventional knowledge benchmark.

---

## 4. Figure 4: Effects of cognitive and implicit biases

### Results
Biases were introduced into either the doctor or patient agent.

#### GPT-4
- Unbiased accuracy: 52%
- Patient cognitive bias: 48%
- Doctor cognitive bias: 50.3%
- Implicit-bias conditions: generally around 50–51%

GPT-4 showed relatively small accuracy decreases.

#### Mixtral-8×7B
- Unbiased accuracy: 37%
- Doctor cognitive bias: approximately 29–32%
- Patient bias: approximately 31–35%
- Implicit bias: approximately 32.7%

Mixtral was more vulnerable to bias, especially when the doctor agent was biased.

### Patient perception results
The simulated patient rated:

- Confidence in the doctor
- Willingness to follow treatment
- Willingness to consult the same doctor again

Implicit biases generally affected these perceptions more strongly than cognitive biases. Education-, gender-, cultural-, and socioeconomic-related biases reduced trust, compliance, and willingness to return.

### Key insight
Diagnostic accuracy alone may underestimate the impact of bias. Even when accuracy changes only slightly, bias can reduce:

- Patient trust
- Treatment adherence
- Follow-up consultation willingness

However, these ratings were generated by simulated LLM patients and should not be treated as direct measures of real patient experience.

---

## 5. Figure 5: Specialties, languages, and tool use

### 5-1. Medical specialties

Claude 3.5 achieved an average accuracy of 66.7%, with particularly strong performance in:

- Internal medicine: 78.3%
- Otolaryngology: 76.7%
- Gynecology: 74.3%

GPT-4 performed relatively well in gynecology and ophthalmology but poorly in emergency medicine and geriatrics.

### Key insight
Interactive diagnostic difficulty varied substantially by specialty. The specialties that appear easy in multiple-choice medical QA are not necessarily easy in dialogue-based diagnosis.

Dialogue-based cases may require:

- Tracking symptom progression
- Interpreting patient descriptions
- Asking targeted follow-up questions
- Handling atypical clinical contexts

---

### 5-2. Multilingual performance

The benchmark covered English, Chinese, French, Spanish, Hindi, Persian, and Korean.

- All models performed best in English.
- Claude 3.5 average accuracy: 48.4%
- GPT-4 average accuracy: 20.9%
- GPT-4 ranged from 11.21% in Chinese to 40.18% in English.
- GPT-4o reached only 3.73% in Korean.
- Chinese was particularly difficult for most models.

### Key insight
Multilingual clinical performance involves more than translation. Models must simultaneously interpret symptoms, medical terminology, and culturally shaped communication patterns.

Claude 3.5 showed the most consistent performance across languages.

---

### 5-3. Agent Toolbox

The evaluated tools included:

- Zero-shot Chain-of-Thought
- One-shot Chain-of-Thought
- Reflection CoT
- Adaptive RAG from textbooks
- Adaptive RAG from the web
- Notebook-based experiential memory

Main findings:

- Claude 3.5 average accuracy: 51.3%
- Claude 3.5 with Notebook: 56.1%
- GPT-4 with Web RAG: 43.9%
- GPT-4 with Reflection CoT: 42.2%
- GPT-4o with Notebook: 43.0%
- Llama 3 70B improved by approximately 9.4% on average across tools.
- Llama 3 reached 41.1% with Notebook and Reflection CoT.
- GPT-3.5 generally declined with tool use.
- GPT-3.5 showed a 27.1% decrease with textbook RAG.

### Key insight
Tools were not uniformly beneficial. Their value depended on whether the model could use them appropriately.

The Notebook tool is particularly important because it supports **experiential learning**: the model stores lessons from previous cases and retrieves them for future cases.

The abstract reports up to a 92% relative improvement for Llama 3 with Notebook, whereas the detailed results report an average improvement of 9.4% and an accuracy of 41.1% under selected tool conditions. The 92% figure should therefore be understood as a maximum relative improvement under a specific comparison.

---

## 6. Figure 6: Multimodal diagnostic performance

### Results
The study used 120 NEJM cases requiring both dialogue and image interpretation.

#### Images provided initially

- Claude 3.5: 37.2%
- GPT-4: 27.7%
- GPT-4o: 21.4%
- GPT-4o-mini: 8.0%

#### Images provided only after being requested

- Claude 3.5: 35.4%
- GPT-4: 25.4%
- GPT-4o: 19.1%
- GPT-4o-mini: 6.1%

### Key insight
Claude 3.5 performed best under both image-delivery conditions.

Performance decreased slightly when models had to request the image, suggesting that multimodal diagnosis requires not only visual interpretation but also the ability to recognize **when additional visual information is needed**.

---

## 7. Human dialogue ratings

### Results
Three physicians rated 20 English-language dialogues on a 1–10 scale.

| Category | Mean score |
|---|---:|
| Doctor realism | 6.2 |
| Patient realism | 6.7 |
| Measurement realism | 6.3 |
| Empathy | 5.8 |

Common criticisms included:

- Abrupt opening statements
- Overcommitment to one diagnosis
- Basic errors
- Repetitive or overly verbose patient responses
- Incomplete test results
- Limited empathy and patient-centered communication

### Key insight
High diagnostic accuracy does not automatically imply high-quality clinical communication. Clinical AI should also be evaluated for:

- Realistic dialogue
- Empathy
- Completeness of test reporting
- Patient-centered interaction

The human evaluation was small, involving only three physicians and 20 dialogues.

---

## 8. Tables and Appendices

The provided article text does not include separate major tables or detailed appendices. Most quantitative results are presented in Figures 1–6 and the main text.

The authors state that they release:

- AgentClinic data
- Code
- Prompts
- Evaluation scripts

Repository:

`github.com/SamuelSchmidgall/AgentClinic`

MIMIC-IV-based data require separate access through PhysioNet procedures.

### Key insight
Open code and prompts improve reproducibility. However, results may still depend on:

- The moderator model and prompt
- The selected patient-agent model
- LLM-generated measurement results
- The reliability of simulated patient ratings
- The small human-evaluation sample

---

## Overall takeaway

AgentClinic’s central message is that:

1. Static medical QA scores do not reliably predict interactive diagnostic ability.
2. Diagnostic accuracy depends heavily on **information-gathering and questioning strategy**.
3. Tool use is highly model-dependent and is not automatically beneficial.
4. Bias can affect not only diagnostic accuracy but also trust, treatment adherence, and follow-up behavior.
5. Multimodal, multilingual, and specialty-specific settings reveal larger differences between models.
6. Clinical AI evaluation should therefore include **sequential decision-making, tool use, bias robustness, dialogue quality, and patient-centered outcomes**, rather than accuracy alone.

<br/>
# refer format:
### BibTeX

```bibtex
@article{schmidgall2026agentclinic,
  author  = {Schmidgall, Samuel and Ziaei, Rojin and Harris, Carl and Kim, JiWoong and Pontes Reis, Eduardo and Jopling, Jeffrey and Moor, Michael},
  title   = {AgentClinic: A Multimodal Benchmark for Tool-Using Clinical AI Agents},
  journal = {npj Digital Medicine},
  year    = {2026},
  volume  = {9},
  number  = {1},
  pages   = {499},
  doi     = {10.1038/s41746-026-02674-7},
  url     = {https://doi.org/10.1038/s41746-026-02674-7}
}
```

### Chicago 

Schmidgall, Samuel, Rojin Ziaei, Carl Harris, JiWoong Kim, Eduardo Pontes Reis, Jeffrey Jopling, and Michael Moor. “AgentClinic: A Multimodal Benchmark for Tool-Using Clinical AI Agents.” *npj Digital Medicine* 9 (2026): 499. https://doi.org/10.1038/s41746-026-02674-7.


