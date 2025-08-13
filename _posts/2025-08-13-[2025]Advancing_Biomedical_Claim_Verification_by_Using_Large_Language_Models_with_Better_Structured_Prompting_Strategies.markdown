---
layout: post
title:  "[2025]Advancing Biomedical Claim Verification by Using Large Language Models with Better Structured Prompting Strategies"
date:   2025-08-13 16:54:28 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

LLM으로 biomedical claim verification((1) 주장 이해, (2) 증거 분석, (3) 중간 결론 도출, (4) 포함 관계 결정)
-> 오류 추적 및 모델의 추론 과정을 이해하는 데 도움     

짧은 요약(Abstract) :

이 논문에서는 생물 의학적 주장 검증(biomedical claim verification) 과정을 개선하기 위해 대형 언어 모델(LLM)을 활용한 구조화된 4단계 프롬프트 전략을 제안합니다. 이 과정은 (1) 주장 이해, (2) 증거 분석, (3) 중간 결론 도출, (4) 포함 관계 결정의 네 가지 단계로 구성되어 있습니다. 이 전략은 구성적이고 인간과 유사한 추론을 활용하여 논리적 일관성과 사실적 기반을 강화하고, LLM이 다양한 생물 의학적 주장 검증 작업에서 추론 패턴을 일반화할 수 있도록 돕습니다. 실험 결과, 각 단계가 독특하면서도 상호 보완적인 역할을 수행함을 보여주며, 체계적인 프롬프트와 단계별 지침이 LLM의 잠재적 인지 능력을 발휘하게 하고 오류 추적 및 모델의 추론 과정을 이해하는 데 도움을 줍니다. 이 연구는 AI 기반 생물 의학적 주장 검증의 신뢰성을 향상시키는 것을 목표로 합니다.


This paper proposes a structured four-step prompting strategy that utilizes large language models (LLMs) to improve the process of biomedical claim verification. The process consists of four stages: (1) claim comprehension, (2) evidence analysis, (3) intermediate conclusion, and (4) entailment decision-making. This strategy leverages compositional and human-like reasoning to enhance logical consistency and factual grounding, helping LLMs generalize reasoning patterns across various biomedical claim verification tasks. Experimental results demonstrate that each step plays distinct yet complementary roles, and systematic prompting along with carefully designed step-wise instructions unlocks the latent cognitive abilities of LLMs, facilitating error tracing and understanding of the model's reasoning process. This research aims to enhance the reliability of AI-driven biomedical claim verification.


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


이 연구에서는 생물 의학 주장 검증을 위한 새로운 접근 방식을 제안합니다. 이 방법은 대형 언어 모델(LLM)을 활용하여 주장을 이해하고, 증거를 분석하며, 중간 결론을 도출하고, 최종적으로 주장과 증거 간의 관계를 결정하는 네 단계의 구조화된 프롬프트 전략을 포함합니다. 이 방법은 다음과 같은 주요 요소로 구성됩니다.

1. **모델 선택**: 연구에서는 다양한 경량 LLM을 사용하여 실험을 수행했습니다. 이 모델들은 주로 3.6억에서 14억 개의 파라미터를 가진 모델들로, GPT-3.5와 GPT-4o-mini 모델도 포함됩니다. 이러한 모델들은 비용 효율적이며, 생물 의학 데이터에 대한 추론 능력을 향상시키기 위해 설계되었습니다.

2. **프롬프트 설계**: 각 단계는 명확한 지침을 제공하여 모델이 특정 작업을 수행하도록 유도합니다. 첫 번째 단계인 '주장 이해'에서는 모델이 주장의 주요 용어와 개념을 해석하도록 지시합니다. 두 번째 단계인 '증거 분석'에서는 모델이 제공된 임상 시험 데이터를 바탕으로 주장을 검증하기 위해 관련 데이터를 식별하고 비교하도록 합니다. 세 번째 단계인 '중간 결론'에서는 모델이 증거를 바탕으로 주장이 논리적으로 따르는지를 결론짓도록 유도합니다. 마지막으로, 네 번째 단계인 '관계 결정'에서는 모델이 최종적으로 '포함' 또는 '모순'과 같은 관계를 예측하도록 합니다.

3. **훈련 데이터**: 연구에서는 NLI4CT 데이터셋을 주요 평가 작업으로 사용했습니다. 이 데이터셋은 생물 의학 주장 검증에 필요한 복잡한 수치 및 도메인 전문 지식 추론을 포함하고 있습니다. 또한, SciFact와 HealthVer와 같은 관련 벤치마크를 통해 일반화 능력을 평가했습니다.

4. **지도 학습**: 경량 LLM의 성능을 향상시키기 위해, 연구에서는 GPT-4o-mini 모델을 사용하여 NLI4CT 훈련 세트에서 고품질의 훈련 예제를 생성했습니다. 이 예제들은 모델의 추론 과정을 개선하고, 일관된 응답 형식을 유지하는 데 도움을 주었습니다.

5. **성능 평가**: 모델의 성능은 F1 점수를 사용하여 평가되었습니다. 실험 결과, 제안된 4단계 프롬프트 전략이 기존의 2단계 방법보다 성능을 크게 향상시켰음을 보여주었습니다. 특히, 경량 LLM이 복잡한 생물 의학 주장 검증 작업을 수행하는 데 있어 더 높은 정확도를 달성했습니다.

이러한 방법론은 생물 의학 분야에서 AI 기반 주장 검증의 신뢰성을 높이는 데 기여할 것으로 기대됩니다.

---



This study proposes a novel approach for biomedical claim verification, incorporating a structured four-step prompting strategy that utilizes large language models (LLMs). This method consists of the following key components:

1. **Model Selection**: The research employs various lightweight LLMs for experimentation, primarily models with parameters ranging from 3.6 billion to 14 billion, including GPT-3.5 and GPT-4o-mini. These models are cost-effective and designed to enhance reasoning capabilities in biomedical data.

2. **Prompt Design**: Each step provides clear instructions to guide the model in performing specific tasks. In the first step, 'Claim Comprehension,' the model is instructed to interpret the key terms and concepts within the claim. The second step, 'Evidence Analysis,' directs the model to identify and compare relevant data from the provided clinical trial data to verify the claim. The third step, 'Interim Conclusion,' encourages the model to draw conclusions based on the evidence regarding whether the claim logically follows. Finally, the fourth step, 'Entailment Decision-making,' prompts the model to predict the relationship as either 'Entailment' or 'Contradiction.'

3. **Training Data**: The study uses the NLI4CT dataset as the primary evaluation task, which includes complex numerical and domain-specific reasoning necessary for biomedical claim verification. Additionally, related benchmarks such as SciFact and HealthVer are utilized to assess generalization capabilities.

4. **Supervised Fine-Tuning**: To improve the performance of lightweight LLMs, the study employs the GPT-4o-mini model to generate high-quality training examples from the NLI4CT training set. These examples help refine the model's reasoning process and maintain a consistent response format.

5. **Performance Evaluation**: The performance of the models is evaluated using the F1 score. Experimental results demonstrate that the proposed four-step prompting strategy significantly enhances performance compared to the existing two-step method. Notably, lightweight LLMs achieved higher accuracy in performing complex biomedical claim verification tasks.

This methodology is expected to contribute to improving the reliability of AI-driven claim verification in the biomedical field.


<br/>
# Results


이 연구에서는 생물 의학 주장 검증을 위한 4단계 프롬프트 전략을 제안하고, 이를 통해 다양한 경량 대형 언어 모델(LLM)의 성능을 평가했습니다. 실험은 NLI4CT 데이터셋을 포함하여 SciFact와 HealthVer와 같은 관련 벤치마크에서 수행되었습니다. 각 모델의 성능은 F1 점수로 평가되었으며, 이 점수는 모델이 주어진 주장과 증거 간의 관계를 얼마나 정확하게 예측하는지를 나타냅니다.

#### 경쟁 모델
연구에서는 여러 경량 LLM 모델을 비교했습니다. 주요 모델로는 GPT3.5, GPT-4o-mini, Phi3.5-3.6B, Mistral-7B, Llama3.1-8B, Gemma2-9B, Mistral-12B, Phi3-14B 등이 포함되었습니다. 이들 모델은 각각의 프롬프트 전략에 따라 성능이 달라졌습니다.

#### 테스트 데이터
NLI4CT 데이터셋은 생물 의학 주장 검증을 위한 주요 평가 데이터셋으로, 각 주장과 관련된 임상 시험 데이터가 포함되어 있습니다. 이 데이터셋은 주장과 증거 간의 관계를 평가하는 데 필요한 복잡한 수치적 및 도메인 특화 지식 추론을 요구합니다.

#### 메트릭
모델의 성능 평가는 F1 점수를 사용하여 이루어졌습니다. F1 점수는 정밀도와 재현율의 조화 평균으로, 모델이 얼마나 정확하게 주장을 검증하는지를 나타냅니다. F1 점수는 다음과 같이 계산됩니다:

\[ F1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \]

#### 비교 결과
실험 결과, 4단계 프롬프트 전략을 적용한 모델들이 단순 2단계 프롬프트와 비교했을 때 성능이 크게 향상되었습니다. 예를 들어, NLI4CT 데이터셋에서 GPT3.5 모델은 2단계 프롬프트에서 0.53의 F1 점수를 기록했으나, 4단계 프롬프트를 적용했을 때 0.75로 증가했습니다. 비슷한 방식으로, GPT-4o-mini 모델은 2단계에서 0.77에서 4단계에서 0.86으로 성능이 향상되었습니다.

이러한 결과는 4단계 프롬프트 전략이 LLM의 추론 능력을 극대화하고, 생물 의학 주장 검증 작업에서의 정확성을 높이는 데 효과적임을 보여줍니다. 특히, 각 단계가 명확하게 정의되어 있어 모델이 특정 하위 작업에 집중할 수 있도록 도와주며, 이는 모호성을 줄이고 정확성을 높이는 데 기여합니다.



This study proposes a four-step prompting strategy for biomedical claim verification and evaluates the performance of various lightweight large language models (LLMs) using this approach. The experiments were conducted on the NLI4CT dataset, along with related benchmarks such as SciFact and HealthVer. The performance of each model was assessed using the F1 score, which indicates how accurately the model predicts the relationship between a given claim and evidence.

#### Competing Models
The study compared several lightweight LLMs, including GPT3.5, GPT-4o-mini, Phi3.5-3.6B, Mistral-7B, Llama3.1-8B, Gemma2-9B, Mistral-12B, and Phi3-14B. The performance of these models varied depending on the prompting strategies employed.

#### Test Data
The NLI4CT dataset serves as the primary evaluation dataset for biomedical claim verification, containing clinical trial data related to each claim. This dataset requires complex numerical and domain-specific knowledge reasoning to assess the relationship between claims and evidence.

#### Metrics
The performance of the models was evaluated using the F1 score, which is a harmonic mean of precision and recall, indicating how accurately the model verifies claims. The F1 score is calculated as follows:

\[ F1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \]

#### Comparison Results
The experimental results showed that models applying the four-step prompting strategy significantly outperformed those using a simple two-step prompting approach. For instance, the GPT3.5 model recorded an F1 score of 0.53 with the two-step prompt but increased to 0.75 when the four-step prompt was applied. Similarly, the GPT-4o-mini model improved from 0.77 in the two-step to 0.86 in the four-step approach.

These results demonstrate that the four-step prompting strategy effectively maximizes the reasoning capabilities of LLMs and enhances accuracy in biomedical claim verification tasks. Notably, the clear definition of each step allows the model to focus on specific subtasks, contributing to reduced ambiguity and increased accuracy.


<br/>
# 예제


이 논문에서는 생물 의학 주장 검증(biomedical claim verification) 작업을 수행하기 위해 대형 언어 모델(LLM)을 활용하는 방법을 제안합니다. 이 작업은 주어진 주장(claim)과 임상 시험 보고서(clinical trial reports)에서 파생된 증거(evidence) 간의 관계를 판단하는 것입니다. 연구자들은 이 과정을 네 단계로 나누어 구조화된 프롬프트(prompt) 전략을 사용하여 LLM의 성능을 향상시키고자 했습니다.

#### 예시: 트레이닝 데이터와 테스트 데이터

1. **트레이닝 데이터**:
   - **입력**: "임상 시험에서 100명의 환자 중 98명이 무병 생존(recurring-free survival)을 달성했습니다. 따라서 2명의 환자는 무병 생존을 달성하지 못했습니다."
   - **출력**: "주장: '임상 시험에서 단 2명의 환자가 무병 생존을 달성하지 못했습니다.' 이 주장은 임상 시험 데이터에 의해 지지됩니다. 따라서 이 주장은 'Entailment'입니다."

2. **테스트 데이터**:
   - **입력**: "임상 시험에서 100명의 환자 중 98명이 무병 생존을 달성했습니다. 따라서 2명의 환자는 무병 생존을 달성하지 못했습니다."
   - **출력**: "주장: '임상 시험에서 단 2명의 환자가 무병 생존을 달성하지 못했습니다.' 이 주장은 임상 시험 데이터에 의해 지지됩니다. 따라서 이 주장은 'Entailment'입니다."

#### 구체적인 작업 설명
- **작업 정의**: 주어진 주장(C)과 임상 시험 데이터(P) 간의 관계를 판단하는 이진 분류 문제로 설정합니다. 모델은 주장이 임상 시험 데이터에서 논리적으로 따르는지(Entailment) 또는 그렇지 않은지(Contradiction)를 결정합니다.
- **프롬프트 단계**:
  1. **주장 이해(Claim Comprehension)**: 모델이 주장을 해석하도록 유도합니다.
  2. **증거 분석(Evidence Analysis)**: 모델이 임상 시험 데이터를 분석하여 주장을 검증하는 데 필요한 증거를 식별합니다.
  3. **중간 결론(Interim Conclusion)**: 모델이 증거를 바탕으로 주장이 논리적으로 따르는지를 결론짓도록 유도합니다.
  4. **관계 결정(Entailment Decision-making)**: 최종적으로 모델이 주장이 임상 시험 데이터에서 논리적으로 따르는지 여부를 결정합니다.

이러한 구조화된 접근 방식은 LLM이 복잡한 생물 의학 주장 검증 작업을 수행하는 데 있어 더 높은 정확성을 달성하도록 돕습니다.

---



This paper proposes a method for utilizing large language models (LLMs) in the task of biomedical claim verification. This task involves determining the relationship between a given claim and evidence derived from clinical trial reports. The researchers aim to enhance the performance of LLMs by employing a structured prompting strategy that decomposes the verification process into four distinct steps.

#### Example: Training Data and Test Data

1. **Training Data**:
   - **Input**: "In the clinical trial, 98 out of 100 patients achieved recurrence-free survival. Therefore, 2 patients did not achieve recurrence-free survival."
   - **Output**: "Claim: 'Only 2 patients in the primary trial did not achieve recurrence-free survival.' This claim is supported by the clinical trial data. Therefore, the claim is 'Entailment.'"

2. **Test Data**:
   - **Input**: "In the clinical trial, 98 out of 100 patients achieved recurrence-free survival. Therefore, 2 patients did not achieve recurrence-free survival."
   - **Output**: "Claim: 'Only 2 patients in the primary trial did not achieve recurrence-free survival.' This claim is supported by the clinical trial data. Therefore, the claim is 'Entailment.'"

#### Detailed Task Description
- **Task Definition**: The task is framed as a binary classification problem where the model determines whether the claim (C) logically follows from the premise (P) provided in clinical trial data. The model predicts whether the claim is entailed (Entailment) or contradicted (Contradiction) by the evidence.
- **Prompting Steps**:
  1. **Claim Comprehension**: The model is prompted to interpret the claim.
  2. **Evidence Analysis**: The model analyzes the clinical trial data to identify the evidence needed to verify the claim.
  3. **Interim Conclusion**: The model is guided to draw a conclusion based on the evidence regarding whether the claim logically follows.
  4. **Entailment Decision-making**: Finally, the model predicts whether the claim logically follows from the clinical trial data.

This structured approach helps LLMs achieve higher accuracy in performing complex biomedical claim verification tasks.

<br/>
# 요약

이 연구에서는 생물 의학 주장 검증을 위한 4단계 구조화된 프롬프트 전략을 제안하며, 각 단계는 주장 이해, 증거 분석, 중간 결론 도출, 그리고 함의 결정으로 구성된다. 실험 결과, 이 접근 방식이 경량 LLM의 성능을 약 10% 향상시키는 것으로 나타났으며, 특히 증거 분석 단계에서의 개선이 두드러졌다. 최종적으로, 이 방법은 LLM의 추론 능력을 향상시키고 해석 가능성을 높이는 데 기여한다.


This study proposes a structured four-step prompting strategy for biomedical claim verification, consisting of claim comprehension, evidence analysis, interim conclusion, and entailment decision-making. Experimental results show that this approach improves the performance of lightweight LLMs by approximately 10%, with significant enhancements observed particularly in the evidence analysis stage. Ultimately, this method contributes to enhancing the reasoning capabilities of LLMs and increasing interpretability.

<br/>
# 기타


#### 1. 다이어그램 및 피규어
- **Figure 1**: NLI4CT 데이터셋의 예시를 통해 임상 시험 데이터와 주장의 관계를 시각적으로 설명합니다. 이 그림은 NLI 작업의 복잡성을 강조하며, 모델이 주장을 검증하기 위해 필요한 도메인 지식과 다단계 추론을 요구함을 보여줍니다.
  
- **Figure 3**: 제안된 4단계 프롬프트 프레임워크의 구조를 나타냅니다. 각 단계는 주장을 이해하고, 증거를 분석하며, 중간 결론을 도출하고, 최종적으로 관계 예측을 수행하는 과정을 명확히 합니다. 이 구조는 LLM이 특정 하위 작업에 집중하도록 돕고, 모호성을 줄이며 정확성을 높입니다.

#### 2. 테이블
- **Table 1**: NLI4CT, SciFact, HealthVer 데이터셋의 각 관계 클래스에 대한 인스턴스 분포를 보여줍니다. 이 표는 각 데이터셋에서의 주장과 증거 간의 관계를 이해하는 데 필요한 샘플 수를 제공합니다.

- **Table 3**: 다양한 모델의 F1 점수를 비교하여 4단계 프롬프트 프레임워크가 성능을 어떻게 향상시키는지를 보여줍니다. 4단계 접근 방식이 단순한 2단계 및 기본 프롬프트 템플릿보다 상당한 성능 향상을 가져온 것을 확인할 수 있습니다.

- **Table 4**: 경량 LLM의 SFT(지도 학습) 결과를 보여줍니다. SFT가 모델의 성능을 어떻게 향상시키는지를 나타내며, 훈련 샘플 수가 증가함에 따라 성능이 향상됨을 보여줍니다.

- **Table 10-15**: 다양한 모델의 응답을 비교하여 2단계 및 4단계 프롬프트 프레임워크에서의 성능 차이를 보여줍니다. 각 모델이 주장을 어떻게 해석하고 증거를 분석하는지에 대한 예시를 제공합니다. 이 표들은 4단계 접근 방식이 더 정확한 결론을 도출하는 데 어떻게 기여하는지를 강조합니다.

#### 3. 어펜딕스
- 어펜딕스에는 모델의 응답 예시와 함께 각 단계에서의 성능을 비교하는 추가 데이터가 포함되어 있습니다. 이 데이터는 모델이 주장을 분석하고 증거를 평가하는 과정에서의 오류 유형을 진단하는 데 유용합니다.



#### 1. Diagrams and Figures
- **Figure 1**: This figure illustrates an example from the NLI4CT dataset, visually explaining the relationship between clinical trial data and claims. It emphasizes the complexity of NLI tasks, highlighting the need for domain knowledge and multi-step reasoning required by the model to validate claims.

- **Figure 3**: This figure represents the structure of the proposed 4-step prompting framework. Each step clearly outlines the process of understanding the claim, analyzing evidence, drawing intermediate conclusions, and finally making relational predictions. This structure helps LLMs focus on specific subtasks, reducing ambiguity and enhancing accuracy.

#### 2. Tables
- **Table 1**: This table shows the instance distribution for each relation class in the NLI4CT, SciFact, and HealthVer datasets. It provides the number of samples needed to understand the relationship between claims and evidence in each dataset.

- **Table 3**: This table compares the F1 scores of various models, demonstrating how the 4-step prompting framework enhances performance. It confirms that the 4-step approach leads to significant performance improvements over simpler 2-step and baseline prompt templates.

- **Table 4**: This table presents the results of supervised fine-tuning (SFT) for lightweight LLMs. It shows how SFT improves model performance, indicating that performance increases as the number of training samples grows.

- **Tables 10-15**: These tables compare the responses of different models, illustrating the performance differences between the 2-step and 4-step prompting frameworks. They provide examples of how each model interprets claims and analyzes evidence, highlighting how the 4-step approach contributes to more accurate conclusions.

#### 3. Appendix
- The appendix includes additional data comparing model responses and performance at each step. This data is useful for diagnosing error types in the models during the claim analysis and evidence evaluation processes.

<br/>
# refer format:
### BibTeX Citation

```bibtex
@inproceedings{Liang2025,
  author    = {Siting Liang and Daniel Sonntag},
  title     = {Advancing Biomedical Claim Verification by Using Large Language Models with Better Structured Prompting Strategies},
  booktitle = {Proceedings of the 24th Workshop on Biomedical Language Processing (BioNLP 2025)},
  pages     = {148--166},
  year      = {2025},
  month     = {August},
  publisher = {Association for Computational Linguistics},
  address   = {Germany}
}
```

### Chicago Style Citation

Liang, Siting, and Daniel Sonntag. "Advancing Biomedical Claim Verification by Using Large Language Models with Better Structured Prompting Strategies." In *Proceedings of the 24th Workshop on Biomedical Language Processing (BioNLP 2025)*, 148–166. Germany: Association for Computational Linguistics, August 1, 2025.
