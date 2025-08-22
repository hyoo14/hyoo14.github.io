---
layout: post
title:  "[2025]LoGU: Long-form Generation with Uncertainty Expressions"
date:   2025-08-22 02:04:03 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 LLMs가 긴 형식의 응답에서 불확실성을 표현하도록 돕기 위해 LoGU(불확실성을 명시적으로 표현하는 긴 형식의 생성 작업)라는 새로운 과제를 소개하고, 이를 해결하기 위해 데이터 수집 프레임워크(Fact-Checking, Revision )와 두 단계의 학습 파이프라인(불확실성 표현의 정확성을 높이기 위해, 정확한 불확실성 표현과 부정확한 불확실성 표현을 포함하는 응답 쌍을 사용하여 모델, supervised fine tuning)을 제안


짧은 요약(Abstract) :


이 논문은 대형 언어 모델(LLM)이 긴 형식의 텍스트를 생성할 때 불확실성을 표현하는 방법을 제안합니다. LLM은 종종 사실과 다른 정보를 생성하는 문제를 가지고 있으며, 이를 완화하기 위해 모델이 확실하지 않을 때 불확실성을 표현하도록 하는 것이 중요합니다. 기존 연구는 주로 짧은 형식의 질문 응답(QA)에 초점을 맞췄지만, 실제 응용에서는 더 긴 응답이 필요합니다. 이 연구에서는 불확실성을 명시적으로 표현하는 긴 형식의 생성 작업(LoGU)을 소개하고, 두 가지 주요 문제인 불확실성 억제와 불확실성 불일치를 해결하기 위해 새로운 데이터 수집 프레임워크와 두 단계의 학습 파이프라인을 제안합니다. 실험 결과, 제안된 방법이 사실적 정확성을 개선하고 잘못된 진술을 줄이며, 생성된 응답의 전반적인 포괄성을 유지하는 데 효과적임을 보여줍니다.



This paper introduces a method for enabling large language models (LLMs) to express uncertainty in long-form text generation. LLMs often struggle with generating factually incorrect statements, and a promising approach to mitigate this is to allow models to express uncertainty when unsure. Previous research has primarily focused on short-form question answering (QA), but real-world applications often require much longer responses. In this work, we introduce the task of Long-form Generation with Uncertainty (LoGU), which requires models to explicitly express uncertainty during generation. We identify two key challenges: Uncertainty Suppression, where models hesitate to express uncertainty, and Uncertainty Misalignment, where models convey uncertainty inaccurately. To tackle these challenges, we propose a novel decomposition-based data collection framework and a two-stage training pipeline. Experiments on three long-form datasets demonstrate the effectiveness of our approach, showing improvements in factual accuracy, reduction of incorrect statements, and preservation of the overall comprehensiveness of the generated responses.


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



이 논문에서는 LLM(대형 언어 모델)이 긴 형식의 텍스트를 생성할 때 불확실성을 표현할 수 있도록 하는 새로운 과제인 LoGU(Long-form Generation with Uncertainty)를 소개합니다. 이 과제는 모델이 잘 모르는 정보에 대해 불확실성을 명시적으로 표현함으로써 잘못된 정보(즉, 환각)를 줄이는 것을 목표로 합니다. 이를 위해 두 가지 주요 문제를 해결하고자 합니다: 불확실성 억제(Uncertainty Suppression)와 불확실성 불일치(Uncertainty Misalignment)입니다.

1. **불확실성 억제(Uncertainty Suppression)**: 모델이 불확실성을 표현하는 것을 꺼리는 경향이 있습니다. 이는 모델이 과도한 자신감을 가지게 하여 사실적 정확성을 감소시킬 수 있습니다.

2. **불확실성 불일치(Uncertainty Misalignment)**: 모델이 실제로 알고 있는 정보에 대해서도 불확실성을 표현하는 경우가 있습니다. 이는 모델의 지식과 불확실성 표현이 일치하지 않는 문제를 야기합니다.

이 문제들을 해결하기 위해, 저자들은 새로운 데이터 수집 프레임워크와 두 단계의 학습 파이프라인을 제안합니다.

#### 데이터 수집 프레임워크

1. **사실 확인(Fact-Checking)**: 모델의 응답을 원자적 주장으로 분해하고, 외부 지식 소스를 통해 각 주장의 진위를 확인합니다. 이를 통해 지지되는 주장(Cs)과 반박되는 주장(Cns)으로 분류합니다.

2. **수정(Revision)**: 반박된 주장(Cns)을 불확실성 표현으로 수정하고, 지지된 주장(Cs)은 그대로 둡니다. 이 과정에서 불확실성 표현의 비율을 조절하여 과도한 불확실성 표현을 방지합니다.

3. **조립(Assembling)**: 수정된 원자적 주장들을 하나의 일관된 응답으로 조립합니다.

#### 두 단계 학습 파이프라인

1. **LOGU-SFT (Supervised Fine-Tuning)**: 모델이 불확실성을 표현하도록 유도하여 불확실성 억제 문제를 해결합니다. 이를 위해 긍정적으로 수정된 응답(Rpos)만을 사용하여 모델을 미세 조정합니다.

2. **LOGU-DPO (Direct Preference Optimization)**: 불확실성 표현의 정확성을 높이기 위해, 정확한 불확실성 표현과 부정확한 불확실성 표현을 포함하는 응답 쌍을 사용하여 모델을 학습시킵니다. 이를 통해 불확실성 불일치 문제를 해결합니다.

이 방법론은 세 가지 긴 형식의 데이터셋에서 실험을 통해 그 효과가 입증되었습니다. 이 접근법은 사실적 정확성을 향상시키고 잘못된 진술을 줄이며, 생성된 응답의 전반적인 포괄성을 유지하는 데 효과적임을 보여줍니다.

---




This paper introduces a new task called LoGU (Long-form Generation with Uncertainty), which aims to enable large language models (LLMs) to express uncertainty in long-form text generation. The goal is to reduce hallucinations by explicitly expressing uncertainty when the model is unsure about certain information. To achieve this, the authors address two key challenges: Uncertainty Suppression and Uncertainty Misalignment.

1. **Uncertainty Suppression**: Models tend to hesitate to express uncertainty, which can lead to overconfidence and reduced factual accuracy.

2. **Uncertainty Misalignment**: There are cases where models express uncertainty inaccurately, even when they have sufficient knowledge about a fact.

To tackle these challenges, the authors propose a novel data collection framework and a two-stage training pipeline.

#### Data Collection Framework

1. **Fact-Checking**: The model's responses are decomposed into atomic claims, and the veracity of each claim is checked against external knowledge sources. Claims are categorized into supported claims (Cs) and refuted claims (Cns).

2. **Revision**: Refuted claims (Cns) are revised into uncertainty expressions, while supported claims (Cs) remain unchanged. The ratio of uncertainty expressions is controlled to prevent excessive uncertainty.

3. **Assembling**: The revised atomic claims are assembled into a coherent response.

#### Two-Stage Training Pipeline

1. **LOGU-SFT (Supervised Fine-Tuning)**: This stage encourages the model to express uncertainty, addressing the uncertainty suppression problem. The model is fine-tuned using only positively revised responses (Rpos).

2. **LOGU-DPO (Direct Preference Optimization)**: This stage improves the accuracy of uncertainty expressions by training the model with response pairs that include both accurate and inaccurate uncertainty expressions, addressing the uncertainty misalignment problem.

The methodology is validated through experiments on three long-form datasets, demonstrating its effectiveness in improving factual accuracy, reducing incorrect statements, and preserving the overall comprehensiveness of the generated responses.


<br/>
# Results



이 연구에서는 Long-form Generation with Uncertainty (LoGU)라는 새로운 과제를 소개하고, 이를 해결하기 위한 두 가지 주요 문제인 불확실성 억제(Uncertainty Suppression)와 불확실성 불일치(Uncertainty Misalignment)를 해결하기 위해 새로운 데이터 수집 프레임워크와 두 단계의 학습 파이프라인을 제안했습니다. 이 연구는 Mistral-7B-Instruct와 Llama3-8B-Instruct 모델을 사용하여 세 가지 데이터셋(Bios, LongFact, WildHallu)에서 실험을 수행했습니다. 

#### 경쟁 모델 및 비교

1. **경쟁 모델**: 
   - **Unc-Zero**: 모델이 불확실한 정보에 대해 불확실성을 표현하도록 직접 프롬프트를 제공.
   - **Unc-Few**: Unc-Zero에 추가로 불확실성을 포함한 10개의 QA 예제를 제공하여 학습.
   - **Pair-Few**: Unc-Few에 추가로 불확실성을 포함한 답변과 그렇지 않은 답변을 쌍으로 제공.
   - **Self-Refine**: 초기 응답을 생성한 후, 두 번째 단계에서 불확실한 사실을 명시적으로 표현하도록 수정.

2. **비교 결과**:
   - **LoGU-DPO**는 모든 데이터셋에서 가장 높은 성능을 보였습니다. 예를 들어, Mistral 모델의 경우 Bios 데이터셋에서 정확도가 38.8%에서 65.4%로 향상되었습니다.
   - **LoGU-SFT**는 불확실성 억제 문제를 해결하는 데 효과적이었으나, 불확실성 표현의 정확성은 LoGU-DPO에 비해 낮았습니다.
   - **Prompt-Based 방법**들은 불확실성을 어느 정도 표현할 수 있었으나, 정확성 면에서는 LoGU-DPO에 미치지 못했습니다.

#### 테스트 데이터 및 메트릭

- **테스트 데이터**: Bios, LongFact, WildHallu 세 가지 데이터셋을 사용하였으며, 추가적으로 ASQA 데이터셋을 사용하여 모델의 일반화 성능을 평가했습니다.
- **메트릭**:
  - **Factual Accuracy (FA)**: 생성된 응답의 사실적 정확성을 측정.
  - **Uncertain Accuracy (UA)**: 불확실성 표현의 정확성을 측정.
  - **#Incor**: 생성된 응답에서의 잘못된 진술의 수를 측정.

#### 결론

LoGU-DPO는 불확실성을 정확하게 표현하고, 잘못된 진술을 줄이며, 전반적인 사실적 정확성을 향상시키는 데 효과적임을 입증했습니다. 이는 모델이 불확실성을 보다 구체적이고 타겟화된 방식으로 표현할 수 있도록 하여, 실제 응용에서 더 신뢰할 수 있는 결과를 제공합니다.

---




### Summary of Results

This study introduces a new task called Long-form Generation with Uncertainty (LoGU) and proposes a novel data collection framework and a two-stage training pipeline to address two main challenges: Uncertainty Suppression and Uncertainty Misalignment. The experiments were conducted using Mistral-7B-Instruct and Llama3-8B-Instruct models on three datasets: Bios, LongFact, and WildHallu.

#### Competing Models and Comparison

1. **Competing Models**:
   - **Unc-Zero**: Directly prompts the model to express uncertainty for any information it is unsure about.
   - **Unc-Few**: Builds on Unc-Zero by providing 10 additional QA examples with explicit uncertainty expressions.
   - **Pair-Few**: Extends Unc-Few by providing paired answers with and without uncertainty expressions.
   - **Self-Refine**: Uses a draft-and-refine approach where the model first generates an initial response and then refines it to express uncertainty.

2. **Comparison Results**:
   - **LoGU-DPO** consistently achieved the best performance across all datasets. For instance, it improved accuracy from 38.8% to 65.4% on the Bios dataset for the Mistral model.
   - **LoGU-SFT** effectively addressed the uncertainty suppression issue but had lower accuracy in uncertainty expression compared to LoGU-DPO.
   - **Prompt-Based methods** were able to express some uncertainty but did not match the accuracy of LoGU-DPO.

#### Test Data and Metrics

- **Test Data**: The study used three datasets: Bios, LongFact, and WildHallu, and additionally evaluated the model's generalization performance using the ASQA dataset.
- **Metrics**:
  - **Factual Accuracy (FA)**: Measures the factual accuracy of the generated responses.
  - **Uncertain Accuracy (UA)**: Measures the accuracy of uncertainty expressions.
  - **#Incor**: Measures the number of incorrect statements in the generated responses.

#### Conclusion

LoGU-DPO proved effective in accurately expressing uncertainty, reducing incorrect statements, and improving overall factual accuracy. This allows the model to express uncertainty in a more specific and targeted manner, providing more reliable results in real-world applications.


<br/>
# 예제

논문 "LoGU: Long-form Generation with Uncertainty Expressions"에서는 대형 언어 모델(LLM)이 긴 형식의 텍스트를 생성할 때 불확실성을 표현하도록 하는 새로운 과제인 LoGU(Long-form Generation with Uncertainty)를 소개합니다. 이 과제는 모델이 잘 모르는 정보에 대해 불확실성을 명시적으로 표현하도록 요구합니다. 이를 통해 모델의 환각(hallucination) 문제를 완화하고, 생성된 응답의 사실적 정확성을 높이는 것을 목표로 합니다.

### 예시 설명

#### 트레이닝 데이터
1. **입력 (Input)**: 
   - 질문: "Annabel Pitcher에 대해 알고 있는 것을 말해줄 수 있나요?"
   - 원본 응답: "Annabel Pitcher는 영국의 작가입니다. 그녀는 1974년 8월 15일 런던에서 태어났습니다."

2. **출력 (Output)**:
   - 수정된 응답: "Annabel Pitcher는 영국의 작가입니다. 그녀는 런던에서 태어났지만, 언제 태어났는지는 확실하지 않습니다."

#### 테스트 데이터
1. **입력 (Input)**:
   - 질문: "Aegon the Conqueror에 대해 알고 있는 것을 말해줄 수 있나요?"

2. **출력 (Output)**:
   - 원본 응답: "Aegon Targaryen, 일명 Aegon the Conqueror는 Targaryen 왕조의 첫 번째 왕입니다. 그는 27 BC에 태어났습니다."
   - 수정된 응답: "Aegon Targaryen, 일명 Aegon the Conqueror는 Targaryen 왕조의 첫 번째 왕입니다. 그는 27 BC에 태어났지만, 정확한 날짜는 확실하지 않습니다."

### 구체적인 테스크
1. **불확실성 억제 (Uncertainty Suppression)**: 모델이 불확실성을 표현하는 것을 주저하는 문제를 해결합니다. 이를 위해, 모델이 불확실성을 표현하도록 지도 학습(Supervised Fine-Tuning, SFT)을 사용합니다.

2. **불확실성 불일치 (Uncertainty Misalignment)**: 모델이 실제로 알고 있는 정보에 대해 부정확하게 불확실성을 표현하는 문제를 해결합니다. 이를 위해, 직접 선호 최적화(Direct Preference Optimization, DPO)를 사용하여 불확실성 표현의 정확성을 높입니다.




In the paper "LoGU: Long-form Generation with Uncertainty Expressions," the authors introduce a new task called LoGU (Long-form Generation with Uncertainty), which requires large language models (LLMs) to explicitly express uncertainty when generating long-form text. This task aims to mitigate the hallucination problem of models and improve the factual accuracy of generated responses.

### Example Explanation

#### Training Data
1. **Input**:
   - Question: "Could you tell me what you know about Annabel Pitcher?"
   - Original Response: "Annabel Pitcher is a British author. She was born on August 15, 1974, in London."

2. **Output**:
   - Revised Response: "Annabel Pitcher is a British author. She was born in London, but I am not sure when she was born."

#### Test Data
1. **Input**:
   - Question: "Could you tell me what you know about Aegon the Conqueror?"

2. **Output**:
   - Original Response: "Aegon Targaryen, also known as Aegon the Conqueror, was the first king of the Targaryen dynasty. He was born in 27 BC."
   - Revised Response: "Aegon Targaryen, also known as Aegon the Conqueror, was the first king of the Targaryen dynasty. He was born in 27 BC, but the exact date is uncertain."

### Specific Tasks
1. **Uncertainty Suppression**: Address the issue where models hesitate to express uncertainty. Supervised Fine-Tuning (SFT) is used to encourage models to express uncertainty.

2. **Uncertainty Misalignment**: Address the issue where models inaccurately express uncertainty about information they actually know. Direct Preference Optimization (DPO) is used to improve the accuracy of uncertainty expressions.

<br/>
# 요약

이 논문에서는 LLMs가 긴 형식의 응답에서 불확실성을 표현하도록 돕기 위해 LoGU라는 새로운 과제를 소개하고, 이를 해결하기 위해 데이터 수집 프레임워크와 두 단계의 학습 파이프라인을 제안합니다. 실험 결과, 제안된 방법이 사실적 정확성을 향상시키고 잘못된 진술을 줄이며, 생성된 응답의 포괄성을 유지하는 데 효과적임을 보여줍니다. 예시로는, LoGU-DPO가 원래 모델과 비교하여 더 정확한 불확실성 표현을 생성하는 방법을 설명합니다.

In this paper, a new task called LoGU is introduced to help LLMs express uncertainty in long-form responses, and a data collection framework and two-stage training pipeline are proposed to address this. Experimental results demonstrate that the proposed method effectively improves factual accuracy, reduces incorrect statements, and preserves the comprehensiveness of generated responses. An example illustrates how LoGU-DPO generates more accurate uncertainty expressions compared to the original model.

<br/>
# 기타



1. **Figure 1**: 이 다이어그램은 짧은 형태의 질문 응답(QA)과 긴 형태의 QA의 차이를 보여줍니다. 짧은 형태의 QA는 단일 주장에 대한 불확실성을 표현하는 반면, 긴 형태의 QA는 여러 주장을 포함하고 있어 더 세밀한 불확실성 표현이 필요함을 강조합니다.

2. **Figure 2**: 이 그림은 LoGU(긴 형태의 생성에서 불확실성 표현) 작업을 위한 데이터 수집 및 훈련 파이프라인을 설명합니다. 데이터 수집은 응답을 원자적 주장으로 분해하고, 선택적으로 불확실성 표현을 포함하도록 수정한 후, 이를 다시 조립하는 과정을 포함합니다. 두 단계의 훈련 파이프라인은 불확실성 억제와 불확실성 불일치 문제를 해결하는 데 중점을 둡니다.

3. **Table 1**: 이 테이블은 다양한 방법론의 성능을 비교합니다. LoGU-DPO가 모든 데이터셋에서 가장 높은 정확도와 불확실성 정확도를 달성했음을 보여줍니다. 이는 이 방법이 환각을 줄이고 전반적인 사실성을 향상시키는 데 효과적임을 나타냅니다.

4. **Table 2**: 이 테이블은 도메인 외 데이터셋(ASQA)에서의 성능을 보여줍니다. LoGU-DPO는 도메인 외 데이터셋에서도 우수한 성능을 발휘하여 일반화 가능성을 입증합니다.

5. **Table 3**: 이 테이블은 다른 소스 모델에서 수집된 훈련 데이터를 사용한 경우의 성능을 비교합니다. 특히, DPO 단계에서는 모델의 자체 생성 데이터를 사용하는 것이 중요함을 강조합니다.

6. **Figure 3**: 이 다이어그램은 불확실성 표현의 카테고리를 분석한 결과를 보여줍니다. 훈련 기반 방법, 특히 LoGU-DPO는 불확실성을 더 구체적으로 표현하여 중요한 세부 사항에 대한 오류를 줄이는 데 효과적임을 나타냅니다.

7. **Table 4**: 이 테이블은 원래 모델의 생성과 LoGU-SFT 및 LoGU-DPO의 출력을 비교한 통계치를 보여줍니다. LoGU-DPO는 잘못된 주장의 수를 크게 줄이면서 정보의 총량을 유지합니다.

8. **Figure 4**: 이 그림은 다양한 방법으로 생성된 응답의 유용성과 유창성에 대한 인간 평가 결과를 보여줍니다. LoGU-DPO는 유용성에서 가장 높은 점수를 받았으며, 적절한 불확실성 표현이 사용자 경험을 향상시킴을 나타냅니다.

9. **Appendix**: 부록은 실험의 구현 세부 사항, 사용된 프롬프트, 추가 실험 결과 등을 제공합니다. 이는 연구의 재현 가능성을 높이고, 연구 방법론에 대한 깊은 이해를 돕습니다.

---






1. **Figure 1**: This diagram illustrates the difference between short-form question answering (QA) and long-form QA. Short-form QA expresses uncertainty for a single claim, whereas long-form QA involves multiple claims, requiring more granular uncertainty expressions.

2. **Figure 2**: This figure describes the data collection and training pipeline for the LoGU (Long-form Generation with Uncertainty) task. Data collection involves decomposing responses into atomic claims, selectively revising them to include uncertainty expressions, and reassembling them. The two-stage training pipeline focuses on addressing uncertainty suppression and misalignment.

3. **Table 1**: This table compares the performance of various methodologies. It shows that LoGU-DPO achieves the highest accuracy and uncertainty accuracy across all datasets, indicating its effectiveness in reducing hallucinations and improving overall factuality.

4. **Table 2**: This table shows performance on an out-of-domain dataset (ASQA). LoGU-DPO demonstrates excellent performance on out-of-domain datasets, proving its generalizability.

5. **Table 3**: This table compares performance when training data collected from different source models is used. It emphasizes the importance of using the model's own generations during the DPO stage.

6. **Figure 3**: This diagram shows the analysis of uncertainty expression categories. Training-based methods, especially LoGU-DPO, express uncertainty more specifically, effectively reducing errors in critical details.

7. **Table 4**: This table shows statistics comparing the original model's generation with the outputs of LoGU-SFT and LoGU-DPO. LoGU-DPO significantly reduces the number of incorrect claims while maintaining the total amount of information.

8. **Figure 4**: This figure shows human evaluation results on the helpfulness and fluency of responses generated using different methods. LoGU-DPO scores highest in helpfulness, indicating that appropriate uncertainty expressions enhance user experience.

9. **Appendix**: The appendix provides implementation details of the experiments, prompts used, additional experimental results, etc. This enhances the reproducibility of the research and aids in a deeper understanding of the research methodology.

<br/>
# refer format:



**BibTeX 형식:**
```bibtex
@inproceedings{yang2025logu,
  title={LoGU: Long-form Generation with Uncertainty Expressions},
  author={Yang, Ruihan and Zhang, Caiqi and Zhang, Zhisong and Huang, Xinting and Yang, Sen and Collier, Nigel and Yu, Dong and Yang, Deqing},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={18947--18968},
  year={2025},
  organization={Association for Computational Linguistics}
}
```

**시카고 스타일:**
Yang, Ruihan, Caiqi Zhang, Zhisong Zhang, Xinting Huang, Sen Yang, Nigel Collier, Dong Yu, and Deqing Yang. 2025. "LoGU: Long-form Generation with Uncertainty Expressions." In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 18947–18968. Association for Computational Linguistics.
