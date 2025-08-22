---
layout: post
title:  "[2025]Conservative Bias in Large Language Models: Measuring Relation Predictions"
date:   2025-08-22 01:35:04 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

모델이 안전하지만 비정보적인 레이블을 선택하는 "Hobson's choice" 시나리오를 도입  

보수적인 편향이 환각(hallucination)보다 두 배 더 자주 발생한다는 것을 발견

이 효과를 정량화하기 위해 SBERT와 LLM 프롬프트를 사용하여 제약된 프롬프트에서의 보수적인 편향 행동과 반제약 및 개방형 프롬프트에서 생성된 레이블 간의 의미적 유사성을 포착



짧은 요약(Abstract) :



이 논문에서는 대형 언어 모델(LLM)이 관계 추출 작업에서 보수적인 편향(conservative bias)을 보이는 경향을 분석합니다. LLM은 적절한 옵션이 없을 때 자주 NO_RELATION 레이블을 선택하는데, 이는 잘못된 관계 할당을 방지하는 데 도움이 되지만, 동시에 출력에 명시적으로 포함되지 않은 추론으로 인해 상당한 정보 손실을 초래합니다. 저자들은 여러 프롬프트, 데이터셋 및 관계 유형을 통해 이 트레이드오프를 체계적으로 평가하며, 모델이 안전하지만 비정보적인 레이블을 선택하는 "Hobson's choice" 시나리오를 도입합니다. 연구 결과, 보수적인 편향이 환각(hallucination)보다 두 배 더 자주 발생한다는 것을 발견했습니다. 이 효과를 정량화하기 위해 SBERT와 LLM 프롬프트를 사용하여 제약된 프롬프트에서의 보수적인 편향 행동과 반제약 및 개방형 프롬프트에서 생성된 레이블 간의 의미적 유사성을 포착합니다.




This paper analyzes the tendency of large language models (LLMs) to exhibit conservative bias in relation extraction tasks. LLMs frequently default to the NO_RELATION label when an appropriate option is unavailable, which helps prevent incorrect relation assignments but also leads to significant information loss when reasoning is not explicitly included in the output. The authors systematically evaluate this trade-off across multiple prompts, datasets, and relation types, introducing the concept of Hobson's choice to capture scenarios where models opt for safe but uninformative labels over hallucinated ones. The findings reveal that conservative bias occurs twice as often as hallucination. To quantify this effect, SBERT and LLM prompts are used to capture the semantic similarity between conservative bias behaviors in constrained prompts and labels generated from semi-constrained and open-ended prompts.


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



이 연구에서는 대형 언어 모델(LLM)에서 보이는 보수적 편향(Conservative Bias, CB)을 분석하기 위해 여러 가지 방법론을 사용했습니다. 연구의 주요 목표는 LLM이 관계 추출 작업에서 NO_RELATION이라는 안전한 선택을 선호하는 경향을 정량화하고, 이 경향이 환각(hallucination) 방지와 어떤 관계가 있는지를 탐구하는 것이었습니다.

1. **모델 선택**: 연구에서는 GPT-4, Llama3.1-8B-Instruct, Mistral-7B-Instruct-v0.3와 같은 여러 LLM을 사용했습니다. 각 모델은 두 가지 온도 설정(0.2 및 0.5)을 사용하여 다양한 출력을 생성했습니다. 이러한 설정은 모델의 결정성을 높이고 여러 번의 반복을 통해 출력의 일관성을 평가하는 데 도움을 주었습니다.

2. **데이터셋**: 연구는 REFinD와 TACRED라는 두 개의 데이터셋을 사용했습니다. REFinD는 금융 관련 데이터로 구성되어 있으며, TACRED는 뉴스 및 웹 텍스트에서 파생된 대규모 관계 추출 데이터셋입니다. 두 데이터셋 모두 NO_RELATION 레이블이 포함된 인스턴스를 포함하고 있어 CB와 HR(환각 비율)을 분석하는 데 적합했습니다.

3. **프롬프트 설계**: 연구에서는 세 가지 유형의 프롬프트를 설계했습니다:
   - **제한된 프롬프트(Constrained Prompt)**: 모델이 미리 정의된 관계 클래스 중에서 선택하도록 요구합니다.
   - **반제한 프롬프트(Semi-Constrained Prompt)**: 모델이 선택할 수 있는 관계 목록을 제공하되, 적절한 옵션이 없을 경우 새로운 관계를 제안할 수 있도록 합니다.
   - **개방형 프롬프트(Open-ended Prompt)**: 모델이 주어진 데이터에 따라 가장 적합한 관계를 생성하도록 요구합니다.

4. **측정 지표**: 연구에서는 CB, HR, Hobson’s Choice Rate(HCR), New Relation Rate(NRR)와 같은 여러 지표를 사용하여 모델의 출력을 평가했습니다. CB는 모델이 NO_RELATION을 선택하는 빈도를 측정하고, HR은 모델이 잘못된 관계를 생성하는 빈도를 측정합니다.

5. **실험 및 결과**: 실험 결과, GPT-4는 제한된 프롬프트에서 낮은 HR(0.02-0.04%)를 보였지만, CB는 상대적으로 높은 빈도로 나타났습니다. 반면, Llama3.1은 높은 HR을 보였지만 CB는 낮았습니다. 이러한 결과는 CB와 HR 간의 상반된 관계를 나타내며, 모델의 정확성과 혁신 간의 균형을 고려해야 함을 시사합니다.

이 연구는 LLM의 보수적 편향을 정량화하고, 이를 통해 관계 추출 작업의 품질을 향상시킬 수 있는 방법을 제시합니다.

---




This study employed various methodologies to analyze the Conservative Bias (CB) exhibited by large language models (LLMs). The primary goal of the research was to quantify the tendency of LLMs to prefer the NO_RELATION label in relation extraction tasks and to explore the relationship between this tendency and hallucination prevention.

1. **Model Selection**: The study utilized several LLMs, including GPT-4, Llama3.1-8B-Instruct, and Mistral-7B-Instruct-v0.3. Each model was tested under two temperature settings (0.2 and 0.5) to generate diverse outputs. These settings helped enhance the determinism of the models and assess the consistency of outputs through multiple iterations.

2. **Datasets**: The research focused on two datasets: REFinD and TACRED. REFinD consists of financial-related data, while TACRED is a large-scale relation extraction dataset derived from news and web text. Both datasets include instances labeled as NO_RELATION, making them suitable for analyzing CB and Hallucination Rate (HR).

3. **Prompt Design**: The study designed three types of prompts:
   - **Constrained Prompt**: Requires the model to select from a predefined list of relation classes.
   - **Semi-Constrained Prompt**: Provides a list of relations but allows the model to propose a new relation if none of the options are deemed appropriate.
   - **Open-Ended Prompt**: Tasks the model with generating the most suitable relation based on the input data.

4. **Metrics**: The research employed several metrics, including Conservative Bias Rate (CBR), Hallucination Rate (HR), Hobson’s Choice Rate (HCR), and New Relation Rate (NRR) to evaluate the model outputs. CBR measures how often the model defaults to NO_RELATION, while HR quantifies how often the model generates incorrect relations.

5. **Experiments and Results**: The experimental results showed that GPT-4 exhibited a low HR (0.02-0.04%) under constrained prompts but had a relatively high frequency of CB. In contrast, Llama3.1 showed a higher HR but lower CB. These results indicate an inverse relationship between CB and HR, suggesting a trade-off between accuracy and innovation in model behavior.

This study quantifies the Conservative Bias in LLMs and proposes methods to leverage this understanding to improve the quality of relation extraction tasks.


<br/>
# Results



이 논문에서는 대형 언어 모델(LLM)의 보수적 편향(Conservative Bias, CB)과 환각(hallucination) 현상을 비교 분석하고, 이들이 관계 추출(relation extraction) 작업에 미치는 영향을 평가합니다. 연구는 두 가지 주요 데이터셋인 REFinD와 TACRED를 사용하여 진행되었습니다.

#### 1. 모델 성능
- **모델**: GPT-4, Llama3.1-8B-Instruct, Mistral-7B-Instruct-v0.3
- **데이터셋**: REFinD는 28,676개의 인스턴스와 22개의 관계 유형을 포함하고 있으며, TACRED는 106,264개의 예제를 포함하고 있습니다.
- **온도 설정**: 모델의 출력을 조절하기 위해 두 가지 온도 설정(0.2 및 0.5)을 사용했습니다.

#### 2. 메트릭
- **보수적 편향 비율(Conservative Bias Rate, CBR)**: 모델이 NO_RELATION을 선택한 비율을 측정합니다.
- **환각 비율(Hallucination Rate, HR)**: 모델이 제공된 옵션 외의 관계를 생성한 비율을 측정합니다.
- **새로운 관계 비율(New Relation Rate, NRR)**: 모델이 제공된 옵션에 없는 유효한 관계를 제안한 비율을 측정합니다.
- **Hobson’s Choice 비율(Hobson’s Choice Rate, HCR)**: 모델이 적절한 옵션이 없을 때 NO_RELATION을 선택한 비율을 측정합니다.

#### 3. 결과 요약
- **GPT-4**: REFinD 데이터셋에서 CBR은 1.14%에서 1.33%로 나타났고, HR은 0.02%에서 0.06%로 낮았습니다. TACRED 데이터셋에서는 CBR이 7.99%에서 7.11%로 나타났고, HR은 15.47%에서 13.87%로 상대적으로 높았습니다.
- **Llama3.1**: REFinD에서 CBR은 0.29%에서 1.07%로 증가했으며, HR은 8.18%에서 4.67%로 감소했습니다. TACRED에서는 CBR이 10.80%에서 4.00%로 감소했지만, HR은 9.60%에서 2.60%로 감소했습니다.
- **Mistral**: REFinD에서 CBR은 2.51%에서 19.15%로 증가했으며, HR은 3.84%에서 2.97%로 감소했습니다. TACRED에서는 CBR이 13.01%에서 15.57%로 증가했습니다.

#### 4. 결론
연구 결과, GPT-4는 환각 저항력이 뛰어나지만 보수적 편향이 더 높다는 것을 보여주었습니다. Llama3.1은 보수적 편향이 낮지만 환각 비율이 높았습니다. 이는 모델의 정확성과 혁신성 간의 상충 관계를 나타냅니다. 향후 연구는 보수적 편향과 새로운 관계 식별 간의 균형을 맞추는 방향으로 진행될 필요가 있습니다.

---




This paper analyzes the Conservative Bias (CB) and hallucination phenomena in large language models (LLMs) and evaluates their impact on relation extraction tasks. The study is conducted using two primary datasets: REFinD and TACRED.

#### 1. Model Performance
- **Models**: GPT-4, Llama3.1-8B-Instruct, Mistral-7B-Instruct-v0.3
- **Datasets**: REFinD consists of 28,676 instances and 22 relation types, while TACRED includes 106,264 examples.
- **Temperature Settings**: Two temperature settings (0.2 and 0.5) were used to control the model outputs.

#### 2. Metrics
- **Conservative Bias Rate (CBR)**: Measures the rate at which the model selects NO_RELATION.
- **Hallucination Rate (HR)**: Measures the rate at which the model generates relations outside the provided options.
- **New Relation Rate (NRR)**: Measures the rate at which the model proposes valid relations not present in the provided options.
- **Hobson’s Choice Rate (HCR)**: Measures the rate at which the model selects NO_RELATION when no suitable option is available.

#### 3. Summary of Results
- **GPT-4**: In the REFinD dataset, CBR ranged from 1.14% to 1.33%, while HR was low at 0.02% to 0.06%. In the TACRED dataset, CBR was 7.99% to 7.11%, with HR at 15.47% to 13.87%.
- **Llama3.1**: In REFinD, CBR increased from 0.29% to 1.07%, and HR decreased from 8.18% to 4.67%. In TACRED, CBR decreased from 10.80% to 4.00%, while HR decreased from 9.60% to 2.60%.
- **Mistral**: In REFinD, CBR increased from 2.51% to 19.15%, and HR decreased from 3.84% to 2.97%. In TACRED, CBR increased from 13.01% to 15.57%.

#### 4. Conclusion
The findings indicate that while GPT-4 demonstrates strong resistance to hallucinations, it exhibits a higher frequency of conservative bias. In contrast, Llama3.1 shows lower conservative bias but higher hallucination rates. This reflects a trade-off between accuracy and innovation in model behavior. Future research should focus on balancing conservative bias with the need for novel relation identification.


<br/>
# 예제



이 논문에서는 대형 언어 모델(LLM)의 관계 추출 작업에서 나타나는 보수적 편향(Conservative Bias, CB)을 분석하고 있습니다. 연구의 주요 목표는 LLM이 주어진 옵션 중에서 가장 안전한 선택인 "NO_RELATION"으로 기본값을 설정하는 경향을 이해하고, 이로 인해 발생하는 정보 손실을 측정하는 것입니다.

#### 예시 설명

1. **트레이닝 데이터**: 
   - 데이터셋: REFinD
   - 예시 문장: "Apple Inc. is the owner of Beats Electronics."
   - 주어진 엔티티: 
     - 주체(subject): "Apple Inc."
     - 객체(object): "Beats Electronics"
   - 가능한 관계 옵션: ["OWNER_OF", "SHAREHOLDER_OF", "NO_RELATION"]

2. **테스트 데이터**:
   - 예시 문장: "Apple Inc. has a partnership with Beats Electronics."
   - 주어진 엔티티:
     - 주체: "Apple Inc."
     - 객체: "Beats Electronics"
   - 가능한 관계 옵션: ["PARTNERSHIP", "NO_RELATION"]

3. **모델의 출력**:
   - 모델이 "Apple Inc. has a partnership with Beats Electronics."라는 문장을 처리할 때, 주어진 옵션 중 "PARTNERSHIP"이 없기 때문에 모델은 "NO_RELATION"을 선택할 수 있습니다. 
   - 그러나 모델은 "PARTNERSHIP"이라는 관계가 존재함을 인식하고, 그에 대한 이유를 제시할 수 있지만, 최종적으로는 "NO_RELATION"을 선택하게 됩니다. 이는 보수적 편향의 예시입니다.

4. **결과 분석**:
   - 모델의 출력에서 "NO_RELATION"을 선택한 비율(Hobson’s Choice Rate, HCR)과 보수적 편향 비율(Conservative Bias Rate, CBR)을 측정합니다.
   - 예를 들어, REFinD 데이터셋에서 모델이 "NO_RELATION"을 선택한 비율이 57.72%이고, 보수적 편향이 1.14%로 나타날 수 있습니다.

이러한 방식으로 연구는 LLM의 보수적 편향이 정보 손실을 초래하는 방식과 그 빈도를 정량적으로 평가하고 있습니다.

---




This paper analyzes the Conservative Bias (CB) exhibited by large language models (LLMs) in relation extraction tasks. The primary goal of the research is to understand the tendency of LLMs to default to the safest choice, "NO_RELATION," from the given options, and to measure the information loss that results from this behavior.

#### Example Explanation

1. **Training Data**:
   - Dataset: REFinD
   - Example Sentence: "Apple Inc. is the owner of Beats Electronics."
   - Given Entities: 
     - Subject: "Apple Inc."
     - Object: "Beats Electronics"
   - Possible Relation Options: ["OWNER_OF", "SHAREHOLDER_OF", "NO_RELATION"]

2. **Test Data**:
   - Example Sentence: "Apple Inc. has a partnership with Beats Electronics."
   - Given Entities:
     - Subject: "Apple Inc."
     - Object: "Beats Electronics"
   - Possible Relation Options: ["PARTNERSHIP", "NO_RELATION"]

3. **Model Output**:
   - When the model processes the sentence "Apple Inc. has a partnership with Beats Electronics," it may choose "NO_RELATION" because "PARTNERSHIP" is not among the provided options.
   - However, the model may recognize that a "PARTNERSHIP" relation exists and can provide reasoning for it, but ultimately defaults to "NO_RELATION." This is an example of Conservative Bias.

4. **Result Analysis**:
   - The study measures the rate at which the model selects "NO_RELATION" (Hobson’s Choice Rate, HCR) and the rate of Conservative Bias (CBR).
   - For instance, in the REFinD dataset, the model might show a HCR of 57.72% and a CBR of 1.14%.

In this way, the research quantitatively evaluates how the Conservative Bias in LLMs leads to information loss and the frequency of this behavior.

<br/>
# 요약


이 논문에서는 대형 언어 모델(LLM)의 보수적 편향(Conservative Bias, CB)을 측정하기 위해 다양한 프롬프트와 데이터셋을 사용하여 관계 추출 작업을 분석하였다. 실험 결과, LLM은 적절한 관계가 없을 때 'NO_RELATION'으로 기본값을 설정하는 경향이 있으며, 이는 정보 손실을 초래하는 것으로 나타났다. 예를 들어, 모델은 'Hobson’s choice' 상황에서 더 적합한 관계가 제시되더라도 안전한 선택을 선호하는 경향을 보였다.

---

This paper analyzes the Conservative Bias (CB) in large language models (LLMs) during relation extraction tasks using various prompts and datasets. The results show that LLMs tend to default to 'NO_RELATION' when an appropriate relation is unavailable, leading to significant information loss. For instance, the models exhibit a preference for safe choices in 'Hobson’s choice' scenarios, even when more suitable relations are suggested.

<br/>
# 기타



#### 다이어그램 및 피규어
1. **Figure 1**: LLM의 출력 예시를 보여주며, "Hobson’s Choice"와 보수적 편향(Conservative Bias, CB) 행동을 설명합니다. 모델이 적절한 관계가 없을 때 "NO_RELATION"을 선택하는 경향을 보여줍니다. 이 예시는 모델이 더 정확한 관계(예: OWNER_OF 또는 SHAREHOLDER_OF)를 제안할 수 있지만, 안전한 선택을 위해 이를 무시하는 모습을 나타냅니다.

2. **Figure 2**: 프롬프트 설계의 프로세스 워크플로우를 보여줍니다. 다양한 제약 수준을 가진 프롬프트가 LLM의 관계 생성 및 선택 능력에 미치는 영향을 탐구합니다.

#### 테이블
1. **Table 1**: 다양한 프롬프트 유형에 따른 LLM 출력 성능을 요약합니다. GPT-4는 REFinD 데이터셋에서 낮은 환각률(HR)과 높은 보수적 편향률(CBR)을 보였습니다. 반면, Llama3.1은 더 높은 HR을 보였지만 낮은 CBR을 나타냈습니다. 이는 모델의 보수적 경향과 환각 저항 간의 상반된 관계를 보여줍니다.

2. **Table 2**: REFinD 및 TACRED 데이터셋에서 GPT-4 출력의 의미적 유사성 점수를 보여줍니다. REFinD 데이터셋에서 보수적 편향이 포착된 경우, 반제약 프롬프트의 출력과 의미적으로 유사한 비율이 높았습니다. 이는 보수적 편향 레이블이 관계 추출 작업을 개선하는 데 유용할 수 있음을 시사합니다.

#### 어펜딕스
1. **A.1**: 의미적 유사성 점수에 대한 추가 정보를 제공합니다. REFinD 데이터셋에서 보수적 편향이 포착된 경우, 반제약 프롬프트의 출력과의 유사성이 62%에 달했습니다.

2. **A.4**: 환각과 보수적 편향 간의 트레이드오프를 설명합니다. 환각은 제공된 옵션 외부에서 새로운 관계를 생성하는 반면, 보수적 편향은 주어진 옵션에서 가장 덜 잘못된 선택을 하며, 더 적절한 관계를 제안하는 경향이 있습니다.




#### Diagrams and Figures
1. **Figure 1**: This figure illustrates an example of LLM output, demonstrating "Hobson’s Choice" and Conservative Bias (CB) behavior. It shows the model's tendency to default to "NO_RELATION" when appropriate relations are not available. The example highlights that the model could suggest more accurate relations (e.g., OWNER_OF or SHAREHOLDER_OF) but opts for a safer choice, leading to information loss.

2. **Figure 2**: This figure presents the process workflow of prompt design. It explores how different levels of constraints in prompts affect the LLM's ability to generate and select appropriate relations.

#### Tables
1. **Table 1**: This table summarizes the performance of LLM outputs across different prompt types. GPT-4 exhibited low hallucination rates (HR) and high Conservative Bias Rates (CBR) on the REFinD dataset. In contrast, Llama3.1 showed higher HR but lower CBR, indicating a trade-off between conservative tendencies and hallucination resistance.

2. **Table 2**: This table displays semantic similarity scores for GPT-4 outputs on the REFinD and TACRED datasets. It indicates that when Conservative Bias is captured, the outputs from semi-constrained prompts show a high degree of semantic similarity, suggesting that CB labels could be beneficial for improving relation extraction tasks.

#### Appendix
1. **A.1**: Provides additional information on semantic similarity scores. It shows that when Conservative Bias is captured in the REFinD dataset, the similarity with outputs from semi-constrained prompts reaches 62%.

2. **A.4**: Discusses the trade-off between hallucination and Conservative Bias. Hallucination involves generating new relations outside the provided options, while Conservative Bias reflects the model's tendency to select the least incorrect option while suggesting a more appropriate relation in reasoning.

<br/>
# refer format:
### BibTeX Citation

```bibtex
@inproceedings{Aguda2025,
  author    = {Toyin Aguda and Erik Wilson and Allan Anzagira and Simerjot Kaur and Charese Smiley},
  title     = {Conservative Bias in Large Language Models: Measuring Relation Predictions},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2025},
  pages     = {18989--18998},
  year      = {2025},
  month     = {July},
  publisher = {Association for Computational Linguistics},
  address   = {Bangkok, Thailand},
}
```

### Chicago Style Citation

Aguda, Toyin, Erik Wilson, Allan Anzagira, Simerjot Kaur, and Charese Smiley. "Conservative Bias in Large Language Models: Measuring Relation Predictions." In *Findings of the Association for Computational Linguistics: ACL 2025*, 18989–18998. Bangkok, Thailand: Association for Computational Linguistics, July 2025.
