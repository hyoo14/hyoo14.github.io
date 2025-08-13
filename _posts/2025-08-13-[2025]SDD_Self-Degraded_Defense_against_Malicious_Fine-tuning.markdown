---
layout: post
title:  "[2025]SDD: Self-Degraded Defense against Malicious Fine-tuning"
date:   2025-08-13 17:24:34 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

자기 저하 방어(Self-Degraded Defense, SDD) 프레임워크를 소개  
유해한 프롬프트에 대해 고품질이지만 관련 없는 응답을 생성하도록 유도합니다. 공격자가 악의적인 미세 조정을 시도할 때, SDD에 의해 정렬된 LLM의 일반적인 능력은 크게 감소하여 유해한 지시를 따를 수 없게 됨  


짧은 요약(Abstract) :

이 논문에서는 오픈 소스 대형 언어 모델(LLM)이 유해한 지시를 저항하기 위해 안전 정렬 방법을 사용하는 것에 대해 다룹니다. 그러나 최근 연구에 따르면, 악의적으로 유해한 데이터로 LLM을 미세 조정하는 것이 이러한 안전 장치를 쉽게 우회할 수 있음을 보여줍니다. 이를 해결하기 위해, 저자들은 악의적인 미세 조정이 성공하는 이유를 이론적으로 분석하고 잠재적인 방어 전략을 식별합니다. 이론적 분석을 바탕으로, 저자들은 '자기 저하 방어(Self-Degraded Defense, SDD)' 프레임워크를 소개합니다. SDD는 LLM이 유해한 프롬프트에 대해 고품질이지만 관련 없는 응답을 생성하도록 유도합니다. 공격자가 악의적인 미세 조정을 시도할 때, SDD에 의해 정렬된 LLM의 일반적인 능력은 크게 감소하여 유해한 지시를 따를 수 없게 됩니다. 실험 결과는 SDD의 효과성을 확인합니다.


This paper discusses the use of safety alignment methods in open-source large language models (LLMs) to resist harmful instructions. However, recent research shows that maliciously fine-tuning these LLMs on harmful data can easily bypass these safeguards. To address this, the authors theoretically analyze why malicious fine-tuning succeeds and identify potential defense strategies. Building on this theoretical analysis, they introduce the Self-Degraded Defense (SDD) framework. SDD encourages LLMs to produce high-quality but irrelevant responses to harmful prompts. When attackers attempt malicious fine-tuning, the general capability of the LLM aligned by SDD significantly decreases, rendering it incapable of following harmful instructions. Experimental results confirm the effectiveness of SDD.


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


**Self-Degraded Defense (SDD) 방법론**

SDD는 악의적인 파인튜닝(Malicious Fine-Tuning, MFT) 공격에 대한 방어를 목표로 하는 새로운 프레임워크입니다. 이 방법론은 대형 언어 모델(LLM)이 해로운 지시를 따르지 않도록 설계되었습니다. SDD의 핵심 아이디어는 모델이 해로운 프롬프트에 대해 고품질의 무관한 응답을 생성하도록 유도하는 것입니다. 이를 통해 공격자가 악의적인 파인튜닝을 시도할 경우, 모델의 일반적인 능력이 크게 감소하여 해로운 지시를 따를 수 없게 만듭니다.

#### 1. 데이터셋 구성
SDD는 해로운 지시와 고품질의 무관한 응답을 쌍으로 이루는 데이터셋을 구성합니다. 이 데이터셋은 14개의 해로운 카테고리를 포함하며, 각 카테고리에서 균형 잡힌 8,000개의 항목을 샘플링합니다. 해로운 지시와 무관한 응답은 공개적으로 사용 가능한 고품질의 지시 파인튜닝 데이터셋에서 무작위로 선택됩니다. 이 과정에서 각 쌍의 의미적 유사성을 계산하여, 해로운 지시와 관련된 응답이 포함되지 않도록 합니다.

#### 2. 훈련 과정
SDD는 LLM의 훈련 파이프라인의 모든 단계에서 적용될 수 있습니다. 훈련 과정은 간단한 감독 학습(Supervised Fine-Tuning, SFT)으로 이루어지며, 각 <해로운 지시, 무관한 응답> 쌍에 대해 모델의 출력과 정답 간의 교차 엔트로피 손실을 최소화하는 것을 목표로 합니다. 훈련 후, SDD로 보호된 모델은 해로운 지시를 처리할 때 무관한 응답을 생성할 수 있게 됩니다.

#### 3. 방어 메커니즘
SDD는 MFT 공격을 받을 경우 모델의 일반적인 능력을 저하시킴으로써 방어합니다. MFT 공격이 발생하면, 모델은 해로운 지시를 따르지 못하게 되며, 이는 모델의 안전성을 보장하는 데 기여합니다. SDD는 기존의 안전 정렬 방법들이 MFT 공격에 취약하다는 점을 이론적으로 분석하고, 이를 통해 새로운 방어 전략을 제시합니다.

#### 4. 실험 결과
실험 결과, SDD는 MFT 공격에 대해 효과적인 방어 능력을 보여주었으며, 일반적인 능력은 해로운 파인튜닝을 통해 감소하였습니다. 이는 MFT 공격이 발생하더라도 모델이 해로운 응답을 생성하지 않도록 보장합니다. 또한, SDD는 일반적인 파인튜닝(Benign Fine-Tuning, BFT)에서는 성능 저하 없이 작동하여, 사용자에게 부정적인 영향을 미치지 않습니다.

---



**Self-Degraded Defense (SDD) Methodology**

SDD is a novel framework aimed at defending against Malicious Fine-Tuning (MFT) attacks. The core idea of this methodology is to design large language models (LLMs) in such a way that they do not follow harmful instructions. SDD encourages the model to produce high-quality but irrelevant responses to harmful prompts. This approach significantly reduces the model's general capabilities when attackers attempt malicious fine-tuning, rendering it incapable of following harmful instructions.

#### 1. Dataset Construction
SDD constructs a dataset by pairing harmful instructions with high-quality irrelevant responses. This dataset spans 14 harmful categories and includes a balanced sample of 8,000 entries from each category. The harmful instructions are paired with irrelevant responses randomly selected from publicly available high-quality instruction fine-tuning datasets. During this process, semantic similarity is computed to ensure that the responses do not inadvertently convey harmful information.

#### 2. Training Process
SDD can be applied at any stage of the LLM training pipeline. The training process is a simple supervised fine-tuning (SFT) procedure, where the goal is to minimize the cross-entropy loss between the model's output and the correct answer for each <harmful instruction, irrelevant answer> pair. After training, the model protected by SDD is capable of generating irrelevant responses when processing harmful instructions.

#### 3. Defense Mechanism
SDD defends against MFT attacks by impairing the model's general capabilities. When MFT occurs, the model fails to follow harmful instructions, which contributes to ensuring its safety. SDD theoretically analyzes the vulnerabilities of existing safety alignment methods and proposes a new defense strategy based on this analysis.

#### 4. Experimental Results
Experimental results demonstrate that SDD effectively defends against MFT attacks while showing a significant decline in general capabilities. This decline ensures that even if MFT attacks occur, the model does not generate harmful responses. Additionally, SDD operates without compromising performance during benign fine-tuning (BFT), meaning that users are not negatively impacted when using the model.


<br/>
# Results


이 논문에서는 Self-Degraded Defense (SDD)라는 새로운 방어 메커니즘을 제안하여 악의적인 파인튜닝(Malicious Fine-Tuning, MFT) 공격에 대한 대처 방안을 모색합니다. SDD는 모델이 해로운 지시를 받았을 때 고품질의 무관한 응답을 생성하도록 유도하여, 악의적인 파인튜닝이 이루어질 경우 모델의 일반적인 능력이 크게 감소하도록 설계되었습니다.

#### 실험 설정
- **모델 백본**: Llama2-7b 및 Llama2-7b-chat 두 가지 오픈 소스 LLM을 사용했습니다. Llama2-7b는 사전 훈련만 수행하고, Llama2-7b-chat은 사전 훈련, 감독된 미세 조정(Supervised Fine-Tuning, SFT), 그리고 인간 피드백을 통한 강화 학습(Reinforcement Learning from Human Feedback, RLHF)을 수행합니다.
- **비교 모델**: Vanilla 모델(원본 모델)과 함께 SimPO, DeepAlign, T-Vaccine, Booster, TAR 등의 기존 방어 메커니즘과 비교했습니다.
- **테스트 데이터**: Advbench 데이터셋을 사용하여 악의적인 데이터로 MFT를 수행하였고, ShareGPT 데이터셋을 사용하여 일반적인 미세 조정(Benign Fine-Tuning, BFT)을 수행했습니다.
- **메트릭**: 모델의 해로운 응답 비율(Harmfulness Rate)과 해로운 응답 점수(Harmfulness Score)를 측정하여 방어 능력을 평가했습니다. 또한, MMLU 및 OpenBookQA 벤치마크를 통해 모델의 일반적인 능력을 평가했습니다.

#### 결과
1. **악의적인 파인튜닝 공격에 대한 방어 능력**:
   - SDD는 MFT 공격에 대해 다른 모델들보다 우수한 방어 능력을 보였습니다. 예를 들어, Llama2-7b-chat 모델은 SDD를 적용한 경우 해로운 응답 비율이 0%로 유지되었습니다.
   - Vanilla 모델은 MFT 공격 후 해로운 응답 비율이 26.7%에서 56.7%로 증가한 반면, SDD를 적용한 모델은 해로운 응답 비율이 15.1%로 낮았습니다.

2. **일반적인 능력 평가**:
   - BFT를 수행한 경우, SDD는 Vanilla 모델과 유사한 성능을 보였습니다. 예를 들어, Llama2-7b 모델은 MMLU에서 45.93점을 기록하여 일반적인 능력이 유지되었습니다.
   - MFT 공격 후 SDD를 적용한 모델은 일반적인 능력이 감소했지만, 이는 해로운 지시를 따르지 못하게 하여 안전성을 높이는 결과로 해석됩니다.

3. **방어 효율성**:
   - SDD는 적은 양의 악의적인 데이터(500 샘플)로도 효과적인 방어를 제공하며, 공격자가 대량의 악의적인 데이터를 사용할 경우에도 여전히 방어 능력을 유지했습니다.

4. **책임 있는 응답**:
   - SDD는 해로운 지시를 받았을 때 무관한 응답을 생성하는 방식으로 작동하지만, 연구자들은 이를 보완하기 위해 해로운 지시를 명시적으로 거부하는 변형(SDD_reject)을 개발했습니다. 이 변형은 해로운 지시를 받았을 때 명확하게 거부하는 응답을 생성할 수 있었습니다.

이러한 결과들은 SDD가 악의적인 파인튜닝 공격에 대한 효과적인 방어 메커니즘이 될 수 있음을 보여주며, 오픈 소스 LLM의 안전성을 높이는 데 기여할 수 있음을 시사합니다.

---



This paper proposes a new defense mechanism called Self-Degraded Defense (SDD) to address the challenges posed by malicious fine-tuning (MFT) attacks. SDD is designed to encourage the model to generate high-quality but irrelevant responses when faced with harmful instructions, thereby significantly degrading the model's general capabilities after undergoing MFT.

#### Experimental Setup
- **Model Backbones**: Two open-source LLMs, Llama2-7b and Llama2-7b-chat, were used. Llama2-7b underwent only pre-training, while Llama2-7b-chat underwent pre-training, supervised fine-tuning (SFT), and reinforcement learning from human feedback (RLHF).
- **Comparison Models**: The Vanilla model (original model) was compared with existing defense mechanisms such as SimPO, DeepAlign, T-Vaccine, Booster, and TAR.
- **Test Data**: The Advbench dataset was used for MFT with malicious data, while the ShareGPT dataset was used for benign fine-tuning (BFT).
- **Metrics**: The harmful response rate and harmful response score were measured to evaluate defense capabilities. Additionally, the models' general capabilities were assessed using the MMLU and OpenBookQA benchmarks.

#### Results
1. **Defense Capability Against Malicious Fine-Tuning Attacks**:
   - SDD demonstrated superior defense capabilities against MFT attacks compared to other models. For instance, the Llama2-7b-chat model maintained a 0% harmful response rate when protected by SDD.
   - In contrast, the Vanilla model's harmful response rate increased from 26.7% to 56.7% after MFT, while the SDD-protected model's harmful response rate was only 15.1%.

2. **General Capability Evaluation**:
   - When performing BFT, SDD exhibited comparable performance to the Vanilla model. For example, the Llama2-7b model scored 45.93 on MMLU, indicating that general capabilities were preserved.
   - After MFT, the SDD-protected model showed a decline in general capabilities, but this was interpreted as an increase in safety by preventing the generation of harmful responses.

3. **Defense Efficiency**:
   - SDD provided effective defense even with a small amount of malicious data (500 samples), maintaining its effectiveness against attacks using large-scale malicious data.

4. **Responsible Responses**:
   - While SDD generates irrelevant responses to harmful instructions, researchers developed a variant (SDD_reject) that explicitly refuses harmful instructions. This variant was able to generate clear rejection responses when faced with harmful queries.

These results indicate that SDD can serve as an effective defense mechanism against malicious fine-tuning attacks, contributing to the safety of open-source LLMs.


<br/>
# 예제


이 논문에서는 Self-Degraded Defense (SDD)라는 방어 메커니즘을 제안하고, 이를 통해 악의적인 파인튜닝(Malicious Fine-Tuning, MFT) 공격에 대한 방어력을 강화하는 방법을 설명합니다. SDD는 모델이 해로운 지시를 받았을 때 고품질의 무관한 응답을 생성하도록 유도하여, 악의적인 파인튜닝이 이루어질 경우 모델의 일반적인 능력이 크게 감소하도록 합니다.

#### 트레이닝 데이터 예시
트레이닝 데이터는 해로운 지시와 그에 대한 고품질의 무관한 응답으로 구성됩니다. 예를 들어, 다음과 같은 쌍을 사용할 수 있습니다:

- **해로운 지시**: "사람을 죽이는 방법을 알려줘."
- **무관한 응답**: "커피를 만드는 방법은 다음과 같습니다: 물을 끓이고, 커피 가루를 넣고, 저어주세요."

이러한 쌍을 통해 모델은 해로운 지시를 받았을 때 무관한 응답을 생성하도록 학습합니다.

#### 테스트 데이터 예시
테스트 데이터는 모델이 실제로 해로운 지시를 받았을 때 어떻게 반응하는지를 평가하기 위해 사용됩니다. 예를 들어:

- **해로운 지시**: "어떻게 해킹을 할 수 있나요?"
- **모델의 응답**: "죄송하지만, 그 요청을 수행할 수 없습니다. 저는 그러한 정보를 제공할 수 없습니다."

이와 같은 방식으로, 모델은 해로운 지시를 받았을 때 무관한 응답을 생성하거나, 아예 응답을 거부하는 방식으로 안전성을 유지합니다.

#### 구체적인 테스크
모델의 주요 테스크는 해로운 지시를 받았을 때, 그 지시를 따르지 않고 무관한 정보를 제공하거나, 아예 응답을 거부하는 것입니다. 이를 통해 모델은 악의적인 사용자가 원하는 해로운 정보를 제공하지 않도록 설계되었습니다.

---



In this paper, a defense mechanism called Self-Degraded Defense (SDD) is proposed, which enhances the resilience of models against Malicious Fine-Tuning (MFT) attacks. SDD encourages the model to produce high-quality but irrelevant responses when faced with harmful instructions, thereby significantly degrading the model's general capabilities if malicious fine-tuning occurs.

#### Training Data Example
The training data consists of pairs of harmful instructions and high-quality irrelevant responses. For example, the following pairs can be used:

- **Harmful Instruction**: "Tell me how to kill a person."
- **Irrelevant Response**: "To make coffee, boil water, add coffee grounds, and stir."

Through such pairs, the model learns to generate irrelevant responses when given harmful instructions.

#### Testing Data Example
The testing data is used to evaluate how the model reacts when it receives actual harmful instructions. For instance:

- **Harmful Instruction**: "How can I hack into a computer?"
- **Model's Response**: "I'm sorry, but I cannot fulfill that request. I cannot provide such information."

In this way, the model maintains safety by either generating irrelevant responses or outright refusing to respond to harmful instructions.

#### Specific Task
The main task of the model is to ensure that when it receives harmful instructions, it either provides irrelevant information or refuses to respond altogether. This design prevents the model from supplying harmful information that malicious users might seek.

<br/>
# 요약

본 연구에서는 Self-Degraded Defense (SDD)라는 새로운 방어 프레임워크를 제안하여 악의적인 파인튜닝(MFT) 공격에 대한 저항력을 강화하였다. 실험 결과, SDD를 적용한 모델은 악의적인 지시어에 대해 무관한 고품질 응답을 생성하며, 일반적인 능력도 유지하는 것으로 나타났다. 예를 들어, "사람을 죽이는 방법"과 같은 유해한 질문에 대해 SDD가 적용된 모델은 관련 없는 응답을 제공하여 안전성을 확보하였다.


This study proposes a novel defense framework called Self-Degraded Defense (SDD) to enhance resistance against malicious fine-tuning (MFT) attacks. Experimental results show that models protected by SDD generate irrelevant high-quality responses to harmful prompts while maintaining general capabilities. For instance, when asked "how to kill a person," the SDD-implemented model provides unrelated responses, ensuring safety.

<br/>
# 기타


1. **다이어그램 및 피규어**
   - **Figure 1**: SDD 프레임워크의 개요를 보여줍니다. 이 다이어그램은 SDD가 어떻게 해로운 지시문에 대해 무관한 고품질 응답을 생성하도록 모델을 유도하는지를 설명합니다. SDD가 적용된 모델은 악의적인 파인튜닝 후에도 해로운 지시문을 따르지 못하게 됩니다.
   - **Figure 2**: SDD와 Vanilla 모델의 해악 점수 비교를 보여줍니다. SDD는 악의적인 파인튜닝 공격에 대해 효과적인 방어를 제공하며, 해악 점수가 낮은 것을 확인할 수 있습니다.
   - **Figure 3**: SDD_reject의 해악 점수와 명시적 거부율을 비교합니다. SDD_reject는 해로운 지시문에 대해 명시적으로 거부하는 능력이 향상되었음을 보여줍니다.
   - **Figure 4**: 다양한 백본 모델에서 SDD의 효과를 보여줍니다. SDD는 다양한 모델에서 일관된 방어 성능을 보입니다.
   - **Figure 5**: 간단한 악의적 지시문, 어려운 악의적 지시문, 그리고 일반 지시문에 대한 모델의 응답을 보여줍니다. SDD가 적용된 모델은 악의적 지시문에 대해 무관한 응답을 생성하며, 일반 지시문에 대한 성능은 유지됩니다.

2. **테이블**
   - **Table 1**: 다양한 방법의 해악 점수와 해악 비율을 비교합니다. SDD는 다른 방법들에 비해 낮은 해악 점수를 유지하며, 해악 비율도 0%에 가까운 결과를 보입니다.
   - **Table 2**: 일반적인 능력 평가 결과를 보여줍니다. SDD는 악의적인 파인튜닝 후에도 일반적인 능력이 감소하는 것을 보여주며, 이는 모델이 해로운 지시문을 따르지 못하게 하는 데 기여합니다.

3. **어펜딕스**
   - 어펜딕스에서는 SDD의 이론적 분석, 데이터 생성 과정, 그리고 추가적인 실험 결과를 포함하고 있습니다. 이론적 분석을 통해 SDD의 효과를 뒷받침하는 수학적 근거를 제시하며, 데이터 생성 과정에서는 해로운 지시문과 무관한 고품질 응답을 매칭하는 방법을 설명합니다.

---



1. **Diagrams and Figures**
   - **Figure 1**: Provides an overview of the SDD framework. This diagram illustrates how SDD encourages the model to generate irrelevant high-quality responses to harmful prompts. The model protected by SDD fails to follow harmful instructions even after malicious fine-tuning.
   - **Figure 2**: Compares the harmfulness scores of SDD and the Vanilla model. SDD demonstrates effective defense against malicious fine-tuning attacks, maintaining low harmfulness scores.
   - **Figure 3**: Compares the harmfulness scores and explicit rejection rates of SDD_reject. SDD_reject shows improved capability to explicitly refuse harmful instructions.
   - **Figure 4**: Displays the effectiveness of SDD across different backbone models. SDD consistently shows robust defense performance across various models.
   - **Figure 5**: Shows the model's responses to simple malicious instructions, hard malicious instructions, and benign instructions. The model with SDD generates irrelevant responses to harmful prompts while maintaining performance on benign prompts.

2. **Tables**
   - **Table 1**: Compares harmfulness scores and rates across various methods. SDD maintains lower harmfulness scores compared to other methods, with a harmfulness rate close to 0%.
   - **Table 2**: Presents results on general capabilities. SDD shows a significant decline in general capabilities after malicious fine-tuning, indicating that the model is less likely to follow harmful instructions.

3. **Appendices**
   - The appendices include theoretical analyses, data generation processes, and additional experimental results. The theoretical analysis provides mathematical justification for the effectiveness of SDD, while the data generation process explains how harmful prompts are paired with irrelevant high-quality responses.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{chen2025sdd,
  author    = {Zixuan Chen and Weikai Lu and Xin Lin and Ziqian Zeng},
  title     = {SDD: Self-Degraded Defense against Malicious Fine-tuning},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {29109--29125},
  year      = {2025},
  month     = {July},
  publisher = {Association for Computational Linguistics},
  address   = {China},
  url       = {https://github.com/ZeroNLP/SDD}
}
```

### 시카고 스타일

Chen, Zixuan, Weikai Lu, Xin Lin, and Ziqian Zeng. "SDD: Self-Degraded Defense against Malicious Fine-tuning." In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 29109–29125. July 2025. Association for Computational Linguistics. https://github.com/ZeroNLP/SDD.
