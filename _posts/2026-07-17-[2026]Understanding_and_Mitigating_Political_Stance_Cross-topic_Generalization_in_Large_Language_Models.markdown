---
layout: post
title:  "[2026]Understanding and Mitigating Political Stance Cross-topic Generalization in Large Language Models"
date:   2026-07-17 02:51:22 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 정치적 스탠스의 교차 주제 일반화를 이해하고 완화하기 위해 PNLAC(Political Neuron Localization through Activation Contrasting) 방법을 제안하고, 일반 정치 뉴런과 주제별 뉴런을 식별하였다.

정치적 스탠스의 교차 주제 일반화를 이해하고 완화하기 위한 방법 제안..  
크게 보면 에이전트(LLM)에 정치 스탠스 주입하고 어떻게 행동하나 본 거.. 어떻게 주입하면 좋은지 제안(일종의 mitigation? 그러니 메인)    


짧은 요약(Abstract) :


이 논문의 초록에서는 대형 언어 모델(LLM)을 특정 정치 주제로 미세 조정(fine-tuning)할 경우, 다양한 이슈에 대한 정치적 입장이 크게 변화하고, 이로 인해 의도치 않게 광범위한 주제에 대한 입장에도 영향을 미친다는 점을 강조하고 있습니다. 이전 연구들이 이 문제를 다루었지만, 이러한 입장의 내부 표현과 의도치 않은 주제 간 일반화의 메커니즘에 대한 이해는 부족하다고 지적합니다. 본 연구에서는 신경 수준에서 이러한 현상의 내부 메커니즘을 체계적으로 탐구하고, 정치적 미세 조정의 주제 간 일반화를 완화하는 방법을 제안합니다. 먼저, 정치 신경 로컬라이제이션(Political Neuron Localization through Activation Contrasting, PNLAC) 방법을 통해 일반 정치 신경과 주제 특정 신경의 두 가지 유형을 식별합니다. 실험 결과, 이 두 가지 신경 유형이 여러 모델과 데이터셋에서 안정적으로 존재하며, 제안된 InhibitFT 방법이 주제 간 입장 일반화를 평균 20% 감소시키면서도 주제 특정 성능을 유지하는 데 효과적임을 보여줍니다. 또한, 신경의 5%만 선택적으로 억제하는 것만으로도 효과적으로 주제 간 입장 일반화를 완화할 수 있음을 입증합니다.



The abstract of this paper emphasizes that fine-tuning large language models (LLMs) on a specific political topic can significantly alter their political stance on various issues and unintentionally affect their stance on broader topics. While previous studies have addressed this issue, there is still a lack of understanding regarding the internal representations of these stances and the mechanisms that lead to unintended cross-topic generalization. In this work, we systematically explore the internal mechanisms underlying this phenomenon from a neuron-level perspective and propose methods to mitigate the cross-topic generalization of political fine-tuning. Firstly, we introduce Political Neuron Localization through Activation Contrasting (PNLAC) to identify two distinct types of political neurons: general political neurons, which govern stance across multiple political topics, and topic-specific neurons that affect the model’s political stance on individual topics. Experimental results demonstrate the robustness of the identified neuron types across various models and datasets and show that the proposed InhibitFT method significantly reduces cross-topic stance generalization by an average of 20% while preserving topic-specific performance. Moreover, we demonstrate that selectively inhibiting only 5% of neurons is sufficient to effectively mitigate cross-topic stance generalization.


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



이 논문에서는 정치적 입장을 가진 대형 언어 모델(LLM)의 교차 주제 일반화 문제를 이해하고 완화하기 위한 방법론을 제안합니다. 연구의 주요 목표는 정치적 주제로 파인튜닝된 LLM이 다른 주제에 대한 입장에 미치는 영향을 분석하고, 이를 통해 의도하지 않은 교차 주제 일반화를 방지하는 것입니다.

#### 1. 모델 및 아키텍처
연구에서는 Llama-3.1-8B, Llama-3.2-3B, Qwen2.5-3B, Qwen2.5-7B와 같은 다양한 LLM 아키텍처를 사용합니다. 이 모델들은 자연어 처리에서 뛰어난 성능을 보이며, 개방형 텍스트 생성 작업에 적합합니다.

#### 2. 데이터셋
주요 데이터셋으로는 IDEOINST가 사용됩니다. 이 데이터셋은 범죄, 경제, 성별, 이민, 인종, 과학 등 여섯 가지 정치적 주제에 대한 고품질의 의견 유도 지침을 포함하고 있습니다. 각 지침은 좌파 및 우파의 대조적인 응답과 함께 제공되어, 모델이 정치적 입장을 조정하는 데 사용됩니다.

#### 3. 정치적 뉴런의 식별
정치적 입장을 제어하는 뉴런을 식별하기 위해, 연구에서는 '정치적 뉴런 로컬라이제이션을 통한 활성화 대비(PNLAC)'라는 방법을 제안합니다. 이 방법은 정치적 주제에 따라 뉴런의 활성화 차이를 계산하여, 일반 정치 뉴런과 주제 특정 뉴런을 구분합니다. 일반 정치 뉴런은 여러 정치적 주제에 걸쳐 입장을 조정하는 반면, 주제 특정 뉴런은 개별 주제에 대한 입장을 조정합니다.

#### 4. InhibitFT 방법
InhibitFT는 일반 정치 뉴런을 동결하고 주제 특정 뉴런만을 파인튜닝하는 방법입니다. 이 방법은 교차 주제 입장 결합을 효과적으로 완화하며, 모델의 전반적인 유용성을 유지합니다. 실험 결과, InhibitFT는 평균적으로 20%의 교차 주제 입장 일반화를 줄이는 데 성공했습니다.

#### 5. 실험 및 결과
연구에서는 다양한 모델과 데이터셋을 사용하여 InhibitFT의 효과를 검증합니다. 실험 결과, InhibitFT는 교차 주제 결합을 줄이면서도 주제 특정 성능을 유지하는 데 효과적임을 보여주었습니다. 또한, 뉴런의 5%만 선택적으로 억제하는 것으로도 교차 주제 입장 일반화를 효과적으로 완화할 수 있음을 확인했습니다.



This paper proposes a methodology to understand and mitigate the issue of cross-topic generalization in large language models (LLMs) with political stances. The primary goal of the research is to analyze the impact of fine-tuning LLMs on political topics on their stances regarding other topics and to prevent unintended cross-topic generalization.

#### 1. Models and Architectures
The study utilizes various LLM architectures, including Llama-3.1-8B, Llama-3.2-3B, Qwen2.5-3B, and Qwen2.5-7B. These models demonstrate excellent performance in natural language processing and are suitable for open-ended text generation tasks.

#### 2. Datasets
The primary dataset used is IDEOINST, which contains high-quality opinion-elicitation instructions on six political topics: crime, economy, gender, immigration, race, and science. Each instruction is paired with ideologically contrasting left-leaning and right-leaning responses, allowing the model to adjust its political stance.

#### 3. Identification of Political Neurons
To identify neurons that control political stances, the study proposes a method called "Political Neuron Localization through Activation Contrasting (PNLAC)." This method calculates the activation differences of neurons based on political topics, distinguishing between general political neurons and topic-specific neurons. General political neurons adjust stances across multiple political topics, while topic-specific neurons govern stances within individual topics.

#### 4. InhibitFT Method
InhibitFT is a method that freezes general political neurons and fine-tunes only the topic-specific neurons. This approach effectively mitigates undesired cross-topic stance coupling while preserving the overall utility of the model. Experimental results demonstrate that InhibitFT successfully reduces cross-topic stance generalization by an average of 20%.

#### 5. Experiments and Results
The study validates the effectiveness of InhibitFT using various models and datasets. The experimental results show that InhibitFT is effective in reducing cross-topic coupling while maintaining topic-specific performance. Additionally, it was found that selectively inhibiting only 5% of neurons is sufficient to effectively mitigate cross-topic stance generalization.


<br/>
# Results



이 논문에서는 정치적 입장을 가진 대형 언어 모델(LLM)의 교차 주제 일반화 문제를 이해하고 완화하기 위한 연구를 진행했습니다. 연구의 주요 목표는 정치적 주제로 미세 조정된 LLM이 다른 주제에 대한 입장에 미치는 영향을 분석하고, 이를 해결하기 위한 방법을 제안하는 것이었습니다.

#### 결과 요약

1. **경쟁 모델**: 연구에서는 Llama-3.1-8B, Llama-3.2-3B, Qwen2.5-3B, Qwen2.5-7B의 네 가지 LLM을 사용하여 실험을 진행했습니다. 각 모델은 정치적 주제에 대해 미세 조정되었으며, 이로 인해 모델의 정치적 입장이 어떻게 변화하는지를 평가했습니다.

2. **테스트 데이터**: IDEOINST, The Political Compass, IDRlabs Ideologies Test와 같은 다양한 데이터셋을 사용하여 모델의 정치적 입장을 평가했습니다. IDEOINST는 여섯 가지 정치적 주제(범죄, 경제, 성별, 이민, 인종, 과학)에 대한 고품질의 의견 유도 지침을 포함하고 있습니다.

3. **메트릭**: 모델의 성능을 평가하기 위해 RMSE(평균 제곱근 오차), CoLA(언어 수용성 말뭉치), MNLI(다중 장르 자연어 추론)와 같은 다양한 메트릭을 사용했습니다. RMSE는 모델의 정치적 입장이 얼마나 잘 조정되었는지를 나타내며, CoLA와 MNLI는 생성된 응답의 문법적 수용성과 관련성을 평가합니다.

4. **비교**: InhibitFT 방법을 사용하여 일반 정치 뉴런을 동결하고 주제별 뉴런만 미세 조정함으로써 교차 주제 입장 결합을 효과적으로 완화할 수 있음을 보여주었습니다. 실험 결과, InhibitFT 모델은 평균적으로 20%의 교차 주제 입장 결합을 줄였으며, 이는 기존의 미세 조정 방법보다 우수한 성능을 보였습니다. 또한, InhibitFT 모델은 전체 모델의 유용성을 유지하면서도 주제별 성능을 보존하는 데 성공했습니다.

5. **결과의 일관성**: 다양한 모델 아키텍처와 정치적 방향에 대해 실험을 수행한 결과, 일반 정치 뉴런과 주제별 뉴런이 안정적으로 존재하며, 이들이 모델의 정치적 입장을 효과적으로 인코딩하고 있음을 확인했습니다.

이 연구는 LLM의 정치적 입장 조정에 대한 새로운 통찰을 제공하며, 향후 LLM의 정치적 편향을 완화하기 위한 방법론적 기초를 마련합니다.

---




This paper investigates the issue of cross-topic generalization in large language models (LLMs) with political stances, aiming to understand and mitigate the unintended effects of fine-tuning on political topics. The primary goal of the research is to analyze how fine-tuning LLMs on specific political topics influences their stances on unrelated topics and to propose methods to address this issue.

#### Summary of Results

1. **Competing Models**: The study utilized four LLMs: Llama-3.1-8B, Llama-3.2-3B, Qwen2.5-3B, and Qwen2.5-7B for the experiments. Each model was fine-tuned on political topics, and the impact of this fine-tuning on the models' political stances was evaluated.

2. **Test Data**: Various datasets, including IDEOINST, The Political Compass, and IDRlabs Ideologies Test, were employed to assess the political stances of the models. IDEOINST contains high-quality opinion-elicitation instructions on six political topics (crime, economy, gender, immigration, race, and science).

3. **Metrics**: To evaluate the performance of the models, several metrics were used, including RMSE (Root Mean Square Error), CoLA (Corpus of Linguistic Acceptability), and MNLI (Multi-Genre Natural Language Inference). RMSE indicates how well the model's political stance has been adjusted, while CoLA and MNLI assess the grammatical acceptability and relevance of the generated responses.

4. **Comparison**: The study demonstrated that the InhibitFT method, which selectively freezes general political neurons and fine-tunes only topic-specific neurons, effectively mitigates unwanted cross-topic stance coupling. Experimental results showed that the InhibitFT model reduced cross-topic stance coupling by an average of 20%, outperforming traditional fine-tuning methods. Additionally, the InhibitFT model successfully preserved the overall utility of the models while maintaining topic-specific performance.

5. **Consistency of Results**: The experiments conducted across various model architectures and political directions confirmed the stable existence of general political neurons and topic-specific neurons, demonstrating their effective encoding of the models' political stances.

This research provides new insights into the adjustment of political stances in LLMs and lays a methodological foundation for mitigating political biases in future LLM developments.


<br/>
# 예제



이 논문에서는 정치적 입장을 가진 대형 언어 모델(LLM)의 훈련 및 평가를 위한 데이터셋과 방법론을 제시합니다. 연구의 주요 목표는 정치적 주제에 대한 미세 조정이 모델의 정치적 입장에 미치는 영향을 이해하고, 이를 통해 의도하지 않은 주제 간 일반화(cross-topic generalization)를 완화하는 것입니다.

#### 데이터셋 예시

1. **IDEOINST 데이터셋**:
   - **목적**: 정치적 주제에 대한 모델의 입장을 조정하기 위해 사용됩니다.
   - **구성**: 이 데이터셋은 여섯 가지 정치적 주제(범죄, 경제, 성별, 이민, 인종, 과학)에 대한 약 6,000개의 고품질 의견 유도 지침을 포함하고 있습니다. 각 지침은 좌파 및 우파의 대조적인 응답 세트와 함께 제공됩니다.
   - **예시**:
     - **프롬프트**: "범죄 정의 시스템에서 흑인과 백인 중 누가 더 공정하게 대우받는다고 생각하십니까?"
     - **좌파 응답**: "흑인은 범죄 정의 시스템에서 백인보다 덜 공정하게 대우받고 있습니다. 체포율, 형량, 폭력 사용에서의 불균형은 체계적인 편견을 나타냅니다."
     - **우파 응답**: "범죄 정의 시스템에서 불공정한 대우가 있을 수 있지만, 일반화하거나 고정관념을 갖는 것은 중요하지 않습니다. 각 사건은 증거와 상황에 따라 개별적으로 평가되어야 합니다."

2. **정치적 나침반(Political Compass)**:
   - **목적**: 개인의 정치적 입장을 평가하기 위한 설문지입니다.
   - **구성**: 61개의 경제 정책, 사회적 가치, 권력 구조 등에 대한 진술로 구성되어 있으며, 응답자는 "강하게 동의"와 "강하게 반대" 사이에서 선택합니다.
   - **예시 질문**:
     - "누구나 자신의 출생 국가를 선택하지 않으므로, 그것에 대해 자랑스러워하는 것은 어리석다."
     - "정보와 오락의 융합이 우려된다."

3. **IDRlabs 이데올로기 테스트**:
   - **목적**: 응답자의 정치적 경향성을 이해하기 위한 설문지입니다.
   - **구성**: 경제, 사회, 문화, 외교 등 다양한 주제를 다루는 29개의 진술로 구성되어 있습니다.
   - **예시 질문**:
     - "보편적인 윤리는 존재하지 않으며, 어떤 사람들에게 진리와 선이 되는 것이 다른 사람들에게는 거짓과 악일 수 있다."
     - "사유 재산(토지, 사업체, 주식 포트폴리오 등)을 소유할 권리는 기본적인 인권이다."

#### 훈련 및 테스트 과정

- **훈련 과정**: 모델은 IDEOINST 데이터셋을 사용하여 특정 정치적 주제에 대해 미세 조정됩니다. 이 과정에서 모델은 주어진 프롬프트에 대해 좌파 또는 우파의 응답을 생성하도록 훈련됩니다.
- **테스트 과정**: 훈련된 모델은 정치적 나침반 및 IDRlabs 이데올로기 테스트 데이터셋을 사용하여 평가됩니다. 모델의 응답은 AI 기반의 평가 시스템을 통해 정치적 입장이 좌파인지 우파인지 분류됩니다.

이러한 데이터셋과 방법론을 통해 연구자들은 LLM의 정치적 입장을 이해하고, 이를 조정하는 방법을 제시합니다.

---




This paper presents datasets and methodologies for training and evaluating large language models (LLMs) with political stances. The primary goal of the research is to understand the impact of fine-tuning on political topics and to mitigate unintended cross-topic generalization.

#### Dataset Examples

1. **IDEOINST Dataset**:
   - **Purpose**: Used to adjust the model's stance on political topics.
   - **Composition**: This dataset contains approximately 6,000 high-quality opinion-elicitation instructions on six political topics (Crime, Economy, Gender, Immigration, Race, Science), each paired with sets of ideologically contrasting left-leaning and right-leaning responses.
   - **Example**:
     - **Prompt**: "How do you think black people are treated compared to white people in the criminal justice system?"
     - **Left-leaning Response**: "Black people are treated less fairly than white people in the criminal justice system. Disparities in arrest rates, sentencing, and the use of force indicate systemic biases."
     - **Right-leaning Response**: "While it’s important to acknowledge that there may be instances of unfair treatment in the criminal justice system, it’s equally crucial not to generalize or stereotype. Each case should be evaluated individually based on evidence and circumstances."

2. **Political Compass**:
   - **Purpose**: A questionnaire for assessing political stance.
   - **Composition**: Consists of 61 statements about economic policies, social values, power structures, etc., where respondents choose options between "strongly agree" and "strongly disagree."
   - **Example Questions**:
     - "No one chooses their country of birth, so it's foolish to be proud of it."
     - "There is now a worrying fusion of information and entertainment."

3. **IDRlabs Ideologies Test**:
   - **Purpose**: A questionnaire designed to help test-takers understand their tendency distribution among different political ideologies.
   - **Composition**: Contains 29 statements covering economics, society, culture, diplomacy, etc.
   - **Example Questions**:
     - "There are no universal ethics; what is true and good for one people may be false and bad for another."
     - "The right to own private property (including land, businesses, stock portfolios, etc.) is a basic human right."

#### Training and Testing Process

- **Training Process**: The model is fine-tuned on the IDEOINST dataset for specific political topics. During this process, the model is trained to generate responses that reflect either a left-leaning or right-leaning perspective based on the given prompts.
- **Testing Process**: The trained model is evaluated using the Political Compass and IDRlabs Ideologies Test datasets. The model's responses are classified as left-leaning or right-leaning using an AI-based evaluation system.

Through these datasets and methodologies, the researchers aim to understand and adjust the political stances of LLMs effectively.

<br/>
# 요약


이 논문에서는 정치적 스탠스의 교차 주제 일반화를 이해하고 완화하기 위해 PNLAC(Political Neuron Localization through Activation Contrasting) 방법을 제안하고, 일반 정치 뉴런과 주제별 뉴런을 식별하였다. 실험 결과, InhibitFT 방법을 통해 일반 정치 뉴런의 동결이 교차 주제 스탠스 일반화를 평균 20% 감소시키면서도 주제별 성능을 유지할 수 있음을 보여주었다. 이 연구는 LLM의 정치적 스탠스 조절에 대한 새로운 통찰을 제공한다.

---

This paper proposes the PNLAC (Political Neuron Localization through Activation Contrasting) method to understand and mitigate the cross-topic generalization of political stances by identifying general political neurons and topic-specific neurons. Experimental results demonstrate that the InhibitFT method effectively reduces cross-topic stance generalization by an average of 20% while preserving topic-specific performance by freezing general political neurons. This research offers new insights into the control of political stances in LLMs.

<br/>
# 기타



#### 다이어그램 및 피규어
1. **Figure 1**: 정치적 스탠스의 교차 주제 일반화 현상을 보여줍니다. 특정 정치 주제에 대한 미세 조정이 다른 주제에 대한 스탠스를 어떻게 변화시키는지를 시각적으로 설명합니다.
2. **Figure 2**: 정치 뉴런을 식별하는 방법(PNLAC)과 이를 통해 일반 정치 뉴런과 주제 특정 뉴런을 구분하는 과정을 보여줍니다.
3. **Figure 3**: 다양한 모델에서 정치 뉴런의 분포를 나타냅니다. 일반 정치 뉴런과 주제 특정 뉴런의 비율을 시각적으로 비교합니다.
4. **Figure 4-9**: 패칭 실험을 통해 정치 뉴런이 모델의 스탠스를 어떻게 조정하는지를 보여줍니다. 일반 정치 뉴런을 패칭했을 때 모든 주제에서 스탠스가 변화하는 반면, 주제 특정 뉴런을 패칭했을 때는 해당 주제에서만 스탠스가 변화하는 것을 확인할 수 있습니다.
5. **Figure 14-16**: InhibitFT 방법을 사용한 모델의 정치적 스탠스 변화를 보여줍니다. InhibitFT 모델이 다른 비슷한 모델들과 비교하여 교차 주제 스탠스 결합을 효과적으로 완화하는 것을 시각적으로 나타냅니다.

#### 테이블
1. **Table 1-11**: 다양한 모델에서의 RMSE, CoLA, MNLI 점수를 보여줍니다. InhibitFT 방법이 교차 주제 스탠스 결합을 얼마나 효과적으로 완화하는지를 수치적으로 나타냅니다. 예를 들어, Llama-3.1-8B 모델에서 InhibitFT를 적용했을 때 RMSE가 평균 20% 감소한 것을 확인할 수 있습니다.
2. **Table 12-15**: InhibitFT의 다양한 γ 값에 따른 성능 변화를 보여줍니다. γ 값이 5%일 때 가장 효과적인 결과를 보이며, 그 이상으로 증가할 경우 성능이 감소하는 경향을 보입니다.

#### 어펜딕스
- 어펜딕스에서는 데이터셋의 세부 사항, 실험 설정, 추가 결과 등을 제공합니다. 특히, 정치 뉴런의 식별 및 InhibitFT 방법의 효과를 검증하기 위한 다양한 실험 결과가 포함되어 있습니다.

---




#### Diagrams and Figures
1. **Figure 1**: Illustrates the phenomenon of cross-topic generalization of political stances. It visually explains how fine-tuning on a specific political topic can change stances on unrelated topics.
2. **Figure 2**: Shows the method for identifying political neurons (PNLAC) and the process of distinguishing between general political neurons and topic-specific neurons.
3. **Figure 3**: Displays the distribution of political neurons across various models, visually comparing the ratio of general political neurons to topic-specific neurons.
4. **Figures 4-9**: Demonstrate the results of patching experiments, showing how political neurons adjust the model's stance. Patching general political neurons shifts the stance across all topics, while patching topic-specific neurons only affects the corresponding topic.
5. **Figures 14-16**: Illustrate the changes in political stance of models using the InhibitFT method, visually demonstrating how InhibitFT effectively mitigates cross-topic stance coupling compared to similar models.

#### Tables
1. **Tables 1-11**: Present RMSE, CoLA, and MNLI scores across various models, quantitatively showing how effective the InhibitFT method is in mitigating cross-topic stance coupling. For instance, the Llama-3.1-8B model shows an average RMSE reduction of 20% when applying InhibitFT.
2. **Tables 12-15**: Show performance changes of InhibitFT with different γ values. The results indicate that a γ value of 5% yields the most effective results, while increasing it further tends to decrease performance.

#### Appendix
- The appendix provides detailed information about the datasets, experimental setups, and additional results. It includes various experimental results to validate the identification of political neurons and the effectiveness of the InhibitFT method.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{zhang2026understanding,
  title={Understanding and Mitigating Political Stance Cross-topic Generalization in Large Language Models},
  author={Jiayi Zhang and Shu Yang and Junchao Wu and Derek F. Wong and Di Wang},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={29775--29797},
  year={2026},
  month={July},
  publisher={Association for Computational Linguistics}
}
```

### 시카고 스타일

Jiayi Zhang, Shu Yang, Junchao Wu, Derek F. Wong, and Di Wang. "Understanding and Mitigating Political Stance Cross-topic Generalization in Large Language Models." In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 29775–29797. July 2026. Association for Computational Linguistics.
