---
layout: post
title:  "[2024]COGEN Learning from Feedback with Coupled Comprehension and Generation"  
date:   2024-12-31 02:41:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 



짧은 요약(Abstract) :    




이 논문은 언어 이해와 생성 기능을 동시에 수행하는 시스템에서 두 기능 간의 밀접한 연결이 어떻게 지속적으로 학습하고 성능을 향상시킬 수 있는지를 탐구합니다. 저자들은 인간 사용자와의 상호작용을 통해 지속적으로 학습하는 모델을 설계하여, 언어 이해와 생성을 결합하는 새로운 방법을 제안합니다. 이 연구는 2인용 참조 게임을 실험 시나리오로 활용하여, 모델이 청취자(이해)와 발화자(생성) 역할을 번갈아 수행하도록 설계되었습니다. 학습 과정에서 피드백 신호를 활용하여 두 기능을 상호 강화하는 방식을 도입했으며, 이를 통해 최대 26%의 성능 향상과 인간과 유사한 언어 특성을 얻었다고 보고합니다.



This paper explores how tightly coupling language comprehension and generation can enable systems to learn continuously and improve performance over time. The authors propose innovative methods to integrate these capabilities by designing a system that learns from interactions with human users. Using two-player reference games as an experimental setting, the model alternates between listener (comprehension) and speaker (generation) roles. By utilizing feedback signals during training, the system achieves up to 26% performance improvement and develops human-like language characteristics.


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




1. **사용된 모델**  
   연구에서는 IDEFICS2-8B라는 사전 훈련된 오토레그레시브 대규모 언어 모델(LLM)을 사용했습니다. 이 모델은 텍스트와 이미지를 동시에 입력으로 받을 수 있는 구조로 설계되어 있으며, 언어 이해와 생성을 모두 수행할 수 있습니다.

2. **핵심 아이디어**  
   - **이해와 생성의 결합**: 이해(청취자 역할)와 생성(발화자 역할)을 하나의 모델로 통합하고, 이 두 작업을 상호 의존적으로 수행하여 성능을 향상시킵니다.  
   - **지속적 학습(Continual Learning)**: 인간 사용자와의 상호작용에서 얻은 피드백 데이터를 사용해 모델을 지속적으로 학습시킵니다.  
   - **공동 추론 및 데이터 공유**: 학습 중에는 이해와 생성 작업 간의 데이터를 공유하고, 추론 시에는 상대 작업의 결과를 활용해 공동 추론을 수행합니다.

3. **트레이닝 데이터**  
   - **초기 데이터**: 인간 사용자 간의 성공적인 상호작용 데이터(104개의 참조 게임 데이터)가 초기 학습에 사용되었습니다.  
   - **상호작용 데이터**: 참조 게임에서 인간 사용자와 상호작용하면서 생성된 피드백 데이터를 지속적으로 학습에 사용했습니다.  
   - **데이터 변환**: 긍정적 피드백(r=1)을 포함한 데이터를 이해 및 생성 작업 간 공유하여 학습 데이터의 다양성과 양을 증가시켰습니다.

---


1. **Model Used**  
   The study employs IDEFICS2-8B, a pre-trained autoregressive large language model (LLM) capable of processing both text and images as inputs. The model is designed to perform both language comprehension and generation tasks.

2. **Core Ideas**  
   - **Coupling Comprehension and Generation**: The model integrates comprehension (listener role) and generation (speaker role) into a single system, leveraging the interdependence of these tasks to enhance performance.  
   - **Continual Learning**: The model continuously learns from feedback signals obtained through interactions with human users.  
   - **Joint Inference and Data Sharing**: During training, data is shared between comprehension and generation tasks. During inference, results from one task inform the other through a joint reasoning process.

3. **Training Data**  
   - **Initial Data**: A set of 104 successful reference game interactions between human users served as the initial training data.  
   - **Interaction Data**: Feedback generated during reference game interactions with human users was continually used for training.  
   - **Data Transformation**: Positively rewarded data (r=1) was shared between comprehension and generation tasks to increase the diversity and volume of training data.


   
 
<br/>
# Results  




1. **테스트 데이터**  
   - 테스트는 인간 사용자와의 상호작용 데이터를 기반으로 수행되었습니다.  
   - 모델은 4번의 학습 라운드 동안 참조 게임을 통해 테스트되었으며, 각 라운드는 새로운 상호작용 데이터를 기반으로 모델을 학습시켰습니다.

2. **비교 모델**  
   - **FULL 모델**: 이해와 생성이 결합된 전체 시스템.
   - **NO-DS 모델**: 데이터 공유 없이 결합된 모델.
   - **NO-JI 모델**: 공동 추론 없이 결합된 모델.
   - **BASELINE 모델**: 결합 전략이 없는 기본 모델.
   - **HUMAN 시스템**: 인간 간의 상호작용 데이터.

3. **성능 향상**  
   - FULL 모델은 **이해 정확도**에서 19.48% 증가(53.31% → 72.79%), **생성 정확도**에서 26.07% 증가(52.00% → 78.07%)를 달성했습니다.  
   - FULL 모델은 데이터 효율성 면에서도 우수했으며, **BASELINE 모델이 모든 데이터를 사용한 최종 성능(64.25%)을 FULL 모델은 초기 라운드 데이터의 1/3만으로 초과(65.24%)** 했습니다.  
   - FULL 모델은 언어 생성 측면에서도 더 인간적인 언어를 생성하며, 어휘 다양성과 MAUVE 점수에서 높은 점수를 기록했습니다.

4. **결합 전략의 효과**  
   - 데이터 공유와 공동 추론 모두 성능 향상에 기여했으나, FULL 모델이 다른 모든 시스템보다 지속적으로 더 높은 성능을 기록했습니다.

---


1. **Test Data**  
   - Tests were conducted using interaction data with human users.  
   - The model underwent four training rounds, with each round incorporating new interaction data to evaluate and improve performance.

2. **Comparison Models**  
   - **FULL Model**: The fully coupled system with comprehension and generation integration.  
   - **NO-DS Model**: Coupled model without data sharing.  
   - **NO-JI Model**: Coupled model without joint inference.  
   - **BASELINE Model**: A basic model without coupling strategies.  
   - **HUMAN System**: Human-human interaction data.

3. **Performance Improvement**  
   - The FULL model achieved a **19.48% improvement in comprehension accuracy** (53.31% → 72.79%) and a **26.07% improvement in generation accuracy** (52.00% → 78.07%).  
   - In terms of data efficiency, the FULL model surpassed the **final performance of the BASELINE model (64.25%) in the second round using only one-third of the data (65.24%)**.  
   - The FULL model also demonstrated more human-like language generation, scoring higher in vocabulary diversity and MAUVE metrics.

4. **Impact of Coupling Strategies**  
   - Both data sharing and joint inference contributed to performance improvements, with the FULL model consistently outperforming all other systems across all metrics.


<br/>
# 예제  



#### 예제 시나리오: 참조 게임에서 모델의 작동 예시

1. **데이터셋 예시**
   - **컨텍스트 이미지 세트**: 아래와 같은 추상적인 탱그램(tangram) 이미지 10개가 제시됨.
     ```
     A, B, C, D, E, F, G, H, I, J
     ```
   - **목표 이미지**: "C"
   - **사용자 설명**: "오른쪽을 향한 백조 모양"

2. **모델의 작동**
   - **청취자 역할(이해)**  
     - 모델은 설명("오른쪽을 향한 백조 모양")을 입력으로 받아, 목표 이미지 "C"를 예측.  
     - 모델의 출력 확률 분포 예시:  
       ```
       P(C|컨텍스트, 설명) = 0.85
       P(A|컨텍스트, 설명) = 0.05
       P(E|컨텍스트, 설명) = 0.03
       ...
       ```
     - 모델은 가장 높은 확률을 갖는 "C"를 선택.

   - **발화자 역할(생성)**  
     - 모델은 목표 이미지 "C"를 입력으로 받아, 그에 대한 설명을 생성:  
       ```
       생성된 설명: "오른쪽을 향한 백조 모양"
       ```

3. **성능 평가**
   - **청취자 역할**: 모델이 "C"를 정확히 선택했으므로 이해 정확도 100% 기록.  
   - **발화자 역할**: 인간 청취자가 모델의 설명을 보고 "C"를 정확히 선택했으므로 생성 정확도 100% 기록.  
   - **MAUVE 점수**: 모델이 생성한 설명이 인간 생성 설명과 얼마나 유사한지를 측정하는 지표. 예를 들어, MAUVE 점수 = 0.85.

---


#### Example Scenario: Model Operation in a Reference Game

1. **Dataset Example**
   - **Context Image Set**: The following abstract tangram images are presented:
     ```
     A, B, C, D, E, F, G, H, I, J
     ```
   - **Target Image**: "C"
   - **User Description**: "A swan facing right."

2. **Model Operation**
   - **Listener Role (Comprehension)**  
     - The model takes the description ("A swan facing right") as input and predicts the target image "C."  
     - Example output probability distribution:  
       ```
       P(C|context, description) = 0.85
       P(A|context, description) = 0.05
       P(E|context, description) = 0.03
       ...
       ```
     - The model selects "C" as it has the highest probability.

   - **Speaker Role (Generation)**  
     - The model takes the target image "C" as input and generates the following description:  
       ```
       Generated Description: "A swan facing right."
       ```

3. **Performance Evaluation**
   - **Listener Role**: The model correctly identifies "C," recording 100% comprehension accuracy.  
   - **Speaker Role**: A human listener correctly identifies "C" based on the model's description, achieving 100% generation accuracy.  
   - **MAUVE Score**: Measures the similarity between the model's generated description and human-generated descriptions. For example, MAUVE score = 0.85.


<br/>  
# 요약   




이 연구는 언어 이해와 생성을 결합하여 지속적 학습을 수행하는 시스템을 개발했으며, IDEFICS2-8B 모델을 사용하여 인간 사용자와의 상호작용에서 피드백 데이터를 학습에 활용했습니다. 결과적으로 이해와 생성 정확도가 각각 19.48%와 26.07% 향상되었으며, 데이터 효율성을 크게 증대시켰습니다. 예를 들어, "오른쪽을 향한 백조 모양"이라는 설명에서 목표 이미지 "C"를 정확히 선택하고 생성하는 성능을 보여주며, 모델의 MAUVE 점수는 인간 언어와의 유사성을 반영해 0.85를 기록했습니다.

---


This study developed a system that integrates language comprehension and generation for continual learning, leveraging the IDEFICS2-8B model to learn from feedback obtained during interactions with human users. The results showed a 19.48% and 26.07% improvement in comprehension and generation accuracy, respectively, while significantly enhancing data efficiency. For example, when given the description "A swan facing right," the model accurately identified and generated the target image "C," achieving a MAUVE score of 0.85, reflecting its similarity to human language.


<br/>  
# 기타  



1. **Figure 1: 참조 게임 상호작용 시나리오**  
   - 이 그림은 참조 게임의 구조를 보여줍니다.  
   - 발화자는 목표 이미지를 설명하여 청취자가 올바른 이미지를 선택하도록 유도합니다.  
   - 성공적인 게임은 청취자가 올바른 이미지를 선택할 때 이루어집니다.  

2. **Figure 2: 지속적 학습 시나리오와 결합된 이해 및 생성**  
   - 이 그림은 학습 과정에서 모델이 청취자와 발화자 역할을 번갈아 수행하는 구조를 보여줍니다.  
   - 모델은 상호작용에서 피드백을 수집하고 이를 바탕으로 학습을 반복합니다.  
   - 결합된 접근법은 피드백 데이터를 양쪽 역할에 활용하여 성능을 지속적으로 개선합니다.

3. **Figure 3: 이해 및 생성 성능 비교**  
   - 모델 성능이 라운드별로 어떻게 개선되었는지 시각화한 그래프입니다.  
   - FULL 모델이 NO-DS, NO-JI, BASELINE 모델보다 지속적으로 높은 성능을 보여줍니다.  
   - FULL 모델은 4라운드 동안 이해 정확도는 약 72.79%, 생성 정확도는 약 78.07%에 도달했습니다.

4. **Figure 5: 언어 분석 결과**  
   - 언어 길이, 어휘 다양성, 새로운 단어 추가 수, MAUVE 점수 등 모델 언어의 변화를 보여줍니다.  
   - FULL 모델은 언어 길이와 어휘 다양성에서 인간 언어와 유사한 패턴을 보이며, MAUVE 점수가 가장 높았습니다.

---


1. **Figure 1: Reference Game Interaction Scenario**  
   - This figure illustrates the structure of a reference game.  
   - The speaker describes the target image to help the listener select the correct one.  
   - A successful game occurs when the listener selects the correct image.

2. **Figure 2: Continual Learning Scenario with Coupled Comprehension and Generation**  
   - This figure depicts how the model alternates between listener and speaker roles during training.  
   - Feedback collected from interactions is used for iterative learning.  
   - The coupled approach utilizes feedback data for both roles to continuously improve performance.

3. **Figure 3: Comprehension and Generation Performance Comparison**  
   - This graph visualizes how model performance improves across rounds.  
   - The FULL model consistently outperforms NO-DS, NO-JI, and BASELINE models.  
   - By the 4th round, the FULL model achieves approximately 72.79% comprehension accuracy and 78.07% generation accuracy.

4. **Figure 5: Language Analysis Results**  
   - This figure shows changes in model language, including utterance length, vocabulary diversity, new word additions, and MAUVE scores.  
   - The FULL model demonstrates patterns similar to human language in length and vocabulary diversity, with the highest MAUVE score.


<br/>
# refer format:     


@inproceedings{Gul2024COGEN,
  title = {COGEN: Learning from Feedback with Coupled Comprehension and Generation},
  author = {Mustafa Omer Gul and Yoav Artzi},
  booktitle = {Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages = {12966--12982},
  year = {2024},
  month = {November},
  organization = {Association for Computational Linguistics},
  address = {Singapore},
  url = {https://github.com/lil-lab/cogen}
}



Mustafa Omer Gul and Yoav Artzi. "COGEN: Learning from Feedback with Coupled Comprehension and Generation." In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 12966–12982. Singapore: Association for Computational Linguistics, 2024. https://github.com/lil-lab/cogen.

	

