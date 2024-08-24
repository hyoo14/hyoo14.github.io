---
layout: post
title:  "[2020]Reasoning about Goals, Steps, and Temporal Ordering with WikiHow"  
date:   2024-08-24 18:49:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

이 연구는 WikiHow에서 추출한 데이터셋을 기반으로, BERT, XLNet, RoBERTa, GPT-2와 같은 AI 모델을 활용해 목표-단계 및 단계-단계 시간적 관계를 추론하는 방법을 제안

짧은 요약(Abstract) :    

이 논문의 abstract 부분은 WikiHow라는 사이트의 지식을 활용하여 두 가지 관계에 대한 추론 작업을 제안합니다. 첫 번째는 **목표-단계 관계**로, 특정 목표를 달성하기 위해 필요한 단계들을 이해하는 것이고, 두 번째는 **단계-단계 시간적 관계**로, 특정 절차에서 단계들이 어떤 순서로 진행되는지를 이해하는 것입니다. 연구진은 WikiHow에서 데이터를 수집하여 목표와 단계 간의 관계를 추론하는 데이터셋을 만들었으며, 이 데이터셋은 인간 상식에 기반한 추론을 위한 신뢰할 수 있는 기준을 제공합니다. 실험 결과, 최신 AI 모델들이 인간 성능보다 10~20% 낮은 성과를 보였고, 이 데이터셋을 학습한 모델은 다른 도메인의 작업에서도 성능이 크게 향상되었습니다.

쉽게 설명하면, WikiHow라는 사이트에 있는 ‘요가하는 방법’ 같은 설명을 바탕으로, ‘요가 매트 구입’이라는 단계가 ‘요가 배우기’ 전에 일어나는 것처럼 목표와 단계를 추론하는 AI 시스템을 만들고, 이를 다른 작업에 적용했을 때도 효과적이라는 것을 보여준 연구입니다.

---

The abstract of this paper proposes a suite of reasoning tasks based on two types of relationships between procedural events. The first is the **Goal-Step relationship**, which involves understanding the steps needed to achieve a particular goal, and the second is the **Step-Step Temporal relationship**, which involves understanding the sequence in which steps typically occur in a procedure. The researchers collected data from WikiHow to build a dataset for inferring these relationships, creating a reliable benchmark for commonsense reasoning. Experimental results show that state-of-the-art AI models perform 10-20% worse than humans, but models trained on this dataset significantly improved performance in other out-of-domain tasks.

In simpler terms, the study built an AI system that can understand the sequence of steps, such as knowing that "buying a yoga mat" usually happens before "learning yoga poses" in a tutorial, and the system was proven to be effective in various tasks.

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


논문의 방법론에서는 WikiHow 사이트에서 데이터를 자동으로 추출하고, 이를 학습 데이터셋으로 사용하여 AI 모델들이 목표-단계(Goal-Step) 관계와 단계-단계 시간적 관계(Step-Step Temporal relationship)를 추론하도록 설정했습니다.

1. **데이터 수집**: 
   - WikiHow 사이트에서 11만 개 이상의 how-to 기사를 크롤링하여 데이터셋을 만들었습니다. 각 기사에는 **목표(Goal)**와 그 목표를 달성하기 위한 **단계(Step)**들이 포함되어 있으며, 이 정보가 핵심 데이터로 사용되었습니다.

2. **데이터셋 구성**:
   - 각 기사의 제목을 목표로 정의하고, 해당 기사의 각 절 제목을 단계로 간주했습니다. 이를 기반으로 세 가지 추론 작업을 설계했습니다:
     1. **단계 추론(Step Inference)**: 주어진 목표에 맞는 단계를 선택하는 작업.
     2. **목표 추론(Goal Inference)**: 주어진 단계에 해당하는 목표를 추론하는 작업.
     3. **단계 순서 추론(Step Ordering)**: 두 단계의 시간적 순서를 결정하는 작업.

3. **데이터 생성 방법**:
   - 저자들은 **부정 샘플링(Negative Sampling)** 전략을 사용하여 자동으로 학습 데이터를 생성했습니다. 부정 샘플링이란, 주어진 목표나 단계와 유사하지만 틀린 답이 되도록 부정 후보를 선택하는 방법입니다. 이를 위해 **BERT 임베딩**을 사용하여 단계 간의 **코사인 유사도**를 계산하고, 유사한 단계를 부정 후보로 선택했습니다.
   - 자동 생성된 데이터셋에는 일정량의 노이즈가 있을 수 있기 때문에, 일부 데이터는 인간 평가자가 검증하여 신뢰도를 높였습니다.

4. **모델 사용**:
   - 저자들은 **BERT**, **XLNet**, **RoBERTa**, **GPT-2** 같은 사전 학습된 트랜스포머 모델을 사용하여 목표-단계 및 단계-단계 시간적 관계를 학습했습니다. 모델들은 이 데이터셋을 기반으로 학습(fine-tuning)되었고, 성능 평가가 진행되었습니다.
   - 실험 결과, 이 모델들은 인간 성능보다 10~20% 낮은 성과를 보였지만, 다른 도메인의 작업에 대해 매우 높은 전이 학습 성능을 보였습니다.

결론적으로, 저자들은 WikiHow 데이터로부터 자동으로 데이터를 추출하고, 이를 AI 모델 학습에 사용하여 목표-단계 및 단계-단계 시간적 관계를 추론하는 방법을 제안했습니다.

---

**Methodology (English version)**:

The methodology of this paper involves automatically extracting data from WikiHow and using it as a dataset to train AI models to infer Goal-Step and Step-Step Temporal relationships.

1. **Data Collection**: 
   - The researchers crawled over 110,000 how-to articles from the WikiHow website. Each article contains a **goal** and the corresponding **steps** to achieve that goal, and this information was used as the core dataset.

2. **Dataset Construction**:
   - The title of each article was defined as the goal, and the section headers were considered as steps. Based on this, the authors designed three inference tasks:
     1. **Step Inference**: Selecting the appropriate step given a goal.
     2. **Goal Inference**: Inferring the correct goal from a given step.
     3. **Step Ordering**: Determining the temporal order between two steps.

3. **Data Generation Method**:
   - The authors employed a **Negative Sampling** strategy to automatically generate the dataset. Negative sampling involves selecting distractor steps or goals that are semantically similar but incorrect. BERT embeddings were used to calculate **cosine similarity** between steps, and highly similar steps were selected as negative candidates.
   - Since automatically generated datasets may include noise, some portions of the dataset were verified by human annotators to ensure quality.

4. **Model Usage**:
   - Pretrained transformer models such as **BERT**, **XLNet**, **RoBERTa**, and **GPT-2** were fine-tuned on the dataset to learn the Goal-Step and Step-Step Temporal relationships. These models were then evaluated for their performance.
   - While the models performed 10-20% lower than human accuracy, they demonstrated strong transfer learning capabilities to other domains.

In conclusion, the researchers proposed a method for automatically extracting data from WikiHow and training AI models to infer Goal-Step and Step-Step Temporal relationships.




<br/>
# Results  




1. **모델 성능 비교**:
   - **BERT**, **XLNet**, **RoBERTa**, **GPT-2** 등의 사전 학습된 트랜스포머 모델들을 WikiHow 데이터셋으로 학습시켰습니다.
   - 각 모델은 **목표-단계(Goal-Step)** 관계와 **단계-단계 시간적 관계(Step-Step Temporal)**를 추론하는 작업에서 성능을 비교했으며, **RoBERTa** 모델이 가장 높은 정확도를 보였습니다. **BERT**, **XLNet**, **GPT-2**가 그 뒤를 이었습니다.
   - 인간 성능(96.5-98%)에 비해, **RoBERTa**는 약 88%의 정확도를 기록하여 인간과 AI 모델 간에는 약 10-20%의 성능 차이가 있었습니다.

2. **전이 학습(Transfer Learning) 결과**:
   - WikiHow 데이터셋을 학습한 모델들은 다른 도메인 작업에서도 성능을 크게 향상시켰습니다.
     - **SWAG**(비디오 캡션 작업)에서는 24%의 성능 향상을 보였습니다.
     - **Snips**(대화 시스템에서의 의도 탐지 작업)에서는 78%의 무학습(zero-shot) 성능을 기록했습니다.
     - **Story Cloze Test**(이야기 이해 작업)에서는 64%의 성능 향상을 기록했습니다.
   - 이처럼 WikiHow 데이터셋을 기반으로 학습한 모델은 다른 도메인의 작업에서도 매우 높은 전이 학습 성능을 보여주었습니다.

3. **모델별 성능 차이**:
   - **RoBERTa**가 가장 높은 성능을 보였으며, **BERT**와 **XLNet**이 그 뒤를 이었습니다. **GPT-2**는 상대적으로 낮은 성능을 보였지만, 여전히 목표-단계 추론과 단계 순서 추론에서 유의미한 성과를 냈습니다.

---



1. **Model Performance Comparison**:
   - Pretrained transformer models such as **BERT**, **XLNet**, **RoBERTa**, and **GPT-2** were fine-tuned on the WikiHow dataset.
   - When comparing the accuracy in inferring **Goal-Step** and **Step-Step Temporal relationships**, **RoBERTa** showed the best performance, followed by **BERT**, **XLNet**, and **GPT-2**.
   - While human performance ranged between 96.5% and 98%, **RoBERTa** achieved around 88% accuracy, leaving a gap of 10-20% between human and model performance.

2. **Transfer Learning Results**:
   - The models trained on the WikiHow dataset showed significant improvements in other domain tasks:
     - On the **SWAG** (video captioning) task, a 24% improvement in accuracy was achieved.
     - On the **Snips** (intent detection) task, a 78% zero-shot performance was recorded.
     - On the **Story Cloze Test** (story understanding), a 64% improvement was observed.
   - These results demonstrate the effectiveness of using WikiHow-based tasks for transfer learning, especially in low-resource scenarios.

3. **Model Performance Differences**:
   - **RoBERTa** outperformed other models, with **BERT** and **XLNet** following closely. While **GPT-2** showed relatively lower performance, it still performed well in goal-step inference and step ordering tasks.

Hope this helps! Let me know if you need further clarification.

<br/>
# 예시  



### 1. **단계 추론 작업(Step Inference Task)**:
   **예시**:
   - **목표**: "코로나바이러스 예방하기(Prevent Coronavirus)"
   - **후보 단계**:
     - A. 손을 씻으세요 (정답)
     - B. 고양이를 씻기세요
     - C. 손뼉을 치세요
     - D. 단백질을 드세요

   **결과**:
   - **RoBERTa**: 정답인 "손을 씻으세요"를 정확히 선택하여 가장 높은 성능을 보였습니다.
   - **BERT**와 **XLNet**: 유사한 후보를 선택하긴 했지만, 일부 혼동이 있는 경우가 있었습니다.
   - **GPT-2**: 상대적으로 낮은 정확도를 기록했으며, 일부 부정확한 후보를 선택하는 경향을 보였습니다.

### 2. **목표 추론 작업(Goal Inference Task)**:
   **예시**:
   - **단계**: "립스틱 색상을 고르세요(choose a color of lipstick)"
   - **후보 목표**:
     - A. 핑크색 입술 만들기 (정답)
     - B. 사람의 입술 읽기
     - C. 립싱크하기
     - D. 입술 그리기

   **결과**:
   - **RoBERTa**: 주어진 단계를 보고 정확하게 "핑크색 입술 만들기"라는 목표를 추론해냈습니다.
   - **BERT**와 **XLNet**: 대체로 정확하게 목표를 예측했으나, 때때로 유사한 의미의 잘못된 목표를 선택하기도 했습니다.
   - **GPT-2**: 목표 추론 작업에서 다른 모델보다 낮은 성능을 보였으며, 정답률이 떨어졌습니다.

### 3. **단계 순서 추론 작업(Step Ordering Task)**:
   **예시**:
   - **목표**: "은 세척하기(Clean Silver)"
   - **후보 단계**:
     - A. 은을 건조시키세요
     - B. 은을 손세척하세요 (정답)

   **결과**:
   - **RoBERTa**: 정확히 "은을 손세척하세요"가 "은을 건조시키세요"보다 먼저 일어난다는 것을 추론했습니다.
   - **BERT**와 **XLNet**: 유사한 성능을 보였으나, 일부 경우에서 시간적 순서에 혼동을 일으켰습니다.
   - **GPT-2**: 단계 순서 추론 작업에서 다른 모델에 비해 낮은 성능을 보였습니다.

### 성능 비교:
- **RoBERTa**는 각 작업에서 가장 높은 정확도를 기록하며, 주어진 목표와 단계를 정확하게 예측했습니다.
- **BERT**와 **XLNet**은 비슷한 성능을 보였으며, 일부 작업에서는 혼동을 일으킬 수 있는 유사한 답변을 선택하는 경향이 있었습니다.
- **GPT-2**는 상대적으로 낮은 성능을 보였으며, 특히 시간적 순서 추론 작업에서 어려움을 겪었습니다.

### 전이 학습 예시:
- **SWAG** 작업에서는, **RoBERTa**가 학습 없이도 24% 성능 향상을 보였으며, 다른 모델들 역시 WikiHow 데이터셋으로 학습된 후 성능이 크게 개선되었습니다.
- **Snips** 작업에서 **RoBERTa**는 78%의 무학습(zero-shot) 성능을 기록하며 거의 완벽한 성과를 보였습니다.

---



### 1. **Step Inference Task**:
   **Example**:
   - **Goal**: "Prevent Coronavirus"
   - **Candidate Steps**:
     - A. Wash your hands (Correct)
     - B. Wash your cat
     - C. Clap your hands
     - D. Eat your protein

   **Results**:
   - **RoBERTa**: Correctly selected "Wash your hands" as the right step, showing the best performance.
   - **BERT** and **XLNet**: Generally performed well but occasionally selected incorrect but semantically similar options.
   - **GPT-2**: Performed relatively poorly, selecting incorrect candidates more frequently.

### 2. **Goal Inference Task**:
   **Example**:
   - **Step**: "Choose a color of lipstick"
   - **Candidate Goals**:
     - A. Get Pink Lips (Correct)
     - B. Read One’s Lips
     - C. Lip Sync
     - D. Draw Lips

   **Results**:
   - **RoBERTa**: Correctly inferred that the step corresponds to the goal "Get Pink Lips."
   - **BERT** and **XLNet**: Performed similarly well but occasionally chose semantically incorrect goals.
   - **GPT-2**: Performed worse, struggling to correctly infer the goal from the step.

### 3. **Step Ordering Task**:
   **Example**:
   - **Goal**: "Clean Silver"
   - **Candidate Steps**:
     - A. Dry the silver
     - B. Handwash the silver (Correct)

   **Results**:
   - **RoBERTa**: Correctly inferred that "Handwash the silver" comes before "Dry the silver."
   - **BERT** and **XLNet**: Performed well but occasionally confused the temporal order.
   - **GPT-2**: Struggled the most with step ordering tasks, with lower accuracy.

### Performance Comparison:
- **RoBERTa** consistently outperformed other models, making accurate predictions in each task.
- **BERT** and **XLNet** performed similarly but sometimes struggled with semantically similar distractors.
- **GPT-2** performed the worst, especially in the step ordering task.

### Transfer Learning Example:
- In the **SWAG** task, **RoBERTa** showed a 24% improvement in zero-shot performance, and other models also improved significantly after being trained on the WikiHow dataset.
- In the **Snips** task, **RoBERTa** achieved an impressive 78% zero-shot performance, nearly perfect for intent detection.

These examples demonstrate the effectiveness of each model and how the WikiHow dataset improved their performance across various tasks.

<br/>  
# 요약 

이 연구는 WikiHow에서 수집한 데이터를 기반으로, 목표-단계 관계와 단계-단계 시간적 관계를 추론하는 AI 모델을 학습시키는 방법을 제안했습니다. 저자들은 자동으로 데이터를 생성하고 부정 샘플링을 통해 학습 데이터를 구축했으며, 이를 BERT, XLNet, RoBERTa, GPT-2 같은 트랜스포머 모델로 학습시켰습니다. RoBERTa 모델이 가장 높은 성능을 보였으며, 인간 성능과 약 10~20% 차이가 있었습니다. 전이 학습 실험에서는 다른 도메인 작업에서도 성능이 크게 향상되었습니다. 특히 단계 추론, 목표 추론, 단계 순서 추론 작업에서 RoBERTa는 뛰어난 성능을 기록했습니다.

---

This study proposes a method to train AI models to infer Goal-Step and Step-Step Temporal relationships based on data collected from WikiHow. The authors automatically generated the dataset using negative sampling and fine-tuned transformer models like BERT, XLNet, RoBERTa, and GPT-2. The RoBERTa model showed the best performance, with a 10-20% gap compared to human accuracy. In transfer learning experiments, the models significantly improved performance in other domain tasks. RoBERTa particularly excelled in step inference, goal inference, and step ordering tasks.


# 기타  


<br/>
# refer format:     


@inproceedings{zhang2020reasoning,
  title={Reasoning about Goals, Steps, and Temporal Ordering with WikiHow},
  author={Zhang, Li and Lyu, Qing and Callison-Burch, Chris},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={4630--4639},
  year={2020},
  organization={Association for Computational Linguistics}
}



Zhang Li, Qing Lyu, and Chris Callison-Burch. "Reasoning about Goals, Steps, and Temporal Ordering with WikiHow." In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 4630-4639. Association for Computational Linguistics, 2020.