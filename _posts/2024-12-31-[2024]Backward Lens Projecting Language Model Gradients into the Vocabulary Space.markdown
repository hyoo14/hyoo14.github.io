---
layout: post
title:  "[2024]Backward Lens: Projecting Language Model Gradients into the Vocabulary Space"  
date:   2024-12-31 02:30:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 



짧은 요약(Abstract) :    




이 연구는 Transformer 기반 언어 모델(Transformer-based Language Models)의 역전파(Backward Pass)와 기울기(Gradient)에 대해 연구하며, 이를 어휘 공간(Vocabulary Space)으로 투영하는 방법론을 제안합니다. 기존의 해석 가능성 연구는 주로 순전파(Forward Pass)로부터 얻은 가중치와 은닉 상태를 어휘 공간으로 투영하여 언어 모델 내 정보 흐름을 분석했습니다. 본 연구는 이를 확장하여 역전파에서의 기울기를 어휘 항목으로 투영하고, 새로운 정보가 모델의 뉴런에 저장되는 메커니즘을 탐구합니다. 또한, 기울기 행렬의 저랭크 성질(Low-Rank)을 활용해 모델의 학습 과정을 "Imprint"와 "Shift"라는 두 가지 단계로 정의하고, 이를 기반으로 모델의 내부 지식을 편집하는 새로운 방법론을 제안합니다. 연구는 기울기 행렬의 분석, 기울기의 어휘 항목 투영, 그리고 학습 메커니즘의 해석에 기여하며, 제안된 방법론이 기존의 최첨단 지식 편집 기법들과 유사한 성능을 보이는 것을 확인했습니다.

---


This study explores the backward pass and gradients in Transformer-based Language Models, proposing a methodology to project them into the vocabulary space. While prior interpretability studies primarily focus on projecting weights and hidden states from the forward pass into the vocabulary space, this work extends this approach to gradients from the backward pass. It investigates how new information is stored in the neurons of the model, utilizing the low-rank nature of gradient matrices. The study identifies a two-phase mechanism, "Imprint" and "Shift," which describes how knowledge is stored in feed-forward layers during training. A novel method for editing internal model knowledge is introduced, demonstrating performance comparable to state-of-the-art editing techniques. Contributions include the analysis of gradient ranks, the projection of gradients into tokens, and the interpretability of learning mechanisms in language models.



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



이 논문에서는 Transformer 기반 언어 모델(GPT-2 및 Llama2-7B)을 대상으로 역전파(backward pass) 과정에서 생성되는 기울기(gradient)를 어휘 공간(vocabulary space)에 투영하는 새로운 방법론을 제안했습니다. 이 연구의 핵심은 모델의 기울기 행렬이 저랭크(low-rank) 구조를 가진다는 점을 활용하여 학습 과정을 분석하고, 이를 통해 모델이 새로운 정보를 저장하고 학습하는 메커니즘을 "Imprint(각인)"와 "Shift(변환)"라는 두 가지 단계로 정의한 것입니다.

#### 주요 내용:
1. **사용 모델**: 
   - GPT-2 (Radford et al., 2019)
   - Llama2-7B (Touvron et al., 2023)

2. **방법론의 핵심**:
   - 기울기 행렬을 구성하는 입력 벡터와 기울기 백터(VJP; Vector-Jacobian Product)를 기반으로, 언어 모델이 새로운 정보를 어떻게 저장하는지 설명.
   - "Imprint" 단계는 모델의 가중치가 입력 벡터를 통해 업데이트되는 과정을 나타냄.
   - "Shift" 단계는 목표 어휘 항목(target token)의 임베딩을 통해 모델 출력이 변환되는 과정을 설명.

3. **트레이닝 데이터**:
   - CounterFact 데이터셋(Meng et al., 2022)을 사용하여, 특정 프롬프트에 대한 목표 어휘 항목(target token)을 학습하도록 모델을 수정.
   - 실험에서 100개의 샘플 프롬프트를 무작위로 선택하고, 각 프롬프트에 대해 역전파를 단일 실행(single backward pass)하여 분석.

4. **실험 설계**:
   - 모델의 기울기 행렬을 투영(Logit Lens)하여 기울기의 의미를 해석하고, "Imprint"와 "Shift"의 영향을 평가.
   - 기존의 모델 편집 방법(MEND, ROME 등)과 비교하여 성능을 평가.

---



This paper introduces a novel methodology to project gradients from the backward pass of Transformer-based language models (e.g., GPT-2 and Llama2-7B) into the vocabulary space. The study leverages the low-rank structure of gradient matrices to analyze the learning process and identifies a two-phase mechanism, termed "Imprint" and "Shift," that describes how new information is stored in the model.

#### Key Details:
1. **Models Used**:
   - GPT-2 (Radford et al., 2019)
   - Llama2-7B (Touvron et al., 2023)

2. **Core Methodology**:
   - Analyze gradients formed by the input vectors and Vector-Jacobian Products (VJPs) to explain how language models store new information.
   - The "Imprint" phase describes how the model weights are updated based on the input vectors.
   - The "Shift" phase explains how the model's output is adjusted toward the target token embedding.

3. **Training Data**:
   - CounterFact dataset (Meng et al., 2022) was used to train the model for specific prompts and their corresponding target tokens.
   - Experiments involved selecting 100 random prompt-target pairs and performing a single backward pass for analysis.

4. **Experimental Design**:
   - Gradient matrices were projected using the Logit Lens to interpret their meaning and assess the effects of the "Imprint" and "Shift" mechanisms.
   - The proposed methodology was compared against existing model editing techniques like MEND and ROME to evaluate its effectiveness.



   
 
<br/>
# Results  




이 논문에서는 **CounterFact 데이터셋**을 사용하여 GPT-2와 Llama2-7B 모델에 대해 실험을 수행하였습니다. 연구는 기존의 최첨단 모델 편집 기법(MEND, ROME, MEMIT)과 비교하여 제안된 방법론의 성능을 평가했습니다.

#### 주요 결과:
1. **테스트 데이터**:
   - CounterFact 데이터셋의 1000개의 샘플을 사용해 모델 편집 성능을 평가.
   - 주어진 프롬프트(prompt)에 대해 원하는 목표 단어(target word)를 모델이 정확히 생성하는지 확인.

2. **비교 모델**:
   - MEND, ROME, MEMIT 등 현재 사용되는 최첨단 모델 편집 기법과 성능 비교.
   - 편집 후 일반화(generalization)와 특정성(specificity) 측면에서 성능 분석.

3. **성능 향상**:
   - 제안된 방법론은 단일 프롬프트 편집에서 ROME 및 MEMIT과 유사한 수준의 정확도를 달성 (99.4% 성공률).
   - ROME과 비교했을 때 일반화 성능(즉, 편집된 문장에서 파생된 표현에 대한 정확도)은 약간 낮았지만, 계산 복잡도는 현저히 낮음.
   - 기존 기법과 달리 역전파를 사용하지 않고 단일 순전파(forward pass)만으로 모델 편집을 수행 가능.

4. **주요 발견**:
   - "Imprint"와 "Shift" 메커니즘이 모델의 편집 효율성을 향상시킴.
   - 특히, 제안된 방법은 더 낮은 계산 복잡도로도 높은 편집 성공률을 달성하여 모델 튜닝을 위한 실용적인 대안을 제시.

---



The study conducted experiments using the **CounterFact dataset** to evaluate the performance of GPT-2 and Llama2-7B models. The proposed methodology was compared with state-of-the-art model editing techniques such as MEND, ROME, and MEMIT.

#### Key Results:
1. **Test Data**:
   - Evaluated on 1000 samples from the CounterFact dataset.
   - Assessed the model's ability to generate the desired target word for given prompts.

2. **Comparison Models**:
   - Benchmarked against MEND, ROME, and MEMIT.
   - Performance analyzed in terms of editing efficacy, generalization (e.g., accuracy on paraphrases), and specificity.

3. **Performance Improvements**:
   - Achieved a success rate of 99.4% for single-prompt edits, comparable to ROME and MEMIT.
   - While the generalization performance was slightly lower than ROME, the computational complexity was significantly reduced.
   - Unlike existing methods, the proposed approach successfully edited models using only a single forward pass, eliminating the need for a full backward pass.

4. **Key Insights**:
   - The "Imprint" and "Shift" mechanisms enhanced editing efficiency.
   - The proposed method offers a practical alternative for model tuning, achieving high success rates with reduced computational overhead.



<br/>
# 예제  




논문에서 다룬 구체적인 예시는 **CounterFact 데이터셋**을 사용하여 특정 프롬프트와 목표 단어를 통해 모델의 편집 결과를 평가한 사례입니다.

#### 예제 1: "Obama grew up in" 프롬프트
- **입력 프롬프트**: "Obama grew up in"
- **목표 단어(Target)**: "Paris"
- **결과**:
  - 기존 모델(GPT-2)은 "Obama grew up in Hawaii"를 출력하며 "Hawaii"가 높은 확률 점수를 가짐.
  - 편집 후:
    - 모델은 "Obama grew up in Paris"를 정확히 예측.
    - 편집된 모델에서 "Paris"는 **Logit Lens 점수**에서 가장 높은 확률을 가짐.
    - 하위 계층(early layers)에서는 "Paris"와 관련된 단어(예: "Macron", "France")도 높은 점수를 보임.

#### 예제 2: "Jack Dorsey founded" 프롬프트
- **입력 프롬프트**: "Jack Dorsey founded"
- **목표 단어(Target)**: "IBM"
- **결과**:
  - 기존 모델은 "Twitter"를 출력하며 "Twitter"가 가장 높은 확률을 가짐.
  - 편집 후:
    - 모델은 "Jack Dorsey founded IBM"으로 수정됨.
    - "IBM"은 **Logit Lens 투영**에서 높은 순위를 보이며, 편집된 모델이 성공적으로 새로운 지식을 반영했음을 확인.

#### 스코어:
- **편집 성공률**: 99.4%
- **일반화 성능**: Paraphrases(재구성된 문장)에서 41.6% 정확도 (ROME의 71.9% 대비 낮음).

---



The paper provides concrete examples using the **CounterFact dataset** to evaluate the model's editing results with specific prompts and target words.

#### Example 1: "Obama grew up in" Prompt
- **Input Prompt**: "Obama grew up in"
- **Target Word**: "Paris"
- **Results**:
  - Before editing:
    - The base model (GPT-2) outputs "Obama grew up in Hawaii," with "Hawaii" having the highest probability score.
  - After editing:
    - The model correctly predicts "Obama grew up in Paris."
    - In the edited model, "Paris" has the highest probability according to **Logit Lens scores**.
    - Early layers also show high scores for related words like "Macron" and "France."

#### Example 2: "Jack Dorsey founded" Prompt
- **Input Prompt**: "Jack Dorsey founded"
- **Target Word**: "IBM"
- **Results**:
  - Before editing:
    - The base model outputs "Twitter," with "Twitter" being the most probable word.
  - After editing:
    - The model outputs "Jack Dorsey founded IBM," reflecting the new knowledge.
    - "IBM" ranks highly in the **Logit Lens projection**, confirming successful knowledge editing.

#### Scores:
- **Editing Success Rate**: 99.4%
- **Generalization Performance**: 41.6% accuracy on paraphrased sentences (lower than ROME's 71.9%).


<br/>  
# 요약   




이 연구는 Transformer 기반 언어 모델(GPT-2, Llama2-7B)의 역전파 기울기를 어휘 공간으로 투영하여 학습 메커니즘을 분석하고, 이를 "Imprint(각인)"와 "Shift(변환)"라는 두 단계로 설명하는 새로운 방법론을 제안했습니다. CounterFact 데이터셋을 사용하여 특정 프롬프트(예: "Obama grew up in")를 목표 단어("Paris")로 편집한 결과, 편집 성공률은 99.4%로 높은 정확도를 기록했으며, 모델은 편집 후 "Paris"를 정확히 예측했습니다. 본 연구는 단일 순전파만으로 편집이 가능하며, 기존 기법(ROME, MEMIT 등) 대비 계산 복잡도가 낮은 실용적인 대안을 제시했습니다.

---


This study introduces a novel methodology for analyzing the learning mechanism of Transformer-based language models (GPT-2, Llama2-7B) by projecting backward gradients into the vocabulary space, explaining the process through two phases: "Imprint" and "Shift." Using the CounterFact dataset, the method successfully edited specific prompts (e.g., "Obama grew up in") to target words ("Paris"), achieving a high success rate of 99.4% with accurate predictions. The proposed approach, requiring only a single forward pass, provides a practical alternative with reduced computational complexity compared to existing methods like ROME and MEMIT.

<br/>  
# 기타 




1. **Figure 1**: 
   - 모델의 "Imprint"와 "Shift" 메커니즘을 시각적으로 설명합니다. 예를 들어, "Lionel Messi plays for"라는 프롬프트에 대해 "Paris"를 목표로 설정했을 때, 역전파 과정에서 각 계층의 기울기가 새로운 목표 단어("team")와 관련된 정보를 모델 가중치에 어떻게 추가하는지 보여줍니다.

2. **Figure 3**: 
   - 프롬프트 길이에 따라 기울기 행렬의 랭크가 계층별로 어떻게 변화하는지 보여줍니다. 모든 계층에서 기울기 랭크는 일반적으로 프롬프트 길이와 동일하며, 마지막 계층에서는 항상 랭크가 1로 나타나는 특징을 시각화합니다.

3. **Figure 4**: 
   - GPT-2 모델의 역전파 기울기(VJP)를 어휘 공간으로 투영(Logit Lens)한 결과를 나타냅니다. (a) "Obama grew up in" → "Paris", (b) "Jack Dorsey founded" → "IBM" 사례에서 각 계층과 토큰의 관계를 시각화하여, 목표 단어("Paris", "IBM")가 높은 관련성을 갖는 계층과 토큰을 강조합니다.

4. **Figure 5**: 
   - GPT-2 모델의 역전파 기울기 크기(norm)가 계층과 프롬프트 세그먼트(토큰)별로 어떻게 분포되는지 나타냅니다. 대부분의 업데이트는 프롬프트의 주제 토큰(예: "Obama" 또는 "Jack Dorsey") 및 마지막 토큰에서 발생함을 보여줍니다.

5. **Figure 6**: 
   - 각 계층의 기울기(VJP)가 목표 단어를 어휘 공간에서 얼마나 높은 순위로 투영하는지를 보여줍니다. 마지막 프롬프트 토큰의 기울기(VJP)는 목표 단어를 가장 낮은 확률로 투영하는 경향이 있지만, 이는 기울기 크기(norm)가 낮아지는 초기 계층에서 두드러집니다.

---



1. **Figure 1**: 
   - Visualizes the "Imprint" and "Shift" mechanisms of the model. For example, given the prompt "Lionel Messi plays for" and the target "Paris," it illustrates how gradients in each layer embed new target-related information ("team") into the model weights during the backward pass.

2. **Figure 3**: 
   - Shows how the rank of gradient matrices changes across layers relative to the prompt length. In all layers, the gradient rank generally equals the prompt length, while the last layer consistently exhibits a rank of 1.

3. **Figure 4**: 
   - Depicts the projection of backward gradients (VJP) into the vocabulary space using Logit Lens. In examples like (a) "Obama grew up in" → "Paris" and (b) "Jack Dorsey founded" → "IBM," it highlights layers and tokens that strongly relate to the target words ("Paris" and "IBM").

4. **Figure 5**: 
   - Illustrates the distribution of gradient norms across layers and prompt segments (tokens) in the GPT-2 model. Most updates occur around the subject tokens (e.g., "Obama" or "Jack Dorsey") and the final token in the prompt.

5. **Figure 6**: 
   - Displays how each layer’s gradients (VJP) rank the target word in the vocabulary space. Gradients for the final prompt token tend to rank the target word lowest, a phenomenon amplified in early layers due to diminishing gradient norms. 


<br/>
# refer format:     


@inproceedings{Katz2024BackwardLens,
  title        = {Backward Lens: Projecting Language Model Gradients into the Vocabulary Space},
  author       = {Shahar Katz and Yonatan Belinkov and Mor Geva and Lior Wolf},
  booktitle    = {Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  pages        = {2390--2422},
  year         = {2024},
  organization = {Association for Computational Linguistics},
  address      = {Singapore},
  month        = {November},
  url          = {https://github.com/shacharKZ/BackwardLens},
  abstract     = {This work investigates the backward pass and gradients in Transformer-based Language Models by projecting them into the vocabulary space. A novel method for analyzing and editing internal model knowledge is introduced, using the "Imprint and Shift" mechanism. Experimental results demonstrate the efficacy of this interpretability-focused approach.},
}



Katz, Shahar, Yonatan Belinkov, Mor Geva, and Lior Wolf. "Backward Lens: Projecting Language Model Gradients into the Vocabulary Space." In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, 2390–2422. Singapore: Association for Computational Linguistics, November 2024. https://github.com/shacharKZ/BackwardLens.


