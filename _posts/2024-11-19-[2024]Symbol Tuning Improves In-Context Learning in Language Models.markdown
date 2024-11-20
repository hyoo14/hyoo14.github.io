---
layout: post
title:  "[2024]Symbol Tuning Improves In-Context Learning in Language Models"  
date:   2024-11-19 11:00:20 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

음 그러니까 일종의 파인튜닝인 인스트럭트 튜닝을 하는건데  
레이블을 실제 단어 대신에 다른 심볼을 넣는다는 말  

왜 하냐면... 모델이 자연어의 의미를 추론하는 대신 순수한 입력-라벨 관계을 파악함..  
즉, 레이블의 자연어 의미에 과도하게 집중하지 않는 효과가 있다 이말  



짧은 요약(Abstract) :    



이 논문의 초록은 언어 모델의 **심볼 튜닝(Symbol Tuning)**이라는 새로운 방법을 소개합니다. 이 방법은 자연어 라벨(예: "긍정/부정 감정")을 임의의 심볼(예: "foo/bar")로 대체한 입력-라벨 쌍으로 언어 모델을 미세 조정합니다. 심볼 튜닝은 모델이 지시문이나 자연어 라벨에 의존하지 않고 입력-라벨 매핑을 학습하도록 유도합니다.

연구진은 Flan-PaLM 모델(최대 540B 매개변수)을 대상으로 실험하여 다음과 같은 이점을 관찰했습니다:
1. **새로운 작업 성능 향상**: 심볼 튜닝은 지시문이나 자연어 라벨이 없는 상황에서도 높은 성능을 보여줍니다.
2. **알고리즘적 추론 향상**: 리스트 함수 및 튜링 개념 작업에서 각각 최대 18.2%와 15.3%의 성능 향상을 기록했습니다.
3. **역 라벨 상황 처리 능력 강화**: 모델이 사전 지식을 무시하고 맥락 정보에 기반하여 라벨을 올바르게 따를 수 있음을 보였습니다.

이 연구는 심볼 튜닝이 적은 계산 자원으로 구현 가능하며, 언어 모델의 맥락 학습 능력을 효과적으로 개선할 수 있음을 입증합니다.

---


This paper introduces **Symbol Tuning**, a novel method for fine-tuning language models on input-label pairs where natural language labels (e.g., "positive/negative sentiment") are replaced with arbitrary symbols (e.g., "foo/bar"). Symbol Tuning encourages the model to learn input-label mappings without relying on instructions or natural language labels.

The researchers experimented with Flan-PaLM models (up to 540B parameters) and observed the following benefits:
1. **Improved Performance on New Tasks**: Symbol-tuned models perform better even in underspecified prompts without instructions or natural language labels.
2. **Enhanced Algorithmic Reasoning**: Achieved up to 18.2% and 15.3% improvements on list function and Turing concept benchmarks, respectively.
3. **Better Handling of Flipped Labels**: Symbol-tuned models can override prior semantic knowledge and accurately follow in-context flipped labels.

The study demonstrates that Symbol Tuning can be implemented with minimal computational cost and effectively enhances the in-context learning capabilities of language models.



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




이 논문에서는 **심볼 튜닝(Symbol Tuning)**이라는 새로운 방법론을 제안했습니다. 심볼 튜닝은 입력-라벨 쌍에서 자연어 라벨(예: "긍정", "부정")을 임의의 심볼(예: "foo", "bar")로 대체하여 모델을 미세 조정하는 방식입니다. 이 방법은 모델이 지시문이나 자연어 라벨에 의존하지 않고, 대신 맥락 안에 주어진 입력-라벨 매핑을 학습하도록 유도합니다.

#### 사용된 구체적 모델:
- **Flan-PaLM 모델**: 8B, 62B, 62B-Cont, 540B 매개변수 크기의 모델.
- **학습 데이터**: 22개의 공개 NLP 데이터셋과 약 30,000개의 임의의 심볼 사용.
- **튜닝 과정**: Adafactor 옵티마이저와 최대 4,000단계의 학습. 입력과 목표 시퀀스 길이는 각각 2048 및 512로 설정.

#### 기존 메서드와의 비교:
1. **유사한 기존 메서드**:
   - **Instruction Tuning**: 모델이 자연어로 된 지시문을 학습하도록 미세 조정하는 방법.
   - **Label Augmentation**: 기존 연구에서는 라벨에 노이즈를 추가하거나 무작위 라벨을 사용하여 모델의 학습 능력을 강화.

2. **심볼 튜닝의 차별점**:
   - **심볼 사용**: 심볼 튜닝은 임의의 심볼을 사용하여 자연어 라벨 대신 매핑을 학습합니다. 이는 지시문과 자연어 라벨 없이 작업을 학습하도록 모델을 강제합니다.
   - **지시문 비의존성**: 심볼 튜닝은 작업 정의에 지시문이 포함되지 않는 경우에도 높은 성능을 보입니다.
   - **알고리즘적 학습**: 기존 방법들이 주로 자연어 데이터를 다룬 반면, 심볼 튜닝은 알고리즘적 문제에서도 성능 향상을 보였습니다.

---


This paper introduces a novel methodology called **Symbol Tuning**. Symbol Tuning fine-tunes models on input-label pairs where natural language labels (e.g., "positive", "negative") are replaced with arbitrary symbols (e.g., "foo", "bar"). This forces the model to learn input-label mappings by reasoning with the in-context examples instead of relying on instructions or natural language labels.

#### Specific Models Used:
- **Flan-PaLM Models**: Includes 8B, 62B, 62B-Cont, and 540B parameter sizes.
- **Training Data**: Used 22 publicly available NLP datasets and approximately 30,000 arbitrary symbols.
- **Fine-tuning Process**: Employed the Adafactor optimizer with up to 4,000 steps of training. Input and target sequence lengths were set to 2048 and 512, respectively.

#### Comparison with Existing Methods:
1. **Similar Existing Methods**:
   - **Instruction Tuning**: Fine-tunes models to learn from tasks phrased as natural language instructions.
   - **Label Augmentation**: Previous works added noise or randomization to labels to improve model adaptability.

2. **Differences with Symbol Tuning**:
   - **Use of Symbols**: Symbol Tuning replaces natural language labels with arbitrary symbols, compelling models to infer tasks solely from input-label mappings.
   - **Independence from Instructions**: Unlike instruction tuning, Symbol Tuning achieves robust performance even in the absence of task descriptions.
   - **Algorithmic Learning**: While earlier methods focused primarily on natural language tasks, Symbol Tuning demonstrated significant improvements in algorithmic reasoning tasks.


   
 
<br/>
# Results  



#### 비교 모델:
- **기준 모델**: Instruction-tuned Flan-PaLM 모델(8B, 62B, 62B-Cont, 540B 매개변수).
- Symbol Tuning 모델과 성능을 비교.

#### 사용된 데이터셋:
- **훈련 데이터셋**: HuggingFace의 22개 NLP 데이터셋 (문장 분류, 감정 분석, 자연어 추론 등)과 약 30,000개의 임의 심볼.
- **평가 데이터셋**: 11개 NLP 데이터셋 (훈련 및 Instruction Tuning에 사용되지 않은 데이터셋).

#### 성능 향상 결과:
1. **새로운 작업 성능 향상**:
   - Flan-cont-PaLM-62B 모델 기준:
     - **지시문과 라벨 없음**: 11.1% 성능 향상.
     - **지시문만 있음**: 4.2% 성능 향상.
     - **라벨만 있음**: 1.6% 성능 향상.

2. **알고리즘적 추론 향상**:
   - 리스트 함수 작업(List Functions Benchmark): 
     - Flan-PaLM-8B: 18.2% 성능 향상.
     - Flan-PaLM-62B: 11.1% 성능 향상.
   - 튜링 개념 작업(Simple Turing Concepts Benchmark): 
     - Flan-PaLM-8B: 15.3% 성능 향상.
     - Flan-PaLM-62B: 15.3% 성능 향상.

3. **역 라벨 작업(Flipped Labels)**:
   - 모델이 역 라벨을 올바르게 따를 확률 증가:
     - Flan-PaLM-8B: 평균 26.5% 향상.
     - Flan-PaLM-62B: 평균 33.7% 향상.

심볼 튜닝은 특히 지시문이나 자연어 라벨 없이도 높은 성능을 유지하며, 대규모 모델의 필요성을 줄이는 데 성공적입니다.

---


#### Baseline Models:
- **Comparison Models**: Instruction-tuned Flan-PaLM models (8B, 62B, 62B-Cont, 540B parameters).
- Performance compared with Symbol Tuning models.

#### Datasets Used:
- **Training Datasets**: 22 NLP datasets from HuggingFace (e.g., sentence classification, sentiment analysis, natural language inference) and approximately 30,000 arbitrary symbols.
- **Evaluation Datasets**: 11 NLP datasets that were not included in training or instruction tuning.

#### Performance Improvements:
1. **Improved Performance on New Tasks**:
   - For Flan-cont-PaLM-62B:
     - **Without Instructions or Labels**: 11.1% improvement.
     - **With Instructions Only**: 4.2% improvement.
     - **With Labels Only**: 1.6% improvement.

2. **Enhanced Algorithmic Reasoning**:
   - **List Functions Benchmark**:
     - Flan-PaLM-8B: 18.2% improvement.
     - Flan-PaLM-62B: 11.1% improvement.
   - **Simple Turing Concepts Benchmark**:
     - Flan-PaLM-8B: 15.3% improvement.
     - Flan-PaLM-62B: 15.3% improvement.

3. **Flipped Labels Tasks**:
   - Increased ability to follow flipped labels:
     - Flan-PaLM-8B: Average 26.5% improvement.
     - Flan-PaLM-62B: Average 33.7% improvement.

Symbol Tuning demonstrates robust performance, particularly in scenarios without instructions or natural language labels, and reduces the reliance on larger models.



<br/>
# 예제  



#### 구체적인 예시:
- **데이터셋**: SST-2 (Stanford Sentiment Treebank 2)  
  - 이 데이터셋은 문장의 감정을 긍정(Positive) 또는 부정(Negative)으로 분류하는 감정 분석 작업입니다.

- **처리 방법**:
  1. **기존 라벨**:
     - 입력: "This movie is great."  
       라벨: "Positive"  
       출력: "Positive"  
     - 입력: "Worst film I’ve ever seen."  
       라벨: "Negative"  
       출력: "Negative"  

  2. **심볼 튜닝 적용**:
     - 자연어 라벨을 임의의 심볼로 변경:
       - 입력: "This movie is great."  
         라벨: "Foo"  
         출력: "Foo"  
       - 입력: "Worst film I’ve ever seen."  
         라벨: "Bar"  
         출력: "Bar"  
     - 모델은 맥락에서 제공된 예제를 기반으로 입력과 심볼 라벨 간의 매핑 규칙을 학습.

- **예측 결과**:
  - 새 데이터:
    - 입력: "This movie is terrible."  
      - 심볼 튜닝 적용 전: 모델이 'Positive' 또는 'Negative'를 무작위로 예측.
      - 심볼 튜닝 적용 후: 맥락 정보를 기반으로 "Bar"를 정확히 예측.  
  - 성능 향상:
    - 심볼 튜닝 후, SST-2 데이터셋에서 성능이 평균적으로 15% 이상 향상.

---



#### Concrete Example:
- **Dataset**: SST-2 (Stanford Sentiment Treebank 2)  
  - This dataset involves classifying sentences as expressing Positive or Negative sentiment.

- **Processing Method**:
  1. **Original Labels**:
     - Input: "This movie is great."  
       Label: "Positive"  
       Output: "Positive"  
     - Input: "Worst film I’ve ever seen."  
       Label: "Negative"  
       Output: "Negative"  

  2. **With Symbol Tuning**:
     - Replace natural language labels with arbitrary symbols:
       - Input: "This movie is great."  
         Label: "Foo"  
         Output: "Foo"  
       - Input: "Worst film I’ve ever seen."  
         Label: "Bar"  
         Output: "Bar"  
     - The model learns to map inputs to symbols based on in-context examples.

- **Prediction Results**:
  - New Data:
    - Input: "This movie is terrible."  
      - Before Symbol Tuning: Model randomly predicts 'Positive' or 'Negative.'  
      - After Symbol Tuning: Accurately predicts "Bar" based on in-context rules.  
  - Performance Improvement:
    - Symbol tuning improved the SST-2 dataset performance by over 15% on average.


<br/>  
# 요약   


심볼 튜닝(Symbol Tuning)은 자연어 라벨을 임의의 심볼로 대체하여 언어 모델을 미세 조정하는 새로운 방법론입니다. 이 방법은 모델이 입력-라벨 간의 관계를 학습하도록 강제하며, 지시문이나 자연어 라벨 없이도 높은 성능을 보입니다. 연구진은 Flan-PaLM 모델을 사용해 22개의 NLP 데이터셋으로 훈련하고, 11개의 새로운 작업에서 평가하여 이 방법의 효과를 입증했습니다. 특히 리스트 함수나 튜링 테스트와 같은 알고리즘적 작업에서 최대 18.2%의 성능 향상을 기록했습니다. 예를 들어, 감정 분석 작업에서는 심볼 튜닝 모델이 맥락 정보만으로도 '긍정' 또는 '부정'을 정확히 예측할 수 있었습니다.

---

Symbol Tuning is a novel methodology that fine-tunes language models by replacing natural language labels with arbitrary symbols. This approach forces the model to learn input-label mappings, achieving high performance even without instructions or natural language labels. Researchers trained Flan-PaLM models on 22 NLP datasets and evaluated them on 11 new tasks to demonstrate the effectiveness of this method. Notably, it achieved up to 18.2% improvement in algorithmic tasks like list functions and Turing tests. For example, in sentiment analysis, symbol-tuned models accurately predicted "positive" or "negative" using only contextual information.


<br/>  
# 기타  


<br/>
# refer format:     



@article{Wei2024,
  title={Symbol Tuning Improves In-Context Learning in Language Models},
  author={Jerry Wei and Le Hou and Andrew Lampinen and Xiangning Chen and Da Huang and Yi Tay and Xinyun Chen and Yifeng Lu and Denny Zhou and Tengyu Ma and Quoc V. Le},
  journal={arXiv preprint arXiv:2305.08298v2},
  year={2024},
  month={January},
  note={Available at: \url{https://arxiv.org/abs/2305.08298}}
}



Wei, Jerry, Le Hou, Andrew Lampinen, Xiangning Chen, Da Huang, Yi Tay, Xinyun Chen, Yifeng Lu, Denny Zhou, Tengyu Ma, and Quoc V. Le. 2024. "Symbol Tuning Improves In-Context Learning in Language Models." arXiv preprint arXiv:2305.08298v2. January.   

