---
layout: post
title:  "[2024]Pretraining Data Detection for Large Language Models: A Divergence-based Calibration Method"  
date:   2024-12-31 02:36:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 



짧은 요약(Abstract) :    





이 연구는 대규모 언어 모델(LLMs)의 사전 훈련 데이터 감지를 위한 새로운 캘리브레이션 방법인 DC-PDD를 제안합니다. 기존의 Min-K% Prob 방법은 훈련 데이터에 포함되지 않은 텍스트를 잘못 분류할 가능성이 있었던 반면, DC-PDD는 토큰 확률 분포와 토큰 빈도 분포 간의 발산을 기반으로 더 정확한 점수를 계산합니다. 이를 통해 훈련 데이터 감지의 성능을 크게 향상시켰으며, 영어와 중국어 데이터셋 모두에서 실험적으로 검증되었습니다. 또한, 새로운 중국어 벤치마크인 PatentMIA를 구축하여 다양한 모델과 비교한 결과, 제안된 방법이 기존 기술보다 우수함을 확인했습니다.

---


This study introduces DC-PDD, a novel calibration method for pretraining data detection in large language models (LLMs). Unlike the existing Min-K% Prob approach, which may misclassify non-training texts, DC-PDD calculates a detection score based on the divergence between the token probability distribution and the token frequency distribution. This method significantly improves detection performance, as validated through experiments on both English and Chinese datasets. Additionally, a new Chinese benchmark, PatentMIA, was developed, and comparative evaluations demonstrate the superiority of the proposed approach over existing techniques.


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




이 논문의 방법론은 대규모 언어 모델(LLMs)의 사전 훈련 데이터 감지를 위해 DC-PDD(Divergence-based Calibration for Pretraining Data Detection)라는 새로운 방법을 제안합니다. 

1. **사용된 모델**: 
   - 연구에서 다룬 대상 모델은 OPT, GPT-3, Llama, Baichuan 등 다양한 대규모 언어 모델입니다. 특히, 영어 데이터 감지에서는 WikiMIA와 BookMIA 벤치마크를 사용했고, 중국어 데이터 감지를 위해 PatentMIA라는 새로운 벤치마크를 구축했습니다.

2. **핵심 아이디어**: 
   - 기존의 Min-K% Prob 방법은 텍스트의 일부 토큰 확률만을 기반으로 감지 점수를 계산하지만, DC-PDD는 토큰 확률 분포와 사전 훈련 데이터에서의 토큰 빈도 분포 간의 발산(교차 엔트로피)을 측정하여 더 정교한 감지 점수를 계산합니다. 이 발산 기반 접근법은 LLM의 사전 훈련 데이터를 더 잘 감지할 수 있도록 설계되었습니다.

3. **트레이닝 데이터**: 
   - DC-PDD는 사전 훈련 데이터 자체에 접근하지 않고, 대규모 공개 데이터셋(C4 및 ChineseWebText)을 참조 데이터로 사용하여 토큰 빈도 분포를 추정합니다. 이를 통해 모델이 사전 훈련에 사용된 텍스트를 감지하는 데 필요한 정보를 얻습니다.

4. **방법론 단계**:
   - (1) LLM을 통해 주어진 텍스트의 토큰 확률 분포를 계산합니다.
   - (2) 참조 데이터셋에서 토큰 빈도 분포를 추정합니다.
   - (3) 이 두 분포 간의 발산을 계산해 감지 점수를 도출합니다.
   - (4) 사전 정의된 임계값을 기반으로 주어진 텍스트가 훈련 데이터에 포함되었는지 여부를 결정합니다.



The methodology of this paper introduces DC-PDD (Divergence-based Calibration for Pretraining Data Detection), a novel approach for detecting pretraining data in large language models (LLMs).

1. **Models Used**: 
   - The study focuses on several large language models, including OPT, GPT-3, Llama, and Baichuan. English data detection benchmarks include WikiMIA and BookMIA, while a new Chinese benchmark named PatentMIA was constructed for Chinese data detection.

2. **Key Idea**: 
   - Unlike the Min-K% Prob method, which computes detection scores based on a subset of token probabilities, DC-PDD calculates the divergence (cross-entropy) between the token probability distribution and the token frequency distribution from pretraining data. This divergence-based approach provides a more accurate mechanism for detecting pretraining data.

3. **Training Data**: 
   - Instead of directly accessing pretraining data, DC-PDD leverages large-scale public datasets (C4 and ChineseWebText) as reference corpora to estimate token frequency distributions. These reference distributions are then used to infer whether a text was included in the pretraining data.

4. **Methodology Steps**:
   - (1) Calculate the token probability distribution of a given text using the LLM.
   - (2) Estimate the token frequency distribution using a reference corpus.
   - (3) Compute the divergence between these distributions to derive a detection score.
   - (4) Determine whether the given text is part of the training data based on a predefined threshold.



   
 
<br/>
# Results  


이 연구에서는 DC-PDD의 성능을 평가하기 위해 다양한 언어 모델과 데이터셋을 사용하여 실험을 진행했습니다.  

1. **테스트 데이터**:  
   - 영어 데이터: WikiMIA와 BookMIA 벤치마크  
   - 중국어 데이터: 새롭게 구축된 PatentMIA 벤치마크  

2. **비교 모델**:  
   - 기존의 Min-K% Prob, Min-K%++ Prob, PPL(Perplexity), Zlib, Lowercase, Small Ref와 같은 여러 베이스라인 모델과 DC-PDD의 성능을 비교했습니다.  

3. **향상된 성능**:  
   - **BookMIA** 데이터셋:  
     - DC-PDD는 AUC 점수에서 Lowercase 대비 **5.4% 향상**, TPR@5%FPR에서 **9.6% 향상**을 기록했습니다.  
   - **PatentMIA** 데이터셋:  
     - Min-K% Prob 대비 AUC 점수에서 **5.4% 향상**, TPR@5%FPR에서 **13.2% 향상**을 보였습니다.  
   - **WikiMIA** 데이터셋:  
     - AUC 점수는 기존 Min-K%++ Prob와 유사한 수준을 유지했지만, 다른 방법에 비해 여전히 높은 성능을 보였습니다.  

4. **주요 특징**:  
   - DC-PDD는 데이터와 모델에 구애받지 않는 더 일반적인 성능을 제공했으며, 특히 중국어 데이터 감지와 같은 비영어 환경에서도 우수한 결과를 달성했습니다.  
   - 작은 참조 모델(Small Ref)을 사용할 수 없는 GPT-3와 같은 폐쇄형 모델에서도 효과적으로 작동했습니다.  

---



The performance of DC-PDD was evaluated through experiments with various language models and datasets.  

1. **Test Data**:  
   - English Data: WikiMIA and BookMIA benchmarks  
   - Chinese Data: The newly constructed PatentMIA benchmark  

2. **Comparison Models**:  
   - DC-PDD was compared against several baseline models, including Min-K% Prob, Min-K%++ Prob, PPL (Perplexity), Zlib, Lowercase, and Small Ref.  

3. **Improved Performance**:  
   - **BookMIA** dataset:  
     - DC-PDD improved the AUC score by **5.4%** and TPR@5%FPR by **9.6%** compared to the Lowercase method.  
   - **PatentMIA** dataset:  
     - DC-PDD outperformed Min-K% Prob, achieving a **5.4%** increase in AUC and a **13.2%** increase in TPR@5%FPR.  
   - **WikiMIA** dataset:  
     - While its AUC improvement over Min-K%++ Prob was marginal, it still performed better than most other methods.  

4. **Key Highlights**:  
   - DC-PDD demonstrated more generalizable performance across data and models, especially in non-English scenarios like Chinese text detection.  
   - It effectively worked with closed-source models like GPT-3, where Small Ref methods are inapplicable.  




<br/>
# 예제  



**구체적인 예제**:  
아래는 DC-PDD를 사용해 영어 데이터셋 BookMIA에서 GPT-3 모델이 훈련 데이터 여부를 판단하는 과정입니다.

1. **입력 데이터**:
   - 텍스트: `"The boy walked into the room and picked up a book."`

2. **모델 처리 과정**:
   - **토큰 확률 분포 계산**:  
     GPT-3는 각 단어(`"The"`, `"boy"`, `"walked"`, ...)에 대해 확률을 생성합니다. 예를 들어:
     - `"The"`: 0.08
     - `"boy"`: 0.12
     - `"walked"`: 0.05  
   - **참조 데이터 기반 토큰 빈도 분포 추정**:  
     공개된 C4 참조 데이터셋에서 동일한 단어의 빈도를 추정:
     - `"The"`: 0.10
     - `"boy"`: 0.09
     - `"walked"`: 0.03  

3. **발산(교차 엔트로피) 계산**:  
   각 단어에 대해 발산 점수를 계산:
   - `"The"`: \( -0.08 \times \log(0.10) = 0.016 \)
   - `"boy"`: \( -0.12 \times \log(0.09) = 0.026 \)
   - `"walked"`: \( -0.05 \times \log(0.03) = 0.024 \)

4. **최종 감지 점수**:  
   발산 점수의 평균을 계산해 최종 감지 점수를 도출:
   - \( \beta = \frac{0.016 + 0.026 + 0.024}{3} = 0.022 \)

5. **결정**:
   - DC-PDD가 정의한 임계값이 \( \tau = 0.02 \)라고 가정하면, 최종 점수 \( 0.022 \)는 훈련 데이터로 포함된 것으로 간주됩니다.  
   - 결과: **"훈련 데이터 포함"**

---


**Specific Example**:  
Here is an example of how DC-PDD determines whether a text from the BookMIA dataset was part of GPT-3's training data.

1. **Input Data**:
   - Text: `"The boy walked into the room and picked up a book."`

2. **Model Processing**:
   - **Token Probability Distribution Calculation**:  
     GPT-3 generates probabilities for each word (`"The"`, `"boy"`, `"walked"`, ...). For example:
     - `"The"`: 0.08
     - `"boy"`: 0.12
     - `"walked"`: 0.05  
   - **Reference Token Frequency Estimation**:  
     The same words' frequencies are estimated from the publicly available C4 dataset:
     - `"The"`: 0.10
     - `"boy"`: 0.09
     - `"walked"`: 0.03  

3. **Divergence (Cross-Entropy) Calculation**:
   The divergence score for each word is computed:
   - `"The"`: \( -0.08 \times \log(0.10) = 0.016 \)
   - `"boy"`: \( -0.12 \times \log(0.09) = 0.026 \)
   - `"walked"`: \( -0.05 \times \log(0.03) = 0.024 \)

4. **Final Detection Score**:  
   The average divergence score is calculated to derive the final detection score:
   - \( \beta = \frac{0.016 + 0.026 + 0.024}{3} = 0.022 \)

5. **Decision**:
   - Assuming DC-PDD's predefined threshold \( \tau = 0.02 \), the final score \( 0.022 \) suggests the text was part of the training data.  
   - Result: **"Included in training data"**


<br/>  
# 요약   


이 연구는 대규모 언어 모델(LLMs)의 사전 훈련 데이터 감지를 위해 DC-PDD(Divergence-based Calibration for Pretraining Data Detection) 방법을 제안했습니다. DC-PDD는 토큰 확률 분포와 참조 데이터에서 추정한 토큰 빈도 분포 간의 발산을 기반으로 감지 점수를 계산하여, 기존 Min-K% Prob 방법보다 더 정밀한 결과를 제공합니다. 실험 결과, DC-PDD는 영어 데이터(BookMIA)에서 AUC 점수를 5.4% 향상시키고, 중국어 데이터(PatentMIA)에서도 TPR@5%FPR 기준 13.2%의 향상을 달성했습니다. 예를 들어, `"The boy walked into the room and picked up a book."`라는 텍스트는 최종 점수가 임계값(0.02)을 초과해 훈련 데이터로 판별되었습니다. DC-PDD는 데이터와 모델 유형에 관계없이 높은 정확도를 유지하며 폐쇄형 모델에서도 효과적으로 작동합니다.

---


This study proposes DC-PDD (Divergence-based Calibration for Pretraining Data Detection), a method for detecting pretraining data in large language models (LLMs). DC-PDD calculates detection scores based on the divergence between token probability distributions and token frequency distributions estimated from reference data, offering more accurate results than the Min-K% Prob method. Experimental results show that DC-PDD improved AUC by 5.4% on English data (BookMIA) and TPR@5%FPR by 13.2% on Chinese data (PatentMIA). For instance, the text `"The boy walked into the room and picked up a book."` was identified as part of the training data, as its final score exceeded the threshold (0.02). DC-PDD maintains high accuracy across diverse datasets and models, effectively working with closed-source models.

<br/>  
# 기타  



1. **Figure 1**:  
   이 그림은 Min-K% Prob 방법과 DC-PDD 방법의 차이를 시각적으로 보여줍니다. (a)에서는 Min-K% Prob이 훈련 데이터와 비훈련 데이터의 토큰 확률 분포를 비교하여 k%의 가장 낮은 확률 토큰만을 고려하는 방식을 나타냅니다. 그러나 이 방식은 높은 빈도를 가진 일반적인 단어가 포함된 비훈련 데이터를 잘못 분류할 수 있습니다. (b)에서는 DC-PDD가 토큰 확률 분포와 참조 데이터의 토큰 빈도 분포 간의 발산을 계산하여 더 정확한 점수를 제공함을 시각화합니다.

2. **Figure 2**:  
   DC-PDD의 두 가지 주요 전략, 즉 상한 제한(LUP)과 첫 번째 발생 토큰 선택(SFO)의 효과를 보여주는 그래프입니다. Baichuan-13B와 Qwen1.5-14B에서는 두 전략 모두 성능 향상에 기여했으며, 특히 AUC 점수의 증가를 관찰할 수 있습니다.

3. **Figure 3**:  
   (a)는 모델 크기에 따른 DC-PDD 성능을, (b)는 텍스트 길이에 따른 성능 변화를 보여줍니다. DC-PDD는 모델 크기가 클수록 AUC 점수가 증가하며, 텍스트가 길수록 성능이 향상되는 경향을 나타냅니다. 이는 대규모 모델이 더 많은 데이터를 기억하며, 긴 텍스트가 훈련 데이터로 감지될 가능성이 높기 때문입니다.

---



1. **Figure 1**:  
   This figure illustrates the difference between the Min-K% Prob method and the DC-PDD method. Panel (a) shows how Min-K% Prob compares token probability distributions of training and non-training data by focusing only on the k% of tokens with the lowest probabilities. However, this approach may misclassify non-training data containing frequent common words. Panel (b) visualizes how DC-PDD calculates the divergence between token probability distributions and token frequency distributions from reference data, providing more accurate detection scores.

2. **Figure 2**:  
   This graph demonstrates the impact of two key strategies in DC-PDD: limiting the upper bound (LUP) and selecting the first occurrence of tokens (SFO). For Baichuan-13B and Qwen1.5-14B, both strategies contributed to performance improvements, particularly in terms of increasing AUC scores.

3. **Figure 3**:  
   Panel (a) shows the performance of DC-PDD relative to model size, while panel (b) illustrates the impact of text length on performance. DC-PDD achieves higher AUC scores as model size increases and performs better with longer texts. This trend is attributed to larger models memorizing more pretraining data and longer texts being more likely to be identified as training data.

<br/>
# refer format:     



@inproceedings{zhang2024dcpdd,
  title = {Pretraining Data Detection for Large Language Models: A Divergence-based Calibration Method},
  author = {Weichao Zhang and Ruqing Zhang and Jiafeng Guo and Maarten de Rijke and Yixing Fan and Xueqi Cheng},
  booktitle = {Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages = {5263--5274},
  year = {2024},
  organization = {Association for Computational Linguistics},
  address = {Beijing, China},
  month = {November},
  url = {https://github.com/zhang-wei-chao/DC-PDD},
  abstract = {As the scale of training corpora for large language models (LLMs) grows, model developers become increasingly reluctant to disclose details on their data. This lack of transparency poses challenges to scientific evaluation and ethical deployment. Recently, pretraining data detection approaches, which infer whether a given text was part of an LLM’s training data through black-box access, have been explored. This paper proposes a divergence-based calibration method for token probabilities, improving detection accuracy significantly over existing methods.},
}



Zhang, Weichao, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, and Xueqi Cheng. "Pretraining Data Detection for Large Language Models: A Divergence-based Calibration Method." In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 5263–74. Beijing, China: Association for Computational Linguistics, November 2024. https://github.com/zhang-wei-chao/DC-PDD.


