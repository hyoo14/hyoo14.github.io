---
layout: post
title:  "[2024]Shifting Attention to Relevance: Towards the Predictive Uncertainty Quantification of Free-Form Large Language Models"  
date:   2024-08-27 13:48:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

이 연구는 대형 언어 모델들이 텍스트를 생성할 때, 중요한 단어들과 그렇지 않은 단어들이 섞여 있지만, 기존 방법들은 이를 구별하지 못해 오류를 일으킬 수 있다고 지적합니다. 그래서 중요한 단어에 더 집중해 모델의 신뢰성을 높이는 방법을 제안하고, 이를 통해 다양한 모델에서 더 나은 성능을 보였다고 결론짓고 있습니다.

짧은 요약(Abstract) :    

이 연구는 대형 언어 모델(LLM)이 텍스트 생성에서 뛰어난 성능을 보이지만, 종종 "환각(hallucination)" 문제로 인해 결과의 신뢰도가 떨어지는 문제를 해결하고자 합니다. 기존의 불확실성 측정(UQ) 방법들은 이 문제에 대한 잠재적인 해결책을 제공할 수 있지만, 이를 LLM에 정확하게 적용하기는 어려웠습니다. 본 연구는 모든 단어가 동일한 의미를 전달하지 않는다는 점에 주목하여, 중요하지 않은 단어가 과대 평가되는 현상을 해결하기 위해 **SAR(Shifting Attention to Relevance)**이라는 새로운 방법을 제안합니다. SAR은 중요한 요소에 주의를 집중시켜 불확실성 측정을 개선하며, 여러 LLM을 대상으로 한 실험에서 뛰어난 성능을 입증했습니다.


This study addresses the issue of "hallucinations" in large language models (LLMs), where generated texts often suffer from reduced reliability. While existing uncertainty quantification (UQ) methods offer potential solutions, applying them accurately to LLMs is challenging. The research introduces a new approach called Shifting Attention to Relevance (SAR), which acknowledges that not all words contribute equally to the meaning of a sentence. By focusing attention on more relevant components, SAR improves UQ performance. Extensive experiments on various LLMs demonstrate its superior results.


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


이 논문에서 제안한 **SAR(Shifting Attention to Relevance)** 방법은 대형 언어 모델(LLM)에서 **불확실성 측정(uncertainty quantification)**을 개선하기 위해, **토큰**과 **문장**의 중요도를 평가하고, 해당 점수를 바탕으로 **attention 값을 업데이트**하는 방식으로 작동합니다.

1. **토큰 수준에서의 관련성 평가**: 각 토큰의 중요도는 해당 토큰을 문장에서 제거했을 때 **문장의 의미가 얼마나 변하는지**를 측정하여 계산합니다. 문장의 의미적 변화는 의미 유사도 함수 \( g(x, s) \)를 통해 측정되며, 유사도가 크게 감소하면 해당 토큰은 중요한 것으로 간주됩니다. 이를 통해 **Relevance Score**를 얻습니다.

2. **문장 수준에서의 관련성 평가**: 문장 전체의 중요도는 해당 문장이 다른 문장들과 **의미적으로 얼마나 일치하는지**를 평가하여 계산됩니다. 문장이 다른 문장들과 더 유사할수록 해당 문장은 더 관련성이 높은 것으로 간주되며, 생성 확률과 유사도를 결합하여 관련성을 평가합니다.

3. **attention 값 업데이트**: 각 토큰과 문장에 대해 **attention 메커니즘**을 통해 초기 attention 값을 계산한 후, **Relevance Score**를 이용해 **해당 attention 값을 업데이트**합니다. 즉, 관련성 점수를 반영하여, 중요하지 않은 토큰이나 문장에는 주의를 덜 기울이고, 중요한 토큰이나 문장에 더 많은 주의를 기울이도록 **attention 값을 조정**합니다.

4. **불확실성 재조정**: 이렇게 업데이트된 attention 값을 바탕으로 **불확실성**을 재계산합니다. 이 과정에서 중요하지 않은 요소들이 불필요하게 주목받는 것을 방지하고, 중요한 요소들이 불확실성 평가에 더 크게 기여하도록 만듭니다.

5. **모델 적용 및 실험**: SAR 방법은 **Vicuna, LLaMA-2-chat, WizardLM**과 같은 명령어 기반 모델, 그리고 **OPT** 같은 사전 훈련된 모델에 적용되었으며, **TriviaQA, CoQA, SciQ, MedQA, MedMCQA** 등의 데이터셋에서 실험되었습니다. 실험 결과, SAR은 기존의 방법들보다 뛰어난 성능을 보여주었습니다.

---


The proposed **SAR (Shifting Attention to Relevance)** method enhances **uncertainty quantification** in large language models (LLMs) by evaluating the relevance of **tokens** and **sentences** and updating their **attention scores** based on the obtained relevance scores.

1. **Token-level Relevance Calculation**: The importance of each token is calculated by measuring how much the meaning of the sentence changes when the token is removed. The semantic change is measured using a similarity function \( g(x, s) \). If removing the token significantly alters the meaning, it is considered important, and its **Relevance Score** is calculated.

2. **Sentence-level Relevance Calculation**: The importance of each sentence is evaluated based on how semantically consistent it is with other sentences. Sentences that are more similar to others are considered more relevant. The **Relevance Score** at the sentence level is computed by combining the similarity with other sentences and generation probabilities.

3. **Updating Attention Scores**: After calculating initial **attention values** for each token or sentence using the attention mechanism, these values are **updated** using the **Relevance Scores**. This ensures that more attention is given to tokens or sentences with higher relevance and less to irrelevant ones.

4. **Re-adjusting Uncertainty**: Using the updated attention scores, **uncertainty** is recalculated. This process prevents irrelevant components from being overemphasized and ensures that important components contribute more to the model’s uncertainty estimation.

5. **Model Application and Experiments**: SAR was applied to instruction-tuned models such as **Vicuna, LLaMA-2-chat, WizardLM**, as well as pre-trained models like **OPT**, and tested on datasets including **TriviaQA, CoQA, SciQ, MedQA, MedMCQA**. SAR consistently outperformed previous methods, focusing attention on more relevant components to enhance uncertainty quantification.


<br/>
# Results  


1. **사용한 모델**:  
   - SAR 방법은 **Vicuna-13b/33b**, **LLaMA-2-chat-13b**, **WizardLM-13b**, 그리고 **OPT (2.7b, 6.7b, 13b, 30b)** 같은 명령어 기반 대형 언어 모델과 사전 훈련된 모델에 적용되었습니다.

2. **사용된 데이터셋**:  
   - 실험은 **CoQA, TriviaQA, SciQ, MedQA, MedMCQA**와 같은 자유형 질문-답변 데이터셋에서 수행되었습니다.

3. **평가 메트릭**:  
   - 생성된 답변의 정확도를 평가하기 위해 **Rouge-L**과 **문장 유사도(Sentence Similarity)**를 사용했습니다. 두 지표 모두 0.5 이상의 점수를 받은 경우 올바른 생성물로 간주되었습니다.  
   - **AUROC**(Area Under the Receiver Operating Characteristic Curve) 지표는 불확실성 측정을 평가하는 데 사용되었습니다. AUROC 값이 1에 가까울수록 더 정확한 불확실성 측정을 의미하며, SAR은 AUROC를 향상시키는 데 중점을 두었습니다.

4. **성과**:  
   - SAR은 **CoQA, TriviaQA, SciQ** 데이터셋에서 기존의 방법들보다 뛰어난 성능을 보였습니다. 예를 들어, **OPT-13b 모델**에서 SAR은 **CoQA 데이터셋**에서 AUROC 값 0.773을 기록했습니다.
   - SAR의 **TOKENSAR**와 **SENTSAR** 조합은 성능을 크게 개선했으며, **OPT-30b-CoQA** 실험에서는 AUROC 0.748을 달성했습니다.
   - **MedQA**와 **MedMCQA** 같은 **의료 데이터셋**에서도 SAR은 우수한 성능을 보였으며, **MedMCQA**에서 AUROC 0.717을 기록했습니다.

**결론**:  
SAR은 토큰과 문장의 중요도를 반영해 불확실성을 재조정함으로써 다양한 모델과 데이터셋에서 성능 향상을 일관되게 달성했습니다.

---



1. **Models Used**:  
   - SAR was applied to instruction-tuned models like **Vicuna-13b/33b**, **LLaMA-2-chat-13b**, **WizardLM-13b**, and pre-trained models like **OPT (2.7b, 6.7b, 13b, 30b)**.

2. **Datasets Used**:  
   - Experiments were conducted on free-form question-answering datasets such as **CoQA, TriviaQA, SciQ, MedQA, MedMCQA**.

3. **Evaluation Metrics**:  
   - **Rouge-L** and **Sentence Similarity** were used to evaluate the accuracy of the generated answers. Generations with scores above 0.5 were considered correct.  
   - **AUROC** (Area Under the Receiver Operating Characteristic Curve) was used to measure uncertainty quantification. The closer the AUROC score is to 1, the more accurately uncertainty is assessed, and SAR focused on improving this score.

4. **Performance**:  
   - SAR outperformed existing methods on datasets like **CoQA, TriviaQA, SciQ**. For instance, SAR achieved an AUROC of 0.773 on the **CoQA dataset** using the **OPT-13b model**.
   - The combination of **TOKENSAR** and **SENTSAR** led to significant performance improvements, achieving an AUROC of 0.748 in the **OPT-30b-CoQA** setting.
   - SAR also performed exceptionally well on medical datasets such as **MedQA** and **MedMCQA**, with an AUROC of 0.717 on the **MedMCQA** dataset.

**Conclusion**:  
By recalibrating uncertainty based on the relevance of tokens and sentences, SAR consistently improved performance across a range of models and datasets.



<br/>
# 예시  



**Before SAR 적용**  
문장: "What is the ratio of the mass of an object to its volume?"  
모델이 생성한 답변: "density of an object"  
불확실성 점수 (기존 방식): **1.949**

기존 방식에서는 Predictive Entropy (PE)가 모든 토큰에 동일한 중요도를 부여해, **"of"**와 같은 비중요 토큰이 과도한 불확실성을 일으킬 수 있습니다. 이로 인해, 모델은 중요한 단어와 비중요한 단어를 구별하지 못해 **높은 불확실성 점수**를 반환할 수 있습니다.

**After SAR 적용**  
SAR는 토큰의 중요도를 평가하여 **Relevance Score**를 계산하고, 더 중요한 토큰에 더 높은 가중치를 부여합니다. 예를 들어, **"density"**는 중요한 토큰으로 간주되어 높은 Relevance Score를 받고, **"of"**는 낮은 점수를 받습니다. SAR는 이를 바탕으로 불확실성 점수를 조정합니다.

- **불확실성 점수 (SAR 적용 후)**: **0.650**

SAR를 적용하면 불필요한 불확실성을 줄이고 더 신뢰성 있는 답변을 생성할 수 있습니다.

---

**불확실성 점수 계산 방식**  
SAR는 **Predictive Entropy (PE)**를 토큰 및 문장 수준에서 계산하며, 중요도가 높은 토큰이나 문장에 더 큰 가중치를 부여하여 불확실성을 재조정합니다.

- **토큰 수준에서의 불확실성 재조정**: 각 토큰의 불확실성은 해당 토큰이 문장의 의미에 미치는 영향을 고려해 계산됩니다. Relevance Score를 바탕으로 중요한 토큰의 불확실성은 **증가**하고, 덜 중요한 토큰의 비중은 **감소**합니다.

  ET(zi, sj, x) = -log p(zi|s<i, x) * R̃T(zi, sj, x)

- **문장 수준에서의 불확실성 재조정**: 문장 간 유사도를 바탕으로, 의미적으로 일관성이 높은 문장은 불확실성이 감소합니다. 문장의 관련성을 반영하여 불확실성을 조정합니다.

---



**Before SAR Application**  
Sentence: "What is the ratio of the mass of an object to its volume?"  
Model's generated answer: "density of an object"  
Uncertainty score (before): **1.949**

In the previous method, Predictive Entropy (PE) assigned equal importance to all tokens, causing irrelevant tokens like **"of"** to inflate the overall uncertainty. This led to a **high uncertainty score**, potentially causing the model to be unsure about generating an answer.

**After SAR Application**  
SAR recalculates each token's importance using **Relevance Score**. Important tokens like **"density"** receive a higher weight, while less important tokens like **"of"** are downweighted. This adjustment significantly reduces the uncertainty score.

- **Uncertainty score (after SAR)**: **0.650**

By applying SAR, unnecessary uncertainty is reduced, allowing the model to generate more confident and reliable answers.

---

**Uncertainty Score Calculation**  
SAR adjusts the **Predictive Entropy (PE)** at both the token and sentence levels, giving higher weights to more relevant tokens or sentences, thus recalibrating the uncertainty.

- **Token-Level Uncertainty Adjustment**: The uncertainty for each token is recalculated based on its impact on the sentence’s meaning. Tokens with higher relevance scores have their uncertainty proportion **increased**, while less relevant tokens have their proportion **decreased**.

  ET(zi, sj, x) = -log p(zi|s<i, x) * R̃T(zi, sj, x)

- **Sentence-Level Uncertainty Adjustment**: Sentence-level uncertainty is adjusted based on semantic consistency, with more semantically aligned sentences receiving lower uncertainty. The relevance of each sentence is used to adjust uncertainty.

---



<br/>  
# 요약 


SAR(Shifting Attention to Relevance)은 대형 언어 모델(LLM)의 불확실성 측정을 개선하기 위해 토큰과 문장의 관련성 점수를 계산하여, 중요한 요소에 더 많은 주의를 기울이도록 불확실성 점수를 조정합니다. 이 방식은 기존 Predictive Entropy(PE) 방식에서 발생하던 불필요한 불확실성을 줄여줍니다. SAR은 **CoQA, TriviaQA, MedQA** 등의 데이터셋에서 적용되었으며, **OPT-13b** 모델에서 AUROC가 0.773으로 향상되었습니다. 예시로, 문장 "density of an object"에서 **"density"**는 중요한 단어로 높은 관련성 점수를 받고, 덜 중요한 **"of"**는 낮은 점수를 받아 불확실성이 크게 감소했습니다. SAR을 통해 불확실성 점수가 크게 개선되어 더 신뢰성 있는 결과를 생성할 수 있었습니다.

---


SAR (Shifting Attention to Relevance) improves uncertainty quantification in large language models (LLMs) by calculating relevance scores for tokens and sentences, focusing more attention on key elements to adjust uncertainty scores. This approach reduces unnecessary uncertainty present in the previous Predictive Entropy (PE) method. SAR was applied to datasets such as **CoQA, TriviaQA, MedQA**, showing improved AUROC of 0.773 with the **OPT-13b** model. For example, in the sentence "density of an object," the word **"density"** received a high relevance score, while less important tokens like **"of"** received lower scores, reducing overall uncertainty. By applying SAR, the uncertainty score was significantly improved, resulting in more reliable outputs.


# 기타  


<br/>
# refer format:     
@inproceedings{duan2024shifting,
  title={Shifting Attention to Relevance: Towards the Predictive Uncertainty Quantification of Free-Form Large Language Models},
  author={Duan, Jinhao and Cheng, Hao and Wang, Shiqi and Zavalny, Alex and Wang, Chenan and Xu, Renjing and Kailkhura, Bhavya and Xu, Kaidi},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={5050--5063},
  year={2024},
  organization={Association for Computational Linguistics}
}



Duan Jinhao, Hao Cheng, Shiqi Wang, Alex Zavalny, Chenan Wang, Renjing Xu, Bhavya Kailkhura, and Kaidi Xu. "Shifting Attention to Relevance: Towards the Predictive Uncertainty Quantification of Free-Form Large Language Models." In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 5050-5063. Association for Computational Linguistics, 2024.

