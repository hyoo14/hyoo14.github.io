---
layout: post
title:  "[2024]Bridge the Gap Between CV and NLP! AGradient-based Textual Adversarial Attack Framework"  
date:   2024-06-21 07:22:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

짧은 요약(Abstract) :    


최근 다양한 작업에서 성공을 거두었음에도 불구하고, 심층 학습 기술은 여전히 작은 섭동을 포함한 적대적 예제에 대해 성능이 저조합니다. 최적화 기반의 적대적 공격 방법이 컴퓨터 비전 분야에서는 잘 탐구되었지만, 텍스트의 불연속성으로 인해 자연어 처리에 직접 적용하기는 현실적으로 어렵습니다. 이 문제를 해결하기 위해, 기존의 최적화 기반 적대적 공격 방법을 비전 영역에서 텍스트 적대적 샘플을 생성하는 통합 프레임워크를 제안합니다. 이 프레임워크에서는 지속적으로 최적화된 섭동을 임베딩 레이어에 추가하고 전방 전파 과정에서 증폭합니다. 그런 다음 최종적으로 섭동된 잠재 표현을 마스크된 언어 모델 헤드를 사용하여 잠재적 적대적 샘플을 얻기 위해 디코딩합니다. 본 논문에서는 Textual Projected Gradient Descent(T-PGD)라는 공격 알고리즘을 통해 프레임워크를 구체화합니다. 우리는 프록시 기울기 정보를 사용하는 경우에도 이 알고리즘이 효과적임을 발견했습니다. 따라서 더 도전적인 전이 블랙 박스 공격을 수행하고 여러 모델과 세 가지 벤치마크 데이터셋에서 공격 알고리즘을 평가하기 위해 종합적인 실험을 수행했습니다. 실험 결과, 우리의 방법이 전반적으로 더 나은 성능을 달성하며 기존의 강력한 기준 방법과 비교하여 더 유창하고 문법적으로 올바른 적대적 샘플을 생성함을 보여줍니다. 코드와 데이터는 [여기](https://github.com/Phantivia/T-PGD)에서 이용 가능합니다.



Despite recent success on various tasks, deep learning techniques still perform poorly on adversarial examples with small perturbations. While optimization-based methods for adversarial attacks are well-explored in the field of computer vision, it is impractical to directly apply them in natural language processing due to the discrete nature of the text. To address the problem, we propose a unified framework to extend the existing optimization-based adversarial attack methods in the vision domain to craft textual adversarial samples. In this framework, continuously optimized perturbations are added to the embedding layer and amplified in the forward propagation process. Then the final perturbed latent representations are decoded with a masked language model head to obtain potential adversarial samples. In this paper, we instantiate our framework with an attack algorithm named Textual Projected Gradient Descent (T-PGD). We find our algorithm effective even using proxy gradient information. Therefore, we perform the more challenging transfer black-box attack and conduct comprehensive experiments to evaluate our attack algorithm with several models on three benchmark datasets. Experimental results demonstrate that our method achieves overall better performance and produces more fluent and grammatical adversarial samples compared to strong baseline methods. The code and data are available at [GitHub](https://github.com/Phantivia/T-PGD).





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


이 섹션에서는 프레임워크의 개요와 연속적인 섭동을 추가하고 텍스트를 재구성하는 방법에 대해 설명합니다.

#### 개요

우리는 섭동 생성 과정에서 두 가지 모델을 사용합니다: (1) 적대적 샘플을 최적화하기 위해 기울기 정보를 제공하는 로컬 프록시 모델과 (2) 공격자가 속이려는 실제 희생자 모델입니다. 구체적으로, 로컬 데이터셋에서 미세 조정된 프록시 BERT 모델이 각 불연속 텍스트 인스턴스를 연속적인 토큰 임베딩으로 인코딩하고 여기에 연속적인 섭동을 추가합니다. 섭동은 희생자 모델의 예측 출력에 따라 프록시 모델의 기울기를 사용하여 반복적으로 최적화됩니다. 섭동 후, MLM 헤드가 섭동된 잠재 표현을 디코딩하여 적대적 샘플 후보를 생성합니다.

#### 연속적인 섭동 추가

이전 연구는 트랜스포머 기반 사전 학습 언어 모델의 잠재 표현이 의미론적 및 구문적 특징을 제공하는 데 효과적임을 보여주었습니다. 따라서 로컬 데이터셋에서 미세 조정된 로컬 BERT 모델을 프레임워크의 인코더로 사용합니다.

각 텍스트 입력에 대해, 우리는 먼저 전방 전파 과정에서 작업별 손실을 계산한 다음, 역전파를 수행하여 입력 텍스트의 토큰 임베딩에 대한 손실의 기울기를 얻습니다. 생성된 기울기는 토큰 임베딩에 추가된 섭동을 업데이트하기 위한 정보로 간주됩니다.

#### 텍스트 재구성

연속적인 섭동을 사용하여, 우리는 최적화된 토큰 임베딩에서 의미 있는 적대적 텍스트를 재구성해야 합니다. MLM 헤드는 특정 작업에 대해 미세 조정된 후에도 중간 레이어의 숨겨진 상태에서 입력 문장을 높은 정확도로 재구성할 수 있습니다. 구체적으로, MLM 헤드는 H × V 선형 레이어로, 여기서 H는 숨겨진 상태의 크기이고 V는 어휘의 크기입니다. 연속적인 입력 숨겨진 상태 h가 주어지면, MLM 헤드는 t = hAT + b로 토큰 ID를 예측할 수 있습니다.

  


This section describes an overview of the framework and the method of adding continuous perturbations and reconstructing the text.

#### Overview

We use two models in the perturbation generation process: (1) a local proxy model which provides gradient information to optimize the adversarial samples, and (2) the true victim model that the attacker attempts to deceive. Specifically, a proxy BERT model fine-tuned on the attacker's local dataset encodes each discrete text instance into continuous token embeddings and then adds continuous perturbation to it. The perturbation would be iteratively optimized using the gradient of the proxy model, according to the prediction output of the victim model. After perturbation, an MLM head will decode the perturbed latent representation to generate candidate adversarial samples.

#### Adding Continuous Perturbations

Previous work has shown that the latent representations of transformer-based pre-trained language models are effective in providing semantic and syntactic features. Therefore, we use a local BERT model fine-tuned on our local dataset as the encoder for our framework.

For each text input, we first calculate the task-specific loss in the forward propagation process, and then perform backward propagation to obtain the gradients of the loss with respect to the token embeddings of the input text. The generated gradients are viewed as the information for updating the perturbations added to the token embeddings.

#### Reconstruction

Using continuous perturbations, we need to reconstruct the meaningful adversarial text from the optimized token embeddings. The MLM head is observed to be able to reconstruct input sentences from hidden states in middle layers with high accuracy, even after models have been fine-tuned on specific tasks. Specifically, the MLM head is an H × V linear layer, where H is the size of hidden states and V is the size of the vocabulary. Given continuous input hidden states h, the MLM head can predict token IDs by t = hAT + b.




<br/>
# Results  



#### 실험 설정

우리는 감정 분석, 자연어 추론 및 뉴스 분류 작업에서 프레임워크와 T-PGD 알고리즘의 효과를 평가하기 위해 종합적인 실험을 수행했습니다. 각 데이터셋에 대해, BERT, RoBERTa, ALBERT, 그리고 XLNet을 대상으로 하는 T-PGD의 공격 성능을 평가했습니다. 실험은 세 가지 벤치마크 데이터셋(SST-2, MNLI, AG's News)에서 수행되었습니다.

#### 실험 결과

T-PGD는 공격 성공률 및 적대적 샘플의 품질 측면에서 강력한 기준 방법들을 일관되게 능가했습니다. 특히, 우리의 적대적 샘플은 기준 모델보다 더 높은 USE 점수를 기록하여 의미적 일관성이 더 높음을 나타냈습니다. 또한, 긴 텍스트에서는 문법적 오류가 적고 당혹도가 낮은 적대적 샘플을 생성했습니다.

#### 인간 평가

100개의 SST-2 원본 문장과 100개의 적대적 샘플을 랜덤으로 선택하여 인간 평가를 수행했습니다. 인간 평가 결과, T-PGD가 생성한 적대적 샘플이 기준 방법보다 더 나은 의미적 일관성과 품질을 가지고 있음을 확인했습니다. 

#### 추가 분석

기울기 정보의 중요성을 확인하기 위해, 기울기 정보를 사용하지 않고 랜덤 섭동만을 추가하는 실험을 수행했습니다. 결과는 기울기 정보를 사용하지 않을 경우, 공격 성공률과 USE 점수가 크게 감소함을 보여줍니다.

재구성 작업의 중요성을 확인하기 위해, 재구성 손실을 추가하지 않는 실험을 수행했습니다. 결과는 재구성 손실이 없을 경우, 적대적 샘플의 품질이 크게 떨어짐을 보여줍니다.

#### 효율성 및 눈에 띄지 않음

T-PGD는 상대적으로 높은 쿼리 수를 요구하지만, 긴 텍스트에서는 더 낮은 섭동 비율을 기록했습니다. 이는 T-PGD가 더 낮은 쿼리 예산으로도 강력한 공격을 수행할 수 있음을 나타냅니다.

#### 모델 간 전이 가능성

T-PGD로 생성된 적대적 샘플이 다른 모델에서도 효과적임을 확인했습니다. 예를 들어, BERT를 공격하여 생성된 적대적 샘플이 RoBERTa에서도 높은 공격 성공률을 기록했습니다.



#### Experimental Setting

We conducted comprehensive experiments to evaluate the effectiveness of our framework and T-PGD algorithm on the tasks of sentiment analysis, natural language inference, and news classification. For each dataset, we evaluated the performance of T-PGD by attacking BERT, RoBERTa, ALBERT, and XLNet. The experiments were conducted on three benchmark datasets: SST-2, MNLI, and AG's News.

#### Experimental Results

T-PGD consistently outperformed strong baseline methods in terms of attack success rate and the quality of adversarial samples. Notably, our adversarial samples yielded higher USE scores than baseline models, indicating higher semantic consistency. Additionally, T-PGD produced adversarial samples with fewer grammatical errors and lower perplexity in longer texts.

#### Human Evaluation

We randomly selected 100 original SST-2 sentences and 100 adversarial samples for human evaluation. The results showed that adversarial samples crafted by T-PGD had better semantic consistency and quality compared to the baseline method.

#### Further Analysis

To verify the importance of gradient information, we conducted an ablation experiment by adding only random perturbations without using gradient information. The results showed a significant decrease in attack success rate and USE score without gradient information.

To confirm the importance of the reconstruction task, we performed an experiment without the reconstruction loss. The results showed a substantial decline in the quality of adversarial samples without the reconstruction loss.

#### Efficiency and Imperceptibility

T-PGD requires a relatively high query number but recorded lower perturbation rates in longer texts. This indicates that T-PGD can still perform a strong attack with a lower query budget.

#### Transferability Across Models

We confirmed that adversarial samples generated by T-PGD are effective across different models. For example, adversarial samples generated by attacking BERT also achieved high attack success rates on RoBERTa.





<br/>
# 예시  

#### 예제 1 (SST-2 데이터셋)

- **원본 문장**: "the movie bounces all over the map."
  - **설명**: 이 문장은 영화가 줄거리가 산만하고 일관성이 없음을 나타냅니다.
- **적대적 샘플**: "the movie bounce & all over & map."
  - **설명**: 원본 문장의 의미는 유지하면서도 약간의 섭동을 추가하여 모델이 혼동하도록 합니다. '&' 기호를 추가하여 문장의 문법적 정확성을 약간 훼손시켰습니다.

- **원본 문장**: "looks like a high school film project completed the day before it was due."
  - **설명**: 이 문장은 마감 하루 전에 완성된 고등학교 영화 프로젝트처럼 보인다는 부정적인 평가를 담고 있습니다.
- **적대적 샘플**: "looks like a unique school film project completed the day before it was due."
  - **설명**: 'high'를 'unique'로 바꾸어 문장의 전체 의미를 바꾸지 않으면서도 모델이 혼동하도록 합니다.

#### 예제 2 (MNLI 데이터셋)

- **원본 문장 (전제)**: "and he said , what ’s going on ?"
  - **설명**: 누군가가 "무슨 일이야?"라고 물어보는 상황을 나타냅니다.
- **원본 문장 (가설)**: "he wanted to know what was going on ."
  - **설명**: 그가 무슨 일이 일어나고 있는지 알고 싶어했다는 설명입니다.
- **적대적 샘플 (가설)**: "he wanted to know what was going on ¡"
  - **설명**: 문장의 끝에 특수 문자 '¡'를 추가하여 원본 의미를 거의 변경하지 않으면서도 모델이 혼동하도록 합니다.

- **원본 문장 (전제)**: "they seem to have him on a primary radar ."
  - **설명**: 그들이 그를 주요 레이더에 포착한 것 같다는 의미입니다.
- **원본 문장 (가설)**: "they have got him on a primary radar ."
  - **설명**: 그들이 그를 주요 레이더에 포착했다는 의미입니다.
- **적대적 샘플 (가설)**: "they finally got him on a primary radar."
  - **설명**: 'have got'를 'finally got'로 바꾸어 원본 의미를 유지하면서도 모델이 혼동하도록 합니다.

#### 예제 3 (AG's News 데이터셋)

- **원본 문장**: "nortel lowers expectations nortel said it expects revenue for the third quarter to fall short of expectations."
  - **설명**: 노텔이 3분기 수익이 기대에 못 미칠 것으로 예상한다고 발표했다는 내용입니다.
- **적대적 샘플**: "nortel lowers expectations nortel said , expects income for the third quarter to fall short of expectations."
  - **설명**: 'revenue'를 'income'으로 바꾸어 문장의 의미를 유지하면서도 약간의 섭동을 추가했습니다.

- **원본 문장**: "itunes now selling band aid song ipod owners can download the band aid single after apple reaches agreement with the charity."
  - **설명**: 아이튠즈에서 밴드 에이드 노래를 판매하며, 애플이 자선단체와 합의한 후 아이팟 소유자가 밴드 에이드 싱글을 다운로드할 수 있다는 내용입니다.
- **적대적 샘플**: "the now selling band aid song dar norman can reach the band aid single after apple reaches agreement with the charity."
  - **설명**: 'itunes'를 'the'로, 'ipod owners'를 'dar norman'으로 바꾸어 문장의 의미를 유지하면서도 모델이 혼동하도록 합니다.



#### Example 1 (SST-2 Dataset)

- **Original Sentence**: "the movie bounces all over the map."
  - **Explanation**: This sentence indicates that the movie's plot is disjointed and lacks coherence.
- **Adversarial Sample**: "the movie bounce & all over & map."
  - **Explanation**: By adding '&' symbols, the sentence's grammatical correctness is slightly compromised, creating confusion for the model while maintaining the original meaning.

- **Original Sentence**: "looks like a high school film project completed the day before it was due."
  - **Explanation**: This sentence gives a negative review, indicating that the film looks like a last-minute high school project.
- **Adversarial Sample**: "looks like a unique school film project completed the day before it was due."
  - **Explanation**: Replacing 'high' with 'unique' changes the sentence slightly, causing confusion for the model without altering the overall meaning.

#### Example 2 (MNLI Dataset)

- **Original Sentence (Premise)**: "and he said , what ’s going on ?"
  - **Explanation**: This indicates someone asking, "What's going on?"
- **Original Sentence (Hypothesis)**: "he wanted to know what was going on."
  - **Explanation**: This explains that he wanted to know what was happening.
- **Adversarial Sample (Hypothesis)**: "he wanted to know what was going on ¡"
  - **Explanation**: Adding the special character '¡' at the end of the sentence keeps the original meaning while causing confusion for the model.

- **Original Sentence (Premise)**: "they seem to have him on a primary radar."
  - **Explanation**: This means that they have detected him on the primary radar.
- **Original Sentence (Hypothesis)**: "they have got him on a primary radar."
  - **Explanation**: This indicates that they have detected him on the primary radar.
- **Adversarial Sample (Hypothesis)**: "they finally got him on a primary radar."
  - **Explanation**: Changing 'have got' to 'finally got' maintains the original meaning while confusing the model.

#### Example 3 (AG's News Dataset)

- **Original Sentence**: "nortel lowers expectations nortel said it expects revenue for the third quarter to fall short of expectations."
  - **Explanation**: This statement indicates that Nortel expects its third-quarter revenue to fall short of expectations.
- **Adversarial Sample**: "nortel lowers expectations nortel said , expects income for the third quarter to fall short of expectations."
  - **Explanation**: Changing 'revenue' to 'income' keeps the sentence's meaning while adding slight perturbations.

- **Original Sentence**: "itunes now selling band aid song ipod owners can download the band aid single after apple reaches agreement with the charity."
  - **Explanation**: This indicates that iTunes is selling a Band Aid song and that iPod owners can download the single after Apple reaches an agreement with the charity.
- **Adversarial Sample**: "the now selling band aid song dar norman can reach the band aid single after apple reaches agreement with the charity."
  - **Explanation**: Replacing 'itunes' with 'the' and 'ipod owners' with 'dar norman' maintains the sentence's meaning while confusing the model.


<br/>  
# 요약 



이 연구는 컴퓨터 비전(CV)에서 사용되는 최적화 기반 적대적 공격 방법을 자연어 처리(NLP)에 적용하기 위한 통합 프레임워크를 제안합니다. 우리는 Textual Projected Gradient Descent(T-PGD)라는 알고리즘을 통해 이 프레임워크를 구체화하고, 여러 모델과 세 가지 벤치마크 데이터셋에서 이 알고리즘의 성능을 평가했습니다. 실험 결과, T-PGD는 전반적으로 더 나은 성능을 보였으며, 기존 방법보다 더 유창하고 문법적으로 올바른 적대적 샘플을 생성했습니다. 또한, 인간 평가에서도 T-PGD가 생성한 적대적 샘플이 의미적 일관성과 품질 면에서 우수함을 확인했습니다. 본 연구는 NLP 연구자들이 최적화 기반 방법을 사용하여 적대적 텍스트를 생성할 수 있도록 지원하며, CV와 NLP 간의 연구 자원을 공유하는 데 기여합니다.  


This study proposes a unified framework to apply optimization-based adversarial attack methods used in computer vision (CV) to natural language processing (NLP). We instantiated this framework with an algorithm named Textual Projected Gradient Descent (T-PGD) and evaluated its performance with several models on three benchmark datasets. Experimental results showed that T-PGD achieved overall better performance and produced more fluent and grammatically correct adversarial samples compared to existing methods. Additionally, human evaluations confirmed that adversarial samples generated by T-PGD were superior in terms of semantic consistency and quality. This research supports NLP researchers in generating adversarial texts using optimization-based methods and contributes to sharing research resources between CV and NLP.   


# 기타  



<br/>
# refer format:     
Yuan, Lifan, Zhang, YiChi, Chen, Yangyi, Wei, Wei. (2023). Bridge the Gap Between CV and NLP! A Gradient-based Textual Adversarial Attack Framework. *Findings of the Association for Computational Linguistics: ACL 2023*. Toronto, Canada. (pp. 7132–7146). Association for Computational Linguistics. DOI: 10.18653/v1/2023.findings-acl.446. Available at: https://aclanthology.org/2023.findings-acl.446  


@inproceedings{Yuan2023,
  author    = {Lifan Yuan and YiChi Zhang and Yangyi Chen and Wei Wei},
  title     = {Bridge the Gap Between CV and NLP! A Gradient-based Textual Adversarial Attack Framework},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2023},
  year      = {2023},
  month     = {July},
  pages     = {7132--7146},
  address   = {Toronto, Canada},
  editor    = {Anna Rogers and Jordan Boyd-Graber and Naoaki Okazaki},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2023.findings-acl.446},
  doi       = {10.18653/v1/2023.findings-acl.446}
}
