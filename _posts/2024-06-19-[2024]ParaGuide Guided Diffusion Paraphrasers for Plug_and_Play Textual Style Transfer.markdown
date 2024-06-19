---
layout: post
title:  "[2024]ParaGuide: Guided Diffusion Paraphrasers for Plug-and-Play Textual Style Transfer"  
date:   2024-06-19 11:18:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

짧은 요약(Abstract) :    
### Abstract 번역 (한글)

텍스트 스타일 변환은 텍스트의 의미를 유지하면서 스타일적 특성을 변형하는 작업입니다. 목표 "스타일"은 단일 속성(예: 격식)에서부터 작가의 스타일(예: 셰익스피어)에 이르기까지 여러 방식으로 정의될 수 있습니다. 이전의 비지도 스타일 변환 접근 방식은 고정된 스타일 집합에 대해 상당한 양의 라벨링된 데이터를 필요로 하거나 대규모 언어 모델을 요구하는 경우가 많습니다. 반면에 우리는 추론 시 임의의 목표 스타일에 유연하게 적용할 수 있는 새로운 확산 기반 프레임워크를 소개합니다. 우리의 효율적인 파라미터 방식인 PARAGUIDE는 패러프레이즈 조건 확산 모델과 오프 더 셀프 분류기 및 강력한 기존 스타일 임베더에서의 그래디언트 기반 지침을 활용하여 텍스트의 의미를 유지하면서 스타일을 변환합니다. 우리는 Enron 이메일 코퍼스에서 이 방법을 검증하고, 인간 및 자동 평가를 통해 형식성, 감정, 심지어 작가 스타일 변환에서도 강력한 기준을 능가하는 결과를 얻었습니다.  

### Abstract (영어)

Textual style transfer is the task of transforming stylistic properties of text while preserving meaning. Target “styles” can be defined in numerous ways, ranging from single attributes (e.g., formality) to authorship (e.g., Shakespeare). Previous unsupervised style-transfer approaches generally rely on significant amounts of labeled data for only a fixed set of styles or require large language models. In contrast, we introduce a novel diffusion-based framework for general-purpose style transfer that can be flexibly adapted to arbitrary target styles at inference time. Our parameter-efficient approach, PARAGUIDE, leverages paraphrase-conditioned diffusion models alongside gradient-based guidance from both off-the-shelf classifiers and strong existing style embedders to transform the style of text while preserving semantic information. We validate the method on the Enron Email Corpus, with both human and automatic evaluations, and find that it outperforms strong baselines on formality, sentiment, and even authorship style transfer.  
* Useful sentences :  
*   


{% endhighlight %}  

<br/>

[Paper link](https://drive.google.com/drive/folders/1yRrce_OzRnkWidnQ8k7ejrW7_eCSGiRy?usp=sharing)  
[~~Lecture link~~]()   

<br/>

# 단어정리  
*  
 
<br/>
# Methodology    

* PARAGUIDE는 세 가지 주요 단계로 구성됩니다:  

1. 자가회귀(AR) 모델을 사용하여 입력 텍스트의 초기 패러프레이즈를 생성합니다.  


자가회귀(AR) 모델은 이전 단계에서 생성된 출력을 다음 단계의 입력으로 사용하는 모델입니다. 이는 일반적으로 순차적으로 텍스트를 생성하는 데 사용됩니다. AR 모델은 입력 텍스트의 구조와 의미를 유지하면서 새로운 텍스트를 생성하는 데 도움을 줍니다. 대표적인 예로 GPT 모델이 있습니다.


2. 패러프레이즈 조건 확산 모델을 사용하여 여러 확산 단계를 거쳐 이 패러프레이즈에서 입력 텍스트를 반복적으로 재구성합니다.    


패러프레이즈 조건 확산 모델은 노이즈가 추가된 텍스트 임베딩을 통해 점진적으로 원래의 텍스트를 재구성하는 모델입니다. 이 모델은 패러프레이즈를 조건으로 사용하여 의미를 유지하면서도 스타일을 변화시킬 수 있습니다. 확산 모델은 일반적으로 이미지 생성에서 사용되었지만, 최근 텍스트 생성에도 적용되고 있습니다. 이 모델은 텍스트의 연속적인 표현을 점진적으로 노이즈를 제거하면서 복원합니다.   


3. 각 확산 단계에서 임의의 미분 가능한 손실에 대한 그래디언트를 계산하고 이 그래디언트를 사용하여 목표 스타일을 향한 지침을 제공합니다.  


각 확산 단계에서는 특정 손실 함수에 대한 그래디언트를 계산하여 텍스트가 목표 스타일로 변환되도록 지침을 제공합니다. 이 손실 함수는 텍스트가 얼마나 잘 목표 스타일을 따르는지를 측정합니다. 예를 들어, 형식적인 글쓰기 스타일로 변환하려면, 해당 스타일을 잘 따르는지 평가하는 손실 함수를 사용할 수 있습니다.    



* PARAGUIDE has three primary steps:

1. Generating an initial paraphrase of an input text with an autoregressive (AR) model.  


An autoregressive (AR) model generates text sequentially, where the output from the previous step is used as the input for the next 
step. This type of model helps in maintaining the structure and meaning of the input text while generating new text. A prominent example of an AR model is the GPT (Generative Pre-trained Transformer) model.


2. Using a paraphrase-conditioned text diffusion model to iteratively reconstruct the input text from this paraphrase over a number of diffusion steps.    


The paraphrase-conditioned text diffusion model iteratively reconstructs the original text from noised text embeddings by gradually removing the noise. This model uses the paraphrase as a condition to maintain semantic consistency while altering the style. Diffusion models, originally popular in image generation, have recently been applied to text generation. They work by progressively denoising a continuous representation of the text.    


3. At each diffusion step, computing gradients for arbitrary differentiable losses, and using these gradients for guidance towards a target style.    


At each diffusion step, gradients for a specific loss function are computed to guide the text towards the desired style. This loss function measures how well the text adheres to the target style. For example, if transforming the text to a formal writing style, a loss function that evaluates adherence to this style would be used.    


<br/>
# Results  

* 속성 스타일 변환

이 섹션에서는 속성 변환에 대한 평가 결과를 검토합니다. 대표적인 출력은 부록에 포함되어 있습니다.

** 자동 평가

자동 평가는 모델이 생성한 텍스트가 목표 스타일을 얼마나 잘 따르는지 평가하기 위해 다양한 지표를 사용합니다. 이 지표들은 다음과 같이 평가됩니다:

1. **전환 정확도 (Transfer Accuracy)**:
   - **내부 정확도 (Internal Accuracy)**: 모델이 스타일 전환 시 사용한 분류기를 사용하여 측정한 정확도입니다. 예를 들어, 모델이 형식적인 텍스트를 생성했는지 평가할 때, 형식성을 분류하는 분류기를 사용합니다.
   - **외부 정확도 (External Accuracy)**: 학습에 사용되지 않은 별도의 분류기를 사용하여 측정한 정확도입니다. 이는 모델의 일반화 능력을 평가합니다.

2. **유사성 (Similarity)**:
   - 생성된 텍스트와 원본 텍스트 간의 의미적 유사성을 측정합니다. 이는 주로 상호 암시 점수 (Mutual Implication Score, MIS)로 측정되며, 두 텍스트가 얼마나 의미적으로 일치하는지를 평가합니다.

3. **유창성 (Fluency)**:
   - 생성된 텍스트의 문법적 정확성과 일관성을 평가합니다. 이는 CoLA (Corpus of Linguistic Acceptability) 데이터셋에서 훈련된 모델을 사용하여 측정됩니다.

표 1과 3은 형식성과 감정 전환에 대한 자동 평가 결과를 보여줍니다. 각 접근법에 대해 각 지표의 평균 점수와 형식적/비형식적(→ F, → I) 및 긍정적/부정적(→ P, → N) 변환의 세부 사항을 표시합니다.

PARAGUIDE는 감정 및 형식성 실험 전반에 걸쳐 모든 집계된 공동 지표에서 다른 모든 접근법을 능가합니다. 또한, PARAGUIDE는 전환 정확도에서 모든 기준을 크게 능가합니다. 전환 정확도와 의미 보존 사이의 본질적인 트레이드오프에도 불구하고, 형식성 측면에서 PARAGUIDE(λ = 2e2)는 전환 정확도와 의미 보존 모두에서 모든 기준 접근법을 능가합니다. 감정 전환에서는 PARAGUIDE의 증가된 정확도가 의미 유사성에 더 큰 비용을 초래하지만, 이는 텍스트의 극성을 변경하는 성공적인 감정 전환에서 기대되는 결과입니다(Jin et al. 2022).

** 인간 평가

표 2는 인간 형식성 평가 결과를 보여줍니다. 평가자는 모델 출력의 형식성, 유사성, 유창성을 평가했습니다. 인간 평가에서 PARAGUIDE는 모든 집계된 메트릭에서 최고 성능의 기준을 크게 능가합니다(p = 0.05). 특히, 유창성 메트릭에서도 예상 외로 PARAGUIDE가 더 우수한 평가를 받았습니다. 이는 CoLA 훈련 코퍼스의 구성과 이메일 작성 관행의 차이로 설명될 수 있습니다(Warstadt, Singh, and Bowman 2019). 반면에 STRAP 기준은 인간 평가에서 크게 저조한 성과를 보였습니다. 우리는 STRAP 모델이 매우 반복적인 텍스트를 생성하는 것을 발견했으며, 이는 제한된 데이터셋에서 미세 조정된 결과로 보입니다(Patel, Andrews, and Callison-Burch 2022).

작가 스타일 변환

표 4는 자원 부족 환경에서의 작가 스타일 변환에 대한 결과를 보여줍니다. 스타일 임베딩 공간에서 평가했을 때, PARAGUIDE의 네 가지 구성 중 세 가지가 모든 기준 접근법을 능가했습니다(심지어 ChatGPT-3.5도 포함). 그러나 보류된 UAR 임베딩 공간을 고려할 때, PARAGUIDE는 다른 접근법을 능가하지 못했으며, 이는 STRAP의 높은 유창성 점수 때문입니다. 이와 달리, STRAP 구현은 각 작가에 대해 800백만 매개변수를 가진 110개의 별도 모델을 포함합니다.

* 평가를 위한 데이터셋

우리는 이메일 스타일 전환 접근법이 새로운 작가와 텍스트에 일반화되는지 평가해야 합니다. 이를 위해 저자 및 속성 평가를 위해 주소의 10%를 보류 작가로 무작위 선택했습니다. 이 110명의 작가는 자원 부족 저작 코퍼스를 제공합니다. 저작 실험을 위해 각 보류 소스 작가의 테스트 이메일 중 최대 5개를 선택하고, 이를 5명의 무작위 보류 작가에게 전환합니다.

속성 스타일 전환을 위한 훈련 및 검증 데이터셋을 구축하기 위해, 우리는 보류 작가의 텍스트를 점수화하기 위해 기존의 인기 있는 형식성 및 감정 분류기를 사용합니다. 또한, Enron 코퍼스 외에도, Reddit Million User Dataset (MUD)에서 프리트레이닝 코퍼스를 구축했습니다. 이 데이터셋은 400,000명의 다양한 Reddit 사용자가 작성한 4백만 개의 댓글을 포함합니다. Enron과 Reddit 데이터셋 모두에서 동일한 패러프레이징 절차를 사용하여 (패러프레이즈, 원본 텍스트) 훈련 쌍을 생성합니다.

---



* Attribute Style Transfer

In this section, we review our evaluation results for attribute transfer. We include representative outputs in the Appendix.

** Automatic Evaluations

Automatic evaluations use various metrics to assess how well the generated text follows the target style. These metrics are evaluated as follows:

1. **Transfer Accuracy**:
   - **Internal Accuracy**: This is the accuracy measured using the classifier that was used during style transfer. For example, if the model generates formal text, it uses a classifier trained to detect formality.
   - **External Accuracy**: This is the accuracy measured using a separate classifier that was not used during training. This evaluates the model's generalization ability.

2. **Similarity**:
   - This measures the semantic similarity between the generated text and the original text. It is often measured using the Mutual Implication Score (MIS), which assesses how semantically aligned the two texts are.

3. **Fluency**:
   - This evaluates the grammatical correctness and coherence of the generated text. It uses a model trained on the CoLA (Corpus of Linguistic Acceptability) dataset.

Tables 1 and 3 present our automatic evaluation results for formality and sentiment transfer. For each approach, we display the average score for each metric, along with the breakdown for formal/informal (→ F, → I) and positive/negative (→ P, → N) transfer.

PARAGUIDE outperforms all other approaches on all aggregate Joint metrics, across both sentiment and formality experiments. Additionally, PARAGUIDE significantly surpasses all baselines on transfer accuracy. Despite the inherent trade-off between transfer accuracy and meaning preservation, on formality, PARAGUIDE (λ = 2e2) outperforms all baseline approaches on both transfer accuracy and meaning preservation. On sentiment transfer, PARAGUIDE’s increased accuracy incurs a larger cost to semantic similarity, but this is expected in successful sentiment transfer, which involves changing the polarity of texts (Jin et al. 2022).

** Human Evaluation

Table 2 displays the results of our human formality evaluation, where annotators rated the Formality, Similarity, and Fluency of model outputs. When evaluated by humans, PARAGUIDE significantly outperforms the top-performing baselines across all aggregate metrics (p = 0.05). Notably, this is even true for the Fluency metric, where annotators rated whether outputs were reasonable, coherent emails. This result was unexpected given PARAGUIDE’s comparatively unimpressive automatic Fluency scores, but could be explained by differences between email writing practices and the composition of the CoLA training corpus (Warstadt, Singh, and Bowman 2019). In contrast, the STRAP baseline dramatically underperforms in our human evaluation. Manually inspecting outputs, we found that the STRAP models we fine-tuned for attribute transfer generate highly repetitive text. We suspect that this results from fine-tuning on our limited dataset, and aligns with previous work, which has shown that STRAP’s performance is heavily reliant on dataset size (Patel, Andrews, and Callison-Burch 2022).

Authorship Style Transfer

Table 4 presents our results on the challenging task of low-resource authorship style transfer. When evaluated with the Style embedding space, three of the four PARAGUIDE configurations outperform every single baseline (including ChatGPT-3.5) on Joint and Confusion. When we consider the holdout UAR embedding space, however, ChatGPT-3.5, which notably uses 400x more parameters than PARAGUIDE, outperforms the other approaches. Considering only non-LLM methods, PARAGUIDE outperforms all baselines on UAR Confusion, but is very narrowly outperformed by STRAP on UAR Joint. This can be attributed to STRAP’s higher Fluency score, which was a metric that was not predictive of human ratings on the formality task. Additionally, in contrast to PARAGUIDE’s plug-and-play approach, the STRAP implementation involves 110 separate models, each

 with 800 million parameters, fine-tuned for every author.

* Evaluation Datasets

We need to evaluate whether our email style transfer approach generalizes to new authors and texts. Therefore, we randomly select 10% of addresses to be the holdout authors for both authorship and attribute evaluations. These 110 authors present a low-resource authorship corpus, as the median holdout author has only 23 emails. For our authorship experiments, we evaluate each approach by selecting up to 5 test emails per holdout source author, and transferring these to 5 other random holdout authors.

To build our training and validation datasets for attribute style transfer, we use popular existing formality and sentiment classifiers to score texts from the holdout authors in the Enron dataset. Critically, we set aside these external classifiers and avoid using them as guidance for PARAGUIDE at inference time. In addition to the Enron corpus, we also build a pretraining corpus from the Reddit Million User Dataset (MUD), which includes 4 million comments by 400k different Reddit users. We use the same paraphrasing procedure on both the Enron and Reddit datasets to generate (paraphrase, original text) training pairs.



<br/>  
# 요약 


PARAGUIDE는 텍스트 스타일 변환을 위한 새로운 확산 기반 프레임워크로, 입력 텍스트의 패러프레이즈를 생성한 후 확산 모델을 사용하여 목표 스타일로 재구성합니다. 자동 평가에서 PARAGUIDE는 형식성과 감정 전환에서 다른 모든 접근법을 능가하며, 인간 평가에서도 유사성, 유창성, 형식성에서 뛰어난 성과를 보였습니다. 특히, 작가 스타일 변환 실험에서도 높은 정확도를 기록하였습니다. 평가 데이터셋은 Enron 이메일 코퍼스와 Reddit Million User Dataset을 사용하여 구축되었습니다.  



PARAGUIDE is a novel diffusion-based framework for text style transfer that generates a paraphrase of the input text and then uses a diffusion model to reconstruct it in the target style. In automatic evaluations, PARAGUIDE outperformed all other approaches in formality and sentiment transfer, and it also showed superior performance in human evaluations for similarity, fluency, and formality. Notably, it achieved high accuracy in authorship style transfer experiments. The evaluation datasets were built using the Enron Email Corpus and the Reddit Million User Dataset.  


<br/>
# 예시  
### 예제 1: 형식성 전환 (Formality Transfer)

#### 원본 텍스트 (Original Text):
"Hey, can you send me the report by tomorrow?"

#### 전환된 텍스트 (Transformed Text):
"안녕하세요, 내일까지 보고서를 보내주시겠습니까?"

#### 영어 (English):
"Hello, could you please send me the report by tomorrow?"

### 예제 2: 감정 전환 (Sentiment Transfer)

#### 원본 텍스트 (Original Text):
"I'm really happy with the results!"

#### 전환된 텍스트 (Transformed Text):
"결과에 정말 실망했습니다."

#### 영어 (English):
"I'm really disappointed with the results!"  

# 기타  
Diffusion 모델은 원래 이미지 생성에서 사용되던 방법으로, 데이터를 노이즈로 변환한 후, 이 노이즈를 점진적으로 제거하면서 원본 데이터를 복원하는 과정입니다. 텍스트 생성에서는 이 개념을 차용하여, 텍스트의 연속적 표현(임베딩)에 노이즈를 추가한 후, 이 노이즈를 단계적으로 제거하면서 원본 텍스트를 복원합니다.   



Diffusion models were originally used for image generation, where data is converted into noise and then progressively denoised to reconstruct the original data. In text generation, this concept is adapted by adding noise to the continuous representation (embedding) of the text and then gradually removing the noise to reconstruct the original text.   


<br/>
# refer format:     
Horvitz, Zachary, Ajay Patel, Chris Callison-Burch, Zhou Yu, and Kathleen McKeown. "ParaGuide: Guided Diffusion Paraphrasers for Plug-and-Play Textual Style Transfer." In The Thirty-Eighth AAAI Conference on Artificial Intelligence (AAAI-24) (https://ojs.aaai.org/index.php/AAAI/article/view/29780/31346)  

  
  
@inproceedings{Horvitz2024,
  title={ParaGuide: Guided Diffusion Paraphrasers for Plug-and-Play Textual Style Transfer},
  author={Zachary Horvitz and Ajay Patel and Chris Callison-Burch and Zhou Yu and Kathleen McKeown},
  booktitle={The Thirty-Eighth AAAI Conference on Artificial Intelligence (AAAI-24)},
  year={2024}
}
   
