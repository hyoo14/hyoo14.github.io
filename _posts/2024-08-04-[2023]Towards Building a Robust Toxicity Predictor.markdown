---
layout: post
title:  "[2023]Towards Building a Robust Toxicity Predictor"  
date:   2024-08-04 15:04:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

짧은 요약(Abstract) :    



최근의 자연어 처리(NLP) 문헌에서는 독성 언어 예측기의 강건성에 대해 거의 주목하지 않고 있습니다. 그러나 이러한 시스템은 적대적인 맥락에서 사용될 가능성이 가장 높습니다. 이 논문은 SOTA 텍스트 분류기를 속여 독성 텍스트 샘플을 무해한 것으로 예측하도록 하는 작은 단어 수준의 교란을 도입한 새로운 적대적 공격인 ToxicTrap을 제시합니다. ToxicTrap은 탐욕 기반 검색 전략을 활용하여 빠르고 효과적으로 독성 적대적 예제를 생성할 수 있습니다. 두 가지 새로운 목표 함수 설계를 통해 ToxicTrap은 다중 클래스 및 다중 레이블 독성 언어 탐지기의 약점을 식별할 수 있습니다. 우리의 실증 결과는 SOTA 독성 텍스트 분류기가 제안된 공격에 대해 실제로 취약하다는 것을 보여주며, 다중 레이블 경우에서 98% 이상의 공격 성공률을 달성합니다. 우리는 또한 바닐라 적대적 훈련과 그 개선된 버전이 보이지 않는 공격에 대해서도 독성 탐지기의 강건성을 증가시킬 수 있음을 보여줍니다.


Recent NLP literature pays little attention to the robustness of toxicity language predictors, while these systems are most likely to be used in adversarial contexts. This paper presents a novel adversarial attack, ToxicTrap, introducing small word-level perturbations to fool state-of-the-art (SOTA) text classifiers to predict toxic text samples as benign. ToxicTrap exploits greedy-based search strategies to enable fast and effective generation of toxic adversarial examples. Two novel goal function designs allow ToxicTrap to identify weaknesses in both multiclass and multilabel toxic language detectors. Our empirical results show that SOTA toxicity text classifiers are indeed vulnerable to the proposed attacks, attaining over 98% attack success rates in multilabel cases. We also show how vanilla adversarial training and its improved version can help increase the robustness of a toxicity detector even against unseen attacks.

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


이 논문에서는 ToxicTrap이라는 새로운 적대적 공격을 제안하여 SOTA 텍스트 분류기를 속여 독성 텍스트 샘플을 무해한 것으로 예측하도록 합니다. ToxicTrap은 단어 수준의 작은 교란을 도입하여 독성 텍스트 샘플을 생성하고 이를 통해 모델의 취약성을 평가합니다. 

#### 2.1 ToxicTrap의 단어 변환
ToxicTrap은 입력 텍스트에서 단어를 유사어로 교체하는 변환을 중심으로 설계되었습니다. 주요 변환 방법은 다음과 같습니다:
1. GloVe 단어 임베딩 공간에서 가장 가까운 N개의 유사어로 단어를 교체 (N ∈ {5, 20, 50}).
2. BERT 마스크 언어 모델(MLM)로 예측된 단어로 교체.
3. WordNet에서 가장 가까운 유사어로 단어를 교체.

#### 2.2 새로운 목표 함수
목표 함수 G(F,x’)는 공격의 목적을 정의합니다. ToxicTrap 공격의 목표 함수는 다음과 같습니다:
- 다중 클래스 또는 이진 독성 탐지기: G(F,x‘) = {F(x’) = 0;F(x) ≠ 0}
- 다중 라벨 독성 탐지기: G(F,x‘) = {F_b(x’) = 1;F_b(x) = 0; F_t(x‘) = 0, ∀t ∈ T}

#### 2.3 언어 제약
언어 제약을 사용하여 교란된 텍스트 x’가 원래 텍스트 x의 의미와 유창성을 유지하도록 합니다. 사용된 제약 조건은 다음과 같습니다:
- 교란할 단어 비율 10%로 제한
- Universal Sentence Encoder (USE)로 최소 각도 유사도 0.84
- 품사 일치

#### 2.4 탐욕적 검색 전략
ToxicTrap의 탐색 디자인은 단어 중요도 순위를 사용하여 하나의 단어를 교체하고 적대적 예제를 생성하는 탐욕적 검색에 중점을 둡니다. 주요 탐색 전략은 다음과 같습니다:
1. ”unk“ 기반: 단어가 UNK 토큰으로 대체될 때 변경되는 휴리스틱 점수로 단어 중요도 결정.
2. ”delete“ 기반: 단어가 원래 입력에서 삭제될 때 변경되는 휴리스틱 점수로 단어 중요도 결정.
3. ”가중 중요도“ 또는 ”가중 중요도“: 단어가 UNK 토큰으로 대체될 때 점수 변경과 단어 교란 시 최대 점수 획득의 조합으로 단어 중요도 결정.
4. ”gradient“ 기반: 단어 중요도는 단어에 대한 손실의 그래디언트를 사용하여 계산되며 L1 norm으로 중요도를 측정.

#### 2.5 적대적 훈련
ToxicTrap 공격의 궁극적인 목표는 독성 NLP 모델의 적대적 강건성을 향상시키는 것입니다. 적대적 훈련(AT)은 주요 방어 전략으로, 훈련 데이터를 교란하여 생성된 적대적 예제로 보강합니다. 두 가지 변형이 있습니다:
1. 단일 공격 접근 방식으로 생성된 적대적 예제를 사용하는 AT1.
2. 여러 공격 방법으로 생성된 적대적 예제를 사용하는 AT2.

#### 2.6 ToxicTrap 레시피 및 확장
ToxicTrap의 모듈식 디자인을 통해 다양한 공격 레시피를 구현할 수 있으며, 목표 함수, 제약 조건, 변환 및 검색 전략을 결합할 수 있습니다.


This paper introduces a novel adversarial attack called ToxicTrap, designed to deceive state-of-the-art (SOTA) text classifiers into predicting toxic text samples as benign by introducing small word-level perturbations. ToxicTrap evaluates model vulnerabilities through the generation of adversarial toxic text samples.

#### 2.1 Word Transformations in ToxicTrap
ToxicTrap focuses on replacing words from the input with synonyms. Key transformation methods include:
1. Swapping words with their N nearest neighbors in the counter-fitted GloVe word embedding space (N ∈ {5, 20, 50}).
2. Swapping words with those predicted by the BERT Masked Language Model (MLM).
3. Swapping words with their nearest neighbors in WordNet.

#### 2.2 Novel Goal Functions
A goal function G(F,x‘) defines the purpose of an attack. The goal functions for ToxicTrap attacks are:
- For multiclass or binary toxicity detectors: G(F,x’) = {F(x‘) = 0;F(x) ≠ 0}
- For multilabel toxicity detectors: G(F,x’) = {F_b(x‘) = 1;F_b(x) = 0; F_t(x’) = 0, ∀t ∈ T}

#### 2.3 Language Constraints
Language constraints are used to ensure that the perturbed text x‘ preserves the semantics and fluency of the original text x. The constraints used are:
- Limit the ratio of words to perturb to 10%
- Minimum angular similarity from the Universal Sentence Encoder (USE) is 0.84
- Part-of-speech match

#### 2.4 Greedy Search Strategies
The search design of ToxicTrap focuses on iteratively replacing one word at a time using word importance ranking to generate adversarial examples. Key search strategies include:
1. ”unk“ based: Word importance is determined by how much a heuristic score changes when the word is substituted with an UNK token.
2. ”delete“ based: Word importance is determined by how much the heuristic score changes when the word is deleted from the original input.
3. ”weighted saliency“ or ”wt-saliency“: Words are ordered using a combination of the change in score when the word is substituted with an UNK token multiplied by the maximum score gained by perturbing the word.
4. ”gradient“ based: Word importance is calculated using the gradient of the victim’s loss with respect to the word, taking its L1 norm as the word‘s importance.

#### 2.5 Adversarial Training
The ultimate goal of designing ToxicTrap attacks is to improve the adversarial robustness of toxicity NLP models. Adversarial Training (AT) is a primary defense strategy that involves augmenting the training data with adversarial examples generated by perturbing the training data. Two variations exist:
1. AT1: Augmented adversarial examples generated from a single attack approach.
2. AT2: Augmented adversarial examples generated from multiple attack methods.

#### 2.6 ToxicTrap Recipes and Extensions
The modular design of ToxicTrap allows for the implementation of various attack recipes, combining different goal functions, constraints, transformations, and search strategies.


<br/>
# Results  




이 논문에서는 세 가지 독성 예측기 모델(Jigsaw-BL, Jigsaw-ML, HTweet-MC)을 대상으로 ToxicTrap 공격을 수행한 결과를 제공합니다. 주된 결과는 다음과 같습니다:

1. **공격 성공률**:
   - Jigsaw-BL 모델에서, 검색 방법과 품사 일치 제약을 사용한 경우 공격 성공률은 98.74%에서 99.68% 사이였습니다.
   - Jigsaw-ML 모델에서, 공격 성공률은 88.91%에서 99.54% 사이였습니다.
   - HTweet-MC 모델에서, 공격 성공률은 67.16%에서 90.07% 사이였습니다.

2. **질의 수**:
   - Jigsaw-BL 모델에서, 평균 질의 수는 26.78에서 846.41 사이였습니다.
   - Jigsaw-ML 모델에서, 평균 질의 수는 29.78에서 876.81 사이였습니다.
   - HTweet-MC 모델에서, 평균 질의 수는 48.04에서 1025.66 사이였습니다.

3. **단어 변환 비율**:
   - Jigsaw-BL 모델에서, 변형된 단어의 비율은 6.76%에서 8.73% 사이였습니다.
   - Jigsaw-ML 모델에서, 변형된 단어의 비율은 6.96%에서 8.82% 사이였습니다.
   - HTweet-MC 모델에서, 변형된 단어의 비율은 18.76%에서 24.13% 사이였습니다.

4. **적대적 훈련 효과**:
   - AT1-unk 훈련 방법을 사용한 경우, HTweet-MC 모델의 예측 성능은 AUC 0.938, AP 0.785, F1 0.738, Recall 0.723을 유지했습니다.
   - AT2 훈련 방법을 사용한 경우, HTweet-MC 모델의 예측 성능은 AUC 0.932, AP 0.778, F1 0.685, Recall 0.641을 기록했습니다.

#### 주요 결과 요약
ToxicTrap은 다중 레이블 독성 탐지기에서 98% 이상의 공격 성공률을 달성했습니다. 또한, 적대적 훈련은 보이지 않는 공격에 대해서도 독성 탐지기의 강건성을 증가시키는 데 도움이 됨을 보여주었습니다.



The paper presents the results of performing ToxicTrap attacks on three toxicity predictor models (Jigsaw-BL, Jigsaw-ML, HTweet-MC). The main results are as follows:

1. **Attack Success Rate**:
   - For the Jigsaw-BL model, the attack success rate ranged from 98.74% to 99.68%, depending on the search method and the use of part-of-speech (POS) matching constraint.
   - For the Jigsaw-ML model, the attack success rate ranged from 88.91% to 99.54%.
   - For the HTweet-MC model, the attack success rate ranged from 67.16% to 90.07%.

2. **Number of Queries**:
   - For the Jigsaw-BL model, the average number of queries ranged from 26.78 to 846.41.
   - For the Jigsaw-ML model, the average number of queries ranged from 29.78 to 876.81.
   - For the HTweet-MC model, the average number of queries ranged from 48.04 to 1025.66.

3. **Perturbed Word Percentage**:
   - For the Jigsaw-BL model, the percentage of perturbed words ranged from 6.76% to 8.73%.
   - For the Jigsaw-ML model, the percentage of perturbed words ranged from 6.96% to 8.82%.
   - For the HTweet-MC model, the percentage of perturbed words ranged from 18.76% to 24.13%.

4. **Effect of Adversarial Training**:
   - When using the AT1-unk training method, the prediction performance of the HTweet-MC model maintained AUC 0.938, AP 0.785, F1 0.738, and Recall 0.723.
   - When using the AT2 training method, the prediction performance of the HTweet-MC model recorded AUC 0.932, AP 0.778, F1 0.685, and Recall 0.641.

#### Key Findings
ToxicTrap achieved over 98% attack success rates in multilabel toxicity detectors. Additionally, adversarial training helps increase the robustness of toxicity detectors even against unseen attacks   


### Sentence Quality Evaluation 

이 논문에서는 독성 텍스트 샘플의 품질을 평가하기 위해 다양한 언어 제약 조건을 사용합니다. 주요 평가 지표는 다음과 같습니다:

1. **문장 유사도**:
   - Universal Sentence Encoder (USE)와 같은 문장 인코딩을 사용하여 원본 텍스트와 변형된 텍스트 간의 최소 각도 유사도를 0.84 이상으로 유지합니다.

2. **단어 임베딩 코사인 유사도**:
   - 단어 임베딩을 사용하여 원본 단어와 대체된 단어 간의 코사인 유사도를 0.5 이상으로 유지합니다.

3. **단어 변형 비율**:
   - 변형된 단어의 비율을 전체 단어의 10% 미만으로 제한합니다.

4. **품사 일치**:
   - 대체된 단어가 원본 단어와 같은 품사를 가지도록 합니다.

이러한 언어 제약 조건은 변형된 텍스트가 원본 텍스트의 의미와 유창성을 최대한 유지하도록 보장합니다.



The paper evaluates the quality of toxic text samples using various language constraints. The key evaluation metrics are as follows:

1. **Sentence Similarity**:
   - Uses sentence encoding like the Universal Sentence Encoder (USE) to maintain a minimum angular similarity of 0.84 between the original and perturbed text.

2. **Word Embedding Cosine Similarity**:
   - Maintains a cosine similarity of at least 0.5 between the original and substituted words using word embeddings.

3. **Perturbed Word Ratio**:
   - Limits the ratio of perturbed words to less than 10% of the total words.

4. **Part-of-Speech Match**:
   - Ensures that the substituted words have the same part-of-speech as the original words.

These language constraints ensure that the perturbed text retains the semantics and fluency of the original text as much as possible.



<br/>
# 예시  


- 원래 문장: ”My neighbors hate me cuz i smoke weed and blast the most ghetto music f...“
  - 공격 성공 문장: ”My neighbors hate me cuz i smoke pot and blast the most ghetto music from...“
  - 유사도 점수: 0.554

- 원래 문장: ”thank god FOB didnt come back doing some shitty arena tour...“
  - 공격 성공 문장: ”thank god FOB didnt come back doing some foul arena tour...“
  - 유사도 점수: 0.910

- 원래 문장: ”ugh your with Tebow? You’re officially a fag now and in followed.“
  - 공격 성공 문장: ”ugh your with Tebow? You’re officially a poof now and in followed.“
  - 유사도 점수: 0.897



- Original Sentence: ”My neighbors hate me cuz i smoke weed and blast the most ghetto music f...“
  - Adversarial Sentence: ”My neighbors hate me cuz i smoke pot and blast the most ghetto music from...“
  - Similarity Score: 0.554

- Original Sentence: ”thank god FOB didnt come back doing some shitty arena tour...“
  - Adversarial Sentence: ”thank god FOB didnt come back doing some foul arena tour...“
  - Similarity Score: 0.910

- Original Sentence: ”ugh your with Tebow? You’re officially a fag now and in followed.“
  - Adversarial Sentence: ”ugh your with Tebow? You’re officially a poof now and in followed.“
  - Similarity Score: 0.897



<br/>  
# 요약 



이 논문은 독성 언어 탐지기의 강건성을 평가하기 위해 새로운 적대적 공격인 ToxicTrap을 제안합니다. ToxicTrap은 탐욕적 검색 전략을 사용하여 작은 단어 수준의 교란을 통해 독성 텍스트를 무해한 것으로 분류하도록 만듭니다. 주요 평가 지표는 공격 성공률, 평균 질의 수, 변형된 단어의 비율 및 문장 유사도로 구성됩니다. 실험 결과, 독성 텍스트 분류기는 다중 레이블 독성 탐지기에서 98% 이상의 공격 성공률을 보였으며, 적대적 훈련은 모델의 강건성을 향상시킵니다. ToxicTrap은 SOTA 독성 텍스트 분류기의 취약성을 성공적으로 드러냈습니다.



This paper proposes a novel adversarial attack called ToxicTrap to evaluate the robustness of toxicity language predictors. ToxicTrap uses greedy search strategies to introduce small word-level perturbations, making toxic text classified as benign. The main evaluation metrics include attack success rate, average number of queries, percentage of perturbed words, and sentence similarity. Experimental results show that toxicity text classifiers achieve over 98% attack success rates in multilabel toxicity detectors, and adversarial training enhances model robustness. ToxicTrap successfully reveals the vulnerabilities of SOTA toxicity text classifiers.


# 기타  


<br/>
# refer format:     
@inproceedings{Bespalov2023,
  title={Towards Building a Robust Toxicity Predictor},
  author={Bespalov, Dmitriy and Bhabesh, Sourav and Xiang, Yi and Zhou, Liutong and Qi, Yanjun},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics, Volume 5: Industry Track},
  pages={581--598},
  year={2023},
  organization={Association for Computational Linguistics}
}




Bespalov, Dmitriy, Sourav Bhabesh, Yi Xiang, Liutong Zhou, and Yanjun Qi. "Towards Building a Robust Toxicity Predictor." In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics, Volume 5: Industry Track*, 581-598. Association for Computational Linguistics, 2023.


