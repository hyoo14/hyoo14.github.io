---
layout: post
title:  "[2018]Adversarial Example Generation with Syntactically Controlled Paraphrase Networks"  
date:   2024-05-18 20:43:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 
### Abstract (한글 번역)

우리는 구문을 제어하는 패러프레이즈 네트워크(SCPNs)를 제안하고 이를 사용하여 적대적 예제를 생성합니다. 주어진 문장과 목표 구문 형태(예: 구성 구문 분석)를 가지고 SCPNs는 원하는 구문을 가진 문장의 패러프레이즈를 생성하도록 훈련됩니다. 우리는 먼저 매우 큰 규모로 역번역을 수행한 후, 이 과정에서 자연스럽게 발생하는 구문 변형에 라벨을 붙이는 구문 분석기를 사용하여 이 작업을 위한 훈련 데이터를 생성할 수 있음을 보여줍니다. 이러한 데이터는 목표 구문을 지정하기 위한 추가 입력과 함께 신경 인코더-디코더 모델을 훈련할 수 있게 합니다. 자동화된 평가와 인간 평가를 조합한 결과, SCPNs는 목표 명세를 따르면서 패러프레이즈 품질을 저하시키지 않고 기존의 비제어 패러프레이즈 시스템과 비교할 때 더 나은 성능을 보였습니다. 또한 SCPNs는 (1) 사전 훈련된 모델을 속이고, (2) 훈련 데이터를 증강할 때 구문 변형에 대한 모델의 견고성을 향상시키는 구문 적대적 예제를 생성하는 데 더 능숙합니다.

### Abstract (원문 영어)

We propose syntactically controlled paraphrase networks (SCPNs) and use them to generate adversarial examples. Given a sentence and a target syntactic form (e.g., a constituency parse), SCPNs are trained to produce a paraphrase of the sentence with the desired syntax. We show it is possible to create training data for this task by first doing back-translation at a very large scale, and then using a parser to label the syntactic transformations that naturally occur during this process. Such data allows us to train a neural encoder-decoder model with extra inputs to specify the target syntax. A combination of automated and human evaluations show that SCPNs generate paraphrases that follow their target specifications without decreasing paraphrase quality when compared to baseline (uncontrolled) paraphrase systems. Furthermore, they are more capable of generating syntactically adversarial examples that both (1) “fool” pre-trained models and (2) improve the robustness of these models to syntactic variation when used to augment their training data.

짧은 요약(Abstract) :    


* Useful sentences :  
*   


{% endhighlight %}  

<br/>

[Paper link](https://drive.google.com/drive/folders/1q9FeGH1BjiVOklCE_lkdIILYL1nV-DfT?usp=sharing)  
[Lecture link](https://aclanthology.org/N18-1170.mp4)   

<br/>

# 단어정리  
*  
 
<br/>
# Methodology    
### 방법론 (한글 번역)

일반적인 목적의 구문을 제어하는 패러프레이즈 생성은 도전적인 작업입니다. 

맥키온(1983)의 질문 생성 시스템과 같이 수작업으로 만든 규칙과 문법에 의존하는 접근 방식은 제한된 수의 구문 목표만을 지원합니다.

우리는 이 문제에 대한 첫 번째 학습 접근 방식을 소개하며, 신경 인코더-디코더 모델의 일반성을 바탕으로 다양한 변환을 지원합니다.

이를 통해 두 가지 새로운 도전에 직면하게 됩니다: (1) 훈련을 위해 많은 양의 패러프레이즈 쌍을 확보하는 것, (2) 이러한 쌍에 라벨을 붙일 구문 변환을 정의하는 것.

공개적으로 대규모 문장 패러프레이즈 데이터셋이 존재하지 않기 때문에, 우리는 Wieting et al.(2017)을 따라 신경 역번역을 사용하여 수백만 개의 패러프레이즈 쌍을 자동으로 생성합니다.

역번역은 원래 문장과 역번역된 대응 문장 사이에 자연스럽게 언어적 변화를 주입합니다.

매우 큰 규모로 이 과정을 실행하고 우리가 생성하고자 하는 특정 변화를 테스트함으로써, 다양한 현상에 대한 충분한 입력-출력 쌍을 수집할 수 있습니다.

우리의 초점은 선형화된 구성 구문 분석에서 파생된 템플릿을 사용하여 정의한 구문 변환에 있습니다(§2).

이러한 병렬 데이터를 바탕으로, 우리는 문장과 목표 구문 템플릿을 입력으로 받아 원하는 패러프레이즈를 생성하는 인코더-디코더 모델을 쉽게 훈련할 수 있습니다.

자동화된 평가와 인간 평가를 조합한 결과, 생성된 패러프레이즈가 거의 항상 목표 명세를 따르며, 패러프레이즈 품질이 기존의 신경 역번역(바닐라)과 비교해 크게 저하되지 않음을 보여줍니다(§4).

우리의 모델인 구문 제어 패러프레이즈 네트워크(SCPN)는 사전 훈련된 모델의 성능에 크게 영향을 미치는 감정 분석 및 텍스트 함의 데이터셋에 대한 적대적 예제를 생성할 수 있습니다(그림 1).

또한, 이러한 예제로 훈련 세트를 증강하면 원래 테스트 세트의 정확도를 손상시키지 않으면서 견고성을 향상시킬 수 있음을 보여줍니다(§5).

이 결과는 첫 번째 일반 목적 구문 제어 패러프레이즈 접근 방식을 확립할 뿐만 아니라, 이 일반적인 패러다임이 목표 텍스트의 다른 많은 측면을 제어하는 데 사용할 수 있음을 시사합니다.

### Methods (원문 영어)

General purpose syntactically controlled paraphrase generation is a challenging task.

Approaches that rely on handcrafted rules and grammars, such as the question generation system of McKeown (1983), support only a limited number of syntactic targets.

We introduce the first learning approach for this problem, building on the generality of neural encoder-decoder models to support a wide range of transformations.

In doing so, we face two new challenges: (1) obtaining a large amount of paraphrase pairs for training, and (2) defining syntactic transformations with which to label these pairs.

Since no large-scale dataset of sentential paraphrases exists publicly, we follow Wieting et al. (2017) and automatically generate millions of paraphrase pairs using neural backtranslation.

Backtranslation naturally injects linguistic variation between the original sentence and its back-translated counterpart.

By running the process at a very large scale and testing for the specific variations we want to produce, we can gather ample input-output pairs for a wide range of phenomena.

Our focus is on syntactic transformations, which we define using templates derived from linearized constituency parses (§2).

Given such parallel data, we can easily train an encoder-decoder model that takes a sentence and target syntactic template as input, and produces the desired paraphrase.

A combination of automated and human evaluations show that the generated paraphrases almost always follow their target specifications, while paraphrase quality does not significantly deteriorate compared to vanilla neural backtranslation (§4).

Our model, the syntactically controlled paraphrase network (SCPN), is capable of generating adversarial examples for sentiment analysis and textual entailment datasets that significantly impact the performance of pretrained models (Figure 1).

We also show that augmenting training sets with such examples improves robustness without harming accuracy on the original test sets (§5).

Together these results not only establish the first general purpose syntactically controlled paraphrase approach, but also suggest that this general paradigm could be used for controlling many other aspects of the target text.


<br/>  
# Results  
### 결과 (한글 번역)

결과(표 2)는 SCPN이 대부분의 입력에 대해 구문 제어를 달성했음을 보여줍니다. 우리의 구문 생성기는 거의 항상 목표 템플릿과 일치하는 전체 구문을 생성합니다. 그러나 이러한 구문을 사용하여 생성된 패러프레이즈는 구문 정확도가 떨어집니다. 생성된 구문에 대한 질적 검토 결과, 순서 또는 하위 구성 요소의 존재 측면에서 실제 목표 구문과 다를 수 있음을 보여줍니다. 이러한 차이는 SCPN의 디코더를 혼란스럽게 할 수 있습니다.

NMT-BT 시스템은 입력 문장과 구문적으로 매우 유사한 패러프레이즈를 생성하는 경향이 있습니다: 이러한 패러프레이즈의 28.7%가 입력 문장과 동일한 템플릿을 가지며, 11.1%만이 실제 목표 문장과 동일한 템플릿을 가집니다. 우리는 SCPN을 NMT 역번역으로 생성된 데이터로 훈련함에도 불구하고, 학습 과정에서 구문을 통합하여 이 문제를 피할 수 있습니다.

자동 평가와 인간 평가를 결합한 결과, SCPN이 생성한 패러프레이즈는 거의 항상 목표 명세를 따르며, 패러프레이즈 품질은 기존의 신경 역번역(바닐라)과 비교해 크게 저하되지 않음을 보여줍니다(§4). 또한, SCPN은 사전 훈련된 모델의 성능에 크게 영향을 미치는 감정 분석 및 텍스트 함의 데이터셋에 대한 적대적 예제를 생성할 수 있습니다(그림 1). 이러한 예제로 훈련 세트를 증강하면 원래 테스트 세트의 정확도를 손상시키지 않으면서 견고성을 향상시킬 수 있음을 보여줍니다(§5).

이 결과는 첫 번째 일반 목적 구문 제어 패러프레이즈 접근 방식을 확립할 뿐만 아니라, 이 일반적인 패러다임이 목표 텍스트의 다른 많은 측면을 제어하는 데 사용할 수 있음을 시사합니다.

### Results (원문 영어)

The results (Table 2) show that SCPN does indeed achieve syntactic control over the majority of its inputs. Our parse generator produces full parses that almost always match the target template; however, paraphrases generated using these parses are less syntactically accurate. A qualitative inspection of the generated parses reveals that they can differ from the ground-truth target parse in terms of ordering or existence of lower-level constituents. We theorize that these differences may throw off SCPN’s decoder.

The NMT-BT system produces paraphrases that tend to be syntactically very similar to the input sentences: 28.7% of these paraphrases have the same template as that of the input sentence s1, while only 11.1% have the same template as the ground-truth target s2. Even though we train SCPN on data generated by NMT backtranslation, we avoid this issue by incorporating syntax into our learning process.

A combination of automated and human evaluations show that the generated paraphrases almost always follow their target specifications, while paraphrase quality does not significantly deteriorate compared to vanilla neural backtranslation (§4). Furthermore, SCPNs are capable of generating syntactically adversarial examples that both (1) “fool” pre-trained models and (2) improve the robustness of these models to syntactic variation when used to augment their training data (Figure 1).

Together these results not only establish the first general purpose syntactically controlled paraphrase approach, but also suggest that this general paradigm could be used for controlling many other aspects of the target text.

<br/>  
# 요약  
### 요약 (한글)

이 논문에서는 구문을 제어하는 패러프레이즈 네트워크(SCPNs)를 제안하여 적대적 예제를 생성하는 방법을 소개합니다.

역번역과 구문 분석기를 사용하여 대규모 훈련 데이터를 생성하고, 신경 인코더-디코더 모델을 훈련하여 원하는 구문을 가진 패러프레이즈를 생성합니다.

SCPNs는 기존의 비제어 패러프레이즈 시스템보다 구문 명세를 따르면서도 패러프레이즈 품질이 저하되지 않음을 보여줍니다.

또한, SCPNs는 사전 훈련된 모델을 속이고 훈련 데이터의 견고성을 향상시키는 구문 적대적 예제를 효과적으로 생성합니다.

### Summary (English)

This paper proposes syntactically controlled paraphrase networks (SCPNs) to generate adversarial examples.

By using back-translation and a parser, large-scale training data is generated, and a neural encoder-decoder model is trained to produce paraphrases with the desired syntax.

SCPNs are shown to follow syntactic specifications without degrading paraphrase quality compared to uncontrolled paraphrase systems.

Additionally, SCPNs effectively generate syntactically adversarial examples that both fool pre-trained models and improve the robustness of training data.


<br/>
# 예시  
### 예시 (한글 번역)

표 3은 두 문장에 대해 SCPN이 생성한 구문 제어 패러프레이즈를 보여줍니다.

각 입력 문장에 대해, 네 가지 다른 템플릿의 출력을 보여줍니다.

네 번째 템플릿은 입력 의미를 부적절한 목표 형식에 맞추려 할 때 발생하는 의미적 차이나 비문법적 출력을 포함하는 실패 사례입니다.

감정 분석(왼쪽)과 텍스트 함의(오른쪽)에 대한 SCPN이 생성한 적대적 예제를 보여줍니다.

두 경우 모두, 사전 훈련된 분류기는 원래 문장의 레이블을 올바르게 예측하지만, 해당 패러프레이즈의 레이블은 올바르게 예측하지 못합니다.

### Examples (원문 영어)

Table 3 shows syntactically controlled paraphrases generated by SCPN for two sentences.

For each input sentence, we show the outputs of four different templates.

The fourth template is a failure case, exhibiting semantic divergence and/or ungrammaticality when trying to squeeze the input semantics into an unsuitable target form.

Figure 1 shows adversarial examples for sentiment analysis (left) and textual entailment (right) generated by SCPN.

In both cases, a pretrained classifier correctly predicts the label of the original sentence but not the corresponding paraphrase.

### 예시 (한글 번역)

원본: with the help of captain picard , the borg will be prepared for everything .

패러프레이즈 1: now , the borg will be prepared by picard , will it ?

패러프레이즈 2: the borg here will be prepared for everything .

패러프레이즈 3: with the help of captain picard , the borg will be prepared , and the borg will be prepared for everything ... for everything .

패러프레이즈 4: oh , come on captain picard , the borg line for everything .

원본: you seem to be an excellent burglar when the time comes .

패러프레이즈 1: when the time comes , you ’ll be a great thief .

패러프레이즈 2: “ you seem to be a great burglar , when the time comes . ” you said .

패러프레이즈 3: can i get a good burglar when the time comes ?

패러프레이즈 4: look at the time the thief comes .

### Examples (원문 영어)

Original: with the help of captain picard , the borg will be prepared for everything .

Paraphrase 1: now , the borg will be prepared by picard , will it ?

Paraphrase 2: the borg here will be prepared for everything .

Paraphrase 3: with the help of captain picard , the borg will be prepared , and the borg will be prepared for everything ... for everything .

Paraphrase 4: oh , come on captain picard , the borg line for everything .

Original: you seem to be an excellent burglar when the time comes .

Paraphrase 1: when the time comes , you ’ll be a great thief .

Paraphrase 2: “ you seem to be a great burglar , when the time comes . ” you said .

Paraphrase 3: can i get a good burglar when the time comes ?

Paraphrase 4: look at the time the thief comes .


<br/>
# refre format:     
Iyyer, Mohit, John Wieting, Kevin Gimpel, and Luke Zettlemoyer. "Adversarial Example Generation with Syntactically Controlled Paraphrase Networks." Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers).    