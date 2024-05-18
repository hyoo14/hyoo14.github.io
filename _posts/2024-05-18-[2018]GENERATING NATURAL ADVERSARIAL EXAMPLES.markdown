---
layout: post
title:  "[2018]GENERATING NATURAL ADVERSARIAL EXAMPLES"  
date:   2024-05-18 16:27:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 


짧은 요약(Abstract) :    
### Abstract (한글 번역)

복잡한 특성 때문에, 기계 학습 모델이 배포될 때 어떻게 오작동하거나 악용될 수 있는지 특성화하기 어렵습니다.

경미한 교란으로 인해 모델 예측이 크게 달라지는 입력, 즉 적대적 예제를 최근에 연구하여, 이 모델들이 실패하는 적대적 시나리오를 노출함으로써 이러한 모델의 견고성을 평가하는 데 도움이 됩니다.

그러나 이러한 악의적인 교란은 종종 자연스럽지 않으며, 의미적으로 의미가 없고, 언어와 같은 복잡한 도메인에 적용할 수 없습니다.

이 논문에서는 데이터 다발 위에 위치한 자연스럽고 읽을 수 있는 적대적 예제를 생성하기 위한 프레임워크를 제안합니다.

이는 최근 생성적 적대 신경망의 발전을 활용하여 밀집되고 연속적인 데이터 표현의 의미적 공간에서 검색함으로써 이루어집니다.

우리는 이미지 분류, 텍스트 추론, 기계 번역과 같은 다양한 응용 분야에 대해 블랙 박스 분류기를 대상으로 제안된 접근 방식의 잠재력을 보여주는 생성된 적대적 예제를 제시합니다.

생성된 적대적 예제가 자연스럽고, 인간이 읽을 수 있으며, 블랙 박스 분류기를 평가하고 분석하는 데 유용하다는 실험을 포함합니다.

### Abstract (영문 원문)

Due to their complex nature, it is hard to characterize the ways in which machine learning models can misbehave or be exploited when deployed.

Recent work on adversarial examples, i.e. inputs with minor perturbations that result in substantially different model predictions, is helpful in evaluating the robustness of these models by exposing the adversarial scenarios where they fail.

However, these malicious perturbations are often unnatural, not semantically meaningful, and not applicable to complicated domains such as language.

In this paper, we propose a framework to generate natural and legible adversarial examples that lie on the data manifold, by searching in semantic space of dense and continuous data representation, utilizing the recent advances in generative adversarial networks.

We present generated adversaries to demonstrate the potential of the proposed approach for black-box classifiers for a wide range of applications such as image classification, textual entailment, and machine translation.

We include experiments to show that the generated adversaries are natural, legible to humans, and useful in evaluating and analyzing black-box classifiers.




* Useful sentences :  
*   


{% endhighlight %}  

<br/>

[Paper link](https://drive.google.com/drive/folders/1mcwcPoF37KO2h-4j2WWtG59rRF1Orwcu?usp=sharing)  
[~~Lecture link~~]()   

<br/>

# 단어정리  
*  
 
<br/>
# Methodology    
### 방법론 (한글 번역)

자연어처리를 위한 방법론은 주로 적대적 정규화 오토인코더(ARAE)를 사용하여 이산 텍스트를 연속적인 코드로 인코딩하는 데 기반을 둡니다.

ARAE 모델은 LSTM 인코더를 사용하여 문장을 연속적인 코드로 인코딩하고, 이러한 코드에서 노이즈와 데이터를 사용하여 적대적 훈련을 수행하여 데이터 분포를 추정합니다.

우리는 이러한 연속적인 코드를 z ∈ IR100의 가우시안 공간으로 매핑하는 인버터를 도입합니다.

텍스트를 연속적인 공간 c ∈ IR300로 인코딩하기 위해 4개의 CNN 레이어를 사용하며, 필터 크기(300, 500, 700, 1000), 스트라이드(2, 2, 2) 및 컨텍스트 윈도우(5, 5, 3)를 다양하게 사용합니다.

디코더로는 300차원의 단일 레이어 LSTM을 사용합니다.

우리는 노이즈에서 연속적인 코드로, 연속적인 코드에서 노이즈로의 매핑을 학습하기 위해 두 개의 MLP를 각각 생성기와 인버터로 훈련합니다.

ARAE 모델의 다양한 구성 요소에 대한 손실 함수는 오토인코더 재구성 손실 및 생성기와 비평가의 WGAN 손실 함수로, 식 (4), (5), (6)에 설명되어 있습니다.

먼저 WGAN 전략을 사용하여 ARAE 구성 요소인 인코더, 디코더, 생성기를 훈련한 후, 식 (7)에 있는 손실 함수로 이러한 인코더와 생성기에 대해 인버터를 추가로 훈련합니다.

이 인버터는 역전파를 통해 생성된 연속 코드와 노이즈 샘플 간의 Jensen-Shannon 발산을 최소화합니다.

우리는 Stanford Natural Language Inference (SNLI) 데이터셋의 최대 길이 10인 문장에서 우리의 프레임워크를 훈련합니다.

### Methods (English Original)

The methodology for natural language processing primarily relies on using the Adversarially Regularized Autoencoder (ARAE) to encode discrete text into continuous codes.

The ARAE model encodes a sentence with an LSTM encoder into continuous code and then performs adversarial training on these codes generated from noise and data to approximate the data distribution.

We introduce an inverter that maps these continuous codes into the Gaussian space of z ∈ IR100.

To encode text into continuous space c ∈ IR300, we use 4 layers of CNN with varying filter sizes (300, 500, 700, 1000), strides (2, 2, 2), and context windows (5, 5, 3).

For the decoder, we use a single-layer LSTM with a hidden dimension of 300.

We also train two MLPs, one each for the generator and the inverter, to learn mappings from noise to continuous codes and continuous codes to noise, respectively.

The loss functions for different components of the ARAE model, which are autoencoder reconstruction loss and WGAN loss functions for generator and critic, are described in Equations (4), (5), (6).

We first train the ARAE components of encoder, decoder, and generator using the WGAN strategy, followed by training the inverter on top of these with the loss function in (7), by minimizing the Jensen-Shannon divergence between the inverted continuous codes and noise samples.

We train our framework on sentences up to length 10 from the Stanford Natural Language Inference (SNLI) dataset.

<br/>  
# Results  
### 결과 (한글 번역)

텍스트 함의(Textual Entailment)

텍스트 함의(TE)는 언어를 위한 상식적 추론을 평가하기 위해 설계된 작업으로, 텍스트 단편에 대한 자연어 이해 및 논리적 추론을 요구합니다.

이 작업에서 우리는 전제와 가설 쌍을 분류하여 가설이 전제에 의해 함의되는지, 전제와 모순되는지, 혹은 중립적인지를 판단합니다.

예를 들어, "There are children present" 문장은 "Children smiling and waving at camera" 문장에 의해 함의되고, "The kids are frowning" 문장은 그것과 모순됩니다.

우리는 가설을 속이기 위해 의미 공간에서 가설을 방해하여 분류기를 속이는 적대적 예제를 생성하는 접근법을 사용합니다.

전제는 변경하지 않고 유지합니다.

우리는 단일 층이 평균 단어 임베딩 위에 있는 임베딩 분류기, 단일 층이 문장 표현 위에 있는 LSTM 기반 모델, 그리고 구문 분석에 계층적 LSTM을 사용하는 TE를 위한 최고 성능의 분류기인 TreeLSTM 등 세 가지 분류기를 훈련합니다.

세 가지 분류기를 비교하는 몇 가지 예는 표 3에 나와 있습니다.

세 분류기 모두 라벨을 정확하게 예측하지만, 분류기가 더 정확해질수록 (임베딩에서 LSTM, TreeLSTM으로) 속이기 위해 문장을 더 많이 변경해야 합니다.

기계 번역

기계 번역을 고려하는 이유는 NLP에 대한 신경 접근 방식의 가장 성공적인 응용 중 하나일 뿐만 아니라 대부분의 실용적인 번역 시스템이 블랙 박스 접근 API 뒤에 있기 때문입니다.

그러나 번역 시스템의 출력은 클래스가 아니므로, 여기에서 적대적 예제의 개념은 명확하지 않습니다.

대신 특정 속성에 대해 번역을 테스트하는 탐색 함수에 대해 기계 번역의 적대적 예제를 정의합니다.

이러한 속성은 언어에 대한 언어적 통찰을 제공하거나 잠재적 취약점을 감지할 수 있습니다.

우리는 동일한 생성기와 인버터를 사용하여 Google 번역 모델(2017년 10월 15일 기준)에서 영어에서 독일어로의 API 접근을 통해 이러한 "적대적 예제"를 찾습니다.

먼저 특정 독일어 단어를 독일어 번역에 도입하는 적대적 영어 문장을 생성하려는 시나리오를 고려합니다.

탐색 함수는 번역에서 해당 단어의 존재를 테스트하고, 번역에서 탐색 함수가 통과하면 적대적 예제(영어 문장)를 찾은 것입니다.

우리는 표 4에 "stehen"("stand"를 의미하는 독일어 단어)을 번역에 도입하는 탐색 함수의 예를 제공합니다.

번역 시스템이 매우 강력하기 때문에 이러한 적대적 예제는 모델의 취약점을 드러내지 않지만, 대신 다양한 언어를 이해하거나 배우는 도구로 사용할 수 있습니다 (이 예에서는 독일어 사용자가 영어를 배우는 데 도움을 줄 수 있습니다).

우리는 또한 특정 번역 시스템의 취약점을 겨냥한 더 복잡한 탐색 함수를 설계할 수 있습니다.

예를 들어, 두 개의 능동 동사를 포함하는 영어 문장("People sitting in a restaurant eating")의 번역이 독일어 번역에도 두 개의 동사("essen" 및 "sitzen")가 있는지 확인합니다.

이제 방해된 영어 문장 s'가 두 개의 동사를 포함하지만 번역에는 그 중 하나만 있는 경우에만 통과하는 탐색 함수를 정의합니다.

이러한 탐색 함수에 대한 적대적 예제는 원래 문장(s)과 유사하지만 번역에서 동사 하나가 누락된 영어 문장(s')입니다.

표 5는 이러한 탐색 함수를 사용하여 생성된 적대적 예제의 예를 제공합니다.

예를 들어, "eating"이 원래 문장("People sitting in a living room eating")에 등장할 때 번역에서 "essen"이 누락되는지 테스트하는 탐색 함수입니다.

따라서 이러한 적대적 예제는 Google의 영어-독일어 번역 시스템의 취약점을 제안합니다: 영어 문장에서 동사로 사용되는 단어가 종종 번역에서 누락됩니다.

### Results (English Original)

Textual Entailment

Textual Entailment (TE) is a task designed to evaluate common-sense reasoning for language, requiring both natural language understanding and logical inferences for text snippets.

In this task, we classify a pair of sentences, a premise and a hypothesis, into three categories depending on whether the hypothesis is entailed by the premise, contradicts the premise, or is neutral to it.

For instance, the sentence “There are children present” is entailed by the sentence “Children smiling and waving at camera,” while the sentence “The kids are frowning” contradicts it.

We use our approach to generate adversaries by perturbing the hypothesis to deceive classifiers, keeping the premise unchanged.

We train three classifiers of varying complexity, namely, an embedding classifier that is a single layer on top of the average word embeddings, an LSTM based model consisting of a single layer on top of the sentence representations, and TreeLSTM that uses a hierarchical LSTM on the parses and is a top-performing classifier for this task.

A few examples comparing the three classifiers are shown in Table 3.

Although all classifiers correctly predict the label, as the classifiers get more accurate (from embedding to LSTM to TreeLSTM), they require much more substantial changes to the sentences to be fooled.

Machine Translation

We consider machine translation not only because it is one of the most successful applications of neural approaches to NLP, but also since most practical translation systems lie behind black-box access APIs.

The notion of adversary, however, is not so clear here as the output of a translation system is not a class.

Instead, we define adversary for machine translation relative to a probing function that tests the translation for certain properties, ones that may lead to linguistic insights into the languages, or detect potential vulnerabilities.

We use the same generator and inverter as in entailment, and find such “adversaries” via API access to the currently deployed Google Translate model (as of October 15, 2017) from English to German.

First, let us consider the scenario in which we want to generate adversarial English sentences such that a specific German word is introduced into the German translation.

The probing function here would test the translation for the presence of that word, and we would have found an adversary (an English sentence) if the probing function passes for a translation.

We provide an example of such a probing function that introduces the word “stehen” (“stand” in English) to the translation in Table 4.

Since the translation system is quite strong, such adversaries are not surfacing the vulnerabilities of the model, but instead can be used as a tool to understand or learn different languages (in this example, help a German speaker learn English).

We can design more complex probing functions as well, especially ones that target specific vulnerabilities of the translation system.

Let us consider translations of English sentences that contain two active verbs, e.g. “People sitting in a restaurant eating,” and see that the German translation has the two verbs as well, “essen” and “sitzen,” respectively.

We now define a probing function that passes only if the perturbed English sentence s’ contains both the verbs, but the translation only has one of them.

An adversary for such a probing function will be an English sentence (s’) that is similar to the original sentence (s), but for some reason, its translation is missing one of the verbs.

Table 5 presents examples of generated adversaries using such a probing function.

For example, one that tests whether “essen” is dropped from the translation when its English counterpart “eating” appears in the source sentence (“People sitting in a living room eating”).

These adversaries thus suggest a vulnerability in Google’s English to German translation system: a word acting as a gerund in English often gets dropped from the translation.

<br/>  
# 요약  
### 요약 (한글)

이 논문은 기계 학습 모델의 취약성을 평가하기 위해 자연스럽고 읽을 수 있는 적대적 예제를 생성하는 프레임워크를 제안합니다.

특히 자연어처리를 위해, 적대적 정규화 오토인코더(ARAE)를 사용하여 이산 텍스트를 연속적인 코드로 인코딩한 후, 가우시안 공간에서 적대적 예제를 생성합니다.

이 접근 방식은 텍스트 함의와 기계 번역 작업에 적용되며, 분류기와 번역 시스템의 견고성을 평가하는 데 유용한 통찰을 제공합니다.

실험 결과, 생성된 적대적 예제는 자연스럽고 의미적으로 유사하며, 블랙 박스 모델을 평가하고 분석하는 데 효과적임을 보여줍니다.

### Summary (English)

This paper proposes a framework to generate natural and legible adversarial examples to evaluate the vulnerabilities of machine learning models.

Specifically for natural language processing, it uses the Adversarially Regularized Autoencoder (ARAE) to encode discrete text into continuous codes, and then generates adversarial examples in Gaussian space.

This approach is applied to tasks like textual entailment and machine translation, providing valuable insights into the robustness of classifiers and translation systems.

Experimental results show that the generated adversarial examples are natural, semantically similar, and effective in evaluating and analyzing black-box models.  

# 예시  
적대적 예제를 넣으면 분류 모델이나 번역 시스템의 예측이 원래 예측과 크게 달라집니다. 이는 모델이 실제로는 잘 작동하지 않는 특정 상황이나 입력 패턴을 드러낼 수 있으며, 모델의 취약성을 노출시키는 데 도움이 됩니다.

Classifiers Sentences Label
Original 
p : The man wearing blue jean shorts is grilling. Contradiction 
h : The man is walking his dog.
-> Contradiction  

Embedding h': The man is walking by the dog.     Contradiction → Entailment  
LSTM h' : The person is walking a dog.    Contradiction → Entailment  
TreeLSTM  h': A man is winning a race.     Contradiction → Neutral  


<br/>
# refre format:     
Zhao, Zhengli, Dheeru Dua, and Sameer Singh. "Generating Natural Adversarial Examples." Proceedings of the International Conference on Learning Representations, 2018.