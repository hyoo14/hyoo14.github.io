---
layout: post
title:  "[2018]SYNTHETIC AND NATURAL NOISE BOTH BREAK NEURAL MACHINE TRANSLATION"  
date:   2024-05-18 19:28:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 


짧은 요약(Abstract) :    
### Abstract (Translated to Korean)

문자 기반 신경 기계 번역(NMT) 모델은 어휘 밖 문제를 완화하고 형태론을 학습하며 완전히 종단 간 번역 시스템으로 나아가는 데 도움을 줍니다. 불행히도, 이러한 모델은 매우 취약하며, 노이즈가 있는 데이터를 접했을 때 쉽게 무너집니다. 이 논문에서는 NMT 모델을 인공적 및 자연적인 소음원에 직면시킵니다. 우리는 최신 모델이 인간이 이해하는 데 어려움이 없는 중간 정도의 소음 텍스트를 번역하지 못한다는 것을 발견했습니다. 우리는 모델의 견고성을 높이기 위한 두 가지 접근법을 탐구합니다: 구조 불변 단어 표현과 노이즈 텍스트에 대한 견고한 학습. 문자 컨볼루션 신경망을 기반으로 한 모델이 여러 종류의 소음에 대해 동시에 견고한 표현을 학습할 수 있다는 것을 발견했습니다.

### Abstract (Original in English)

Character-based neural machine translation (NMT) models alleviate out-of-vocabulary issues, learn morphology, and move us closer to completely end-to-end translation systems. Unfortunately, they are also very brittle and easily falter when presented with noisy data. In this paper, we confront NMT models with synthetic and natural sources of noise. We find that state-of-the-art models fail to translate even moderately noisy texts that humans have no trouble comprehending. We explore two approaches to increase model robustness: structure-invariant word representations and robust training on noisy texts. We find that a model based on a character convolutional neural network is able to simultaneously learn representations robust to multiple kinds of noise.



* Useful sentences :  
*   


{% endhighlight %}  

<br/>

[Paper link](https://drive.google.com/drive/folders/1oyr4oZUUqMeEOezAIqvhW20zttfaDYFu?usp=sharing)  
[~~Lecture link~~]()   

<br/>

# 단어정리  
*  
 
<br/>
# Methodology    
### Methodology (Translated to Korean)

우리는 노이즈가 포함된 데이터로 신경 기계 번역(NMT) 모델의 견고성을 높이기 위한 두 가지 접근 방식을 탐구했습니다: 구조 불변 단어 표현과 노이즈 텍스트에 대한 견고한 학습. 

첫 번째 접근 방식은 단어의 구조 불변 표현을 사용하는 것입니다. 이는 문자의 임베딩을 평균화하여 단어 표현을 생성한 다음, 이를 사용하여 단어 수준 인코더를 통해 번역하는 방식입니다. 이러한 모델(meanChar)은 글자 순서가 바뀌는 것에 대해 민감하지 않지만, 다른 종류의 노이즈에는 여전히 민감합니다.

두 번째 접근 방식은 노이즈가 포함된 데이터로 모델을 학습시키는 것입니다. 이를 통해 모델은 다양한 종류의 노이즈에 대해 견고한 표현을 학습할 수 있습니다. 특히, 랜덤 노이즈, 키보드 오타, 자연 발생 인간 오류와 같은 다양한 노이즈를 혼합한 데이터로 학습된 charCNN 모델이 모든 종류의 노이즈에 대해 견고한 성능을 보였음을 발견했습니다.

우리는 또한, charCNN 모델이 다양한 종류의 노이즈에 대해 견고하게 학습할 수 있는지 테스트했습니다. 다양한 필터가 다른 종류의 노이즈에 대해 견고하게 학습되는지 확인하기 위해, 네 가지 조건(랜덤, 키보드 오타, 자연 발생 인간 오류, 그리고 이 세 가지 노이즈가 혼합된 데이터)에서 charCNN 모델이 학습된 가중치를 분석했습니다.

### Methodology (Original in English)

We explore two approaches to increase model robustness when faced with noisy data: structure-invariant word representations and robust training on noisy texts.

The first approach uses structure-invariant representations by averaging character embeddings to create a word representation and then proceeding with a word-level encoder similar to the charCNN model. This model (meanChar) is by definition insensitive to scrambling, though it is still sensitive to other kinds of noise.

The second approach involves training the model on noisy data, enabling it to learn robust representations against various types of noise. We found that a charCNN model trained on a mix of random noise, keyboard typos, and natural human errors was robust to all kinds of noise.

We also tested whether the charCNN model could robustly learn multiple types of noise. To check this, we analyzed the weights learned by charCNN models trained under four conditions: on completely scrambled words (Rand), keyboard typos (Key), natural human errors (Nat), and an ensemble model trained on a mix of Rand+Key+Nat kinds of noise.

<br/>  
# Results  
### Results (Translated to Korean)

charCNN 모델이 모든 종류의 노이즈에 대해 학습할 수 있는지 테스트했습니다.

charCNN 모델은 특정 종류의 노이즈에 대해 학습되었을 때, 그 노이즈에 대해 잘 동작했습니다.

모든 모델은 순수 텍스트에 대해 비교적 좋은 품질을 유지했습니다.

강건한 학습은 노이즈의 종류에 민감합니다.

스크램블링 방법들(Swap/Mid/Rand) 중에서, 더 많은 노이즈가 학습에 도움이 됩니다.

랜덤 노이즈로 학습된 모델은 Swap/Mid 노이즈를 번역할 수 있지만 그 반대는 불가능합니다.

스크램블링, 키보드 오타, 자연 발생 노이즈의 세 가지 범주에서 한 종류의 노이즈로 학습된 모델은 다른 종류의 노이즈에 대해 잘 작동하지 않습니다.

특히, 자연 발생 노이즈로 학습된 모델만이 테스트 시 자연 발생 노이즈를 적절히 번역할 수 있습니다.

이 결과는 인간의 성능과 컴퓨터 모델 간의 중요한 차이를 나타냅니다. 인간은 이러한 형태의 학습 없이도 랜덤 문자 배열을 해독할 수 있습니다.

다음으로, 학습 중 모델에 여러 종류의 노이즈를 노출시켜 학습 견고성을 높일 수 있는지 테스트했습니다.

우리의 동기는 모델이 여러 종류의 노이즈에서 잘 작동할 수 있는지 보는 것이었습니다.

그래서 우리는 각 문장에 대해 무작위로 노이즈 방법을 균일하게 샘플링하여 최대 세 가지 종류의 노이즈를 섞습니다.

그런 다음 혼합된 노이즈 훈련 세트에서 모델을 학습시키고 순수한 테스트 세트와 (혼합되지 않은) 노이즈 버전의 테스트 세트에서 테스트합니다.

혼합된 노이즈로 학습된 모델은 혼합되지 않은 노이즈로 학습된 모델보다 약간 더 나쁩니다.

그러나 혼합된 노이즈로 학습된 모델은 학습된 특정 종류의 노이즈에 대해 견고합니다.

특히, Rand, Key, Nat 노이즈의 혼합으로 학습된 모델은 모든 종류의 노이즈에 견고합니다.

한 가지 노이즈에서 최고는 아니지만 평균적으로 최고의 결과를 얻었습니다.

이 모델은 스크램블된 밈을 상당히 잘 번역할 수 있습니다.

### Results (Original in English)

We tested whether the charCNN model could robustly learn multiple types of noise.

charCNN models trained on a specific kind of noise performed well on the same kind of noise at test time.

All models also maintained fairly good quality on vanilla texts.

The robust training is sensitive to the kind of noise.

Among the scrambling methods (Swap/Mid/Rand), more noise helps in training.

Models trained on random noise can still translate Swap/Mid noise, but not vice versa.

The three broad classes of noise (scrambling, Key, Nat) are not mutually-beneficial.

Models trained on one do not perform well on the others.

In particular, only models trained on natural noise can reasonably translate natural noise at test time.

We find this result indicates an important difference between computational models and human performance, since humans can decipher random letter orderings without explicit training of this form.

Next, we tested whether we could increase training robustness by exposing the model to multiple types of noise during training.

Our motivation was to see if models could perform well on more than one kind of noise.

We therefore mixed up to three kinds of noise by sampling a noise method uniformly at random for each sentence.

We then trained a model on the mixed noisy training set and tested it on both vanilla and (unmixed) noisy versions of the test set.

We found that models trained on mixed noise were slightly worse than models trained on unmixed noise.

However, the models trained on mixed noise were robust to the specific types of noise they were trained on.

In particular, the model trained on a mix of Rand, Key, and Nat noise was robust to all noise kinds.

Even though it is not the best on any one kind of noise, it achieved the best result on average.

This model was also able to translate the scrambled meme reasonably well.

<br/>  
# 요약  
### 요약 (한글)

이 논문은 신경 기계 번역(NMT) 모델이 노이즈 데이터에 대해 매우 취약하다는 것을 발견했습니다.

모델의 견고성을 높이기 위해 두 가지 접근 방식을 탐구했습니다: 구조 불변 단어 표현과 노이즈 텍스트에 대한 견고한 학습.

첫 번째 방법은 문자 임베딩을 평균화하여 단어 표현을 생성하는 것이고, 두 번째 방법은 다양한 노이즈로 학습 데이터를 만들어 모델을 학습시키는 것입니다.

결과적으로, 랜덤 노이즈, 키보드 오타, 자연 발생 오류가 혼합된 데이터로 학습된 charCNN 모델이 모든 종류의 노이즈에 대해 견고한 성능을 보였습니다.

### Summary (English)

This paper finds that neural machine translation (NMT) models are extremely brittle when faced with noisy data.

Two approaches were explored to increase model robustness: structure-invariant word representations and robust training on noisy texts.

The first method involves creating word representations by averaging character embeddings, while the second method trains the model on data containing various types of noise.

As a result, a charCNN model trained on a mix of random noise, keyboard typos, and natural errors showed robust performance against all types of noise.

# 예시  
### 예시 (한글)

논문에서 사용된 몇 가지 예시를 통해 설명해드리겠습니다.

**랜덤 노이즈:**

원문: "According to a research at Cambridge University"

랜덤 노이즈: "Aoccdrnig to a rscheearch at Cmabrigde Uinervtisy"

**키보드 오타:**

원문: "This is a test sentence"

키보드 오타: "This is a test sentenve" (마지막 글자가 'c'에서 'v'로 잘못 입력됨)

**자연 발생 오류:**

원문: "This is another example"

자연 발생 오류: "This is antoher exmaple" (단어의 철자가 잘못됨)

**문자 임베딩을 평균화하여 단어 표현:**

문자 기반 모델에서는 각 문자의 임베딩 벡터를 평균내어 단어 표현을 생성합니다. 예를 들어, "cat"이라는 단어의 경우 'c', 'a', 't' 각 문자의 임베딩 벡터를 평균내어 하나의 단어 표현을 만듭니다.

### Examples (English)

Here are some examples used in the paper:

**Random Noise:**

Original: "According to a research at Cambridge University"

Random Noise: "Aoccdrnig to a rscheearch at Cmabrigde Uinervtisy"

**Keyboard Typos:**

Original: "This is a test sentence"

Keyboard Typos: "This is a test sentenve" (last letter 'c' mistakenly typed as 'v')

**Natural Human Errors:**

Original: "This is another example"

Natural Human Errors: "This is antoher exmaple" (spelling errors in words)

**Averaging Character Embeddings to Create Word Representations:**

In a character-based model, the embeddings of each character are averaged to create the word representation. For example, for the word "cat," the embeddings of 'c', 'a', and 't' are averaged to form a single word representation.

<br/>
# refre format:     
Belinkov, Yonatan, and Yonatan Bisk. "Synthetic and Natural Noise Both Break Neural Machine Translation." Proceedings of the International Conference on Learning Representations, 2018.  