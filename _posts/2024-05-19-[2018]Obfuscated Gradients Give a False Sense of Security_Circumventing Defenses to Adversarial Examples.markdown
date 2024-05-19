---
layout: post
title:  "[2018]Obfuscated Gradients Give a False Sense of Security_Circumventing Defenses to Adversarial Examples"  
date:   2024-05-19 13:35:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

짧은 요약(Abstract) :    
### 요약 (한글)

우리는 경사도 마스킹의 한 형태인 '가려진 경사도(obfuscated gradients)'가 적대적 예제 방어에 대해 잘못된 보안 감각을 초래하는 현상을 식별합니다.

가려진 경사도를 유발하는 방어책이 반복 최적화 기반 공격을 물리치는 것처럼 보이지만, 이러한 효과에 의존하는 방어책은 우회될 수 있습니다.

우리는 이 효과를 나타내는 방어의 특징적인 행동을 설명하고, 우리가 발견한 세 가지 유형의 가려진 경사도 각각에 대해 이를 극복하기 위한 공격 기술을 개발합니다.

ICLR 2018에서 비공인 백색 상자 보안 방어를 조사하는 사례 연구에서 가려진 경사도가 흔히 발생하며, 9개의 방어책 중 7개가 가려진 경사도에 의존하고 있음을 발견했습니다.

새로운 공격으로 원래의 위협 모델을 고려한 각 논문에서 6개의 방어책을 완전히 우회하고 1개를 부분적으로 우회합니다.

---

### Abstract (영어)

We identify obfuscated gradients, a kind of gradient masking, as a phenomenon that leads to a false sense of security in defenses against adversarial examples.

While defenses that cause obfuscated gradients appear to defeat iterative optimization-based attacks, we find defenses relying on this effect can be circumvented.

We describe characteristic behaviors of defenses exhibiting the effect, and for each of the three types of obfuscated gradients we discover, we develop attack techniques to overcome it.

In a case study, examining non-certified white-box-secure defenses at ICLR 2018, we find obfuscated gradients are a common occurrence, with 7 of 9 defenses relying on obfuscated gradients.

Our new attacks successfully circumvent 6 completely, and 1 partially, in the original threat model each paper considers.

* Useful sentences :  
*   


{% endhighlight %}  

<br/>

[Paper link](https://drive.google.com/drive/folders/1XzRbKPmKROkfJM1GAKY_5DBx7WUCxDi1?usp=sharing)  
[~~Lecture link~~]()   

<br/>

# 단어정리  
*  
 
<br/>
# Methodology    

### 방법론 (한글)

우리는 가려진 경사도(obfuscated gradients)가 적대적 예제를 방어하는 방어책에 대해 잘못된 보안 감각을 초래하는 현상을 식별했습니다.

이러한 방어책이 반복 최적화 기반 공격을 물리치는 것처럼 보이지만, 우리는 이러한 효과에 의존하는 방어책이 우회될 수 있음을 발견했습니다.

우리는 이 효과를 나타내는 방어의 특징적인 행동을 설명하고, 우리가 발견한 세 가지 유형의 가려진 경사도 각각에 대해 이를 극복하기 위한 공격 기술을 개발했습니다.

방어가 의도적으로 비차별화(differentiable)를 깨뜨리고 그라디언트 셰터링(gradient shattering), 확률적 경사도(stochastic gradients), 폭발/소실 경사도(vanishing/exploding gradients)로 인해 그라디언트 하강법이 실패하도록 만드는 경우, 이러한 효과가 발생할 수 있습니다.

그라디언트 셰터링은 방어가 비차별화되거나 수치적으로 불안정한 경우, 혹은 의도적으로 잘못된 그라디언트를 생성하는 경우 발생합니다.

확률적 경사도는 네트워크 자체가 무작위화되거나 입력이 무작위로 변환되어 분류기에 공급될 때 발생합니다.

폭발/소실 경사도는 여러 번의 신경망 평가를 포함하는 방어책에서 발생할 수 있으며, 이는 매우 깊은 신경망 평가로 이어져 그라디언트가 소실되거나 폭발하게 만듭니다.

우리는 이러한 세 가지 현상을 극복하기 위해 다음과 같은 새로운 기술을 제안합니다.

- 백워드 패스 분화 가능 근사(Backward Pass Differentiable Approximation, BPDA)
- 변환 기대값(Expectation over Transformation, EOT)
- 재파라미터화(Reparameterization)

백워드 패스 분화 가능 근사는 비차별화 방어책을 공격하기 위해 사용되며, 그라디언트 셰터링을 극복합니다.

변환 기대값은 무작위 변환을 사용하는 방어책을 공격할 때 사용되며, 확률적 경사도를 극복합니다.

재파라미터화는 폭발/소실 경사도를 해결합니다.

### Methodology (영어)

We identify obfuscated gradients, a kind of gradient masking, as a phenomenon that leads to a false sense of security in defenses against adversarial examples.

While defenses that cause obfuscated gradients appear to defeat iterative optimization-based attacks, we find defenses relying on this effect can be circumvented.

We describe characteristic behaviors of defenses exhibiting the effect, and for each of the three types of obfuscated gradients we discover, we develop attack techniques to overcome it.

When defenses intentionally break differentiability and cause gradient shattering, stochastic gradients, or vanishing/exploding gradients, gradient descent fails.

Gradient shattering occurs when a defense is non-differentiable, introduces numerical instability, or intentionally generates incorrect gradients.

Stochastic gradients arise when the network itself is randomized or the input is randomly transformed before being fed to the classifier.

Vanishing/exploding gradients occur in defenses that involve multiple iterations of neural network evaluation, leading to very deep neural network evaluations that cause gradients to vanish or explode.

We propose new techniques to overcome obfuscated gradients caused by these three phenomena.

- Backward Pass Differentiable Approximation (BPDA)
- Expectation over Transformation (EOT)
- Reparameterization

BPDA is used to attack non-differentiable defenses and overcome gradient shattering.

EOT is applied to attack defenses employing randomized transformations, overcoming stochastic gradients.

Reparameterization resolves vanishing/exploding gradients.


<br/>
# Results  
### 결과 (한글)

우리는 ICLR 2018에서 비공인 백색 상자 보안 방어를 조사하는 사례 연구를 사용하여 가려진 경사도의 발생 빈도를 조사하고 이러한 공격 기술의 적용 가능성을 이해합니다.

우리는 9개의 방어책 중 7개가 이 현상에 의존하고 있음을 발견했습니다.

우리가 개발한 새로운 공격 기술을 적용하여 원래의 위협 모델에서 6개의 방어책을 완전히 우회하고 1개를 부분적으로 우회했습니다.

우리는 각 논문에서 수행된 평가에 대한 분석도 제공합니다.

또한 연구자들에게 공통의 지식 기반, 공격 기술 설명 및 공통 평가 함정을 제공하여 미래의 방어책이 이러한 동일한 공격 접근 방식에 취약하지 않도록 돕고자 합니다.

재현 가능한 연구를 촉진하기 위해, 우리는 각 방어책의 재구현과 각 방어에 대한 공격 구현을 공개합니다.

### Results (영어)

We investigate the prevalence of obfuscated gradients and understand the applicability of these attack techniques using a case study of ICLR 2018 non-certified defenses that claim white-box robustness.

We find that obfuscated gradients are a common occurrence, with 7 of 9 defenses relying on this phenomenon.

Applying the new attack techniques we develop, we overcome obfuscated gradients and circumvent 6 of them completely, and 1 partially, under the original threat model of each paper.

Along with this, we offer an analysis of the evaluations performed in the papers.

Additionally, we hope to provide researchers with a common baseline of knowledge, description of attack techniques, and common evaluation pitfalls, so that future defenses can avoid falling vulnerable to these same attack approaches.

To promote reproducible research, we release our re-implementation of each of these defenses, along with implementations of our attacks for each.


<br/>  
# 요약  
### 주요 내용 (한글)

이 논문은 가려진 경사도(obfuscated gradients)라는 현상이 적대적 예제 방어에 대한 잘못된 보안 감각을 초래한다고 설명합니다.

가려진 경사도는 비차별화(differentiable)를 깨뜨리고, 경사도 셰터링(gradient shattering), 확률적 경사도(stochastic gradients), 폭발/소실 경사도(vanishing/exploding gradients)로 인해 발생합니다.

이러한 문제를 극복하기 위해, 우리는 백워드 패스 분화 가능 근사(Backward Pass Differentiable Approximation, BPDA), 변환 기대값(Expectation over Transformation, EOT), 재파라미터화(Reparameterization) 등의 기술을 제안합니다.

예를 들어, BPDA는 비차별화 방어를 공격할 때 유용하며, EOT는 무작위 변환을 사용하는 방어를 극복하는 데 사용됩니다.

우리는 ICLR 2018에서 9개의 방어책 중 7개가 가려진 경사도에 의존하고 있음을 발견하고, 새로운 공격 기술로 6개의 방어책을 완전히 우회했습니다.

### Summary (English)

This paper explains that the phenomenon of obfuscated gradients leads to a false sense of security in defenses against adversarial examples.

Obfuscated gradients occur due to breaking differentiability, causing gradient shattering, stochastic gradients, and vanishing/exploding gradients.

To overcome these issues, we propose techniques such as Backward Pass Differentiable Approximation (BPDA), Expectation over Transformation (EOT), and Reparameterization.

For example, BPDA is useful for attacking non-differentiable defenses, and EOT is used to overcome defenses employing randomized transformations.

We found that 7 out of 9 defenses at ICLR 2018 rely on obfuscated gradients, and we completely circumvented 6 defenses with our new attack techniques.

<br/>
# 예시  
### 판다 예시 (한글)

이 논문은 가려진 경사도(obfuscated gradients)라는 현상이 적대적 예제 방어에 대한 잘못된 보안 감각을 초래한다고 설명합니다.

이를 극복하기 위해 우리는 백워드 패스 분화 가능 근사(Backward Pass Differentiable Approximation, BPDA) 등의 기술을 제안합니다.

예를 들어, 판다 이미지를 식별하는 신경망 모델이 있다고 가정해보겠습니다.

이 모델에 비차별화 방어가 적용되면, 일반적인 경사도 기반 공격으로는 모델을 속일 수 없습니다.

하지만 BPDA를 사용하면 비차별화된 부분을 근사하여 판다 이미지를 다른 동물로 잘못 식별하게 만들 수 있습니다.

### Panda Example (English)

This paper explains that the phenomenon of obfuscated gradients leads to a false sense of security in defenses against adversarial examples.

To overcome this, we propose techniques such as Backward Pass Differentiable Approximation (BPDA).

For example, consider a neural network model that identifies panda images.

If non-differentiable defenses are applied to this model, normal gradient-based attacks cannot deceive the model.

However, by using BPDA, we can approximate the non-differentiable parts and make the model misidentify the panda image as another animal.

<br/>
# refer format:     
Anish Athalye, Nicholas Carlini, and David Wagner. (2018). Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples. Proceedings of the 35th International Conference on Machine Learning, PMLR, 80, 274-283.  