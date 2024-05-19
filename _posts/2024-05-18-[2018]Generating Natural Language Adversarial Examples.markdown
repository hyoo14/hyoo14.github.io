---
layout: post
title:  "[2018]Generating Natural Language Adversarial Examples"  
date:   2024-05-18 23:56:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

짧은 요약(Abstract) :    
### Abstract

딥 신경망(DNN)은 올바르게 분류된 예제에 대한 작은 변형인 적대적 예제에 취약하다.

이미지 분야에서는 이러한 변형이 인간의 인식에 거의 구별되지 않아 인간과 최첨단 모델 간의 의견 차이를 초래한다.

그러나 자연어 분야에서는 작은 변형이 명확하게 인지되며 단어 하나의 교체만으로도 문서의 의미가 크게 달라질 수 있다.

이러한 도전 과제를 해결하기 위해 우리는 잘 훈련된 감정 분석 및 텍스트 함축 모델을 성공적으로 속일 수 있는 의미적으로나 문법적으로 유사한 적대적 예제를 생성하기 위해 블랙박스 기반의 개체군 최적화 알고리즘을 사용한다.

우리는 또한 성공적인 감정 분석 적대적 예제의 92.3%가 20명의 인간 주석자에 의해 원래 레이블로 분류되며 예제가 인지적으로 매우 유사하다는 것을 보여준다.

마지막으로, 방어 수단으로서의 적대적 훈련을 시도했지만 개선에 실패하여 우리의 적대적 예제의 강도와 다양성을 입증한다.

우리는 우리의 발견이 연구자들이 자연어 분야에서 DNN의 강인성을 향상시키는 연구를 계속하도록 장려하기를 바란다.

### Original Abstract

Deep neural networks (DNNs) are vulnerable to adversarial examples, perturbations to correctly classified examples which can cause the model to misclassify.

Deep neural networks (DNNs) are vulnerable to adversarial examples, perturbations to correctly classified examples which can cause the model to misclassify.

In the image domain, these perturbations are often virtually indistinguishable to human perception, causing humans and state-of-the-art models to disagree.

However, in the natural language domain, small perturbations are clearly perceptible, and the replacement of a single word can drastically alter the semantics of the document.

Given these challenges, we use a black-box population-based optimization algorithm to generate semantically and syntactically similar adversarial examples that fool well-trained sentiment analysis and textual entailment models with success rates of 97% and 70%, respectively.

We additionally demonstrate that 92.3% of the successful sentiment analysis adversarial examples are classified to their original label by 20 human annotators, and that the examples are perceptibly quite similar.

Finally, we discuss an attempt to use adversarial training as a defense, but fail to yield improvement, demonstrating the strength and diversity of our adversarial examples.

We hope our findings encourage researchers to pursue improving the robustness of DNNs in the natural language domain.

* Useful sentences :  
*   


{% endhighlight %}  

<br/>

[Paper link](https://drive.google.com/drive/folders/1V7vf64HqQVxM73iYKApYE8dypaJWK3AY?usp=sharing)  
[~~Lecture link~~]()   

<br/>

# 단어정리  
*  
 
<br/>
# Methodology    
### 방법론 (Methodology)

#### 3.1 위협 모델 (Threat Model)

우리는 공격자가 대상 모델에 블랙박스 접근을 가지고 있다고 가정한다.

공격자는 모델 아키텍처, 매개변수 또는 훈련 데이터에 대해 알지 못하며, 공급된 입력을 통해 대상 모델에 쿼리하고 출력 예측 및 신뢰도 점수를 얻을 수만 있다.

이 설정은 이미지 도메인에서 광범위하게 연구되었지만, 자연어의 맥락에서는 아직 탐구되지 않았다.

We assume the attacker has black-box access to the target model.

The attacker is not aware of the model architecture, parameters, or training data, and is only capable of querying the target model with supplied inputs and obtaining the output predictions and their confidence scores.

This setting has been extensively studied in the image domain but has yet to be explored in the context of natural language.


#### 3.2 알고리즘 (Algorithm)

그래디언트 기반 공격 방법의 제한을 피하기 위해, 우리는 다음 목표를 염두에 두고 적대적 예제를 구성하기 위한 알고리즘을 설계한다.

우리는 원본 예제와 적대적 예제 사이의 수정된 단어 수를 최소화하고, 원본의 의미적 유사성과 구문적 일관성을 유지하는 수정만 수행한다.

이 목표를 달성하기 위해, 그래디언트 기반 최적화 대신 유전자 알고리즘을 활용한 개체군 기반 그래디언트 없는 최적화를 이용한 공격 알고리즘을 개발했다.

그래디언트 없는 최적화를 사용하면 블랙박스 경우에도 사용할 수 있는 추가적인 이점이 있다.

그래디언트에 의존하는 알고리즘은 모델이 미분 가능하고 내부 접근이 가능한 경우에만 적용 가능하다.

To avoid the limitations of gradient-based attack methods, we design an algorithm for constructing adversarial examples with the following goals in mind.

We aim to minimize the number of modified words between the original and adversarial examples, but only perform modifications which retain semantic similarity with the original and syntactic coherence.

To achieve these goals, instead of relying on gradient-based optimization, we developed an attack algorithm that exploits population-based gradient-free optimization via genetic algorithms.

An added benefit of using gradient-free optimization is enabling use in the black-box case.

Gradient-reliant algorithms are inapplicable in this case, as they are dependent on the model being differentiable and the internals being accessible.


#### 유전자 알고리즘 (Genetic Algorithms)

유전자 알고리즘은 자연 선택 과정을 모방하여, 후보 해의 집합을 점진적으로 더 나은 해로 진화시킨다.

각 반복의 집합은 세대(generation)라고 한다.

각 세대의 개체 품질은 적합도 함수(fitness function)를 사용하여 평가된다.

"더 적합한(fitter)" 해는 다음 세대를 번식할 때 더 많이 선택될 가능성이 높다.

다음 세대는 교차(crossover)와 돌연변이(mutation)의 조합을 통해 생성된다.

교차는 두 개 이상의 부모 해로부터 자식 해를 생성하는 과정이며, 이는 생식과 생물학적 교차와 유사하다.

돌연변이는 개체의 다양성을 증가시키고 탐색 공간의 더 나은 탐색을 제공하기 위해 수행된다.

유전자 알고리즘은 조합 최적화 문제를 해결하는 데 있어 좋은 성능을 발휘하는 것으로 알려져 있으며, 후보 해의 집합을 사용하여 더 적은 수정으로도 성공적인 적대적 예제를 찾을 수 있다.

Genetic algorithms are inspired by the process of natural selection, iteratively evolving a population of candidate solutions towards better solutions.

The population of each iteration is called a generation.

In each generation, the quality of population members is evaluated using a fitness function.

“Fitter” solutions are more likely to be selected for breeding the next generation.

The next generation is generated through a combination of crossover and mutation.

Crossover is the process of taking more than one parent solution and producing a child solution from them; it is analogous to reproduction and biological crossover.

Mutation is done in order to increase the diversity of population members and provide better exploration of the search space.

Genetic algorithms are known to perform well in solving combinatorial optimization problems, and due to employing a population of candidate solutions, these algorithms can find successful adversarial examples with fewer modifications.


#### 교란 서브루틴 (Perturb Subroutine)

우리 알고리즘을 설명하기 위해, 먼저 교란(Perturb) 서브루틴을 소개한다.

이 서브루틴은 수정된 문장이나 원래 문장이 될 수 있는 입력 문장 xcur를 받아들인다.

이 서브루틴은 문장 xcur에서 단어 w를 무작위로 선택한 다음, 유사한 의미를 가지며 주변 문맥에 적합하고 목표 레이블 예측 점수를 증가시키는 적절한 대체 단어를 선택한다.

최적의 대체 단어를 선택하기 위해, 교란 서브루틴은 다음 단계를 적용한다:

1. GloVe 임베딩 공간에서 선택된 단어의 N개의 가장 가까운 이웃을 계산한다.
우리는 유클리드 거리(euclidean distance)를 사용했으며, 코사인 거리(cosine distance)를 사용했을 때 눈에 띄는 개선을 보지 못했다.
우리는 선택된 단어와의 거리가 δ보다 큰 후보를 필터링한다.
우리는 반대 맞춤 방법(counter-fitting method)을 사용하여 적대적 GloVe 벡터를 후처리하여 가장 가까운 이웃이 동의어가 되도록 한다.
결과 임베딩은 희생자 모델이 사용하는 임베딩과 독립적이다.

2. Google 1 billion words 언어 모델을 사용하여 단어 w 주변의 문맥에 맞지 않는 단어를 필터링한다.
후보 단어를 해당 교체 문맥에 맞춘 언어 모델 점수를 기반으로 순위 매겨 상위 K개의 단어만 유지한다.

3. 남은 단어 집합에서, 단어 w를 xcur에서 대체할 때 목표 레이블 예측 확률을 최대화할 단어를 선택한다.

4. 최종적으로 선택된 단어를 w 대신 삽입하고, 교란 서브루틴은 결과 문장을 반환한다.

To explain our algorithm, we first introduce the subroutine Perturb.

This subroutine accepts an input sentence xcur which can be either a modified sentence or the same as xorig.

It randomly selects a word w in the sentence xcur and then selects a suitable replacement word that has similar semantic meaning, fits within the surrounding context, and increases the target label prediction score.

In order to select the best replacement word, Perturb applies the following steps:

1. Computes the N nearest neighbors of the selected word according to the distance in the GloVe embedding space.

We used euclidean distance, as we did not see noticeable improvement using cosine.

We filter out candidates with distance to the selected word greater than δ.

We use the counter-fitting method to post-process the adversary’s GloVe vectors to ensure that the nearest neighbors are synonyms.

The resulting embedding is independent of the embeddings used by victim models.

2. Second, we use the Google 1 billion words language model to filter out words that do not fit within the context surrounding the word w in xcur.

We do so by ranking the candidate words based on their language model scores when fit within the replacement context, and keeping only the top K words with the highest scores.

3. From the remaining set of words, we pick the one that will maximize the target label prediction probability when it replaces the word w in xcur.

4. Finally, the selected word is inserted in place of w, and Perturb returns the resulting sentence.


#### 최적화 절차 (Optimization Procedure)

최적화 알고리즘은 알고리즘 1에서 볼 수 있다.

알고리즘은 초기 세대 P0를 생성하는 것으로 시작되며, 이를 위해 원본 문장의 다양한 수정을 생성하기 위해 교란 서브루틴을 S번 호출한다.

그런 다음, 현재 세대의 각 개체군 구성원의 적합도는 대상 레이블 예측 확률로 계산된다.

만약 개체군 구성원의 예측 레이블이 목표 레이블과 일치하면 최적화는 완료된다.

그렇지 않으면, 현재 세대에서 적합도 값에 비례하여 개체군 구성원의 쌍을 무작위로 샘플링한다.

그런 다음, 두 부모 문장으로부터 새로운 자식 문장을 균일 분포를 사용하여 독립적으로 샘플링하여 합성한다.

마지막으로, 교란 서브루틴이 생성된 자식 문장에 적용된다.

The optimization algorithm can be seen in Algorithm 1.

The algorithm starts by creating the initial generation P0 of size S by calling the Perturb subroutine S times to create a set of distinct modifications to the original sentence.

Then, the fitness of each population member in the current generation is computed as the target label prediction probability.

If a population member’s predicted label is equal to the target label, the optimization is complete.

Otherwise, pairs of population members from the current generation are randomly sampled with probability proportional to their fitness values.

A new child sentence is then synthesized from a pair of parent sentences by independently sampling from the two using a uniform distribution.

Finally, the Perturb subroutine is applied to the resulting children.




<br/>
# Results  
### 결과 (Results)

#### 4.1 공격 평가 결과 (Attack Evaluation Results)

우리는 우리의 알고리즘을 평가하기 위해 두 과제의 테스트 세트에서 무작위로 1000개의 감정 분석 예제와 500개의 텍스트 함축 예제를 샘플링했다.

올바르게 분류된 예제를 선택하여 희생자 모델의 정확도 수준이 결과에 영향을 미치지 않도록 했다.

감정 분석 과제에서는 공격자가 예측 결과를 긍정에서 부정으로, 또는 그 반대로 변경하는 것을 목표로 한다.

텍스트 함축 과제에서는 공격자가 가설만 수정할 수 있으며, 예측 결과를 '함축'에서 '모순'으로, 또는 그 반대로 변경하는 것을 목표로 한다.

우리는 공격자를 최대 20번의 반복으로 제한하고, 하이퍼파라미터 값을 S=60, N=8, K=4, δ=0.5로 고정했다.

또한 문서에서 허용되는 최대 변경 비율을 두 과제에 대해 각각 20%와 25%로 설정했다.

이 비율을 증가시키면 성공률은 증가하지만 평균 품질은 감소할 것이다.

공격이 반복 한도 내에서 성공하지 못하거나 지정된 임계값을 초과하면 실패로 간주된다.

We randomly sampled 1000, and 500 correctly classified examples from the test sets of the two tasks to evaluate our algorithm.

Correctly classified examples were chosen to limit the accuracy levels of the victim models from confounding our results.

For the sentiment analysis task, the attacker aims to divert the prediction result from positive to negative, and vice versa.

For the textual entailment task, the attacker is only allowed to modify the hypothesis, and aims to divert the prediction result from ‘entailment’ to ‘contradiction’, and vice versa.

We limit the attacker to maximum 20 iterations, and fix the hyper-parameter values to S=60, N=8, K=4, and δ=0.5.

We also fixed the maximum percentage of allowed changes to the document to be 20% and 25% for the two tasks, respectively.

If increased, the success rate would increase but the mean quality would decrease.

If the attack does not succeed within the iterations limit or exceeds the specified threshold, it is counted as a failure.


우리의 공격으로 생성된 샘플 출력은 표 1과 2에 나와 있다.

추가 출력은 보충 자료에서 찾을 수 있다.

표 3은 각 과제에 대한 공격 성공률과 수정된 단어의 평균 비율을 보여준다.

우리는 개체군 기반 최적화의 사용을 검증하기 위해 탐욕적으로 교란 서브루틴을 적용하는 Perturb 기준선과 비교했다.

결과에서 알 수 있듯이, 우리는 두 과제에서 제한된 수정으로 높은 성공률을 달성할 수 있었다.

또한, 유전자 알고리즘은 성공률과 수정된 단어 비율 모두에서 Perturb 기준선을 크게 능가하여 개체군 기반 최적화의 추가적인 이점을 보여준다.

단일 TitanX GPU를 사용하여 감정 분석과 텍스트 함축에 대해 성공 시 평균 실행 시간을 각각 예제당 43.5초와 5초로 측정했다.

높은 성공률과 합리적인 실행 시간은 우리의 접근 방식이 IMDB 데이터셋에서 발견되는 것과 같은 긴 문장으로 확장할 때도 실용적임을 보여준다.

Sample outputs produced by our attack are shown in Tables 1 and 2.

Additional outputs can be found in the supplementary material.

Table 3 shows the attack success rate and mean percentage of modified words on each task.

We compare to the Perturb baseline, which greedily applies the Perturb subroutine, to validate the use of population-based optimization.

As can be seen from our results, we are able to achieve high success rate with a limited number of modifications on both tasks.

In addition, the genetic algorithm significantly outperformed the Perturb baseline in both success rate and percentage of words modified, demonstrating the additional benefit yielded by using population-based optimization.

Testing using a single TitanX GPU, for sentiment analysis and textual entailment, we measured average runtimes on success to be 43.5 and 5 seconds per example, respectively.

The high success rate and reasonable runtimes demonstrate the practicality of our approach, even when scaling to long sentences, such as those found in the IMDB dataset.


텍스트 함축에서 우리의 성공률은 문장 길이의 큰 차이로 인해 낮다.

SNLI 코퍼스에서 평균 가설 문장은 9단어로, IMDB(실험을 위해 100단어로 제한됨)와 비교하여 매우 짧다.

이렇게 짧은 문장에서는 성공적인 교란을 적용하는 것이 훨씬 더 어렵지만, 우리는 여전히 70%의 성공률을 달성할 수 있었다.

같은 이유로, 우리는 텍스트 함축 과제에서 Perturb 기준선을 적용하지 않았는데, Perturb 기준선은 최대 허용 변경 한도 내에서 어떤 성공도 거두지 못했다.

Our success rate on textual entailment is lower due to the large disparity in sentence length.

On average, hypothesis sentences in the SNLI corpus are 9 words long, which is very short compared to IMDB (limited to 100 words for experiments).

With sentences that short, applying successful perturbations becomes much harder, however we were still able to achieve a success rate of 70%.

For the same reason, we didn’t apply the Perturb baseline on the textual entailment task, as the Perturb baseline fails to achieve any success under the limits of the maximum allowed changes constraint.


#### 4.2 사용자 연구 (User Study)

우리는 우리의 적대적 교란이 얼마나 인지될 수 있는지 평가하기 위해 20명의 자원봉사자와 함께 감정 분석 과제에 대한 사용자 연구를 수행했다.

참여한 자원봉사자의 수는 이전 연구들에 비해 상당히 많다.

사용자 연구는 두 부분으로 구성되었다.

첫 번째로, 우리는 참가자들에게 100개의 적대적 예제를 제시하고 텍스트의 감정을 라벨링하도록 요청했다(즉, 긍정 또는 부정).

응답의 92.3%가 원본 텍스트의 감정과 일치하여 우리의 수정이 텍스트 감정에 대한 인간의 판단에 크게 영향을 미치지 않았음을 나타낸다.

두 번째로, 우리는 각 질문에 원본 예제와 해당 적대적 예제가 쌍으로 포함된 100개의 질문을 준비했다.

참가자들은 각 쌍의 유사성을 1(매우 유사)에서 4(매우 다름)까지의 척도로 평가하도록 요청받았다.

평균 평점은 2.23 ± 0.25로, 인지된 차이도 작다는 것을 보여준다.

We performed a user study on the sentiment analysis task with 20 volunteers to evaluate how perceptible our adversarial perturbations are.

Note that the number of participating volunteers is significantly larger than used in previous studies.

The user study was composed of two parts.

First, we presented 100 adversarial examples to the participants and asked them to label the sentiment of the text (i.e., positive or negative.)

92.3% of the responses matched the original text sentiment, indicating that our modification did not significantly affect human judgment on the text sentiment.

Second, we prepared 100 questions, each question includes the original example and the corresponding adversarial example in a pair.

Participants were asked to judge the similarity of each pair on a scale from 1 (very similar) to 4 (very different).

The average rating is 2.23 ± 0.25, which shows the perceived difference is also small.


#### 4.3 적대적 훈련 (Adversarial Training)

섹션 4.1에서 시연된 결과는 다음 질문을 제기한다: 이러한 공격을 어떻게 방어할 수 있을까?

우리는 이미지 도메인에서 유일하게 효과적인 방어인 적대적 훈련이 공격 성공률을 낮출 수 있는지 확인하기 위해 예비 실험을 수행했다.

우리는 IMDB 훈련 세트를 사용하여 깨끗하게 훈련된 감정 분석 모델에서 1000개의 적대적 예제를 생성하고, 이를 기존 훈련 세트에 추가하여 처음부터 모델을 적대적으로 훈련시켰다.

우리는 적대적 훈련이 테스트 세트를 사용한 실험에서 추가적인 강인성 이점을 제공하지 않는다는 것을 발견했다.

모델이 훈련 세트에 포함된 적대적 예제를 거의 100% 정확도로 분류함에도 불구하고 말이다.

이 결과는 우리의 공격 알고리즘이 생성한 교란의 다양성을 보여주며, 적대적 공격에 대한 방어의 어려움을 보여준다.

우리는 이러한 결과가 자연어 모델의 강인성을 높이는 추가 연구를 고무하기를 바란다.

The results demonstrated in section 4.1 raise the following question: How can we defend against these attacks?

We performed a preliminary experiment to see if adversarial training, the only effective defense in the image domain, can be used to lower the attack success rate.

We generated 1000 adversarial examples on the cleanly trained sentiment analysis model using the IMDB training set, appended them to the existing training set, and used the updated dataset to adversarially train a model

 from scratch.

We found that adversarial training provided no additional robustness benefit in our experiments using the test set, despite the fact that the model achieves near 100% accuracy classifying adversarial examples included in the training set.

These results demonstrate the diversity in the perturbations generated by our attack algorithm, and illustrate the difficulty in defending against adversarial attacks.

We hope these results inspire further work in increasing the robustness of natural language models.



<br/>  
# 요약  
이 논문은 딥 신경망(DNN)이 자연어 처리(NLP)에서 적대적 예제에 취약하다는 점을 강조한다.

연구자들은 블랙박스 개체군 최적화 알고리즘을 사용하여 감정 분석 및 텍스트 함축 모델을 속일 수 있는 의미적, 구문적으로 유사한 적대적 예제를 생성했다.

이 알고리즘은 유전자 알고리즘을 활용하여 의미적 유사성과 구문적 일관성을 유지하면서도 모델을 속이는 단어 교체를 수행한다.

실험 결과, 높은 성공률과 적은 단어 수정으로 적대적 예제를 생성할 수 있음을 보여주었으며, 인간 평가자들이 원본 텍스트와 유사하게 분류하는 것으로 나타났다.

적대적 훈련은 이러한 공격에 대한 방어에 효과적이지 않은 것으로 나타났다.

---

This paper highlights the vulnerability of deep neural networks (DNNs) to adversarial examples in natural language processing (NLP).

Researchers used a black-box population-based optimization algorithm to generate semantically and syntactically similar adversarial examples that could deceive sentiment analysis and textual entailment models.

This algorithm employs genetic algorithms to perform word replacements that maintain semantic similarity and syntactic coherence while fooling the models.

Experimental results demonstrated high success rates with minimal word modifications, and human evaluators classified the adversarial examples similarly to the original texts.

Adversarial training was found to be ineffective in defending against these attacks.

<br/>
# 예시  
### 예시

**원본 텍스트 예측: 부정 (신뢰도: 78.0%)**

이 영화는 형편없는 연기, 형편없는 줄거리, 그리고 형편없는 배우 선택이 있었다. (레슬리 닐슨... 정말!!!) 내가 조금 재미있다고 생각한 유일한 부분은 싸우는 FBI/CIA 요원들이었지만, 관객 대부분이 아이들이었기 때문에 그 주제를 이해하지 못했다.

**적대적 텍스트 예측: 긍정 (신뢰도: 59.8%)**

이 영화는 끔찍한 연기, 끔찍한 줄거리, 그리고 무서운 배우 선택이 있었다. (레슬리 닐슨... 정말!!!) 내가 조금 재미있다고 생각한 유일한 부분은 싸우는 FBI/CIA 요원들이었지만, 관객 대부분이 어린이들이었기 때문에 그 주제를 이해하지 못했다.

**Original Text Prediction: Negative (Confidence: 78.0%)**

This movie had terrible acting, terrible plot, and terrible choice of actors. (Leslie Nielsen ...come on!!!) the one part I considered slightly funny was the battling FBI/CIA agents, but because the audience was mainly kids they didn’t understand that theme.

**Adversarial Text Prediction: Positive (Confidence: 59.8%)**

This movie had horrific acting, horrific plot, and horrifying choice of actors. (Leslie Nielsen ...come on!!!) the one part I regarded slightly funny was the battling FBI/CIA agents, but because the audience was mainly youngsters they didn’t understand that theme.

---

이 예시는 원본 텍스트에서 몇 개의 단어만 교체하여 모델의 예측을 바꿀 수 있음을 보여준다.

예를 들어, "terrible"이라는 단어가 "horrific"으로, "kids"가 "youngsters"로 교체되었을 때, 모델은 부정적인 예측에서 긍정적인 예측으로 변환되었다.

이러한 단어 교체는 텍스트의 전체적인 의미를 크게 변경하지 않으면서도 모델을 속이는 데 효과적이었다.

이 접근법은 의미적 유사성과 문법적 일관성을 유지하면서 적대적 예제를 생성하는 데 유용하다.

<br/>
# refer format:     
Alzantot, Moustafa, Yash Sharma, Ahmed Elgohary, Bo-Jhang Ho, Mani Srivastava, and Kai-Wei Chang. "Generating Natural Language Adversarial Examples." Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, October-November 2018.  
