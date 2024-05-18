---
layout: post
title:  "[2017]Adversarial Examples for Evaluating Reading Comprehension Systems"  
date:   2024-05-18 15:10:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 


짧은 요약(Abstract) :    
### Abstract

표준 정확도 지표는 독해 시스템이 빠르게 발전하고 있음을 나타내지만, 이러한 시스템이 진정으로 언어를 이해하는 정도는 불분명합니다.  

시스템이 실제 언어 이해 능력을 보상하기 위해, 우리는 스탠포드 질문 응답 데이터셋(SQuAD)에 대한 적대적 평가 체계를 제안합니다.  

우리의 방법은 시스템이 답을 변경하지 않거나 사람을 혼동시키지 않고 컴퓨터 시스템을 방해하도록 자동으로 생성된 문장을 포함하는 단락에 대한 질문에 답할 수 있는지 테스트합니다.  

이러한 적대적 설정에서는 16개의 발표된 모델의 정확도가 평균 75%의 F1 점수에서 36%로 떨어집니다. 비문법적 단어 시퀀스를 추가할 수 있는 경우, 4개의 모델의 평균 정확도는 7%로 더욱 감소합니다.  

우리는 이러한 통찰력이 언어를 보다 정확하게 이해하는 새로운 모델 개발을 촉진하기를 바랍니다.  

### Original Abstract

Standard accuracy metrics indicate that reading comprehension systems are making rapid progress, but the extent to which these systems truly understand language remains unclear.  

To reward systems with real language understanding abilities, we propose an adversarial evaluation scheme for the Stanford Question Answering Dataset (SQuAD).  

Our method tests whether systems can answer questions about paragraphs that contain adversarially inserted sentences, which are automatically generated to distract computer systems without changing the correct answer or misleading humans.  

In this adversarial setting, the accuracy of sixteen published models drops from an average of 75% F1 score to 36%; when the adversary is allowed to add ungrammatical sequences of words, average accuracy on four models decreases further to 7%.  

We hope our insights will motivate the development of new models that understand language more precisely.   


* Useful sentences :  
*   


{% endhighlight %}  

<br/>

[Paper link](https://drive.google.com/drive/folders/1s9zUjtwX3LIOeNYAZ4khcSKBCDwEnyKU?usp=sharing)  
[Lecture link](https://aclanthology.org/D17-1215.mp4)   

<br/>

# 단어정리  
*  
 
<br/>
# Methodology    
## 적대적 평가의 일반적인 프레임워크  

표면적인 단서를 의존하면서 언어를 이해하지 못하는 모델은 대부분의 경우 예측에 도움이 되는 단서를 인식함으로써 평균 F1 점수에 따라 성공할 수 있습니다.

기존의 모델이 간단한 패턴을 넘어서서 배웠는지 여부를 확인하기 위해, 우리는 테스트 예제를 변경하여 부족한 모델을 혼동시키는 적대적 평가 방법을 도입합니다.

Figure 1의 예를 고려해보면: BiDAF 앙상블 모델은 원래 올바른 답을 제공했지만, 적대적 방해 문장이 추가되면 혼동됩니다.

우리는 적대자를 (p, q, a) 예제를 받아 새로운 예제 (p', q', a')를 반환하는 함수로 정의합니다.

적대적 정확도는 다음과 같이 정의됩니다: 𝐴𝑑𝑣(𝑓)=(1/∣𝐷𝑡𝑒𝑠𝑡∣)∑(𝑝,𝑞,𝑎)∈𝐷𝑡𝑒𝑠𝑡𝑣(𝐴(𝑝,𝑞,𝑎,𝑓),𝑓))

표준 테스트 오류는 모델이 올바른 답을 얻는 테스트 분포의 비율을 측정하지만, 적대적 정확도는 적대적으로 선택된 변경에도 불구하고 모델이 견고하게 올바른 비율을 측정합니다.

이 양이 의미를 가지려면, 적대자는 두 가지 기본 요구 사항을 충족해야 합니다: 첫째, (p', q', a') 튜플이 유효해야 하며, 인간이 (p', q', a')을 보고 올바른 답이라고 판단할 수 있어야 합니다. 둘째, (p', q', a')은 원래 예제 (p, q, a)와 "가까워야" 합니다.

## 의미를 유지하는 적대자

이미지 분류에서는 적대적 예제가 입력에 눈에 띄지 않는 양의 노이즈를 추가하여 생성됩니다. 이러한 섭동은 이미지의 의미를 변경하지 않지만, 의미를 유지하는 변화에 과민한 모델의 예측을 변경할 수 있습니다.  

언어에 대한 직접적인 유사점은 입력의 패러프레이징입니다. 그러나 높은 정밀도의 패러프레이즈 생성은 어려우며, 대부분의 문장 수정은 실제로 의미를 변경합니다.  

## 연결형 적대자

패러프레이징에 의존하는 대신, 우리는 의미를 변경하는 섭동을 사용하여 연결형 적대자를 구축합니다. 연결형 적대자는 (p + s, q, a) 형식의 예제를 생성합니다.  

연결형 적대자는 새 문장을 단락 끝에 추가하고 질문과 답변은 변경하지 않습니다. 유효한 적대적 예제는 s가 올바른 답과 모순되지 않는 경우입니다.  

기존 모델은 이러한 문장을 실제로 질문을 다루는 문장과 구분하는 데 어려움을 겪으며, 이는 모델이 의미 변경에 과민한 것이 아니라 과잉 안정성을 갖는다는 것을 나타냅니다.  

이제 두 가지 구체적인 연결형 적대자와 두 가지 변형을 설명합니다. ADDSENT는 질문과 유사해 보이는 문법적인 문장을 추가합니다. 반면 ADDANY는 임의의 영어 단어 시퀀스를 추가하여 모델을 혼동시킵니다.  

## ADDSENT  

ADDSENT는 질문과 유사하지만 실제로는 올바른 답과 모순되지 않는 문장을 생성하는 4단계 절차를 사용합니다.

1단계에서, 우리는 질문에 의미를 변경하는 섭동을 적용하여 결과적인 적대적 문장이 호환 가능하도록 보장합니다. 우리는 WordNet의 동의어를 사용하여 명사와 형용사를 교체하고, GloVe 단어 벡터 공간의 가장 가까운 단어로 이름있는 엔티티와 숫자를 변경합니다.

2단계에서, 우리는 원래 답변과 같은 "유형"을 가진 가짜 답변을 생성합니다.

3단계에서, 우리는 변경된 질문과 가짜 답변을 사용하여 서술 형태로 결합합니다.

4단계에서, 우리는 군중 소싱을 통해 이러한 문장의 오류를 수정합니다. 각 문장은 5명의 작업자가 독립적으로 편집하여, 각 원시 문장에 대해 최대 5개의 문장이 생성됩니다. 그런 다음 3명의 추가 작업자가 비문법적이거나 호환되지 않는 문장을 필터링하여, 더 작은 집합의 인간이 승인한 문장을 생성합니다.

## ADDANY

ADDANY의 목표는 문법적이지 않더라도 단어 시퀀스를 선택하는 것입니다. 우리는 지역 탐색을 사용하여 혼란스러운 문장을 적대적으로 선택합니다.

우리는 일반 영어 단어 목록에서 단어를 무작위로 초기화한 다음, 각 단어에 대해 최선의 단어를 선택합니다.

ADDANY는 ADDSENT보다 훨씬 더 많은 모델 접근을 요구합니다.

ADDANY는 원래 답과 모순되지 않는 문장을 생성하는 보장을 하지 않으며, 실제로 생성된 문장은 의미가 없는 문법적이지 않은 단어 시퀀스입니다.


## ADDSENTMOD

ADDSENT의 변형인 ADDSENTMOD는 다른 가짜 답변을 사용하고, 적대적 문장을 단락의 끝에 추가하는 대신 시작 부분에 추가합니다.

------  

## General Framework

A model that relies on superficial cues without understanding language can do well according to average F1 score if these cues happen to be predictive most of the time.

To determine whether existing models have learned much beyond such simple patterns, we introduce adversaries that confuse deficient models by altering test examples.

Consider the example in Figure 1: the BiDAF Ensemble model originally gives the right answer but gets confused when an adversarial distracting sentence is added to the paragraph.

We define an adversary to be a function that takes in an example (p, q, a) and returns a new example (p′, q′, a′).

The adversarial accuracy is defined as: Adv(f)= (1/∣Dtest∣) ∑ (p,q,a)∈D test v(A(p,q,a,f),f)).

While standard test error measures the fraction of the test distribution over which the model gets the correct answer, the adversarial accuracy measures the fraction over which the model is robustly correct, even in the face of adversarially-chosen alterations.

For this quantity to be meaningful, the adversary must satisfy two basic requirements: first, it should always generate (p′, q′, a′) tuples that are valid—a human would judge a′ as the correct answer to q′ given p′. Second, (p′, q′, a′) should be somehow “close” to the original example (p, q, a).

## Semantics-preserving Adversaries

In image classification, adversarial examples are commonly generated by adding an imperceptible amount of noise to the input. These perturbations do not change the semantics of the image, but they can change the predictions of models that are oversensitive to semantics-preserving changes.

For language, the direct analogue would be to paraphrase the input. However, high-precision paraphrase generation is challenging, as most edits to a sentence do actually change its meaning.

## Concatenative Adversaries

Instead of relying on paraphrasing, we use perturbations that do alter semantics to build concatenative adversaries. Concatenative adversaries generate examples of the form (p + s, q, a).

Concatenative adversaries add a new sentence to the end of the paragraph and leave the question and answer unchanged. Valid adversarial examples are precisely those for which s does not contradict the correct answer.

Existing models are bad at distinguishing these sentences from sentences that do in fact address the question, indicating that they suffer not from oversensitivity but from overstability to semantics-altering edits.

Now, we describe two concrete concatenative adversaries, as well as two variants. ADDSENT, our main adversary, adds grammatical sentences that look similar to the question. In contrast, ADDANY adds arbitrary sequences of English words, giving it more power to confuse models.

## ADDSENT

ADDSENT uses a four-step procedure to generate sentences that look similar to the question, but do not actually contradict the correct answer.

In Step 1, we apply semantics-altering perturbations to the question, in order to guarantee that the resulting adversarial sentence is compatible. We replace nouns and adjectives with antonyms from WordNet, and change named entities and numbers to the nearest word in GloVe word vector space.

In Step 2, we create a fake answer that has the same “type” as the original answer.

In Step 3, we combine the altered question and fake answer into declarative form.

In Step 4, we fix errors in these sentences via crowdsourcing. Each sentence is edited independently by five workers on Amazon Mechanical Turk, resulting in up to five sentences for each raw sentence. Three additional crowdworkers then filter out sentences that are ungrammatical or incompatible, resulting in a smaller set of human-approved sentences.

## ADDANY

The goal of ADDANY is to choose any sequence of words, regardless of grammaticality. We use local search to adversarially choose a distracting sentence.

We first initialize words randomly from a list of common English words and then choose the best word for each position.

ADDANY requires significantly more model access than ADDSENT.

ADDANY does not ensure that the sentences generated do not contradict the original answer; in practice, the generated sentences are gibberish sequences of words.

## ADDSENTMOD

ADDSENTMOD, a variant of ADDSENT, uses different fake answers and prepends the adversarial sentence to the beginning of the paragraph instead of appending it to the end.

<br/>  
# Results  
### 결과 (Results)

**주요 실험**

Table 2는 Match-LSTM과 BiDAF 모델이 네 가지 적대자에 대해 어떻게 수행했는지를 보여줍니다.

각 모델은 모든 형태의 적대적 평가에서 상당한 정확도 하락을 겪었습니다.

ADDSENT는 네 모델의 평균 F1 점수를 75.7%에서 31.3%로 감소시켰습니다.

ADDANY는 더 효과적이어서 평균 F1 점수를 6.7%로 떨어뜨렸습니다.

ADDONESENT는 모델 독립적임에도 불구하고 ADDSENT의 효과를 많이 유지했습니다.

마지막으로, ADDCOMMON은 일반적인 단어만 추가했음에도 불구하고 평균 F1 점수를 46.1%로 떨어뜨렸습니다.

우리는 또한 우리의 적대자가 개발 중에 사용되지 않은 모델을 혼동시킬 만큼 일반적이라는 것을 확인했습니다.

우리는 공개된 테스트 시간 코드가 있는 열두 개의 모델에서 ADDSENT를 실행했으며, 모든 모델이 적대적 평가에 대해 강건하지 않음을 확인했습니다. 16개의 모델에 걸쳐 평균 F1 점수는 75.4%에서 36.4%로 떨어졌습니다.

**사람 평가**

우리의 결과가 유효한지 확인하기 위해 사람들도 적대적 예제에 혼동되지 않는지 검증했습니다.

ADDANY는 사람을 대상으로 실행하기에는 모델 쿼리가 너무 많기 때문에 ADDSENT에 집중했습니다.

각 원래와 적대적 단락-질문 쌍을 세 명의 작업자에게 제시하고, 단락에서 복사-붙여넣기로 올바른 답을 선택하게 했습니다.

그런 다음 세 개의 응답에 대해 다수결 투표를 했습니다(모두 다른 경우 무작위로 하나를 선택).

이 결과는 Table 4에 나와 있습니다.

원래 예제에서 우리의 사람들은 전체 개발 세트에서 보고된 91.2 F1보다 약간 더 잘 수행했습니다.

ADDSENT에서는 인간의 정확도가 13.1 F1 포인트 떨어졌지만, 컴퓨터 시스템보다 훨씬 적은 감소를 보였습니다.

게다가, 이 감소의 대부분은 우리의 적대적 문장과 관련이 없는 실수로 설명될 수 있습니다.

**오류 분석**

다음으로, 우리는 적대적 평가에서 우리의 네 가지 주요 모델의 행동을 더 잘 이해하려고 했습니다.

적대자에 의해 발생한 오류를 강조하기 위해, 우리는 모델이 원래 정확한 답을 예측한 예제를 중심으로 분석했습니다.

우리는 이 집합을 "모델 성공"과 "모델 실패"로 나누었습니다.

**ADDSENT 문장의 범주화**

우리는 ADDSENT가 생성한 문장을 수동으로 조사했습니다.

100개의 BiDAF Ensemble 실패 사례 중, 75건은 적대적 문장에서 엔티티 이름이 변경된 경우였고, 17건은 숫자나 날짜가 변경된 경우였으며, 33건은 질문 단어의 반의어가 사용된 경우였습니다.

또한, 군중 소싱 중에 작업자들이 가한 기타 섭동이 있는 7개의 문장이 있었습니다.

**모델 성공의 이유**

마지막으로, 특정 예제에서 모델이 적대적 섭동에 견고한지 여부를 결정하는 요인을 이해하려고 했습니다.

모델은 질문과 원래 단락의 정확한 n-그램 일치가 있는 경우 잘 수행하는 경향이 있음을 발견했습니다.

질문이 짧을수록 모델이 성공할 가능성이 높았습니다.

**모델 간 전이 가능성**

ADDONESENT에서 생성된 예제는 명확하게 모델 간에 전이될 수 있습니다.

ADDSENT와 ADDANY에서 생성된 예제도 다른 모델을 혼동시키는 경향이 있음을 확인했습니다.

**적대적 예제에서의 훈련**

마지막으로, 우리는 적대적 예제에서 훈련하여 기존 모델이 더 강건해질 수 있는지 확인했습니다.

ADDSENT의 경우, 원시 적대 문장을 생성한 다음 원래 훈련 데이터와 결합하여 BiDAF 모델을 훈련시켰습니다.

결과는 Table 6에 나와 있습니다.

ADDSENT에 대해 재훈련된 모델은 거의 방어적이었지만, ADDSENTMOD에 대해 거의 동일한 성능 저하를 보였습니다.

이는 모델이 마지막 문장을 무시하고 ADDSENT가 제안한 가짜 답을 거부하는 것을 학습했음을 시사합니다.

### Original Results

**Main Experiments**

Table 2 shows the performance of the Match-LSTM and BiDAF models against all four adversaries.

Each model incurred a significant accuracy drop under every form of adversarial evaluation.

ADDSENT made average F1 score across the four models fall from 75.7% to 31.3%.

ADDANY was even more effective, making average F1 score fall to 6.7%.

ADDONESENT retained much of the effectiveness of ADDSENT, despite being model-independent.

Finally, ADDCOMMON caused average F1 score to fall to 46.1%, despite only adding common words.

We also verified that our adversaries were general enough to fool models that we did not use during development.

We ran ADDSENT on twelve published models for which we found publicly available test-time code; all models were not robust to adversarial evaluation. Average F1 score fell from 75.4% to 36.4% across the sixteen total models tested.

**Human Evaluation**

To ensure our results are valid, we verified that humans are not also fooled by our adversarial examples.

As ADDANY requires too many model queries to run against humans, we focused on ADDSENT.

We presented each original and adversarial paragraph-question pair to three crowdworkers and asked them to select the correct answer by copy-and-pasting from the paragraph.

We then took a majority vote over the three responses (if all three responses were different, we picked one at random).

These results are shown in Table 4.

On original examples, our humans are actually slightly better than the reported number of 91.2 F1 on the entire development set.

On ADDSENT, human accuracy drops by 13.1 F1 points, much less than the computer systems.

Moreover, much of this decrease can be explained by mistakes unrelated to our adversarial sentences.

**Error Analysis**

Next, we sought to better understand the behavior of our four main models under adversarial evaluation.

To highlight errors caused by the adversary, we focused on examples where the model originally predicted the (exact) correct answer.

We divided this set into "model successes" and "model failures."

**Categorizing ADDSENT Sentences**

We manually examined sentences generated by ADDSENT.

In 100 BiDAF Ensemble failures, we found 75 cases where an entity name was changed, 17 cases where numbers or dates were changed, and 33 cases where an antonym of a question word was used.

Additionally, there were 7 sentences with other perturbations made by crowdworkers during Step 4 of ADDSENT.

**Reasons for Model Successes**

Finally, we sought to understand the factors that influence whether the model will be robust to adversarial perturbations on a particular example.

We found that models do well when the question has an exact n-gram match with the original paragraph.

Models succeeded more often on short questions.

**Transferability Across Models**

Examples from ADDONESENT clearly transfer across models.

Examples generated by ADDSENT and ADDANY also tended to fool other models.

**Training on Adversarial Examples**

Finally, we tried training on adversarial examples to see if existing models can become more robust.

For ADDSENT, we generated raw adversarial sentences and combined them with the original training data to train the BiDAF model.

The results are shown in Table 6.

The retrained model was nearly robust against ADDSENT but performed poorly on ADDSENTMOD.

This suggests the model learned to ignore the last sentence and reject the fake answers proposed by ADDSENT.


<br/>  
# 요약  
이 논문은 독해 시스템의 실제 언어 이해 능력을 평가하기 위해 적대적 평가 체계를 제안합니다.

적대적 평가의 주요 방법론으로는 질문과 유사한 문장을 추가하는 ADDSENT와 임의의 단어 시퀀스를 추가하는 ADDANY가 있습니다.

이 방법들은 모델의 약점을 드러내며, 대부분의 모델은 이 설정에서 성능이 크게 저하됩니다.

연구 결과는 ADDSENT와 ADDANY 모두에서 평균 F1 점수가 크게 떨어짐을 보여주며, 코드와 데이터를 공개하여 후속 연구를 촉진합니다.

---

This paper proposes an adversarial evaluation scheme to assess the true language understanding abilities of reading comprehension systems.

Key methodologies of adversarial evaluation include ADDSENT, which adds sentences similar to the question, and ADDANY, which adds arbitrary sequences of words.

These methods expose weaknesses in the models, with most models showing significant performance drops in this setting.

The research findings show that average F1 scores drop significantly for both ADDSENT and ADDANY, and the code and data are made public to promote further research.

<br/>
# refre format:     
Jia, Robin, and Percy Liang. "Adversarial Examples for Evaluating Reading Comprehension Systems." Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, September 2017.    