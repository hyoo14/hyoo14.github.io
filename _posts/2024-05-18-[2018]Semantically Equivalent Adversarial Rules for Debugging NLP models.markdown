---
layout: post
title:  "[2018]Semantically Equivalent Adversarial Rules for Debugging NLP models"  
date:   2024-05-18 23:20:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

짧은 요약(Abstract) :    
### Abstract 번역 (Korean)

복잡한 자연어 처리(NLP) 모델들은 종종 취약하여, 의미적으로 매우 유사한 입력 인스턴스에 대해 다른 예측을 합니다. 이러한 개별 인스턴스의 행동을 자동으로 감지하기 위해, 우리는 의미를 보존하는 변화가 모델의 예측을 유도하는 의미적으로 동등한 적대적 샘플(SEAs)을 제시합니다. 우리는 이러한 적대적 샘플을 많은 인스턴스에서 적대적 샘플을 유도하는 간단하고 보편적인 대체 규칙으로 일반화한 의미적으로 동등한 적대적 규칙(SEARs)을 제시합니다. 우리는 SEAs와 SEARs의 유용성과 유연성을 기계 이해, 시각적 질문 응답, 그리고 감정 분석을 위한 최신 블랙박스 모델에서 버그를 탐지하는 데 입증합니다. 사용자 연구를 통해 우리는 인간보다 더 많은 인스턴스에 대해 고품질의 로컬 적대적 샘플을 생성하고, SEARs가 인간 전문가가 발견한 버그보다 네 배 더 많은 실수를 유도함을 입증합니다. SEARs는 또한 실용적입니다. 데이터 증강을 사용한 모델 재훈련은 정확도를 유지하면서 버그를 크게 줄입니다.

---

### Abstract (English)

Complex machine learning models for NLP are often brittle, making different predictions for input instances that are extremely similar semantically.

To automatically detect this behavior for individual instances, we present semantically equivalent adversaries (SEAs) – semantic-preserving perturbations that induce changes in the model’s predictions.

We generalize these adversaries into semantically equivalent adversarial rules (SEARs) – simple, universal replacement rules that induce adversaries on many instances.

We demonstrate the usefulness and flexibility of SEAs and SEARs by detecting bugs in black-box state-of-the-art models for three domains: machine comprehension, visual question answering, and sentiment analysis.

Via user studies, we demonstrate that we generate high-quality local adversaries for more instances than humans, and that SEARs induce four times as many mistakes as the bugs discovered by human experts.

SEARs are also actionable: retraining models using data augmentation significantly reduces bugs, while maintaining accuracy.


* Useful sentences :  
*   


{% endhighlight %}  

<br/>

[Paper link](https://drive.google.com/drive/folders/11Db1l1xlNa0cQjBJ_U78cUTncTjJ3KF9?usp=sharing)  
[Lecture link](https://aclanthology.org/P18-1079.mp4)   

<br/>

# 단어정리  
*  
 
<br/>
# Methodology    

### 사용된 방법론 번역 (Korean)

블랙 박스 모델 \( f \)가 문장 \( x \)를 받아 예측 \( f(x) \)을 수행한다고 가정합시다. 우리는 \( x \)의 패러프레이즈를 생성하고 \( f \)로부터 예측을 얻어 원래의 예측이 변경될 때까지 이를 반복하여 적대적 샘플을 식별합니다.

주어진 지표 함수 SemEq(x, x′)는 \( x \)가 \( x′ \)와 의미적으로 동일하면 1이고, 그렇지 않으면 0입니다. 우리는 의미적으로 동일한 인스턴스가 모델 예측을 변경하는 것을 의미적으로 동등한 적대적 샘플(SEA)로 정의합니다 (식 (1)).

이러한 적대적 샘플은 \( f \)의 견고성을 평가하는 데 중요하며, 각 샘플은 바람직하지 않은 버그입니다.

---

### Semantically Equivalent Adversaries (English)

Consider a black box model \( f \) that takes a sentence \( x \) and makes a prediction \( f(x) \), which we want to debug. We identify adversaries by generating paraphrases of \( x \), and getting predictions from \( f \) until the original prediction is changed.

Given an indicator function SemEq(x, x′) that is 1 if \( x \) is semantically equivalent to \( x′ \) and 0 otherwise, we define a semantically equivalent adversary (SEA) as a semantically equivalent instance that changes the model prediction in Eq (1). Such adversaries are important in evaluating the robustness of \( f \), as each is an undesirable bug.

---

### 방법론 세부 설명 (Korean)

패러프레이징은 주변 문맥이 필요하거나 별도의 모델을 훈련해야 하므로, 우리는 대신 신경 기계 번역(Lapata et al., 2017)을 기반으로 패러프레이징을 선택합니다. 이 방법에서 P(x′|x) (원본 문장 \( x \)가 주어졌을 때 패러프레이즈 \( x′ \)의 확률)는 여러 피벗 언어로 \( x \)를 번역한 후 번역들을 다시 원래 언어로 번역하는 점수에 비례합니다. 이 접근법은 의미와 "가능성"을 동시에 평가하며(번역 모델에는 "내장된" 언어 모델이 있음), 각 백-디코더의 경로를 선형적으로 결합하여 패러프레이즈 생성을 쉽게 합니다.

불행히도, 주어진 소스 문장 \( x \)와 \( z \)의 경우, P(x′|x)는 P(z′|z)와 비교할 수 없으며, 각 문장은 다른 정규화 상수를 가지며, \( x \) 또는 \( z \) 주위의 분포 모양에 크게 의존합니다. \( x \) 근처에 여러 개의 완벽한 패러프레이즈가 있으면, 이들은 모두 확률 질량을 공유할 것입니다. 반면, \( z \) 근처에 나머지보다 훨씬 나은 패러프레이즈가 있다면, 품질이 동일하더라도 \( x \) 근처의 것들보다 더 높은 점수를 받을 것입니다. 따라서 우리는 패러프레이즈의 확률과 문장 자체의 확률 사이의 비율로 의미 점수 \( S(x, x′) \)를 정의합니다.

---

### Detailed Methodology (English)

While there are various ways of scoring semantic similarity between pairs of texts based on embeddings (Le and Mikolov, 2014; Wieting and Gimpel, 2017), they do not explicitly penalize unnatural sentences, and generating sentences requires surrounding context (Le and Mikolov, 2014) or training a separate model. We turn instead to paraphrasing based on neural machine translation (Lapata et al., 2017), where \( P(x′|x) \) (the probability of a paraphrase \( x′ \) given original sentence \( x \)) is proportional to translating \( x \) into multiple pivot languages and then taking the score of back-translating the translations into the original language. This approach scores semantics and “plausibility” simultaneously (as translation models have “built in” language models) and allows for easy paraphrase generation, by linearly combining the paths of each back-decoder when back-translating.

Unfortunately, given source sentences \( x \) and \( z \), \( P(x′|x) \) is not comparable to \( P(z′|z) \), as each has a different normalization constant, and heavily depends on the shape of the distribution around \( x \) or \( z \). If there are multiple perfect paraphrases near \( x \), they will all share probability mass, while if there is a paraphrase much better than the rest near \( z \), it will have a higher score than the ones near \( x \), even if the paraphrase quality is the same. We thus define the semantic score \( S(x, x′) \) as a ratio between the probability of a paraphrase and the probability of the sentence itself:

---

### SEAs 생성 및 SEARs로 일반화 (Korean)

의미적으로 동등한 적대적 샘플(SEAs)을 생성하기 위해 우리는 패러프레이즈 생성 기술(Lapata et al., 2017)을 기반으로 하는 접근법을 제시하며, 이는 모델에 무관합니다(즉, 모든 블랙박스 모델에 작동합니다). 다음으로, 우리는 SEAs를 의미적으로 동등한 규칙(SEARs)으로 일반화하고, 최적의 규칙 세트를 위한 속성들을 설명합니다: 의미적 동등성, 높은 적대적 샘플 수, 비중복성. 우리는 이러한 세트를 찾는 문제를 서브모듈 최적화 문제로 설정하여, 정확하면서도 효율적인 알고리즘을 도출합니다.

사람을 루프에 포함시켜, 사용자 연구를 통해 SEARs가 다양한 최신 모델들(감정 분류, 시각적 질문 응답 등)의 중요한 버그를 발견하는 데 도움을 준다는 것을 입증합니다. 우리의 실험 결과 SEAs와 SEARs가 사람보다 더 많은 영향을 미치는 버그를 감지하는 데 크게 기여하며, SEARs는 사람이 발견한 버그보다 3~4배 더 많은 실수를 유도합니다. 마지막으로, 우리는 SEARs가 실용적이라는 것을 보여줍니다. 데이터 증강 절차를 사용한 모델 재훈련은 정확성을 유지하면서 버그를 크게 줄입니다.

---

### Generation of SEAs and Generalization into SEARs (English)

We first present an approach to generate semantically equivalent adversaries (SEAs), based on paraphrase generation techniques (Lapata et al., 2017), that is model-agnostic (i.e., works for any black box model). Next, we generalize SEAs into semantically equivalent rules (SEARs), and outline the properties for optimal rule sets: semantic equivalence, high adversary count, and non-redundancy. We frame the problem of finding such a set as a submodular optimization problem, leading to an accurate yet efficient algorithm.

Including the human into the loop, we demonstrate via user studies that SEARs help users uncover important bugs on a variety of state-of-the-art models for different tasks (sentiment classification, visual question answering). Our experiments indicate that SEAs and SEARs make humans significantly better at detecting impactful bugs – SEARs uncover bugs that cause 3 to 4 times more mistakes than human-generated rules, in much less time. Finally, we show that SEARs are actionable, enabling the human to close the loop by fixing the discovered bugs using a data augmentation procedure.

<br/>
# Results  

### 결과 번역 (Korean)

우리는 자동으로 발견된 SEAs 및 SEARs와 사용자가 생성한 적대적 샘플과 규칙을 비교하고, SEARs로 유도된 버그를 수정하는 방법을 제안합니다.

우리의 평가 벤치마크에는 두 가지 작업이 포함됩니다: 시각적 질문 응답(VQA)과 영화 리뷰 문장의 감정 분석. 우리는 이러한 작업을 선택한 이유는 사용자가 예측이 올바른지 여부를 신속하게 판단할 수 있고, 인스턴스를 쉽게 변경할 수 있으며, 두 인스턴스가 의미적으로 동등한지 여부를 판단할 수 있기 때문입니다. 우리의 초점은 디버깅이므로, 실험 내내 원래 올바르게 예측된 예제에 대해서만 SEAs 및 SEARs를 고려했습니다(즉, 각 적대적 샘플은 구성상 실수입니다). 이 섹션의 모든 실험에 대한 사용자 인터페이스는 보충 자료에 포함되어 있습니다.

---

### Results (English)

We compare automatically discovered SEAs and SEARs to user-generated adversaries and rules, and propose a way to fix the bugs induced by SEARs.

Our evaluation benchmark includes two tasks: visual question answering (VQA) and sentiment analysis on movie review sentences. We choose these tasks because a human can quickly look at a prediction and judge if it is correct or incorrect, can easily perturb instances, and judge if two instances in a pair are semantically equivalent or not. Since our focus is debugging, throughout the experiment we only considered SEAs and SEARs on examples that are originally predicted correctly (i.e. every adversary is also by construction a mistake). The user interfaces for all experiments in this section are included in the supplementary material.

---

### 구현 세부 사항 번역 (Korean)

패러프레이징 모델(Lapata et al., 2017)은 다른 언어로의 번역 모델이 필요합니다. 우리는 EuroParl, 뉴스 및 기타 소스(Tiedemann, 2012)에서 2백만 및 1백만 평행 문장에 대해 기본 매개변수로 OpenNMT-py(Klein et al., 2017)를 사용하여 영어↔포르투갈어 및 영어↔프랑스어 모델을 훈련시켰습니다. 우리는 spacy 라이브러리(http://spacy.io)를 품사 태깅에 사용합니다. SEAR 생성의 경우, 우리는 δ = 0.1(즉, 적어도 90%의 동등성)을 설정합니다. 우리는 후보 적대적 샘플 세트를 생성하고, 기계 작업자에게 그것들을 의미적으로 동등한지 판단하게 합니다.

---

### Implementation Details (English)

The paraphrasing model (Lapata et al., 2017) requires translation models to and from different languages. We train neural machine translation models using the default parameters of OpenNMT-py (Klein et al., 2017) for English↔Portuguese and English↔French models, on 2 million and 1 million parallel sentences (respectively) from EuroParl, news, and other sources (Tiedemann, 2012). We use the spacy library (http://spacy.io) for POS tagging. For SEAR generation, we set δ = 0.1 (i.e. at least 90% equivalence). We generate a set of candidate adversaries as described in Section 2, and ask mechanical turkers to judge them.

---

### 실험 결과 번역 (Korean)

자동으로 생성된 SEAs 및 SEARs를 사용자가 생성한 적대적 샘플과 규칙과 비교한 결과는 일관적이었습니다. 두 가지 작업 모두에서 모델은 많은 비율의 예측에 대해 SEAs에 취약했으며, 완전 자동화된 방법이 사람만큼 자주 SEAs를 생성할 수 있음을 보여주었습니다. 한편, 생성된 SEAs에서 사용자가 선택한 경우(HSEA)는 사용자가 적대적 샘플을 생성하거나 가장 높은 점수의 SEAs를 사용하는 것보다 훨씬 더 좋은 결과를 냈습니다. 의미적 평가자는 실수를 저지르므로 최상위 적대적 샘플이 항상 의미적으로 동등하지는 않지만, 좋은 품질의 SEA는 종종 상위 5위 안에 있으며 사용자가 쉽게 식별할 수 있습니다.

---

### Experimental Results (English)

The results in Table 4a and 4b are consistent across tasks: both models are susceptible to SEAs for a large fraction of predictions, and our fully automated method is able to produce SEAs as often as humans (left columns). On the other hand, asking humans to choose from generated SEAs (HSEA) yields much better results than asking humans to generate them (right columns), or using the highest scored SEA. The semantic scorer does make mistakes, so the top adversary is not always semantically equivalent, but a good quality SEA is often in the top 5, and is easily identified by users.

---

### 전문가와의 비교 번역 (Korean)

여기에서는 전문가들이 많은 예측을 뒤집는 규칙을 고안할 수 있는지, 그리고 생성된 SEARs와 비교하여 높은 영향을 미치는 글로벌 버그를 감지할 수 있는지 조사합니다. AMT 작업자 대신, 우리는 최소한 기계 학습 또는 NLP에서 대학원 과정을 이수한 학생, 졸업생 또는 교수를 포함한 26명의 전문가를 대상으로 합니다. 각 작업에 대해 피험자는 검증 데이터의 예제를 보고, 그 예제를 변경하고, 예측을 얻을 수 있는 인터페이스를 받습니다. 인터페이스는 또한 검색 및 대체 규칙을 만들 수 있게 하며, 규칙이 유도하는 실수의 수에 대한 즉각적인 피드백을 제공합니다. 그들은 또한 규칙이 적용되는 예제 목록을 보며 의미적 동등성을 확인할 수 있습니다. 피험자는 검증 데이터에서 유도된 실수의 수를 최대화하기 위해(즉, "실수 커버리지"를 최대화) 노력하라는 지시를 받지만, 의미적으로 동등한 규칙을 통해서만 가능합니다. 그들은 원하는 만큼 많은 규칙을 시도할 수 있으며, 마지막에 최대 10개의 규칙을 선택하라는 요청을 받습니다. 이는 인간에게는 상당히 도전적인 작업입니다(알고리즘 접근 방식을 선호하는 또 다른 이유), 하지만 우리는 기존의 자동화된 방법을 알지 못합니다.

---

### Comparison with Experts (English)

Here we investigate whether experts are able to detect high-impact global bugs, i.e. devise rules that flip many predictions, and compare them to generated SEARs. Instead of AMT workers, we have 26 expert subjects: students, graduates, or professors who have taken at least a graduate course in machine learning or NLP. The experiment setup is as follows: for each task, subjects are given an interface where they see examples in the validation data, perturb those examples, and get predictions. The interface also allows them to create search and replace rules, with immediate feedback on how many mistakes are induced by their rules. They also see the list of examples where the rules apply, so they can verify semantic equivalence. Subjects are instructed to try to maximize the number of mistakes induced in the validation data (i.e. maximize “mistake coverage”), but only through semantically equivalent rules. They can try as many rules as they like, and are asked to select the best set of at most 10 rules at the end. This is quite a challenging task for humans (yet another reason to prefer algorithmic approaches), but we are not aware of any existing automated methods.

---

### SEARs로 버그 수정 번역 (Korean)

이러한 버그가 발견되면 자연스럽게 그것들을 수정하고 싶어집니다. SEARs의 글로벌하고 결정론적인 특성은 그것들을 체계적인 방식으로 나타내므로 실용적입니다. 영향력이 큰 버그가 식별되면, 우리는 간단한 데이터 증강 절차를 사용합니다: SEARs를 훈련 데이터에 적용하고, 생성된 예제로 원래 훈련 데이터를 증강하여 모델을 다시 훈련합니다.

우리는 ≥ 20명의 피험자가 버그로 받아들인 규칙을 취하여, VQA의 경우 4개의 규칙(표 2에 있음)과 감정 분석의 경우 16개의 규칙(표 3 포함)을 사용합니다. 그런 다음 이 규칙을 훈련 데이터에 적용하고 모델을 다시 훈련합니다. 버그가 여전히 존재하는지 확인하기 위해, 검증에서 올바르게 예측된 인스턴스에 SEARs를 적용하여 민감도 데이터를 생성합니다. 이러한 규칙에 의해 설명된 버그에 취약하지 않은 모델은 그 예측을 변경하지 않아야 하며, 따라서 이 민감도 데이터에 대한 오류율은 0%여야 합니다. 우리는 또한 원래 검증 데이터에 대한 정확성을 측정하여, 버그 수정 절차가 정확성을 떨어뜨리지 않는지 확인합니다.

표 6은 증강 후 이러한 오류의 발생률이 크게 감소했으며, 검증 정확성에 미미한 변화가 있음을 보여줍니다(두 작업 모두에서, 이러한 변화는 다른 시드로 재훈련하는 효과와 일치합니다). 이러한 결과는 SEARs가 버그를 발견하는 데 유용할 뿐만 아니라, 모든 모델

에 대한 간단한 증강 기법을 통해 실용적이라는 것을 보여줍니다.

---

### Fixing Bugs Using SEARs (English)

Once such bugs are discovered, it is natural to want to fix them. The global and deterministic nature of SEARs make them actionable, as they represent bugs in a systematic manner. Once impactful bugs are identified, we use a simple data augmentation procedure: applying SEARs to the training data, and retraining the model on the original training augmented with the generated examples.

We take the rules that are accepted by ≥ 20 subjects as accepted bugs, a total of 4 rules (in Table 2) for VQA, and 16 rules for sentiment (including ones in Table 3). We then augment the training data by applying these rules to it, and retrain the models. To check if the bugs are still present, we create a sensitivity dataset by applying these SEARs to instances predicted correctly on the validation. A model not prone to the bugs described by these rules should not change any of its predictions, and should thus have error rate 0% on this sensitivity data. We also measure accuracy on the original validation data, to make sure that our bug-fixing procedure is not decreasing accuracy.

Table 6 shows that the incidence of these errors is greatly reduced after augmentation, with negligible changes to the validation accuracy (on both tasks, the changes are consistent with the effect of retraining with different seeds). These results show that SEARs are useful not only for discovering bugs, but are also actionable through a simple augmentation technique for any model.


<br/>  
# 요약  
### 논문의 주요 포인트 요약 (Korean)

이 논문은 자연어 처리(NLP) 모델의 버그를 탐지하기 위해 의미를 보존하는 적대적 샘플(SEAs)과 의미적으로 동등한 적대적 규칙(SEARs)을 제시합니다.

SEAs는 패러프레이징을 사용하여 모델의 예측을 변경하는 입력을 생성하며, SEARs는 이러한 샘플을 일반화한 규칙입니다.

이 방법론은 기계 이해, 시각적 질문 응답, 감정 분석 등의 다양한 모델에서 버그를 감지하는 데 효과적임을 보여줍니다.

또한, 데이터 증강을 통해 이러한 버그를 수정할 수 있음을 입증합니다.

---

### Summary of Key Points (English)

This paper presents semantically equivalent adversaries (SEAs) and semantically equivalent adversarial rules (SEARs) to detect bugs in natural language processing (NLP) models.

SEAs generate inputs that change the model's predictions using paraphrasing, and SEARs generalize these samples into rules.

This methodology is shown to be effective in detecting bugs in various models, including machine comprehension, visual question answering, and sentiment analysis.

Additionally, it demonstrates that these bugs can be fixed through data augmentation.


<br/>
# 예시  
### 예시와 설명 (Korean)

**예시:**

원본 질문: "What has been the result of this publicity?"

대답: "increased scrutiny on teacher misconduct."

변형된 질문: "What’s been the result of this publicity?"

대답: "teacher misconduct."

---

### Explanation (English)

**Example:**

Original question: "What has been the result of this publicity?"

Answer: "increased scrutiny on teacher misconduct."

Changed question: "What’s been the result of this publicity?"

Answer: "teacher misconduct."

---

### 예시 설명 (Korean)

이 예시는 SEAs와 SEARs가 NLP 모델에서 버그를 어떻게 유도하는지 보여줍니다. 원본 질문 "What has been the result of this publicity?"는 올바른 답변 "increased scrutiny on teacher misconduct"을 생성합니다. 그러나 "What’s been the result of this publicity?"로 단순히 축약형을 사용한 변형된 질문은 잘못된 답변 "teacher misconduct"을 생성합니다. 이는 의미적으로 동등한 변형이 모델의 예측을 어떻게 변경할 수 있는지를 보여주며, SEARs가 모델의 전반적인 취약성을 발견하는 데 유용하다는 것을 강조합니다.

---

### Explanation (English)

This example demonstrates how SEAs and SEARs can induce bugs in NLP models. The original question "What has been the result of this publicity?" produces the correct answer "increased scrutiny on teacher misconduct." However, simply using the contraction "What’s been the result of this publicity?" in the changed question results in the incorrect answer "teacher misconduct." This shows how semantically equivalent transformations can alter a model's prediction, highlighting the usefulness of SEARs in discovering overall vulnerabilities in the model.

### SEAs와 SEARs 예시 (Korean)

**예시 1: SEAs**

원본 질문: "What color is the tray?"

대답: "Pink"

변형된 질문 1: "What colour is the tray?"

대답: "Green"

변형된 질문 2: "Which color is the tray?"

대답: "Green"

변형된 질문 3: "What color is it?"

대답: "Green"

변형된 질문 4: "How color is tray?"

대답: "Green"

---

**예시 2: SEARs**

**규칙:**

변환 규칙: "WP is→WP’s"

변환 전: "What is the oncorhynchus also called?"

대답: "chum salmon"

변환 후: "What’s the oncorhynchus also called?"

대답: "keta"

---

**규칙:**

변환 규칙: "?→??"

변환 전: "How long is the Rhine?"

대답: "1,230 km"

변환 후: "How long is the Rhine??"

대답: "more than 1,050,000"

---

### Example of SEAs and SEARs (English)

**Example 1: SEAs**

Original question: "What color is the tray?"

Answer: "Pink"

Changed question 1: "What colour is the tray?"

Answer: "Green"

Changed question 2: "Which color is the tray?"

Answer: "Green"

Changed question 3: "What color is it?"

Answer: "Green"

Changed question 4: "How color is tray?"

Answer: "Green"

---

**Example 2: SEARs**

**Rule:**

Transformation rule: "WP is→WP’s"

Before transformation: "What is the oncorhynchus also called?"

Answer: "chum salmon"

After transformation: "What’s the oncorhynchus also called?"

Answer: "keta"

---

**Rule:**

Transformation rule: "?→??"

Before transformation: "How long is the Rhine?"

Answer: "1,230 km"

After transformation: "How long is the Rhine??"

Answer: "more than 1,050,000"

---

### SEAs와 SEARs 예시 설명 (Korean)

이 예시는 SEAs와 SEARs가 NLP 모델에서 어떻게 작동하는지를 보여줍니다. 첫 번째 예시 세트(SEAs)에서는 질문에 약간의 수정을 가함으로써 답변이 크게 달라지는 것을 볼 수 있으며, 이는 모델이 작은 변화에도 민감하다는 것을 나타냅니다. 두 번째 예시 세트(SEARs)에서는 간단한 변환 규칙을 통해 여러 개의 적대적 예제를 생성하고, 이로 인해 잘못된 답변이 생성되는 것을 보여줍니다. 이는 모델의 전반적인 취약성을 강조합니다.

---

### Explanation of SEAs and SEARs Examples (English)

These examples illustrate how SEAs and SEARs work in NLP models. In the first set of examples (SEAs), slight modifications to the questions lead to significantly different answers, demonstrating the model's sensitivity to small changes. In the second set of examples (SEARs), simple transformation rules generate multiple adversarial examples that result in incorrect answers, highlighting the model's global vulnerabilities.

<br/>
# refre format:     
Ribeiro, Marco Tulio, Sameer Singh, and Carlos Guestrin. "Semantically Equivalent Adversarial Rules for Debugging NLP Models." Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), July 2018.