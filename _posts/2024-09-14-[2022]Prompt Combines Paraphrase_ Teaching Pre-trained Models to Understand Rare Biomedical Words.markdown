---
layout: post
title:  "[2022]Prompt Combines Paraphrase: Teaching Pre-trained Models to Understand Rare Biomedical Words"  
date:   2024-09-14 14:25:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

짧은 요약(Abstract) :    



프롬프트 기반 미세 조정(fine-tuning)은 일반 도메인에서 소수의 샘플만을 사용하는 환경에서도 자연어 처리(NLP) 작업에 효과적인 것으로 입증되었습니다. 그러나 생의학 도메인에서 프롬프트 기반 미세 조정은 충분히 조사되지 않았습니다. 생의학 용어는 일반 도메인에서는 드물지만, 생의학 문맥에서는 매우 흔하게 사용되며, 이는 사전 학습된 모델이 생의학 애플리케이션에서 미세 조정을 거친 후에도 성능이 크게 저하되는 주요 원인입니다. 특히, 저자원 시나리오에서 이러한 문제가 두드러집니다. 우리는 모델이 프롬프트를 사용하여 드문 생의학 용어를 학습하는 데 도움을 주는 간단하지만 효과적인 접근 방식을 제안합니다. 실험 결과, 우리의 방법은 추가적인 매개변수나 훈련 단계 없이 소수의 샘플로 프롬프트를 적용하여 생의학 자연어 추론(NLI) 작업에서 최대 6%의 성능 향상을 달성할 수 있음을 보여줍니다.

---

Prompt-based fine-tuning for pre-trained models has proven effective for many natural language processing tasks under few-shot settings in the general domain. However, tuning with prompt in the biomedical domain has not been investigated thoroughly. Biomedical words are often rare in the general domain, but quite ubiquitous in biomedical contexts, which dramatically deteriorates the performance of pre-trained models on downstream biomedical applications even after fine-tuning, especially in low-resource scenarios. We propose a simple yet effective approach to helping models learn rare biomedical words during tuning with prompt. Experimental results show that our method can achieve up to 6% improvement in biomedical natural language inference tasks without any extra parameters or training steps using few-shot vanilla prompt settings.



* Useful sentences :  


{% endhighlight %}  

<br/>

[Paper link]()  
[~~Lecture link~~]()   

<br/>

# 단어정리  
*  

<br/>
# Introduction   


사전 학습된 모델들은 자연어 처리(NLP)에서 큰 성공을 거두었으며, 다양한 작업에 대해 새로운 패러다임을 제시했습니다(Peters et al., 2018; Devlin et al., 2019; Liu et al., 2019; Qiu et al., 2020). 많은 연구가 생의학 자연어 처리 작업에서 사전 학습된 모델에 주목하고 있습니다(Lee et al., 2020; Lewis et al., 2020; Zhao et al., 2021). 그러나 일반적인 사전 학습된 모델은 생의학 NLP 작업에서 뛰어난 성능을 보이지 못하는 경우가 있습니다. 생의학 NLP 작업에서 사전 학습된 모델의 잠재력을 충분히 발휘하는 데는 두 가지 주요 문제가 있습니다. (1) 제한된 데이터와 (2) 드문 생의학 용어들입니다. 첫째, 개인정보 보호 정책 제한(Šuster et al., 2017), 데이터 주석 작업의 높은 비용과 전문적 요구 사항으로 인해 생의학 레이블 데이터의 양이 제한적인 경우가 많습니다. 사전 학습된 모델들은 충분한 샘플이 없을 경우, 작업과 관련된 매개변수를 최적화하는 데 어려움을 겪습니다(Liu et al., 2021a). 둘째, 생의학 용어들은 저빈도 단어들이지만 생의학 텍스트를 이해하는 데 매우 중요한 역할을 합니다. 예를 들어, 자연어 추론(NLI) 작업에서 "afebrile"(발열 없음)과 같은 드문 단어를 다룰 때 모델은 그 단어를 충분히 학습하지 못했을 경우, 잘못된 레이블을 예측할 가능성이 높습니다. 따라서, 사전 학습된 모델은 저자원 작업 시 생의학 텍스트의 정확한 의미를 포착하지 못하게 됩니다.

새로운 작업에 대해 주석된 샘플이 거의 없는 상황에서, 추가적인 작업 전용 매개변수로 사전 학습된 모델을 효과적으로 미세 조정하는 것은 어렵습니다. 이는 앞서 언급한 것처럼 생의학 분야에서는 더욱 큰 도전 과제입니다. 프롬프트 기법은 소수의 샘플을 사용하는 환경에서 사전 학습과 다운스트림 작업 간의 간극을 좁혀 미세 조정 과정을 원활하게 하는 데 도입되었습니다(Liu et al., 2021a). 따라서 생의학 NLP 작업에 프롬프트 기반 조정을 적용하는 것이 유익할 수 있습니다.

드문 단어들의 문제가 생의학 사전 학습 모델의 주요 문제임에도 불구하고, 이 문제를 연구한 논문은 소수에 불과하며, 대부분의 연구는 사전 학습 단계에서 드문 단어들의 표현을 강화하는 데 초점을 맞추고 있습니다(Schick and Schütze, 2020; Yu et al., 2021; Wu et al., 2020). 이러한 방법들은 생의학 지식을 사용한 추가적인 사전 학습이나 훈련 단계를 필요로 하며, 이는 시간이 많이 소요되고 비효율적입니다. 반면에 우리는 이 문제를 해결하기 위해 사전 학습 대신 미세 조정 단계에 초점을 맞추고 있습니다. 사람들이 새로운 단어를 접했을 때 사전을 찾아 그 의미를 파악하는 것처럼, 우리는 프롬프트 기반 미세 조정에서 드문 생의학 단어를 설명하는 방법을 제안합니다. 이 새로운 접근 방식은 생의학 단어를 이해하는 데 있어 모델의 미세 조정 능력을 강화할 수 있습니다. 또한, 이 방법은 특정 데이터셋에 국한되지 않고, 다른 도메인으로 쉽게 이전될 수 있는 모델 독립적인 모듈입니다.

---


Pre-trained models have achieved great success in natural language processing (NLP) and become a new paradigm for various tasks (Peters et al., 2018; Devlin et al., 2019; Liu et al., 2019; Qiu et al., 2020). Many studies have paid attention to pre-trained models in biomedical NLP tasks (Lee et al., 2020; Lewis et al., 2020; Zhao et al., 2021). However, plain pre-trained models sometimes cannot do very well in biomedical NLP tasks. In general, there are two challenges to fully exploit the potential of the pre-trained models for biomedical NLP tasks, i.e., (1) limited data and (2) rare biomedical words. Firstly, it is common that the amount of biomedical labeled data is limited due to strict privacy policy constraints (Šuster et al., 2017), high cost, and professional requirements for data annotation. Pre-trained models perform poorly with few samples since abundant training samples are essential to optimize task-related parameters (Liu et al., 2021a). Secondly, biomedical words are usually low-frequency words but critical to understanding biomedical texts. As an example of a natural language inference (NLI) task in Figure 1, the model goes wrong during tuning when faced with a rare word “afebrile” (having no fever) in the premise. It can be difficult for the pre-trained models to predict the correct label if the models haven’t seen the rare biomedical words enough times during pre-training or tuning stages. Thus, pre-trained models cannot capture the precise semantics of biomedical texts in the scenario of low-resource tasks.

With very few annotated samples available for a new task, it is hard to effectively fine-tune pre-trained models with the additional task-specific parameters, which is even more of a challenge in the biomedical domain as mentioned above. The prompt technique has been introduced to smooth the fine-tuning process in the few-shot settings by narrowing down the gap between the pre-training stage and the downstream task in the general domain (Liu et al., 2021a), as demonstrated in Figure 2. Therefore, it is beneficial to adapt prompt-based tuning to biomedical NLP tasks.

Although the challenge of rare words is a critical problem for biomedical pre-trained models, only a handful of works have studied the issue and most of them focus on enriching the representation of rare words through the pre-training stage (Schick and Schütze, 2020; Yu et al., 2021; Wu et al., 2020). Thus, it naturally requires them to involve a second-round pre-training or further training steps with biomedical knowledge to achieve the above goal, which is highly time-consuming and inefficient. Alternatively, we emphasize the tuning stage instead of pre-training to resolve these issues. When coming across an unknown word, humans may seek the dictionary for its paraphrase. Enlightened by this phenomenon, we propose to explain rare biomedical words with paraphrases on the basis of prompt-based tuning. The new approach could enhance tuning capability in understanding biomedical words. Furthermore, as a generic plug-in module for non-specific datasets, our approach is model-agnostic and can be easily transferred to other domains.


 
<br/>
# Methodology    


**3. 방법**

이 섹션에서는 드문 생의학 단어를 찾고, 프롬프트 기반 미세 조정을 통해 사전 학습된 모델에 이 단어들의 패러프레이즈(paraphrase)를 추가하는 방법을 모델 독립적인 플러그인 방식으로 소개합니다.

**3.1 드문 단어**

단어의 드문 정도는 특정 코퍼스에서의 빈도에 따라 결정되며, 이는 문맥에 따라 달라질 수 있습니다. 사전 학습된 코퍼스에서 드문 단어가 다운스트림 작업에서는 드문 단어가 아닐 수도 있습니다. 이 연구에서는 사전 학습 코퍼스에서 특정 임계값 이하로 등장하는 단어를 "드문 단어"로 정의합니다.

한편, 사전 학습된 모델들은 입력된 단어를 토큰으로 분할합니다. Byte-Pair Encoding(Sennrich et al., 2016)이나 WordPiece(Schuster and Nakajima, 2012)와 같은 토크나이저는 빈도나 가능성에 따라 단어를 하위 단어로 분할하는데, 이 과정에서 흔한 단어들이 주로 영향을 미칩니다. 따라서 드문 단어가 흔한 단어들로 분할되어 사전 학습된 모델에 입력될 때, 원래 드문 단어의 의미가 거의 보존되지 않습니다. 또한, 서로 다른 사전 학습 모델의 토크나이저는 같은 드문 단어를 다른 토큰으로 나누어 드문 단어의 토큰화 결과가 모델마다 다를 수 있습니다. 예를 들어, BERT-Large(Devlin et al., 2019) 모델은 "afebrile"이라는 단어를 "af-eb-ril-e"로 분할하고, Biomedical-Clinical-RoBERTa-Large(Lewis et al., 2020) 모델은 "a-fe-brile"로 분할합니다.

**3.2 드문 생의학 단어의 선택**

단어의 빈도를 얻기 위해 PubMed 초록, PubMed Central(PMC) 전체 텍스트 및 MIMIC-III 데이터셋과 같은 생의학 코퍼스를 사용합니다. 이러한 코퍼스는 BC-RoBERTa(Lewis et al., 2020), BioBERT(Lee et al., 2020), PubMedBERT(Gu et al., 2020)와 같은 생의학 언어 모델의 사전 학습에 널리 사용됩니다. 우리는 위의 코퍼스를 반복하여 사전 학습 단계에서 각 단어의 빈도를 얻습니다. 생의학 코퍼스에서 발견된 드문 단어들은 생의학 도메인뿐만 아니라 일반 도메인에서도 포함될 수 있습니다. 모든 드문 단어를 포함하는 대신, 우리는 생의학 도메인의 드문 단어들만 고려하는데, 이는 (1) 도메인 특유의 분포와 (2) 작업 특화 단어 때문입니다. 도메인 특화 용어는 생의학 작업에서 더 큰 기여를 할 수 있기 때문에, 우리는 특정 임계값을 설정하여 드문 단어를 선택하며, Section 4.3에서 다양한 임계값을 실험합니다. 우리는 드문 생의학 단어의 패러프레이즈를 온라인 사전인 Wiktionary에서 가져옵니다. 이를 최적화하기 위해 우리는 의료, 의학, 질병, 증상 및 약리학과 관련된 카테고리로 태그된 단어들만 유지합니다.

**3.3 패러프레이즈 선택**

드문 생의학 단어에 대해 하나 이상의 패러프레이즈가 있을 수 있으며, 가장 적절한 패러프레이즈를 선택하는 것은 어려울 수 있습니다. 따라서 부적절한 패러프레이즈로 인한 노이즈를 줄이기 위해, 하나 이상의 패러프레이즈가 있는 드문 생의학 단어는 제외합니다. 또한, 설정된 임계값 이하의 빈도를 가진 또 다른 드문 단어를 포함하는 패러프레이즈는 무시합니다. 한편, 생의학 약어는 의미 있는 정보가 거의 없기 때문에 모든 생의학 약어에 대해 패러프레이즈를 추가합니다.

**3.4 패러프레이즈를 사용한 프롬프트 기반 미세 조정**

사람들이 새로운 단어를 접할 때, 우리는 그 의미를 파악하기 위해 사전을 찾습니다. 이와 같은 방식으로, 우리는 사전 학습된 모델을 패러프레이즈로 안내합니다. 즉, 드문 단어는 괄호 안에 패러프레이즈를 추가하여 설명됩니다(Figure 2(d)). 이렇게 하면 사전 학습된 모델을 사용할 때, 드문 생의학 단어에 대한 패러프레이즈가 데이터셋에 즉시 생성되어 프롬프트 기반 미세 조정 전에 플러그인 모듈로 사용될 수 있습니다.

---


In this section, we introduce how we find the rare biomedical words and append paraphrases to the rare biomedical words with the prompt-based tuning of pre-trained models in a model-agnostic plug-in manner.

**3.1 Rare Words**

The rarity of a word mostly depends on its frequency in a certain corpus, which can vary from context to context. A rare word in the pre-training corpora is possibly not that rare in the downstream tasks. In this work, we define the “rare words” as the words whose frequency is under a specific threshold in the pre-training corpora as aforementioned.

Meanwhile, although the pre-trained models tokenize the input words into tokens, tokenizers based on byte-pair encoding (Sennrich et al., 2016) or WordPiece (Schuster and Nakajima, 2012) split words into sub-words by frequency or likelihood, which is both dominated by the common words. Thus, although the rare words can be split into possible non-rare tokens, there is not much semantics from the original rare words retained after being tokenized into common tokens for the pre-trained models. Also, tokenizers of different pre-trained models can tokenize the same rare word into different tokens and consequentially make rare tokens model-related. For example, BERT-Large (Devlin et al., 2019) model tokenizes “afebrile” into “af-eb-ril-e” while Biomedical-Clinical-RoBERTa-Large (Lewis et al., 2020) model tokenizes it into “a-fe-brile”.

**3.2 Selection of Rare Biomedical Words**

To obtain the frequency of words, we adopt the biomedical corpora including PubMed abstract, PubMed Central (PMC) full-text, and MIMIC-III dataset, which are widely used for pre-training biomedical language models, such as BC-RoBERTa (Lewis et al., 2020), BioBERT (Lee et al., 2020), and PubMedBERT (Gu et al., 2020). We loop the above corpora to obtain the frequency of each word in the pre-training phase. The rare words found in biomedical corpora are likely to contain words not only in the biomedical domain but also in the general domain. Instead of including all rare words, we consider rare words from the biomedical domain with the following two reasons: (1) Domain-specific distribution: unlike the general domain, distribution of words in the biomedical domain is shaped with domain-specific terms, such as disease, medicine, diagnosis, and treatment (Lee et al., 2020). (2) Task-specific words: rare words from the biomedical domain can contribute more to biomedical tasks than that from the general domain. Therefore, we introduce a threshold, an empirical hyper-parameter, to assist the selection of rare words following Yu et al. (2021), and we also experiment with different thresholds for the rare words in Section 4.3. We retrieve the paraphrases of the rare biomedical words from an online dictionary “Wiktionary.”

To optimize the selection, we only keep the rare words that are tagged with medical-related categories from the Wiktionary, i.e., medical, medicine, disease, symptom, and pharmacology.

**3.3 Selection of Paraphrases**

There can be more than just one paraphrase for a rare biomedical word and it is tricky to choose the most appropriate paraphrases. Therefore, to avoid introducing noise from the inappropriate paraphrases, we exclude rare biomedical words with more than one corresponding paraphrase. In addition, we ignore paraphrases that contain additional rare words whose frequencies are below the set threshold since it only replaces one rare word with another. Meanwhile, considering that biomedical abbreviations are likely to be tokenized into separate letters with no meaningful semantic information, we retrieve and append the paraphrases to all the biomedical abbreviations.

**3.4 Prompt-based Fine-Tuning with Paraphrases**

When coming across new words during reading, humans habitually seek dictionaries for the corresponding paraphrases to help us understand. Following the same idea, we guide the pre-trained models with paraphrases, where rare words are followed by the parenthesis punctuation, as shown in Figure 2(d). In this way, given a pre-trained model, paraphrases of biomedical rare words can be considered as a portable plug-in module and generated for any dataset instantly before prompt-based fine-tuning.

<br/>
# Results  




결과 부분에서는 소수의 샘플을 사용하여 드문 생의학 단어를 처리하는 프롬프트 기반 미세 조정 기법이 기존의 기법들 대비 성능 향상을 이뤘음을 보여줍니다. 특히, 생의학 자연어 추론(MedNLI)과 생의학 의미적 텍스트 유사성(MedSTS) 작업에서 다양한 사전 학습된 모델들에 대해 실험을 진행하였습니다. 

**MedNLI 결과:**
- BERT-Large 모델의 경우, 프롬프트와 패러프레이즈(paraphrase)를 적용한 기법이 소수 샘플 학습에서 최대 6%까지 성능 향상을 보여줬습니다. 예를 들어, 16개의 샘플을 사용한 경우, BERT-Large의 정확도는 38.9%에서 40.8%로 약 2% 향상되었습니다.
- RoBERTa-Large 모델은 성능이 43.2%에서 49.5%로, 즉 약 6.3% 향상되었습니다. 샘플 수가 많아질수록 성능 향상이 줄어들긴 하지만, 여전히 모든 경우에서 패러프레이즈가 적용된 모델이 더 나은 성능을 보였습니다.

**MedSTS 결과:**
- MedSTS 작업에서도 BERT-Large와 RoBERTa-Large 같은 모델에서 유사한 성능 향상이 있었습니다. BERT-Large는 Pearson 상관계수가 14.1%에서 18.5%로 약 4.4% 향상되었습니다.
- RoBERTa-Large 모델은 29.5%에서 34.6%로, 약 5.1% 향상되었습니다.

이 결과들은 프롬프트와 패러프레이즈를 사용한 미세 조정이 드문 생의학 단어들을 더 잘 이해할 수 있게 도와주며, 소수 샘플 환경에서도 기존의 방법들보다 성능이 크게 향상됨을 보여줍니다.

---


**Results:**

The results demonstrate that the prompt-based fine-tuning method with paraphrases for handling rare biomedical words significantly outperforms baseline methods across various pre-trained models in few-shot learning settings, especially for the biomedical natural language inference (MedNLI) and biomedical semantic textual similarity (MedSTS) tasks.

**MedNLI Results:**
- For the BERT-Large model, the method incorporating prompts and paraphrases showed an improvement of up to 6% in few-shot learning. For example, with 16 samples, the accuracy of BERT-Large improved from 38.9% to 40.8%, which is about a 2% increase.
- The RoBERTa-Large model showed an improvement from 43.2% to 49.5%, which is approximately a 6.3% increase. Although the performance improvement decreases as the number of samples increases, the model with paraphrases consistently outperformed the baseline in all cases.

**MedSTS Results:**
- Similarly, in the MedSTS task, models like BERT-Large and RoBERTa-Large demonstrated similar performance gains. The Pearson correlation coefficient for BERT-Large increased from 14.1% to 18.5%, an improvement of about 4.4%.
- For RoBERTa-Large, the performance increased from 29.5% to 34.6%, about a 5.1% improvement.

These results indicate that prompt-based fine-tuning with paraphrases helps models better understand rare biomedical words, significantly improving performance in few-shot scenarios compared to traditional methods.




<br/>
# 예시  



본 논문에서는 생의학 자연어 추론(MedNLI) 작업에서 드문 생의학 단어를 포함한 테스트 예제를 통해, 프롬프트 기반 미세 조정 기법이 어떻게 기존 모델보다 더 나은 성능을 발휘했는지 보여줍니다. 

예를 들어, 아래와 같은 테스트 문장을 사용할 수 있습니다:

- **전제(Premise):** 환자는 직장 검사에서 BRBPR(직장 출혈) 소견을 보였다.
- **가설(Hypothesis):** 환자는 직장 출혈이 있었다.

기존의 사전 학습된 모델들은 "BRBPR"와 같은 드문 생의학 약어를 제대로 처리하지 못해, 이 두 문장이 같은 의미라는 것을 이해하지 못하고 잘못된 예측을 할 수 있습니다. 예를 들어, BERT-Large 모델은 이 문장을 **"중립"**으로 예측했습니다. 하지만 본 논문에서 제안된 프롬프트 기반 미세 조정 기법은 "BRBPR"이라는 약어가 "직장 출혈"이라는 의미로 패러프레이즈가 추가되면서, 이 문장이 같은 의미라는 것을 정확하게 인식하고 **"일치(Entailment)"**로 올바르게 예측했습니다.

또 다른 예제는 아래와 같습니다:

- **전제(Premise):** 환자는 출산 3일 전에 입원했으며, 임신 중 고혈압과 단백뇨 소견이 있었다.
- **가설(Hypothesis):** 환자는 임신 중 단백뇨가 있었다.

기존 모델들은 "단백뇨"라는 드문 단어를 충분히 학습하지 못해 **"일치(Entailment)"**로 예측했지만, 실제로는 이 문장들은 **"중립(Neutral)"**이 맞습니다. 본 논문의 기법은 드문 생의학 용어에 대한 패러프레이즈를 제공하여 이 문제를 해결하고, 정확한 예측을 도와줍니다.

이러한 예제들을 통해 기존 모델은 드문 생의학 단어들을 다루는 데 어려움을 겪지만, 본 논문의 기법은 패러프레이즈를 통해 이러한 문제를 해결하여 더 나은 성능을 보였습니다.

---


In this paper, the authors provide examples from the biomedical natural language inference (MedNLI) task to show how the prompt-based fine-tuning method performs better than traditional models, particularly in handling rare biomedical terms.

For instance, consider the following test example:

- **Premise:** The patient showed BRBPR (bright red blood per rectum) on rectal examination.
- **Hypothesis:** The patient had bright red blood per rectum.

Traditional pre-trained models, such as BERT-Large, struggle to correctly interpret the rare abbreviation "BRBPR" and may incorrectly classify the relationship between the premise and hypothesis. In this case, BERT-Large predicted **"Neutral"** due to its inability to recognize that "BRBPR" means "bright red blood per rectum." However, the prompt-based fine-tuning method used in this paper added a paraphrase explaining the term, allowing the model to correctly classify the relationship as **"Entailment."**

Another example involves the following:

- **Premise:** The patient was admitted three days prior to delivery due to gestational hypertension and proteinuria during pregnancy.
- **Hypothesis:** The patient had proteinuria during pregnancy.

In this case, traditional models misclassify this as **"Entailment"**, as they have not sufficiently learned the meaning of the rare term "proteinuria." However, this should be classified as **"Neutral."** The proposed method in the paper provides a paraphrase for the rare biomedical term, helping the model make the correct prediction.

These examples illustrate that while traditional models struggle with rare biomedical terms, the prompt-based fine-tuning approach in this paper successfully addresses these issues by using paraphrases, resulting in better performance.





<br/>  
# 요약 



본 논문에서는 소수 샘플 환경에서 드문 생의학 단어를 더 잘 이해하도록 돕기 위해 프롬프트 기반 미세 조정 기법을 제안합니다. 이 방법은 사전 학습된 모델에 드문 단어의 패러프레이즈를 제공하여 모델이 더 정확한 의미를 학습할 수 있게 합니다. 실험 결과, BERT-Large와 RoBERTa-Large 같은 모델에서 최대 6%의 성능 향상을 보여주었습니다. 예를 들어, "BRBPR"와 같은 드문 약어를 패러프레이즈로 설명해 주면 모델이 더 정확하게 자연어 추론 작업을 수행할 수 있습니다. 이 방법은 기존 모델이 다루기 어려운 드문 용어 문제를 해결하며, 생의학 텍스트 이해 능력을 크게 향상시킵니다.

---


This paper proposes a prompt-based fine-tuning approach to help models better understand rare biomedical words in few-shot settings. The method provides paraphrases for rare terms during the fine-tuning stage, allowing pre-trained models to learn more accurate meanings. Experimental results show up to a 6% performance improvement for models like BERT-Large and RoBERTa-Large. For example, explaining rare abbreviations like "BRBPR" with a paraphrase enables the model to perform natural language inference tasks more accurately. This approach effectively addresses the challenge of handling rare terms and significantly enhances the models' understanding of biomedical texts.

# 기타  


<br/>
# refer format:     


@inproceedings{wang2022prompt,
  title={Prompt Combines Paraphrase: Teaching Pre-trained Models to Understand Rare Biomedical Words},
  author={Wang, Haochun and Liu, Chi and Xi, Nuwa and Zhao, Sendong and Ju, Meizhi and Zhang, Shiwei and Zhang, Ziheng and Zheng, Yefeng and Qin, Bing and Liu, Ting},
  booktitle={Proceedings of the 29th International Conference on Computational Linguistics},
  pages={1422--1431},
  year={2022},
  organization={International Committee on Computational Linguistics}
}




Wang, Haochun, Chi Liu, Nuwa Xi, Sendong Zhao, Meizhi Ju, Shiwei Zhang, Ziheng Zhang, Yefeng Zheng, Bing Qin, and Ting Liu. "Prompt Combines Paraphrase: Teaching Pre-trained Models to Understand Rare Biomedical Words." In Proceedings of the 29th International Conference on Computational Linguistics, 1422–1431. International Committee on Computational Linguistics, 2022.



