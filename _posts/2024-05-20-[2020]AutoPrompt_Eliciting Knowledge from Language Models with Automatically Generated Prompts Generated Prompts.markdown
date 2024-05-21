---
layout: post
title:  "[2020]AutoPrompt_Eliciting Knowledge from Language Models with Automatically Generated Prompts Generated Prompts"  
date:   2024-05-20 20:43:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

짧은 요약(Abstract) :    

### Abstract 한글 설명:

사전 학습된 언어 모델의 놀라운 성공은 이 모델들이 사전 학습 중에 어떤 종류의 지식을 배우는지를 연구하도록 동기를 부여했습니다.

과제를 빈칸 채우기 문제(예: 클로즈 테스트)로 재구성하는 것은 이러한 지식을 측정하는 자연스러운 접근 방식이지만, 적절한 프롬프트를 작성하는 데 필요한 수작업과 추측으로 인해 사용이 제한됩니다.

이를 해결하기 위해 다양한 작업에 대한 프롬프트를 자동으로 생성하는 방법인 AUTOPROMPT를 개발했습니다.

AUTOPROMPT를 사용하여 마스크된 언어 모델(MLM)이 추가 파라미터나 미세 조정 없이 감정 분석과 자연어 추론을 수행할 수 있는 고유한 능력을 가지고 있음을 보여주었습니다.

또한, LAMA 벤치마크에서 수동으로 생성된 프롬프트보다 더 정확한 사실적 지식을 MLM으로부터 유도하고, MLM이 감독된 관계 추출 모델보다 관계 추출자로 더 효과적으로 사용할 수 있음을 보여주었습니다.

이러한 결과는 자동으로 생성된 프롬프트가 기존의 프로빙 방법에 대한 실질적인 대안이며, 사전 학습된 언어 모델이 더 정교하고 능력이 향상됨에 따라 잠재적으로 미세 조정을 대체할 수 있음을 시사합니다.

### Abstract 영어 설명:

The remarkable success of pretrained language models has motivated the study of what kinds of knowledge these models learn during pretraining.

Reformulating tasks as fill-in-the-blanks problems (e.g., cloze tests) is a natural approach for gauging such knowledge, however, its usage is limited by the manual effort and guesswork required to write suitable prompts.

To address this, we develop AUTOPROMPT, an automated method to create prompts for a diverse set of tasks.

Using AUTOPROMPT, we show that masked language models (MLMs) have an inherent capability to perform sentiment analysis and natural language inference without additional parameters or finetuning.

We also show that our prompts elicit more accurate factual knowledge from MLMs than the manually created prompts on the LAMA benchmark, and that MLMs can be used as relation extractors more effectively than supervised relation extraction models.

These results demonstrate that automatically generated prompts are a viable parameter-free alternative to existing probing methods, and as pretrained LMs become more sophisticated and capable, potentially a replacement for finetuning.

* Useful sentences :  
*   


{% endhighlight %}  

<br/>

[Paper link](https://drive.google.com/drive/folders/15rwpeC8peZg0LM3a9p7_G0uA8Xx8-PWI?usp=sharing)  
[Lecture link](https://slideslive.com/38939188/eliciting-knowledge-from-language-models-with-automatically-generated-prompts)   

<br/>

# 단어정리  
*  
 
<br/>
# Methodology    

### 방법론 한글 설명:

AUTOPROMPT는 다양한 작업에 대한 프롬프트를 자동으로 생성하기 위해 개발된 방법입니다.

이 방법은 마스크된 언어 모델(MLM)이 감정 분석과 자연어 추론을 추가 파라미터나 미세 조정 없이 수행할 수 있도록 합니다.

프롬프트를 생성하기 위해 AUTOPROMPT는 기울기 기반 검색을 사용합니다.

구체적으로, 각 입력 문장을 자연어 프롬프트로 변환하는 템플릿과 트리거 토큰 집합을 결합하여 프롬프트를 만듭니다.

트리거 토큰은 모든 입력에 대해 공통적으로 사용되며, 기울기 기반 검색을 통해 결정됩니다.

MLM 예측은 [MASK] 토큰을 기반으로 하고, 레이블 토큰 집합을 통해 분류 확률을 도출합니다.

이 방법은 레이블 토큰을 자동으로 선택하는 단계도 포함합니다.

먼저, 로지스틱 회귀를 사용하여 [MASK] 토큰의 컨텍스트화된 임베딩을 기반으로 클래스 레이블을 예측합니다.

그 다음, MLM의 출력 임베딩을 사용하여 레이블 토큰을 선택합니다.

### 방법론 영어 설명:

AUTOPROMPT is a method developed to automatically generate prompts for a diverse set of tasks.

This method allows masked language models (MLMs) to perform sentiment analysis and natural language inference without additional parameters or finetuning.

To generate prompts, AUTOPROMPT uses a gradient-based search.

Specifically, it creates a prompt by combining each input sentence with a template and a set of trigger tokens.

The trigger tokens are shared across all inputs and are determined using a gradient-based search.

MLM predictions are based on the [MASK] token, and classification probabilities are derived through a set of label tokens.

This method also includes a step to automatically select label tokens.

First, logistic regression is used to predict class labels based on the contextualized embedding of the [MASK] token.

Then, label tokens are selected using the output embeddings of the MLM.

<br/>
# Results  
### 결과 한글 설명:

AUTOPROMPT는 다양한 실험에서 그 효과를 입증했습니다.

먼저, 감정 분석과 자연어 추론(NLI) 작업에 대해 사전 학습된 마스크된 언어 모델(MLM)을 테스트하기 위해 프롬프트를 구성했습니다.

미세 조정 없이도 MLM은 두 작업 모두에서 뛰어난 성능을 보였습니다.

적절하게 프롬프트된 RoBERTa 모델은 SST-2 데이터셋에서 91%의 정확도를 기록했으며, 이는 미세 조정된 ELMo 모델보다 우수한 성능입니다.

또한, LAMA 벤치마크의 사실 검색 작업에서 AUTOPROMPT는 기존 수동 및 코퍼스 마이닝 방법보다 더 효과적으로 MLM의 사실적 지식을 유도하는 프롬프트를 구성했습니다.

LAMA 벤치마크에서 AUTOPROMPT는 최고 43.3%의 정밀도-1을 달성했으며, 이는 현재 최고 단일 프롬프트 결과인 34.1%를 능가합니다.

AUTOPROMPT는 관계 추출 작업에서도 뛰어난 성능을 보였으며, 실제 사실이 포함된 문맥 문장이 제공되면 기존 관계 추출 모델을 능가했습니다.

결국, AUTOPROMPT는 데이터가 부족한 상황에서도 높은 평균 및 최악의 경우 정확도를 달성했으며, 여러 작업을 위해 모델을 제공할 때 실용적인 장점을 제공합니다.

### 결과 영어 설명:

AUTOPROMPT demonstrated its effectiveness in numerous experiments.

First, prompts were constructed to test pretrained masked language models (MLMs) on sentiment analysis and natural language inference (NLI) tasks.

Without any finetuning, MLMs performed well on both tasks.

A properly-prompted RoBERTa model achieved 91% accuracy on the SST-2 dataset, surpassing a finetuned ELMo model.

Additionally, in the fact retrieval tasks of the LAMA benchmark, AUTOPROMPT constructed prompts that more effectively elicited factual knowledge from MLMs than existing manual and corpus-mining methods.

On the LAMA benchmark, AUTOPROMPT achieved a precision-at-1 of 43.3%, exceeding the current best single-prompt result of 34.1%.

AUTOPROMPT also excelled in relation extraction tasks, outperforming existing relation extraction models when context sentences with real facts were provided.

Ultimately, AUTOPROMPT achieved high average and worst-case accuracy in low-data regimes, offering practical advantages when serving models for multiple tasks.

<br/>  
# 요약 
### 주요 내용 한글 설명:

AUTOPROMPT는 다양한 작업에 대한 프롬프트를 자동으로 생성하여 마스크된 언어 모델(MLM)의 성능을 향상시키는 방법입니다.

이 방법은 기울기 기반 검색을 사용하여 각 입력 문장을 템플릿과 트리거 토큰 집합과 결합하여 프롬프트를 생성합니다.

AUTOPROMPT는 감정 분석, 자연어 추론, 사실 검색 및 관계 추출 작업에서 뛰어난 성능을 보였습니다.

특히, LAMA 벤치마크에서 기존 수동 및 코퍼스 마이닝 방법보다 더 효과적으로 사실적 지식을 유도하는 프롬프트를 구성했습니다.

또한, 데이터가 부족한 상황에서도 높은 정확도를 달성하여 실용적인 장점을 제공합니다.

### 주요 내용 영어 설명:

AUTOPROMPT is a method that automatically generates prompts for various tasks to enhance the performance of masked language models (MLMs).

This method uses gradient-based search to create prompts by combining each input sentence with a template and a set of trigger tokens.

AUTOPROMPT showed excellent performance in sentiment analysis, natural language inference, fact retrieval, and relation extraction tasks.

Specifically, it constructed prompts that more effectively elicited factual knowledge in the LAMA benchmark compared to existing manual and corpus-mining methods.

Additionally, it achieved high accuracy even in low-data scenarios, offering practical advantages. 

<br/>
# 예시  
### 예시 한글 설명:

AUTOPROMPT는 기울기 기반 검색을 사용하여 프롬프트를 생성합니다.

예를 들어, 감정 분석 작업에서 입력 문장이 "이 영화는 정말 좋았다"일 때,

AUTOPROMPT는 템플릿과 트리거 토큰을 결합하여 "이 영화는 정말 좋았다. 분위기 대화 [MASK]"와 같은 프롬프트를 생성할 수 있습니다.

여기서 [MASK] 토큰은 모델이 예측해야 할 부분입니다.

기울기 기반 검색을 통해, 각 트리거 토큰이 [MASK] 토큰의 예측 확률에 미치는 영향을 계산합니다.

AUTOPROMPT는 이러한 기울기를 사용하여 프롬프트에서 가장 효과적인 트리거 토큰을 선택합니다.

이 방법을 통해 모델이 "긍정적"이라는 레이블을 더 정확하게 예측할 수 있습니다.

또 다른 예로, 사실 검색 작업에서 "Obama was born in [MASK]"와 같은 프롬프트를 사용할 수 있습니다.

이 경우 모델은 "Hawaii"를 예측하게 됩니다.

### 예시 영어 설명:

AUTOPROMPT uses gradient-based search to generate prompts.

For example, in a sentiment analysis task with the input sentence "This movie was really good",

AUTOPROMPT can generate a prompt like "This movie was really good. atmosphere dialogue [MASK]" by combining a template with trigger tokens.

Here, the [MASK] token is the part that the model needs to predict.

Using gradient-based search, the impact of each trigger token on the prediction probability of the [MASK] token is calculated.

AUTOPROMPT uses these gradients to select the most effective trigger tokens for the prompt.

This method helps the model to more accurately predict the label "positive".

In another example, for a fact retrieval task, a prompt like "Obama was born in [MASK]" can be used.

In this case, the model would predict "Hawaii".

<br/>
# refer format:     
Shin, Taylor, Yasaman Razeghi, Robert L. Logan IV, Eric Wallace, and Sameer Singh. "AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts." In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), November 2020.  