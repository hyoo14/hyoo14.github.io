---
layout: post
title:  "[2026]Linear Script Representations in Speech Foundation Models Enable Zero-Shot Transliteration"
date:   2026-04-16 19:06:18 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문은 Whisper 같은 음성 기초모델의 디코더 활성값에서 **script(문자 체계)가 선형 방향으로 표현**된다는 점을 이용해, 소스/타깃 script 샘플의 평균 활성 차이로 **script vector**를 추출하고 테스트 시 활성에 더해 원하는 script로 전사되게 하는 방법을 제안합니다.


짧은 요약(Abstract) :




이 논문은 **다국어 음성 인식 모델(특히 Whisper)** 이 출력하는 **문자 체계(script)** 를 사용자가 원하는 대로 조절할 수 있음을 보여줍니다.  
예를 들어 같은 말이라도 **키릴 문자**, **라틴 문자**, **한자** 등 서로 다른 스크립트로 표기할 수 있는데, 기존 모델은 어떤 스크립트로 출력할지가 일관되지 않을 수 있습니다.

저자들은 **스크립트 정보가 모델의 activation space(활성화 공간)에 선형적으로 표현되어 있다**고 주장합니다. 즉, 특정 스크립트를 나타내는 방향(vector)이 존재하며, 이를 테스트 시점에 activation에 더해주면 출력 스크립트를 바꿀 수 있습니다.

이 방법의 핵심은:
- 학습(finetuning) 없이
- 추론(inference)할 때
- activation에 스크립트 벡터를 더해서

원하는 스크립트로 출력하게 만드는 것입니다.

실험 결과, 이 방식은:
- **일반적인 언어-스크립트 조합뿐 아니라**
- **이례적인 조합**에서도 작동했습니다.

예를 들어:
- 이탈리아어를 **키릴 문자**로 출력
- 일본어를 **라틴 문자**로 출력

같은 것도 가능했습니다.

또한 이 방법은 Whisper의 **모든 모델 크기**에서 경쟁력 있는 성능을 보였고, 결과적으로 **speech recognition output의 script를 post-hoc으로 제어**하는 새로운 방법을 제시합니다.

---


This paper shows that the **output script** of multilingual speech recognition models, especially **Whisper**, can be directly controlled.

In many languages, the same spoken content can be written in different scripts, such as **Latin**, **Cyrillic**, or **Chinese characters**. However, ASR models may produce inconsistent scripts for the same language.

The authors argue that **script is linearly encoded in the model’s activation space**. In other words, there are directions in hidden activations that correspond to specific scripts. By adding these script vectors during inference, the model can be steered to output the desired script.

The main idea is:
- no finetuning,
- no retraining,
- only inference-time modification of activations.

Their experiments show that this method can induce script changes not only in normal language-script pairs, but also in unusual ones, such as:
- writing **Italian in Cyrillic**
- writing **Japanese in Latin script**

The method works competitively across all Whisper model sizes and provides a way to control script output **after training**, i.e. post-hoc.

---





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




# 1) 한글 설명: 메서드(모델, 아키텍처, 학습데이터, 특별한 기법)

## 1. 연구가 다루는 문제
이 논문은 **멀티링구얼 음성 인식 모델(특히 Whisper 계열)** 이 같은 언어라도 서로 다른 **문자 체계(script)** 로 결과를 내놓는 문제를 다룹니다.  
예를 들어:

- 세르비아어는 **라틴 문자**와 **키릴 문자** 둘 다 사용 가능
- 중국어는 **간체(Simplified)** 와 **번체(Traditional)** 가 공존
- 어떤 경우에는 원래 언어와 무관한 문자로도 표기하고 싶을 수 있음  
  - 예: 이탈리아어를 키릴 문자로 적기

즉, 단순히 “무슨 언어인지”를 맞히는 것이 아니라, **어떤 script로 출력할지 제어(control)** 하는 것이 핵심입니다.

---

## 2. 사용한 모델: Whisper 계열 speech foundation model
이 논문은 **OpenAI Whisper** 를 기반으로 실험합니다.

### Whisper의 구조
Whisper는 전형적인 **encoder-decoder 구조**의 음성 인식 모델입니다.

- **Encoder**: 입력 음성을 음성 표현(continuous representation)으로 변환
- **Decoder**: encoder 출력을 바탕으로 텍스트를 생성하는 autoregressive language model 역할 수행

논문은 Whisper의 여러 크기 모델을 모두 사용합니다.

- tiny
- base
- small
- medium
- large
- large-v2
- large-v3

즉, 이 방법이 특정 큰 모델에만 되는지, 아니면 작은 모델에도 되는지 확인하기 위해 **모델 크기별로 비교**했습니다.

---

## 3. 핵심 아이디어: script가 activation space에 선형적으로 인코딩되어 있다
이 논문의 중심 가설은 다음입니다.

> **문자 체계(script)는 Whisper의 hidden activation 공간에서 선형 방향(linear direction)으로 표현된다.**

즉, “Cyrillic스럽게 만드는 방향”, “Latin스럽게 만드는 방향” 같은 것이 모델 내부 표현 공간에 존재한다는 것입니다.  
이 방향만 알아내면, **추론 시점(inference time)** 에 hidden state에 더해 줌으로써 출력 script를 조절할 수 있습니다.

이 아이디어는 최근 LLM에서 알려진 **activation steering / representation engineering** 과 유사합니다.

---

## 4. 특별한 기법: Script Vector(= Steering Vector) 추출
논문이 제안하는 핵심 방법은 **script vector** 를 만드는 것입니다.  
이 벡터는 “source script”와 “target script” 사이의 차이를 나타냅니다.

### 4.1 전체 절차
논문은 script vector를 추출하는 과정을 3단계로 설명합니다.

#### (1) Activation collection
같은 음성 샘플에 대해 decoder에 서로 다른 프롬프트를 넣어 두 번 decoding합니다.

- **source prompt**: source script 쪽으로 유도
- **target prompt**: target script 쪽으로 유도

예를 들어:
- Serbian Latin vs Serbian Cyrillic
- Simplified Chinese vs Traditional Chinese
- Romanization vs Cyrillization

각 decoder layer에서 token position별 hidden activation을 모읍니다.

---

#### (2) Direction isolation
각 layer마다 source와 target의 평균 activation을 구한 뒤,

\[
r_\ell = v^{SRC}_\ell - v^{TRG}_\ell
\]

처럼 **차이 벡터**를 만듭니다.

즉:

- source script 쪽 평균 activation
- target script 쪽 평균 activation

의 차이를 script direction으로 보는 것입니다.

이 벡터가 바로 **script vector / steering vector** 입니다.

---

#### (3) Activation addition
추론할 때는 decoder의 hidden state에 이 벡터를 더합니다.

\[
h^{steered}_{t,\ell} = h_{t,\ell} + \sigma r_\ell
\]

여기서:

- \(h_{t,\ell}\): 현재 토큰의 layer \(\ell\) hidden activation
- \(r_\ell\): script vector
- \(\sigma\): steering strength 조절 값

즉, 모델이 텍스트를 생성하는 중간에 **hidden state를 살짝 밀어** 원하는 script로 출력하게 합니다.

---

## 5. 왜 이 방법이 특별한가?
이 방법의 장점은 다음과 같습니다.

### 5.1 재학습이 필요 없음
- finetuning 없이
- inference time에서만
- 벡터를 더하는 방식으로 제어

즉, **학습 비용이 거의 없습니다**.

### 5.2 적은 샘플로도 가능
논문은 심지어 **한 개의 좋은 예시(pair)** 만으로도 script vector를 만들 수 있다고 보고합니다.

즉, 데이터가 매우 적어도 되는 **sample-efficient method** 입니다.

### 5.3 비정상적인 script 변환도 가능
원래 언어-문자 조합이 자연스럽지 않아도 script 변환이 가능합니다.

예:
- Italian in Cyrillic
- Japanese in Latin

이건 단순한 deterministic transliteration tool로는 어렵고, 모델 내부 표현을 직접 조절하기 때문에 가능한 결과입니다.

---

## 6. 학습데이터 / 실험 데이터
이 논문은 **FLEURS** 데이터셋을 사용합니다.

### FLEURS란?
- 100개가 넘는 언어에 대해
- 음성과 transcription이 정렬된 multilingual speech dataset

논문은 여기서 여러 언어를 골라 실험합니다.

### 사용 언어
- Serbian
- Mandarin
- Russian
- Italian
- Hindi
- Greek
- Japanese
- Korean

### 왜 이 언어들?
두 가지 이유가 있습니다.

1. **script confusion이 있는 언어**
   - Serbian: Latin/Cyrillic
   - Mandarin: Simplified/Traditional

2. **transliteration이 의미 있는 언어**
   - Romanization
   - Cyrillization

즉, 문자 전환 실험에 적합한 언어들입니다.

---

## 7. 입력 프롬프트(prompt)의 역할
이 논문은 prompt를 단순한 텍스트 조건으로 사용합니다.

예를 들어:
- “this is a sentence in X” 의미의 문장을 target script로 입력
- decoder를 특정 script로 biasing

즉, prompt는 **원하는 script를 암시**하는 역할을 합니다.

논문은 세 가지 대표 상황을 둡니다.

### 7.1 no-prompt
아무 프롬프트 없이 오디오만 넣음  
→ 모델이 자연스럽게 내놓는 script 확인

### 7.2 prompt
target script로 된 prompt를 넣어 출력 script를 유도

### 7.3 steer
prompt와 함께, 혹은 prompt를 통해 얻은 script vector를 activation에 추가

---

## 8. Filtering: 품질 좋은 activation만 사용
script vector를 만들 때 모든 예시를 그대로 쓰지 않습니다.  
왜냐하면 모델이 엉뚱한 script를 출력했을 수도 있기 때문입니다.

그래서 논문은 **필터링**을 합니다.

### 방법
- 모델 출력과 정답 사이의 normalized Levenshtein edit distance 계산
- threshold \(\theta\)보다 낮은 예시만 사용

즉, **잘 맞는 예시만 골라서 평균 activation을 계산**합니다.  
이렇게 해야 script vector가 더 깨끗해집니다.

논문에서는:
- script confusion / romanization: \(\theta = 0.4\)
- cyrillization: \(\theta = 0.8\)

를 사용합니다.

---

## 9. 실험 설정: 두 가지 주요 task
논문은 크게 두 가지를 봅니다.

### 9.1 Script confusion mitigation
같은 언어 안에서 script가 흔들리는 문제를 해결

예:
- Serbian이 Latin 또는 Cyrillic로 나오는 문제
- Mandarin이 Simplified 또는 Traditional로 나오는 문제

### 9.2 Transliteration
원래 언어를 다른 script로 옮김

예:
- Japanese → Latin
- Greek → Latin
- Italian → Cyrillic

이때 중요한 점은, target script가 source language와 반드시 연결되어 있지 않아도 된다는 것입니다.

---

## 10. Zero-shot과 pseudo-label 방식
transliteration에서는 추가로 두 가지 steering 방법을 씁니다.

### 10.1 zero-shot
한 언어에서 얻은 script vector를 다른 언어에 그대로 적용

예:
- Serbian에서 만든 romanization direction을 Russian, Hindi, Japanese 등에 적용

즉, **언어 간 일반화**를 보는 방식입니다.

### 10.2 pseudo-label
zero-shot으로 얻은 steered output을 임시 정답처럼 사용해서  
다시 script vector를 추출합니다.

즉:

1. 처음 vector로 steering
2. 나온 결과를 pseudo transcription으로 사용
3. 새 vector를 다시 학습하듯 추출

이 방식은 실제로 **더 해당 언어에 맞는 script direction** 을 얻기 위한 개선 절차입니다.

---

## 11. Layer-wise steering
논문은 decoder의 **모든 layer에 steering** 하는 방식을 사용합니다.

이유는:
- script 정보가 모든 layer에서 선형적으로 존재한다고 보기 때문
- 실제로 layer-wise probe 실험에서도 모든 layer에서 script 구분이 가능했기 때문

즉, 특정 한 layer만 건드리기보다  
**모든 decoder layer에 같은 방향 벡터를 주입**하는 것이 더 자연스럽고 효과적입니다.

---

## 12. 평가 방식
평가는 character-level normalized edit similarity를 씁니다.

- 예측 결과와 정답 script transcription 간의 edit distance 계산
- 이를 길이로 정규화
- 1에서 빼서 similarity score로 변환

즉 점수가 높을수록:
- target script를 더 잘 따르고
- transliteration도 더 정확하다는 뜻입니다

---

## 13. 이 논문의 방법을 한 문장으로 요약하면
> Whisper의 decoder hidden state에서 script 차이를 나타내는 선형 방향을 추출하고, 추론 시 그 방향을 activation에 더해 출력 script를 제어하는 방법이다.

---

## 1. Problem being addressed
This paper studies the problem that multilingual speech recognition models, especially Whisper-style models, may produce outputs in different scripts for the same language.

Examples include:

- Serbian in Latin vs. Cyrillic
- Mandarin in Simplified vs. Traditional Chinese
- More unusual cases such as Italian written in Cyrillic

So the goal is not only to recognize the language correctly, but also to **control the output script**.

---

## 2. Model used: Whisper speech foundation models
The paper builds on **OpenAI Whisper**, a multilingual speech recognition model with an **encoder-decoder architecture**.

### Whisper architecture
- **Encoder**: converts speech into latent audio representations
- **Decoder**: autoregressively generates text from those representations

The authors evaluate multiple Whisper sizes:

- tiny
- base
- small
- medium
- large
- large-v2
- large-v3

This allows them to test whether the method works only on large models or also on smaller ones.

---

## 3. Core hypothesis: script is linearly encoded in activation space
The central idea is:

> Script information is represented as a linear direction in the model’s hidden activation space.

In other words, there is a direction in the decoder representation space corresponding to “more Cyrillic” or “more Latin.”  
If we identify that direction, we can add it at inference time and steer the output script.

This is closely related to **activation steering** / **representation engineering**.

---

## 4. Main technique: extracting a script vector
The paper proposes a **script vector** or **steering vector** that captures the difference between two scripts.

### 4.1 Three-step procedure

#### (1) Activation collection
For the same audio sample, the decoder is run twice with different text prompts:

- a **source prompt** that biases the output toward the source script
- a **target prompt** that biases the output toward the target script

The hidden activations are collected at each decoder layer.

---

#### (2) Direction isolation
For each layer, the mean activation under the source script and the mean activation under the target script are computed, and the difference is taken:

\[
r_\ell = v^{SRC}_\ell - v^{TRG}_\ell
\]

This difference vector is treated as the **script direction**.

---

#### (3) Activation addition
At inference time, the vector is added to the decoder hidden states:

\[
h^{steered}_{t,\ell} = h_{t,\ell} + \sigma r_\ell
\]

where \(\sigma\) controls the steering strength.

So instead of retraining the model, the method directly modifies internal activations to influence the output script.

---

## 5. Why this method is special
### 5.1 No fine-tuning required
The method is entirely inference-time and training-free.

### 5.2 Sample-efficient
The paper shows that even **one high-quality example pair** can be enough to extract a useful script vector.

### 5.3 Can induce unconventional script pairings
Because the intervention is done in representation space, it can produce outputs like:

- Italian in Cyrillic
- Japanese in Latin script

These are not straightforwardly handled by deterministic transliteration tools.

---

## 6. Training / evaluation data
The paper uses the **FLEURS** dataset, a multilingual speech dataset with aligned audio and transcripts.

### Languages studied
- Serbian
- Mandarin
- Russian
- Italian
- Hindi
- Greek
- Japanese
- Korean

These languages are selected because they are useful for:

1. **script confusion** tasks
2. **transliteration** tasks

---

## 7. Role of prompts
Prompts are used to bias the decoder toward a script.

The paper considers:

### no-prompt
Audio only, no text biasing

### prompt
A target-script prompt is prepended to bias the output script

### steer
The prompt-based activations are used to derive a script vector, which is then added during decoding

---

## 8. Filtering low-quality activations
To get a clean script vector, the paper filters out examples where the model’s output does not match the desired script well.

They compute normalized Levenshtein edit distance and keep only examples below a threshold \(\theta\).

This removes noisy activations and improves the vector quality.

---

## 9. Two main experimental settings
### 9.1 Script confusion mitigation
They test whether the method can control the script within the same language.

Examples:
- Serbian Latin vs. Cyrillic
- Simplified vs. Traditional Chinese

### 9.2 Transliteration
They also test transliteration into another script.

Examples:
- Japanese → Latin
- Greek → Latin
- Italian → Cyrillic

---

## 10. Zero-shot and pseudo-label variants
### zero-shot
A script vector learned from one language is applied directly to another language.

### pseudo-label
The zero-shot output is used as pseudo transcription to learn a more adapted script vector.

This allows refinement of the script direction for the new language.

---

## 11. Layer-wise steering
The authors steer **all decoder layers** rather than only one layer.

Why?
Because script information appears to be linearly separable in all decoder layers, as confirmed by their probing analysis.

---

## 12. Evaluation
They use character-level normalized edit similarity between the predicted transcription and the reference.

Higher scores mean:
- better script control
- better transliteration quality

---

## 13. One-sentence summary
> The method extracts a linear script direction from Whisper decoder activations and adds it at inference time to control the output script without retraining.

---



<br/>
# Results




## 1) 이 논문의 결과가 무엇을 보여주려는가
이 논문의 핵심 결과는 다음과 같습니다.

1. **Whisper 같은 다국어 음성 foundation model은 출력 script를 일관되게 제어하지 못할 수 있다.**
2. **Script 정보는 activation space에서 선형 방향(linear direction)으로 표현된다.**
3. **이 방향(script vector)을 추출해서 테스트 시점에 hidden state에 더하면, 원하는 script로 출력이 바뀐다.**
4. 이 방법은
   - **스크립트 혼동(script confusion)** 을 줄이는 데 효과적이고,
   - **zero-shot transliteration** 도 가능하게 한다.
5. 특히 **학습 없이(inference-time, training-free)** 동작하며,
   **일부 경우에는 기존 prompting보다 더 잘 작동**한다.

---

## 2) 실험에서 사용한 경쟁 모델(Competitor / Baseline)
논문은 주로 **Whisper** 계열 모델을 사용했습니다.  
실험한 모델 크기는 다음과 같습니다.

- tiny
- base
- small
- medium
- large
- large-v2
- large-v3

즉, **여러 크기의 Whisper 모델 전체를 비교**하여, 모델 크기에 따라 script 제어 성능이 어떻게 달라지는지 봤습니다.

비교 대상 방법은 크게 세 가지입니다.

### (1) no-prompt
- 아무런 script 유도 프롬프트 없이 그냥 ASR 수행
- 즉, **기본 출력이 얼마나 target script를 따르는지** 측정하는 기준선

### (2) prompt
- decoder에 target script 쪽으로 편향시키는 텍스트 프롬프트를 넣음
- 즉, **입력 프롬프트만으로 script를 제어하려는 baseline**

### (3) steer (ours)
- 논문이 제안한 방식
- source script와 target script의 activation 평균 차이로 **script vector**를 만들고,
- inference 때 hidden state에 더해서 script를 바꿈

### (4) one-shot
- script vector를 단 1개의 예시쌍으로 추출해 steering
- 매우 sample-efficient한지 보기 위한 분석 설정

### (5) zero-shot / pseudo-label
transliteration 실험에서 추가된 설정입니다.

- **zero-shot**: 한 언어에서 얻은 script direction을 다른 언어에 그대로 적용
- **pseudo-label**: zero-shot으로 만든 결과를 pseudo transcription으로 다시 사용해 새 script vector를 재학습

---

## 3) 테스트 데이터
논문은 **FLEURS** 데이터셋을 사용했습니다.

### FLEURS란?
- 100개 이상의 언어에 대해 **speech + text transcription**이 있는 다국어 데이터셋
- 이 논문에서는 각 언어의 train/validation/test split을 활용

### script confusion 실험에 사용한 언어/스크립트
- **Serbian**: Latin vs Cyrillic
- **Mandarin**: Simplified Chinese vs Traditional Chinese

### transliteration 실험에 사용한 언어
#### Romanization
다음 언어를 라틴 문자로 바꾸는 실험:
- Russian
- Hindi
- Greek
- Japanese
- Korean

#### Cyrillization
다음 언어를 키릴 문자로 바꾸는 실험:
- Italian
- Hindi
- Greek
- Japanese
- Korean

추가로, deterministic transliteration tool을 정답 생성용으로 사용했습니다.
- Serbian: **cyrtranslit**
- Mandarin: **OpenCC**
- Romanization: **uroman**, **pykakasi**
- Cyrillization: **ICU**

---

## 4) 평가 메트릭
논문은 **character-level normalized edit similarity**를 사용합니다.

### 계산 방식
1. 예측 결과와 정답 사이의 **edit distance**를 계산
2. 이를 길이로 정규화
3. 그 값을 1에서 빼서 **유사도(similarity)** 로 변환

즉, 식으로 보면:

- normalized edit distance
- 그리고 최종 점수는 `1 - normalized edit distance`

### 의미
- **1.0**: 정답과 완전히 일치
- **0.0에 가까울수록** 전혀 맞지 않음

논문에서는 이 점수를 편의상 **accuracy**라고 부릅니다.

### 전처리
평가를 공정하게 하기 위해:
- punctuation 제거
- spacing 제거
- target script가 아닌 부분 제거

즉, **script와 transliteration 자체의 품질**만 보려는 설정입니다.

---

## 5) 결과 1: Script confusion mitigation
이 실험의 목표는:
- 모델이 같은 언어에서도 script를 혼동하는 현상을 얼마나 줄이는가
- prompt보다 steering이 더 좋은가
를 보는 것입니다.

### 주요 결과
논문 결과에 따르면:

#### (a) 작은 모델에서 prompting보다 steering이 더 강함
- Whisper tiny/base/small 같은 작은 모델에서는
  prompt만으로는 script 제어가 약한 경우가 많음
- 반면 **steer는 더 강하게 target script를 유도**

예:
- Serbian Cyrillic에서 특히 차이가 큼
- tiny 모델에서는 prompt가 거의 못 바꾸는 반면 steer는 크게 개선

#### (b) 더 희소한 script에서 개선이 두드러짐
- Serbian: Cyrillic
- Chinese: Traditional

이런 덜 자주 등장하는 script에서
- no-prompt보다 prompt가 좋아지고
- steer도 더 좋아짐

반면,
- Serbian Latin
- Chinese Simplified

처럼 이미 많이 나오는 script는 상대적으로 개선 폭이 작았습니다.

#### (c) one-shot도 꽤 강함
- script vector를 **한 개의 예시**만으로도 학습 가능
- 즉, 매우 sample-efficient

---

## 6) 결과 2: Transliteration
이 실험은 단순히 같은 언어의 다른 script가 아니라,
**아예 다른 언어의 문자 체계로 옮겨 적는 것**입니다.

### 6-1) Romanization 결과
Romanization은 대체로 잘 되었습니다.

#### zero-shot 결과
논문은 Serbian에서 얻은 romanization direction을 다른 언어에 적용했습니다.

결과:
- Hindi: 약 **69%**
- Russian: 약 **67%**
- Greek / Japanese / Korean도 의미 있는 성능

즉, **romanization direction이 언어 간에 꽤 일반화**됨을 보여줍니다.

#### pseudo-label 결과
zero-shot 출력으로부터 다시 vector를 학습하면 성능이 더 상승합니다.

- 모든 언어에서 전반적 향상
- 특히 zero-shot이 불안정한 언어에서 효과적

이 결과는:
- **script direction이 언어 간에 유사하다**
- 그리고 **출력 자체를 다시 학습 신호로 써서 더 정교한 direction을 만들 수 있다**
는 점을 시사합니다.

---

### 6-2) Cyrillization 결과
Cyrillization은 romanization보다 어려웠습니다.

#### 전체 경향
- 대부분의 언어에서 성능이 제한적
- 그런데 **Italian**에서는 상대적으로 잘 됨
- Italian은 **약 43% accuracy**를 보임

논문 해석:
- Italian은 Latin script 기반 고자원 언어이므로
- Whisper의 학습 분포와 더 잘 맞았을 가능성이 큼

#### 의미
이 결과는:
- script steering이 단순히 학습된 특정 언어의 암기라기보다
- **script 수준의 구조를 어느 정도 포착**하고 있음을 보여줍니다.

---

## 7) 결과 3: Script vector는 one-shot으로도 학습 가능
논문은 script vector를 배우는 데 **단 한 쌍의 좋은 예시**만으로도 충분한지 확인했습니다.

### 결과
- one-shot만으로도 유의미한 steering 가능
- 특히 Cyrillic/Traditional Chinese 같은 설정에서 개선됨

### 의미
이것은 이 방법이:
- 데이터가 많이 필요하지 않고
- training 없이도 빠르게 적용 가능하다는 점을 보여줍니다.

---

## 8) 결과 4: Script 정보는 모든 decoder layer에서 선형적으로 분리됨
논문은 probe를 사용해서 script 정보가 layer-wise로 어떻게 존재하는지 분석했습니다.

### 결과
- Serbian과 Mandarin 모두에서
- **모든 decoder layer에서 script가 선형적으로 분리 가능**

즉:
- script 정보는 특정 layer에만 있는 것이 아니라
- decoder 전반에 걸쳐 선형 방향으로 존재한다고 해석할 수 있습니다.

### 의미
이 결과는 왜
- 모든 layer에 steering을 적용하는 방식이 효과적인지
설명해 줍니다.

---

## 9) 결과 5: Steering 결과는 실제 발음에 더 가까운 transliteration을 만들기도 함
논문은 deterministic transliteration과 비교해,
steering 결과가 더 **phonetic, sound-based**인 경우가 많다고 설명합니다.

예:
- Korean, Greek, Japanese, Hindi, Russian 등에서
- deterministic mapping은 철자 규칙에만 충실해 실제 발음과 어긋나는 경우가 있음
- 반면 steering은 실제 발음에 더 맞는 romanization/cyrillization을 생성하는 경우가 많음

즉, 이 방법은 단순한 문자 치환이 아니라,
**발음 기반 transliteration**을 어느 정도 반영합니다.

---

## 10) 종합 해석
이 논문의 실험 결과를 한 문장으로 정리하면:

> Whisper 같은 다국어 음성 모델의 activation space에는 script 정보가 선형적으로 존재하며, 이를 inference-time에 조작하면 prompting보다 더 안정적으로 script control 및 transliteration을 수행할 수 있다.

### 특히 중요한 포인트
- **학습 없이 가능**
- **one-shot도 가능**
- **zero-shot cross-lingual generalization 가능**
- **작은 모델에서 prompt보다 강함**
- **발음 기반 transliteration까지 유도 가능**

---




## 1) What the results are trying to show
The paper’s main empirical claims are:

1. Multilingual speech foundation models such as Whisper do not always produce a consistent output script.
2. Script is encoded as a **linear direction** in the model’s activation space.
3. By adding a script vector to hidden states at inference time, one can steer the model toward a desired script.
4. This works for:
   - **mitigating script confusion**
   - **zero-shot transliteration**
5. The method is **training-free** and often **competitive with or better than prompting**.

---

## 2) Competitor models / baselines
The paper evaluates multiple Whisper model sizes:

- tiny
- base
- small
- medium
- large
- large-v2
- large-v3

So the comparison is not limited to one model; it spans the full Whisper family.

### Baselines
#### (1) no-prompt
- Run ASR with no script-inducing prompt
- Serves as the base reference

#### (2) prompt
- Prepend a text prompt in the target script
- A context-based baseline for controlling output script

#### (3) steer (ours)
- Compute a **script vector** from the difference between source-script and target-script activations
- Add it to hidden states during inference

#### (4) one-shot
- Learn the script vector from only a single example pair
- Tests sample efficiency

#### (5) zero-shot / pseudo-label
Used in transliteration experiments:

- **zero-shot**: apply a script vector learned from one language to others
- **pseudo-label**: use steered outputs as pseudo-transcriptions and relearn the vector

---

## 3) Test data
The paper uses the **FLEURS** dataset, which contains speech-text pairs for 100+ languages.

### Script confusion setting
- Serbian: Latin vs Cyrillic
- Mandarin: Simplified vs Traditional Chinese

### Transliteration setting
#### Romanization
- Russian
- Hindi
- Greek
- Japanese
- Korean

#### Cyrillization
- Italian
- Hindi
- Greek
- Japanese
- Korean

The paper uses deterministic transliteration tools to build references:
- Serbian: cyrtranslit
- Mandarin: OpenCC
- Romanization: uroman, pykakasi
- Cyrillization: ICU

---

## 4) Evaluation metric
The paper uses **character-level normalized edit similarity**.

### Procedure
1. Compute character-level edit distance between prediction and reference
2. Normalize by sequence length
3. Convert to similarity via `1 - normalized edit distance`

### Interpretation
- **1.0** = exact match
- lower values = worse match

The authors call this similarity score **accuracy**.

### Preprocessing
To focus on transliteration quality:
- punctuation is removed
- spacing is removed
- parts not in the target script are removed

---

## 5) Results 1: Script confusion mitigation
This setup measures how well the method reduces script instability.

### Main findings
#### (a) Steering is more effective than prompting in smaller models
- For Whisper tiny/base/small, prompting often has limited effect
- Activation steering improves target-script accuracy more strongly

#### (b) Gains are stronger for less frequent scripts
- Serbian Cyrillic
- Traditional Chinese

These scripts benefit more than the already frequent scripts:
- Serbian Latin
- Simplified Chinese

#### (c) One-shot steering is surprisingly effective
- A single high-quality example can already produce a useful script vector
- Very sample-efficient

---

## 6) Results 2: Transliteration
The transliteration setting tests whether the method can generate output in a different script not associated with the source language.

### 6-1) Romanization
Romanization works well overall.

#### Zero-shot
A romanization vector learned from Serbian transfers reasonably well to other languages.

Reported examples:
- Hindi: around **69%**
- Russian: around **67%**

This suggests romanization directions are broadly shared across languages.

#### Pseudo-label
Using steered outputs as pseudo-labels improves results further across all languages.

---

### 6-2) Cyrillization
Cyrillization is harder than romanization.

#### Overall trend
- Performance is more limited across most languages
- But Italian stands out with about **43% accuracy**

The authors suggest that Italian benefits because it is a high-resource Latin-script language, which better matches Whisper’s training distribution.

---

## 7) One-shot learning of script vectors
The paper shows that script vectors can be learned from just **one example pair**.

### Result
- One-shot steering already works
- It is especially useful for high-accuracy, low-data settings

### Implication
The method is extremely sample-efficient and requires no finetuning.

---

## 8) Script information is linearly separable in all decoder layers
The authors use a linear probe to examine layer-wise script information.

### Result
- Script is linearly separable across **all decoder layers**
- This holds for both Serbian and Mandarin

### Interpretation
This supports the claim that script is represented as a linear direction throughout the decoder, which explains why steering all layers is effective.

---

## 9) Steering often yields more phonetic transliterations
Compared with deterministic transliteration tools, the steering output often better reflects actual pronunciation.

Examples discussed include:
- Korean
- Greek
- Japanese
- Hindi
- Russian

This suggests the method is not just copying orthographic rules, but sometimes captures **sound-based transliteration** better.

---

## 10) Overall takeaway
In short, the results show that:

> script information is linearly encoded in Whisper’s activation space, and inference-time activation steering can control output script and enable zero-shot transliteration without training.

Key strengths:
- training-free
- one-shot possible
- zero-shot cross-lingual transfer
- stronger than prompting in small models
- often more phonetic than deterministic transliteration

---





<br/>
# 예제




## 1) 이 논문의 핵심 테스크는 무엇인가?

이 논문은 Whisper 같은 **멀티링구얼 음성 인식 모델**이 출력하는 문자 결과에서,  
**“언어는 맞는데 script(문자 체계)가 흔들리는 문제”**를 다룹니다.

쉽게 말하면:

- 같은 한국어/세르비아어/중국어라도
- 어떤 경우는 **한글/라틴/키릴/간체/번체** 등 서로 다른 script로 출력될 수 있고
- 사용자는 원하는 script로 결과를 통제하고 싶습니다.

논문은 이를 두 가지 큰 테스크로 나눕니다.

1. **Script confusion mitigation**  
   - 같은 언어 안에서 script가 뒤섞이는 문제를 줄이기
   - 예: 세르비아어가 라틴으로 나올지 키릴로 나올지 통제

2. **Transliteration**
   - 음성의 내용을 **다른 script로 옮겨 적기**
   - 예: 이탈리아어 음성을 **키릴 문자**로 출력하거나  
     일본어를 **라틴 문자(로마자)** 로 출력

---

## 2) 데이터는 무엇을 쓰나?

논문은 주로 **FLEURS**라는 데이터셋을 사용합니다.  
FLEURS는 각 언어별로 **음성(audio)** 와 **정답 전사(text transcription)** 가 있는 멀티링구얼 음성 데이터입니다.

즉, 각 샘플은 대략 이런 구조입니다.

- **입력:** 음성 파일
- **정답:** 그 음성의 텍스트 전사
- **언어별, script별 전사 형태가 다름**

예를 들어:

- 러시아어 음성 → 정답 전사: 키릴 문자
- 힌디어 음성 → 정답 전사: 데바나가리 문자
- 일본어 음성 → 정답 전사: 일본어 문자
- 세르비아어 음성 → 정답 전사: 라틴 또는 키릴

---

## 3) “트레이닝 데이터”는 실제로 무엇을 의미하나?

이 논문에서 말하는 “training”은 일반적인 의미의 모델 재학습이 아닙니다.  
**추가 파인튜닝(finetuning)** 없이, **activation vector(steering vector)** 를 뽑기 위한 **예시 샘플 집합**을 말합니다.

즉:

- Whisper를 새로 학습하는 것이 아님
- 모델 내부 활성값(activations)을 모아서
- “source script”와 “target script”의 차이를 계산함

### 구체적으로 하는 일
논문 방법론에 따르면, 학습/추출용 데이터는 다음처럼 사용됩니다.

#### (1) 같은 음성 샘플에 대해 두 가지 프롬프트로 디코딩
예를 들어 세르비아어 샘플 하나가 있으면:

- **source prompt**: 라틴 스크립트 쪽으로 유도
- **target prompt**: 키릴 스크립트 쪽으로 유도

그러면 같은 음성에 대해 모델이 서로 다른 script로 출력할 수 있습니다.

#### (2) 각 출력에서 decoder layer의 activation을 수집
각 레이어별 hidden state를 모읍니다.

#### (3) source와 target의 평균 activation 차이를 계산
이 차이가 바로 **script direction / script vector** 입니다.

논문 수식은 대략 이런 구조입니다:

- source activation 평균: \(v^{SRC}_\ell\)
- target activation 평균: \(v^{TRG}_\ell\)
- script vector: \(r_\ell = v^{SRC}_\ell - v^{TRG}_\ell\)

즉,  
**“라틴→키릴”, “간체→번체” 같은 script 차이를 나타내는 방향 벡터**를 activation space에서 찾는 것입니다.

---

## 4) 트레이닝 데이터의 구체적 예시

논문은 각 설정에서 보통 **10개 샘플 정도**를 사용합니다.  
그리고 조건을 만족하는 샘플만 필터링합니다.

예를 들어 script confusion 실험에서는:

- 세르비아어의 라틴/키릴
- 중국어의 간체/번체

에 대해 train split에서 일부 샘플을 사용합니다.

### 예시: 세르비아어 script confusion
- 입력:
  - 세르비아어 음성 10개 정도
  - 각 음성에 대해 라틴 스크립트 prompt와 키릴 스크립트 prompt를 각각 넣어 decoding
- 출력:
  - 라틴 전사 / 키릴 전사
- 목적:
  - 라틴과 키릴의 activation 차이를 구해서 script vector 추출

예시로 논문 Table 3의 프롬프트는 다음과 같습니다.

- Serbian (Latin): **Ovo je srpska rečenica**
- Serbian (Cyrillic): **Ово jе српска реченица**

둘 다 “이것은 세르비아어 문장입니다”라는 의미지만, script만 다릅니다.

---

## 5) 테스트 데이터는 무엇이고 입력/출력은 어떻게 되나?

테스트에서는 **새로운 음성**이 들어옵니다.  
그리고 모델의 출력이 원하는 script로 나오게 하는 것이 목표입니다.

---

## 6) 테스크 1: Script confusion mitigation

### 문제 정의
같은 언어인데 script가 불안정하게 출력되는 문제를 해결합니다.

### 대표 언어
- 세르비아어: Latin / Cyrillic
- 중국어: Simplified / Traditional

### 테스트 입력
- 새로운 세르비아어 음성
- 새로운 중국어 음성

### 테스트 출력
- 사용자가 원하는 script의 전사
  - 예: 세르비아어를 키릴로
  - 예: 중국어를 번체로

### 비교 방법
논문은 세 가지를 비교합니다.

1. **no-prompt**
   - 아무 prompt 없이 그냥 음성만 넣음
   - 모델이 알아서 출력
2. **prompt**
   - 원하는 script의 예시 문장을 prompt로 줌
3. **steer**
   - activation에 script vector를 더해 원하는 script로 유도

### 예시
세르비아어 음성의 정답이 “라틴”일 수도 “키릴”일 수도 있는데,  
사용자가 키릴을 원한다고 합시다.

- **입력:** 세르비아어 음성
- **no-prompt 출력:** 라틴으로 나올 수도 있음
- **prompt 출력:** 키릴로 유도
- **steer 출력:** 키릴로 강하게 유도

논문 결과상 특히 작은 Whisper 모델에서  
**prompt보다 steer가 더 잘 script를 바꾸는 경우**가 있었습니다.

---

## 7) 테스크 2: Transliteration

이 테스크가 훨씬 흥미로운 부분입니다.  
여기서는 **소스 언어는 그대로인데, 출력 script를 다른 언어권 script로 바꾸는 것**입니다.

즉, 단순히 같은 언어의 다른 script가 아니라,  
**“전혀 일반적이지 않은 언어-script 조합”**까지 실험합니다.

### 예시
- 이탈리아어 음성 → 키릴 문자로 출력
- 일본어 음성 → 로마자(Latin)로 출력
- 힌디어 음성 → 로마자로 출력
- 그리스어 음성 → 로마자로 출력
- 러시아어 음성 → 로마자로 출력

---

## 8) Transliteration에서의 트레이닝/테스트 구조

논문은 transliteration을 위해 다음 두 가지를 봅니다.

### A. Zero-shot
한 언어에서 뽑은 script vector를 다른 언어에 바로 적용

예:
- 세르비아어에서 얻은 “romanization vector”를
- 러시아어, 힌디어, 일본어, 한국어, 그리스어에 그대로 적용

즉:

- **학습/추출용 예시:** 세르비아어
- **테스트 대상:** 러시아어, 힌디어, 일본어, 한국어, 그리스어
- **목표 출력:** 각 언어를 라틴 문자로 옮겨 적기

이게 zero-shot입니다.

### B. Pseudo-label
zero-shot으로 생성된 결과를 다시 “가짜 정답(pseudo label)”처럼 써서  
새로운 script vector를 다시 추출합니다.

즉:

1. 첫 번째 vector로 한번 transliteration 생성
2. 그 출력물을 pseudo transcription으로 사용
3. 새로운 vector를 다시 뽑음
4. 더 잘 맞는 방향으로 개선

---

## 9) 구체적인 예시: Romanization

### 목표
비라틴 script의 음성을 **라틴 문자로** 출력하기

### 훈련(정확히는 vector 추출)용 예시
Table 4에 있는 romanization prompt 예시:

- Russian: **eto russkoye predlozheniye**
- Greek: **auti einai mia elliniki protasi**
- Hindi: **yah ek hindi vaakya hai**
- Korean: **igeoseun hangugeo munjangibnida**
- Japanese: **kore wa nihongo no bun desu**

이 프롬프트들은 “이것은 X 언어의 문장입니다”라는 의미입니다.  
즉, 디코더가 특정 script로 출력하도록 context를 줍니다.

### 테스트 입력 예시
예를 들어 러시아어 음성이 들어오면:

- **입력:** 러시아어 음성
- **정답 출력:** 라틴 문자로 옮긴 romanization
- **예상 출력 예:**  
  `del patro imel rannee preimushchestvo vo vtorom sete...`

논문 Table 1/8에서처럼, deterministic romanization과 비교해  
steered output이 더 발음에 가까운 경우가 있었습니다.

---

## 10) 구체적인 예시: Cyrillization

### 목표
원래 라틴이나 다른 script로 쓰는 언어를 **키릴 문자로** 출력하기

### 훈련/프롬프트 예시
Table 5의 cyrillization prompt:

- Italian: **куэста э уна фразе итальяна**
- Greek: **афти инэ мия эллиники протаси**
- Hindi: **йе эк хинди вакья хэ**
- Korean: **игосын хангуго мунджанимнида**
- Japanese: **корэ ва нихонго но бун десу**

### 테스트 예시
- **입력:** 이탈리아어 음성
- **출력 목표:** 키릴 문자 transliteration
- **예시 출력:**  
  `нелло специфико си состене ...`

논문에서는 **이탈리아어 → 키릴**이 의외로 꽤 잘 되는 편이었다고 보고합니다.

---

## 11) 입력과 출력의 관계를 표처럼 정리하면

### 11-1. Script confusion
- **입력:** 같은 언어의 음성
- **중간 조작:** prompt 또는 activation steering
- **출력:** 원하는 script의 전사
- **예시**
  - 세르비아어 음성 → 키릴 전사
  - 중국어 음성 → 번체 전사

### 11-2. Transliteration
- **입력:** 특정 언어 음성
- **중간 조작:** 다른 언어/다른 script의 방향 벡터 적용
- **출력:** 발음 기반 transliteration
- **예시**
  - 일본어 음성 → 로마자
  - 이탈리아어 음성 → 키릴

---

## 12) 이 논문에서 “구체적인 인풋”의 실제 형태

논문의 실제 입력은 보통 아래 두 가지가 결합된 형태입니다.

### A. Audio
- FLEURS에서 가져온 음성 클립
- 예: 일본어 문장 녹음, 힌디어 문장 녹음 등

### B. Prompt
- 디코더에 넣는 짧은 텍스트
- 목적: 원하는 script를 context로 유도

예:
- `kore wa nihongo no bun desu`
- `Ово jе српска реченица`

### C. Activation steering vector
- 학습이라기보다 추출된 벡터
- 각 decoder layer에 더해짐

즉, 실제 inference input은:

> 음성 + prompt + script vector

가 됩니다.

---

## 13) 이 논문의 출력은 어떤 모양인가?

출력은 일반 ASR 결과처럼 보이지만,  
**원하는 script로 써진 transcription**입니다.

예를 들어 일본어 음성에서:

- 일반 ASR 출력: 일본어 문자
- romanization 목표 출력: `hong kong no chiheisen wo egai te...`
- cyrillization 목표 출력: 키릴 문자 형태의 음성 전사

또 다른 예로, 세르비아어 음성에서:

- 그냥 나오면 라틴일 수도 있음
- steering하면 키릴로 바뀜

---

## 14) 논문이 강조하는 포인트

이 논문에서 중요한 것은 단순히 “번역”이 아니라:

1. **script가 hidden state에 선형적으로 표현된다**
2. **그 방향을 vector로 뽑을 수 있다**
3. **추가 학습 없이 inference-time에 script를 바꿀 수 있다**
4. **심지어 일반적이지 않은 script 조합도 유도 가능하다**
   - 예: 이탈리아어를 키릴로
   - 일본어를 라틴으로

---

## 15) 아주 간단한 예시로 다시 정리

### 예시 1: 세르비아어 라틴→키릴
- **훈련용 입력:** 세르비아어 음성 몇 개
- **훈련용 출력:** 라틴/키릴 둘 다 생성
- **벡터 추출:** 라틴과 키릴의 차이
- **테스트 입력:** 새로운 세르비아어 음성
- **테스트 출력:** 키릴 문장으로 전사

### 예시 2: 일본어→로마자
- **훈련용 입력:** 세르비아어에서 얻은 romanization vector 혹은 일본어 프롬프트 활용
- **테스트 입력:** 일본어 음성
- **테스트 출력:** `kore wa nihongo no bun desu`

### 예시 3: 이탈리아어→키릴
- **훈련/벡터:** cyrillization vector
- **테스트 입력:** 이탈리아어 음성
- **테스트 출력:** 키릴 문자 transliteration

---



---

## 1) What is the main task in the paper?

The paper studies a problem in multilingual speech recognition models like Whisper:

- the model may recognize the correct **language**, but
- the **script** of the output can be unstable or inconsistent.

For example, a language may be written in multiple scripts:
- Serbian: Latin or Cyrillic
- Chinese: Simplified or Traditional
- Japanese/Korean/Russian/Greek/Hindi: can be transliterated into Latin script
- Italian (in an unusual setup): can be transliterated into Cyrillic

The paper focuses on two main tasks:

1. **Script confusion mitigation**
   - controlling the output script within the same language
   - e.g., Serbian in Latin vs. Cyrillic, Chinese in Simplified vs. Traditional

2. **Transliteration**
   - writing speech from one language in another script
   - e.g., Japanese speech in Latin script, Italian speech in Cyrillic script

---

## 2) What data is used?

The paper uses **FLEURS**, a multilingual speech dataset containing:

- **audio**
- **text transcriptions**

So each example is basically:

- **input:** spoken audio
- **reference:** the transcription in a given script

Depending on the language, the transcription can appear in different scripts.

Examples:
- Russian audio → Cyrillic transcription
- Hindi audio → Devanagari transcription
- Serbian audio → Latin or Cyrillic transcription
- Chinese audio → Simplified or Traditional transcription

---

## 3) What does “training data” mean here?

This paper does **not** fine-tune Whisper in the usual way.  
Instead, it uses a small set of examples to **extract steering vectors** from the model’s activations.

So the “training” stage is really:

- collect activations from the decoder
- compare source-script and target-script behaviors
- compute a vector that represents the script direction

### How it works
For one audio example, the model is decoded with two prompts:

- **source prompt**: biases output toward one script
- **target prompt**: biases output toward another script

Then the decoder activations are collected and averaged, and the difference is taken:

- source mean activation
- target mean activation
- script vector = source minus target

This vector is interpreted as a **linear direction in activation space**.

---

## 4) Concrete example of training/vector extraction

### Example: Serbian script confusion
Suppose we want to move from Latin to Cyrillic.

- **input audio:** Serbian speech
- **source prompt:** Serbian in Latin script
- **target prompt:** Serbian in Cyrillic script
- **output:** two transcriptions of the same audio, one Latin and one Cyrillic
- **goal:** subtract the activations to isolate the Latin-vs-Cyrillic direction

The paper’s example prompts are:

- Serbian (Latin): `Ovo je srpska rečenica`
- Serbian (Cyrillic): `Ово jе српска реченица`

Both mean “This is a Serbian sentence,” but the scripts differ.

---

## 5) What is the test data and what are the test inputs/outputs?

At test time, the input is **new unseen speech audio**.  
The goal is to make the output appear in the desired script.

The model can be controlled in three ways:

1. **no-prompt**
   - just audio only
2. **prompt**
   - audio plus a text prompt in the target script
3. **steer**
   - audio plus a script vector added to the decoder activations

---

## 6) Task 1: Script confusion mitigation

### Task definition
The goal is to reduce instability when a language has multiple scripts.

### Languages studied
- Serbian: Latin vs. Cyrillic
- Chinese: Simplified vs. Traditional

### Test input
- new Serbian audio
- new Chinese audio

### Test output
- transcription in the desired script

### Example
If the audio is Serbian and the desired output is Cyrillic:

- **input:** Serbian speech audio
- **no-prompt output:** may come out in Latin
- **prompt output:** script can be nudged toward Cyrillic
- **steer output:** script can be strongly shifted to Cyrillic

The paper reports that activation steering can be especially effective in smaller Whisper models.

---

## 7) Task 2: Transliteration

This is the more unusual and interesting setting.

Here the model is asked to output a language in a **different script**, often an unconventional language-script pairing.

### Examples
- Italian speech → Cyrillic script
- Japanese speech → Latin script
- Korean speech → Latin script
- Greek speech → Latin script
- Hindi speech → Latin script

---

## 8) Training/test structure for transliteration

The paper considers two strategies:

### A. Zero-shot
A script vector extracted from one language is directly applied to other languages.

Example:
- derive a romanization vector from Serbian
- apply it to Russian, Hindi, Greek, Japanese, Korean

So:
- **training/vector source:** Serbian
- **test languages:** others like Russian/Hindi/Greek/etc.
- **goal:** output Latin-script transliterations

### B. Pseudo-label
The zero-shot outputs are treated as pseudo-transcriptions and used to learn a better vector for the new language.

So:
1. use the initial steering vector
2. generate transliterations
3. treat those outputs as pseudo-labels
4. re-extract a better vector

---

## 9) Concrete example: Romanization

### Goal
Convert speech from a non-Latin script into Latin script.

### Prompts used to extract/guide the vector
The paper uses prompts like:

- Russian: `eto russkoye predlozheniye`
- Greek: `auti einai mia elliniki protasi`
- Hindi: `yah ek hindi vaakya hai`
- Korean: `igeoseun hangugeo munjangibnida`
- Japanese: `kore wa nihongo no bun desu`

These mean “This is a sentence in X language,” serving as script-biasing prompts.

### Test example
- **input:** Russian speech
- **expected output:** Latin-script romanization
- **example output:** something like  
  `del patro imel rannee preimushchestvo...`

---

## 10) Concrete example: Cyrillization

### Goal
Convert speech into Cyrillic script, even from languages that are not normally written in Cyrillic.

### Prompts used
The paper uses prompts like:

- Italian: `куэста э уна фразе итальяна`
- Greek: `афти инэ мия эллиники протаси`
- Hindi: `йе эк хинди вакья хэ`
- Korean: `игосын хангуго мунджанимнида`
- Japanese: `корэ ва нихонго но бун десу`

### Test example
- **input:** Italian speech
- **output:** Cyrillic transliteration
- **example output:**  
  `нелло специфико си состене ...`

The paper notes that Italian performed surprisingly well in the Cyrillic setting.

---

## 11) Simple input/output summary

### Script confusion
- **input:** speech in a language with multiple scripts
- **intervention:** prompt or activation steering
- **output:** the desired script
- **example:** Serbian speech → Cyrillic transcription

### Transliteration
- **input:** speech in a source language
- **intervention:** a learned script vector
- **output:** transcription in a different script
- **example:** Japanese speech → Latin script

---

## 12) What does the actual model input look like?

At inference time, the model uses:

- **audio**
- **prompt**
- **activation steering vector**

So the practical input is effectively:

> audio + prompt + script vector

---

## 13) What does the output look like?

The output is an ASR transcript, but in the target script.

Examples:
- Japanese speech → Latin text transliteration
- Serbian speech → Cyrillic transcription
- Italian speech → Cyrillic transliteration

---

## 14) Main takeaway

The paper shows that:

1. script information is linearly represented in the model’s activation space
2. a script direction can be extracted as a vector
3. that vector can be added at inference time
4. this allows script control without finetuning
5. even unusual transliteration directions are possible

---




<br/>
# 요약


이 논문은 Whisper 같은 음성 기초모델의 디코더 활성값에서 **script(문자 체계)가 선형 방향으로 표현**된다는 점을 이용해, 소스/타깃 script 샘플의 평균 활성 차이로 **script vector**를 추출하고 테스트 시 활성에 더해 원하는 script로 전사되게 하는 방법을 제안합니다.  
결과적으로 별도 파인튜닝 없이도 **script confusion**을 줄이고, **romanization/cyrillization**을 zero-shot으로 유도했으며, 특히 적은 예시(심지어 1개)만으로도 작동하고 Italian→Cyrillic 같은 **비정상적 언어-스크립트 조합**까지 가능함을 보였습니다.  
예시로는 Serbian의 Cyrillic/Latin, Mandarin의 Traditional/Simplified 전환은 물론, 일본어·한국어·그리스어·러시아어를 라틴 문자로, 이탈리아어를 키릴 문자로 바꾸는 transliteration이 가능했고, 생성 결과가 단순 철자 매핑보다 **실제 발음에 더 가깝다**는 점도 관찰했습니다.




This paper proposes a **script vector** method that exploits the fact that **script is linearly represented** in decoder activations of speech foundation models like Whisper: it extracts a script direction from the mean activation difference between source and target script examples, then adds that vector at inference time to steer transcription into the desired script.  
The method improves **script confusion** without fine-tuning, enables **zero-shot romanization/cyrillization**, works with as few as one example, and even generalizes to **unconventional language-script pairings**.  
Examples include switching between Serbian Latin/Cyrillic and Mandarin Traditional/Simplified, romanizing Japanese/Korean/Greek/Russian, and cyrillizing Italian, with outputs often being **more phonetically faithful** than deterministic transliteration.

<br/>
# 기타



## 1) 다이어그램/개념도

### Figure 1: 핵심 아이디어
- Whisper 같은 음성 foundation model의 **activation space** 안에 **script 정보가 선형적으로 존재**한다는 가설을 보여줍니다.
- 오디오를 인코딩한 뒤 디코더 활성값에 **script vector**를 더하면, 출력 스크립트를 바꿀 수 있습니다.
- 예시:
  - 이탈리아어 *buongiorno*를 일반적으로는 라틴 문자로 출력하지만,
  - script vector를 더하면 **키릴 문자 형태(буонджорно)**로 바뀝니다.

**인사이트**
- 스크립트 변환이 단순 후처리나 규칙기반이 아니라, **모델 내부 표현을 조작해서 직접 제어 가능**하다는 점을 시각적으로 제시합니다.
- 기존의 transliteration이 아닌, **zero-shot에 가까운 스크립트 steering** 가능성을 강조합니다.

---

### Figure 2: Script vector 추출 방법
3단계로 설명됩니다.

1. **Collect**
   - source script와 target script 각각에 대해 디코더 activation을 수집
2. **Isolate**
   - source 평균 activation에서 target 평균 activation을 빼서 script direction 추출
   - 식: `r_l = v_SRC_l - v_TRG_l`
3. **Add**
   - 추론 시 해당 vector를 activation에 더해 원하는 script로 유도

**인사이트**
- 이 방법은 학습(finetuning) 없이 가능한 **inference-time intervention**입니다.
- 핵심은 “스크립트도 개념처럼 선형 방향으로 존재한다”는 점입니다.

---

## 2) Figure 결과 해석

### Figure 3: Script confusion mitigation
- 비교 방법:
  - **no-prompt**
  - **prompt**
  - **steer (ours)**
  - **one-shot**
- 대상:
  - Serbian: Latin / Cyrillic
  - Chinese: Simplified / Traditional
- 결과:
  - 작은 모델에서는 prompt만으로는 한계가 있는데, **steering이 더 강하게 script를 바꾸는 경우가 많음**
  - 특히 **Serbian Cyrillic**처럼 덜 자주 등장하는 스크립트에서 개선이 큼
  - 모델이 커질수록 prompt 성능도 좋아지지만, steering은 여전히 경쟁력 있음

**인사이트**
- “스크립트 혼동(script confusion)”은 특히 **저자원/비주류 스크립트**에서 심각함.
- activation steering은 **작은 모델에서도 script control을 강화**하는 데 유효합니다.
- prompt보다 **내부 표현 조작이 더 직접적인 제어 수단**임을 보여줍니다.

---

### Figure 4: Romanization cross-language generalization
- Serbian에서 학습한 romanization vector를 다른 언어에 zero-shot 적용
- 대상 언어:
  - Russian, Hindi, Greek, Japanese, Korean
- 결과:
  - Hindi와 Russian에서 특히 zero-shot 성능이 높음
  - pseudo-label로 한 번 더 적응시키면 **모든 언어에서 더 좋아짐**

**인사이트**
- romanization direction은 **언어를 넘어 상당히 공유되는 표현**일 가능성이 큽니다.
- 즉, script vector는 특정 언어 전용이라기보다 **cross-lingual하게 재사용 가능**합니다.
- pseudo-label 방식은 “steered output을 다시 학습용 예시처럼 쓰는” 방법으로, **적응 성능을 더 끌어올림**.

---

### Figure 5: Cyrillization across languages
- Latin → Cyrillic 방향으로 유도
- 결과:
  - 전반적으로 romanization보다 어려움
  - 하지만 **Italian**은 예외적으로 좋은 성능(43%)
- 해석:
  - Italian은 Latin script 기반이고, Whisper 학습 데이터에서도 Latin/English 계열이 풍부하기 때문이라고 추정

**인사이트**
- 매우 **비전형적인 language-script pairing**도 어느 정도 가능함.
- 다만 Cyrillic 유도는 romanization보다 난도가 높고, **학습 데이터 분포**의 영향을 강하게 받습니다.
- English/Latin 계열과 음운적으로 가까운 언어일수록 유리할 수 있습니다.

---

### Figure 6: Script probing accuracy across layers
- 각 decoder layer에서 script를 linear probe로 분류
- 결과:
  - **모든 layer에서 script가 분리 가능**
  - 이미 초반 layer에서도 script 정보가 충분히 드러남

**인사이트**
- script 정보는 모델의 특정 마지막 층에만 있는 것이 아니라 **전 layer에 걸쳐 존재**합니다.
- 따라서 steering도 일부 층이 아니라 **모든 층에 적용하는 것이 합리적**입니다.
- 또한 실제 응용에서, 초반 layer만 보고도 script를 예측해 **prompt/steer를 선택적으로 적용**할 가능성을 시사합니다.

---

## 3) 테이블 결과 해석

### Table 1: Sample predictions of romanization with Whisper large-v2
- 여러 언어에 대해 deterministic romanization과 비교
- 결과:
  - deterministic output보다 **발음 기반(transcription/pronunciation-based)**에 더 가까운 경우가 많음
  - 굵게 표시된 부분이 특히 그런 사례
- 예:
  - Korean, Greek, Japanese, Hindi, Russian에서
  - 문자 대응 규칙으로는 어색하지만, steering 결과는 **실제 발음에 더 충실**

**인사이트**
- 이 방법은 단순 문자 치환이 아니라 **소리 기반 transliteration**을 유도하는 경향이 있습니다.
- transliteration의 품질을 “orthography consistency”보다 **phonetic faithfulness** 관점에서 개선할 수 있음을 보여줍니다.

---

### Table 2: Sample predictions of cyrillization with Whisper large-v2
- Italian에서 특히 가장 강한 성능
- 다른 언어도 부분적으로 자연스러운 Cyrillic transliteration 가능

**인사이트**
- 비정상적인 방향인 “Italian → Cyrillic”도 가능하다는 점에서 **개념적 일반화력**을 보여줍니다.
- 완벽하진 않지만, 모델 내부에 **script-independent phonetic alignment**가 어느 정도 존재함을 시사합니다.

---

### Table 6: Script confusion across Whisper models
- 모델 크기별로:
  - `no-prompt`
  - `prompt`
  - `steer`
- 핵심 관찰:
  - 작은 모델에서 `no-prompt`는 script confusion이 심함
  - `prompt`는 개선되지만, 특히 Serbian Cyrillic에서는 작은 모델에서 불안정
  - `steer`는 전반적으로 더 안정적이고 높은 점수

**인사이트**
- 모델이 작을수록 prompt만으로는 제어가 부족할 수 있음.
- steering은 **모델 크기와 관계없이 일관된 제어 신호**를 제공함.
- 특히 비주류 script에서 효과가 더 두드러짐.

---

### Table 7: One-shot script steering
- 단 **한 쌍의 예시**만으로도 script vector를 학습 가능
- 결과:
  - 전체적으로 one-shot만으로도 꽤 높은 성능
  - Serbian Cyrillic, Traditional Chinese 등에서 의미 있는 개선

**인사이트**
- script direction은 매우 **sample-efficient**하게 추출 가능
- 즉, 많은 데이터나 finetuning 없이도 **적은 예시로 제어 벡터를 만들 수 있음**
- 실용성 측면에서 상당히 중요합니다.

---

### Table 8: Romanization / Cyrillization performance
- Whisper large-v2에서의 대표적 성능표
- 주요 패턴:
  - Romanization zero-shot은 Hindi, Russian에서 비교적 높음
  - Pseudo-label은 대체로 더 향상
  - Cyrillization은 전체적으로 더 어렵지만 Italian은 상대적으로 좋음

**인사이트**
- romanization 방향은 언어 간 공유성이 크고,
- pseudo-label을 통해 **target language-specific adaptation**이 가능함.
- cyrillization은 더 어려우나, 적절한 source 언어에서는 꽤 유망함.

---

### Table 9: Dataset statistics
- FLEURS의 train/validation/test 샘플 수를 언어별로 제시
- 상대적으로 다양한 언어에 대해 평가했음을 보여줌

**인사이트**
- 실험이 특정 언어 하나에 치우치지 않고,
- multiple scripts와 다양한 언어 조합에서 검증되었음을 확인할 수 있습니다.

---

### Table 10: Best sigma values
- transliteration에서 사용한 steering strength `sigma`의 최적값 제시
- 언어별로 최적 sigma가 다름

**인사이트**
- script steering은 **강도 조절이 중요**하며,
- 언어/스크립트별로 최적 intervention strength가 다릅니다.
- 너무 약하면 효과가 없고, 너무 강하면 원래 발화를 망칠 수 있습니다.

---

## 4) 어펜딕스(부록) 해석

### Table 3: Script confusion prompts
- 중국어 simplified/traditional, Serbian latin/cyrillic에 대한 prompt 예시 제공
- 모두 “This is a sentence in X”라는 의미의 문장을 각 script로 제시

**인사이트**
- prompt baseline은 **스크립트 문맥을 주는 방식**입니다.
- 그러나 이 방법은 activation steering보다 덜 직접적일 수 있습니다.

---

### Table 4: Romanization prompts
- 러시아어, 그리스어, 힌디어, 한국어, 일본어에 대해 romanization prompt 제공

**인사이트**
- source language의 Roman script 문맥을 넣어 transcription을 유도하는 역할입니다.
- 하지만 결과에서 보듯 prompt만으로는 zero-shot transliteration에 한계가 있습니다.

---

### Table 5: Cyrillization prompts
- Italian, Greek, Hindi, Korean, Japanese에 대해 Cyrillic prompt 제공

**인사이트**
- Cyrillization은 prompt로도 가능하지만, 안정성은 제한적입니다.
- 이 때문에 activation steering의 필요성이 부각됩니다.

---

### 부록의 추가 테이블들(Table 6~10)
- 본문 Figure와 유사한 수치 결과를 더 상세히 제공
- 모델 크기별/언어별/방법별 성능 차이를 확인할 수 있음

**인사이트**
- 전체적으로 결론은 일관됨:
  1. script는 선형적으로 표현된다
  2. steering으로 제어 가능하다
  3. zero-shot / few-shot / one-shot으로도 어느 정도 가능하다
  4. transliteration은 phonetic faithfulness를 보이는 경향이 있다

---

## 5) 전체 결론
이 논문의 핵심은 다음입니다.

- Whisper 같은 speech foundation model 내부에 **script 정보가 선형적으로 표현**되어 있다.
- 이를 활용하면 **prompting보다 직접적인 방식으로 output script를 제어**할 수 있다.
- 특히:
  - **script confusion 완화**
  - **zero-shot transliteration**
  - **one-shot 벡터 추출**
  - **cross-lingual generalization**
  이 가능하다.
- 그리고 결과는 단순한 문자 변환보다 **발음 중심의 transliteration**에 더 가깝다.

---


## 1) Diagrams / Conceptual Figures

### Figure 1: Core idea
- The paper shows that **script information is linearly encoded** in the activation space of speech foundation models like Whisper.
- By adding a **script vector** to decoder activations at inference time, the output script can be changed.
- Example:
  - Italian *buongiorno* is normally output in Latin script,
  - but after adding the script vector, it becomes **Cyrillic** (буонджорно).

**Insight**
- Script control is not done by post-processing or explicit rules, but by **directly intervening in the model’s internal representation**.
- This motivates **zero-shot-like script steering**.

---

### Figure 2: How script vectors are extracted
The method has three steps:

1. **Collect**
   - Gather decoder activations for source and target scripts
2. **Isolate**
   - Subtract target mean activations from source mean activations
   - `r_l = v_SRC_l - v_TRG_l`
3. **Add**
   - Add the vector to activations during inference to steer output script

**Insight**
- This is an **inference-time intervention**, not finetuning.
- The key claim is that script behaves like a **linear direction** in representation space.

---

## 2) Figure Results

### Figure 3: Mitigating script confusion
- Methods compared:
  - no-prompt
  - prompt
  - steer (ours)
  - one-shot
- Languages:
  - Serbian: Latin/Cyrillic
  - Chinese: Simplified/Traditional
- Findings:
  - Prompting is limited in smaller models
  - Steering often changes the script more effectively
  - Gains are especially strong for the **less frequent script** in a language
  - Larger models make prompting more effective, but steering remains competitive

**Insight**
- Script confusion is especially problematic for **low-resource or less frequent scripts**.
- Activation steering is effective even for smaller models.
- Internal intervention can be more direct than prompting.

---

### Figure 4: Romanization generalization across languages
- A romanization vector learned from Serbian is applied zero-shot to other languages.
- Languages:
  - Russian, Hindi, Greek, Japanese, Korean
- Findings:
  - Zero-shot performance is already strong for Hindi and Russian
  - Pseudo-label adaptation improves results further across all languages

**Insight**
- Romanization directions appear to be **shared across languages** to a significant extent.
- Script vectors are not language-specific only; they can be **cross-lingually reused**.
- Pseudo-labeling helps adapt the vector to the target language.

---

### Figure 5: Cyrillization across languages
- Latin-to-Cyrillic steering was tested.
- Findings:
  - Generally harder than romanization
  - **Italian** performs best, reaching 43%

**Insight**
- Even unconventional language-script pairings can be induced.
- Cyrillization is more difficult and more sensitive to the training-data distribution.
- Languages phonetically closer to English/Latin may be easier.

---

### Figure 6: Script probing across decoder layers
- A linear probe shows script separability at every decoder layer.
- Findings:
  - Script is separable in **all layers**
  - Script information is visible even early in decoding

**Insight**
- Script is not stored only in the final layers; it is distributed throughout the decoder.
- This supports steering **all layers**, not just one.
- It also suggests early-layer probing could enable more selective control in practice.

---

## 3) Table Results

### Table 1: Romanization sample predictions
- Compared against deterministic romanization tools.
- Findings:
  - Steering often produces outputs that are **closer to pronunciation**
  - Many bolded portions show where deterministic transliteration is orthographically correct but phonetically less faithful

**Insight**
- The method tends to produce **sound-based transliteration**, not just character mapping.

---

### Table 2: Cyrillization sample predictions
- Italian shows the strongest performance.
- Other languages also show partially sensible Cyrillic transliterations.

**Insight**
- The model contains enough cross-script phonetic alignment to handle novel transliteration directions.

---

### Table 6: Script confusion across Whisper models
- Compared no-prompt, prompt, and steer across model sizes.
- Findings:
  - Smaller models suffer more from script confusion
  - Prompting helps, but not always enough
  - Steering is more stable and often better

**Insight**
- Steering provides a more reliable control signal across model sizes.
- The benefit is especially clear for the less frequent script variants.

---

### Table 7: One-shot script steering
- Only one high-quality example pair is enough to extract a usable script vector.
- Findings:
  - One-shot steering already performs well
  - Strong improvements appear for Serbian Cyrillic and Traditional Chinese

**Insight**
- Script vectors are highly **sample-efficient** to learn.
- This is important for practical, low-resource settings.

---

### Table 8: Romanization / Cyrillization performance
- Romanization zero-shot is fairly strong for Hindi and Russian.
- Pseudo-labeling usually improves results.
- Cyrillization is harder, but Italian is relatively strong.

**Insight**
- Romanization has strong cross-language similarity.
- Pseudo-label adaptation improves language-specific performance.
- Cyrillization is more challenging, but still promising.

---

### Table 9: Dataset statistics
- Shows train/validation/test sizes for FLEURS across languages.

**Insight**
- The evaluation covers multiple languages and script pairs rather than a single narrow case.

---

### Table 10: Best sigma values
- Reports the best steering strengths for transliteration.
- Optimal sigma differs by language.

**Insight**
- Steering strength matters a lot.
- Too weak has little effect; too strong may distort the transcription.

---

## 4) Appendix

### Table 3: Script confusion prompts
- Prompts for Serbian and Chinese scripts.
- All prompts mean “This is a sentence in X.”

**Insight**
- Prompting provides script context, but it is less direct than activation steering.

---

### Table 4: Romanization prompts
- Prompts for Russian, Greek, Hindi, Korean, and Japanese.

**Insight**
- These prompts bias the model toward romanized output, but only to a limited extent.

---

### Table 5: Cyrillization prompts
- Prompts for Italian, Greek, Hindi, Korean, and Japanese in Cyrillic script.

**Insight**
- Prompt-based cyrillization is possible, but limited in robustness.

---

## 5) Overall takeaway
The paper’s main message is:

- Script information is **linearly represented** in Whisper-like speech foundation models.
- This enables **direct script control** via activation steering.
- The method works for:
  - script confusion mitigation
  - zero-shot transliteration
  - one-shot vector extraction
  - cross-lingual generalization
- The generated transliterations are often more **phonetic** than deterministic rule-based outputs.

---



<br/>
# refer format:



## 1) BibTeX

```bibtex
@misc{shim2026linear,
  title={Linear Script Representations in Speech Foundation Models Enable Zero-Shot Transliteration},
  author={Shim, Ryan Soh-Eun and Choi, Kwanghee and Chang, Kalvin and Hsu, Ming-Hao and Eichin, Florian and Wu, Zhizheng and Suhr, Alane and Hedderich, Michael A. and Harwath, David and Mortensen, David R. and Plank, Barbara},
  year={2026},
  eprint={2601.02906},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2601.02906}
}
```

## 2) 시카고 스타일(줄글 형태)

Shim, Ryan Soh-Eun, Kwanghee Choi, Kalvin Chang, Ming-Hao Hsu, Florian Eichin, Zhizheng Wu, Alane Suhr, Michael A. Hedderich, David Harwath, David R. Mortensen, and Barbara Plank. “Linear Script Representations in Speech Foundation Models Enable Zero-Shot Transliteration.” arXiv preprint arXiv:2601.02906 (2026). https://arxiv.org/abs/2601.02906.




