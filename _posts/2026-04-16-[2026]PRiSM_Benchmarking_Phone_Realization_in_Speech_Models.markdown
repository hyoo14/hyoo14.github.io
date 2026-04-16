---
layout: post
title:  "[2026]PRiSM: Benchmarking Phone Realization in Speech Models"
date:   2026-04-16 19:08:25 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문은 PRiSM이라는 오픈소스 벤치마크를 제안해, 음성모델의 phone recognition을 **내재적 평가(PFER)**와 **외재적 평가(전사/표현 probe)**로 함께 측정합니다.


짧은 요약(Abstract) :



이 논문은 **PRiSM**이라는 새로운 오픈소스 벤치마크를 제안합니다. PRiSM은 음성 모델이 **phone recognition(음소/음성 단위 인식)** 을 얼마나 잘하는지 평가하기 위한 기준입니다. 기존 평가는 주로 **전사 정확도** 같은 표면적인 성능만 보았지만, PRiSM은 이를 넘어서 모델이 실제로 **음성학적 특성(phonetic ability)** 을 얼마나 잘 이해하고 있는지, 그리고 그 표현이 **다운스트림 작업**에 얼마나 유용한지까지 함께 평가합니다.

PRiSM은 두 가지 관점에서 평가합니다.

1. **Intrinsic evaluation**  
   - 모델이 예측한 음성 전사(transcription)가 정답과 얼마나 다른지 측정합니다.
   - 즉, phone recognition 자체의 핵심 능력을 봅니다.

2. **Extrinsic evaluation**  
   - 모델의 전사 결과나 내부 표현(representation)을 이용해
     - 병리적 발화 평가,
     - 교육용 발음 평가,
     - 다국어/방언 식별
     같은 실제 작업에 얼마나 도움이 되는지 봅니다.

논문 결과에 따르면:

- **학습 과정에서 다양한 언어를 많이 접한 모델**이 더 좋은 성능을 보였습니다.
- 특히 **encoder-CTC 계열 모델**이 가장 안정적이었습니다.
- 반면, **대형 오디오 언어모델(LALMs)** 은 전문적인 phone recognition 모델보다 여전히 성능이 떨어졌습니다.

즉, PRiSM은 음성 모델이 단순히 글자처럼 잘 전사하는지뿐 아니라, **음성학적으로 얼마나 풍부하고 일반화 가능한 정보를 담고 있는지**를 평가하는 기준을 제시하는 논문입니다.

---



This paper introduces **PRiSM**, the first open-source benchmark for evaluating **phone recognition (PR)** in speech models. Unlike previous evaluations that focus mainly on surface-level transcription accuracy, PRiSM is designed to uncover hidden weaknesses in phonetic perception by combining **intrinsic** and **extrinsic** evaluation.

- **Intrinsic evaluation** measures how accurately a model transcribes phones.
- **Extrinsic evaluation** tests whether the model’s transcriptions and internal representations are useful for downstream tasks in **clinical, educational, and multilingual** settings.

The authors find that:

- **Training with diverse language exposure** is important for strong PR performance.
- **Encoder-CTC models** are the most stable across tasks.
- **Specialized PR models outperform Large Audio Language Models (LALMs)**.

Overall, PRiSM provides a reproducible benchmark to move the field toward multilingual speech models with stronger and more robust phonetic abilities.

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




# 1. 메서드 개요

PRiSM은 단순히 “음소를 얼마나 잘 맞추는가”만 보지 않고,  
1) **직접적인 전사 성능(intrinsic)** 과  
2) **하위 태스크에서의 유용성(extrinsic)**  
을 함께 평가하는 벤치마크입니다.

즉, 어떤 음성 모델이 IPA 기반 phone transcription을 잘 만드는지 볼 뿐 아니라,  
그 모델이 만들어낸 **전사(transcription)** 와 **내부 표현(representation)** 이 실제로 다음과 같은 작업에 얼마나 도움이 되는지도 함께 봅니다.

- 병리적 발화 분석
- L2(비원어민) 발음 평가
- 억양/방언/L1 추정
- 다국어 언어 식별
- phone inventory induction(언어의 음소 체계 추정)

이 구조가 PRiSM의 핵심 메서드입니다.

---

# 2. 평가 프레임워크

## 2.1 Intrinsic evaluation: 코어 PR 능력 측정
Intrinsic 평가는 모델이 생성한 **음성→IPA 전사**를 정답과 직접 비교합니다.

### 사용 지표
논문은 **PFER(Phonetic Feature Error Rate)** 를 사용합니다.

- 일반적인 PER(Phone Error Rate)은 phone을 하나의 token으로 보고 편집거리 계산을 합니다.
- PRiSM의 PFER는 더 정교하게,  
  phone 자체의 틀림을 단순 토큰 오차가 아니라  
  **조음적 특징(articulatory features)** 수준에서 측정합니다.

예를 들어:
- voicing
- rounding
- place of articulation
- manner of articulation

같은 특징 벡터를 비교해서 편집거리를 계산합니다.

### 왜 PFER인가?
논문은 PR에서 단순 PER보다 PFER가 더 적절하다고 봅니다. 이유는:
- phone은 lexical unit보다 더 낮은 수준의 음성 단위라서 오차가 더 noisy함
- 완전히 다른 phone처럼 보여도 조음 특징상 가까울 수 있음
- 반대로 표면적으로 비슷해도 중요한 조음 특징이 틀릴 수 있음

즉, **“맞았는가/틀렸는가”보다 “어떤 방식으로 틀렸는가”** 를 더 잘 반영합니다.

---

## 2.2 Extrinsic evaluation: 다운스트림 유용성 측정
Extrinsic 평가는 모델 출력이 실제 downstream task에 얼마나 유용한지를 봅니다.

여기서 두 가지 입력을 따로 평가합니다.

### (1) Transcript Probe (TP)
- 모델이 생성한 **phonetic transcription** 을 입력으로 사용
- 입력 텍스트를 기반으로 한 lightweight probe가 downstream task를 수행

구조:
- PR 모델 → IPA 전사 생성
- 전사 → **text-based bi-GRU probe**
- 결과를 분류/회귀/지리 예측 등에 사용

### (2) Representation Probe (RP)
- 모델의 **hidden representation** 을 입력으로 사용
- 마지막 레이어 representation을 time pooling + attention으로 모은 뒤 MLP로 예측

구조:
- PR 모델 → hidden representation 추출
- 마지막 레이어 특징 → attention pooling
- 이후 MLP → downstream prediction

### 왜 둘 다 보나?
논문은 모델이 phonetic 정보를 활용하는 방식이 두 가지라고 설명합니다.

1. **명시적 전사(explicit transcription)**  
2. **내부 latent representation**

전사만 좋고 representation이 나쁠 수도 있고,  
반대로 전사는 별로여도 representation이 downstream task에 더 좋을 수도 있습니다.  
그래서 둘을 모두 보는 것이 중요하다고 주장합니다.

---

# 3. 평가 태스크 구성

PRiSM은 평가 태스크를 크게 4개 범주로 나눕니다.

## 3.1 Intrinsic: Core Capability
### A. Seen language variation
기존에 학습 중 본 언어이지만, 변이된 발화들에 대한 평가:
- TIMIT
- L2-ARCTIC perceived
- Speech Accent Archive

목적:
- 모델이 익숙한 언어에서도 방언/억양/비원어민 발화에 덜 의존하는지 확인

### B. Unseen languages
학습에서 본 적 없는 언어:
- DoReCo
- VoxAngeles
- Tusom2021

목적:
- 언어-무관적 phonetic generalization 능력 평가

---

## 3.2 Extrinsic: Downstream Utility
### A. Pathological speech
- EasyCall
- UASpeech
- UltraSuite

태스크:
- Dysarthria intelligibility prediction
- Child speech disorder detection

### B. L2 speech
- EdAcc
- CMU Arctic
- L2-ARCTIC
- Speechocean762

태스크:
- L1 classification
- L2 pronunciation assessment

### C. Multilingual speech
- FLEURS-24
- Vaani
- DoReCo

태스크:
- Language ID
- Geolocation
- Phone inventory induction

---

# 4. 평가 대상 모델들

논문은 다양한 PR 시스템을 비교합니다.  
크게 보면 다음 계열로 나뉩니다.

---

## 4.1 Wav2Vec2Phs 계열
### 포함 모델
- MultiIPA
- W2V2P-LV60
- W2V2P-XLSR53

### 특징
- Wav2Vec2 기반 SSL speech model을 PR용으로 fine-tune한 모델들
- 서로 다른 pretraining coverage와 fine-tuning 데이터 차이를 가짐

### 핵심 포인트
이 계열은 보통:
- **자기지도 사전학습(SSL)** 을 활용하고
- 여러 언어의 음성 정보를 학습한 뒤
- IPA 전사로 fine-tuning합니다.

논문에서는 pretraining에 포함된 언어 범위와 supervised fine-tuning 데이터 범위가 성능에 큰 영향을 준다고 봅니다.

---

## 4.2 ZIPA 계열
### 포함 모델
- ZIPA-CTC
- ZIPA-CTC-NS

### 특징
- multilingual data로 scratch부터 학습한 encoder-CTC 모델
- ZIPA-CTC-NS는 pseudo-labeled data를 추가 학습
- encoder-only 구조

### 핵심 포인트
ZIPA-CTC-NS는 특히:
- 더 넓은 다국어 데이터
- pseudo-label 확장
을 통해 unseen language에서 더 안정적인 성능을 보입니다.

즉, **언어 다양성 + noisy pseudo-label** 이 중요한 역할을 한다는 결론을 뒷받침합니다.

---

## 4.3 POWSM 계열
### 포함 모델
- POWSM
- POWSM-CTC

### 특징
- POWSM는 attention-based encoder-decoder(AED)
- 같은 데이터로 학습한 encoder-CTC 변형이 POWSM-CTC
- PRiSM 논문 저자들이 추가로 학습한 비교 모델

### 핵심 포인트
- POWSM는 decoder 기반 language modeling 성격을 어느 정도 가짐
- POWSM-CTC는 encoder-only CTC 방식

논문에서는 **architecture 차이만으로도 phonetic generalization의 성격이 달라진다**고 봅니다.

---

## 4.4 LALMs (Large Audio Language Models)
### 포함 모델
- Gemini 2.5 Flash
- Qwen3-Omni-Instruct

### 특징
- 범용 멀티모달 오디오-텍스트 모델
- 내부 representation 접근이 어렵기 때문에 zero-shot prompting 방식으로 probe

### 핵심 포인트
논문 결론:
- LALMs는 일부 task에서 경쟁력 있지만
- 전반적으로 specialized PR model보다 약함
- 특히 fine-grained phonetic perception이 부족

즉, “범용 멀티모달 모델이 음소 인식까지 잘하느냐”에 대해 아직 한계가 있음을 보입니다.

---

# 5. 아키텍처 차이의 의미

논문은 단순 성능 비교뿐 아니라 **어떤 아키텍처가 왜 그런 성능을 내는가**도 분석합니다.

## 5.1 Encoder-only CTC vs AED
### Encoder-CTC
- 입력 음성을 encoder로 직접 처리
- CTC loss로 음성-문자 정렬을 학습
- temporal alignment와 acoustic fidelity가 비교적 강함

### AED
- encoder + decoder
- decoder가 다음 토큰 예측을 수행
- phonotactics(음운론적 패턴) / language modeling에 더 의존할 가능성 있음

### 논문 관찰
- encoder-only CTC 모델이 새로운 도메인에서 더 안정적
- AED는 phonotactic pattern에 더 많이 의존할 수 있음
- 하지만 어떤 task에서는 phonotactics가 도움이 될 수도 있음

즉, **아키텍처에 따라 “소리 자체” vs “가능한 음운 패턴”을 얼마나 쓰는지 달라집니다.**

---

## 5.2 CR-CTC의 특성
ZIPA는 **Consistency Regularized CTC (CR-CTC)** 를 사용합니다.

논문은 이 loss가:
- acoustic signal만 보는 것이 아니라
- phonotactic regularity도 함께 학습하도록 bias를 줄 수 있다고 봅니다.

결과적으로 ZIPA는:
- unseen language에서 특정 상황에 강하지만
- transcription이 지나치게 정규화되는 경향도 보일 수 있습니다.

---

# 6. 특수한 실험 기법

## 6.1 Phone masking experiment
이 실험은 모델이 **phonotactics(음운론적 패턴)** 에 의존하는지,  
아니면 **acoustic signal** 자체에 의존하는지 보려는 분석입니다.

### 방법
- TIMIT의 time-aligned phone transcript를 사용
- 일정 비율 p%의 phone을 silence로 교체
- 변형된 speech를 PR 모델에 입력
- 남아 있는 phone만을 정답으로 하여 PFER 계산

### 해석
- acoustic 정보만 쓰는 모델이라면 masking 비율이 커져도 성능 곡선이 크게 변하지 않아야 함
- phonotactic pattern에 의존하는 모델은 masking이 심해질수록 성능 저하

### 결과
- Wav2Vec2Phs와 POWSM-CTC가 masking이 커져도 더 잘 버팀
- ZIPA는 phonotactics와 acoustic 둘 다에 의존하는 경향
- 이는 loss/architecture에 따라 모델의 phonetic bias가 다르다는 뜻

---

## 6.2 Zero-shot phone inventory induction
새 언어의 phone inventory를 추정하는 실험입니다.

### 방법
- DoReCo 기반 저자원 언어 사용
- 모델이 예측한 transcript를 수집
- PanPhon 기반 tokenization 후 set union으로 inventory 구성
- gold inventory와 비교하여 precision/recall/F1 계산

### 의미
이 실험은 단순히 “문장 전사”가 아니라  
**언어의 음소 체계 자체를 얼마나 잘 포착하는지**를 봅니다.

### 결론
- 다국어 학습이 중요
- 특히 supervised training에서 언어 수가 많을수록 recall이 좋아짐
- data 양만이 아니라 **언어 다양성(diversity)** 이 핵심

---

## 6.3 Geolocation for dialectal speech
힌디 방언 음성에서 발화자의 지역을 맞추는 실험입니다.

### 관찰
- transcript probe가 representation probe보다 더 잘 나옴

### 이유 가설
- transcript는 phone sequence order를 유지하는 RNN 기반 probe를 사용
- 방언별로 phone inventory는 비슷해도 **분포와 순서**가 다를 수 있음
- 따라서 전화 전사 기반 probe가 더 유리할 수 있음

---

## 6.4 LALMs의 phonetic perception 한계
LALM은 일부 태스크에서 괜찮지만:
- dialectal variation
- fine-grained accent discrimination
- geographic localization

에서는 매우 약합니다.

### 분석 결과
- 특정 지역/억양으로 mode collapse 발생
- higher-resource accent 쪽으로 편향
- thinking mode를 켜면 오히려 bias가 악화되기도 함

즉, LALM은 아직 **정교한 음성학적 지각**이 부족하다는 해석입니다.

---

# 7. 이 논문의 메서드가 중요한 이유

PRiSM의 메서드적 기여는 단순한 새 모델 제안이 아니라:

1. **PR 평가 표준화**
2. **전사와 representation을 함께 평가**
3. **seen language / unseen language를 분리**
4. **downstream utility를 실제로 검증**
5. **아키텍처와 학습 데이터의 영향을 분석**

하는 데 있습니다.

즉,  
“전화 전사 성능이 좋다 = 실제로 phonetic ability가 좋다”는 가정을 깨고,  
**무엇을 잘하는지 세분화해서 보자**는 것이 핵심입니다.

---

# 8. 한 줄 요약

PRiSM은 **IPA phone transcription 성능(PFER)** 과 **downstream task에서의 유용성(TP/RP)** 을 함께 평가하는 벤치마크이며,  
모델은 **Wav2Vec2Phs, ZIPA, POWSM, LALM** 등으로 나뉘고,  
**encoder-CTC, AED, CR-CTC, pseudo-labeling, multilingual pretraining** 같은 설계가 성능에 큰 영향을 줍니다.

---



---

## 1. Overview of the method

PRiSM is not just a benchmark for “how accurately a model transcribes phones.”  
Instead, it evaluates phone recognition systems in two complementary ways:

- **Intrinsic evaluation**: how accurately the model transcribes speech into IPA phones
- **Extrinsic evaluation**: how useful those transcriptions or hidden representations are for downstream tasks

So the method asks both:

1. **Does the model produce good phone transcriptions?**
2. **Are those outputs actually useful for clinical, educational, and multilingual speech tasks?**

This dual evaluation is the core methodological idea of PRiSM.

---

## 2. Evaluation framework

### 2.1 Intrinsic evaluation: core phone recognition ability
Intrinsic evaluation directly compares the model’s predicted phonetic transcriptions against gold labels.

#### Metric: PFER
The paper uses **Phonetic Feature Error Rate (PFER)**.

- Standard PER (Phone Error Rate) treats phones as discrete tokens.
- PFER instead computes edit distance over **articulatory feature vectors** such as:
  - voicing
  - rounding
  - place of articulation
  - manner of articulation

#### Why PFER?
The authors argue that PR errors are noisier than ASR errors because phones are lower-level articulatory units.  
PFER is therefore more sensitive to *how* a phone is wrong, not just whether it is wrong.

---

### 2.2 Extrinsic evaluation: downstream utility
The extrinsic part tests whether the model’s output is useful in real applications.

Two types of probes are used:

#### (1) Transcript Probe (TP)
- Input: predicted phonetic transcriptions
- Probe model: a text-based **bi-GRU**
- Purpose: measure how useful explicit phone strings are

#### (2) Representation Probe (RP)
- Input: hidden representations from the PR model
- Probe model: attention pooling + MLP
- Purpose: measure whether latent representations retain phonetic information

#### Why both?
Because phonetic information can be carried in two channels:

1. **explicit transcriptions**
2. **latent representations**

A model might be weak in one and strong in the other, so the paper evaluates both.

---

## 3. Task setup

PRiSM evaluates systems on four groups of tasks.

### 3.1 Intrinsic: core capability
#### A. Variation of seen language
Datasets:
- TIMIT
- L2-ARCTIC perceived
- Speech Accent Archive

Purpose:
- test whether models remain robust on accented, dialectal, or non-native speech in languages they have already seen

#### B. Unseen languages
Datasets:
- DoReCo
- VoxAngeles
- Tusom2021

Purpose:
- test language-agnostic phonetic generalization

---

### 3.2 Extrinsic: downstream utility
#### A. Pathological speech
Datasets:
- EasyCall
- UASpeech
- UltraSuite

Tasks:
- dysarthria intelligibility prediction
- child speech disorder detection

#### B. L2 speech
Datasets:
- EdAcc
- CMU Arctic
- L2-ARCTIC
- Speechocean762

Tasks:
- L1 classification
- L2 pronunciation assessment

#### C. Multilingual speech
Datasets:
- FLEURS-24
- Vaani
- DoReCo

Tasks:
- language ID
- geolocation
- phone inventory induction

---

## 4. Model families compared

The paper benchmarks several types of PR systems.

### 4.1 Wav2Vec2Phs family
Models:
- MultiIPA
- W2V2P-LV60
- W2V2P-XLSR53

#### Characteristics
- Fine-tuned variants of Wav2Vec2 SSL models
- Differ in pretraining coverage and fine-tuning datasets

#### Key point
The paper finds that broader language exposure during training is important.

---

### 4.2 ZIPA family
Models:
- ZIPA-CTC
- ZIPA-CTC-NS

#### Characteristics
- Encoder-only CTC models trained from scratch on multilingual data
- ZIPA-CTC-NS is further trained on pseudo-labeled data

#### Key point
More multilingual coverage and pseudo-labels improve robustness, especially for unseen languages.

---

### 4.3 POWSM family
Models:
- POWSM
- POWSM-CTC

#### Characteristics
- POWSM is an encoder-decoder model with attention
- POWSM-CTC is an encoder-CTC variant trained on the same data

#### Key point
Architecture matters: encoder-only CTC and AED models behave differently in how they use phonotactics vs acoustic cues.

---

### 4.4 LALMs
Models:
- Gemini 2.5 Flash
- Qwen3-Omni-Instruct

#### Characteristics
- Large audio-language models
- Probed via zero-shot prompting because hidden representations are hard to access

#### Key point
They are competitive in some cases but generally underperform specialized PR models, especially on fine-grained phonetic perception.

---

## 5. Architectural interpretation

### Encoder-only CTC vs AED
#### CTC
- Directly maps acoustic frames to output phones
- Encourages stronger temporal alignment and acoustic fidelity

#### AED
- Uses a decoder to predict the next token
- Can rely more on phonotactics and implicit language modeling

#### Observation
- Encoder-CTC models are generally more stable
- AED may benefit from learned phonological patterns in some tasks

---

## 6. Special experimental techniques

### 6.1 Phone masking experiment
This experiment tests whether a model relies on:

- **acoustic signal**, or
- **phonotactic patterns**

#### Procedure
- Use time-aligned TIMIT transcripts
- Replace a percentage of phones with silence
- Re-run PR and compute PFER against the remaining phones

#### Interpretation
- If a model depends only on acoustics, masking should not strongly change the curve
- If it depends on phonotactics, performance should degrade as masking increases

---

### 6.2 Zero-shot phone inventory induction
This task checks whether a model can infer the set of phones used in a new language.

#### Procedure
- Use DoReCo languages unseen in training
- Collect predicted transcriptions
- Tokenize phones using PanPhon features
- Union detected phones into an inventory
- Compare with gold inventory using set-based metrics

#### Meaning
This is not just transcription; it measures whether the model captures the phonological system of a language.

---

### 6.3 Geolocation for dialectal speech
This task predicts geographic origin from dialectal speech.

#### Observation
Transcript probes can outperform representation probes.

#### Hypothesis
Sequence order matters:
- dialects may share similar phone inventories
- but differ in distributional patterns and phone sequences

Thus, transcript-based sequence models may capture dialect distinctions better.

---

### 6.4 LALMs lack fine-grained phonetic perception
The paper shows that LALMs often fail on:
- dialectal variation
- accent discrimination
- geographic localization

They show:
- mode collapse
- bias toward high-resource accents
- limited sensitivity to subtle phonetic cues

So the paper concludes that LALMs still lack robust phonetic perception.

---

## 7. Why this method matters

PRiSM’s methodological contribution is not a new speech model, but a **standardized benchmark** that:

1. evaluates both transcription and downstream utility,
2. separates seen-language variation from unseen-language generalization,
3. compares explicit transcripts with hidden representations,
4. tests different architectures fairly,
5. analyzes the role of multilingual training and model design.

In other words, it challenges the assumption that good transcription accuracy alone fully captures phonetic competence.

---

## 8. One-sentence summary

PRiSM is a benchmark that evaluates phone recognition systems using both **intrinsic transcription accuracy (PFER)** and **extrinsic downstream utility (TP/RP)**, comparing model families such as **Wav2Vec2Phs, ZIPA, POWSM, and LALMs**, while showing that **multilingual training, encoder-CTC architecture, and phonetic feature-aware evaluation** are crucial for robust performance.

---




<br/>
# Results



## 1. 논문의 결과 요약

이 논문은 음성모델의 **phone recognition(PR, 음소/음성단위 인식)** 성능을 단순 전사 정확도만으로 보지 않고,  
1) **내재적 평가(intrinsic)**: 실제 전사 성능  
2) **외재적 평가(extrinsic)**: 그 전사/표현이 실제 다운스트림 작업에 얼마나 유용한지  
두 축으로 평가합니다.

핵심 결론은 다음과 같습니다.

- **다양한 언어를 폭넓게 학습한 모델이 전반적으로 유리**합니다.
- **encoder-CTC 계열 모델이 가장 안정적**입니다.
- **특화된 PR 모델이 여전히 LALM(Large Audio Language Model)보다 우수**합니다.
- 단순히 전사 오류율만 낮다고 해서 실제 downstream 유용성이 높은 것은 아닙니다.
- 모델마다 **전사(Transcript Probe, TP)**와 **표현(Representation Probe, RP)**에서 성향이 다르게 나타납니다.

---

## 2. 경쟁 모델(비교 대상) 정리

논문에서 비교한 모델들은 크게 아래와 같습니다.

### 2.1 Wav2Vec2 기반 PR 모델
- **MultiIPA**
- **W2V2P-LV60**
- **W2V2P-XLSR53**

특징:
- 사전학습된 wav2vec2/XLSR 계열을 PR용으로 파인튜닝한 모델들입니다.
- 언어 커버리지와 파인튜닝 데이터가 다릅니다.
- 비교적 전통적인 강력한 베이스라인입니다.

### 2.2 ZIPA 계열
- **ZIPA-CTC**
- **ZIPA-CTC-NS**

특징:
- multilingual 데이터로 학습한 encoder-CTC 모델입니다.
- ZIPA-CTC-NS는 pseudo-labeled data를 더 활용해 확장된 버전입니다.
- 논문에서 **가장 강하고 안정적인 계열 중 하나**로 나타납니다.

### 2.3 POWSM 계열
- **POWSM**
- **POWSM-CTC** (논문 저자 구현)

특징:
- POWSM은 attention-based encoder-decoder(AED) 모델
- POWSM-CTC는 같은 데이터로 학습한 encoder-CTC 변형
- 아키텍처 차이가 성능에 미치는 영향을 분석하기 위한 비교군입니다.

### 2.4 LALMs (Large Audio Language Models)
- **Gemini 2.5 Flash**
- **Qwen3-Omni-Instruct**

특징:
- 범용 오디오-언어 모델
- PR 전용 모델보다 강할 것 같지만, 실제로는 PR/phonetic task에서 약점이 드러남
- 특히 zero-shot 설정에서 취약

### 2.5 기타 표현 평가용 베이스라인
- **WavLM**
- **Whisper**

특징:
- transcription 자체보다 hidden representation probing에 강한지 보기 위한 비교군입니다.

---

## 3. 테스트 데이터셋과 평가 과제

이 논문은 데이터셋을 **intrinsic**과 **extrinsic**으로 나누어 평가합니다.

---

### 3.1 Intrinsic 평가용 데이터

#### A. Seen language variation
이미 학습에서 어느 정도 접한 영어 기반 변이/억양/비원어민 발화 등을 평가합니다.

- **PR-tmt**: TIMIT
  - 영어 지역 변이
- **PR-arc**: L2-ARCTIC perceived set
  - 비원어민 영어
- **PR-saa**: Speech Accent Archive
  - 다양한 L1 배경 화자의 영어 발화

#### B. Unseen languages
학습 중 보지 못한 언어에서의 일반화 능력을 평가합니다.

- **PR-drc**: DoReCo (45개 언어)
- **PR-vox**: VoxAngeles (95개 언어)
- **PR-tsm**: Tusom2021 (Tusom)

이 부분은 “새 언어에서 phone recognition이 얼마나 가능한가?”를 보는 핵심 테스트입니다.

---

### 3.2 Extrinsic 평가용 데이터

#### A. Pathological speech
- **DYS-ez**: EasyCall (이탈리아어 dysarthria)
- **DYS-ua**: UASpeech (영어 dysarthria)
- **CSD-us**: UltraSuite (아동 speech sound disorder)

이들은 발화의 병리적/비정형적 특성을 얼마나 잘 반영하는지 봅니다.

#### B. L2 speech
- **L1-eda**: EdAcc
- **L1-arc**: CMU ARCTIC + L2-ARCTIC
- **L2-so**: Speechocean762

이 과제들은 억양, 발음 정확도, L1 배경 추론 등을 봅니다.

#### C. Multilingual speech
- **LID-fl**: FLEURS-24
- **GEO-v**: Vaani Hindi-belt 지리 위치 추정
- **PI-drc**: DoReCo phone inventory induction

다국어/방언/언어식별 관련 능력을 평가합니다.

---

## 4. 사용된 메트릭

### 4.1 Intrinsic: PFER
논문은 일반적인 PER(Phone Error Rate) 대신 **PFER(Phonetic Feature Error Rate)**를 사용합니다.

- 단순히 phone token을 틀렸는지를 보는 것이 아니라,
- 각 phone의 **조음적 특성(feature)** 차이를 edit distance로 계산합니다.
- 예: voicing, roundness 같은 속성 차이를 반영

즉, “완전히 다른 phone”과 “비슷한 조음 특징을 가진 phone”을 구분해 더 정교한 평가를 합니다.

---

### 4.2 Extrinsic: Transcript Probe vs Representation Probe

#### Transcript Probe (TP)
- 모델이 생성한 **전사 결과**를 입력으로 사용
- bi-GRU 기반 text probe로 downstream task를 예측

#### Representation Probe (RP)
- 모델의 **마지막 hidden representation**을 입력으로 사용
- attention pooling + MLP로 downstream task를 예측

이 둘의 차이는 매우 중요합니다.

- TP: 명시적인 phonetic content 활용
- RP: 내부 표현에 담긴 더 풍부한 phonetic/acoustic 정보 활용 가능

---

## 5. 주요 결과 정리

---

## 5.1 Intrinsic 결과: PFER

### 5.1.1 Seen language variation

표 3을 보면 다음 경향이 나타납니다.

- **W2V2P-XLSR53**, **ZIPA-CTC-NS**, **ZIPA-CTC**가 전반적으로 강함
- **POWSM**은 일부 조건에서 약함, 특히 Speech Accent Archive에서 매우 불안정
- **LALMs(Gemini, Qwen)**는 전반적으로 PR 전용 모델보다 약함

#### 대표적으로:
- **PR-tmt**
  - W2V2P-LV60: 13.2
  - W2V2P-XLSR53: 13.5
  - ZIPA-CTC / ZIPA-CTC-NS: 13.1
- **PR-arc**
  - W2V2P-XLSR53: 9.9
  - ZIPA-CTC / ZIPA-CTC-NS: 9.7
- **PR-saa**
  - W2V2P-XLSR53: 9.0
  - ZIPA-CTC: 9.0
  - ZIPA-CTC-NS: 8.9

즉, seen language variation에서는 **멀티링구얼 학습 + CTC 구조**가 강합니다.

---

### 5.1.2 Unseen languages

unseen language에서는 모델 간 격차가 더 분명해집니다.

- **ZIPA-CTC-NS**: 평균 19.0으로 매우 좋음
- **W2V2P-LV60**: 19.5
- **W2V2P-XLSR53**: 21.0
- **POWSM**: 18.7
- **LALMs**: 매우 나쁨
  - Gemini 평균 53.8
  - Qwen 평균 105.4

여기서 중요한 점은:
- unseen language에서 **LALM은 반복 출력, degeneracy** 같은 문제가 나타남
- PR 전용 모델은 여전히 훨씬 안정적

---

### 5.1.3 해석
논문은 다음을 강조합니다.

- **seen language variation**은 기존에 본 패턴에 맞춰 transcribe하는 것이 유리
- **unseen language**는 다국어 학습과 phonological pattern 학습이 중요
- encoder-CTC 모델이 일반적으로 가장 안정적
- LALM은 phone recognition에서 아직 한계가 큼

---

## 5.2 Extrinsic 결과: Downstream utility

외재적 평가는 TP와 RP로 나뉩니다.

---

### 5.2.1 Pathological speech

#### TP 결과
- ZIPA 계열, W2V2P 계열이 전반적으로 강함
- GPT형 LALM보다는 PR 전용 모델이 낫거나 비슷

#### RP 결과
- **Whisper**가 매우 강함
- 특히 표현 자체가 병리적 speech의 acoustic-phonetic 정보를 잘 담는 것으로 보임

#### 해석
병리적 speech는 발음 오류, timbre, prosody 등 **세밀한 acoustic cue**가 중요하므로,
단순 전사보다 representation이 더 유리한 경우가 많습니다.

---

### 5.2.2 L2 speech

- TP와 RP 간 차이가 상황에 따라 달라짐
- Wav2Vec2Phs 계열은 TP에서 강하지만 RP 향상은 제한적
- ZIPA 계열은 TP에서는 약간 떨어지지만 RP에서 더 나을 수 있음

즉, L2 speech에서는 “전사 자체”보다 “표현에 담긴 정보”가 유용할 때가 있습니다.

---

### 5.2.3 Multilingual speech

- TP는 multilingual task에서 강한 편
- 특히 **LID, phone inventory induction**에서는 multilingual training의 이점이 큼
- GEO-v 같은 dialect geolocation에서는 TP가 RP보다 훨씬 좋음

논문은 지리 방언 판별에는 phone sequence order가 중요하다고 설명합니다.

---

## 6. 중요한 비교 포인트

---

### 6.1 Language exposure matters
- 학습 중 **본 언어/비슷한 언어**에 강함
- unseen language는 다양하고 폭넓은 언어 exposure가 중요

즉, 언어 커버리지가 넓을수록 좋습니다.

---

### 6.2 Architecture matters
- **Encoder-only + CTC**가 가장 안정적
- AED는 특정 조건에서 phonotactics에 과도하게 의존하거나 search instability가 생길 수 있음

특히 Figure 2에서 phone masking을 늘렸을 때:
- acoustic signal에 더 의존하는 모델은 성능이 비교적 유지
- phonotactics에 더 의존하는 모델은 PFER가 악화

논문은 **Wav2Vec2Phs가 더 acoustic-driven**,  
**ZIPA는 phonotactics와 acoustics를 같이 학습**하는 경향이 있다고 해석합니다.

---

### 6.3 Data diversity matters
- 단순 데이터 양보다 **언어 다양성**이 중요
- ZIPA-CTC-NS는 pseudo-labeled 데이터까지 활용해 더 정확한 unseen language phone prediction을 보임

---

### 6.4 LALMs still lag behind
- Gemini, Qwen 모두 PR 및 dialectal/phonetic discrimination에서 제한적
- 특히 zero-shot geolocation과 accent classification에서 편향과 mode collapse 발생
- 영어/고자원 방언에 끌리는 경향

---

## 7. 한 줄 결론

이 논문의 핵심 결과는 다음과 같이 요약됩니다.

> **PRiSM benchmark에서 다국어 데이터와 encoder-CTC 구조를 가진 특화 PR 모델이 가장 안정적이며, LALM은 아직 phone recognition과 세밀한 phonetic discrimination에서 뒤처진다. 또한 transcription만 보는 intrinsic 평가와 실제 downstream 유용성을 보는 extrinsic 평가를 함께 해야 모델을 제대로 평가할 수 있다.**

---



---

## 1. High-level result summary

The paper evaluates phone recognition (PR) systems along two dimensions:

1. **Intrinsic evaluation**: direct transcription accuracy
2. **Extrinsic evaluation**: how useful the transcriptions or hidden representations are for downstream tasks

The main conclusions are:

- **Broad multilingual exposure during training is crucial**
- **Encoder-CTC models are the most stable**
- **Specialized PR models still outperform Large Audio Language Models (LALMs)**
- Good transcription accuracy does not always imply good downstream utility
- Models behave differently depending on whether the probe uses **transcripts** or **hidden representations**

---

## 2. Competing models

The paper compares several model families.

### 2.1 Wav2Vec2-based PR models
- **MultiIPA**
- **W2V2P-LV60**
- **W2V2P-XLSR53**

These are fine-tuned variants of wav2vec2/XLSR speech SSL models for PR.

---

### 2.2 ZIPA family
- **ZIPA-CTC**
- **ZIPA-CTC-NS**

These are multilingual encoder-CTC PR models.
ZIPA-CTC-NS additionally uses pseudo-labeled data and is generally one of the strongest systems.

---

### 2.3 POWSM family
- **POWSM**
- **POWSM-CTC** (authors’ variant)

POWSM is an attention-based encoder-decoder model, while POWSM-CTC is an encoder-CTC variant built for comparison.

---

### 2.4 LALMs
- **Gemini 2.5 Flash**
- **Qwen3-Omni-Instruct**

These are general-purpose audio-language models. They are competitive in some settings but generally lag behind specialized PR systems.

---

### 2.5 Additional baselines for representation probing
- **WavLM**
- **Whisper**

These are used mainly to test whether hidden representations encode useful phonetic information.

---

## 3. Test datasets and tasks

The benchmark separates evaluation into intrinsic and extrinsic tasks.

---

### 3.1 Intrinsic datasets

#### A. Seen-language variation
These test variation within languages or accents seen in training:

- **PR-tmt**: TIMIT
- **PR-arc**: L2-ARCTIC perceived set
- **PR-saa**: Speech Accent Archive

#### B. Unseen languages
These assess generalization to languages not seen in training:

- **PR-drc**: DoReCo (45 languages)
- **PR-vox**: VoxAngeles (95 languages)
- **PR-tsm**: Tusom2021

---

### 3.2 Extrinsic datasets

#### A. Pathological speech
- **DYS-ez**: EasyCall
- **DYS-ua**: UASpeech
- **CSD-us**: UltraSuite

#### B. L2 speech
- **L1-eda**: EdAcc
- **L1-arc**: CMU ARCTIC + L2-ARCTIC
- **L2-so**: Speechocean762

#### C. Multilingual speech
- **LID-fl**: FLEURS-24
- **GEO-v**: Vaani Hindi-belt geolocation
- **PI-drc**: DoReCo phone inventory induction

---

## 4. Metrics

### 4.1 Intrinsic metric: PFER
Instead of standard PER, the paper uses **PFER (Phonetic Feature Error Rate)**.

- It compares phones using **articulatory feature distance**
- This is more linguistically meaningful than token-level phone mismatch
- It captures whether two phones differ in features like voicing or rounding

---

### 4.2 Extrinsic probes

#### Transcript Probe (TP)
- Uses predicted phone transcriptions as input
- A text-based bi-GRU predicts the downstream task

#### Representation Probe (RP)
- Uses the model’s last hidden representations
- An attention pooling + MLP probe is applied

This distinction is important because transcript and representation may encode different kinds of phonetic information.

---

## 5. Main results

---

## 5.1 Intrinsic results: PFER

### 5.1.1 Seen-language variation
From Table 3:

- **W2V2P-XLSR53**
- **ZIPA-CTC**
- **ZIPA-CTC-NS**

perform strongly across seen-language variation tasks.

Example scores:
- **PR-tmt**
  - W2V2P-LV60: 13.2
  - W2V2P-XLSR53: 13.5
  - ZIPA-CTC / ZIPA-CTC-NS: 13.1
- **PR-arc**
  - W2V2P-XLSR53: 9.9
  - ZIPA-CTC / ZIPA-CTC-NS: 9.7
- **PR-saa**
  - ZIPA-CTC-NS: 8.9
  - W2V2P-XLSR53: 9.0

Overall, multilingual training and CTC-style decoding are strong for seen-language variation.

---

### 5.1.2 Unseen languages
Performance gaps become more visible:

- **ZIPA-CTC-NS**: average PFER 19.0
- **W2V2P-LV60**: 19.5
- **W2V2P-XLSR53**: 21.0
- **POWSM**: 18.7
- **LALMs** are much worse:
  - Gemini: 53.8
  - Qwen: 105.4

The paper notes that LALMs often produce degenerate or repetitive outputs on unseen languages.

---

### 5.1.3 Interpretation
The authors conclude that:

- Seen-language variation benefits from familiar phonetic patterns
- Unseen languages benefit from multilingual training and learned phonological regularities
- Encoder-CTC architectures are the most stable
- LALMs are still weak in PR

---

## 5.2 Extrinsic results: downstream utility

The paper evaluates both transcript probes and representation probes.

---

### 5.2.1 Pathological speech
- PR models such as ZIPA and W2V2P are competitive in TP
- **Whisper** is especially strong in RP
- Hidden representations seem to preserve acoustic-phonetic details useful for pathological speech tasks

This suggests that for dysarthria or child speech disorder detection, representations can be more informative than transcripts.

---

### 5.2.2 L2 speech
- Wav2Vec2-based PR models are often strong in TP
- ZIPA models may be less strong in TP but can do well in RP
- The utility of transcripts vs representations depends on the task

---

### 5.2.3 Multilingual speech
- TP works well for multilingual tasks
- Especially strong for:
  - language identification
  - phone inventory induction
  - geolocation from dialectal speech

For geolocation, transcript order information is particularly useful.

---

## 6. Key comparison takeaways

### 6.1 Language exposure matters
More diverse multilingual exposure leads to better generalization.

### 6.2 Architecture matters
Encoder-only CTC models are more stable than AED models.

### 6.3 Data diversity matters
Language diversity is as important as raw data volume.

### 6.4 LALMs still lag behind
Despite being large and general-purpose, LALMs remain less reliable than specialized PR systems on phonetic tasks.

---

## 7. One-sentence conclusion

> PRiSM shows that specialized multilingual encoder-CTC PR models remain the strongest and most stable approach for phone recognition, while LALMs are still limited in fine-grained phonetic perception; both intrinsic and extrinsic evaluations are necessary to fully assess PR systems.

---




<br/>
# 예제




## 1. 논문이 다루는 큰 문제: “폰(Phone) 재현(Realization)”을 어떻게 평가할 것인가?
이 논문은 음성 모델이 단순히 “문장을 잘 받아쓰는가”를 넘어,  
**실제 발화된 소리의 세부 음성학적 특징을 얼마나 정확히 포착하는지**를 평가합니다.

여기서 핵심은 **phone recognition(PR)** 입니다.  
PR은 입력 음성을 받아서 이를 **IPA(International Phonetic Alphabet)** 같은 음성 기호로 변환하는 작업입니다.

즉, 모델의 입력은 보통:
- 사람의 음성 오디오

모델의 출력은 보통:
- 해당 음성을 음성 단위로 바꾼 전사(예: IPA)

이 논문은 이런 PR 시스템을 다음 두 관점에서 평가합니다.

1. **Intrinsic evaluation (내재적 평가)**  
   → 모델이 생성한 전사가 정답 전사와 얼마나 비슷한가?

2. **Extrinsic evaluation (외재적 평가)**  
   → 모델의 전사 또는 내부 표현이 실제 다운스트림 과제에 얼마나 유용한가?

---

## 2. PRiSM의 전체 구조

PRiSM은 크게 두 종류의 평가를 합니다.

### A. Intrinsic: 코어 능력 평가
- **입력**: 음성
- **출력**: phonetic transcription(음성 전사)
- **비교 대상**: 사람이 만든 정답 IPA 전사
- **목표**: “이 모델이 소리를 얼마나 정확히 음성 단위로 재현하는가?”

### B. Extrinsic: 다운스트림 유용성 평가
- **입력 1**: 모델이 생성한 전사(transcription)
- **입력 2**: 모델의 hidden representation(내부 표현)
- **출력**: 분류/회귀/식별 등 downstream task 결과
- **목표**: “이 PR 시스템이 실제 응용에 얼마나 도움 되는가?”

---

## 3. 인트린식 평가의 구체적 예시

## 3.1 태스크: Phone Recognition
논문에서는 **PFER(Phonetic Feature Error Rate)** 를 사용해 평가합니다.

### 입력
- 음성 오디오

### 출력
- IPA 기반 전사

### 정답
- 사람이 주석한 참조 전사(reference transcription)

### 평가 방식
- 모델 출력과 정답 전사 사이의 **음성학적 feature 차이**를 계산
- 단순히 phone token이 맞는지(PER)만 보는 것이 아니라  
  **voicing, rounding, place, manner** 같은 특성 차이를 반영

---

## 3.2 구체적 데이터셋 예시

논문은 seen language variation과 unseen language를 나눠 평가합니다.

### (1) Seen language variation
이 범주는 “학습 중에 본 언어이지만, 발음이 다소 변형된 경우”입니다.

#### 데이터셋 예시
- **TIMIT**
- **L2-ARCTIC**
- **Speech Accent Archive**

#### 예시 입력
- 영어 음성
- 지역 억양이 다르거나
- 비원어민 영어 발화일 수 있음

#### 예시 출력
- TIMIT: 지역적 영어 발음을 IPA로 전사
- L2-ARCTIC: 비원어민 영어의 실제 발음 전사
- Speech Accent Archive: 다양한 L1 배경 화자의 “Please call Stella” 발화 전사

#### 예시 형태
예를 들어, “tell”이라는 단어가:
- 미국 영어에서는 [thEë]처럼 들릴 수 있고
- 스코틀랜드 영어에서는 [thEl]처럼 들릴 수 있음

즉, **표준 철자(tell)** 가 아니라 **실제 발음 형태**를 출력해야 합니다.

---

### (2) Unseen languages
이 범주는 “훈련 중 보지 못한 언어”에서의 일반화 능력을 봅니다.

#### 데이터셋 예시
- **DoReCo**
- **VoxAngeles**
- **Tusom2021**

#### 예시 입력
- 학습 데이터에 없던 저자원 언어 음성

#### 예시 출력
- 해당 언어의 실제 발음을 IPA로 가능한 정확하게 전사

#### 왜 중요한가?
모델이 특정 언어의 패턴을 외운 것이 아니라,  
**보편적인 음성학적 지식**을 학습했는지 확인할 수 있기 때문입니다.

---

## 4. Extrinsic 평가의 구체적 예시

논문은 PR 시스템의 결과를 두 가지 방식으로 downstream task에 넣습니다.

---

## 4.1 Transcript Probe (TP)
### 입력
- PR 모델이 생성한 **문자열 전사**
- 즉, IPA 기호들의 시퀀스

### probe 모델
- 텍스트 기반 Bi-GRU

### 출력
- 태스크에 따라 분류 결과 또는 점수

### 핵심 의미
전사가 얼마나 유용한 정보를 담고 있는지 평가

즉, 모델이 만든 IPA 전사만 가지고도  
아래 같은 태스크를 잘 풀 수 있는지 봅니다.

---

## 4.2 Representation Probe (RP)
### 입력
- PR 모델의 **hidden representation**
- 마지막 layer의 내부 특징 벡터

### probe 모델
- attention pooling + MLP

### 출력
- 태스크별 예측값

### 핵심 의미
전사가 아니라 **내부 표현 자체가 얼마나 유용한지** 평가

즉, 모델이 음향-음성학적 정보를 내부적으로 잘 담고 있는지 봅니다.

---

## 5. Extrinsic 태스크별 구체적 설명

---

## 5.1 Pathological Speech: 병리적 발화 평가

### (A) Dysarthria Intelligibility Prediction
논문에서는 발음장애(dysarthria) 음성의 **명료도(intelligibility)** 를 예측합니다.

#### 데이터셋
- **EasyCall**: 이탈리아어 dysarthric speech
- **UASpeech**: 영어 dysarthric speech

#### 입력
- 장애가 있는 화자의 음성

#### 출력
- 발화 명료도 점수 또는 등급

#### 예시
- UASpeech에서는 사람 청취자 기반으로 화자의 intelligibility를 구간화
- 모델은 이 점수를 맞추도록 학습/평가

#### 의미
발음 이상이 있는 화자의 소리를 PR 모델이 얼마나 잘 반영하는지 확인

---

### (B) Child Speech Disorder Detection
#### 데이터셋
- **UltraSuite**

#### 입력
- 아동의 음성

#### 출력
- 0: typically developing
- 1: speech sound disorder

#### 의미
아이의 발화가 정상 발달인지,  
아니면 speech disorder가 있는지 분류하는 작업입니다.

이 경우 중요한 것은:
- 단순 단어 인식이 아니라
- **비정상적 발음 패턴**을 잘 잡아내는지입니다.

---

## 5.2 L2 Speech: 제2언어 발화 평가

### (A) L2 Assessment
#### 데이터셋
- **Speechocean762**

#### 입력
- 비원어민 영어 발화

#### 출력
- 0~10 사이의 발음 품질/정확도 점수

#### 의미
얼마나 자연스럽고 정확하게 발음했는지 평가

예:
- 9~10: 매우 정확
- 5~6: 이해 가능하지만 오류가 많음
- 0~2: 거의 이해 불가

---

### (B) L1 Classification
#### 데이터셋
- **EdAcc**
- **L2-ARCTIC + CMU-Arctic**

#### 입력
- 영어로 말하는 비원어민 화자의 음성

#### 출력
- 화자의 모국어(L1) 또는 accent cluster

#### 예시
EdAcc에서는 13개 accent cluster 중 하나로 분류:
- South Asian
- Romance
- East Asian
- Slavic/Balkan
- African English
등

#### 의미
모국어가 영어 발화에 남기는 체계적 흔적을 이용해  
화자의 배경 언어를 추정

---

## 5.3 Multilingual Speech: 다국어 식별/지리 추정

### (A) Language Identification (LID)
#### 데이터셋
- **FLEURS-24**

#### 입력
- 다양한 언어의 음성

#### 출력
- 24개 언어 중 하나

#### 예시
입력 음성이 Gujarati인지, Persian인지, Pashto인지 등 분류

#### 의미
모델이 여러 언어의 음성적 차이를 얼마나 잘 구분하는지 평가

---

### (B) Speech Geolocation
#### 데이터셋
- **Vaani-Hi**

#### 입력
- 힌디어 방언 음성

#### 출력
- 위도/경도 좌표

#### 의미
화자가 인도 어느 지역 출신인지 음성만 보고 예측

이건 단순한 언어 식별보다 더 세밀합니다.  
왜냐하면 같은 언어 안에서도 지역에 따라:
- 발음
- 억양
- 자음 중첩
- 모음 품질
같은 특징이 다르기 때문입니다.

---

### (C) Phone Inventory Induction
#### 데이터셋
- **DoReCo**

#### 입력
- 한 언어의 음성 샘플들

#### 출력
- 그 언어에 존재하는 phone inventory 추정

#### 의미
모델이 해당 언어의 음소 체계를 추측할 수 있는가를 보는 것

즉, 전사 결과를 모아서:
- 어떤 IPA phone이 나타나는지
- 실제 언어의 inventory와 얼마나 겹치는지
를 비교합니다.

---

## 6. “구체적인 입력/출력” 관점에서 다시 정리

아래는 사용자가 원한 형식에 맞춰 아주 직접적으로 정리한 것입니다.

---

### 6.1 Intrinsic task 예시

#### 입력
- 1개의 음성 파일
- 예: 영어 문장, 비원어민 영어, 저자원 언어 발화

#### 출력
- IPA 전사 한 줄
- 예: `[həˈloʊ]`, `[thEë]` 같은 형태의 phonetic string

#### 정답
- 사람이 라벨링한 IPA 전사

#### 평가
- PFER 낮을수록 좋음

---

### 6.2 Transcript Probe 예시

#### 입력
- 모델이 생성한 전사 문자열
- 예: `/k æ t/` 비슷한 phone sequence

#### probe의 역할
- 이 문자열만 보고 화자 분류, 언어 식별, 명료도 예측 등을 수행

#### 출력
- 예:
  - 0/1 class
  - 0~10 점수
  - 위도/경도
  - 언어 ID

---

### 6.3 Representation Probe 예시

#### 입력
- PR 모델의 내부 hidden state sequence

#### probe의 역할
- 내부 표현을 받아 다운스트림 예측

#### 출력
- 전사 대신 직접 태스크 결과

#### 의미
- 전사가 틀려도 내부 표현이 유용할 수 있음
- 반대로 전사가 좋아도 내부 표현이 약할 수 있음

---

## 7. 논문이 강조하는 중요한 관찰

논문은 단순히 “누가 더 잘 맞추나”를 넘어서 다음을 보여줍니다.

### 1) seen language와 unseen language는 다르게 평가해야 함
- seen language: 익숙한 발음 패턴이 도움이 됨
- unseen language: multilingual training과 phonological generalization이 중요

### 2) 전사 정확도만으로는 충분하지 않음
- transcription error rate가 낮아도 downstream utility가 낮을 수 있음
- 반대로 전사가 다소 부정확해도 representation이 유용할 수 있음

### 3) encoder-CTC가 안정적
- 다양한 조건에서 안정적인 PR 성능을 보임

### 4) LALM은 아직 PR에 약함
- 대형 audio language model은 일반적인 음성 과제에서는 유용할 수 있지만
- 세밀한 phonetic perception에서는 specialized PR model보다 뒤처짐

---

## 8. 한 문장 요약
이 논문은 **음성 모델이 단순히 단어를 맞히는 수준을 넘어서, 실제 발음의 세부 음성학적 특징을 얼마나 잘 재현하고 활용하는지**를 보기 위해,  
**음성→IPA 전사** 및 **그 전사/내부표현을 이용한 다운스트림 과제**를 체계적으로 평가하는 벤치마크입니다.

---



## 1. The main goal of the paper
PRiSM benchmarks **phone realization (PR)** in speech models.  
The idea is to evaluate not only whether a model can transcribe speech, but whether it can capture the **fine-grained phonetic realization** of actual spoken language.

A PR system takes:
- **Input:** speech audio
- **Output:** phonetic transcription, usually in IPA

The benchmark evaluates this in two ways:

1. **Intrinsic evaluation**  
   → How close is the predicted phonetic transcription to the gold transcription?

2. **Extrinsic evaluation**  
   → How useful are the transcription and/or internal representations for downstream tasks?

---

## 2. Overall PRiSM framework

### A. Intrinsic evaluation
- **Input:** speech audio
- **Output:** phonetic transcription
- **Reference:** human-annotated IPA transcription
- **Goal:** measure how accurately the model realizes the spoken phones

### B. Extrinsic evaluation
- **Input 1:** predicted transcript
- **Input 2:** hidden representation from the speech model
- **Output:** downstream prediction such as classification or regression
- **Goal:** measure practical utility of phonetic information

---

## 3. Intrinsic evaluation: concrete example

## 3.1 Phone Recognition
The paper uses **PFER (Phonetic Feature Error Rate)** to score performance.

### Input
- Speech audio

### Output
- IPA-based phonetic transcription

### Reference
- Gold human annotation

### What is measured?
- Not just token-level phone errors
- But feature-level differences such as:
  - voicing
  - rounding
  - place of articulation
  - manner of articulation

So the model is judged on how well it reproduces the actual phonetic realization.

---

## 3.2 Datasets used for intrinsic evaluation

### (1) Seen-language variation
These are languages seen during training, but with different accents or speaking styles.

#### Examples
- **TIMIT**
- **L2-ARCTIC**
- **Speech Accent Archive**

#### Example input
- English speech
- Regional English accents
- Non-native English speech

#### Example output
- IPA transcription reflecting actual pronunciation

For example, the word “tell” may be realized differently in different English varieties.  
The model must transcribe the pronunciation, not just the orthographic word.

---

### (2) Unseen languages
These test generalization to languages not seen in training.

#### Examples
- **DoReCo**
- **VoxAngeles**
- **Tusom2021**

#### Example input
- Speech in a low-resource language absent from training

#### Example output
- IPA transcription of the spoken utterance

#### Why is this important?
It tests whether the system learned general phonetic knowledge rather than memorizing language-specific patterns.

---

## 4. Extrinsic evaluation: concrete example

The paper uses two kinds of inputs for downstream tasks.

---

## 4.1 Transcript Probe (TP)
### Input
- The predicted phonetic transcription from the PR model

### Probe model
- A text-based bi-GRU

### Output
- Task-specific prediction:
  - classification
  - regression
  - localization, etc.

### Meaning
This tests how useful the generated transcript is as a compact, interpretable representation of phonetic information.

---

## 4.2 Representation Probe (RP)
### Input
- The last-layer hidden representations from the PR model

### Probe model
- attention pooling + MLP

### Output
- Task-specific prediction

### Meaning
This tests whether the internal speech representation itself contains useful phonetic information, even without explicit transcription.

---

## 5. Downstream tasks in detail

---

## 5.1 Pathological speech

### (A) Dysarthria intelligibility prediction
#### Datasets
- **EasyCall**
- **UASpeech**

#### Input
- Speech from dysarthric speakers

#### Output
- Intelligibility or severity score

#### Goal
To see whether phonetic cues can support clinical assessment of speech disorders.

---

### (B) Child speech disorder detection
#### Dataset
- **UltraSuite**

#### Input
- Child speech recordings

#### Output
- 0 = typically developing
- 1 = speech sound disorder

#### Goal
To classify whether a child has atypical speech production.

---

## 5.2 L2 speech

### (A) L2 proficiency / pronunciation assessment
#### Dataset
- **Speechocean762**

#### Input
- Non-native English speech

#### Output
- Sentence-level pronunciation score from 0 to 10

#### Goal
To measure pronunciation quality and intelligibility.

---

### (B) L1 classification
#### Datasets
- **EdAcc**
- **L2-ARCTIC + CMU ARCTIC**

#### Input
- English speech from a non-native speaker

#### Output
- Speaker’s native language or accent cluster

#### Goal
To infer the speaker’s L1 from segmental and prosodic cues.

---

## 5.3 Multilingual speech

### (A) Language identification
#### Dataset
- **FLEURS-24**

#### Input
- Speech in one of 24 languages

#### Output
- Language class ID

#### Goal
To see whether the system can distinguish languages acoustically.

---

### (B) Speech geolocation
#### Dataset
- **Vaani-Hi**

#### Input
- Hindi dialect speech

#### Output
- Geographic coordinates: latitude and longitude

#### Goal
To infer regional origin from dialectal features.

---

### (C) Phone inventory induction
#### Dataset
- **DoReCo**

#### Input
- Speech from a language

#### Output
- Estimated phone inventory

#### Goal
To determine whether the model can infer the set of phones used in an unseen language.

---

## 6. Very concrete input-output summary

### Intrinsic task
- **Input:** one speech utterance
- **Output:** one IPA transcription string
- **Reference:** gold IPA transcript
- **Metric:** PFER

### Transcript probe
- **Input:** predicted transcription string
- **Output:** downstream label/score

### Representation probe
- **Input:** hidden speech representations
- **Output:** downstream label/score

---

## 7. Key takeaways from the paper

1. Seen and unseen languages need different evaluation perspectives.
2. Transcription accuracy alone is not enough.
3. Internal representations can be highly informative.
4. Encoder-CTC models are generally more stable.
5. Large Audio Language Models still lag behind specialized PR models on fine phonetic perception.

---

## 8. One-sentence summary
PRiSM is a benchmark for evaluating whether speech models can accurately realize phonetic detail, not just transcribe words, by testing both **IPA transcription** and **downstream usefulness of transcripts and hidden representations**.



<br/>
# 요약


이 논문은 PRiSM이라는 오픈소스 벤치마크를 제안해, 음성모델의 phone recognition을 **내재적 평가(PFER)**와 **외재적 평가(전사/표현 probe)**로 함께 측정합니다.  
결과적으로 **다국어 노출이 많을수록 성능이 좋아지고**, 특히 **encoder-CTC 계열이 가장 안정적**이었으며, **전문 PR 모델이 LALM보다 전반적으로 우수**했습니다.  
예를 들어 **언어 변이/미지 언어/병리적 발화/L2 발화/다국어 식별** 같은 실제 과제에서 전사(Transcript probe)와 표현(Representation probe)을 모두 비교해, 단순 전사 정확도만으로는 포착되지 않는 모델 차이를 보여줍니다.  

This paper introduces PRiSM, an open-source benchmark that evaluates phone recognition in speech models through both **intrinsic evaluation (PFER)** and **extrinsic evaluation (transcript/representation probes)**.  
The main findings are that **more diverse multilingual exposure improves performance**, **encoder-CTC models are the most stable**, and **specialized PR models generally outperform LALMs**.  
For example, PRiSM compares models on real downstream tasks such as **language variation, unseen languages, pathological speech, L2 speech, and multilingual identification**, showing differences that plain transcription accuracy alone cannot capture.

<br/>
# 기타




## 1) 다이어그램/개요 그림 (Figure 1)
**핵심 내용**
- PRiSM은 **Phone Recognition(PR)** 시스템을 평가하는 첫 **오픈소스 벤치마크**입니다.
- 평가를 두 축으로 나눕니다:
  1. **Intrinsic**: 전사 정확도(코어 능력)
  2. **Extrinsic**: 다운스트림 과제에서의 유용성
- 전사(transcript)뿐 아니라 **내부 representation**도 함께 봅니다.

**인사이트**
- 단순히 “맞게 전사했는가”만 보면 PR 모델의 실제 가치를 놓칠 수 있음.
- **전사 성능과 다운스트림 성능은 반드시 일치하지 않음**.
- 따라서 **표면적 정확도 + 실제 활용성**을 함께 봐야 함.

---

## 2) 테이블 1: 평가 태스크 구성
**핵심 내용**
- 평가 태스크를 크게 두 범주로 구성:
  - **Intrinsic**
    - seen language variation: TIMIT, L2-ARCTIC, Speech Accent Archive
    - unseen languages: DoReCo, VoxAngeles, Tusom2021
  - **Extrinsic**
    - pathological speech
    - L2 speech
    - multilingual speech

**인사이트**
- PR 모델이 잘해야 하는 것은 단지 영어만이 아니라:
  - **변이된 발화**
  - **낯선 언어**
  - **임상/교육/다국어 상황**
  까지 포괄함.
- 즉, **언어 범용성 + 실제 응용성**이 benchmark 설계의 핵심.

---

## 3) 테이블 2: 비교한 모델들
**핵심 내용**
- 모델군:
  - **Wav2Vec2 계열**
  - **ZIPA 계열**
  - **POWSM 계열**
  - **LALMs**(Gemini, Qwen3-Omni)
- 아키텍처/학습 방식이 다양함:
  - CTC
  - AED(encoder-decoder)
  - CR-CTC
  - 대규모 멀티링구얼 사전학습 등

**인사이트**
- PR 성능은 **모델 크기만의 문제가 아님**.
- **언어 커버리지**, **아키텍처(특히 encoder-CTC)**, **학습 데이터 다양성**이 중요.
- LALM은 범용성은 높아 보이지만, PR에서는 아직 전문 PR 모델보다 뒤처짐.

---

## 4) 테이블 3: Intrinsic 결과(PFER)
**핵심 결과**
- 전반적으로 **CTC 기반 모델이 더 안정적**.
- **ZIPA-CTC-NS, W2V2P-XLSR53** 등이 좋은 성능.
- **Gemini, Qwen3-Omni 같은 LALM은 취약**:
  - unseen language에서 특히 성능이 매우 낮거나 불안정
  - 반복/퇴화 출력(degenerate output) 발생
- seen language variation과 unseen languages의 패턴이 다름.

**인사이트**
- **seen language variation**에서는 익숙한 패턴에 기반한 출력이 유리.
- **unseen languages**에서는 **멀티링구얼 훈련**이 훨씬 중요.
- 즉, PR은 단순 음향 인식이 아니라 **음운 패턴과 언어 다양성 학습**이 중요함.

---

## 5) 테이블 4: Extrinsic 결과(TP vs RP)
**핵심 결과**
- **Transcript Probe(TP)**와 **Representation Probe(RP)**의 우열이 task마다 다름.
- **Whisper는 RP에서 특히 강함**.
- **ZIPA류는 RP가 강하고 TP는 상대적으로 약한 편**.
- **Wav2Vec2Phs는 TP에서 강하지만 RP에서 제한적**.
- **LALM은 전반적으로 downstream에서도 약함**.

**인사이트**
- 전사가 좋다고 representation이 반드시 좋은 것은 아님.
- 반대로 representation이 좋다고 전사가 좋은 것도 아님.
- **다른 downstream task는 서로 다른 phonetic 정보를 요구**한다는 점이 중요.
- 임상/발달/다국어 태스크에서 **TP와 RP의 유불리가 다르게 나타남**.

---

## 6) Figure 2: Phone masking 실험
**핵심 내용**
- 입력 음성의 일부 phone을 silence로 가린 뒤, PR이 얼마나 영향을 받는지 봄.
- 만약 모델이 **오직 음향(acoustic signal)**만 본다면, masking이 늘어도 성능 저하가 제한적이어야 함.
- 그런데 실제로는 일부 모델이 masking에 민감.

**결과**
- **Wav2Vec2Phs, POWSM-CTC**는 masking이 커져도 비교적 잘 버팀.
- **ZIPA, POWSM**는 masking 증가 시 성능이 더 나빠짐.
- 이는 모델이 음향뿐 아니라 **phonotactics(음운배열 규칙)**도 많이 활용한다는 뜻.

**인사이트**
- PR 모델은 단순히 소리를 “듣는” 것이 아니라, **언어 내부의 음운적 선호 패턴을 학습**할 수 있음.
- 특히 ZIPA는 **phonotactics + acoustics**를 함께 쓰는 경향.
- **encoder-only + CTC**가 음향 fidelity 측면에서 더 유리할 수 있음.

---

## 7) Figure 3: Zero-shot phone inventory induction
**핵심 결과**
- **POWSM-CTC가 가장 강함**.
- **encoder-only 구조**가 unseen phonetic environment에서 precision을 높이는 데 중요.
- **ZIPA-CTC-NS**는 ZIPA-CTC보다 더 precise:
  - pseudo-labeled data + 더 많은 언어 커버리지의 효과
- 데이터 양만큼이나 **언어 다양성**이 중요.

**인사이트**
- 새로운 언어의 phone inventory를 추론하는 작업에서는
  - **많은 데이터**보다
  - **다양한 언어 exposure**
  가 더 중요할 수 있음.
- unseen language에서 좋은 모델은 단순히 “더 많이 본 모델”이 아니라 **더 넓게 본 모델**.

---

## 8) Figure 4: Dialect geolocation attribution
**핵심 결과**
- TP가 RP보다 **힌디 방언 geolocation**에서 더 좋음.
- TP는 **phone order 정보**를 살려 지역 차이를 구분하는 데 유리.
- 예시에서 doubled consonant 같은 **미세한 발음 차이**를 잘 잡아냄.

**인사이트**
- 방언 판별은 단순한 발화 벡터보다 **순서가 유지된 phonetic sequence**가 더 중요할 수 있음.
- RNN 기반 transcript probe가 **지역별 분포 차이**를 잘 활용.
- 즉, **dialect 정보는 세그먼트 순서와 음운 변화 패턴에 강하게 의존**.

---

## 9) Figure 5 / Section 6.4: LALM bias 분석
**핵심 결과**
- LALM은:
  - GEO-v에서 **거의 chance 수준**
  - L1-eda에서도 특정 accent cluster로 편향
- 특히 thinking mode가 오히려 성능을 떨어뜨림.
- 뉴델리/로망스 계열 등 특정 중심 클래스로 **mode collapse** 발생.

**인사이트**
- LALM은 **미세한 음성학적 차이**에 둔감.
- reasoning을 붙인다고 해결되지 않고, 오히려 **편향이 강화**될 수 있음.
- 즉, LALM은 현재 PR/phonetic discrimination에 적합하지 않음.

---

## 10) Appendix D: LALM 프롬프트
**핵심 내용**
- 각 태스크에 대해 매우 구체적인 프롬프트를 사용.
- 예: IPA 전사, dysarthria severity, accent cluster 분류, LID, geolocation 등.

**인사이트**
- LALM 성능은 단순 모델 능력만이 아니라 **프롬프트 민감도**도 큼.
- 그런데도 결과가 전반적으로 낮다면, 이는 **근본적인 phonetic perception 한계**를 시사.

---

## 11) Appendix E: EdAcc → accent cluster 매핑
**핵심 내용**
- 41개 L1 라벨을 13개 accent cluster로 통합.
- 기준:
  - 언어 계통
  - 모음 체계
  - prosody
  - 영어로의 transfer pattern

**인사이트**
- accent 인식은 세부 국가/언어 단위보다 **phonological similarity 기반 clustering**이 더 타당할 수 있음.
- 즉, 인간이 듣는 accent 차이도 **음운적·운율적 범주**로 묶는 것이 합리적.

---

## 12) Appendix F: Vaani-Hi 구축
**핵심 내용**
- Hindi-belt 지역의 방언 geolocation 태스크를 위해
  - 여러 주/구역에서 데이터 샘플링
  - pincode별 좌표화
  - train/val/test를 location leakage 없게 분리

**인사이트**
- 지리 추론 태스크는 데이터 누수 방지가 매우 중요.
- location leakage를 막아야 진짜 dialectal cue를 평가 가능.
- 따라서 이 태스크는 **방언음성학 + 엄격한 데이터 설계**가 결합된 예.

---

## 전체 요약 인사이트
- PRiSM의 핵심 메시지는:
  1. **전사 정확도만으로 PR 모델을 평가하면 부족**
  2. **intrinsic + extrinsic을 함께 봐야 함**
  3. **멀티링구얼 학습과 CTC 기반 encoder 구조가 강함**
  4. **LALM은 아직 PR/phonetic perception에 약함**
  5. **다운스트림 태스크는 phonetic info의 종류(전사 vs representation)에 따라 다르게 반응함**

---




## 1) Overview diagram / Figure 1
**Main point**
- PRiSM is the first **open-source benchmark** for **Phone Recognition (PR)** systems.
- It evaluates models along two axes:
  1. **Intrinsic**: core transcription accuracy
  2. **Extrinsic**: downstream utility
- It probes both **transcripts** and **internal representations**.

**Insight**
- Surface-level transcription accuracy alone is insufficient.
- **Transcription quality and downstream usefulness do not always align**.
- So PR systems should be evaluated on both **core phonetic accuracy** and **practical utility**.

---

## 2) Table 1: Evaluation tasks
**Main point**
- The benchmark is divided into:
  - **Intrinsic**
    - seen-language variation: TIMIT, L2-ARCTIC, Speech Accent Archive
    - unseen languages: DoReCo, VoxAngeles, Tusom2021
  - **Extrinsic**
    - pathological speech
    - L2 speech
    - multilingual speech

**Insight**
- A good PR model must handle:
  - variant speech,
  - unseen languages,
  - and real-world clinical/educational/multilingual settings.
- So the benchmark emphasizes **generalization + practical usefulness**.

---

## 3) Table 2: Compared model families
**Main point**
- Model families include:
  - **Wav2Vec2-based**
  - **ZIPA-based**
  - **POWSM-based**
  - **LALMs** such as Gemini and Qwen3-Omni
- They differ in:
  - CTC vs AED
  - consistency-regularized CTC
  - multilingual pretraining and supervised data scale

**Insight**
- PR performance is not just about model size.
- **Language coverage**, **architecture**—especially **encoder-CTC**—and **training diversity** matter a lot.
- LALMs may be broad-purpose models, but they still lag behind specialized PR systems.

---

## 4) Table 3: Intrinsic results (PFER)
**Main result**
- **CTC-based models are generally more stable**.
- Strong models include **ZIPA-CTC-NS** and **W2V2P-XLSR53**.
- **LALMs perform poorly**, especially on unseen languages:
  - unstable output
  - repeated/degenerated generations
- Seen-language variation and unseen-language behavior differ substantially.

**Insight**
- For **seen-language variation**, outputs grounded in familiar patterns help.
- For **unseen languages**, **multilingual training** becomes crucial.
- PR is therefore not just acoustic decoding; it requires learning **phonological patterns** across languages.

---

## 5) Table 4: Extrinsic results (TP vs RP)
**Main result**
- The ranking differs by task and probe type:
  - **Whisper is very strong on RP**
  - **ZIPA models are strong on RP but weaker on TP**
  - **Wav2Vec2Phs is strong on TP but gains less on RP**
  - **LALMs are weak overall**

**Insight**
- Good transcription does not guarantee good representations, and vice versa.
- Different downstream tasks require **different types of phonetic information**.
- So benchmarking both TP and RP is necessary.

---

## 6) Figure 2: Phone masking experiment
**Main point**
- Parts of the phone sequence are replaced with silence.
- If a model relies purely on acoustics, performance should not degrade much as masking increases.

**Result**
- **Wav2Vec2Phs** and **POWSM-CTC** remain relatively robust.
- **ZIPA** and **POWSM** degrade more with higher masking.
- This suggests reliance on **phonotactics** in addition to acoustics.

**Insight**
- PR models may learn not only acoustics but also **language-specific phonotactic preferences**.
- ZIPA appears to use both **phonotactics + acoustics**.
- Encoder-only CTC models seem more faithful to the acoustic signal.

---

## 7) Figure 3: Zero-shot phone inventory induction
**Main result**
- **POWSM-CTC performs best**.
- **Encoder-only architecture** is important for precision in unseen phonetic environments.
- **ZIPA-CTC-NS** improves over ZIPA-CTC due to:
  - pseudo-labeled data
  - broader multilingual coverage

**Insight**
- For new-language phone inventory discovery, **diversity of languages** matters as much as data volume.
- The best models are not necessarily those that saw the most data, but those that saw the **widest linguistic diversity**.

---

## 8) Figure 4: Dialect geolocation attribution
**Main result**
- Transcript probes outperform representation probes on Hindi dialect geolocation.
- TP leverages **phone order information** better.
- It can detect fine-grained features like **double consonants**.

**Insight**
- Dialect identification may depend heavily on **ordered phonetic sequences**, not just pooled representations.
- Transcript-based RNN probes can exploit **distributional differences in phone sequences** more effectively.

---

## 9) Figure 5 / Section 6.4: LALM bias analysis
**Main result**
- LALMs are near chance on geolocation.
- On accent classification, they show strong bias toward a few clusters.
- “Thinking mode” makes performance worse, not better.

**Insight**
- LALMs are insensitive to fine phonetic and dialectal differences.
- Added reasoning does not fix this; it may even amplify bias.
- So current LALMs are not reliable for fine-grained phonetic discrimination.

---

## 10) Appendix D: LALM prompts
**Main point**
- The paper uses carefully designed prompts for IPA transcription, dysarthria rating, accent classification, LID, and geolocation.

**Insight**
- LALM performance depends on prompting, but the poor results still suggest a **fundamental limitation in phonetic perception**.

---

## 11) Appendix E: EdAcc to accent-cluster mapping
**Main point**
- 41 L1 labels are collapsed into 13 accent clusters using:
  - language family
  - vowel system
  - prosody
  - transfer patterns

**Insight**
- Accent variation is better modeled via **phonological similarity** rather than only country or language labels.
- This is a linguistically sensible grouping.

---

## 12) Appendix F: Vaani-Hi construction
**Main point**
- The Hindi-belt geolocation dataset is carefully built by:
  - sampling across states/districts
  - mapping pin codes to coordinates
  - splitting to avoid location leakage

**Insight**
- Geolocation tasks require strict leakage control.
- This makes the benchmark genuinely test **dialectal cues**, not memorized location artifacts.

---

## Overall takeaway
PRiSM’s main message is:
1. **Transcription accuracy alone is not enough**
2. **Intrinsic + extrinsic evaluation are both necessary**
3. **Multilingual training and encoder-CTC architectures are strongest**
4. **LALMs still lag behind specialized PR models**
5. **Different downstream tasks reward different kinds of phonetic information**



<br/>
# refer format:




## BibTeX

```bibtex
@article{bharadwaj2026prism,
  title={PRiSM: Benchmarking Phone Realization in Speech Models},
  author={Bharadwaj, Shikhar and Li, Chin-Jou and Kim, Yoonjae and Choi, Kwanghee and Yeo, Eunjung and Shim, Ryan Soh-Eun and Zhou, Hanyu and Boldt, Brendon and Rosero Jacome, Karen and Chang, Kalvin and Agrawal, Darsh and Xu, Keer and Yang, Chao-Han Huck and Zhu, Jian and Watanabe, Shinji and Mortensen, David R.},
  journal={arXiv preprint arXiv:2601.14046},
  year={2026},
  url={https://arxiv.org/abs/2601.14046}
}
```

---

## 시카고 스타일 참고문헌(줄글)

Bharadwaj, Shikhar, Chin-Jou Li, Yoonjae Kim, Kwanghee Choi, Eunjung Yeo, Ryan Soh-Eun Shim, Hanyu Zhou, Brendon Boldt, Karen Rosero Jacome, Kalvin Chang, Darsh Agrawal, Keer Xu, Chao-Han Huck Yang, Jian Zhu, Shinji Watanabe, and David R. Mortensen. “PRiSM: Benchmarking Phone Realization in Speech Models.” *arXiv* (2026). https://arxiv.org/abs/2601.14046.

---


