---
layout: post
title:  "[2026]Self-supervised Speech Models Discover Phonological Vector Arithmeti"
date:   2026-04-16 19:10:45 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 메써드: WavLM 같은 자기지도 음성모델의 phone 표현에서 PanPhon 특징으로 음운적 사분조각(예: [b]:[p]=[d]:[t])을 만들고, \(r_{p1}\approx r_{p2}+r_{p3}-r_{p4}\) 형태의 선형 유추가 성립하는지 코사인 유사도로 검증했다.


짧은 요약(Abstract) :


이 논문은 **자기지도학습 음성 모델(self-supervised speech models, S3Ms)** 이 음성의 음운적 정보(phonology)를 단순히 담고 있는 것이 아니라, 그 정보를 **벡터의 선형 연산으로 표현**한다는 점을 보여줍니다.

핵심은 다음과 같습니다:

1. **음운적 벡터가 선형 방향으로 존재함**
   - 예를 들어, /d/와 /t/의 차이를 벡터로 잡으면 이것이 **유성음(voicing)** 을 나타내는 방향이 됩니다.
   - 이 벡터를 /p/에 더하면 /b/가 됩니다.
   - 즉, 음성 모델의 표현 공간 안에서 음운 특징이 **벡터 덧셈/뺄셈**으로 조합될 수 있습니다.

2. **벡터의 크기(scale)가 연속적인 음향 변화와 연결됨**
   - 단순히 “있다/없다”의 이진 구분이 아니라,
   - 벡터를 얼마나 크게 적용하느냐에 따라 발음의 정도가 점점 달라지는 **연속적 변화**가 나타납니다.
   - 예를 들어 voicing vector의 크기를 키우면, 유성성이 점진적으로 강해집니다.

3. **96개 언어에 걸쳐 분석함**
   - 영어뿐 아니라 여러 언어의 음성 데이터에서 이런 현상이 나타나는지 넓게 검증했습니다.
   - 그 결과, 모델이 음운 구조를 **일반적이고 언어를 넘어서** 학습하고 있음을 확인했습니다.

4. **결론**
   - S3M은 음성을 임의의 숫자 벡터로 표현하는 것이 아니라,
   - **음운적으로 해석 가능한(composable and interpretable) 벡터**로 표현한다는 점을 보여줍니다.
   - 즉, 음성 모델에서도 word2vec처럼 **벡터 산술(vector arithmetic)** 이 성립한다고 주장합니다.

---


This paper shows that self-supervised speech models (S3Ms) encode phonological information in a **linear and compositional way**.

Main findings:

1. **Phonological features correspond to linear directions in representation space**
   - For example, the vector difference between /d/ and /t/ acts as a **voicing vector**.
   - Adding this vector to /p/ produces /b/, demonstrating phonological vector arithmetic.

2. **The magnitude of these vectors reflects a continuous degree of phonological realization**
   - The features are not represented only as binary contrasts.
   - Scaling a phonological vector gradually changes the acoustic properties of the resynthesized speech.

3. **The study evaluates 96 languages**
   - The authors show that these phonological patterns generalize beyond English and apply cross-linguistically.

4. **Conclusion**
   - S3Ms encode speech using **phonologically interpretable, compositional vectors**.
   - This provides evidence that speech models exhibit **phonological vector arithmetic**, similar to semantic vector arithmetic in word embeddings.

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


## 1. 이 논문이 다루는 핵심 메서드 개요
이 논문은 **self-supervised speech models (S3Ms)**, 즉 **자기지도 학습 음성 모델**이 음성의 **음운론적(phonological) 정보**를 어떻게 내부에 표현하는지 분석합니다.  
핵심 아이디어는 텍스트 임베딩에서 word2vec이 보인 **벡터 산술(vector arithmetic)**처럼, 음성 모델의 표현 공간에서도 **음운 특징(예: 유성/무성, 조음 위치, 비음성 등)**이 **선형 방향(linear direction)**으로 존재할 수 있다는 것입니다.

즉, 이 논문은 단순히 “모델이 무엇을 알고 있나?”를 보는 것이 아니라,  
**“그 정보가 어떤 구조로 저장되어 있나?”**를 보는 연구입니다.

---

## 2. 사용한 모델: Self-supervised Speech Models (S3Ms)
논문에서 주요하게 사용한 모델은 아래 3개입니다.

1. **wav2vec 2.0**
2. **HuBERT**
3. **WavLM**

이들은 모두 **대규모 비라벨 음성 데이터**로 사전학습된 음성 표현 학습 모델입니다.

### 공통점
- 모두 **self-supervised learning** 기반
- 음성 텍스트 정답 없이 음성만으로 학습
- 중간 표현이 음소/음운/조음 같은 언어학적 정보를 담는지 분석 가능
- 각 모델의 **layer별 representation**을 추출해서 비교함

---

## 3. 모델 아키텍처
논문은 세 모델 모두 **LARGE configuration**을 사용합니다.

### 구조
- **7개의 1D CNN layer**
- 그 뒤에 **24개의 Transformer block**
- 총 파라미터 수는 약 **300M**

즉, 구조는 대략 다음과 같습니다.

**입력 음성 → CNN feature encoder → Transformer layers → contextual speech representation**

### 레이어별 표현 추출
- CNN 쪽에서 나온 표현을 **layer 0**
- Transformer block의 각 층을 **layer 1~24**
- 총 **25개 layerwise representation**을 분석

이렇게 한 이유는,  
음운 정보가 **초기 층 / 중간 층 / 마지막 층** 중 어디에 더 잘 나타나는지 보기 위해서입니다.

---

## 4. 사용한 데이터셋
논문은 두 개의 음성-음소 정렬 데이터셋을 사용합니다.

### 4.1 TIMIT
- 영어 음성 데이터
- 약 630명의 영어 화자
- 음소 segmentation이 정교하게 되어 있음
- 음운 특징 실험에 적합
- 영어 내 phonological structure 분석에 사용

### 4.2 VoxAngeles
- UCLA Phonetics Archive 기반
- **95개 언어**, **21개 language family**
- 영어 밖의 언어들까지 포함
- cross-lingual generalization 평가에 사용

### 왜 두 데이터셋을 썼나?
- **TIMIT**: 영어 내부에서 정밀 분석
- **VoxAngeles**: 영어에 없는 phone까지 포함해 일반화 확인

즉, 이 논문은 단순히 영어에만 맞는 현상이 아니라,  
**다른 언어의 음운 구조에도 같은 벡터 성질이 있는지**를 보려는 목적이 있습니다.

---

## 5. 음운 특징 추출 방법: PanPhon
논문은 phonological feature를 얻기 위해 **PanPhon**을 사용합니다.

### PanPhon이 하는 일
각 phone을 다음과 같은 **21개 음운 특징** 벡터로 바꿉니다.

예:
- voice
- labial
- nasal
- coronal
- distributed
- high
- low
- back
- round
- tense
- long
- strident
- sonorant
- etc.

각 특징은 원래 **+ / 0 / -** 형태인데,  
논문에서는 이를 **binary 확장**해서 사용합니다.

- `+` → `[1,0]`
- `0` → `[0,0]`
- `-` → `[0,1]`

그래서 최종적으로는 **42차원 이진 표현**이 됩니다.

### 왜 필요한가?
이렇게 해야 phone들 사이에서  
“특정 phonological feature가 같거나 다른 관계”를 체계적으로 찾을 수 있기 때문입니다.

---

## 6. 메인 실험 1: Phonological Vector Arithmetic
논문의 가장 중요한 메서드 중 하나입니다.

### 핵심 가설
S3M 표현 공간 안에는 phonological feature에 대응하는 **선형 벡터 방향**이 존재한다.

예를 들어:

- `[d] - [t]` 는 **voicing vector**
- 이 벡터를 `[p]`에 더하면 `[b]`에 가까워진다

즉,

\[
r[b] \approx r[p] + (r[d] - r[t])
\]

이런 식의 벡터 산술이 성립한다는 것입니다.

### 어떻게 검증했나?
논문은 phone quadruplet을 사용합니다. 예를 들어:

- `[b] : [p] = [d] : [t]`

이런 관계가 있으면, 대응하는 표현 벡터도 비슷한 구조를 가져야 한다고 봅니다.

### 평가 방식
논문은 cosine similarity를 사용합니다.

- target phone representation과
- `r[p1] + r[p3] - r[p4]`

가 얼마나 가까운지 봅니다.

그리고 다음 ordering이 만족되는지를 봅니다.

\[
cos^- < cos < cos^+
\]

즉,
- 다른 phone보다
- analogy로 만든 추정치가 더 가깝고
- 같은 phone의 다른 instance보다는 덜 가깝도록

평가합니다.

### 결과
- S3Ms는 spectral feature(MFCC, MelSpec)보다 훨씬 좋은 성능
- 특히 **WavLM**, **HuBERT**, **wav2vec 2.0**이 phonological analogy를 잘 포착
- 마지막 층이나 중간 층에서 성능이 높음
- 영어에 없는 phone도 어느 정도 일반화

---

## 7. 메인 실험 2: Phonological Vector Scale
두 번째 핵심 메서드는,  
벡터의 **방향(direction)**뿐 아니라 **크기(scale)**도 의미가 있는지 보는 것입니다.

### 기본식
\[
r[b] \approx r[p] + \lambda (r[d] - r[t])
\]

여기서 \(\lambda\)는 scaling factor입니다.

### 질문
- \(\lambda\)를 키우면 voicing이 더 강해지는가?
- \(\lambda\)를 줄이면 특징이 약해지거나 반대로 가는가?
- 즉, phonological feature가 이산적(binary)인지, 연속적(continuous)인지?

---

## 8. 음성 재합성(speech resynthesis) 기법
이 논문의 중요한 특별한 기법입니다.

### 왜 필요한가?
representation 공간에서 벡터를 바꿨을 때,  
그게 실제 음성의 어떤 변화로 이어지는지를 보기 위해서입니다.

### 방법
1. 입력 음성 \(x\)를 S3M에 넣어 representation \(R=f(x)\)를 얻음
2. 특정 phone 구간의 representation에 phonological vector \(v\)를 더함
3. scaling factor \(\lambda\)를 조절
4. vocoder로 다시 음성 생성:
   \[
   \tilde{x} = f^{-1}(\tilde{R})
   \]

즉, **representation 수정 → vocoder로 다시 음성 합성** 방식입니다.

---

## 9. Vocoder: Vocos
representation을 다시 음성으로 바꾸기 위해 논문은 **Vocos vocoder**를 사용했습니다.

### Vocos의 특징
- time-domain과 Fourier-based 방식을 잘 연결하는 neural vocoder
- 음질이 안정적
- oversampling/upsampling artifact가 상대적으로 적음
- OOD input에도 비교적 robust

### 학습한 vocoder 2종
- **English model**: LibriTTS로 학습
- **Multilingual model**: FLEURS-R로 학습

즉, synthesis 실험에서도 영어뿐 아니라 여러 언어로 일반화를 테스트했습니다.

---

## 10. 어떤 phonological features를 실험했나?
논문은 총 **8개 특징**을 scale 분석에서 중점적으로 다룹니다.

### 모음 관련
- high
- low
- back
- round

### 자음 관련
- nasal
- sonorant
- strident
- voice

이들 각각에 대해 acoustic measurement와 대응시켰습니다.

예:
- high/low → F1
- back/round → F2
- nasal → F1 bandwidth
- sonorant → HNR
- strident/voice → COG

---

## 11. acoustic measurement와의 비교
논문은 resynthesis된 음성에서 실제 acoustic measurement를 계산하고,  
\(\lambda\)와의 상관을 봅니다.

### 사용한 지표
- **F1**
- **F2**
- **F1 bandwidth**
- **HNR**
- **COG**

### 결과 해석
- \(\lambda\)가 커질수록 해당 음운 특징이 더 강하게 나타남
- 기대한 방향의 상관관계가 실제로 관찰됨
- 단순히 이진적인 on/off가 아니라 **연속적인 변화**를 보임

예를 들어 voicing vector를 키우면:
- voice onset time이 변하고
- voiced cue가 더 강해지고
- COG도 연속적으로 바뀜

---

## 12. 왜 이 메서드가 중요한가?
이 논문의 방법론은 크게 두 층위입니다.

### 1) 구조 분석
- phonological analogy가 되는가?
- representation space 안에 phonological vector가 있는가?

### 2) 조작 가능성 분석
- 그 vector를 더하고 빼고 스케일하면 실제 음성 특성이 변하는가?

즉, 단순 probing이 아니라  
**representation의 기하학적 구조와 생성적 효과**까지 본다는 점이 중요합니다.

---

## 13. 이 논문 메서드의 요약
한 줄로 정리하면:

> **Self-supervised speech model의 layerwise representation을 추출한 뒤, PanPhon 기반 phonological feature를 이용해 phone analogy를 만들고, cosine similarity로 vector arithmetic를 검증하며, Vocos vocoder로 resynthesis하여 vector scale이 실제 acoustic cue를 연속적으로 조절하는지 분석한 연구**입니다.

---



## 1. High-level method overview
This paper studies how **self-supervised speech models (S3Ms)** encode **phonological information** internally.  
The main idea is analogous to word embeddings in word2vec: just as semantic relations can appear as **vector arithmetic** in text embeddings, phonological relations may appear as **linear directions** in speech representations.

So the paper asks not only:

- “What information is encoded?”

but also:

- “How is that information structured in the representation space?”

---

## 2. Models used: Self-supervised speech models
The paper focuses on three major S3Ms:

1. **wav2vec 2.0**
2. **HuBERT**
3. **WavLM**

These models are pretrained on large amounts of **unlabeled speech** and are widely used as general speech representation learners.

### Shared properties
- self-supervised pretraining
- no phonological supervision
- layerwise speech representations can be extracted
- suitable for analyzing phonetic/phonological structure

---

## 3. Model architecture
All three models are evaluated in their **LARGE configuration**.

### Architecture
- **7 layers of 1D CNNs**
- followed by **24 Transformer blocks**
- around **300M parameters** in total

So the pipeline is roughly:

**raw speech → CNN feature encoder → Transformer layers → contextual speech representations**

### Layerwise representations
- CNN output is treated as **layer 0**
- Transformer blocks are treated as **layers 1–24**
- In total, **25 layerwise representations** are analyzed

This allows the authors to examine where phonological structure emerges across depth.

---

## 4. Datasets
Two phonetic/phonologically segmented datasets are used.

### 4.1 TIMIT
- English speech corpus
- 630 speakers
- carefully segmented phonetic annotations
- used for English-specific analysis

### 4.2 VoxAngeles
- based on the UCLA Phonetics Archive
- **95 languages** across **21 language families**
- used to test cross-linguistic generalization
- includes phones not present in English

### Why both datasets?
- **TIMIT**: detailed English analysis
- **VoxAngeles**: broader multilingual generalization

---

## 5. Phonological feature extraction: PanPhon
The paper uses **PanPhon** to map each phone to a set of phonological features.

### PanPhon provides
21 phonological features such as:
- voice
- labial
- nasal
- coronal
- high
- low
- back
- round
- tense
- long
- strident
- sonorant
- etc.

Each feature is originally encoded as:
- `+`
- `0`
- `-`

The paper binarizes this into:
- `+` → `[1,0]`
- `0` → `[0,0]`
- `-` → `[0,1]`

giving a **42-dimensional binary feature vector**.

---

## 6. Main Experiment 1: Phonological vector arithmetic
This is the central structural test.

### Hypothesis
Phonological features correspond to **linear directions** in S3M representation space.

For example:
- `[d] - [t]` forms a **voicing vector**
- adding that vector to `[p]` should move it toward `[b]`

Formally:

\[
r[b] \approx r[p] + (r[d] - r[t])
\]

### How it is tested
The authors construct **phone quadruplets** such as:

- `[b] : [p] = [d] : [t]`

and test whether the corresponding representation offsets follow the same relation.

### Evaluation
They use **cosine similarity** and check whether:

\[
cos^- < cos < cos^+
\]

where:
- `cos` = analogy-based similarity
- `cos+` = same-phone baseline
- `cos-` = different-phone baseline

### Results
- S3Ms outperform spectral baselines like **MFCC** and **MelSpec**
- **WavLM**, **HuBERT**, and **wav2vec 2.0** capture phonological analogies well
- stronger performance usually appears in middle or later layers
- the models generalize to unseen phones and unseen languages to some extent

---

## 7. Main Experiment 2: Scale of phonological vectors
The second key question is whether not only the **direction** but also the **magnitude** of a phonological vector matters.

### Core equation
\[
r[b] \approx r[p] + \lambda (r[d] - r[t])
\]

Here, \(\lambda\) controls the strength of the phonological feature.

### Question
Does increasing \(\lambda\) produce a gradual increase in the relevant acoustic cue?

This tests whether phonological features are represented as:
- binary contrasts, or
- continuous dimensions

---

## 8. Speech resynthesis with a vocoder
To test the effect of vector scaling on actual speech output, the authors perform **resynthesis**.

### Procedure
1. Extract representation \(R=f(x)\) from input speech \(x\)
2. Add a scaled phonological vector \(\lambda v\) to the target phone frames
3. Reconstruct speech using a vocoder:
   \[
   \tilde{x} = f^{-1}(\tilde{R})
   \]

This lets them verify whether modifying representation space changes the produced audio in linguistically meaningful ways.

---

## 9. Vocoder used: Vocos
The paper uses **Vocos** as the neural vocoder.

### Why Vocos?
- high-quality neural vocoder
- less prone to upsampling artifacts
- robust to out-of-distribution inputs

### Two vocoder training setups
- **English vocoder** trained on **LibriTTS**
- **Multilingual vocoder** trained on **FLEURS-R**

---

## 10. Phonological vectors analyzed in the scaling experiment
The paper focuses on eight features:

### Vowels
- high
- low
- back
- round

### Consonants
- nasal
- sonorant
- strident
- voice

Each is linked to a known acoustic correlate such as:
- F1
- F2
- F1 bandwidth
- HNR
- COG

---

## 11. Acoustic measurement analysis
The authors compute acoustic measurements on the resynthesized speech and correlate them with \(\lambda\).

### Measurements used
- **F1**
- **F2**
- **F1 bandwidth**
- **HNR**
- **COG**

### Main finding
The measurements change **monotonically** with \(\lambda\), and the direction of change matches phonological expectations.

This indicates that phonological vectors are:
- linearly compositional
- continuously scalable
- acoustically interpretable

---

## 12. Why this method matters
The paper combines two levels of analysis:

### 1) Structural analysis
- Do phonological analogies exist?
- Are there linear directions in the representation space?

### 2) Controlled manipulation
- Does scaling a vector change actual speech acoustics in a predictable way?

So the paper is not just a probing study; it is also a **representation manipulation and synthesis** study.

---

## 13. One-sentence summary
In short, the paper extracts **layerwise S3M representations**, uses **PanPhon** to build **phonological analogies**, validates them with **cosine-similarity-based vector arithmetic**, and then uses **Vocos-based speech resynthesis** to show that scaling those vectors continuously controls real acoustic phonological cues.

---



<br/>
# Results




## 1. 이 논문이 무엇을 보려는가
이 논문은 self-supervised speech model(S3M), 즉 **wav2vec 2.0, HuBERT, WavLM** 같은 음성 자기지도 모델이 단순히 “음성 인식에 필요한 정보”만 담는 것이 아니라,  
**음운론적 특징(phonological features)** 을 **선형적으로 조합 가능한 벡터 구조**로 표현하는지 확인합니다.

핵심 질문은 두 가지입니다.

1. **방향(Direction)**:  
   모델 표현 공간 안에 특정 음운론적 차이를 나타내는 **벡터 방향**이 존재하는가?  
   예: `[d] - [t]`가 voicing vector라면, `[p] + ([d]-[t]) ≃ [b]`가 되는가?

2. **크기(Scale)**:  
   그 벡터의 크기 λ를 조절하면, 해당 음운론적 특징이 **연속적으로** 강화/약화되는가?

---

## 2. 실험 데이터셋
논문은 두 개의 음성-음운 주석 데이터셋을 사용합니다.

### (1) TIMIT
- 영어 음성 데이터
- 630명 화자
- 음소와 세그먼트 정보가 수동 정렬됨
- 영어 내부에서의 분석에 사용

### (2) VoxAngeles
- UCLA Phonetics Archive 기반
- **95개 언어**, **21개 어족**
- 영어에 없는 phone도 포함
- **교차언어 일반화**를 보기 위한 데이터셋

즉,
- **TIMIT**은 영어 중심 평가
- **VoxAngeles**는 영어 밖으로 일반화되는지 확인

---

## 3. 경쟁 모델(비교 대상)
논문은 S3M들을 서로 비교하고, 또 전통적인 음향 특징과도 비교합니다.

### 3.1 Self-supervised speech models
1. **wav2vec 2.0**
2. **HuBERT**
3. **WavLM**

이들은 모두 **large configuration**을 사용했고,
- 7개 CNN layer
- 24개 Transformer block
- 총 25개 layer-wise representation을 추출

### 3.2 Spectral baseline
1. **MFCC**
2. **MelSpec**

즉, 논문은
- **학습된 S3M 표현**
vs
- **전통적 음향 표현**
을 비교합니다.

---

## 4. 실험 1: Phonological vector arithmetic의 “방향” 검증

## 4.1 실험 목적
음운론적 analogical relation이 S3M 표현에서 실제로 성립하는지 보려는 것입니다.

예를 들어:
- voicing: `[b]:[p] = [d]:[t]`
- place of articulation(POA): `[b]:[d] = [p]:[t]`

이런 관계가 벡터 연산으로 표현되면:
- `r[b] ≃ r[p] + (r[d] - r[t])`
- `r[b] ≃ r[d] + (r[p] - r[t])`

즉, 특정 음소 차이를 나타내는 **선형 방향**이 존재하는지를 확인합니다.

---

## 4.2 테스트 데이터와 쿼드러플 구성
논문은 PanPhon을 이용해 음소의 음운론적 특징을 추출하고,  
특징 차이가 정확히 맞는 **4개 음소 조합(quadruplet)** 을 만듭니다.

예:
- `p1 : p2 = p3 : p4`

조건은 PanPhon feature vector의 차이가 같아야 함:
- `h_p1 - h_p2 = h_p3 - h_p4`

이렇게 만들어진 quadruplet을 이용해 analogy를 평가합니다.

### 사용된 quadruplet 수
- **TIMIT test set**: 236개 quadruplet
- **VoxAngeles full set**: 468개 quadruplet

주의:
- 영어 phone set에 없는 phone이 포함될 수도 있음
- 각 phone이 최소 50회 이상 등장하는 경우만 사용
- 최종적으로 19개의 phonological feature를 테스트

---

## 4.3 평가 메트릭
이 실험의 핵심 메트릭은 **cosine similarity**와 **success rate**입니다.

### (1) cosine similarity
quadruplet \(p=(p1,p2,p3,p4)\)에 대해
\[
\cos(p) = E[\cos(r_{p1}, r_{p2}+r_{p3}-r_{p4})]
\]
를 계산합니다.

즉,
- `p2 + p3 - p4`가
- `p1`에 얼마나 가까운지
확인합니다.

### (2) baseline 비교
두 baseline과 비교합니다.

#### a. same-phone baseline
- 같은 phone의 다른 발화끼리 비교
- 상한선 역할
\[
\cos^+(p)=E[\cos(r_{p1}, r'_{p1})]
\]

#### b. different-phone baseline
- 다른 phone과의 비교
- 하한선 역할
\[
\cos^-(p)=E[\cos(r_{p1}, r_{\text{not }p1})]
\]

### (3) 성공 조건
논문은 다음 순서가 성립하면 성공으로 봅니다.
\[
\cos^-(p) < \cos(p) < \cos^+(p)
\]

즉,
- 무작위 다른 phone보다 더 가깝고
- 같은 phone 다른 예시보다는 덜 가깝다면
정상적인 phonological analogy가 성립한다고 본 것입니다.

### (4) success rate
전체 quadruplet 중 위 조건을 만족하는 비율:
\[
S(Q)=\frac{1}{|Q|}\sum_{p\in Q}1[\cos^-<\cos<\cos^+]
\]

---

## 4.4 결과: 무엇과 무엇을 비교했는가
논문은 다음을 비교합니다.

### 비교 1: S3Ms vs spectral features
- wav2vec 2.0
- HuBERT
- WavLM
vs
- MFCC
- MelSpec

### 비교 2: layer-wise 비교
각 S3M의 0~24 layer를 모두 비교

### 비교 3: TIMIT vs VoxAngeles
- 영어 내부 성능
- 교차언어 generalization

---

## 4.5 결과 요약
### TIMIT
- **HuBERT 마지막 layer: 94%**
- **WavLM 마지막 layer: 92%**
- **wav2vec 2.0 중간 layer: 61%**
- **MFCC: 19%**
- **MelSpec: 0%**

즉, S3M이 spectral baseline보다 훨씬 우수합니다.

### VoxAngeles
- **WavLM: 93%**
- **HuBERT: 45%**
- **wav2vec 2.0: 39%**
- **MFCC: 19%**
- **MelSpec: 0%**

즉, 영어로만 학습된 S3M도 **다른 언어 phone**에 대해 phonological structure를 꽤 잘 일반화합니다.

---

## 4.6 layer-wise 결과 해석
세 모델의 layer-wise 패턴이 다릅니다.

### wav2vec 2.0
- 중간 layer에서 성능 최고
- 마지막 layer에서는 오히려 덜 좋을 수 있음

### HuBERT / WavLM
- 마지막 layer에서 최고 성능

이 논문은 이를 다음처럼 해석합니다.
- 깊은 layer일수록 더 많은 문맥(context)을 반영
- phonological abstraction이 문맥 정보와 결합되며 형성됨

---

## 4.7 모음 vs 자음
논문은 WavLM을 이용해 음운론적 analogies를 모음과 자음으로 나눠 분석합니다.

### 결과
- **모음**: 비교적 이른 layer에서 peak
- **자음**: 더 복잡한 layer-wise behavior
- 하지만 둘 다 마지막 layer에서는 강함

해석:
- 모음은 시간적으로 더 국소적인 단서로도 판단 가능
- 자음은 주변 문맥이 더 중요

---

## 5. 실험 2: Phonological vector의 “크기(scale)” 검증

## 5.1 실험 목적
벡터 방향만 존재하는지 보는 것이 아니라,  
그 벡터에 스칼라 λ를 곱했을 때:
- 해당 phonological feature가 **연속적으로** 변하는지
확인합니다.

즉,
\[
r[b] \simeq r[p] + \lambda(r[d]-r[t])
\]
에서 λ를 조절하면 voicing이 얼마나 강해지는지 보는 것입니다.

---

## 5.2 실험 설정
논문은 phonological vector를 다음처럼 정의합니다.

\[
v_i = E[r \mid h_i=+1] - E[r \mid h_i=-1]
\]

즉,
- 해당 특징이 있는 phone들의 평균 representation
- 없는 phone들의 평균 representation
의 차이입니다.

### 테스트한 8개 phonological vector
- vowels:
  - high
  - low
  - back
  - round
- consonants:
  - nasal
  - sonorant
  - strident
  - voice

---

## 5.3 평가 방법
WavLM 마지막 layer representation에 벡터를 더한 뒤,
vocoder로 재합성하고,
그 결과 음성에서 acoustic measurement를 측정합니다.

### 사용한 acoustic measurements
- **F1**: vowel height
- **F2**: backness / roundness
- **F1 bandwidth**: nasality
- **HNR**: sonority
- **COG**: voicing / stridency

그리고 λ와 측정치 변화 사이의 상관을 **Spearman correlation**으로 봅니다.

---

## 5.4 결과
### 핵심 결과
모든 feature에서:
- 상관 방향이 이론적 예측과 일치
- λ가 커질수록 acoustic cue가 **단조롭게(monotonic)** 변화
- 변화는 선형적일 필요는 없지만, **연속적**임

즉,
- voicing vector를 더하면 voicing이 갑자기 on/off 되는 것이 아니라,
  점진적으로 바뀜
- rounding vector를 더하면 formant가 점차 내려감
- strident vector를 더하면 고주파 frication이 증가

### 예시
- **round vector**: F1/F2/F3가 낮아짐
- **voicing vector**: VOT가 짧아짐, voicing onset이 빨라짐
- **strident vector**: burst가 사라지고 high-frequency energy 증가
- **nasal vector**: burst가 약해지고 low-frequency murmur 증가

---

## 5.5 extrapolation
흥미롭게도
- \(|\lambda| \le 1\) 뿐 아니라
- \(|\lambda| > 1\)에서도

언어학적으로 해석 가능한 출력이 나왔습니다.

즉, vector arithmetic가 단순한 interpolation이 아니라  
어느 정도 **extrapolation**까지 가능한 구조입니다.

---

## 6. 비교의 핵심 포인트 정리

## 6.1 모델 비교
- **S3Ms**: phonological analogy 성공률 높음
- **MFCC / MelSpec**: 매우 낮음

즉, 단순 음향 특징보다 self-supervised representation이 phonological structure를 더 잘 담고 있음.

## 6.2 데이터셋 비교
- **TIMIT**: 영어 내부 검증
- **VoxAngeles**: 다국어 일반화
- 영어만 학습한 모델도 unseen phone에 대해 상당히 잘 작동

## 6.3 metric 비교
- Direction 실험: **cosine similarity**, **success rate**
- Scale 실험: **Spearman correlation**
- 일부 부가 실험: **PCS(AUC)**

---

## 7. 논문의 결론을 한 줄로 요약하면
이 논문은 **S3M 표현 공간 안에 phonological feature를 나타내는 선형 벡터가 존재하며, 그 벡터의 크기는 실제 발화의 연속적 음향 정도를 조절한다**는 것을 보였습니다.

즉,  
self-supervised speech model이 음성을 **카테고리적이면서도 연속적인 phonological space**로 내부 표현한다는 주장입니다.

---




## 1. What the paper investigates
This paper asks whether self-supervised speech models (S3Ms) such as **wav2vec 2.0, HuBERT, and WavLM** encode phonological structure in a **linearly compositional** way, similar to word analogies in text embeddings.

The paper studies two main questions:

1. **Direction**:  
   Do there exist vector directions in the representation space that correspond to phonological features?  
   Example: does `[d] - [t]` act as a voicing vector so that `[p] + ([d]-[t]) ≃ [b]`?

2. **Scale**:  
   If we scale such vectors by a scalar λ, does the strength of the corresponding phonological feature vary continuously?

---

## 2. Datasets
The paper evaluates on two manually segmented phonetic datasets.

### (1) TIMIT
- English speech corpus
- 630 speakers
- Used for English-specific evaluation

### (2) VoxAngeles
- Based on the UCLA Phonetics Archive
- 95 languages across 21 language families
- Includes phones not present in English
- Used to test **cross-linguistic generalization**

So:
- **TIMIT** = English-focused analysis
- **VoxAngeles** = multilingual generalization test

---

## 3. Competing models
The paper compares S3Ms against traditional acoustic features.

### 3.1 Self-supervised speech models
1. **wav2vec 2.0**
2. **HuBERT**
3. **WavLM**

All are evaluated using the **LARGE** configuration:
- 7 CNN layers
- 24 Transformer blocks
- 25 layer-wise representations extracted

### 3.2 Spectral baselines
1. **MFCC**
2. **MelSpec**

Thus, the comparison is:
- **learned S3M representations**
vs.
- **traditional spectral representations**

---

## 4. Experiment 1: Direction of phonological vectors

## 4.1 Goal
The paper tests whether phonological analogies hold in the representation space.

Example:
- voicing: `[b]:[p] = [d]:[t]`
- POA: `[b]:[d] = [p]:[t]`

This yields vector equations such as:
- `r[b] ≃ r[p] + (r[d] - r[t])`
- `r[b] ≃ r[d] + (r[p] - r[t])`

So the paper asks whether S3M representations contain **linear directions** corresponding to phonological features.

---

## 4.2 Quadruplet construction
Using PanPhon, the authors extract binary phonological features for each phone and construct quadruplets `(p1, p2, p3, p4)` such that:
\[
h_{p1} - h_{p2} = h_{p3} - h_{p4}
\]

This creates symmetric phonological analogies:
- `p1:p2 = p3:p4`

### Number of quadruplets
- **TIMIT**: 236 quadruplets
- **VoxAngeles**: 468 quadruplets

They test **19 phonological features** in total.

---

## 4.3 Metrics
The main metric is **cosine similarity**, together with a **success rate**.

For a quadruplet \(p=(p1,p2,p3,p4)\):
\[
\cos(p) = E[\cos(r_{p1}, r_{p2}+r_{p3}-r_{p4})]
\]

### Baselines
#### same-phone baseline
Similarity between two different instances of the same phone:
\[
\cos^+(p)=E[\cos(r_{p1}, r'_{p1})]
\]

#### different-phone baseline
Similarity to an unrelated phone:
\[
\cos^-(p)=E[\cos(r_{p1}, r_{\text{not }p1})]
\]

### Success criterion
A quadruplet is counted as successful if:
\[
\cos^-(p) < \cos(p) < \cos^+(p)
\]

### Success rate
The proportion of quadruplets satisfying the above ordering.

---

## 4.4 What is compared
The paper compares:
- **S3Ms vs spectral features**
- **different layers** within each S3M
- **TIMIT vs VoxAngeles**

---

## 4.5 Main results
### On TIMIT
- **HuBERT last layer: 94%**
- **WavLM last layer: 92%**
- **wav2vec 2.0 middle layer: 61%**
- **MFCC: 19%**
- **MelSpec: 0%**

### On VoxAngeles
- **WavLM: 93%**
- **HuBERT: 45%**
- **wav2vec 2.0: 39%**
- **MFCC: 19%**
- **MelSpec: 0%**

This shows that S3Ms strongly outperform spectral baselines and generalize to unseen phones/languages.

---

## 4.6 Layer-wise behavior
The models behave differently by layer:

### wav2vec 2.0
- Peaks in the middle layers

### HuBERT / WavLM
- Peak in the final layer

The authors interpret this as evidence that deeper layers integrate more contextual information to form abstract phonological structure.

---

## 4.7 Vowels vs. consonants
The authors also split analogies into vowels and consonants.

### Findings
- **Vowels** peak earlier
- **Consonants** show more complex layer-wise patterns
- Both peak in the final layer

Interpretation:
- Vowel cues are more temporally localized
- Consonant cues often require broader context

---

## 5. Experiment 2: Scale of phonological vectors

## 5.1 Goal
The second experiment asks whether the scalar λ controls the strength of a phonological feature continuously.

That is, if we use:
\[
r[b] \simeq r[p] + \lambda(r[d]-r[t])
\]
does changing λ smoothly modulate voicing?

---

## 5.2 Vector definition
The phonological vector for feature \(i\) is defined as:
\[
v_i = E[r \mid h_i=+1] - E[r \mid h_i=-1]
\]

The paper tests eight vectors:
- vowels: **high, low, back, round**
- consonants: **nasal, sonorant, strident, voice**

---

## 5.3 Evaluation
The authors modify WavLM final-layer representations, resynthesize speech with a vocoder, and then measure acoustic correlates.

### Acoustic measurements
- **F1** for height
- **F2** for backness/roundness
- **F1 bandwidth** for nasality
- **HNR** for sonority
- **COG** for voicing and stridency

They then compute **Spearman rank correlation** between λ and the acoustic change.

---

## 5.4 Results
The signs of the observed correlations match linguistic expectations for all tested features.

Key result:
- vector scaling induces **monotonic** and **continuous** changes
- the effect is not binary
- extrapolation beyond \(|\lambda| \le 1\) often remains interpretable

Examples:
- rounding lowers formants
- voicing shifts COG and VOT
- stridency increases high-frequency frication
- nasality introduces low-frequency murmur

---

## 6. Overall comparison summary
### Models
- **S3Ms**: strong phonological compositionality
- **MFCC/MelSpec**: weak or absent phonological arithmetic

### Datasets
- **TIMIT**: English evaluation
- **VoxAngeles**: cross-linguistic generalization

### Metrics
- **Direction experiment**: cosine similarity, success rate
- **Scale experiment**: Spearman correlation
- Additional analysis: PCS/AUC in the appendix

---

## 7. One-sentence takeaway
The paper shows that self-supervised speech models encode phonological features as **linearly composable and scalable vectors**, and that these vectors correspond to **continuous acoustic control** in resynthesized speech.

---


<br/>
# 예제




이 논문은 자기지도학습 음성모델(Self-supervised Speech Models, S3Ms)이 **음운론적 특징(phonological features)** 을 선형 벡터로 담고 있는지를 보입니다.  
즉, 단어 임베딩에서 유명한

- king - man + woman ≈ queen

같은 관계가, 음성에서는

- [d] - [t] + [p] ≈ [b]

처럼 나타나는지 확인한 것입니다.

논문 전체의 핵심은 크게 두 가지입니다.

1. **방향(Direction)**: 모델 내부에 음운 특징을 나타내는 선형 방향이 있는가?
2. **스케일(Scale)**: 그 벡터의 크기를 조절하면 음향적 특징도 연속적으로 변하는가?

아래에서 이걸 실제 실험 입력/출력 관점으로 자세히 설명하겠습니다.

---

## 1. 데이터셋: 무엇을 입력으로 썼나?

논문은 두 개의 음성 데이터셋을 사용했습니다.

### 1) TIMIT
- **영어 음성 데이터**
- 630명의 화자
- 발화마다 **음소 수준의 정밀한 구간 정보(phonetic segmentation)** 가 있음
- 즉, 각 음성 파일에서
  - 어느 구간이 [p]인지
  - 어느 구간이 [t]인지
  - 어느 구간이 모음인지
  를 알고 있음

### 2) VoxAngeles
- UCLA Phonetics Archive 기반
- **95개 언어**에 걸친 더 다양한 다국어 데이터
- 영어에 없는 음소도 포함 가능
- 즉, 영어에만 맞는 현상인지, 다른 언어로 일반화되는지도 확인 가능

---

## 2. Experiment 1: 음운 벡터의 “방향”을 찾는 실험

이 실험은 간단히 말하면:

> “모델 안에서 [d] - [t] 같은 차이가 실제로 ‘voicing(유성성)’ 방향을 나타내는가?”

를 보는 것입니다.

---

### 2.1 구체적인 입력/출력 구조

#### 입력
- 음성 파형 \(x\)
- 이 파형을 S3M에 넣으면 프레임 단위 hidden representation \(R=f(x)\)가 나옴

#### 출력
- 각 음소 구간을 대표하는 벡터 \(r_p\)
- 그리고 이 벡터들 사이의 유사도(cosine similarity)

즉, 모델의 출력은 **음성 파일 자체가 아니라 벡터 표현**입니다.

---

### 2.2 어떤 “테스크”를 했나?

논문은 음운 특징에 따라 네 개의 음소를 묶어서 **유추(analogy)** 를 만듭니다.

예:
- [b], [p], [d], [t]

여기서
- [b] ↔ [p] 는 유성/무성 차이
- [d] ↔ [t] 도 유성/무성 차이

그래서 아래와 같은 관계가 성립하는지 봅니다.

- [b] : [p] = [d] : [t]
- 또는 벡터로 보면  
  \(r_b \approx r_p + (r_d - r_t)\)

즉,
- **[d] - [t] = voicing vector**
- 여기에 [p]를 더하면 [b]가 되는지 확인

---

### 2.3 실제로 어떻게 계산했나?

논문은 음소 4개로 이루어진 quadruplet을 만들고, 다음을 계산합니다.

- \(r_{p1}\)
- \(r_{p2} + r_{p3} - r_{p4}\)

그리고 둘의 cosine similarity를 봅니다.

예를 들어:
- \(r_b\) 와 \(r_p + r_d - r_t\) 가 얼마나 비슷한지 확인

이게 높으면:
- 모델 내부에 “유성성” 방향이 선형적으로 존재한다고 해석합니다.

---

### 2.4 baseline은 무엇인가?

논문은 단순히 similarity가 높다고 끝내지 않고, 두 기준과 비교합니다.

#### (1) same-phone baseline
같은 음소를 다른 발화에서 뽑은 벡터끼리 비교
- 예: [p]의 다른 샘플끼리
- 이것은 상한선에 가까운 기준

#### (2) different-phone baseline
아무 관련 없는 다른 음소와 비교
- 예: [p] vs [m]
- 이것은 하한선에 가까운 기준

기대되는 순서는:

- different-phone similarity < analogy similarity < same-phone similarity

즉, analogy가 진짜 구조를 반영한다면  
무작위보다 높고, 같은 음소만큼 높지는 않아야 합니다.

---

### 2.5 실제 결과는?

- S3Ms(HuBERT, WavLM, wav2vec 2.0)가
  - MFCC
  - MelSpec
  보다 훨씬 좋았음
- 특히 WavLM 마지막 층이 가장 좋음
- 영어에서 학습된 모델도 VoxAngeles의 **보지 못한 음소**에 대해 꽤 잘 작동함

즉, 입력은 영어 음성으로 학습된 representation이지만,  
출력은 **음운적으로 해석 가능한 벡터 조합**이었습니다.

---

## 3. Experiment 2: 음운 벡터의 “스케일”을 조절하는 실험

이 실험은 더 흥미롭습니다.

> 벡터 방향만 있는 게 아니라, 그 벡터를 얼마나 세게 더하느냐에 따라 음향이 연속적으로 변하는가?

를 봅니다.

---

### 3.1 구체적인 입력/출력

#### 입력
- 음성 구간의 representation \(R\)
- 예: 어떤 발화 안의 [b] 구간
- 여기서 phonological vector \(v\) 를 더함
- 예: voicing vector, roundness vector 등

즉
- 원래 representation \(R\)
- 수정된 representation \(\tilde{R} = R + \lambda v\)

여기서 \(\lambda\) 는 스케일입니다.

#### 출력
- vocoder를 통해 다시 합성한 음성 \(\tilde{x}\)
- 그 음성에서 측정한 acoustic features
  - F1
  - F2
  - F1 bandwidth
  - HNR
  - COG 등

---

### 3.2 어떤 “테스크”였나?

이건 사실상 **representation editing / controlled resynthesis** 테스크입니다.

즉:

1. 음성 샘플을 representation으로 바꿈
2. 특정 음운 벡터를 더함
3. 다시 음성으로 복원
4. 복원된 음성이 정말 그 음운 특징을 더 강하게/약하게 갖는지 측정

---

### 3.3 예시

#### 예시 1: voicing vector
- 대상: [b]
- 벡터: voicing vector
- \(\lambda > 0\) 이 커질수록
  - voice onset time(VOT)이 줄어듦
  - 더 유성적으로 변함
- \(\lambda < 0\) 이면
  - 더 무성적인 방향으로 변함

즉, [b]에 유성 벡터를 세게 넣으면  
더 빨리 성대 진동이 시작되는 방향으로 이동합니다.

---

#### 예시 2: round vector
- 대상: [i]
- 벡터: roundness vector
- \(\lambda > 0\) 이면
  - F1, F2, F3가 낮아지는 경향
  - 입술 둥글림의 음향적 특징이 나타남

즉, 영어에는 없는 front rounded vowel 같은 효과도 어느 정도 만들어냄.

---

#### 예시 3: strident vector
- 대상: [b]
- \(\lambda > 0\) 이면
  - 4~8kHz 부근의 마찰음 에너지가 증가
  - burst는 줄어듦
- 즉 파열음이 마찰음처럼 바뀌는 경향

---

#### 예시 4: nasal vector
- 대상: [b]
- \(\lambda > 0\) 이면
  - burst 감소
  - 저주파 nasal murmur 증가

---

### 3.4 이 실험의 핵심 출력

논문은 \(\lambda\) 와 acoustic measurements 간 Spearman 상관을 측정했습니다.

즉,
- 입력: \(\lambda\) 값
- 출력: resynthesized speech에서 측정한 acoustic value

그리고 둘이 단조롭게 변하는지 본 것입니다.

결과:
- 거의 모든 특징에서 이론적으로 기대한 방향의 상관이 나옴
- 연속적 변화가 관찰됨
- 단순한 binary on/off가 아니라 **gradient**가 있음

---

## 4. 논문에서 사용한 phonological feature 예시

논문은 PanPhon의 특징 중 일부를 사용했습니다.

### 방향 실험(Experiment 1)
19개 phonological features:
- syllabic
- sonorant
- continuant
- delayed release
- lateral
- nasal
- strident
- voice
- spread glottis
- anterior
- coronal
- distributed
- labial
- high
- low
- back
- round
- tense
- long

### 스케일 실험(Experiment 2)
그중 acoustic measurement로 직접 측정 가능한 8개:
- high
- low
- back
- round
- nasal
- sonorant
- strident
- voice

---

## 5. 트레이닝 데이터와 테스트 데이터는 어떻게 나뉘었나?

### Experiment 1
- **TIMIT test split**
- **VoxAngeles 전체**

이유:
- phonological analogy가 보이는지 평가하기 위함
- 그리고 영어 외 언어로 일반화되는지 보려는 것

### Experiment 2
- WavLM의 마지막 층 representation 사용
- vocoder는 train/test로 나눠 resynthesis
- train:
  - LibriTTS 기반 vocoder
  - 또는 FLEURS-R 기반 multilingual vocoder
- test:
  - TIMIT test split
  - VoxAngeles의 남은 언어들

즉, representation은 S3M에서 오고, 합성은 vocoder가 담당합니다.

---

## 6. 아주 간단한 “입력-출력” 요약

### 실험 1
- **입력**: 음성 파일 + 음소 구간 정보
- **모델 출력**: 음소 벡터 representation
- **태스크**: [d] - [t] + [p] ≈ [b] 같은 음운 유추가 성립하는지 확인

### 실험 2
- **입력**: 음성 representation + phonological vector + 스케일 \(\lambda\)
- **모델 출력**: 수정된 representation → resynthesized speech
- **태스크**: 음성 특징이 연속적으로 바뀌는지 확인

---

## 7. 한 문장으로 정리하면

이 논문은  
**“자기지도 음성모델은 음소를 단순히 구분만 하는 것이 아니라, voicing/rounding/nasality 같은 음운 특징을 방향과 크기를 가진 벡터로 표현하며, 그 벡터를 더하거나 조절하면 실제 음향도 예측 가능한 방식으로 변한다”**  
는 것을 보여줍니다.

---





## 1. What is the main idea of the paper?

The paper investigates whether self-supervised speech models (S3Ms) represent phonological features in a **linear and compositional** way.

The analogy is similar to word embeddings:

- king - man + woman ≈ queen

In speech, the authors test whether relations like:

- [d] - [t] + [p] ≈ [b]

hold inside the model’s representation space.

The paper mainly studies two things:

1. **Direction**: Do phonological features correspond to linear directions in the representation space?
2. **Scale**: If we scale those directions, do acoustic properties change continuously?

---

## 2. What datasets were used as input?

The paper uses two annotated speech datasets.

### 1) TIMIT
- English speech corpus
- 630 speakers
- Phone-level manual segmentation
- Each utterance has precise phone boundaries

### 2) VoxAngeles
- Based on the UCLA Phonetics Archive
- Covers 95 languages
- Includes much broader phonetic diversity
- Useful for testing generalization beyond English

So the raw input is **speech audio**, plus phone boundary annotations.

---

## 3. Experiment 1: Finding the “direction” of phonological vectors

This experiment asks:

> Is the difference [d] - [t] really a “voicing” direction in the model?

---

### 3.1 Input and output

#### Input
- speech waveform \(x\)
- fed into a self-supervised speech model
- the model produces a frame-level representation matrix \(R=f(x)\)

#### Output
- a phone vector \(r_p\) for each segmented phone
- cosine similarities between vectors

So the model output is **not text or audio directly**, but a latent representation.

---

### 3.2 What is the task?

The task is a **phonological analogy test**.

Example quadruplet:
- [b], [p], [d], [t]

The authors test whether the following relation holds:

- [b] : [p] = [d] : [t]

or equivalently in vector form:

- \(r_b \approx r_p + (r_d - r_t)\)

This means:
- \(r_d - r_t\) is interpreted as a **voicing vector**
- adding that vector to [p] should bring you close to [b]

---

### 3.3 How is it evaluated?

For each quadruplet, they compute:

- the cosine similarity between \(r_{p1}\) and
  \(r_{p2} + r_{p3} - r_{p4}\)

Example:
- compare \(r_b\) with \(r_p + r_d - r_t\)

They also compare this with two baselines:

#### Same-phone baseline
- compare the same phone across different utterances
- gives an upper bound

#### Different-phone baseline
- compare unrelated phones
- gives a lower bound

Expected ordering:

- different-phone similarity < analogy similarity < same-phone similarity

If this holds, the representation is considered phonologically structured.

---

### 3.4 Main result

S3Ms such as WavLM, HuBERT, and wav2vec 2.0 outperform MFCC and MelSpec by a large margin.

Also:
- English-trained models generalize to unseen phones in VoxAngeles
- The strongest performance often appears in middle or final layers

So the output of this experiment is a **validated phonological vector arithmetic effect**.

---

## 4. Experiment 2: Controlling the “scale” of phonological vectors

This experiment asks:

> If we add a phonological vector with strength \(\lambda\), does the speech change gradually?

---

### 4.1 Input and output

#### Input
- a phone segment representation \(R\)
- a phonological vector \(v\)
- a scalar \(\lambda\)

The modified representation is:

- \(\tilde{R} = R + \lambda v\)

#### Output
- resynthesized speech \(\tilde{x}\)
- acoustic measurements computed from \(\tilde{x}\)

The output is therefore **modified speech audio** plus measurable acoustic cues.

---

### 4.2 What is the task?

This is a **controlled resynthesis / representation editing** task.

Procedure:

1. Take speech audio
2. Extract the model representation
3. Add a scaled phonological vector
4. Decode back into speech with a vocoder
5. Measure whether the acoustic properties changed as expected

---

### 4.3 Examples

#### Example 1: Voicing vector
- target phone: [b]
- vector: voicing vector
- larger positive \(\lambda\):
  - earlier voicing onset
  - reduced VOT
- negative \(\lambda\):
  - more voiceless-like behavior

#### Example 2: Roundness vector
- target phone: [i]
- larger positive \(\lambda\):
  - lowering of formants
  - acoustic pattern consistent with lip rounding

#### Example 3: Strident vector
- target phone: [b]
- larger positive \(\lambda\):
  - more high-frequency frication
  - burst characteristics weaken

#### Example 4: Nasal vector
- target phone: [b]
- larger positive \(\lambda\):
  - burst weakens
  - nasal murmur appears

---

### 4.4 How is this evaluated?

They compute Spearman correlation between:
- the vector scale \(\lambda\)
- the resulting acoustic measurements

The key point is that the changes are:
- monotonic
- interpretable
- continuous rather than binary

---

## 5. What phonological features were tested?

### In Experiment 1
19 PanPhon features, including:
- syllabic
- sonorant
- continuant
- delayed release
- lateral
- nasal
- strident
- voice
- spread glottis
- anterior
- coronal
- distributed
- labial
- high
- low
- back
- round
- tense
- long

### In Experiment 2
8 features with clear acoustic correlates:
- high
- low
- back
- round
- nasal
- sonorant
- strident
- voice

---

## 6. How were train and test data used?

### For Experiment 1
- TIMIT test split
- full VoxAngeles set

Used to test:
- whether phonological analogies hold
- whether the model generalizes to unseen phones/languages

### For Experiment 2
- WavLM final-layer representations
- vocoder trained separately on LibriTTS or FLEURS-R
- test utterances from held-out splits/languages

Used to test:
- whether scaling phonological vectors changes the resynthesized speech in a predictable way

---

## 7. Short summary

In short, the paper shows that self-supervised speech models do not merely encode phones as discrete classes.  
They appear to encode phonological properties such as **voicing, rounding, nasality, and stridency** as:

- **linear directions** in representation space
- **continuous scalar controls** that affect the acoustics of resynthesized speech

---




<br/>
# 요약
메써드: WavLM 같은 자기지도 음성모델의 phone 표현에서 PanPhon 특징으로 음운적 사분조각(예: [b]:[p]=[d]:[t])을 만들고, \(r_{p1}\approx r_{p2}+r_{p3}-r_{p4}\) 형태의 선형 유추가 성립하는지 코사인 유사도로 검증했다.  
결과: 19개 음운 특징 전반에서 S3M이 MelSpec/MFCC보다 훨씬 높은 성공률을 보였고, 특히 WavLM의 마지막 층은 TIMIT에서 약 92%, VoxAngeles에서 약 93% 수준까지 도달했으며, 예를 들어 \([d]-[t]\)로 얻은 voicing vector를 \([p]\)에 더하면 \([b]\)가 되는 식의 음운 벡터 산술이 확인됐다.  
예시: 벡터의 크기 \(\lambda\)를 조절하면 성질도 연속적으로 바뀌어, voicing을 키우면 VOT가 앞당겨지고, rounding을 키우면 F1/F2/F3가 낮아지며, stridency나 nasality를 키우면 burst 감소·고주파 마찰음/비강음 같은 음향 변화가 점진적으로 나타났다.  




Method: The paper constructs phonological quadruplets from PanPhon features in phone representations from self-supervised speech models like WavLM, then tests whether linear analogies of the form \(r_{p1}\approx r_{p2}+r_{p3}-r_{p4}\) hold using cosine similarity.  
Result: Across 19 phonological features, S3Ms outperform MelSpec/MFCC by a large margin; WavLM’s final layer reaches about 92% success on TIMIT and 93% on VoxAngeles, confirming phonological vector arithmetic such as adding the voicing vector \([d]-[t]\) to \([p]\) to obtain \([b]\).  
Example: By scaling the vector with \(\lambda\), the realized feature changes continuously—for instance, stronger voicing shifts VOT earlier, rounding lowers formants (F1/F2/F3), and increased stridency or nasality gradually introduces frication, weaker bursts, or nasal murmur.

<br/>
# 기타




# 한국어 요약: 기타 요소별 결과와 인사이트

## 1) Figure 1: 텍스트 word analogy와 speech phonological analogy 비교
### 결과
- word2vec처럼 텍스트 임베딩이  
  **king - man + woman ≈ queen** 같은 의미적 벡터 연산을 보이듯,
- self-supervised speech model(S3M)도  
  **[p] - [t] + [d] ≈ [b]** 같은 **음운론적 벡터 연산**을 보이는지 질문함.

### 인사이트
- 이 논문의 핵심 문제의식 자체를 보여주는 그림입니다.
- 텍스트 임베딩의 “의미 벡터 산술”을 speech로 옮겨와서,
  **speech representation도 인간이 이해 가능한 음운 특징을 선형적으로 담을 수 있다**는 가설을 제시합니다.
- 즉, S3M은 단순히 소리를 구분하는 것이 아니라,
  **voicing, place of articulation 같은 음운 특징을 방향 벡터로 표현**할 수 있다는 출발점입니다.

---

## 2) Figure 2: S3Ms vs. spectral representations on phonological analogies
### 결과
- S3M들(wav2vec 2.0, HuBERT, WavLM)은
  MFCC나 MelSpec보다 **훨씬 높은 success rate**를 보임.
- 특히:
  - **HuBERT 마지막 레이어: 94%**
  - **WavLM 마지막 레이어: 92%**
  - **wav2vec 2.0 중간 레이어: 61%**
- 반면 spectral features:
  - MFCC: **19%**
  - MelSpec: **0%**
- 언어별로는 TIMIT뿐 아니라 VoxAngeles에서도 비슷한 경향.

### 인사이트
- **자기지도 speech 모델은 단순한 스펙트럼 특징보다 음운적 관계를 더 잘 구조화**하고 있음.
- 특히 **깊은 레이어로 갈수록 음운적 추상화가 강화**되는 경향이 드러남.
- WavLM/HuBERT가 wav2vec 2.0보다 좋게 나온 것은,
  모델 학습 방식이 더 phonological structure를 잘 포착하게 만들 수 있음을 시사합니다.
- 또한 **영어로만 학습한 모델도 다른 언어의 phone에 대해 일반화 가능**하다는 점이 중요합니다.

---

## 3) Figure 3: Vowels vs. Consonants layerwise trends
### 결과
- 모음(vowels)과 자음(consonants)의 layerwise peak가 다름.
- WavLM 기준:
  - 모음은 **초중간층에서 먼저 peak**
  - 자음은 **더 뒤쪽 층에서 peak**
  - 둘 다 **final layer에서 최고 성능**
- 즉, 서로 다른 음운 단위가 서로 다른 층에서 더 잘 구조화됨.

### 인사이트
- **모음은 상대적으로 국소적 단서에 의존**
- **자음은 주변 맥락(temporal context)을 더 많이 필요로 함**
- 그래서 S3M의 deeper layer가 더 넓은 문맥을 반영할 수 있고,
  자음 관련 특징은 더 깊은 층에서 잘 드러난다는 해석이 가능합니다.
- 이 결과는 S3M이 **한 층에 모든 언어학적 정보가 다 들어 있는 것이 아니라, 층별로 다른 수준의 음운 정보를 분화해서 담는다**는 점을 보여줍니다.

---

## 4) Figure 4: phonological vector scale λ와 acoustic measurements
### 결과
- 8개 음운 특징(high, low, back, round, nasal, sonorant, strident, voice)에 대해
  λ를 변화시키면 **해당 acoustic measurement가 단조롭게 변함**.
- Spearman 상관의 방향이 **이론적 예측과 정확히 일치**.
- 예:
  - high ↔ F1
  - back/round ↔ F2
  - nasal ↔ F1 bandwidth
  - voicing/strident ↔ COG
  - sonorant ↔ HNR

### 인사이트
- 핵심은 **벡터 방향(direction)**만이 아니라 **벡터 크기(scale)**도 의미를 가진다는 점입니다.
- 즉, 음운 특징이 binary on/off로만 표현되는 것이 아니라,
  **연속적인 정도(continuum)**로 표현될 수 있음을 보여줍니다.
- 이것은 phonological feature를 **scalar feature**처럼 다룰 수 있음을 시사합니다.
- 특히 speech synthesis에서 **정밀한 조절 가능성**을 열어줍니다.

---

## 5) Figure 5–8: qualitative spectrogram analyses
### Figure 5: round vector 적용
#### 결과
- [i]에 round vector를 더하면 formant가 전반적으로 내려감.
- 영어에는 front rounded vowel이 없는데도, 모델이 그 방향으로 자연스럽게 변형 가능.

#### 인사이트
- **학습 데이터에 없는 조합도 벡터 조작으로 생성 가능**
- phonological vector가 단순 분류가 아니라 **생성적/조작 가능한 방향**임을 보여줌.

---

### Figure 6: voicing vector 적용
#### 결과
- λ가 커질수록 **voicing onset이 빨라짐**
- 큰 값에서는 closure 구간까지 voicing이 침범하여 negative VOT처럼 보임.

#### 인사이트
- voicing은 단순히 유/무가 아니라 **시간적 정렬의 변화**로도 표현됨.
- 즉, S3M은 voicing의 **동적 타이밍**까지 반영할 수 있음.

---

### Figure 7: strident vector 적용
#### 결과
- λ 증가 시 고주파 frication이 강화.
- 동시에 plosive burst가 사라짐.

#### 인사이트
- stridency는 단순한 spectral noise가 아니라,
  **burst vs frication 같은 temporal-structural 차이**까지 함께 조작됨.
- 즉, 모델은 **정적 스펙트럼뿐 아니라 내부 시간 구조**를 학습하고 있음.

---

### Figure 8: nasal vector 적용
#### 결과
- λ 증가 시 burst가 약해지고 low-frequency murmur가 추가됨.

#### 인사이트
- nasality 역시 단순한 공명 변화만이 아니라,
  **폐쇄/방출 구조와 비강 음향 단서가 함께 연동**되어 변함.
- 이는 음운 특징이 여러 acoustic cue의 묶음으로 구현된다는 점을 보여줍니다.

---

## 6) Table 1: phonological features와 acoustic measurements 대응표
### 결과
- 각 phonological feature가 대응되는 대표 acoustic measurement를 정리함:
  - high/low → F1
  - back/round → F2
  - nasal → F1BW
  - sonorant → HNR
  - strident/voice → COG

### 인사이트
- 이 표는 실험의 해석 기준입니다.
- 단순히 상관을 본 것이 아니라,
  **음운 이론에서 예측하는 방향과 실제 음향 지표가 맞는지 검증**하는 틀을 제공합니다.
- 논문의 주장인 “phonological interpretable vectors”를 뒷받침하는 핵심 기준표입니다.

---

## 7) Appendix A.1: item-based vs offset-based analogy test
### 결과
- 본문에서는 item-based analogy test를 주로 사용.
- offset-based test도 함께 설명하며,
  둘은 평가 철학이 다름.
- offset-based는 더 robust하지만,
  speech에서는 phone pair 수가 적어서 많은 analogy가 탈락함.

### 인사이트
- speech 데이터는 word analogy처럼 대규모가 아니기 때문에,
  **평가 방식 선택이 결과에 큰 영향을 줌**.
- 그래서 본 논문은 success rate 기반 item test를 택했으며,
  이는 speech에 더 적합한 선택이라는 논리를 제시합니다.
- 즉, **speech analogies는 word analogies와 같은 방식으로만 평가하면 안 된다**는 점을 보여줍니다.

---

## 8) Appendix B.1: offset-based PCS 결과
### 결과
- PCS에서도 S3Ms가 spectral features보다 우수.
- 다만 success rate와는 달리,
  최종층보다는 중간층에서 peak가 나타나는 경향이 있음.

### 인사이트
- 평가 지표가 바뀌면 layerwise 결론의 세부 양상이 달라질 수 있음.
- 그러나 **S3M이 spectral representation보다 phonological regularity를 잘 담는다는 큰 결론은 유지**됨.
- 따라서 논문의 핵심은 특정 metric 하나에만 의존하지 않는다는 점입니다.

---

## 9) Appendix B.2: Feature slicing vs Audio slicing
### 결과
- S3M에서는 **feature slicing**이 더 좋음.
- Spectral representation에서는 **audio slicing**이 더 좋음.
- Audio slicing은 context를 줄이므로, S3M의 장점인 contextualization을 약화시킴.

### 인사이트
- S3M의 phonological 구조는 **문맥 포함된 representation**에서 더 잘 드러남.
- 반면 MFCC는 오히려 segment만 잘라 쓰는 것이 도움이 됨.
- 즉, **모델 종류에 따라 적절한 분석 단위가 다르다**는 점을 보여줍니다.

---

## 10) Appendix B.3: anisotropic collapse 분석
### 결과
- wav2vec 2.0과 MFCC는 유사도가 1.0 근처로 몰리는 collapse 현상이 있음.
- WavLM은 그런 collapse가 덜함.

### 인사이트
- representation space가 너무 뭉개지면 phonological vector를 안정적으로 뽑기 어려움.
- WavLM이 synthesis에서 더 잘 작동한 이유 중 하나를 설명합니다.
- 즉, **좋은 phonological vector는 공간 전체가 너무 anisotropic하면 얻기 어렵다**는 점을 보여줍니다.

---

## 11) Appendix B.4: phone recognition fine-tuning 영향
### 결과
- fine-tuned phone recognizer들은 pre-trained S3M보다 phonological structure가 더 강해짐.
- 다만 layerwise behavior는 모델별로 다름.

### 인사이트
- phone recognition 목적의 fine-tuning은
  **phonological abstraction을 강화할 수 있음**.
- 하지만 language coverage와 학습 목적에 따라 어떤 층에서 그 구조가 나타나는지는 달라집니다.
- 즉, **task-specific optimization이 phonological geometry를 재구성할 수 있음**을 시사합니다.

---

## 12) Appendix B.5–B.6: feature별/거리별 layerwise trend
### 결과
- 개별 phonological feature나 phonological distance가 layerwise trend를 크게 바꾸지 않음.

### 인사이트
- 특정 feature만의 특수 현상이라기보다,
  **전반적인 phonological structure가 공통적으로 layer에 분산**되어 있음.
- 즉, 모델은 “voicing만” 잘하는 것이 아니라,
  **음운 체계 전반의 선형 구조**를 학습하고 있다는 주장에 힘을 줍니다.

---

## 13) Appendix B.7–B.8: sample efficiency / single phone pair
### 결과
- 수백 개 샘플이면 phonological vector를 꽤 정확히 추정 가능.
- 단일 phone pair만 쓰면 방향은 어느 정도 맞지만 ground truth와는 차이가 큼.

### 인사이트
- phonological vector는 **몇 개 예시만으로도 대략 추정 가능**하지만,
  **여러 phone을 평균내는 것이 훨씬 안정적**입니다.
- 즉, phonological vector는 특정 pair의 우연한 대비가 아니라
  **feature-level 공통 방향**으로 이해해야 합니다.

---

## 14) Appendix B.9: phonological vectors 간 cosine similarity
### 결과
- vowel 관련 벡터끼리, consonant 관련 벡터끼리 phonologically 타당한 관계를 보임.
- 예:
  - high vs low: 강한 음의 상관
  - nasal vs sonorant vs voice: 양의 상관
  - strident vs sonorant: 음의 상관

### 인사이트
- 벡터들은 서로 독립적이지 않고,
  **음운 이론에서 예측되는 상호관계를 반영**함.
- 따라서 이 벡터들은 단순한 임의 방향이 아니라,
  **phonological feature system 자체의 구조를 재현**합니다.

---

## 15) Appendix B.10: unseen languages에 대한 resynthesis
### 결과
- VoxAngeles + multilingual vocoder에서도 TIMIT와 거의 동일한 경향.
- unseen language에서도 잘 작동함.

### 인사이트
- 이 결과는 논문의 일반화 주장에 매우 중요합니다.
- 즉, phonological vector는 영어에만 국한된 현상이 아니라,
  **cross-linguistic phonological regularity**를 반영할 가능성이 큽니다.

---

## 16) Appendix B.11: vowel rounding case study
### 결과
- round vector를 적용하면 F1, F2, F3가 dataset마다 조금 다르게 변함.
- 하지만 공통적으로 rounding에 맞는 방향의 변화가 관찰됨.

### 인사이트
- phonological feature는 언어/데이터셋에 따라 **acoustic realization이 조금씩 달라질 수 있음**
- 그런데도 vector는 그 차이를 반영하면서도 일관된 방향성을 유지합니다.
- 즉, **“보편성 + 데이터셋 특이성”을 동시에 담을 수 있는 표현**입니다.

---

## 17) Appendix B.12: resynthesis stability
### 결과
- λ=0일 때 original audio와 resynthesized audio의 acoustic difference가 거의 0.
- vocoder와 측정 파이프라인이 안정적임.

### 인사이트
- 실험 결과가 vocoder artifact 때문이 아니라,
  **실제 representation modification 효과**임을 뒷받침합니다.
- 즉, 논문의 조작 결과에 대한 신뢰성을 높여주는 검증입니다.

---

## 18) Appendix B.13: MFCC-derived vectors는 synthesis에 부적합
### 결과
- MFCC 기반 phonological vector는 acoustic measurement와 거의 상관이 없음.
- controllability가 매우 약함.

### 인사이트
- 단순한 spectral feature는 phonological vector arithmetic의 기반으로 충분하지 않음.
- 즉, **self-supervised model의 representation이 핵심**이지,
  단순 전통 음향 특징만으로는 이런 구조를 얻기 어렵습니다.

---

# 전체 핵심 인사이트 한 줄 요약
이 논문은 **self-supervised speech model이 음운 특징을 “선형 방향 + 연속적 스케일”로 인코딩하며, 그 벡터를 조작하면 실제 발화의 음향적 성질까지 제어할 수 있다**는 것을 다양한 피규어, 표, 부록 실험으로 입증합니다.

---





## 1) Figure 1: Text word analogies vs. speech phonological analogies
### Result
- Just as word embeddings support semantic vector arithmetic like  
  **king - man + woman ≈ queen**,  
  the paper asks whether speech representations support phonological vector arithmetic like  
  **[p] - [t] + [d] ≈ [b]**.

### Insight
- This figure introduces the paper’s main question.
- It transfers the idea of vector arithmetic from text embeddings to speech models.
- The core hypothesis is that **speech representations can encode phonological features as linear directions**.

---

## 2) Figure 2: S3Ms vs. spectral representations on phonological analogies
### Result
- Self-supervised speech models (S3Ms) outperform MFCC and MelSpec by a large margin.
- Examples:
  - **HuBERT final layer: 94%**
  - **WavLM final layer: 92%**
  - **wav2vec 2.0 middle layer: 61%**
- Baselines are much weaker:
  - MFCC: **19%**
  - MelSpec: **0%**
- Similar trends hold on both TIMIT and VoxAngeles.

### Insight
- S3Ms encode phonological structure much better than traditional spectral features.
- Deeper layers tend to be more abstract and more phonologically organized.
- The result also shows cross-lingual generalization to phones unseen in English.

---

## 3) Figure 3: Vowels vs. consonants
### Result
- Vowels and consonants peak at different layers.
- For WavLM:
  - vowels peak earlier,
  - consonants peak later,
  - both peak in the final layer.

### Insight
- Vowels rely more on local cues.
- Consonants depend more on broader temporal context.
- This suggests that different phonological units are distributed across layers in different ways.

---

## 4) Figure 4: Scale λ and acoustic measurements
### Result
- For eight phonological features, increasing or decreasing λ changes acoustic measurements in the expected direction.
- The observed correlation signs match phonological theory exactly.

### Insight
- Not only the **direction** but also the **scale** of a phonological vector matters.
- This means phonological features are represented as **continuous dimensions**, not just binary categories.
- It also enables fine-grained control over speech synthesis.

---

## 5) Figures 5–8: Qualitative spectrogram analyses
### Figure 5: Roundness vector
- Applying the roundness vector to [i] lowers formants, even though English lacks front rounded vowels.
- Insight: phonological vectors can generalize to unseen combinations.

### Figure 6: Voicing vector
- Larger λ moves voicing onset earlier, even into closure.
- Insight: voicing is encoded as a dynamic timing phenomenon, not just a binary feature.

### Figure 7: Strident vector
- Increasing λ adds frication and removes burst structure.
- Insight: the model captures temporal structure, not only static spectra.

### Figure 8: Nasal vector
- Increasing λ weakens the burst and adds nasal murmur.
- Insight: nasality is represented as a coordinated set of acoustic cues.

---

## 6) Table 1: Mapping phonological features to acoustic measurements
### Result
- The paper maps features to measurements such as:
  - high/low → F1
  - back/round → F2
  - nasal → F1 bandwidth
  - sonorant → HNR
  - strident/voice → COG

### Insight
- This table is the interpretive backbone of Experiment 2.
- It formalizes how phonological theory predicts acoustic correlates.

---

## 7) Appendix A.1: Item-based vs. offset-based analogy tests
### Result
- The paper mainly uses item-based analogy tests.
- Offset-based tests are discussed as a secondary analysis.
- Offset-based evaluation is more robust in principle but discards many analogies in speech data.

### Insight
- Speech analogy evaluation differs from word analogy evaluation because speech has fewer usable phone pairs.
- This justifies the paper’s choice of evaluation metric.

---

## 8) Appendix B.1: Offset-based PCS results
### Result
- S3Ms still outperform spectral features.
- However, the layerwise peak often appears in intermediate layers rather than the final layer.

### Insight
- The detailed layerwise pattern depends on the evaluation metric.
- Still, the broad conclusion remains: S3Ms encode stronger phonological regularities than spectral features.

---

## 9) Appendix B.2: Feature slicing vs. audio slicing
### Result
- Feature slicing works better for S3Ms.
- Audio slicing works better for spectral features.

### Insight
- S3Ms benefit from contextualized representations.
- MFCCs are more local and can benefit from direct segment-level slicing.
- The appropriate analysis method depends on the representation type.

---

## 10) Appendix B.3: Anisotropic collapse
### Result
- wav2vec 2.0 and MFCC show near-1.0 similarity collapse.
- WavLM is less collapsed.

### Insight
- If the representation space collapses too much, phonological vectors become harder to extract.
- This helps explain why WavLM works better for synthesis tasks.

---

## 11) Appendix B.4: Impact of phone-recognition fine-tuning
### Result
- Fine-tuned phone recognizers strengthen phonological structure.
- Their layerwise behavior differs depending on the model.

### Insight
- Fine-tuning for phone recognition can enhance phonological abstraction.
- Training objective and language coverage influence where phonological structure emerges.

---

## 12) Appendices B.5–B.6: Feature-wise and distance-wise trends
### Result
- Layerwise trends are mostly consistent across individual features and phonological distances.

### Insight
- The phenomenon is not feature-specific.
- It supports the claim that S3Ms encode a broad phonological system, not just isolated contrasts.

---

## 13) Appendices B.7–B.8: Sample efficiency and single phone pair
### Result
- A few hundred samples are enough to approximate phonological vectors well.
- Single phone pairs produce noisy approximations.

### Insight
- Phonological vectors should be understood as feature-level directions, not accidental pairwise contrasts.

---

## 14) Appendix B.9: Similarities between phonological vectors
### Result
- Vowel-related and consonant-related vectors show interpretable relationships.
- For example:
  - high vs. low: strong negative similarity
  - nasal vs. sonorant vs. voice: positive similarity
  - strident vs. sonorant: negative similarity

### Insight
- The extracted vectors reflect the structure of phonological theory.
- They are not arbitrary directions; they form a meaningful feature system.

---

## 15) Appendix B.10: Resynthesis on unseen languages
### Result
- Very similar trends appear on VoxAngeles with a multilingual vocoder.

### Insight
- The method generalizes beyond English.
- This supports a cross-linguistic interpretation of phonological vectors.

---

## 16) Appendix B.11: Vowel rounding case study
### Result
- The rounding vector affects F1, F2, and F3 in dataset-specific ways.
- Yet the direction remains phonologically consistent.

### Insight
- Phonological realization is language/data dependent, but the learned vector still captures the shared underlying dimension.

---

## 17) Appendix B.12: Resynthesis stability
### Result
- With λ = 0, original and resynthesized speech are nearly identical acoustically.

### Insight
- This confirms that the observed effects are due to vector manipulation, not vocoder artifacts.

---

## 18) Appendix B.13: MFCC-derived vectors are ineffective for synthesis
### Result
- MFCC-based phonological vectors show little or no correlation with acoustic measurements.
- Controllability is weak.

### Insight
- Traditional spectral features are not sufficient to produce usable phonological vector arithmetic.
- The self-supervised representation itself is the key factor.

---

# One-sentence overall takeaway
The paper shows that **self-supervised speech models encode phonological structure as linear, scalable vectors**, and that manipulating those vectors can produce **interpretable, continuous changes in acoustic realizations**.



<br/>
# refer format:


---

## 1) BibTeX

```bibtex
@article{choi2026phonologicalvectorarithmetic,
  title={\{[b]\}={\,[d]\}-\{[t]\}+\{[p]\}: Self-supervised Speech Models Discover Phonological Vector Arithmetic},
  author={Choi, Kwanghee and Yeo, Eunjung and Cho, Cheol Jun and Harwath, David and Mortensen, David R.},
  journal={arXiv preprint arXiv:2602.18899v3},
  year={2026},
  month={apr},
  url={https://github.com/juice500ml/phonetic-arithmetic}
}
```



```bibtex
@article{choi2026phonologicalvectorarithmetic,
  title={[b]=[d]-[t]+[p]: Self-supervised Speech Models Discover Phonological Vector Arithmetic},
  author={Choi, Kwanghee and Yeo, Eunjung and Cho, Cheol Jun and Harwath, David and Mortensen, David R.},
  journal={arXiv preprint arXiv:2602.18899v3},
  year={2026},
  month={apr},
  url={https://github.com/juice500ml/phonetic-arithmetic}
}
```

---

## 2) 시카고 스타일(줄글형)

Choi, Kwanghee, Eunjung Yeo, Cheol Jun Cho, David Harwath, and David R. Mortensen. 2026. “[b]=[d]-[t]+[p]: Self-supervised Speech Models Discover Phonological Vector Arithmetic.” arXiv preprint arXiv:2602.18899v3, April 13, 2026. https://github.com/juice500ml/phonetic-arithmetic.

---




