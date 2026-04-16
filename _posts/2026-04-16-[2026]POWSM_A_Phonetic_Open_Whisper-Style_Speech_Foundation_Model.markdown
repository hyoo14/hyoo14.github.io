---
layout: post
title:  "[2026]POWSM: A Phonetic Open Whisper-Style Speech Foundation Model"
date:   2026-04-16 19:04:10 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: POWSM는 17,000시간 규모의 공개 다국어 음성·음소 데이터(IPAPack++)로부터, 하나의 AED(encoder-decoder) 구조에서 PR, ASR, audio-guided G2P, audio-guided P2G를 함께 학습하도록 설계했고, CTC/attention 혼합 손실과 언어 토큰·태스크 토큰을 사용했습니다.


짧은 요약(Abstract) :




이 논문은 **POWSM(Phonetic Open Whisper-Style Speech Model)**이라는 새로운 음성 기반 파운데이션 모델을 소개합니다.  
이 모델의 핵심은 **음성 인식(ASR), 음소 인식(PR), grapheme-to-phoneme 변환(G2P), phoneme-to-grapheme 변환(P2G)** 같은 여러 음운(phonetic) 관련 작업을 **하나의 통합된 프레임워크**에서 동시에 처리할 수 있다는 점입니다.

기존에는 이런 작업들이 서로 비슷한 개념임에도 불구하고, 각각 별도의 모델과 데이터셋으로 따로 연구되어 왔습니다.  
하지만 POWSM은 **오디오, 문자(grapheme), 음소(phone)** 사이의 변환을 자연스럽게 지원해서, 언어 간 일반화와 저자원(low-resource) 음성 처리에 더 유리한 구조를 제공합니다.

실험 결과, POWSM은 **비슷한 크기의 특화된 음소 인식 모델들(Wav2Vec2Phoneme, ZIPA)**과 비교해 **동등하거나 더 좋은 성능**을 보였고, 동시에 **G2P, P2G, ASR**까지 함께 지원했습니다.  
또한 이 논문은 **학습 데이터, 코드, 모델**을 공개하여 오픈 사이언스를 장려한다고 밝히고 있습니다.

---



This paper introduces **POWSM (Phonetic Open Whisper-Style Speech Model)**, a new speech foundation model designed to handle multiple phonetic tasks within a single unified framework.  
These tasks include **automatic speech recognition (ASR), phone recognition (PR), grapheme-to-phoneme conversion (G2P), and phoneme-to-grapheme conversion (P2G)**.

Although these tasks are conceptually related, they have traditionally been studied separately, each with its own task-specific architecture and dataset.  
POWSM addresses this fragmentation by enabling seamless conversion among **audio, text (graphemes), and phones**, which makes it useful for universal and low-resource speech processing.

The paper reports that POWSM **matches or outperforms specialized PR models of similar size**, such as **Wav2Vec2Phoneme** and **ZIPA**, while also supporting **G2P, P2G, and ASR** in one model.  
The authors also release their **training data, code, and model** to support open science.

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





## 1. 모델의 목적과 전체 개요
POWSM는 이름 그대로 **“phonetic(음성학/음운 단위) 중심”**의 **speech foundation model**입니다.  
기존 음성 foundation model인 Whisper나 OWSM이 주로 **ASR(음성인식)** 중심이었다면, POWSM는 한 단계 더 나아가 다음 **4가지 phone-related task**를 하나의 프레임워크에서 같이 처리하도록 설계되었습니다.

1. **PR (Phone Recognition)**  
   - 음성에서 phone(음운 단위)을 예측
2. **ASR (Automatic Speech Recognition)**  
   - 음성에서 grapheme/text(문자열)로 변환
3. **Audio-guided G2P (Grapheme-to-Phoneme)**  
   - 음성과 텍스트를 함께 이용해 발음/phone 시퀀스를 예측
4. **Audio-guided P2G (Phoneme-to-Grapheme)**  
   - 음성과 phone을 이용해 문자로 변환

즉, 하나의 모델이 **speech ↔ phones ↔ graphemes** 사이를 오가며 학습하고 추론할 수 있도록 만든 **통합 phonetic foundation model**입니다.

---

## 2. 핵심 아이디어: 왜 phone 중심인가?
논문은 phone을 다음과 같이 봅니다.

- phone은 speech의 가장 작은 음성 단위
- grapheme(문자)보다 언어 간에 더 공유되기 쉬움
- IPA(International Phonetic Alphabet)로 표현 가능
- 따라서 여러 언어에 대해 **cross-lingual generalization**에 유리

이 때문에 POWSM는 단순히 “문자 기반 ASR”이 아니라,  
**음성의 발음적/음운적 구조를 직접 배우는 모델**로 설계되었습니다.

---

## 3. 사용한 데이터: IPAPack++ 기반 멀티링구얼 음성 데이터
### 3.1 기본 학습 데이터
POWSM는 **IPAPack++**라는 오픈소스 코퍼스를 사용합니다.

- 약 **17,000시간 규모**
- 다국어 음성 데이터
- **orthographic transcription(철자/문자 전사)**와 **phonemic transcription(음운 전사)**가 함께 존재
- 논문에서는 이 데이터를 기반으로 멀티태스크 학습을 수행

즉, 단순한 음성-문자 쌍이 아니라,  
음성에 대해 **문자열과 phoneme 정보가 함께 연결된 데이터**를 활용합니다.

### 3.2 데이터 전처리와 정규화
논문에서 꽤 중요한 부분입니다.

#### (1) G2P 생성 전사 정제
- G2P로 생성한 transcription은 **사람이 직접 검수하고 정제**
- 너무 긴 발화는 제거:
  - **300 phones 초과 utterance 필터링**

#### (2) IPA normalization
- IPA sequence는 **Unicode NFD** 형태로 정규화
- 결합 기호/변형 요소가 안정적으로 처리되도록 함

#### (3) 영어 G2P 보정
영어는 표기와 실제 발음 차이가 크기 때문에 rule-based correction을 적용했습니다.

예:
1. word-initial 무성 파열음 /p t k/는 aspirated 처리
2. word-initial 유성 파열음 /b d g/는 무성화 처리
3. /l/의 syllable-final 위치에서는 velarization 반영
4. 비음 앞 모음 nasalization 반영

이건 영어의 실제 발음 특성을 phone-level 학습에 더 잘 반영하기 위한 보정입니다.

#### (4) phone token 분할 방식
IPA token이 grapheme과 헷갈리지 않도록:
- **PanPhon phone entries**를 이용해 greedy trie search
- diacritic/modifier를 붙인 상태로 token화
- `/phOs@m/` 같은 예시를 `/ph/ /O/ /s/ /@/ /m/`처럼 분리
- phone 시퀀스는 슬래시로 감싸서 명확히 구분

이 설계는 **phone token과 문자 token의 혼동을 줄이기 위한 기법**입니다.

---

## 4. 멀티태스크 데이터 포맷
POWSM의 중요한 특징 중 하나는 **한 utterance를 task별로 재구성해서 학습**한다는 점입니다.

각 utterance는 4개 task 중 하나의 형식으로 들어가며,  
각 task마다 다음 요소들이 조합됩니다.

- **text prompt**
- **language token**
- **task token**
- **target output**

### 4.1 PR과 ASR
- prompt는 비워둠:
  - `<na>` token 사용
- PR:
  - 음성 → phones
- ASR:
  - 음성 → text/graphemes

### 4.2 G2P와 P2G
- G2P:
  - text prompt로 grapheme을 넣고
  - target은 phone sequence
- P2G:
  - prompt로 phone sequence를 넣고
  - target은 grapheme sequence

즉, 모델은 입력 형식만 달리해도  
**“무엇을 예측할지”를 task token과 prompt를 통해 구분**합니다.

이 방식은 Whisper 스타일의 multitask prompting과 유사하지만,  
여기서는 phone-related task에 맞게 더 세분화되어 있습니다.

---

## 5. 모델 구조: Attention-based Encoder-Decoder(AED)
POWSM는 **attention-based encoder-decoder (AED)** 구조를 사용합니다.  
이 구조는 Whisper/OWSM 계열과 유사합니다.

### 5.1 Encoder
- **E-Branchformer encoder** 사용
- speech input을 받아 일반적인 음향 표현을 추출
- encoder는 **phonetic-aware representation**을 만들도록 유도됨

E-Branchformer는 self-attention과 convolution/branch-style 구조를 결합한 encoder 계열로,  
음성의 local + global 패턴을 함께 포착하는 데 유리합니다.

### 5.2 Decoder
- **Transformer decoder**
- autoregressive 방식
- text prompt와 language token, task token을 조건으로 output 생성
- encoder output에 **cross-attention** 수행

즉,
- encoder는 음향 정보를 압축하고
- decoder는 해당 음향을 바탕으로 task별 출력 형식(phone/text)을 생성합니다.

### 5.3 Whisper 스타일 설계
논문은 이 구조를 **Whisper / OWSM 스타일**의 open speech model 철학에 맞춘다고 설명합니다.  
즉,
- 다국어 지원
- 멀티태스크
- prompt 기반 제어
- 오픈소스 재현성

이 모두를 반영합니다.

---

## 6. 학습 목표: CTC + Attention의 하이브리드
POWSM는 **hybrid CTC/attention loss**를 사용합니다.

수식은 다음과 같습니다.

\[
L = \alpha_{ctc} L_{ctc} + (1 - \alpha_{ctc}) L_{attention}
\]

여기서:
- \(L_{ctc}\): CTC loss
- \(L_{attention}\): attention decoder loss
- \(\alpha_{ctc} = 0.3\)

### 6.1 CTC의 역할
CTC는 encoder 출력과 target sequence를 정렬시키는 역할을 합니다.

논문에서는 특히:
- encoder target을 **simplified phone sequence**로 둠
- break marker(/./, /</ 등)와 length diacritic 제거
- 이렇게 해서 encoder가 더 빠르게 안정적인 alignment를 학습하도록 함

즉, CTC는  
**“encoder가 발화와 phone 시퀀스 사이의 정렬을 빨리 익히도록 돕는 장치”**입니다.

### 6.2 Attention loss의 역할
attention decoder는:
- task token
- language token
- prompt
- encoder representation

을 이용해 최종 output sequence를 생성합니다.

이는 단순 정렬보다 더 유연하게:
- 긴 출력
- 언어별 변형
- phone/text 변환
을 처리할 수 있게 해줍니다.

---

## 7. 특별한 설계 포인트 1: encoder target을 phone으로 통일
논문에서 중요한 관찰은 다음입니다.

- encoder target을 **phone + orthography**로 섞으면 학습이 어려워짐
- 같은 speech input이 task에 따라 서로 다른 CTC target을 갖기 때문

그래서 encoder CTC target은 **phones로 통일**했습니다.

이 선택의 의미:
- encoder는 language-independent acoustic/phonetic representation을 학습
- decoder가 task-specific한 language modeling 역할을 담당
- 결과적으로 멀티태스크 간 충돌을 줄임

즉,  
**encoder는 “소리”를, decoder는 “표현 형식”을 담당**하는 분업 구조입니다.

---

## 8. 특별한 설계 포인트 2: PanPhon 단위와 suprasegmental 제거
논문은 encoder CTC target 단위를 비교하는 실험도 했습니다.

비교한 조건:
1. Unicode code points
2. PanPhon token
3. suprasegmental 포함/제외

결론적으로:
- **PanPhon tokenization**
- **suprasegmental 제거**

가 가장 효율적이었습니다.

### 왜 그런가?
- Unicode code points는 phone을 자연스럽지 않은 조각으로 쪼갤 수 있음
- PanPhon은 phone-diacritic 조합을 더 자연스럽게 한 단위로 표현
- 하지만 길이/브레이크 같은 suprasegmental은 PR에 혼란을 줄 수 있음

즉, CTC encoder에는  
**“fine-grained phone은 좋지만, suprasegmental은 제거하는 게 안정적”**이라는 결론입니다.

---

## 9. 학습 방식과 하이퍼파라미터
논문에 따르면:

- 학습 프레임워크: **ESPnet**
- 모델 크기: 약 **350M parameters**
- Encoder/Decoder: 각각 **9 layers**
- 입력:
  - **16kHz audio**
  - 최대 **20초 padding**
- global batch size: **256**
- stride: **40ms**
- vocabulary:
  - 약 **40k tokens**
  - 그중 약 **6k phone tokens**
  - language token, timestamp token, BPE token 포함
- 학습 자원:
  - **H100 GPU 약 200 GPU hours**
- decoding에서 기본적으로:
  - `ctc = 0.3`
  - `beam = 3`

---

## 10. PR 성능을 높이는 특별한 기법: decoder/encoder weighting
논문에서는 encoder를 더 강하게 활용하면 out-of-domain PR에 유리하다고 분석합니다.

### 핵심 관찰
- CTC weight를 높이면:
  - **out-of-domain** PR이 좋아짐
  - **in-domain** 성능은 다소 나빠질 수 있음

즉, encoder 쪽을 더 강조하면  
언어 모델적 decoder의 영향이 줄어들어  
새로운 언어/방언에 덜 편향된 phone 예측이 가능해집니다.

하지만 너무 강한 CTC는 in-domain에서 이득이 줄 수 있어  
**generalization과 in-domain performance 사이의 trade-off**가 존재합니다.

---

## 11. language token과 task token의 역할
POWSM는 단순히 task token만 쓰는 것이 아니라  
**language token**도 적극적으로 활용합니다.

### language token이 하는 일
- 언어별 phonotactic pattern을 모델링
- 출력 분포를 특정 언어 쪽으로 유도
- PR/ASR/P2G/G2P 모두에서 language-aware behavior를 가능하게 함

논문에서 언어 토큰을 바꿔 실험한 결과:
- 언어 토큰이 PR 결과에 실제로 영향을 줌
- detected language token을 쓰면 English 고정보다 나음
- unknown token이 가장 잘 작동하는 경우도 있음

즉, 모델은 **언어 토큰을 통해 phonotactic prior**를 학습합니다.

---

## 12. G2P와 P2G의 특징
### 12.1 Audio-guided G2P
POWSM의 G2P는 text만 보는 기존 G2P와 달리, **speech를 함께 사용**할 수 있습니다.

논문 결과:
- speech만 사용할 때는 PR과 유사한 동작
- speech + text를 같이 넣으면 오히려 표준화된 발음으로 수렴하는 경향
- text만 쓰면 더 강하게 “정규화된” 발음 출력

즉, G2P는 단순한 문장-발음 변환이 아니라  
**speech 기반의 실제 발음 variation을 반영할 수 있는 장치**입니다.

### 12.2 Audio-guided P2G
P2G는 phone 정보를 이용해 grapheme을 생성합니다.

논문에서는 low-resource ASR에 대해:
- P2G가 ASR보다 성능이 좋을 수 있음
- 특히 gold phone labels를 쓸 때 효과적
- PR 결과를 이용한 PR-P2G도 일부 언어에서 성능 향상

즉, phone context가 있으면  
저자원 언어에서 문자 복원이나 텍스트 생성이 더 쉬워질 수 있습니다.

---

## 13. 요약: POWSM 방법론의 핵심
한 문장으로 요약하면 POWSM는:

> **오픈소스 멀티링구얼 speech foundation model로서, phone을 중심 표현으로 삼아 PR/ASR/G2P/P2G를 하나의 AED 구조에서 공동 학습하는 모델**입니다.

핵심 특징은 다음과 같습니다.

- **IPAPack++** 기반의 약 17k시간 멀티링구얼 데이터 사용
- **phones 중심**의 통합 표현
- **Whisper/OWSM 스타일 AED 구조**
- **E-Branchformer encoder + Transformer decoder**
- **hybrid CTC/attention loss**
- **PanPhon tokenization**
- **suprasegmental 제거**
- **language token / task token 기반 멀티태스크 제어**
- **out-of-domain 일반화에 강한 phone recognition**
- **audio-guided G2P / P2G** 지원



---

## 1. Goal and Overall Idea
POWSM is a **phonetic speech foundation model** designed to operate around **phones** rather than only graphemes/text.  
While Whisper- and OWSM-style models mainly focus on ASR, POWSM unifies four phone-related tasks in a single framework:

1. **Phone Recognition (PR)**  
   - Speech → phones
2. **Automatic Speech Recognition (ASR)**  
   - Speech → graphemes/text
3. **Audio-guided Grapheme-to-Phoneme (G2P)**  
   - Speech + text → phones
4. **Audio-guided Phoneme-to-Grapheme (P2G)**  
   - Speech + phones → text

So the model learns to bridge **speech, phones, and orthography** within one architecture.

---

## 2. Why Phones?
The paper motivates phones as the core unit because:

- phones are the smallest sound units in speech
- they are shared across languages more naturally than graphemes
- they can be represented using **IPA**
- they support **cross-lingual generalization**

Thus, POWSM is built as a model that learns **phonetic structure** directly, rather than only orthographic mappings.

---

## 3. Training Data: IPAPack++
### 3.1 Main Dataset
POWSM is trained on **IPAPack++**, an open multilingual speech corpus of roughly **17,000 hours**.  
It contains:

- speech audio
- orthographic transcriptions
- phonemic transcriptions

This makes it suitable for a unified multi-task setup.

### 3.2 Data Cleaning and Normalization
Important preprocessing steps include:

- manual inspection and cleaning of G2P-generated transcriptions
- filtering utterances longer than **300 phones**
- normalizing IPA strings to **Unicode NFD**
- applying rule-based corrections to English G2P outputs for:
  - aspiration of word-initial voiceless stops
  - devoicing of word-initial voiced stops
  - velarization of syllable-final /l/
  - vowel nasalization before nasals

### 3.3 Phone Tokenization
To prevent confusion between IPA phones and graphemes:

- phone sequences are tokenized using **PanPhon phone entries**
- diacritics and modifiers are attached by greedy trie search
- phone tokens are enclosed in slashes

This ensures phone units are treated as meaningful tokens rather than arbitrary character fragments.

---

## 4. Multi-task Data Format
Each utterance is reformatted into one of four task-specific formats using:

- a **text prompt**
- a **language token**
- a **task token**
- a **target output**

### PR and ASR
- prompt is empty: `<na>`
- PR predicts phones from speech
- ASR predicts text from speech

### G2P and P2G
- G2P uses text as prompt and predicts phones
- P2G uses phones as prompt and predicts graphemes

This is similar to Whisper-style prompting, but extended specifically for phonetic tasks.

---

## 5. Model Architecture: Attention-Based Encoder-Decoder
POWSM uses an **attention-based encoder-decoder (AED)** architecture, following the general design of Whisper and OWSM.

### 5.1 Encoder
- **E-Branchformer encoder**
- extracts acoustic representations from speech
- encourages phonetic-aware hidden states

### 5.2 Decoder
- **Transformer decoder**
- autoregressive generation
- conditioned on prompt, language token, and task token
- uses cross-attention over encoder outputs

So the encoder handles acoustic representation, while the decoder handles task-specific sequence generation.

---

## 6. Training Objective: Hybrid CTC/Attention
POWSM is trained with a **hybrid CTC/attention loss**:

\[
L = \alpha_{ctc} L_{ctc} + (1 - \alpha_{ctc}) L_{attention}
\]

with:

- \(\alpha_{ctc} = 0.3\)

### Role of CTC
CTC aligns encoder outputs with a simplified phone target sequence.  
The paper strips break symbols and length diacritics from the CTC target to make alignment easier and faster.

### Role of Attention
The attention decoder generates the final sequence conditioned on the prompt and encoder outputs, allowing flexible multi-task generation.

---

## 7. Special Design Choice 1: Use Phones as Encoder Targets
A key finding in the paper is that mixing phones and orthography as encoder targets hurts training, because the same speech input would have different CTC targets across tasks.

Therefore, the encoder CTC target is unified to **phones only**.

This makes the encoder learn a more general acoustic/phonetic representation, while the decoder handles task-specific output formatting.

---

## 8. Special Design Choice 2: PanPhon Tokenization and Removing Suprasegmentals
The authors compare different encoder targets and find that:

- **PanPhon tokenization**
- **without suprasegmentals**

gives the best training efficiency.

Why?

- Unicode code points may split phones into unnatural units
- PanPhon provides more natural phone-diacritic units
- suprasegmentals like length marks can confuse PR

So the encoder benefits from **fine-grained phones**, but not from suprasegmental complexity.

---

## 9. Training Details
Key training settings:

- framework: **ESPnet**
- model size: about **350M parameters**
- encoder and decoder: **9 layers each**
- audio: **16 kHz**
- max length: **20 seconds**
- global batch size: **256**
- encoder stride: **40 ms**
- vocabulary: about **40k tokens**, including:
  - ~6k phone tokens
  - language tokens
  - timestamp tokens
  - BPE tokens
- training compute: around **200 H100 GPU hours**
- default decoding:
  - `ctc = 0.3`
  - `beam = 3`

---

## 10. Encoder vs Decoder Weighting and Generalization
The paper shows that giving more weight to the encoder can improve out-of-domain PR performance.

Main observation:

- higher CTC weight often helps **unseen languages**
- but may hurt **in-domain** performance

This suggests a trade-off:

- decoder helps by acting like a phonotactic language model
- stronger encoder weighting improves cross-lingual generalization

---

## 11. Language Token and Task Token
POWSM also uses **language tokens**, not just task tokens.

The language token helps:

- model phonotactic patterns
- condition output distribution by language
- improve language-aware behavior in PR, ASR, and conversion tasks

The paper shows that changing the language token at inference affects PR outputs, indicating that the model has learned language-specific prior distributions.

---

## 12. G2P and P2G Behavior
### Audio-guided G2P
Unlike text-only G2P, POWSM can use speech as an additional signal.  
This helps capture phonetic variation and pronunciation differences.

### Audio-guided P2G
P2G leverages phone context to generate text, and can be especially useful in low-resource settings.

The paper shows that phone-based prompting can significantly improve low-resource recognition/transcription performance.

---

## 13. Summary
In short, POWSM is:

> an open-source multilingual speech foundation model that uses phones as the central representation and jointly trains PR, ASR, audio-guided G2P, and audio-guided P2G in a single AED architecture.

Its main methodological features are:

- IPAPack++ multilingual training data
- phone-centric representation
- E-Branchformer encoder + Transformer decoder
- hybrid CTC/attention loss
- PanPhon-based phone tokenization
- suprasegmental removal for CTC
- task token + language token prompting
- strong out-of-domain PR generalization
- support for audio-guided G2P and P2G

---



<br/>
# Results





## 1. 결과의 전체적인 의미
이 논문에서 POWSM은 단일 목적 모델이 아니라, **Phone Recognition(PR), Automatic Speech Recognition(ASR), audio-guided G2P, audio-guided P2G**를 함께 수행하는 **멀티태스크 phonetic speech foundation model**로 제안됩니다.  
결과 파트의 핵심은 다음과 같습니다.

- **PR(Phone Recognition)** 에서 경쟁모델보다 같거나 더 좋은 성능을 보임
- **ASR** 에서도 특히 **저자원 언어**에서 웹 규모 모델과 비슷한 수준의 성능을 보임
- **보지 못한 언어(unseen languages)** 와 **언어 변이(dialect, L2 speech)** 에 대해서도 강한 일반화 성능을 보임
- **G2P/P2G** 는 음성 조건을 활용하여 기존 텍스트 기반 변환보다 유연한 결과를 보임

즉, POWSM은 “전화(phonetic) 정보 중심의 통합 모델”로서 다중 작업을 한 프레임워크 안에서 수행 가능함을 보여주는 것이 결과의 핵심입니다.

---

## 2. 평가에 사용한 메트릭

### 2.1 PR 평가 지표: PFER
논문은 PR에서 **PFER(Phonetic Feature Error Rate)** 를 주요 지표로 사용합니다.

- 단순히 phone이 정확히 맞는지 보는 PER(Phone Error Rate)보다 더 정교함
- PanPhon의 **articulatory feature** 를 이용해, 예를 들어 유성/무성, 조음 위치, 조음 방법 같은 **음성학적 유사성**까지 반영
- 삽입/삭제는 1, 특징 차이는 feature distance 기준으로 계산

논문은 PFER를 다음과 같이 설명합니다:
- PER은 exact match만 봄
- PTER는 diacritic/modifier를 별도 token으로 취급
- **PFER는 발음적 유사성을 세밀하게 반영**

즉, POWSM의 PR 성능을 볼 때 단순 정답률보다 **발음적으로 얼마나 가까운 오류인지**까지 평가합니다.

### 2.2 ASR 평가 지표: WER / CER
- **WER(Word Error Rate)**: 영어 및 단어 단위 텍스트 인식 성능 평가
- **CER(Character Error Rate)**: 중국어(예: Mandarin)처럼 문자 단위가 더 자연스러운 경우 사용

논문에서는 ASR 결과를 주로 **WER** 로 제시하고, 중국어는 **CER** 을 사용합니다.

### 2.3 P2G 관련 비교
P2G는 주로 **WER** 로 평가합니다.  
특히 음성에서 추출한 phone 정보를 prompt로 넣는 방식과, gold phone을 쓰는 방식, PR 출력 phone을 쓰는 방식(PR-P2G)을 비교합니다.

---

## 3. 비교에 사용된 경쟁모델(Baselines)

## 3.1 PR 경쟁모델
논문에서 PR 비교 대상으로 사용한 모델은 다음과 같습니다.

1. **Allosaurus**
   - language-agnostic phone recognition
   - allophone-to-phoneme mapping 사용

2. **Allophant**
   - articulatory attributes 기반
   - XLS-R fine-tuning

3. **Wav2Vec2Phoneme**
   - zero-shot cross-lingual phoneme recognition
   - articulatory feature 기반 unseen phoneme 처리

4. **MultIPA**
   - 고품질 G2P 데이터를 활용

5. **ZIPA**
   - ZipFormer 기반
   - IPAPack++에서 대규모로 학습
   - 추가 pseudo-labeled data를 쓴 버전도 비교:
     - **ZIPA-CR-Large**
     - **ZIPA-CR-NS-Large**

6. **POWSM**
   - 본 논문 제안 모델
   - 350M 파라미터
   - PR, ASR, G2P, P2G를 함께 수행

---

## 4. 테스트 데이터셋 구성

## 4.1 In-domain 데이터
훈련에 사용된 **IPAPack++** 내부 데이터셋에서 평가합니다.  
언어는 다음이 포함됩니다:

- 영어(eng)
- 독일어(deu)
- 네덜란드어(nld)
- 프랑스어(fra)
- 이탈리아어(ita)
- 스페인어(spa)
- 포르투갈어(por)
- 폴란드어(pol)
- 타밀어(tam)
- 카자흐어(kaz)
- 만다린(cmn)

이 데이터는 모델이 본 적 있는 분포에서의 성능을 봅니다.

---

## 4.2 Out-of-domain: unseen languages
훈련에 포함되지 않은 언어들에 대해 평가합니다.

- **DoReCo**: 45개 언어 subset
- **VoxAngeles**: 95개 언어를 포함하는 UCLA Phonetics Lab Archive 기반 데이터
- **Tusom2021**: 저자원, 종료위기(endangered) 언어 데이터

이 세 데이터는 **새로운 언어에 대한 일반화 능력**을 봅니다.

---

## 4.3 Out-of-domain: language variation
같은 영어 계열 혹은 L2 발화를 포함하는 데이터에서 평가합니다.

- **Buckeye Corpus**: 영어 방언/대화체
- **DoReCo South-England**: 영어 방언
- **L2-ARCTIC**: 비원어민 영어
- **EpaDB**: 발음 평가용 비원어민 데이터
- **SpeechOcean762**: 비원어민 영어 발화

이들은 모델이 **방언, L2 발음, 사회음성학적 변이**에 얼마나 강한지 보여줍니다.

---

## 5. PR 결과 해석

## 5.1 In-domain PR 결과
표 2에서 POWSM은 **IPAPack++ 내 평가**에서 평균적으로 가장 낮거나 매우 경쟁력 있는 PFER를 보입니다.

핵심 해석:
- POWSM은 **Decoder의 언어모델링 능력**이 강해 phone sequence를 잘 생성
- ZIPA-CR-NS-Large 같은 강력한 경쟁모델과도 비교 가능
- 일부 Germanic language에서는 English 데이터 정제 영향으로 성능이 다소 약해졌을 가능성도 언급

즉, in-domain에서는 **POWSM이 최고 수준의 PR 성능**을 보였고, 특히 phone-level modeling이 잘 작동함을 보여줍니다.

### 수치적으로 보면
- POWSM 평균 PFER: **2.62**
- ZIPA-CR-NS-Large 평균 PFER: **2.70**
- ZIPA-CR-Large 평균 PFER: **2.99**

즉, POWSM이 더 낮은 error를 보여 가장 좋습니다.

---

## 5.2 Out-of-domain PR 결과: unseen languages
표 4에서 POWSM은 **보지 못한 언어들**에서도 좋은 성능을 보입니다.

### 평균 PFER
- POWSM: **18.71**
- ZIPA-CR-NS-Large: **19.01**
- ZIPA-CR-Large: **19.54**
- Wav2Vec2Phoneme: **21.02**
- MultIPA: **21.35**
- Allosaurus: **32.52**

해석:
- POWSM은 ZIPA보다도 조금 더 좋거나 비슷한 수준
- 특히 **같은 데이터로 학습한 ZIPA보다 더 좋고**, 추가 pseudo-label을 쓴 ZIPA-CR-NS-Large보다도 평균적으로 우세
- 따라서 **멀티태스크 학습이 unseen language 일반화에 도움**을 준다고 해석 가능

---

## 5.3 Out-of-domain PR 결과: language variation
같은 표 4에서 방언/L2 데이터에서도 POWSM은 강한 성능을 보입니다.

### 평균 PFER
- POWSM: **14.40**
- ZIPA-CR-NS-Large: **14.34**
- ZIPA-CR-Large: **14.53**
- Wav2Vec2Phoneme: **12.89**
- MultIPA: **18.90**
- Allosaurus: **18.99**

해석:
- POWSM은 ZIPA 계열과 비슷한 수준
- 다만 **Wav2Vec2Phoneme가 일부 사회음성학적 변이에서 더 좋음**
- 논문은 이를 **Wav2Vec2Phoneme가 self-supervised pretraining에서 60k+ 시간 이상의 음성을 사용**했기 때문이라고 설명

즉, POWSM은 방언/L2에도 강하지만, 일부 변이에서는 self-supervised 사전학습이 강한 모델이 우위일 수 있습니다.

---

## 6. ASR 결과 해석

## 6.1 ASR 테스트셋
ASR은 주로 **FLEURS**에서 평가합니다.  
저자원 언어를 골라 다음 언어들에서 비교합니다.

- afr
- orm
- aze
- pan
- tgk
- mkd
- bos
- slv

즉, **8시간 미만의 훈련 데이터**를 가진 저자원 언어들입니다.

---

## 6.2 ASR 비교 모델
비교 대상:

1. **OWLS 0.5B**
2. **OWLS 1B**
3. **OWSM-CTC v4 1B**
4. **POWSM 0.35B, ASR**
5. **POWSM 0.35B, PR-P2G**

여기서 중요한 점은:
- POWSM은 350M 규모로 더 작은 모델
- 경쟁모델은 500M~1B로 더 크고, 웹 규모 혹은 대규모 데이터 기반

---

## 6.3 ASR 결과 해석
표 3에서 POWSM은 일반 ASR 설정만 써도 경쟁력 있지만,  
**PR에서 얻은 phone을 텍스트 prompt로 사용하는 PR-P2G 방식**이 WER를 더 낮춥니다.

### 대표적인 해석
- **POWSM 0.35B, ASR**는 web-scale multilingual ASR 모델과 비슷한 수준
- **POWSM 0.35B, PR-P2G**는 더 좋아서, 어떤 언어에서는 최고 수준
- gold phone을 쓰는 P2G가 가장 좋은 경우도 있음

즉, **phone 정보를 활용하면 저자원 ASR 성능이 향상**됨을 보여줍니다.

---

## 7. 추가 분석 결과의 의미

## 7.1 CTC encoder target 분석
논문은 encoder의 CTC 목표로 무엇을 쓰는 것이 좋은지 실험합니다.

비교:
- Unicode code points
- PanPhon phones
- suprasegmentals 포함/미포함

결과:
- **PanPhon + suprasegmentals 제거**가 가장 안정적으로 빠르게 수렴
- 이유:
  - phones를 더 자연스러운 단위로 표현
  - suprasegmental이 PR을 혼란스럽게 만들 수 있음

즉, **encoder는 fine-grained phones를 선호**하지만, 너무 세부적인 break/length 표시는 학습을 방해할 수 있음을 보입니다.

---

## 7.2 CTC weight 분석
decoder를 얼마나 믿을지에 따라 성능이 달라집니다.

- **CTC weight를 높이면**
  - out-of-domain PR 성능 향상
  - in-domain 성능은 감소 가능

해석:
- decoder는 phonotactic language modeling을 하며, 발음 변이를 “평균화”하는 경향이 있음
- 반면 CTC를 강조하면 encoder가 더 직접적으로 acoustic-phonetic alignment를 학습
- 결과적으로 **일반화는 좋아지지만 in-domain 최적성은 떨어질 수 있음**

---

## 7.3 G2P/P2G 분석
- speech-guided G2P는 **음성을 주면 phonetic variation을 더 잘 반영**
- text prompt만 쓰면 pronunciation이 더 표준화됨
- P2G는 gold phone이나 PR phone을 활용하면 low-resource language에서 상당히 강함

즉, POWSM은 **speech와 text를 함께 써서 narrow transcription과 broad transcription 사이를 조절**할 수 있습니다.

---

## 8. 종합 결론
결과를 종합하면 POWSM의 장점은 다음과 같습니다.

1. **PR에서 state-of-the-art 수준**
2. **저자원 ASR에서도 경쟁적**
3. **unseen language 일반화가 강함**
4. **방언/L2와 같은 언어 변이에도 잘 작동**
5. **G2P/P2G까지 한 모델로 수행 가능**
6. phone-level supervision이 multi-task speech model에서 유효함을 실증

즉, 이 논문은 POWSM이 단순한 ASR 모델이 아니라,  
**phone 기반 음성처리를 통합하는 범용 foundation model**임을 실험적으로 보여줍니다.

---


## 1. Overall meaning of the results
POWSM is not just a single-purpose model; it is proposed as a **multitask phonetic speech foundation model** that jointly performs:

- **Phone Recognition (PR)**
- **Automatic Speech Recognition (ASR)**
- **audio-guided Grapheme-to-Phoneme conversion (G2P)**
- **audio-guided Phoneme-to-Grapheme conversion (P2G)**

The main result claim is that POWSM:

- performs **as well as or better than** strong PR baselines,
- is **competitive with web-scale ASR models** in low-resource settings,
- generalizes well to **unseen languages** and **language variation**,
- and supports G2P/P2G in a unified framework.

---

## 2. Evaluation metrics

### 2.1 PR metric: PFER
For phone recognition, the paper uses **PFER (Phonetic Feature Error Rate)**.

Why PFER?
- It is more fine-grained than PER (Phone Error Rate)
- It uses **PanPhon articulatory features**
- It measures phonetic similarity, not just exact segment match

So, instead of only checking whether phones are identical, PFER also captures errors that are phonetically close.

### 2.2 ASR metrics: WER / CER
- **WER (Word Error Rate)** is used for most ASR experiments
- **CER (Character Error Rate)** is used for Mandarin, where character-level evaluation is more appropriate

### 2.3 P2G comparison
P2G is also evaluated with **WER**, comparing:
- gold phones
- PR-predicted phones
- ASR directly
- different prompt settings

---

## 3. Baselines / competing models

### 3.1 PR baselines
The paper compares POWSM against:

1. **Allosaurus**
   - language-agnostic PR system
   - uses allophone-to-phoneme mapping

2. **Allophant**
   - uses articulatory attributes

3. **Wav2Vec2Phoneme**
   - zero-shot cross-lingual phoneme recognition

4. **MultIPA**
   - leverages high-quality G2P data

5. **ZIPA**
   - large-scale PR model trained on IPAPack++
   - includes **ZIPA-CR-Large** and **ZIPA-CR-NS-Large**

6. **POWSM**
   - the proposed 350M model
   - jointly trained on PR, ASR, G2P, and P2G

### 3.2 ASR baselines
For ASR, the paper compares with:
- **OWLS 0.5B**
- **OWLS 1B**
- **OWSM-CTC v4 1B**
- **POWSM 0.35B, ASR**
- **POWSM 0.35B, PR-P2G**

---

## 4. Test datasets

### 4.1 In-domain data
Evaluation is done on the training distribution from **IPAPack++**, covering languages such as:
- English, German, Dutch, French, Italian, Spanish, Portuguese, Polish, Tamil, Kazakh, Mandarin

### 4.2 Out-of-domain: unseen languages
The model is also tested on:
- **DoReCo**
- **VoxAngeles**
- **Tusom2021**

These evaluate generalization to **new languages**.

### 4.3 Out-of-domain: language variation
To test dialect/L2 robustness, the paper uses:
- **Buckeye Corpus**
- **DoReCo South-England**
- **L2-ARCTIC**
- **EpaDB**
- **SpeechOcean762**

These assess **dialectal variation, L2 speech, and socio-phonetic variation**.

---

## 5. PR results

### 5.1 In-domain PR
POWSM achieves the **lowest average PFER** in the in-domain setting.

Reported average PFER:
- **POWSM: 2.62**
- ZIPA-CR-NS-Large: 2.70
- ZIPA-CR-Large: 2.99

Interpretation:
- POWSM is state-of-the-art or near state-of-the-art
- The decoder’s language modeling ability helps phonetic sequence prediction
- It remains strong despite being multi-task rather than PR-only

### 5.2 Unseen languages
On unseen languages, POWSM remains highly competitive.

Average PFER:
- **POWSM: 18.71**
- ZIPA-CR-NS-Large: 19.01
- ZIPA-CR-Large: 19.54
- Wav2Vec2Phoneme: 21.02
- MultIPA: 21.35
- Allosaurus: 32.52

Interpretation:
- POWSM generalizes better than most baselines
- It even outperforms ZIPA variants trained on similar or more data
- This suggests multitask phonetic training helps cross-lingual generalization

### 5.3 Language variation
For dialect and L2 speech, POWSM is still strong.

Average PFER:
- **POWSM: 14.40**
- ZIPA-CR-NS-Large: 14.34
- ZIPA-CR-Large: 14.53
- Wav2Vec2Phoneme: 12.89

Interpretation:
- POWSM is competitive
- However, Wav2Vec2Phoneme does better in some socio-phonetic settings, likely due to its strong self-supervised pretraining on much larger speech data

---

## 6. ASR results

### 6.1 Test data
ASR is evaluated on **low-resource languages from FLEURS**, including:
- afr, orm, aze, pan, tgk, mkd, bos, slv

These are languages with less than 8 hours of training speech in the paper’s selection scheme.

### 6.2 ASR baselines
The paper compares against:
- OWLS 0.5B
- OWLS 1B
- OWSM-CTC v4 1B
- POWSM 0.35B, ASR
- POWSM 0.35B, PR-P2G

### 6.3 ASR interpretation
POWSM is competitive with larger multilingual ASR systems, and its performance improves further when using phone information from PR as prompts in PR-P2G.

Key takeaway:
- **Phone supervision helps low-resource ASR**
- Using PR outputs as phone prompts can reduce WER substantially
- In some languages, gold-phone P2G performs even better

---

## 7. Additional analyses

### 7.1 CTC target analysis
The authors test what encoder target works best:
- Unicode code points
- PanPhon phones
- with/without suprasegmentals

Finding:
- PanPhon phones without suprasegmentals yield the best and fastest convergence

### 7.2 CTC weight analysis
Increasing CTC weight:
- improves out-of-domain generalization
- can hurt in-domain performance

This suggests a trade-off between:
- encoder-driven phonetic alignment
- decoder-driven phonotactic smoothing

### 7.3 G2P/P2G behavior
- speech-guided G2P preserves phonetic variation better
- text prompts push output toward standardized pronunciations
- P2G works well in low-resource settings, especially when phone context is available

---

## 8. Final takeaway
In short, the results show that POWSM:

1. achieves strong PR performance,
2. is competitive in low-resource ASR,
3. generalizes well to unseen languages,
4. handles dialect/L2 variation reasonably well,
5. supports G2P and P2G in the same model,
6. and demonstrates that phone-level supervision is useful for a unified speech foundation model.

---



<br/>
# 예제



## 1. POWSM이 하는 일 한눈에 보기

POWSM은 하나의 모델로 다음 4가지 작업을 동시에 수행하는 **phonetic speech foundation model**입니다.

1. **PR (Phone Recognition)**  
   음성(audio) → 음소/폰(phone) 시퀀스

2. **ASR (Automatic Speech Recognition)**  
   음성(audio) → 문자(grapheme, orthography) 시퀀스

3. **Audio-guided G2P (Grapheme-to-Phoneme)**  
   음성 + 텍스트 힌트 → 발음 기호(phone) 시퀀스

4. **Audio-guided P2G (Phoneme-to-Grapheme)**  
   음성 + phone 힌트 → 문자(grapheme) 시퀀스

즉, 같은 음성 데이터를 가지고도  
- 어떤 때는 **발음기호를 맞히고**,  
- 어떤 때는 **문자 전사본을 맞히고**,  
- 어떤 때는 **문자에서 발음을 추론하고**,  
- 어떤 때는 **발음에서 문자로 복원**합니다.

---

## 2. 학습 데이터는 어떤 형태인가?

논문에서는 **IPAPack++**라는 공개 코퍼스를 사용합니다.  
대략 **17,000시간 규모의 다국어 음성 데이터**이고, 각 발화에는 보통 다음이 같이 있습니다.

- 음성 파일
- 정답 문자 전사(orthographic transcription)
- 정답 또는 G2P로 생성된 phoneme/IPA 전사

즉, 한 발화가 예를 들면 다음과 같은 정보를 가질 수 있습니다.

- 음성: “who is that”라고 말한 오디오
- 문자 정답: `who is that`
- phone 정답: `/h/ /u/ /I/ /z/ /ð/ /æ/ /t/`

논문에서는 이런 동일한 발화를 **4가지 task format으로 재구성**합니다.

---

## 3. 각 테스크별 구체적인 입력/출력 예시

논문 본문에 나온 예시를 최대한 그대로 풀어 설명하겠습니다.

---

### 3.1 PR (Phone Recognition)

#### 목적
음성을 듣고, 그 음성이 어떤 phone sequence인지 맞히는 작업입니다.

#### 입력
- 오디오
- 언어 토큰(language token)
- task token: `<pr>`
- text prompt는 비어 있음 (`<na>`)

#### 출력
- phone sequence

#### 논문 예시
예를 들어 Buckeye corpus의 한 발화가 아래와 같다고 합시다.

- ASR transcription:  
  `any holidays at all they just kind of ignore`
- phonetic transcription(정답):  
  `/EnihAl2deIsERAlsoUDeIdZ2stkA˜ r2vIgnOô/`

이때 PR의 목표는 음성을 듣고 위의 phonetic transcription과 비슷한 phone sequence를 생성하는 것입니다.

논문 Table 6의 예시를 보면 PR 출력은 다음처럼 나타납니다.

- PR 출력 예:  
  `12.63/˜ EnihAl@deIzætOlsoUDeItS2stkh˜ æn@vIgnOô/`

여기서 중요한 점은:
- 완전히 문자로 쓰는 것이 아니라
- phone 단위의 transcription을 생성한다는 점입니다.

#### 요약
- **입력:** 음성
- **출력:** phone

---

### 3.2 ASR (Automatic Speech Recognition)

#### 목적
음성을 듣고 문자 전사본을 만드는 작업입니다.

#### 입력
- 오디오
- 언어 토큰
- task token: `<asr>`
- text prompt는 비어 있음 (`<na>`)

#### 출력
- grapheme/orthographic transcription

#### 논문 예시
위와 같은 오디오에 대해 ASR은 다음을 출력해야 합니다.

- 정답 문자 전사:  
  `any holidays at all they just kind of ignore`

즉, PR은 발음을 맞히는 작업이고, ASR은 일반적인 받아쓰기처럼 문장을 맞히는 작업입니다.

#### 요약
- **입력:** 음성
- **출력:** 문자

---

### 3.3 Audio-guided G2P

#### 목적
문자 텍스트를 주면 그 단어의 발음을 예측하는 작업인데, POWSM에서는 **음성도 함께 넣는 audio-guided 방식**입니다.  
즉, 같은 철자라도 실제로 어떻게 발음되었는지를 음성을 통해 반영할 수 있습니다.

#### 입력
- 오디오
- 문자 prompt
- 언어 토큰
- task token: `<g2p>`

#### 출력
- phone sequence

#### 논문 예시
논문에서 예시로 든 문장은:

- text prompt: `who is that`
- target output:  
  `<eng><g2p><notimestamps> /h//u//I//z//ð//æ//t/`

즉,
- 입력으로는 “who is that”라는 문자가 주어지고,
- 동시에 그 문장을 실제로 발화한 음성이 들어가며,
- 모델은 그에 대응하는 phone sequence를 출력합니다.

#### 왜 음성이 중요한가?
논문에서는 **text-only G2P**보다 **audio-guided G2P**가 더 유용할 수 있다고 설명합니다.  
이유는 실제 발음 변이가 있기 때문입니다.

예를 들어 “crayon”은 영어에서 여러 발음이 가능하지만, 사전은 하나만 줄 수 있습니다.  
반면 audio-guided G2P는 실제 발화된 음성을 보고, 그 사람/그 지역/그 억양에 가까운 발음을 반영할 수 있습니다.

#### 논문 Table 6의 핵심
Buckeye corpus 예시에서:
- speech만 넣은 G2P는 PR과 비슷한 수준의 phonetic variation을 잘 반영
- speech + text prompt를 같이 넣으면 더 표준화된 발음 쪽으로 가는 경향
- text prompt만 쓰면 더 강하게 표준 발음으로 정규화됨

#### 요약
- **입력:** 음성 + 문자 prompt
- **출력:** phone
- **특징:** 실제 발음 변이 반영 가능

---

### 3.4 Audio-guided P2G

#### 목적
이번에는 phone sequence를 바탕으로 문자를 복원하는 작업입니다.  
이것도 음성을 함께 넣는 audio-guided 방식입니다.

#### 입력
- 오디오
- phone prompt
- 언어 토큰
- task token: `<p2g>`

#### 출력
- grapheme/orthographic transcription

#### 논문 설명
논문은 P2G가 특히 **저자원 언어(low-resource languages)**에서 유용할 수 있다고 봅니다.  
왜냐하면 phone 정보가 있으면 문자를 직접 맞히는 것보다 더 안정적으로 언어 구조를 이용할 수 있기 때문입니다.

#### 예시적 이해
만약 phone prompt가 영어식 발음을 담고 있고, 음성도 함께 주어졌다면, 모델은 해당 phone과 음향을 바탕으로 문자로 복원합니다.

예:
- phone prompt: `/h/ /u/ /I/ /z/ /ð/ /æ/ /t/`
- 출력: `who is that`

#### 요약
- **입력:** 음성 + phone prompt
- **출력:** 문자

---

## 4. 학습 시 데이터는 어떻게 재구성되나?

논문에서 중요한 부분은, **하나의 발화(utterance)를 4개 테스크에 모두 재활용**한다는 점입니다.

예를 들어 어떤 발화가 있다고 합시다.

- 오디오: 사람의 실제 발화
- 문자 전사: `who is that`
- phone 전사: `/h/ /u/ /I/ /z/ /ð/ /æ/ /t/`

이 한 샘플은 다음 4개 학습 샘플로 바뀝니다.

### (1) PR용
- 입력: 오디오 + `<pr>`
- 출력: phone 전사

### (2) ASR용
- 입력: 오디오 + `<asr>`
- 출력: 문자 전사

### (3) G2P용
- 입력: 오디오 + 문자 prompt + `<g2p>`
- 출력: phone 전사

### (4) P2G용
- 입력: 오디오 + phone prompt + `<p2g>`
- 출력: 문자 전사

즉, 데이터는 하나지만 **task-specific format으로 네 번 사용**됩니다.

---

## 5. 테스트 데이터는 어떤 식으로 평가하나?

논문은 in-domain과 out-of-domain 둘 다 봅니다.

---

### 5.1 In-domain 테스트
훈련에 포함된 IPAPack++ 기반 데이터에서 평가합니다.

예:
- 영어 `eng`
- 독일어 `deu`
- 네덜란드어 `nld`
- 프랑스어 `fra`
- 이탈리아어 `ita`
- 스페인어 `spa`
- 포르투갈어 `por`
- 폴란드어 `pol`
- 타밀어 `tam`
- 카자흐어 `kaz`
- 중국어 `cmn`

여기서는 주로 **PR 성능(PFER)**을 봅니다.

---

### 5.2 Out-of-domain 테스트
훈련 때 보지 못한 언어나 언어 변이를 봅니다.

예:
- **DoReCo**
- **VoxAngeles**
- **Tusom2021**
- **Buckeye**
- **DoReCo South-England**
- **L2-ARCTIC**
- **EpaDB**
- **SpeechOcean762**

이 데이터들은 다음 특징이 있습니다.

- unseen languages 포함
- 방언(dialect) 포함
- L2 발화(비원어민 영어) 포함
- 아주 적은 자원만 있는 언어 포함

즉, 모델이 단순히 훈련 언어를 외운 건지, 아니면 **새 언어/새 변이에도 일반화하는지** 확인합니다.

---

## 6. 논문에 나온 구체 예시 정리

---

### 예시 A: Buckeye corpus에서 G2P 비교
논문 Table 6에서 Buckeye 예시가 나옵니다.

#### ASR transcription
`any holidays at all they just kind of ignore`

#### phonetic transcription
`/EnihAl2deIsERAlsoUDeIdZ2stkA˜ r2vIgnOô/`

#### 여러 조건의 G2P 출력
- **G2P (speech)**: 음성만 기반
- **G2P (both)**: 음성 + text prompt
- **G2P (text prompt)**: text만 기반

핵심 관찰:
- speech-only는 실제 발음 변이를 잘 반영
- text-only는 표준화된 발음으로 기울어짐
- speech+text는 중간이지만, 때로는 오히려 정규화가 강해짐

---

### 예시 B: PR-P2G를 ASR 보조로 사용
논문 Table 3에서 low-resource ASR에 대해 **PR-P2G**가 도움이 된다고 설명합니다.

흐름:
1. 먼저 PR이 음성에서 phone을 예측
2. 그 phone을 P2G의 입력 prompt로 사용
3. 최종적으로 문자를 생성

즉, 모델이 바로 문자만 맞히는 것보다  
**음성 → phone → 문자**라는 경로가 더 잘 작동할 수 있습니다.

---

### 예시 C: language token의 영향
논문 Table 8은 언어 토큰이 PR 결과에 영향을 준다고 보여줍니다.

예를 들어 unseen language에 대해:
- `<unk>`
- detected language token
- `<eng>`

를 각각 넣어보면 결과가 달라집니다.

즉, language token은 단순한 메타정보가 아니라  
모델 출력 분포를 특정 언어의 phonotactics 쪽으로 **유도하는 역할**을 합니다.

---

## 7. 논문이 말하는 핵심 해석

### 7.1 Encoder는 phonetic alignment를 담당
논문은 CTC encoder가 **phone-level alignment**를 잘 배우도록 설계했다고 설명합니다.

- phone을 encoder target으로 쓰면 더 안정적
- suprasegmentals(길이, break 등)를 제거한 단순 phone target이 더 빨리 수렴
- encoder가 공통 음향 패턴을 학습

### 7.2 Decoder는 language/task-specific modeling을 담당
decoder는:
- 어떤 task인지
- 어떤 언어인지
- 어떤 prompt가 주어졌는지

를 보고 최종 출력을 생성합니다.

즉,
- encoder = 음향/phonetic representation
- decoder = 출력 형식과 언어적 제약

---

## 8. 한 문장으로 정리하면

POWSM은 **음성, 문자, phone을 서로 오가게 만드는 멀티태스크 음성 기초모델**이며,  
학습 시에는 하나의 발화를 **PR / ASR / G2P / P2G** 네 형식으로 바꿔 사용하고,  
테스트 시에는 **보지 못한 언어, 방언, L2 발화**에서도 일반화되는지를 평가합니다.

---

## 1. What POWSM does at a glance

POWSM is a **phonetic speech foundation model** that performs four phone-related tasks in one unified system:

1. **PR (Phone Recognition)**  
   audio → phone sequence

2. **ASR (Automatic Speech Recognition)**  
   audio → grapheme/orthographic transcript

3. **Audio-guided G2P (Grapheme-to-Phoneme)**  
   audio + text prompt → phone sequence

4. **Audio-guided P2G (Phoneme-to-Grapheme)**  
   audio + phone prompt → grapheme/orthographic transcript

So the same speech input can be used to:
- predict phones,
- predict text,
- infer pronunciation from text using audio,
- or recover text from phones using audio.

---

## 2. What does the training data look like?

The paper trains on **IPAPack++**, a public corpus of about **17,000 hours of multilingual speech** with paired orthographic and phonemic transcriptions.

A single utterance may contain:
- audio,
- orthographic transcript,
- phoneme/IPA transcript.

For example, one utterance might have:
- audio of someone saying: “who is that”
- text transcript: `who is that`
- phoneme transcript: `/h/ /u/ /I/ /z/ /ð/ /æ/ /t/`

The paper reformulates each utterance into **four task-specific training formats**.

---

## 3. Concrete input/output examples by task

---

### 3.1 PR (Phone Recognition)

#### Goal
Listen to speech and output the corresponding phone sequence.

#### Input
- audio
- language token
- task token: `<pr>`
- blank text prompt (`<na>`)

#### Output
- phone sequence

#### Example from the paper
For a Buckeye utterance:

- ASR transcription:  
  `any holidays at all they just kind of ignore`
- phonetic transcription:  
  `/EnihAl2deIsERAlsoUDeIdZ2stkA˜ r2vIgnOô/`

PR tries to predict a phone sequence like this from audio.

#### Summary
- **Input:** audio
- **Output:** phones

---

### 3.2 ASR (Automatic Speech Recognition)

#### Goal
Listen to speech and output text.

#### Input
- audio
- language token
- task token: `<asr>`
- blank text prompt (`<na>`)

#### Output
- orthographic transcript

#### Example
For the same utterance, ASR should output:

- `any holidays at all they just kind of ignore`

#### Summary
- **Input:** audio
- **Output:** text

---

### 3.3 Audio-guided G2P

#### Goal
Predict pronunciation from text, but with audio as additional guidance.

#### Input
- audio
- text prompt
- language token
- task token: `<g2p>`

#### Output
- phone sequence

#### Example from the paper
- text prompt: `who is that`
- target output:  
  `<eng><g2p><notimestamps> /h//u//I//z//ð//æ//t/`

So the model sees both:
- the text “who is that”
- the corresponding speech signal

and predicts the pronunciation.

#### Why audio matters
The paper emphasizes that audio-guided G2P can capture **phonetic variation** better than text-only G2P.

For example, a word like “crayon” may have multiple pronunciations in American English, but a dictionary may only list one. Audio-guided G2P can reflect the actual spoken variant.

#### Summary
- **Input:** audio + text prompt
- **Output:** phones

---

### 3.4 Audio-guided P2G

#### Goal
Recover text from phone information, again with audio as guidance.

#### Input
- audio
- phone prompt
- language token
- task token: `<p2g>`

#### Output
- text

#### Example
If the phone prompt is:
- `/h/ /u/ /I/ /z/ /ð/ /æ/ /t/`

the model may output:
- `who is that`

#### Summary
- **Input:** audio + phone prompt
- **Output:** text

---

## 4. How the training data is reused

A single utterance is reused in four formats:

1. **PR**
   - Input: audio + `<pr>`
   - Output: phones

2. **ASR**
   - Input: audio + `<asr>`
   - Output: text

3. **G2P**
   - Input: audio + text prompt + `<g2p>`
   - Output: phones

4. **P2G**
   - Input: audio + phone prompt + `<p2g>`
   - Output: text

So the model learns a consistent mapping across audio, phones, and graphemes.

---

## 5. What are the test sets?

The paper evaluates both **in-domain** and **out-of-domain** data.

### In-domain
Languages in IPAPack++ seen during training, such as:
- English, German, Dutch, French, Italian, Spanish, Portuguese, Polish, Tamil, Kazakh, Mandarin

### Out-of-domain
Unseen languages and language variation:
- DoReCo
- VoxAngeles
- Tusom2021
- Buckeye
- DoReCo South-England
- L2-ARCTIC
- EpaDB
- SpeechOcean762

These test whether the model generalizes to:
- unseen languages,
- dialects,
- L2 speech,
- low-resource settings.

---

## 6. Specific examples discussed in the paper

### Example A: G2P on Buckeye
The paper compares:
- speech-only G2P
- speech + text G2P
- text-only G2P

Main finding:
- speech-only better preserves actual phonetic variation
- text-only tends to normalize pronunciations
- speech + text can also become more standardized

### Example B: PR-P2G for low-resource ASR
The model can first predict phones via PR and then use P2G to generate text.  
This can improve ASR in low-resource languages.

### Example C: language token effects
The language token influences the output distribution.  
For unseen languages, changing the language token changes PR performance, suggesting the model learns language-specific phonotactic biases.

---

## 7. Core interpretation

The paper frames POWSM as:
- **Encoder:** learns phonetic/acoustic alignment
- **Decoder:** handles task- and language-specific output generation

They also note that:
- removing suprasegmentals from phone targets helps training,
- higher CTC emphasis can help out-of-domain PR,
- language tokens affect phonotactic behavior.

---

## 8. One-sentence summary

POWSM is a unified multilingual speech model that converts between **audio, phones, and text**; during training, each utterance is reformatted into **PR, ASR, G2P, and P2G** examples, and during testing it is evaluated on **seen languages, unseen languages, dialects, and L2 speech**.

---


<br/>
# 요약


POWSM는 17,000시간 규모의 공개 다국어 음성·음소 데이터(IPAPack++)로부터, 하나의 AED(encoder-decoder) 구조에서 PR, ASR, audio-guided G2P, audio-guided P2G를 함께 학습하도록 설계했고, CTC/attention 혼합 손실과 언어 토큰·태스크 토큰을 사용했습니다.  
결과적으로 PR에서 in-domain 평균 PFER 2.62로 ZIPA 계열과 경쟁적이었고, unseen language 및 변이 데이터에서도 강한 일반화 성능을 보였으며, 저자원 ASR에서도 OWSM/OWLS와 비슷한 수준을 달성했습니다.  
예시로, Buckeye에서 speech-only G2P는 발음 변이를 잘 보존했지만 text prompt만 쓰면 발음이 더 표준화되었고, 언어 토큰을 바꾸면 PR 결과도 달라져 모델이 음운론적 패턴과 언어별 분포를 함께 학습했음을 보여줍니다.

**English ve


POWSM is trained on 17,000 hours of open multilingual speech-plus-phone data (IPAPack++) with a single AED encoder-decoder architecture, jointly learning PR, ASR, audio-guided G2P, and audio-guided P2G using a hybrid CTC/attention loss plus language/task tokens.  
As a result, it achieves a strong in-domain PR average PFER of 2.62, generalizes well to unseen languages and variation, and reaches low-resource ASR performance comparable to OWSM/OWLS.  
For example, on Buckeye, speech-only G2P preserved pronunciation variation better, while text-only prompts normalized pronunciations more; changing the language token also changed PR outputs, showing that the model learns both phonotactic patterns and language-specific output distributions.

<br/>
# 기타

---

# 1) 다이어그램 / Figure 1: POWSM 전체 구조
## 결과
- POWSM는 **하나의 통합된 프레임워크** 안에서 4가지 작업을 수행합니다:
  1. **PR (Phone Recognition)**
  2. **ASR (Automatic Speech Recognition)**
  3. **Audio-guided G2P**
  4. **Audio-guided P2G**
- 입력은 음성, 텍스트(철자), phone이며, **task token / language token / prompt**를 이용해 작업을 구분합니다.

## 인사이트
- 기존에는 PR, ASR, G2P, P2G가 각각 따로 연구됐는데, POWSM는 이를 **phonetic 중심의 하나의 모델**로 묶었습니다.
- 단순히 “멀티태스크 모델”이 아니라, **음성–phone–문자 간 변환을 하나의 구조에서 모두 처리**한다는 점이 핵심입니다.
- 특히 **phone-level representation**을 중심에 둬서, 언어 간 일반화와 저자원 환경에 유리합니다.

---

# 2) Figure 2: CTC encoder 학습 곡선
## 결과
- 인코더 CTC target으로 여러 단위를 비교했을 때,
  - **PanPhon tokenization + suprasegmental 제거**가 가장 빠르게 수렴했습니다.
- 즉, **세분화된 phone 단위**를 쓰되, 길이/억양 같은 suprasegmental 표지는 제거하는 것이 가장 효율적이었습니다.

## 인사이트
- CTC encoder는 **정확한 정렬(alignment)**을 빨리 배우는 것이 중요합니다.
- suprasegmental 정보가 많으면 PR 학습에는 오히려 혼란을 줄 수 있습니다.
- 따라서 POWSM는 **decoder는 풍부하게**, **encoder는 단순하고 정렬 가능한 단위로** 학습시키는 전략을 취했습니다.

---

# 3) Table 2: In-domain PR 성능
## 결과
- POWSM는 IPAPack++ in-domain PR에서 **가장 낮은 평균 PFER(2.62)**를 기록했습니다.
- 비교 대상:
  - ZIPA-CR-NS-Large: 2.70
  - ZIPA-CR-Large: 2.99
  - Wav2Vec2Phoneme: 11.11
  - Allophant: 16.14
  - Allosaurus: 16.14 수준
- 일부 언어에서는 ZIPA가 약간 더 좋지만, **전체 평균은 POWSM가 최고**입니다.

## 인사이트
- POWSM는 **PR 전용 모델과 비슷하거나 더 좋은 성능**을 보이면서도,
- 동시에 ASR, G2P, P2G까지 지원하는 점이 강점입니다.
- 즉, **성능 + 범용성**을 동시에 잡은 모델입니다.
- 특히 출력 decoder가 강한 language modeling 역할을 해서 PR 성능에 도움이 된 것으로 해석합니다.

---

# 4) Table 3: Low-resource ASR / PR-P2G 성능
## 결과
- 낮은 자원 언어 ASR에서 POWSM(ASR)는 web-scale ASR 모델(OWLS, OWSM)과 **비슷한 수준**입니다.
- 특히 **PR-P2G**를 쓰면 일부 언어에서 WER가 더 좋아져서, ASR보다 우수하거나 비슷한 결과를 냅니다.
- 예:
  - Afroasiatic 계열 일부 언어에서 PR-P2G가 ASR보다 확실히 개선됨.
  - 다만 언어별 차이는 존재합니다.

## 인사이트
- phone 정보를 먼저 예측한 뒤 P2G로 가는 방식이, 저자원 언어에서는 **오히려 더 강력한 텍스트 복원 방식**이 될 수 있습니다.
- 즉, ASR을 바로 하는 것보다 **phonetic intermediate representation**을 거치는 것이 유리할 수 있습니다.
- 이 결과는 “**phone이 speech-to-text의 중간 표현으로 유효하다**”는 점을 보여줍니다.

---

# 5) Table 4: Unseen language / dialect / L2 일반화
## 결과
- POWSM는 **보지 못한 언어(unseen languages)** 와 **방언, L2 발화**에서도 강한 PR 성능을 보였습니다.
- DoReCo, VoxAngeles, Tusom2021, Buckeye, DRC-SE, L2-ARCTIC, EpaDB, SpeechOcean762 등 다양한 세팅에서 비교적 안정적입니다.
- ZIPA와 비교해도 POWSM가 더 좋거나 비슷하며, 일부 경우 **추가 pseudo-labeled data를 쓴 ZIPA보다도 우수**합니다.

## 인사이트
- 이 결과는 POWSM가 단순히 학습 언어를 외우는 것이 아니라,
  **phone-level acoustic generalization**을 어느 정도 학습했다는 뜻입니다.
- 특히 **언어 변이(dialect, L2 speech)**를 다루는 데 유용합니다.
- 즉, POWSM는 “고자원 언어 인식기”보다는 **보편적 음성 phone 모델**에 가깝습니다.

---

# 6) Table 5: CTC weight 설정 실험
## 결과
- inference 시 CTC weight를 높이면:
  - **out-of-domain 성능은 좋아지지만**
  - **in-domain 성능은 나빠질 수 있음**
- 학습 시에도 높은 α_ctc를 처음부터 주는 경우가 out-of-domain에 유리했습니다.
- 반대로 fine-tuning 단계에서만 갑자기 높여도 큰 개선이 없었습니다.

## 인사이트
- decoder는 phonotactic language modeling처럼 작동해서, 때로는 **일반화를 약간 “정규화”**해버릴 수 있습니다.
- CTC를 더 강하게 주면 encoder 기반의 **순수한 음향 일반화**가 강화됩니다.
- 즉, **encoder vs decoder의 균형**이 핵심입니다.
- in-domain 최적화와 out-of-domain 일반화는 서로 trade-off 관계입니다.

---

# 7) Table 6: G2P에서 speech prompt vs text prompt
## 결과
- **speech만 넣은 G2P**는 PR과 매우 비슷한 결과를 보였습니다.
- **speech + text prompt**를 같이 넣으면 성능이 오히려 떨어지고, 더 표준화된 발음으로 출력되는 경향이 있었습니다.
- **text만 넣으면** 더 강하게 표준형 pronunciation으로 수렴했습니다.

## 인사이트
- POWSM의 G2P는 단순한 문자→음소 변환이 아니라,
  **speech evidence를 통해 실제 발음을 반영**할 수 있습니다.
- speech prompt는 narrow transcription 성격을 살리고,
- text prompt는 broad/standard pronunciation으로 유도합니다.
- 즉, 모델이 **phonetic variation과 phonological normalization 사이를 조절**할 수 있다는 점이 중요합니다.

---

# 8) Table 7: P2G 실험
## 결과
- P2G는 ASR보다 좋은 성능을 보이는 경우가 많았습니다.
- PR 결과를 입력으로 쓰는 PR-P2G는 일부 언어에서 개선되지만, 언어별 편차가 있습니다.
- 언어 토큰을 영어로 고정하고 후처리한 실험도 일부 언어에서 성능이 좋아졌습니다.

## 인사이트
- P2G도 음성 정보에 크게 의존하며, phone context를 잘 활용합니다.
- 언어별 phonotactic similarity가 성능에 영향을 줄 수 있습니다.
- 즉, 텍스트 복원도 단순히 문자 매핑이 아니라 **언어 음운 구조의 영향**을 받습니다.

---

# 9) Table 8: Language token 실험
## 결과
- unseen language에서 language token을 바꾸면 PR 성능이 달라집니다.
- detected language token을 쓰는 것이 영어 고정보다 낫고,
- **<unk>**(unknown language)가 가장 좋은 경우도 있었습니다.

## 인사이트
- language token은 단순 메타정보가 아니라, 모델의 출력 분포를 실제로 바꾸는 **phonotactic prior**로 작동합니다.
- 즉, POWSM는 언어 토큰을 통해 “이 언어에서 가능한 소리 패턴”을 내부적으로 반영합니다.
- 이는 언어 인식과 phone recognition이 분리되지 않고 상호작용함을 보여줍니다.

---

# 10) Appendix A.1: English G2P refinement
## 결과
- 영어 G2P는 다음 규칙으로 정제했습니다:
  1. word-initial 무성 파열음은 aspirated
  2. word-initial 유성 파열음은 무성화
  3. syllable-final /l/은 velarized
  4. 비음 앞 모음은 nasalization

## 인사이트
- 영어의 G2P는 표면 철자보다 실제 발음 차이가 크므로, **데이터 정제가 PR 일반화에 중요**합니다.
- 특히 VOT 문제를 바로잡아 다른 언어에 대한 혼동을 줄이려 했다는 점이 인상적입니다.

---

# 11) Appendix A.4: In-domain ASR 성능
## 결과
- ASR 성능은 완전 최상위는 아니지만, **유사 규모의 Whisper/OWSM 계열보다 경쟁력**이 있습니다.
- 특히 데이터가 많은 언어에서 더 좋고, 중간 빈도 언어에서는 격차가 큽니다.

## 인사이트
- POWSM는 ASR 전용 대형 모델은 아니지만,
  **phone recognition을 같이 학습하면서도 ASR을 충분히 수행**할 수 있습니다.
- 이는 phonetic pretraining의 이점을 보여줍니다.

---

# 12) Appendix A.5 / Table 12: 멀티태스크 스케일 실험
## 결과
- 멀티태스크가 무조건 좋지는 않았습니다.
- 데이터가 적거나 모델이 너무 크거나 작을 때 PR 성능이 악화될 수 있습니다.
- 1-task, 2-task, 4-task 사이에 뚜렷한 단조 증가 경향은 없습니다.

## 인사이트
- 멀티태스크는 항상 성능을 올리는 마법이 아니라,
  **데이터 규모와 모델 용량의 균형**이 중요합니다.
- 즉, task 수가 늘어난다고 자동으로 좋아지지 않고, 학습 신호가 잘 정렬돼야 합니다.

---

# 13) Appendix A.7 / Table 13: ASR text normalization 수정 후 비교
## 결과
- ASR 텍스트 정규화 이슈를 수정한 `POWSM-fix`는
  - ASR 성능이 개선되거나 유사했고
  - out-of-domain PR 성능도 큰 손실 없이 유지했습니다.

## 인사이트
- 성능 문제의 일부는 모델 구조가 아니라 **데이터 전처리 오류**였음을 보여줍니다.
- 따라서 고성능 speech model에서는 **데이터 정제와 normalization이 매우 중요**합니다.

---

# 최종 요약
POWSM의 핵심 인사이트는 다음과 같습니다.

1. **phone을 중심으로 ASR, PR, G2P, P2G를 통합**할 수 있다.  
2. **PR 전용 모델 수준의 성능**을 내면서도 여러 작업을 함께 지원한다.  
3. **speech prompt와 language token**이 발음 변이와 음운 일반화에 의미 있게 작동한다.  
4. **CTC/decoder 균형**이 in-domain vs out-of-domain 성능을 좌우한다.  
5. 멀티태스크와 phonetic supervision은 **저자원·미지 언어·방언·L2 speech**에 특히 유리하다.  

---



## 1) Diagram / Figure 1: Overall POWSM architecture
### Results
- POWSM is a unified framework for four tasks:
  1. **Phone Recognition (PR)**
  2. **Automatic Speech Recognition (ASR)**
  3. **Audio-guided G2P**
  4. **Audio-guided P2G**
- It uses task tokens, language tokens, and prompts to distinguish tasks.

### Insights
- Unlike prior work that treats these tasks separately, POWSM unifies them in a **single phonetic framework**.
- The key idea is to model the relations among **speech, phones, and graphemes** jointly.
- This makes the model more suitable for multilingual and low-resource settings.

---

## 2) Figure 2: CTC encoder learning curves
### Results
- The best encoder target was **PanPhon tokenization without suprasegmentals**.
- This setting led to the fastest convergence.

### Insights
- The encoder benefits from a cleaner, more aligned phonetic target.
- Suprasegmentals can confuse PR-oriented CTC training.
- The model therefore separates roles: the **encoder learns simplified alignment**, while the **decoder handles richer output modeling**.

---

## 3) Table 2: In-domain PR results
### Results
- POWSM achieved the best average in-domain PFER: **2.62**.
- It outperformed or matched specialized PR systems such as ZIPA, Wav2Vec2Phoneme, Allophant, and Allosaurus.

### Insights
- POWSM matches specialized PR models while also supporting ASR, G2P, and P2G.
- This shows that a multi-task phonetic model can be both **accurate and general-purpose**.

---

## 4) Table 3: Low-resource ASR and PR-P2G
### Results
- POWSM ASR is comparable to web-scale ASR models.
- PR-P2G often improves WER further, sometimes surpassing ASR.

### Insights
- Phones can serve as an effective intermediate representation for speech-to-text in low-resource settings.
- This suggests that phonetic supervision helps bridge speech and orthography.

---

## 5) Table 4: Generalization to unseen languages and variation
### Results
- POWSM performs strongly on unseen languages, dialects, and L2 speech.
- It often matches or beats ZIPA, even when ZIPA uses extra pseudo-labeled data.

### Insights
- POWSM learns more than memorized language-specific patterns.
- It captures more general acoustic and phonetic structure, which helps with cross-lingual transfer.

---

## 6) Table 5: CTC weight experiments
### Results
- Higher CTC weight improves out-of-domain PR but can hurt in-domain PR.
- Changing CTC weight only during fine-tuning does not help much; using a higher weight from the start is better for OOD data.

### Insights
- There is a clear trade-off between **in-domain accuracy** and **generalization**.
- The decoder seems to behave like a phonotactic language model, which can smooth rare phonetic patterns.

---

## 7) Table 6: Speech vs text prompts in G2P
### Results
- Speech-only G2P preserves phonetic variation best.
- Adding text prompts pushes outputs toward more standardized pronunciations.
- Text-only G2P performs worst and is the most normalized.

### Insights
- POWSM can control the balance between **narrow phonetic detail** and **broad standardized pronunciation**.
- Speech prompts help capture real pronunciation variation.

---

## 8) Table 7: P2G experiments
### Results
- P2G often performs better than ASR in low-resource languages.
- PR-P2G helps in some languages but not all.

### Insights
- P2G relies heavily on both speech input and phonotactic structure.
- Language-specific phonological patterns matter for text reconstruction.

---

## 9) Table 8: Language token effects
### Results
- Changing the language token changes PR behavior on unseen languages.
- Using the detected language token helps more than always using English.
- In some cases, the unknown token performs best.

### Insights
- Language tokens act as phonotactic priors.
- They influence the output distribution rather than simply providing metadata.

---

## 10) Appendix A.1: English G2P refinement
### Results
- The authors manually refined English G2P to better handle aspiration, voicing, velarized /l/, and vowel nasalization.

### Insights
- Data cleaning matters a lot for cross-lingual phonetic generalization.
- Small transcription artifacts can affect the model’s learned phonetic space.

---

## 11) Appendix A.4: In-domain ASR performance
### Results
- POWSM is competitive with Whisper/OWSM models of similar size.

### Insights
- Even though POWSM is phonetics-centered, it remains a strong ASR system.
- Phonetic supervision appears to improve speech-to-text generalization.

---

## 12) Appendix A.5 / Table 12: Multi-task scaling
### Results
- More tasks do not automatically improve PR.
- Performance depends on data scale and model capacity.

### Insights
- Multi-task learning is not universally beneficial; balance matters.
- Too little data or too much capacity can hurt performance.

---

## 13) Appendix A.7 / Table 13: Fixed ASR normalization
### Results
- After fixing text normalization issues, ASR improved and OOD PR remained stable.

### Insights
- Some errors came from preprocessing, not the model itself.
- Data normalization is crucial for speech foundation models.

---

## Overall takeaway
POWSM shows that a **single phonetic foundation model** can unify PR, ASR, G2P, and P2G effectively.  
Its main strength is not only strong in-domain performance, but also **cross-lingual robustness, low-resource utility, and interpretable phonetic behavior**.


<br/>
# refer format:

## 1) BibTeX

```bibtex
@article{li2026powsm,
  title={POWSM: A Phonetic Open Whisper-Style Speech Foundation Model},
  author={Li, Chin-Jou and Chang, Kalvin and Bharadwaj, Shikhar and Yeo, Eunjung and Choi, Kwanghee and Zhu, Jian and Mortensen, David R. and Watanabe, Shinji},
  journal={arXiv preprint arXiv:2510.24992v2},
  year={2026},
  month={jan},
  note={Released 16 Jan 2026}
}
```

## 2) 시카고 스타일(줄글형)

Li, Chin-Jou, Kalvin Chang, Shikhar Bharadwaj, Eunjung Yeo, Kwanghee Choi, Jian Zhu, David R. Mortensen, and Shinji Watanabe. “POWSM: A Phonetic Open Whisper-Style Speech Foundation Model.” arXiv preprint arXiv:2510.24992v2 (January 16, 2026).


