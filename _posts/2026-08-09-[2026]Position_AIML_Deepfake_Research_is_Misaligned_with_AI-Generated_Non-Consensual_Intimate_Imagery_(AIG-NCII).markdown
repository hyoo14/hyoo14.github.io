---
layout: post
title:  "[2026]Position: AI/ML Deepfake Research is Misaligned with AI-Generated Non-Consensual Intimate Imagery (AIG-NCII)"
date:   2026-08-09 17:18:20 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 딥페이크 연구가 윤리적으로 더 나아가야할 방향 제시  


짧은 요약(Abstract) :


이 논문은 **AI/ML 분야의 딥페이크 연구가 실제로 많이 발생하는 피해인 AIG-NCII를 충분히 다루지 못하고 있다**고 주장합니다. AIG-NCII는 특정 인물의 동의 없이 AI로 성적 이미지나 영상을 만들어 유포하는 행위입니다.

현재 딥페이크 연구는 주로 다음과 같은 문제에 집중합니다.

- 가짜 뉴스와 정치적 허위정보
- 사기와 금융 범죄
- 이미지나 영상이 진짜인지 가짜인지 판별하는 문제

이러한 피해는 주로 콘텐츠를 보는 사람이 속거나 사회적 신뢰가 무너지는 **“시청자 중심의 인식적 피해”**입니다. 반면 AIG-NCII의 핵심 피해는 이미지의 진위 여부가 아니라, 당사자의 동의 없이 얼굴과 신체가 성적으로 이용되어 존엄성과 자율성이 침해되는 **“당사자 중심의 존엄성 피해”**입니다.

논문은 많이 인용된 딥페이크 연구들을 분석한 결과, 대부분의 연구가 AIG-NCII를 언급하지 않거나 기술적으로 대응하지 않는다는 점을 보여줍니다. 특히 현재의 탐지·출처 추적·워터마킹 기술은 콘텐츠가 합성되었는지는 알려줄 수 있지만, 그 이미지가 **동의 없이 만들어졌는지**까지 판단하지는 못합니다. 따라서 “AI로 만든 가짜”라고 표시하는 것만으로는 피해자를 보호할 수 없으며, 오히려 이미지가 계속 노출되거나 가해자가 피해자를 더 쉽게 식별하는 등 피해를 키울 수도 있습니다.

저자들은 연구 방향을 다음과 같이 바꿔야 한다고 제안합니다.

1. 진위성 문제와 존엄성 침해를 별개의 피해로 다룰 것  
2. 공개 라벨링보다 삭제·차단·검토를 위한 내부 탐지에 집중할 것  
3. 사람의 신원을 복제하거나 옷을 제거하는 생성 기술에 대한 예방책을 개발할 것  
4. 탐지 정확도보다 실제 피해 감소를 성과 기준으로 삼을 것  
5. AI 안전 연구에 AIG-NCII를 포함할 것  
6. 성폭력 예방 전문가와 피해자·생존자 옹호 단체와 협력할 것  

결국 이 논문은 AI 안전이 단순히 **“무엇이 진짜인가”**만 보호해서는 안 되며, **“누가 동의 없이 피해를 입고 있는가”**와 인간의 존엄성까지 보호해야 한다고 강조합니다.

---




This paper argues that AI/ML research on deepfakes is poorly aligned with one of the most common and harmful uses of generative AI: **AI-generated non-consensual intimate imagery (AIG-NCII)**. AIG-NCII refers to sexualized images or videos of identifiable people that are created or altered without their consent.

Current deepfake research mainly focuses on:

- misinformation and political manipulation,
- fraud and scams, and
- detecting whether media is authentic or synthetic.

These concerns involve **viewer-centric epistemic harms**, such as being deceived or losing trust in shared information. AIG-NCII, however, creates **subject-centric dignity harms**. The central problem is not whether the image is real or fake, but that a person’s identity and body are used sexually without consent.

The authors’ analysis of highly cited deepfake research shows that most studies either do not mention AIG-NCII or do not develop technical defenses specifically for it. Existing tools—including detection, provenance tracking, and watermarking—can indicate whether content is AI-generated, but they generally cannot determine whether the content was created consensually. As a result, labeling an image as “synthetic” may fail to protect the victim and can sometimes worsen the harm by keeping the image visible or helping abusers identify and organize abusive content.

The paper recommends that researchers:

1. separate authenticity-related harms from dignity-related harms;
2. use detection mainly for private backend moderation, suppression, or triage rather than public labeling;
3. develop proactive defenses against identity preservation and nudification;
4. evaluate success by actual harm reduction, not detection accuracy alone;
5. include AIG-NCII in AI safety research; and
6. work with experts in sexual-violence prevention and victim-survivor advocacy.

The main conclusion is that AI safety should protect not only **truth and authenticity**, but also **consent, autonomy, and human dignity**.


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



### 1. 논문의 성격

이 논문은 새로운 딥페이크 탐지 모델을 제안하는 연구가 아니라, 기존 AI/ML 딥페이크 연구가 **AI 생성 비동의 친밀 이미지(AIG-NCII)** 문제와 얼마나 동떨어져 있는지를 분석하는 **포지션 논문**이다.

논문이 다루는 방법은 크게 다음 세 가지다.

1. 기존 연구에 대한 **문헌 지형 분석**
2. 딥페이크 대응 기술인 **탐지·출처 추적·워터마킹 분석**
3. AIG-NCII를 줄이기 위한 **예방·보호 기술 및 연구 방향 제안**

---

### 2. 문헌 지형 분석 방법

저자들은 2020~2025년의 딥페이크 및 합성 미디어 방어 연구를 체계적으로 조사했다.

#### 검색 및 선별 과정

- Google Scholar에서 다음과 같은 키워드를 조합해 검색했다.
  - 기술 키워드: `detection`, `detector`, `forensics`, `recognition`, `watermark`
  - 대상 키워드: `deepfake`, `synthetic image`, `fake image`, `diffusion`
- 최초 검색 결과: **965편**
- CVPR, ICCV, ECCV, NeurIPS, ICML, ICLR 등 주요 학회 논문으로 제한
- 다른 학회 논문 중 80회 이상 인용된 논문도 포함
- 최종적으로 인용 수가 높은 논문을 추린 뒤, 종양·자동차·철강 균열 탐지처럼 딥페이크와 무관한 논문을 제외
- 최종 분석 대상: **39편**

각 논문에서 다음 용어가 AIG-NCII와 관련되어 언급되었는지 확인했다.

- NCII
- revenge porn
- sexual violence
- porn
- nudity
- undress
- obscene

#### 분석 결과

39편은 다음 세 등급으로 분류되었다.

- **언급 없음:** 34편  
  주로 허위정보, 사기, 정치적 조작 또는 합성 흔적 탐지만 다룸
- **단순 언급:** 5편  
  서론이나 영향 분석에서 AIG-NCII를 언급하지만, 기술 자체는 일반적인 합성 이미지 탐지
- **AIG-NCII에 특화된 기술 구현:** 0편

즉, 기존 연구는 실제 피해자인 이미지 속 인물보다, 이미지를 보는 사람의 “진짜인가 가짜인가” 판단에 집중했다.

---

### 3. AIG-NCII 생성에 사용되는 모델과 기법

논문은 AIG-NCII 생성 기술이 어떻게 발전했는지도 설명한다.

#### 3.1 얼굴 교체와 오토인코더

초기에는 주로 얼굴을 다른 신체나 영상에 합성하는 방식이 사용되었다.

- 오토인코더 기반 얼굴 교체
- 대표적인 오픈소스 도구: **DeepFaceLab**
- 특정 인물의 얼굴 특징을 학습한 뒤 다른 영상이나 신체에 삽입

이 방식은 주로 **얼굴 정체성(identity)을 보존하면서 대상의 신체나 행동을 변경**하는 데 사용된다.

#### 3.2 GAN과 Pix2Pix

초기 “옷 제거” 도구인 DeepNude는 다음과 같은 이미지 변환 기술과 관련되어 있다.

- 조건부 GAN(Conditional GAN)
- 대표적 구조: **Pix2Pix**
- 입력 이미지의 의미를 유지하면서 특정 영역을 변환
- 결과적으로 옷을 제거한 것처럼 보이는 이미지를 생성

#### 3.3 확산 모델

최근에는 GAN보다 **확산 모델(diffusion model)**이 AIG-NCII 생성의 중심이 되었다.

대표적인 기반 모델은 다음과 같다.

- Stable Diffusion
- Latent Diffusion Model

확산 모델은 텍스트 프롬프트를 통해 성적 이미지나 특정 상황의 이미지를 생성할 수 있다. 따라서 단순한 얼굴 교체를 넘어, 대상의 자세·복장·배경·신체 표현 등을 새롭게 만들어낼 수 있다.

#### 3.4 DreamBooth와 LoRA

특정 개인의 얼굴이나 외모를 모델에 학습시키기 위해 다음 기법들이 사용된다.

- **DreamBooth:** 소수의 참조 사진으로 특정 인물의 정체성을 모델에 맞춤 학습
- **LoRA:** 전체 모델을 다시 학습하지 않고 일부 저랭크 파라미터만 조정하여 특정 인물의 특징을 학습

이 기법들은 적은 수의 사진만으로도 특정 인물의 외모를 재현할 수 있게 하므로, AIG-NCII 제작의 기술적 장벽을 크게 낮춘다.

#### 3.5 학술 연구와 오픈소스의 연결

논문은 학술 논문에서 공개된 코드와 모델이 실제 AIG-NCII 도구에 직접 활용되고 있다고 지적한다.

- 오픈소스 연구 코드가 그대로 포크되거나 수정됨
- 일부 상용·비공식 애플리케이션은 연구 코드를 감싼 래퍼(wrapper)에 가까움
- 비동의로 수집된 누드 이미지가 학습 데이터에 포함될 경우, 이러한 생성 능력이 강화될 수 있음

---

### 4. 기존 방어 기술의 세 가지 방법

논문은 기존 AI/ML 딥페이크 방어 기술을 세 가지 범주로 나눈다.

#### 4.1 탐지(Detection)

탐지 모델은 이미지가 실제 이미지인지 합성 이미지인지 분류한다.

일반적인 학습 방식은 다음과 같다.

- 실제 이미지와 생성 이미지를 데이터셋으로 구성
- 두 분포를 구분하는 결정 경계(decision boundary)를 학습
- 입력 이미지가 어느 쪽에 속하는지 판별

주요 특징 추출 방식은 생성 모델의 종류에 따라 달라진다.

- GAN 이미지:
  - 주파수 영역의 이상 패턴
  - 생성 과정에서 생긴 시각적 아티팩트
- 확산 모델 이미지:
  - 역확산 또는 재구성 오차
  - 반복적 노이즈 제거 과정에서 생긴 흔적
  - 주파수·스펙트럼 특성
- 범용 탐지:
  - CLIP과 같은 비전-언어 모델의 특징 공간을 활용
  - 여러 생성 모델에 일반화되는 의미적 패턴을 찾음

대표적 예로 논문은 DIRE를 언급한다. DIRE는 확산 과정을 거꾸로 적용했을 때 합성 이미지와 실제 이미지의 재구성 오차가 다르다는 점을 이용한다.

**한계:**  
탐지는 “합성 여부”는 판별할 수 있지만, 그 이미지가 동의에 기반했는지 또는 피해자의 존엄성을 침해하는지는 판단하지 못한다.

---

#### 4.2 출처 추적(Provenance)

출처 추적 방식은 이미지가 어디에서 만들어지고 어떻게 수정되었는지 기록한다.

주요 요소는 다음과 같다.

- 이미지의 픽셀 데이터에 대한 해시
- 생성·수정 시점
- 소유권이나 편집 이력
- 암호학적 디지털 서명
- 콘텐츠 메타데이터와 매니페스트

C2PA와 같은 표준은 이미지의 **생성 및 수정 이력(chain of custody)**을 확인하려는 방식이다.

**기본 가정:**  
미디어의 출처와 변경 이력을 검증할 수 있으면 그 미디어의 신뢰성을 판단할 수 있다는 것이다.

**한계:**  
출처가 확인되더라도, 동의 없이 특정 인물의 모습을 성적으로 변형했다는 피해 자체는 사라지지 않는다.

---

#### 4.3 워터마킹(Watermarking)

워터마킹은 이미지 생성 단계에서 사람이 보기 어려운 신호를 이미지 안에 삽입한다.

주요 방식은 다음과 같다.

- 잠재공간(latent space)에 워터마크 삽입
- 이미지 샘플링 과정에 워터마크 패턴 삽입
- 이미지의 일부가 잘리거나 필터가 적용되어도 신호가 남도록 설계
- 전용 탐지기나 디코더로 워터마크 확인

예시로 Google의 **SynthID**가 언급된다.

**용도:**  
생성된 미디어를 사후에 식별하거나 AI 생성 여부를 표시하는 데 사용된다.

**한계:**  
“AI가 만든 이미지”라는 표시가 이미지의 삭제, 확산 차단, 피해자 보호를 자동으로 보장하지는 않는다.

---

### 5. 논문이 제안하는 보호·예방 방법

논문은 단순히 합성 여부를 표시하는 것보다, 피해를 직접 줄이는 방향으로 연구 목표를 바꿔야 한다고 주장한다.

#### 5.1 백엔드 플래그와 콘텐츠 억제

AIG-NCII가 의심되는 콘텐츠를 공개적으로 “AI 생성”이라고 표시하기보다,

- 플랫폼 내부 플래그로 처리
- 노출 억제
- 검토 또는 분류(triage)
- 기존 NCII와 유사하게 삭제·차단

하는 방식을 제안한다.

공개 라벨은 콘텐츠를 계속 노출시키거나, 가해자가 “가짜임을 밝혔으므로 문제없다”고 주장하게 만들 수 있기 때문이다.

#### 5.2 정체성 보존 방지

탐지 정확도를 높이는 대신, 특정 개인의 정체성이 모델에 의해 재현되지 않도록 하는 방법을 강조한다.

- 모델이 개인의 얼굴 특징을 학습하지 못하게 만들기
- 특정 인물의 외모 복제를 어렵게 만들기
- 개인 식별성이 유지되지 않도록 생성 결과를 교란하기

이 접근은 **합성 이미지 판별**보다 **개인 정체성의 비동의 재현 방지**에 초점을 둔다.

#### 5.3 적대적 이미지 보호(Adversarial Immunization)

사진에 사람이 인지하기 어려운 작은 변형을 넣어 생성 모델의 내부 표현을 방해하는 방법이다.

목표는 다음과 같다.

- 스타일 모방 방해
- DreamBooth 기반 개인화 학습 방해
- 인페인팅을 통한 옷 제거 방해
- 특정 인물의 얼굴 특징 추출 방해

대표적으로 Anti-DreamBooth, Glaze, AdvPaint, DiffVax, Anti-inpainting 계열의 연구가 관련된다.

이 방식은 완벽한 차단보다는 가해자가 우회하기 위해 더 많은 기술과 비용을 들이게 하는 **마찰(friction)**을 제공하는 것이 목적이다.

#### 5.4 고위험 모델과 코드의 접근 제한

특정 인물의 외모를 소수 사진만으로 복제하는 모델이나 파인튜닝 기법은 무제한 공개하기보다 다음과 같은 접근을 제안한다.

- 연구자 인증 기반 접근
- 제한적 배포
- 모델 가중치와 파인튜닝 코드의 비공개
- 고위험 기능에 대한 게이팅(gating)

#### 5.5 안전 중심 평가 지표

성능을 단순한 정확도나 F1 점수로 평가하지 않고 다음을 측정해야 한다고 주장한다.

- 실제 AIG-NCII 발생량 감소
- 피해 콘텐츠의 노출 및 확산 감소
- 오탐으로 인한 합법적·동의된 콘텐츠의 피해
- 미탐으로 인한 성폭력 피해
- 피해자 개인정보와 안전의 보호 정도

즉, 목표 지표는 **탐지 정확도**가 아니라 **피해 감소량**이어야 한다.

---

### 6. 핵심적인 방법론적 주장

이 논문의 가장 중요한 방법론적 주장은 다음과 같다.

> 합성 여부와 안전 여부는 서로 다른 축이다.

기존 탐지 모델은 다음만 구분한다.

- 실제 이미지인가?
- AI 생성 이미지인가?

그러나 AIG-NCII에서 중요한 기준은 다음이다.

- 이미지 속 인물이 동의했는가?
- 특정 인물의 정체성이 비동의로 사용되었는가?
- 이미지의 유통이 피해자의 존엄성과 자율성을 침해하는가?

따라서 **authenticity(진위성)**를 안전성의 대리 지표로 사용할 수 없다. 논문은 이를 “authentic ≠ safe”라는 논리로 요약한다.

---




### 1. Nature of the Paper

This is a **position paper**, not a paper proposing a new deepfake detection architecture. Its main goal is to show that existing AI/ML deepfake research is misaligned with the dominant harms associated with **AI-generated non-consensual intimate imagery (AIG-NCII)**.

The paper uses three main methodological components:

1. A **landscape analysis** of existing literature
2. A review of three authenticity-based defenses:
   - detection
   - provenance
   - watermarking
3. Recommendations for **prevention-oriented and safety-aligned interventions**

---

### 2. Literature Landscape Analysis

The authors examined technical defense papers published between 2020 and 2025.

#### Search and filtering procedure

They searched Google Scholar using combinations of:

- Technical terms: `detection`, `detector`, `forensics`, `recognition`, `watermark`
- Media terms: `deepfake`, `synthetic image`, `fake image`, `diffusion`

The procedure was:

- Initial search: **965 papers**
- Filtering to major venues such as CVPR, ICCV, ECCV, NeurIPS, ICML, and ICLR
- Including highly cited papers from other venues
- Removing papers using diffusion models for unrelated tasks
- Final qualitative dataset: **39 papers**

The authors checked whether papers mentioned terms such as:

- NCII
- revenge porn
- sexual violence
- pornography
- nudity
- undressing

#### Results

The 39 papers were divided into three categories:

- **No mention:** 34 papers
- **Passing mention:** 5 papers
- **Technical implementation specifically addressing AIG-NCII:** 0 papers

This demonstrates that existing research mainly focuses on whether media is real or fake, rather than on whether a person’s likeness was used without consent.

---

### 3. Models and Techniques Used to Create AIG-NCII

#### 3.1 Face swapping and autoencoders

Early AIG-NCII systems mainly used face-swapping techniques.

- Autoencoder-based face replacement
- Tools such as DeepFaceLab
- Learning a person’s facial identity and placing it onto another body or video

The key capability is **preserving a recognizable identity while changing the surrounding body or context**.

#### 3.2 GANs and Pix2Pix

Earlier “undressing” tools such as DeepNude were associated with conditional GAN architectures, particularly Pix2Pix.

- Conditional image-to-image translation
- Transforming selected regions of an image
- Producing an image that appears to remove clothing

#### 3.3 Diffusion models

More recent AIG-NCII generation is largely driven by diffusion models.

- Stable Diffusion
- Latent diffusion models
- Text-to-image generation
- Image-to-image transformation and inpainting

Diffusion models allow users to modify clothing, pose, body appearance, and context instead of merely replacing a face.

#### 3.4 DreamBooth and LoRA

Personalized generation methods make it possible to reproduce a specific person from a small number of reference images.

- **DreamBooth:** fine-tunes a text-to-image model to represent a particular individual
- **LoRA:** adjusts a small set of low-rank parameters instead of retraining the entire model

These techniques significantly reduce the technical barrier to identity-specific image generation.

#### 3.5 Open-source research code and training data

The paper argues that academic code and models are often reused in real-world AIG-NCII tools.

- Research repositories may be forked directly
- Some applications act as wrappers around open-source research implementations
- Non-consensually collected nude or sexualized images may contribute to unsafe model capabilities

---

### 4. Three Existing Defense Paradigms

#### 4.1 Detection

Detection models classify media as authentic or synthetic.

Typical process:

- Train on real and generated images
- Learn a decision boundary between the two distributions
- Predict whether an input image is synthetic

Examples of features include:

- GAN artifacts and frequency-domain patterns
- Reconstruction error through an inverse diffusion process
- Spectral traces from diffusion noise schedules
- CLIP or vision-language feature spaces for cross-model generalization

For example, DIRE uses differences in reconstruction error after reversing a diffusion process.

**Limitation:**  
Detection can estimate whether an image is synthetic, but it cannot determine whether the subject consented to the image or whether the image violates the subject’s dignity.

---

#### 4.2 Provenance

Provenance systems record the history of a media asset.

They may include:

- Pixel-data hashes
- Digital signatures
- Timestamps
- Ownership information
- Editing history
- Metadata manifests

Standards such as C2PA aim to provide a cryptographically verifiable chain of custody.

**Limitation:**  
Even if the origin of an image is known, that does not remove the harm caused by using a person’s likeness sexually without consent.

---

#### 4.3 Watermarking

Watermarking embeds an invisible or difficult-to-remove signal into generated media.

Methods include:

- Latent-space watermarking
- Sampling-based watermarking
- Watermarks designed to survive cropping, filtering, or other transformations
- Detection through a dedicated decoder or detector

Google’s SynthID is given as an industry example.

**Limitation:**  
A label saying “AI-generated” does not automatically remove the content, stop its distribution, or protect the person depicted.

---

### 5. Prevention and Protection Methods Recommended by the Paper

#### 5.1 Backend flagging and suppression

Instead of publicly labeling suspected AIG-NCII as “AI-generated,” platforms should use detection as a backend signal to trigger:

- Suppression
- Triage
- Human review
- Removal or restriction similar to traditional NCII

Public labeling may leave the harmful content visible and may protect abusers from moderation if they openly admit that the image is synthetic.

#### 5.2 Preventing identity preservation

The paper recommends shifting the goal from maximizing detection accuracy to minimizing unauthorized identity reproduction.

Possible approaches include:

- Preventing models from learning a person’s facial features
- Disrupting reproduction of a specific identity
- Making identity-specific fine-tuning more difficult
- Reducing the recognizability of the subject in generated outputs

#### 5.3 Adversarial image immunization

These methods add imperceptible perturbations to images in order to interfere with generative models.

Potential targets include:

- Style imitation
- DreamBooth personalization
- Identity extraction
- Inpainting-based undressing

Related research includes Anti-DreamBooth, Glaze, AdvPaint, DiffVax, and anti-inpainting methods.

The objective is not perfect protection, but **friction**: increasing the technical cost and difficulty of abuse.

#### 5.4 Restricting high-risk models and code

Models that can reproduce a person’s likeness from only a few photographs should not necessarily be released without restrictions.

Suggested controls include:

- Gated access
- Researcher-only access
- Restricted model weights
- Limited release of fine-tuning code
- Additional safeguards for high-risk capabilities

#### 5.5 Harm-reduction metrics

The paper argues that evaluation should go beyond accuracy, precision, recall, or F1 score.

Relevant metrics should include:

- Reduction in the prevalence of AIG-NCII
- Reduction in exposure and redistribution
- False-positive costs for consensual content
- False-negative costs for victims
- Protection of victim privacy and safety

The ultimate objective should be **harm reduction**, not merely better synthetic-image classification.

---

### 6. Central Methodological Claim

The central claim is:

> Authenticity and safety are different dimensions.

Existing detectors mainly ask:

- Is the image authentic?
- Is the image AI-generated?

For AIG-NCII, the more important questions are:

- Did the subject consent?
- Was the subject’s identity used without permission?
- Does the distribution violate the subject’s dignity and autonomy?

Therefore, **authenticity cannot be treated as a sufficient proxy for safety**. The paper summarizes this distinction as:

> **Authentic ≠ Safe**


<br/>
# Results



### 1. 연구의 성격
이 논문은 새로운 딥페이크 탐지 모델의 성능을 비교한 실험 논문이 아니라, **기존 AI/ML 딥페이크 연구가 AIG-NCII를 얼마나 다루고 있는지 분석한 포지션 논문**입니다. 따라서 일반적인 의미의 경쟁 모델, 테스트 데이터셋, 정확도 비교표는 제시되지 않습니다.

---

### 2. 문헌 분석 대상과 선정 과정

논문은 2020~2025년에 발표된 딥페이크·합성미디어 방어 연구를 대상으로 문헌 분석을 수행했습니다.

| 단계 | 논문 수 |
|---|---:|
| Google Scholar 키워드 검색 결과 | 965편 |
| 주요 학회 및 인용 수 기준 필터링 후 | 379편 |
| 상위 인용 논문 선정 | 100편 |
| 관련 없는 확산모델 연구 및 철회 논문 제외 후 최종 분석 | **39편** |

검색 키워드는 다음과 같은 범주를 포함했습니다.

- 탐지: detection, detector, forensics, recognition
- 출처·표시: watermark
- 대상: deepfake, synthetic image, fake image, diffusion

주요 학회로는 CVPR, ICCV, ECCV, NeurIPS, ICML, ICLR 등이 포함되었습니다. 또한 다른 학회 논문 중 80회 이상 인용된 논문도 일부 포함했습니다.

---

### 3. 결과: AIG-NCII를 다룬 정도

최종 39편을 AIG-NCII 관련성에 따라 세 단계로 분류했습니다.

| 분류 | 논문 수 | 비율 | 의미 |
|---|---:|---:|---|
| 언급 없음 | 34편 | 약 87.2% | 허위정보, 사기, 정치적 조작 또는 기술적 흔적 탐지에만 초점 |
| 단순 언급 | 5편 | 약 12.8% | 영향·한계·배경에서 AIG-NCII를 잠깐 언급하지만 기술 설계에는 반영하지 않음 |
| AIG-NCII 특화 기술 구현 | 0편 | 0% | AIG-NCII를 명시적 위협 모델로 삼은 방어 기술은 없음 |

즉, 논문의 핵심 결과는 **39편 중 AIG-NCII를 기술적으로 직접 다룬 연구가 전혀 없었다는 것**입니다.

부록의 Table 2에서는 39편의 개별 논문을 나열하고, AIG-NCII 관련 용어의 언급 여부를 표시합니다. 그러나 단순히 용어가 등장했다고 해서 해당 연구가 AIG-NCII 방어를 수행한 것은 아닙니다.

---

### 4. 경쟁 모델 및 비교 대상

#### 경쟁 모델
이 논문에는 탐지 모델 간의 직접적인 경쟁 비교가 없습니다. 예를 들어 다음과 같은 방식의 비교는 수행하지 않았습니다.

- Model A vs. Model B
- CNN vs. Transformer
- GAN 탐지기 vs. Diffusion 탐지기
- 정확도, F1-score, AUROC 순위 비교

다만 기존 연구에서 사용되는 기술적 방어 패러다임을 세 가지로 묶어 비교·비판합니다.

#### 기존 방어 패러다임

1. **Detection**
   - 이미지가 실제인지 합성인지 분류
   - GAN의 주파수 흔적이나 diffusion 모델의 노이즈·복원 흔적 등을 이용
   - 예: DIRE, diffusion fingerprint, CLIP 기반 탐지

2. **Provenance**
   - 이미지 생성·수정 이력을 기록
   - C2PA와 같은 암호학적 서명 및 메타데이터로 출처와 변경 과정을 검증

3. **Watermarking**
   - 생성 시 이미지에 보이지 않는 신호를 삽입
   - SynthID, latent watermarking, sampling-based watermarking 등이 해당

이 세 방식은 모두 주로 **“이 이미지가 진짜인가, 합성인가?”**를 판단하기 위한 기술입니다.

---

### 5. 테스트 데이터와 평가 메트릭

#### 테스트 데이터
논문 자체는 새로운 테스트셋을 만들거나, 39개 연구의 데이터셋을 통합하여 재평가하지 않았습니다. 또한 실제 AIG-NCII 이미지를 생성하거나 수집해 모델을 테스트하지도 않았습니다.

따라서 다음과 같은 정보는 논문에 제시되지 않습니다.

- 공통 테스트 데이터셋
- 학습·검증·테스트 분할
- AIG-NCII 전용 벤치마크
- 피해자 보호를 위한 표준 평가셋

#### 평가 메트릭
탐지 모델의 성능을 직접 측정한 논문이 아니므로 다음과 같은 수치 비교도 없습니다.

- Accuracy
- Precision / Recall
- F1-score
- AUROC
- False Positive Rate
- False Negative Rate

대신 문헌 분석의 핵심 평가지표는 **AIG-NCII에 대한 연구 참여 수준**이었습니다.

- AIG-NCII 언급 여부
- 단순 언급인지 여부
- AIG-NCII 특화 위협 모델과 기술 구현 여부

---

### 6. 논문이 제시하는 핵심 비교

논문은 기존 딥페이크 연구의 안전 기준과 AIG-NCII의 실제 피해 기준이 서로 다르다고 비교합니다.

| 기존 연구가 주로 측정하는 축 | AIG-NCII에서 실제로 중요한 축 |
|---|---|
| 합성인지 실제인지 | 동의가 있었는지 |
| 진위성·출처·신뢰 | 존엄성·자율성·성적 프라이버시 |
| 시청자가 속았는지 | 당사자의 이미지가 동의 없이 사용됐는지 |
| 공개 라벨링 | 삭제·억제·피해자 보호 |

논문은 이를 **“authentic ≠ safe”**로 요약합니다. 이미지가 AI로 생성되었다는 사실을 표시해도, 당사자의 동의 없이 성적으로 묘사된 피해는 사라지지 않습니다.

---

### 7. 결론적인 결과 해석

이 논문의 결과는 “어떤 탐지 모델이 가장 우수한가”가 아니라 다음과 같습니다.

1. 기존 연구는 딥페이크를 주로 허위정보, 사기, 정치적 조작 문제로 다뤘습니다.
2. 39편 중 34편은 AIG-NCII를 전혀 언급하지 않았습니다.
3. 5편은 언급만 했으며, AIG-NCII에 맞춘 기술적 위협 모델은 없었습니다.
4. 따라서 AIG-NCII에 특화된 기술적 방어 연구는 분석 대상에서 0편이었습니다.
5. 기존 탐지·출처·워터마킹 기술은 합성 여부를 판단할 수 있지만, 동의 여부나 피해자의 존엄성은 판단하지 못합니다.
6. 그러므로 향후 평가는 단순한 탐지 정확도가 아니라 **실제 피해 감소 여부**를 중심으로 설계되어야 한다고 주장합니다.

논문이 권고하는 방향은 공개 라벨링보다는 백엔드 탐지 후 콘텐츠 억제·삭제·검토로 연결하는 방식, 신원 재현을 어렵게 하는 방어 기술, 사전 예방형 이미지 보호, 피해자·성폭력 방지 전문가와의 공동 설계 등입니다.

---




### 1. Nature of the study
This is not a conventional model-benchmarking paper. It is a **position paper and landscape analysis** examining how much existing AI/ML deepfake-defense research addresses AI-generated non-consensual intimate imagery (AIG-NCII).

Therefore, it does not provide a standard comparison of competing models, datasets, or accuracy metrics.

---

### 2. Literature selection

The authors analyzed technical defense papers published between 2020 and 2025.

| Stage | Number of papers |
|---|---:|
| Initial Google Scholar search | 965 |
| After venue and citation filtering | 379 |
| Top-cited papers selected | 100 |
| Final set after excluding unrelated or retracted papers | **39** |

The search covered terms related to detection, forensics, recognition, watermarking, deepfakes, synthetic images, fake images, and diffusion models. Papers from venues such as CVPR, ICCV, ECCV, NeurIPS, ICML, and ICLR were prioritized.

---

### 3. Main result: engagement with AIG-NCII

The 39 papers were divided into three categories.

| Category | Number | Percentage | Description |
|---|---:|---:|---|
| No mention | 34 | about 87.2% | Focused only on misinformation, fraud, political manipulation, or technical artifacts |
| Mention only | 5 | about 12.8% | Mentioned AIG-NCII briefly, but did not incorporate it into the technical method |
| Technical implementation | 0 | 0% | Used an AIG-NCII-specific threat model or intervention |

The central finding is therefore:

> **None of the 39 analyzed papers technically addressed AIG-NCII as a specific harm category.**

The appendix lists the 39 papers and marks whether relevant terms were mentioned. However, a passing mention does not mean that a paper developed an AIG-NCII defense.

---

### 4. Competing models and comparison targets

The paper does not conduct direct model-versus-model comparisons such as:

- CNN versus Transformer
- GAN detection versus diffusion detection
- Model A versus Model B
- Accuracy, F1-score, or AUROC rankings

Instead, it compares three broad intervention paradigms:

1. **Detection**
   - Classifies media as authentic or synthetic.
   - Uses artifacts such as frequency traces, diffusion reconstruction errors, or semantic features.

2. **Provenance**
   - Records the creation and editing history of media.
   - Uses cryptographic signatures and metadata, such as C2PA.

3. **Watermarking**
   - Embeds an invisible signal into AI-generated media.
   - Examples include latent watermarking, sampling-based watermarking, and systems such as SynthID.

The authors argue that all three paradigms mainly answer the question:

> “Is this media authentic or synthetic?”

They do not directly answer:

> “Was the person depicted in the image consenting to this use?”

---

### 5. Test data and evaluation metrics

#### Test data
The paper does not introduce a new dataset, conduct a shared benchmark, or re-evaluate the 39 papers on a common test set. It also does not generate or collect AIG-NCII content for model testing.

Thus, it does not provide:

- A common benchmark dataset
- Train/validation/test splits
- An AIG-NCII-specific test set
- A standardized victim-protection evaluation set

#### Metrics
Because this is a literature landscape analysis rather than a model evaluation, it does not report:

- Accuracy
- Precision or recall
- F1-score
- AUROC
- False-positive or false-negative rates

Its main analytical measures are:

- Whether AIG-NCII was mentioned
- Whether it was merely mentioned or technically incorporated
- Whether the work used an AIG-NCII-specific threat model

---

### 6. The paper’s central comparison

The paper contrasts the dominant evaluation axis in deepfake research with the axis that matters for AIG-NCII.

| Dominant deepfake research focus | AIG-NCII safety requirement |
|---|---|
| Synthetic versus authentic | Consent versus non-consent |
| Truth, provenance, and trust | Dignity, autonomy, and sexual privacy |
| Whether viewers were deceived | Whether the subject’s likeness was used without consent |
| Public labeling | Suppression, removal, and victim protection |

The paper summarizes this distinction as:

> **Authentic does not equal safe.**

Labeling an image as AI-generated does not remove the harm when a real person has been sexualized without consent.

---

### 7. Overall interpretation

The paper’s key result is not that one detection model outperforms another. Rather, it shows that:

1. Existing deepfake research is mainly motivated by misinformation, fraud, and political deception.
2. In the analyzed set, 34 of 39 papers did not mention AIG-NCII at all.
3. Five papers only mentioned it in passing.
4. No paper used an AIG-NCII-specific threat model or technical intervention.
5. Detection, provenance, and watermarking can help establish whether content is synthetic, but they do not determine consent or protect the subject’s dignity.
6. Future evaluation should therefore focus on **actual harm reduction**, not only detection accuracy.

The authors recommend using detection as a backend signal for suppression, triage, or removal rather than merely applying public labels; developing proactive identity-protection methods; restricting high-risk model releases; and designing interventions with victim advocates and experts in online sexual violence prevention.


<br/>
# 예제


이 논문이 지적하는 핵심은 **기존 딥페이크 연구가 “이미지가 진짜인가, AI가 만든 것인가”를 판별하는 데 집중하지만, AIG-NCII의 핵심 문제는 “당사자가 동의했는가”**라는 점이다. 따라서 기존 연구의 입력·출력·테스크는 대체로 다음과 같다.

### 1. 기존 딥페이크 연구의 전형적인 예시

| 연구 유형 | 트레이닝 데이터의 입력과 정답 | 테스트 데이터 | 모델 출력 | 구체적 테스크 |
|---|---|---|---|---|
| **합성 이미지 탐지** | 입력: 실제 이미지와 AI 생성 이미지. 정답: `real` 또는 `synthetic` | 학습에 사용하지 않은 실제·합성 이미지 | 합성 확률 또는 이진 분류 결과 | 이미지가 AI 생성인지 판별 |
| **생성 방식·모델 탐지** | 입력: 여러 생성 모델이 만든 이미지와 실제 이미지 | 새로운 생성 모델 또는 새로운 데이터셋의 이미지 | 생성 여부, 또는 사용된 생성 모델의 종류 | 생성 모델이 바뀌어도 탐지 가능한지 평가 |
| **위조 영역 탐지** | 입력: 얼굴 합성·인페인팅 등으로 일부가 변형된 이미지. 정답: 변형된 픽셀 또는 영역의 위치 | 새로운 조작 이미지 | 조작 영역의 마스크·좌표 | 이미지의 어느 부분이 조작되었는지 탐지 |
| **출처·프로비넌스 확인** | 입력: 원본, 편집본, 생성·수정 이력과 연결된 메타데이터 | 메타데이터가 있는 새 이미지 | 생성자, 수정 이력, 시간 등의 검증 결과 | 이미지가 어디서 왔고 어떻게 변경되었는지 확인 |
| **워터마크 검출** | 입력: 워터마크가 삽입된 이미지와 삽입되지 않은 이미지 | 압축·크롭·필터가 적용된 이미지 | 워터마크 존재 여부 또는 복호화된 식별 정보 | AI 생성 이미지의 워터마크 확인 |

예를 들어 합성 이미지 탐지 모델은 다음처럼 학습될 수 있다.

- **트레이닝 입력:** 실제 인물의 일반 사진, AI로 생성된 일반 인물 사진  
- **트레이닝 출력/정답:** `실제 이미지 = 0`, `AI 생성 이미지 = 1`
- **테스트 입력:** 학습에 사용하지 않은 새로운 인물 사진과 새로운 생성 모델의 이미지
- **테스트 출력:** “이 이미지는 합성일 확률 97%”와 같은 결과
- **테스크:** 진짜와 합성 이미지를 얼마나 정확히 구분하는지 측정

하지만 이 모델은 이미지가 **동의 없이 특정 인물을 성적으로 묘사했는지**는 반드시 판단하지 못한다.

---

### 2. AIG-NCII에서 필요한 테스크와 기존 테스크의 차이

AIG-NCII에서는 다음 네 가지 경우를 구분해야 한다.

| 이미지의 인공성 | 당사자 동의 | 안전성 |
|---|---|---|
| 합성 | 동의 있음 | 상대적으로 안전할 수 있음 |
| 합성 | 동의 없음 | **AIG-NCII, 유해** |
| 실제 | 동의 있음 | 합법적·동의된 콘텐츠일 수 있음 |
| 실제 | 동의 없음 | 전통적 NCII, 유해 |

즉, **“합성인가?”와 “동의가 있는가?”는 서로 다른 축**이다. 기존 탐지 모델은 주로 첫 번째 축만 다룬다.

AIG-NCII에 맞춘 예시적 테스크는 다음과 같다.

#### A. 비동의 성적 이미지 위험 선별

- **트레이닝 입력:** 사람의 성적·친밀한 이미지가 아니라, 안전한 비식별화·합성 데이터 또는 전문가가 승인한 특징 정보
- **정답:** `AIG-NCII 의심`, `동의된 합성 콘텐츠`, `일반 이미지` 등
- **테스트 입력:** 플랫폼에 업로드된 새로운 이미지
- **출력:** 공개적으로 표시할 “가짜” 라벨이 아니라, 내부적인 `검토 필요` 플래그
- **테스크:** 콘텐츠를 공개적으로 인증하는 것이 아니라, 노출을 줄이고 사람의 검토·삭제 절차로 연결

논문은 AIG-NCII가 탐지되더라도 공개 라벨을 붙이기보다 **삭제, 비공개 처리, 인간 검토를 위한 백엔드 신호**로 사용하는 것이 바람직하다고 주장한다.

#### B. 특정 인물의 동일성 재현 방지

- **트레이닝 입력:** 동의를 받은 일반 인물 사진과 해당 인물의 얼굴·신체 특징
- **학습 목표:** 모델이 특정 인물의 고유한 특징을 그대로 보존하거나 재현하지 못하도록 함
- **테스트 입력:** 몇 장의 공개 사진과 특정 인물을 재현하려는 프롬프트
- **출력:** 해당 인물의 식별 가능한 특징이 유지되지 않은 이미지, 또는 생성 거부
- **테스크:** 탐지 정확도보다 **비동의적 신원 보존(identity preservation)을 최소화**하는지 평가

이는 “생성된 결과가 가짜인지 판별”하는 사후 대응이 아니라, 애초에 특정 개인을 재현하기 어렵게 만드는 예방적 접근이다.

#### C. 누디파잉·인페인팅 방지

- **트레이닝 입력:** 보호 기능이 적용된 일반 사진과 그 사진을 변형하려는 인페인팅 입력
- **정답:** 보호된 사진에서 특정 신체·의상 정보가 보존되거나 안전하게 처리되는 결과
- **테스트 입력:** 크롭, 압축, 필터 등을 거친 보호 이미지와 새로운 생성 모델
- **출력:** 변형 실패, 신원 특징 훼손, 또는 안전한 원본 유지
- **테스크:** 사진이 AI에 의해 성적으로 변형되는 것을 어렵게 만드는 것

논문은 이러한 적대적 이미지 보호(adversarial immunization)가 완벽하지 않더라도 가해자의 비용과 기술적 장벽을 높이는 “마찰(friction)”을 제공할 수 있다고 설명한다.

#### D. 피해 감소 중심 평가

기존 평가는 보통 정확도, 정밀도, 재현율, AUROC 등으로 탐지 성능을 측정한다. 그러나 AIG-NCII에서는 다음과 같은 평가가 더 중요하다.

- 유해 콘텐츠의 실제 노출·확산이 감소했는가?
- 피해 이미지가 탐지 후 신속히 비공개·삭제되는가?
- 오탐으로 인해 동의된 표현이나 피해자의 권리가 침해되지 않는가?
- 탐지 결과가 가해자에게 피해자의 실제 노출 여부를 확인해 주지는 않는가?
- 민감한 원본을 직접 생성하거나 공개하지 않고도 평가할 수 있는가?

---

### 3. 논문의 핵심적인 설계 원칙

1. **진위성과 안전성을 분리한다.**  
   “AI가 만든 이미지”라는 사실만으로 안전하지도, 유해하지도 않다.

2. **공개 라벨보다 비공개 대응을 우선한다.**  
   AIG-NCII 의심 결과는 공개적인 “가짜” 표시가 아니라 삭제·차단·검토를 위한 내부 신호가 되어야 한다.

3. **시스템의 목표를 탐지 정확도에서 피해 감소로 바꾼다.**  
   높은 탐지 성능이 실제 피해 감소를 보장하지 않는다.

4. **특정 인물의 재현 자체를 어렵게 한다.**  
   모델이 소수의 사진만으로 개인의 외모를 복제하지 못하도록 하는 예방 기술이 필요하다.

5. **도메인 전문가와 공동 설계한다.**  
   온라인 성폭력 예방 전문가, 피해자·생존자 옹호 단체, 정책·법률 전문가가 위협 모델과 평가 기준에 참여해야 한다.

6. **민감한 연구에는 안전장치를 둔다.**  
   연구자에게 발생할 수 있는 2차 트라우마를 줄이고, 실제 성적 피해 이미지를 불필요하게 수집·생성·공개하지 않아야 한다.

---




The paper’s central argument is that **most deepfake research asks whether an image is real or AI-generated, whereas the central issue in AIG-NCII is whether the depicted person consented**. The typical inputs, outputs, and tasks in existing research are therefore as follows.

### 1. Typical examples of existing deepfake research

| Research type | Training input and labels | Test data | Model output | Main task |
|---|---|---|---|---|
| **Synthetic-image detection** | Real images and AI-generated images; labels: `real` or `synthetic` | Unseen real and synthetic images | Synthetic probability or binary label | Determine whether an image was AI-generated |
| **Generator/model attribution** | Images from multiple generators and real images | Images from unseen generators or datasets | Synthetic status or generator identity | Generalize across changing generation models |
| **Manipulation localization** | Images with face swaps or inpainting; labels identify altered pixels or regions | New manipulated images | A manipulation mask or region coordinates | Locate which parts of an image were altered |
| **Provenance verification** | Original and edited media linked to metadata and modification histories | New media with provenance records | Verified creator, timestamp, and edit history | Establish where the media came from and how it changed |
| **Watermark detection** | Images with and without embedded watermarks | Images transformed by compression, cropping, or filtering | Watermark presence or decoded identifier | Verify whether content was generated by a particular AI system |

For example:

- **Training input:** ordinary photographs and AI-generated ordinary-person images  
- **Training labels:** `real = 0`, `synthetic = 1`
- **Test input:** unseen people and images from a new generator
- **Test output:** “synthetic probability: 97%”
- **Task:** Measure how accurately the model distinguishes real from generated images

However, this model does not necessarily determine whether an image depicts a **specific person in a sexualized context without consent**.

---

### 2. How AIG-NCII requires different tasks

AIG-NCII requires separating two independent dimensions:

| Artificiality | Consent | Safety implication |
|---|---|---|
| Synthetic | Consensual | May be relatively safe |
| Synthetic | Non-consensual | **AIG-NCII and harmful** |
| Authentic | Consensual | May be legitimate consensual content |
| Authentic | Non-consensual | Traditional NCII and harmful |

Thus, **“Is it synthetic?” and “Was it consensual?” are different questions**. Existing detectors mainly address the first question.

#### A. Risk triage for non-consensual sexual imagery

- **Training input:** Safe, de-identified, or synthetically constructed examples, or expert-approved feature representations rather than unnecessary collections of real intimate images
- **Labels:** `suspected AIG-NCII`, `consensual synthetic content`, `ordinary image`, etc.
- **Test input:** Newly uploaded platform content
- **Output:** A private backend flag such as `requires review`, rather than a public “fake” label
- **Task:** Reduce exposure and route content to human review, suppression, or removal

The paper recommends that suspected AIG-NCII should generally trigger **backend moderation actions**, not public labeling that leaves the material visible.

#### B. Preventing non-consensual identity preservation

- **Training input:** Consent-based ordinary photographs and facial or bodily features
- **Learning objective:** Prevent a model from preserving and reproducing a specific person’s distinctive identity
- **Test input:** A small number of reference photographs and prompts attempting to reproduce that person
- **Output:** A generation that does not preserve identifiable features, or a refusal to generate
- **Task:** Minimize unauthorized identity preservation rather than merely classify the final image

This is a preventive approach: instead of detecting abuse after generation, it makes it harder to reproduce a particular individual in the first place.

#### C. Preventing nudification and malicious inpainting

- **Training input:** Ordinary images with protective perturbations and attempted inpainting transformations
- **Target output:** The protected image remains unchanged or the identity-specific transformation fails
- **Test input:** Protected images after cropping, compression, filtering, or other transformations, evaluated against new generators
- **Output:** Failed manipulation, degraded identity reconstruction, or preservation of the safe original
- **Task:** Increase the difficulty of sexually modifying a person’s image with generative AI

The paper describes this as adversarial image protection or immunization. It may not be perfect, but it can create **friction** by increasing the technical cost of abuse.

#### D. Harm-reduction-oriented evaluation

Traditional evaluations emphasize accuracy, precision, recall, or AUROC. For AIG-NCII, more relevant questions include:

- Did exposure and redistribution of harmful content decrease?
- Was suspected content quickly suppressed or removed?
- Did false positives unnecessarily restrict consensual expression?
- Could the detector itself help abusers verify that a victim’s authentic image was exposed?
- Can the system be evaluated without generating or publicly releasing harmful material?

---

### 3. Main design principles from the paper

1. **Separate authenticity from safety.**  
   Synthetic content is not automatically safe, and authentic content is not automatically harmless.

2. **Prefer private moderation signals to public labels.**  
   AIG-NCII detection should support suppression, removal, or triage rather than simply labeling content as fake.

3. **Measure harm reduction, not only detection accuracy.**  
   A highly accurate detector may still fail to reduce real-world harm.

4. **Prevent identity reproduction proactively.**  
   Models should be designed so that a person cannot be recreated from a small number of photographs without consent.

5. **Co-design with domain experts.**  
   Researchers should work with online sexual-violence experts, victim-survivor advocates, legal specialists, and policymakers.

6. **Use strong ethical guardrails.**  
   Research should avoid unnecessary collection, generation, or disclosure of intimate abuse material and should protect researchers from secondary traumatic stress.

<br/>
# 요약


1. 연구진은 2020~2025년 주요 학회와 고인용 논문 중 딥페이크 방어 논문 39편을 분석하고, 탐지·출처 추적·워터마킹 방법이 AIG-NCII를 어떻게 다루는지 조사했다.  
2. 그 결과 34편은 AIG-NCII를 전혀 언급하지 않았고 5편만 간단히 언급했으며, 이를 위협 모델에 반영해 기술적으로 대응한 논문은 0편으로, 연구가 주로 사기·허위정보 같은 시청자 중심의 진위성 피해에 치우쳐 있음을 보였다.  
3. 예를 들어 합성 이미지라는 라벨은 동의 없이 성적 이미지가 유포되는 피해를 줄이지 못하고 오히려 가해자의 검색·피해자 식별을 도울 수 있으므로, 연구진은 공개 라벨링보다 삭제·차단 중심의 대응과 존엄성 피해를 반영한 위협 모델 및 성폭력 전문가와의 협력을 제안한다.  



1. The authors analyzed 39 highly cited deepfake-defense papers from major venues published between 2020 and 2025, examining how detection, provenance, and watermarking methods address AIG-NCII.  
2. They found that 34 papers did not mention AIG-NCII, five only mentioned it briefly, and none developed a technical intervention with an AIG-NCII-specific threat model, showing that research mainly prioritizes viewer-centered harms such as fraud and misinformation.  
3. For example, labeling an image as synthetic does not reduce the harm of distributing sexual imagery without consent and may help abusers identify or search for victims, so the authors recommend backend suppression or removal, dignity-aware threat models, and collaboration with sexual-violence experts.

<br/>
# 기타



논문에서 본문 외에 제시된 **그림·표·어펜딕스**의 핵심 결과와 인사이트는 다음과 같습니다.

### Figure 1. AIG-NCII의 실제 비중과 연구 비중 비교

**결과**
- 생성형 AI 콘텐츠 중 AIG-NCII가 차지하는 비중은 매우 큰 것으로 제시됩니다.
- 반면, 기술적 방어 연구 39편 중:
  - **34편(약 87%)**: AIG-NCII를 전혀 언급하지 않음
  - **5편(약 13%)**: 관련 용어만 간단히 언급
  - **0편**: AIG-NCII에 특화된 위협 모델이나 기술적 개입을 제안

**인사이트**
- 이 그림은 논문의 핵심 주장인 **“현실의 피해 규모와 AI/ML 연구 의제 사이의 불일치”**를 시각적으로 보여줍니다.
- AIG-NCII가 생성형 AI 악용의 주요 형태일 가능성이 큰데도, 연구는 주로 허위정보·사기·정치적 딥페이크 탐지에 집중되어 있습니다.
- 즉, 연구가 많이 다루는 문제와 실제로 많이 발생하는 피해가 서로 다릅니다.

---

### Figure 2. 딥페이크 피해의 대상: 시청자, 피사체, 또는 양쪽 모두

**결과**
- 딥페이크 피해를 다음 세 유형으로 구분하는 개념적 그림입니다.
  1. **시청자 중심 피해**: 시청자가 속거나 잘못된 정보를 믿게 되는 경우  
     - 예: 투자 사기, 정치적 허위 영상
  2. **피사체 중심 피해**: 당사자의 동의 없이 얼굴이나 신체가 이용되어 존엄·자율성이 침해되는 경우  
     - 예: AIG-NCII
  3. **양쪽 모두에 대한 피해**: 시청자를 속이는 동시에 피사체의 평판이나 권리를 침해하는 경우  
     - 예: 정치인의 얼굴을 사용한 허위 영상

**인사이트**
- 딥페이크 피해는 단순히 “누가 진짜인지 판별하지 못하는 문제”가 아닙니다.
- 특히 AIG-NCII에서는 시청자가 속지 않더라도, 피사체는 이미 동의 없는 성적 이미지화로 피해를 입습니다.
- 따라서 탐지 시스템은 시청자 보호뿐 아니라 **이미지 속 인물의 존엄과 동의 여부**도 고려해야 합니다.

---

### Table 1. 인공성(authenticity)과 동의(consent)는 서로 다른 축

**결과**
표는 이미지의 안전성을 다음 두 축으로 나눕니다.

|  | 동의 있음 / 안전 | 동의 없음 / 유해 |
|---|---|---|
| **합성 이미지** | AI를 이용한 예술적 누드 등 | **AIG-NCII** |
| **실제 이미지** | 합의된 포르노그래피 | **전통적 NCII** |

**인사이트**
- 기존 탐지 기술은 주로 이미지가 **합성인지 실제인지**만 구분합니다.
- 그러나 실제 안전성을 결정하는 핵심 기준은 **동의 여부**입니다.
- 따라서 “AI로 생성되었다”는 사실만으로 안전성을 판단할 수 없습니다.
  - 합성 이미지라도 동의가 없으면 유해합니다.
  - 실제 이미지라도 동의가 있으면 반드시 유해한 것은 아닙니다.
- 논문의 핵심 표현인 **“Authentic ≠ Safe”**를 가장 직접적으로 보여주는 표입니다.
- 즉, 인공성의 축과 피해·안전의 축은 서로 직교하며, 탐지 정확도만 높이는 것으로는 AIG-NCII 문제를 해결할 수 없습니다.

---

### Table 2. 검토한 39편의 논문 목록과 AIG-NCII 언급 여부

**결과**
- 2020~2025년의 주요 AI/ML·컴퓨터비전 관련 논문 중 인용 수가 높은 39편을 검토했습니다.
- 표의 체크 표시가 있는 논문은 다음 5편입니다.
  - Hsu et al. (2020)
  - Raza et al. (2022)
  - Patel et al. (2023)
  - Guarnera et al. (2024)
  - Liao et al. (2021)
- 그러나 이 논문들도 AIG-NCII를 본격적으로 기술적 문제로 다루지는 않았습니다.

**인사이트**
- 39편 중 일부는 AIG-NCII 또는 관련 용어를 언급하지만, 그 언급은 대체로 서론이나 영향 분석 수준에 머뭅니다.
- AIG-NCII에 특화된 데이터셋, 위협 모델, 평가 지표, 방어 방법을 제시한 논문은 **0편**입니다.
- 따라서 논문 저자들은 단순히 “관련 단어가 언급되었는가”보다, 실제로 **AIG-NCII의 피해를 줄이도록 설계되었는가**를 기준으로 연구 참여를 평가해야 한다고 주장합니다.

---

### 어펜딕스의 역할

이 논문의 어펜딕스에는 주로 **Table 2의 39편 전체 목록**이 제시되어 있습니다.

**인사이트**
- 어펜딕스는 본문의 주장을 재현할 수 있도록 연구 대상과 선정 결과를 공개하는 역할을 합니다.
- 이를 통해 저자들은 “딥페이크 연구가 AIG-NCII를 무시한다”는 주장이 단순한 인상비평이 아니라, 정해진 검색어·학회·인용 기준에 따라 수행한 **문헌 검토 결과**임을 보여줍니다.
- 다만 이 분석은 39편의 고인용 기술 논문에 한정되어 있으므로, 모든 딥페이크 연구 전체를 대표한다고 보기는 어렵다는 한계도 있습니다.

---

### 전체 시각자료의 종합 인사이트

이 논문의 그림과 표는 다음의 논리 흐름을 구성합니다.

1. **Figure 1**: AIG-NCII는 현실에서 큰 비중을 차지하지만 연구에서는 거의 다뤄지지 않음.
2. **Figure 2**: 딥페이크 피해는 시청자뿐 아니라 피사체에게도 발생함.
3. **Table 1**: 합성 여부와 동의 여부는 별개의 기준이므로, authenticity만으로 safety를 판단할 수 없음.
4. **Table 2**: 실제 주요 연구 39편에서도 AIG-NCII를 구체적으로 다룬 사례는 없음.

따라서 저자들은 딥페이크 연구가 단순한 **진위 판별 기술**에서 벗어나, 콘텐츠의 **동의 여부·피사체의 존엄·피해 감소**를 직접 고려하는 방향으로 재정렬되어야 한다고 주장합니다.

---




### Figure 1. Comparing the prevalence of AIG-NCII with research attention

**Result**
- AIG-NCII is presented as a major form of generative-AI abuse.
- Among the 39 highly cited technical papers examined:
  - **34 papers, about 87%**, did not mention AIG-NCII at all.
  - **5 papers, about 13%**, mentioned it only briefly.
  - **0 papers** developed a technical intervention or threat model specifically focused on AIG-NCII.

**Insight**
- The figure visualizes the paper’s central claim: there is a major **mismatch between the prevalence of AIG-NCII in practice and the priorities of AI/ML research**.
- Deepfake research largely focuses on misinformation, fraud, and political deception, even though sexualized abuse may represent a dominant use of generative imagery.

---

### Figure 2. Deepfake harms to viewers, subjects, or both

**Result**
The figure conceptually distinguishes three types of harm:

1. **Viewer-centric harm**: viewers are deceived or exposed to false information, such as scams or political misinformation.
2. **Subject-centric harm**: a person’s likeness is used without consent, violating their dignity, autonomy, or sexual privacy, as in AIG-NCII.
3. **Overlapping harm**: both viewers and the depicted subject are harmed, such as when a political deepfake deceives the public while damaging a politician’s reputation.

**Insight**
- Deepfake harm is not only a problem of whether viewers can identify what is real.
- In AIG-NCII cases, the subject can be seriously harmed even when everyone knows the image is synthetic.
- Detection systems therefore need to consider **consent and subject dignity**, not only viewer deception.

---

### Table 1. Authenticity and consent are separate dimensions

**Result**

|  | Consensual / Safe | Non-consensual / Harmful |
|---|---|---|
| **Synthetic** | Artistic nudity generated with AI | **AIG-NCII** |
| **Authentic** | Consensual pornography | **Traditional NCII** |

**Insight**
- Existing detection tools primarily distinguish between **synthetic and authentic media**.
- However, safety depends mainly on **whether the content was created or shared with consent**.
- This means that synthetic content is not automatically safe, and authentic content is not automatically harmful.
- The table illustrates the paper’s key claim: **“Authentic ≠ Safe.”**
- Authenticity and consent are orthogonal dimensions, so improving authenticity detection alone cannot solve AIG-NCII.

---

### Table 2. The 39 surveyed papers and their engagement with AIG-NCII

**Result**
The authors reviewed 39 highly cited papers from 2020 to 2025. Five papers were marked as mentioning AIG-NCII-related issues:

- Hsu et al. (2020)
- Liao et al. (2021)
- Raza et al. (2022)
- Patel et al. (2023)
- Guarnera et al. (2024)

However, none of these papers meaningfully designed their methods around AIG-NCII.

**Insight**
- A brief mention in an introduction or impact statement does not equal substantive engagement.
- No paper in the sample proposed an AIG-NCII-specific dataset, threat model, evaluation metric, or defense.
- The authors therefore distinguish between merely **mentioning AIG-NCII** and actually developing interventions intended to **reduce its harms**.

---

### Role of the Appendix

The appendix primarily provides the full list of the 39 surveyed papers and their AIG-NCII-related mentions.

**Insight**
- It makes the literature analysis more transparent and allows readers to inspect the evidence behind the paper’s claim.
- The appendix supports the argument that the field’s neglect of AIG-NCII is based on a systematic review rather than only on anecdotal impressions.
- At the same time, the analysis is limited to a selected set of highly cited technical papers, so it should not be interpreted as a complete review of every deepfake publication.

---

### Overall insight from the visual materials

The figures and tables build a consistent argument:

1. **Figure 1** shows that AIG-NCII is prevalent in practice but largely absent from research.
2. **Figure 2** shows that deepfakes can harm subjects as well as viewers.
3. **Table 1** shows that synthetic/authentic status is different from consensual/non-consensual status.
4. **Table 2** shows that major technical papers rarely engage with AIG-NCII in a substantive way.

Together, these materials support the authors’ call to move beyond simple **authenticity detection** and toward interventions that directly address **consent, subject dignity, identity protection, and actual harm reduction**.

<br/>
# refer format:
### BibTeX

```bibtex
@inproceedings{qiwei2026misaligned,
  author    = {Qiwei, Li and Wells, Lucas Santo and Schoenebeck, Sarita and Gilbert, Eric},
  title     = {{Position: AI/ML Deepfake Research is Misaligned with AI-Generated Non-Consensual Intimate Imagery (AIG-NCII)}},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  year      = {2026},
  address   = {Seoul, South Korea},
  publisher = {PMLR}
}
```

### 시카고 스타일 참고문헌

Qiwei, Li, Lucas Santo Wells, Sarita Schoenebeck, and Eric Gilbert. “Position: AI/ML Deepfake Research Is Misaligned with AI-Generated Non-Consensual Intimate Imagery (AIG-NCII).” In *Proceedings of the 43rd International Conference on Machine Learning*. Seoul, South Korea: PMLR, 2026, vol. 306.

### 시카고 스타일 각주

1. Li Qiwei, Lucas Santo Wells, Sarita Schoenebeck, and Eric Gilbert, “Position: AI/ML Deepfake Research Is Misaligned with AI-Generated Non-Consensual Intimate Imagery (AIG-NCII),” in *Proceedings of the 43rd International Conference on Machine Learning*, vol. 306 (Seoul, South Korea: PMLR, 2026).




