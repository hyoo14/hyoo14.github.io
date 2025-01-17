---
layout: post
title:  "[2024]Stable Diffusion Exposed: Gender Bias from Prompt to Image"  
date:   2025-01-17 17:18:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

스테이블디퓨전 편향 평가 위한 데이터셋 제안(프롬프트로 구성된)  



짧은 요약(Abstract) :    




이 논문은 이미지 생성 모델에서 나타나는 성별 편향에 대해 조사하며, 특히 Stable Diffusion 모델이 프롬프트의 성별 지시자(예: 남성, 여성, 중립적 표현)에 따라 이미지 생성 과정에서 어떻게 다르게 반응하는지 분석합니다. 연구 결과, 중립적인 프롬프트로 생성된 이미지가 여성 지시자보다 남성 지시자에 더 가깝게 생성되는 경향이 발견되었습니다. 또한, 프롬프트에서 명시적으로 언급되지 않은 객체들도 성별에 따라 차별적으로 생성되었습니다. 논문은 이러한 편향을 줄이기 위한 모델 개발자와 사용자에게 각각의 권고 사항을 제공합니다.

---



Several studies have raised awareness about social biases in image generative models, demonstrating their predisposition towards stereotypes and imbalances. This paper contributes to this growing body of research by introducing an evaluation protocol that analyzes the impact of gender indicators at every step of the generation process on Stable Diffusion images. Leveraging insights from prior work, we explore how gender indicators not only affect gender presentation but also the representation of objects and layouts within the generated images. Our findings include the existence of differences in the depiction of objects, such as instruments tailored for specific genders, and shifts in overall layouts. We also reveal that neutral prompts tend to produce images more aligned with masculine prompts than their feminine counterparts. We further explore where bias originates through representational disparities and how it manifests in the images via prompt-image dependencies, and provide recommendations for developers and users to mitigate potential bias in image generation.



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





**1. 메서드 (Method)**  
이 논문은 Stable Diffusion 모델에서 성별 편향을 평가하기 위한 새로운 자동화된 평가 프로토콜을 제안합니다. 이 메서드는 다음 두 가지 주요 요소를 분석합니다:
- **표현의 불균형**: 프롬프트 공간, 디노이징 공간, 이미지 공간에서 성별 지시자가 생성 과정에 미치는 영향을 측정합니다.
- **프롬프트-이미지 종속성**: 프롬프트와 생성된 이미지 간의 관계를 기반으로 객체를 다섯 가지 그룹(명시적/암시적 가이드 여부 등)으로 분류합니다.

**2. 제안하는 데이터셋 (Dataset)**  
이 연구는 다음 네 가지 데이터셋과 ChatGPT로 생성된 직업 프롬프트 데이터를 사용합니다:
- GCC (Google Conceptual Captions)
- COCO (Common Objects in Context)
- TextCaps (Textual Descriptions for Captions)
- Flickr30k (Image Captioning Dataset)  
각 데이터셋의 프롬프트는 중립적, 여성적, 남성적으로 변환되어 사용됩니다.

**3. 비교하는 모델 (Comparative Models)**  
세 가지 Stable Diffusion 모델을 비교 분석했습니다:
- Stable Diffusion v1.4
- Stable Diffusion v2.0
- Stable Diffusion v2.1

**4. 트레이닝 데이터 및 아키텍처 (Training Data and Architecture)**  
Stable Diffusion 모델 자체는 CLIP 텍스트 인코더와 U-Net 구조를 활용하여 초기 노이즈에서 이미지를 생성합니다. 이 연구는 모델을 재트레이닝하지 않고, 기존 Stable Diffusion 모델의 성별 편향을 분석하는 데 초점을 맞췄습니다.

---



**1. Method**  
The paper introduces a novel automated evaluation protocol to assess gender bias in Stable Diffusion models. The method includes two main components:
- **Representational Disparities**: Measuring how gender indicators affect the generation process across the prompt space, denoising space, and image space.
- **Prompt-Image Dependencies**: Classifying objects into five groups based on their relationship with the prompt and the generated image, such as explicitly/implicitly guided or independent objects.

**2. Proposed Dataset**  
The study uses the following datasets along with profession-specific prompts generated via ChatGPT:
- GCC (Google Conceptual Captions)
- COCO (Common Objects in Context)
- TextCaps (Textual Descriptions for Captions)
- Flickr30k (Image Captioning Dataset)  
Prompts from these datasets were transformed into neutral, feminine, and masculine variations.

**3. Comparative Models**  
Three Stable Diffusion models were compared in the study:
- Stable Diffusion v1.4
- Stable Diffusion v2.0
- Stable Diffusion v2.1

**4. Training Data and Architecture**  
The Stable Diffusion models analyzed in this study use the CLIP text encoder and U-Net architecture to generate images from initial noise. The paper focuses on analyzing the pre-trained Stable Diffusion models for gender bias without re-training them.


   
 
<br/>
# Results  





**1. 주요 결과 요약**  
논문은 Stable Diffusion 모델이 성별 지시자(중립, 남성, 여성)에 따라 이미지 생성에서 성별 편향을 보인다고 결론지었습니다. 중립 프롬프트로 생성된 이미지는 여성 프롬프트보다 남성 프롬프트와 더 유사했습니다.

---

**2. 결과 세부사항**  
#### **(1) 데이터셋 및 메트릭**
논문에서는 다음 데이터셋과 메트릭을 사용했습니다:
- 데이터셋: GCC, COCO, TextCaps, Flickr30k
- 메트릭:
  - **SSIM (Structural Similarity Index Measure)**: 이미지 구조적 유사성 평가
  - **Diff. Pix (Different Pixels)**: 다른 픽셀 수 평가
  - **ResNet, CLIP, DINO**: 사전 학습된 모델에서 얻은 잠재 벡터 간의 코사인 유사도
  - **Split-product**: 이미지 패치 간 최대 코사인 유사도

#### **(2) 모델별 결과 (Stable Diffusion v1.4, v2.0, v2.1)**
각 Stable Diffusion 버전에서 공통적으로 관찰된 경향:
- **중립 프롬프트**로 생성된 이미지가 **남성 프롬프트**로 생성된 이미지와 더 유사.
- **여성 프롬프트**는 객체나 배경에서 더 적은 유사성을 보임.

##### GCC 데이터셋 (예시):
| 모델 | 메트릭 | 중립 vs 여성 | 중립 vs 남성 |
|------|--------|--------------|--------------|
| SD v1.4 | SSIM | 0.770 | 0.798 |
| SD v2.0 | SSIM | 0.767 | 0.790 |
| SD v2.1 | SSIM | 0.755 | 0.782 |

- 모든 모델과 데이터셋에서 **중립-남성** 쌍이 **중립-여성** 쌍보다 높은 유사도를 기록.

---

**3. 주요 관찰**
1. **명시적 객체**: 프롬프트에서 명시된 객체(예: 기타, 우산)는 성별에 관계없이 유사한 빈도로 생성됨.
2. **암시적 객체**: 프롬프트에 명시되지 않은 객체는 성별에 따라 큰 차이를 보임.
   - 남성 프롬프트: 스포츠 관련 객체(예: 축구공, 선수)
   - 여성 프롬프트: 음식 및 가족 관련 객체(예: 샐러드, 아이)
3. **배경**: 성별에 따라 배경의 디테일도 다름(예: 공원은 여성 프롬프트에서 더 자주 등장).

---


**1. Summary of Key Results**  
The study found that Stable Diffusion models exhibit significant gender bias depending on the gender indicators (neutral, masculine, feminine) in the prompts. Images generated from neutral prompts were consistently more similar to those generated with masculine prompts than to feminine prompts.

---

**2. Detailed Results**
#### **(1) Datasets and Metrics**
- **Datasets**: GCC, COCO, TextCaps, Flickr30k
- **Metrics**:
  - **SSIM**: Measures structural similarity of images.
  - **Diff. Pix**: Evaluates the number of differing pixels.
  - **ResNet, CLIP, DINO**: Cosine similarity between latent vectors from pre-trained models.
  - **Split-product**: Maximum cosine similarity across image patches.

#### **(2) Model Performance (Stable Diffusion v1.4, v2.0, v2.1)**
Observed trends across all versions:
- Images from **neutral prompts** were more similar to those from **masculine prompts**.
- **Feminine prompts** produced less similar results in object or background representation.

##### GCC Dataset (Example):
| Model  | Metric | Neutral vs Feminine | Neutral vs Masculine |
|--------|--------|---------------------|---------------------|
| SD v1.4 | SSIM   | 0.770               | 0.798               |
| SD v2.0 | SSIM   | 0.767               | 0.790               |
| SD v2.1 | SSIM   | 0.755               | 0.782               |

- Across all datasets and models, **Neutral-Masculine** pairs showed higher similarity than **Neutral-Feminine** pairs.

---

**3. Key Observations**
1. **Explicit Objects**: Objects explicitly mentioned in the prompts (e.g., guitar, umbrella) were generated at similar rates regardless of gender.
2. **Implicit Objects**: Objects not explicitly mentioned in the prompts varied significantly by gender:
   - Masculine prompts: More sports-related objects (e.g., football, players).
   - Feminine prompts: More food and family-related objects (e.g., salad, child).
3. **Backgrounds**: Background details also differed by gender (e.g., parks appeared more frequently with feminine prompts).






<br/>
# 예제  



#### **예제: GCC 데이터셋**
**프롬프트**
- **중립적**: "A person on the shore of a lake."
- **남성적**: "A man on the shore of a lake."
- **여성적**: "A woman on the shore of a lake."

---

#### **1. 모델별 생성된 이미지 분석**
- **Stable Diffusion v1.4**
  - **중립적 프롬프트**: 호수와 인물이 포함된 기본 이미지.
  - **남성적 프롬프트**: 중립적 프롬프트와 유사하지만 인물이 더 강인한 모습으로 묘사됨.
  - **여성적 프롬프트**: 인물이 더 부드럽거나 날씬한 모습으로 묘사됨. 배경의 밝기와 채도가 약간 더 높음.

- **Stable Diffusion v2.0**
  - **중립적 프롬프트**: 호수 주변의 평범한 장면. 남성적 이미지와 유사.
  - **남성적 프롬프트**: 중립적 프롬프트보다 더 강렬한 색감과 인물의 크기가 큼.
  - **여성적 프롬프트**: 인물이 작아지고 배경에 꽃과 같은 추가 요소가 포함됨.

- **Stable Diffusion v2.1**
  - **중립적 프롬프트**: 인물의 자세와 배경이 남성적 프롬프트와 유사.
  - **남성적 프롬프트**: 강한 대조와 깊이감 있는 배경.
  - **여성적 프롬프트**: 배경에 나뭇잎, 꽃 등이 추가되고, 인물이 더 세밀하게 묘사됨.

---

#### **2. 메트릭 비교 (GCC 데이터셋)**
| 모델       | 메트릭          | 중립 vs 여성 | 중립 vs 남성 |
|------------|------------------|--------------|--------------|
| SD v1.4    | SSIM            | 0.770        | 0.798        |
|            | ResNet Cosine   | 0.516        | 0.543        |
| SD v2.0    | SSIM            | 0.767        | 0.790        |
|            | ResNet Cosine   | 0.543        | 0.571        |
| SD v2.1    | SSIM            | 0.755        | 0.782        |
|            | ResNet Cosine   | 0.522        | 0.552        |

---



#### **Example: GCC Dataset**
**Prompt**
- **Neutral**: "A person on the shore of a lake."
- **Masculine**: "A man on the shore of a lake."
- **Feminine**: "A woman on the shore of a lake."

---

#### **1. Analysis of Generated Images by Model**
- **Stable Diffusion v1.4**
  - **Neutral Prompt**: A generic scene with a person by the lake.
  - **Masculine Prompt**: Similar to the neutral prompt but with the person appearing more robust.
  - **Feminine Prompt**: The person appears slimmer or softer, and the background is slightly brighter and more saturated.

- **Stable Diffusion v2.0**
  - **Neutral Prompt**: A standard scene by the lake, resembling the masculine image.
  - **Masculine Prompt**: Brighter colors and a larger person depicted compared to the neutral prompt.
  - **Feminine Prompt**: Smaller person and additional elements like flowers in the background.

- **Stable Diffusion v2.1**
  - **Neutral Prompt**: Similar poses and backgrounds to the masculine prompt.
  - **Masculine Prompt**: Strong contrasts and depth in the background.
  - **Feminine Prompt**: Added leaves, flowers in the background, and more detailed depiction of the person.

---

#### **2. Metric Comparison (GCC Dataset)**
| Model      | Metric           | Neutral vs Feminine | Neutral vs Masculine |
|------------|------------------|---------------------|---------------------|
| SD v1.4    | SSIM             | 0.770               | 0.798               |
|            | ResNet Cosine    | 0.516               | 0.543               |
| SD v2.0    | SSIM             | 0.767               | 0.790               |
|            | ResNet Cosine    | 0.543               | 0.571               |
| SD v2.1    | SSIM             | 0.755               | 0.782               |
|            | ResNet Cosine    | 0.522               | 0.552               |



<br/>  
# 요약   



Stable Diffusion 모델의 성별 편향 문제를 해결하기 위해, 이 논문은 자동화된 평가 프로토콜을 제안합니다. 제안된 방법론은 프롬프트 공간, 디노이징 공간, 그리고 이미지 공간에서 성별 지시자가 이미지 생성 과정에 미치는 영향을 측정합니다. 연구 결과, 중립적인 프롬프트로 생성된 이미지가 남성 지시자와 더 유사하며, 여성 지시자와의 유사성은 상대적으로 낮게 나타났습니다. 예를 들어, GCC 데이터셋의 프롬프트 "A person on the shore of a lake"에서, 중립 프롬프트로 생성된 이미지는 남성 프롬프트와 유사하지만 여성 프롬프트는 배경이나 객체 표현에서 차이를 보였습니다. 이를 통해 연구진은 성별 편향을 완화하기 위해 프롬프트 설계와 텍스트 임베딩 디바이싱(debiasing)을 추천합니다.

---


To address gender bias in Stable Diffusion models, this paper proposes an automated evaluation protocol. The methodology measures the impact of gender indicators in the prompt space, denoising space, and image space during the image generation process. The study found that images generated from neutral prompts were more similar to those from masculine prompts, while feminine prompts showed lower similarity. For instance, in the GCC dataset, the prompt "A person on the shore of a lake" resulted in neutral images resembling masculine ones, but feminine prompts differed in object and background representation. Based on these findings, the researchers recommend prompt design adjustments and debiasing text embeddings to mitigate gender bias.

<br/>  
# 기타  




논문에서 제시된 다이어그램과 그래프는 Stable Diffusion 모델의 성별 편향을 시각적으로 설명하고 있습니다. 주요 피규어의 요약은 아래와 같습니다:

1. **Figure 1: 성별 지시자에 따른 이미지 생성 비교**  
   동일한 프롬프트에서 "중립", "남성", "여성" 지시자만 변경했을 때 생성된 이미지를 비교합니다. 예를 들어, "A person playing an instrument" 프롬프트는 중립적 이미지가 남성 이미지와 더 유사하며, 여성 이미지는 악기 종류나 배치가 달라지는 경향을 보였습니다.

2. **Figure 3: 프롬프트-이미지 종속성 그룹**  
   이미지 생성 중 객체가 프롬프트와 어떤 관계로 생성되었는지 5가지 그룹으로 분류합니다:
   - **Explicitly guided**: 프롬프트에 명시적으로 언급되고 교차 주의로 가이드된 객체 (예: "woman", "picnic").
   - **Implicitly guided**: 프롬프트에 명시되지 않았지만 연관된 객체 (예: "basket", "blanket").
   - **Explicitly independent**: 프롬프트에 명시되었으나 교차 주의로 가이드되지 않은 객체 (예: "park").
   - **Implicitly independent**: 프롬프트에 언급되지 않았고 배경에서 유추된 객체 (예: "grass").
   - **Hidden**: 이미지에 포함되지 않은 객체 (예: "daytime").

3. **Figure 4: 객체 생성 빈도 비교**  
   각 프롬프트-이미지 종속성 그룹에서 자주 생성된 객체를 나열하며, SD v2.0 모델에서 성별에 따른 차이를 나타냅니다. 예를 들어, **Implicitly guided** 그룹에서 남성 프롬프트는 "shirt", "microphone" 등의 객체를 더 자주 생성했고, 여성 프롬프트는 "dress", "scarf" 등이 더 많았습니다.

4. **Figure 5: 편향 점수 그래프**  
   텍스트 캡션과 GCC 데이터셋에서 특정 객체가 남성 또는 여성 프롬프트에 더 많이 편향되었는지 점수로 표시합니다. 파란색은 남성 프롬프트에 편향된 객체, 주황색은 여성 프롬프트에 편향된 객체를 나타냅니다.

---



The diagrams and graphs in the paper visually illustrate the gender bias in Stable Diffusion models. Key figures are summarized below:

1. **Figure 1: Comparison of Generated Images by Gender Indicator**  
   This figure compares images generated with the same prompt but differing gender indicators ("neutral," "masculine," "feminine"). For example, the prompt "A person playing an instrument" shows that neutral images resemble masculine ones more closely, while feminine images feature variations in instrument type and layout.

2. **Figure 3: Prompt-Image Dependency Groups**  
   Objects generated during the image creation process are classified into five dependency groups:
   - **Explicitly guided**: Explicitly mentioned in the prompt and guided by cross-attention (e.g., "woman," "picnic").
   - **Implicitly guided**: Not explicitly mentioned but contextually related (e.g., "basket," "blanket").
   - **Explicitly independent**: Explicitly mentioned but not guided by cross-attention (e.g., "park").
   - **Implicitly independent**: Not mentioned in the prompt, inferred from the background (e.g., "grass").
   - **Hidden**: Not included in the image (e.g., "daytime").

3. **Figure 4: Object Frequency Comparison**  
   This figure lists frequently generated objects in each prompt-image dependency group, highlighting gender-based differences in SD v2.0. For instance, the **Implicitly guided** group shows that masculine prompts often generate objects like "shirt" and "microphone," while feminine prompts generate "dress" and "scarf."

4. **Figure 5: Bias Score Graph**  
   Bias scores indicate whether specific objects are more biased toward masculine or feminine prompts in TextCaps and GCC datasets. Blue represents objects biased toward masculine prompts, while orange indicates bias toward feminine prompts.


<br/>
# refer format:     


@inproceedings{wu2024stable,
  title = {Stable Diffusion Exposed: Gender Bias from Prompt to Image},
  author = {Wu, Yankun and Nakashima, Yuta and Garcia, Noa},
  booktitle = {Proceedings of the Seventh AAAI/ACM Conference on AI, Ethics, and Society (AIES 2024)},
  year = {2024},
  pages = {1648--1658},
  publisher = {Association for the Advancement of Artificial Intelligence (AAAI)},

  address = {Osaka, Japan},

}




Wu, Yankun, Yuta Nakashima, and Noa Garcia. “Stable Diffusion Exposed: Gender Bias from Prompt to Image.” In Proceedings of the Seventh AAAI/ACM Conference on AI, Ethics, and Society (AIES 2024), 1648–1658. Osaka, Japan: Association for the Advancement of Artificial Intelligence (AAAI), 2024.
