---
layout: post
title:  "[2025]Enhancing Stress Detection on Social Media Through Multi-Modal Fusion of Text and Synthesized Visuals"  
date:   2025-08-02 07:23:40 +0200
categories: study
---

{% highlight ruby %}


한줄 요약: 

제일 인상적이고 응용 가능하기 좋았던 거 같은 논문  
텍스트 기반 스트레스 탐지에서 이미지를 추가로 사용
근데 이 이미지는 텍스트를 디퓨전 모델에 넣어서 만든 것.. 이렇게 만든 이미지를 활용하여 더 나은 성능 달성..굿!  



짧은 요약(Abstract) :    




이 논문은 기존의 **텍스트 기반 스트레스 탐지 기법의 한계**를 보완하기 위해, **텍스트와 생성된 이미지(시각 정보)를 함께 사용하는 새로운 다중모달 프레임워크**를 제안합니다.
연구진은 **DALL·E 모델**을 이용해 소셜 미디어 게시글에서 시각적 이미지를 생성하고, 이를 **CLIP 모델**을 통해 텍스트와 함께 통합된 의미 공간(embedding space)으로 변환하여 스트레스 여부를 분류합니다.

실험은 **Dreaddit 데이터셋**에서 수행되었으며,

* CLIP의 feature를 고정하고 분류기만 학습한 경우 **94.90%의 정확도**
* 전체 CLIP 모델을 정교하게 튜닝한 경우 **98.41%의 정확도**를 달성했습니다.

이 결과는, **생성된 이미지가 텍스트만으로는 포착할 수 없는 정서적 단서나 맥락 정보를 보완**하며,
스트레스 탐지를 위한 더 강력하고 정교한 방법을 가능하게 한다는 것을 보여줍니다.
이 접근법은 향후 **정신 건강 모니터링 및 소셜 미디어 분석** 분야에 혁신적인 방향을 제시합니다.

---



> **Abstract**
> Social media platforms generate an enormous volume of multi-modal data, yet stress detection research has predominantly relied on text-based analysis. In this work, we propose a novel framework that integrates textual content with synthesized visual cues to enhance stress detection. Using the generative model DALL·E, we synthesize images from social media posts, which are then fused with text through the multi-modal capabilities of a pre-trained CLIP model, which encodes both text and image data into a shared semantic space. Our approach is evaluated on the Dreaddit dataset, where a classifier trained on frozen CLIP features achieves 94.90% accuracy, and full fine-tuning further improves performance to 98.41%. These results underscore the integration of synthesized visuals with textual data not only enhances stress detection but also offers a robust method over traditional text-only methods, paving the way for innovative approaches in mental health monitoring and social media analytics.





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



이 연구는 **텍스트와 생성 이미지**를 함께 활용해 스트레스 여부를 판단하는 **다중모달 프레임워크**를 제안합니다. 전체 모델 아키텍처는 다음과 같은 두 단계로 이루어져 있습니다.

#### 1. 이미지 생성 (Image Generation)

* **DALL·E 3** 모델을 사용하여, 소셜 미디어 글을 바탕으로 **스트레스 상태를 시각적으로 표현한 이미지**를 생성합니다.
* 입력된 텍스트는 먼저 **텍스트 인코더**를 통해 의미 정보를 포함한 벡터로 변환되고,
* 이 벡터는 **디퓨전 prior 모듈**을 통해 이미지 표현으로 변환됩니다.
* 마지막으로 **이미지 디코더**가 이 벡터를 바탕으로 실제 이미지를 생성합니다.
* 생성된 이미지는 예: 어두운 색감, 불안한 분위기, 걱정스러운 표정 등 스트레스를 시각적으로 묘사한 형태입니다.

#### 2. CLIP 기반 다중모달 표현 (Multi-modal Representation with CLIP)

* **사전학습된 CLIP 모델**을 활용하여 텍스트와 이미지 모두를 **공통 임베딩 공간**에 매핑합니다.
* 텍스트 인코더와 이미지 인코더를 통해 각각의 표현을 얻은 뒤, 이 둘을 정규화하고 결합하여 최종 임베딩을 구성합니다.
* 두 가지 학습 전략이 사용됩니다:

  * **Classifier-Only Training**: CLIP의 텍스트/이미지 인코더는 고정시키고, 분류기(fully connected layers)만 학습합니다.
  * **Full Fine-Tuning**: CLIP 전체 모델과 분류기를 모두 정교하게 조정(fine-tune)하여 스트레스 표현에 더 잘 맞도록 만듭니다.

이러한 접근은 **텍스트만으로는 포착하기 어려운 감정 신호**를 **이미지와 결합함으로써 보완**할 수 있도록 설계되었습니다.

---


The proposed method integrates two main components to enhance stress detection on social media:

#### 1. Image Generation

The authors use the **DALL·E 3** generative model to synthesize images from textual descriptions of social media posts. The text is encoded using a specialized encoder, which captures the semantic essence. A **diffusion prior** maps the text representation to an image embedding, which is then decoded to generate a **synthetic image** that visually reflects stress-related cues such as dark tones, anxious facial expressions, or cluttered environments.

#### 2. Multi-modal Representation with CLIP

They utilize the **pre-trained CLIP model**, which encodes both text and images into a **shared semantic embedding space**. Each post is represented by combining features from the CLIP text and image encoders. Two training strategies are applied:

* **Classifier-Only Training**: Freezes CLIP’s encoders and trains only the classifier layers.
* **Full Fine-Tuning**: Fine-tunes the entire CLIP model along with the classifier to better adapt to stress detection.

This two-stage framework enables the model to capture **complementary affective signals** across modalities, improving detection accuracy beyond text-only methods.



   
 
<br/>
# Results  


####  테스트 데이터

* 사용된 데이터셋은 **Dreaddit**로, 소셜 미디어 (주로 Reddit)에서 수집된 **스트레스 여부가 라벨링된 게시글**들로 구성되어 있습니다.
* 전체 데이터는 **학습 2,837개, 테스트 414개 샘플**로 구성되어 있으며, 두 클래스(스트레스/비스트레스)가 균형을 이루고 있습니다.

####  비교 모델 (Baselines)

1. **텍스트 전용 모델들 (Text-Only)**

   * BERT-base
   * RoBERTa-base
   * MentalRoBERTa (정신건강 특화 사전학습 모델)
   * MentalBERT

2. **제안된 다중모달 모델 (Ours)**

   * **CLIP Classifier-Only**: CLIP의 본체는 고정하고 분류기만 학습
   * **CLIP Full Fine-Tuning**: CLIP 전체와 분류기 모두 정교 조정

####  주요 성능 지표 (Metrics)

* **정확도 (Accuracy)**
* **가중 F1 점수 (Weighted F1 Score)**: 클래스 간 불균형을 고려한 정밀도와 재현율의 조화 평균

####  주요 결과 요약

| 모델                             | 정확도 (%)   | F1 점수 (%) |
| ------------------------------ | --------- | --------- |
| **MentalRoBERTa**              | 96.14     | 94.24     |
| **MentalBERT**                 | 69.32     | 78.75     |
| **BERT-base**                  | 96.14     | 94.67     |
| **CLIP Classifier-Only (제안)**  | 94.90     | 92.42     |
| **CLIP Full Fine-Tuning (제안)** | **98.41** | **98.27** |

* 제안된 모델 중 **CLIP Full Fine-Tuning**이 모든 모델을 **압도적으로 능가**함.
* CLIP을 고정하고 분류기만 학습해도 94.90%로 상당히 강력했으나, 전체 모델을 조정하니 98.41%로 크게 향상됨.
* **MentalBERT는 성능이 낮았으며**, 일반 언어모델(RoBERTa, BERT-base)과 정신건강 특화 모델(MentalRoBERTa)보다 성능이 떨어졌음.

####  Ablation Study (단일 모달 vs 다중모달 비교)

| 입력 모달리티                  | 정확도 (%)   | F1 점수 (%) |
| ------------------------ | --------- | --------- |
| 이미지만 사용 (Image-Only)     | 95.22     | 93.17     |
| 텍스트만 사용 (Text-Only)      | 96.82     | 96.31     |
| 텍스트 + 이미지 (Image + Text) | **98.41** | **98.27** |

* **텍스트만 사용해도 강력한 성능**을 보였지만, 이미지와 결합할 때 **감정 신호가 보완되어 성능이 더욱 향상**됨.
* 이는 생성 이미지가 실제로 **텍스트의 정서적/맥락적 신호를 증폭**해주는 역할을 한다는 것을 보여줍니다.

---


####  Test Dataset

* The authors use the **Dreaddit** dataset, containing social media posts (primarily from Reddit) labeled for the presence or absence of stress.
* The dataset consists of **2,837 training samples** and **414 test samples**, balanced across the two classes.

####  Baseline Models

1. **Text-Only Models**:

   * BERT-base
   * RoBERTa-base
   * MentalRoBERTa (domain-specific pretrained model)
   * MentalBERT

2. **Proposed Multi-Modal Models**:

   * **CLIP Classifier-Only**: Freezes the CLIP backbone and trains only the classifier.
   * **CLIP Full Fine-Tuning**: Fine-tunes both the CLIP model and classifier jointly.

#### Evaluation Metrics

* **Accuracy**
* **Weighted F1 Score** (to account for class balance)

####  Main Results

| Model                            | Accuracy (%) | Weighted F1 (%) |
| -------------------------------- | ------------ | --------------- |
| **MentalRoBERTa**                | 96.14        | 94.24           |
| **MentalBERT**                   | 69.32        | 78.75           |
| **BERT-base**                    | 96.14        | 94.67           |
| **CLIP Classifier-Only (Ours)**  | 94.90        | 92.42           |
| **CLIP Full Fine-Tuning (Ours)** | **98.41**    | **98.27**       |

* The **CLIP Full Fine-Tuning** model outperformed all baselines, demonstrating the power of integrating synthesized images with text.
* Even the classifier-only CLIP setup achieved strong results (94.90%), but full fine-tuning led to significant gains.
* **MentalBERT underperformed**, while general-purpose models like BERT and RoBERTa performed comparably to MentalRoBERTa.

####  Ablation Study: Modal Contributions

| Modality     | Accuracy (%) | Weighted F1 (%) |
| ------------ | ------------ | --------------- |
| Image Only   | 95.22        | 93.17           |
| Text Only    | 96.82        | 96.31           |
| Image + Text | **98.41**    | **98.27**       |

* Text-only models were already highly effective.
* However, fusing text with **synthesized visual information** yielded the **highest performance**, indicating that visuals offer **complementary emotional signals** not captured by text alone.

---



<br/>
# 예제  



####  수행 테스크

이 논문에서의 핵심 테스크는 다음과 같습니다:

> **이진 분류 (Binary Classification)**
> 소셜 미디어 게시글을 입력받아,
> → **해당 글을 쓴 사람이 스트레스를 겪고 있는지 아닌지**를 예측

---

####  사용된 데이터셋: **Dreaddit**

* 출처: Reddit의 정신건강 관련 서브레딧 (예: `r/depression`, `r/anxiety`, `r/relationships` 등)
* 구성:

  * **학습 데이터**: 2,837개
  * **테스트 데이터**: 414개
  * 각 게시물은 **텍스트 1개 + 이진 라벨(스트레스 여부)** 로 구성됨
  * 라벨링은 **3명의 크라우드워커의 투표**로 결정됨
  * 클래스 비율은 거의 **균형** (스트레스 / 비스트레스)

---

####  구체적인 입력 예시 (인풋)

1. **텍스트 입력 (Social Media Post)**
   예시:

   > *"I feel like I’m drowning in responsibilities. Nothing I do seems to help and I’m so tired of pretending I’m okay."*
   > → DALL·E 3를 통해 다음과 같은 **시각 이미지로 변환**됨:

   * 어두운 조명
   * 피곤해 보이는 사람의 얼굴
   * 복잡하고 혼란스러운 환경 배경

2. **시각 정보 입력 (Synthesized Image)**

   * 위 텍스트를 기반으로 **DALL·E 3**가 생성한 이미지 (1024×1024 해상도)
   * 텍스트의 감정을 시각적으로 반영 (예: 불안, 탈진, 우울함 등)

3. **다중모달 입력 처리 방식**

   * 텍스트와 이미지 각각을 CLIP 모델로 인코딩하고, **공동 임베딩 공간에서 결합**하여 사용

---

####  출력 (Output Label)

* **0 = 스트레스 없음**
* **1 = 스트레스 있음**

예:

| 텍스트                                                      | 생성 이미지                   | 출력 라벨 |
| -------------------------------------------------------- | ------------------------ | ----- |
| “I’m falling apart and no one seems to notice.”          | 우울한 배경, 흐릿한 실내, 혼자 있는 인물 | 1     |
| “Had a peaceful walk with my dog today, feeling better.” | 맑은 날씨, 반려동물, 공원 배경       | 0     |

---


####  Task Description

The primary task is:

> **Binary classification**
> Given a social media post (text),
> → Predict whether the author is **experiencing stress** or not.

---

####  Dataset Used: **Dreaddit**

* Source: Reddit posts from mental health-related subreddits (e.g., `r/depression`, `r/anxiety`, `r/relationships`)
* Composition:

  * **Training samples**: 2,837
  * **Test samples**: 414
  * Each post is paired with a **binary stress label** (0 or 1)
  * Labels were assigned by **three crowd annotators** with majority voting
  * The dataset is **approximately balanced** between the two classes

---

####  Input Format

1. **Textual Input (Post Content)**
   Example:

   > *"I feel like I’m drowning in responsibilities. Nothing I do seems to help and I’m so tired of pretending I’m okay."*

   → This text is used to **generate a visual image** via DALL·E 3:

   * Dim lighting
   * A tired facial expression
   * Cluttered or overwhelming surroundings

2. **Visual Input (Synthesized Image)**

   * DALL·E 3 generates a 1024×1024 image reflecting the affective state implied in the post
   * Visual cues include emotional tone, color scheme, facial features (if any), and background elements

3. **Multi-Modal Encoding**

   * Both the text and image are encoded using **CLIP**
   * The resulting embeddings are **fused into a joint representation** for classification

---

####  Output Labels

* **0 = No Stress**
* **1 = Stress**

Example:

| Post                                                     | Synthesized Image                          | Output Label |
| -------------------------------------------------------- | ------------------------------------------ | ------------ |
| “I’m falling apart and no one seems to notice.”          | Sad environment, solitary person, dim room | 1            |
| “Had a peaceful walk with my dog today, feeling better.” | Park scene, sunshine, dog                  | 0            |





<br/>  
# 요약   




이 연구는 DALL·E를 이용해 소셜 미디어 게시글로부터 스트레스를 시각적으로 표현한 이미지를 생성하고, 이를 텍스트와 함께 CLIP 모델에 입력하는 다중모달 스트레스 탐지 프레임워크를 제안한다.
Dreaddit 데이터셋에서 전체 모델을 정교하게 튜닝했을 때 98.41%의 정확도를 달성하며 기존 텍스트 기반 모델보다 성능이 우수했다.
예시로 “지치고 아무도 날 신경 안 써” 같은 게시글은 어두운 분위기의 이미지와 함께 스트레스가 있는 것으로 분류된다.



This study proposes a multi-modal stress detection framework that generates visuals from social media posts using DALL·E and integrates them with text using the CLIP model.
On the Dreaddit dataset, full fine-tuning of the CLIP model achieved 98.41% accuracy, outperforming traditional text-only models.
For example, a post like “I’m exhausted and no one seems to care” is paired with a dark, emotionally charged image and classified as stressful.




<br/>  
# 기타  




####  테이블 1: 모델별 성능 비교

* 다양한 텍스트 기반 모델(MentalBERT, BERT, RoBERTa 등)과 제안한 CLIP 기반 다중모달 모델 간 성능을 비교함.
* \*\*CLIP Full Fine-Tuning 모델이 최고 성능(정확도 98.41%, F1 98.27%)\*\*을 기록, 기존 모델보다 의미 있는 향상을 보여줌.
* 특히, **MentalBERT가 다른 모델에 비해 성능이 현저히 낮음**을 통해, 단순히 정신건강 특화 사전학습만으로는 한계가 있음을 시사함.

####  테이블 2: Ablation Study (모달리티 별 기여)

* \*\*텍스트만 사용해도 높은 정확도(96.82%)\*\*를 기록했지만, 이미지와 결합할 경우 성능이 **98.41%까지 증가**함.
* **이미지만 사용해도 95.22%의 정확도**를 보여, 생성 이미지가 단독으로도 감정 신호를 어느 정도 포착할 수 있음을 보여줌.
* 결론적으로 **텍스트 + 이미지의 결합이 정서적 신호를 보완하여 성능을 극대화**함을 입증함.

####  Figure 1: 전체 모델 아키텍처 다이어그램

* 왼쪽은 DALL·E 기반 이미지 생성, 오른쪽은 텍스트+이미지를 CLIP으로 인코딩하여 공동 임베딩 공간에서 분류하는 구조를 시각적으로 설명.
* **Classifier-Only vs Full Fine-Tuning 전략**의 차이도 표시되어 있어, 학습 방식의 선택에 따른 유연성도 시사함.
* 이 그림을 통해 전체 파이프라인의 직관적인 이해와 두 단계(생성 + 표현)의 결합이 명확히 드러남.

####  Figure 2: Dreaddit 예시 텍스트와 생성 이미지

* 스트레스 있는 예시 게시글과 없는 게시글 각각에 대해 DALL·E가 생성한 이미지를 시각적으로 제시함.
* **감정 상태에 따라 시각적 요소(배경 색, 표정, 조도 등)가 확연히 구분**되어, 생성 이미지가 실제로 의미 있는 신호를 제공함을 보여줌.
* 인간의 직관으로도 스트레스 여부를 유추할 수 있을 만큼 **감성적 일관성이 확보**됨.

####  어펜딕스: 하이퍼파라미터 및 생성 프롬프트

* DALL·E 3에 사용된 **프롬프트 디자인**이 상세히 기술되어 있음: 어두운 조명, 불안한 표정, 차가운 색감 등 스트레스를 시각적으로 유도.
* CLIP 학습 설정도 **Classifier 전용 학습 vs Full Fine-Tuning**으로 나누어, 각각에 맞는 학습률과 옵티마이저 조합을 제시함.
* 생성 이미지가 실제로 감정 정보를 얼마나 반영하는지에 대한 **시스템적 평가가 빠져 있음** → 후속 연구 필요.

---



####  Table 1: Performance Comparison Across Models

* Compares traditional text-only models (MentalBERT, BERT, RoBERTa) with the proposed CLIP-based multi-modal models.
* **CLIP Full Fine-Tuning achieves the highest performance (98.41% accuracy, 98.27% F1)**, showing significant improvement.
* MentalBERT performs poorly, suggesting that domain-specific pretraining alone may not suffice.

####  Table 2: Ablation Study – Modality Contributions

* **Text-only model performs well (96.82% accuracy)**, but fusing text with generated images increases performance to **98.41%**.
* Interestingly, the image-only model still reaches **95.22%**, implying that the synthesized visuals carry useful affective cues.
* Overall, the combination of both modalities clearly yields the best result, reinforcing the value of multi-modal fusion.

####  Figure 1: Model Architecture Diagram

* Illustrates the full pipeline: image generation using DALL·E on the left, and multi-modal representation using CLIP on the right.
* Shows two training strategies: Classifier-Only and Full Fine-Tuning, highlighting flexibility in implementation.
* The diagram effectively conveys the **two-stage pipeline and how textual and visual cues are jointly encoded**.

####  Figure 2: Examples from Dreaddit with Synthesized Images

* Displays side-by-side examples of posts labeled with stress and no-stress, along with their generated images.
* **Visual cues (e.g., lighting, facial expressions, backgrounds)** align with the affective content of the text.
* The emotional consistency between text and image reinforces that DALL·E’s outputs provide meaningful features.

####  Appendix: Hyperparameters and Prompts

* Details the **DALL·E prompt engineering**, including stress-related visuals (dim lighting, distress, cold colors).
* Specifies different hyperparameters for **Classifier-Only vs Full Fine-Tuning** strategies in CLIP.
* Notably, the study does **not quantitatively evaluate whether generated visuals accurately reflect the intended emotion**, suggesting a future research direction.




<br/>
# refer format:     


@inproceedings{Soufleri2025StressDetection,
  author    = {Efstathia Soufleri and Sophia Ananiadou},
  title     = {Enhancing Stress Detection on Social Media Through Multi-Modal Fusion of Text and Synthesized Visuals},
  booktitle = {Proceedings of the 24th Workshop on Biomedical Language Processing (BioNLP 2025)},
  year      = {2025},
  pages     = {34--43},
  address   = {Vienna, Austria},
  month     = {August},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2025.bionlp-1.4}
}



Soufleri, Efstathia, and Sophia Ananiadou. "Enhancing Stress Detection on Social Media Through Multi-Modal Fusion of Text and Synthesized Visuals." Proceedings of the 24th Workshop on Biomedical Language Processing (BioNLP 2025), 34–43. Vienna, Austria: Association for Computational Linguistics, August 2025. https://aclanthology.org/2025.bionlp-1.4.





