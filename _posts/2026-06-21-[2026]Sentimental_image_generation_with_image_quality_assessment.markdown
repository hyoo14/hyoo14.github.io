---
layout: post
title:  "[2026]Sentimental image generation with image quality assessment"
date:   2026-06-21 08:16:48 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 감정 기반 이미지 생성을 위한 방법인 SIGQA(Sentimental Image Generation with Image Quality Assessment)를 제안합니다.


짧은 요약(Abstract) :



최근 텍스트 기반의 세부 사항 기반 감정 분석(ABSA)에서 강력한 성과가 나타났지만, 여전히 원시 텍스트 데이터는 제한된 의미적 범위를 제공하는 핵심적인 도전 과제가 남아 있습니다. 이를 해결하기 위해 연구자들은 추가적인 데이터 증강을 통해 향상시키는 방법을 모색해왔습니다. 그러나 기존의 방법들은 원본 데이터와 중복되는 경우가 많아 보조적인 역할을 제대로 수행하지 못하고, 사용자 게시 이미지에 의존하는 경우에는 인간의 주석에 크게 의존하게 되어 오류가 전파되는 문제를 안고 있습니다. 본 연구에서는 텍스트에 맞춤화된 감정적 이미지를 생성하는 새로운 접근 방식을 제안합니다. 우리는 감정적 이미지 생성 및 이미지 품질 평가(SIGQA)라는 방법을 도입하여 텍스트 추출을 강화하는 정밀한 시각적 증강을 제공합니다. 또한, SIGQA는 생성된 이미지를 세분화하여 품질을 평가하는 비참조 이미지 품질 평가를 통합하여 최적의 이미지를 선택합니다. 광범위한 실험을 통해 ACOS 및 en-Phone 데이터셋에서 새로운 최첨단 결과를 확립하였으며, 우리의 방법의 효과성을 강조하고 기능 확장을 위한 유망한 방향을 제시합니다.




Recent advances in textual Aspect-Based Sentiment Analysis (ABSA) have delivered strong performance. Nevertheless, a core challenge remains: raw textual data can only provide limited semantic coverage. To address this issue, researchers have explored enhancement with additional augmentations; however, these methods either heavily overlap with the original data, undermining their ability to be supplementary, or rely on user-posted images, which are extremely dependent on human annotation, propagating errors derived from human mistakes. In this work, we take a previously unexplored path: generating sentimental images tailored to the text. We introduce Sentimental Image Generation with Image Quality Assessment (SIGQA), a method that delivers precise, ancillary visual augmentation to strengthen textual extraction. Furthermore, SIGQA incorporates a no-reference image quality assessment that segments generated images to perform fine-grained quality evaluation, selecting the optimal image for augmentation. Extensive experiments establish new state-of-the-art results on the ACOS and en-Phone datasets, underscoring the effectiveness of our method and highlighting a promising direction for expanding features.


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



이 논문에서 제안하는 방법은 "Sentimental Image Generation with Image Quality Assessment" (SIGQA)라는 이름을 가지고 있으며, 주로 두 가지 주요 구성 요소로 나뉩니다: 감정적 이미지 생성(Sentimental Image Generation)과 이미지 품질 평가(Image Quality Assessment)입니다.

1. **감정적 이미지 생성 (Sentimental Image Generation)**:
   - 이 단계에서는 고객 리뷰와 같은 텍스트 입력을 기반으로 감정적으로 적합한 이미지를 생성합니다. 텍스트가 너무 추상적이거나 명확한 대상을 결여하고 있을 경우, 텍스트-이미지 모델이 의미를 이해하는 데 어려움을 겪을 수 있습니다. 이를 해결하기 위해, 세 가지 주요 단계를 거칩니다:
     - **강조 예측 (Emphasis Prediction)**: 감정 요소에 대한 예측을 통해 텍스트의 의미를 보강합니다. 이 과정에서 LLM(대형 언어 모델)을 사용하여 감정 요소의 '은색 레이블'을 예측합니다.
     - **장면 재작성 (Scene Rewriting)**: 원래의 리뷰와 강조된 요소를 바탕으로 구체적인 장면 설명을 생성합니다. 이 설명은 텍스트-이미지 모델이 이해할 수 있도록 작성됩니다.
     - **이미지 생성 (Image Generation)**: 재작성된 장면 설명을 텍스트-이미지 모델에 입력하여 후보 이미지 풀을 생성합니다. 이 과정에서 원래 리뷰나 예측된 은색 쿼드러플을 기반으로 추가 이미지를 생성하여 후보 풀을 확장합니다.

2. **이미지 품질 평가 (Image Quality Assessment)**:
   - 생성된 이미지의 품질을 평가하기 위해, 세분화 기반의 새로운 평가 방법을 제안합니다. 이 방법은 이미지의 의미 있는 세그먼트를 식별하고 각 세그먼트의 품질을 평가하여 최종 이미지를 선택합니다. 이 과정은 다음과 같이 진행됩니다:
     - **이미지 세분화 (Image Segmentation)**: Segment Anything Model(SAM)을 사용하여 생성된 이미지를 의미 있는 세그먼트로 나눕니다.
     - **세그먼트 평가 (Region Evaluation)**: 각 세그먼트에 대해 VLM을 사용하여 의미적 내용과 품질 평가를 수행합니다.
     - **품질 스코어러 (Quality Scorer)**: 최종적으로, 원래 리뷰와 세그먼트 평가를 결합하여 최종 품질 점수를 산출합니다. 이 점수는 생성된 이미지의 품질을 평가하는 데 사용됩니다.

이 방법은 기존의 사용자 생성 이미지에 의존하는 방법과는 달리, 기계 생성 이미지를 사용하여 감정 분석의 성능을 향상시키는 데 중점을 두고 있습니다. SIGQA는 다양한 데이터셋에서 최첨단 성능을 달성하며, 감정 분석의 새로운 방향을 제시합니다.

---



The method proposed in this paper is called "Sentimental Image Generation with Image Quality Assessment" (SIGQA), which is primarily divided into two main components: Sentimental Image Generation and Image Quality Assessment.

1. **Sentimental Image Generation**:
   - In this stage, emotionally appropriate images are generated based on text inputs such as customer reviews. When the text is too abstract or lacks a clear target, the text-to-image model may struggle to understand the semantics. To address this, the process involves three main steps:
     - **Emphasis Prediction**: This step enhances the meaning of the text by predicting emotional elements. A large language model (LLM) is used to predict "silver labels" for sentiment elements.
     - **Scene Rewriting**: A concrete scene description is generated based on the original review and the emphasized elements. This description is crafted to be interpretable by the text-to-image model.
     - **Image Generation**: The rewritten scene description is fed into a text-to-image model to produce a candidate pool of images. Additional images are generated based on the original review or the predicted silver quadruple to expand the candidate pool.

2. **Image Quality Assessment**:
   - To evaluate the quality of the generated images, a novel segmentation-based assessment method is proposed. This method identifies meaningful segments of the image and evaluates the quality of each segment to select the final image. The process proceeds as follows:
     - **Image Segmentation**: The Segment Anything Model (SAM) is employed to partition the generated image into meaningful segments.
     - **Region Evaluation**: Each segment is evaluated for semantic content and quality using a Vision Language Model (VLM).
     - **Quality Scorer**: Finally, the original review and the segment evaluations are combined to produce a final quality score. This score is used to assess the quality of the generated image.

This method focuses on enhancing sentiment analysis performance by using machine-generated images rather than relying on user-generated images. SIGQA achieves state-of-the-art performance across various datasets, revealing a new direction for sentiment analysis.


<br/>
# Results



이 논문에서는 Sentimental Image Generation with Image Quality Assessment (SIGQA)라는 새로운 방법을 제안하여, 텍스트 기반의 감정 분석(Aspect-Based Sentiment Analysis, ABSA)에서의 성능을 향상시키고자 하였다. SIGQA는 감정에 맞는 이미지를 생성하고, 생성된 이미지의 품질을 평가하여 최적의 이미지를 선택하는 두 가지 주요 구성 요소로 이루어져 있다.

#### 결과 요약

1. **경쟁 모델**: SIGQA는 여러 최신 모델들과 비교되었다. 여기에는 TAS-BERT, Extract-Classify, GAS, Paraphrase, DLO, MvP, MUL, ChatGPT, LLaMA 등이 포함된다. 이들 모델은 전통적인 분류 기반 접근법과 생성 모델을 포함하여 다양한 방법론을 사용하였다.

2. **테스트 데이터**: 실험은 ABSA-ACOS 및 en-Phone 데이터셋을 사용하여 수행되었다. 이 데이터셋들은 감정 분석을 위한 다양한 리뷰와 그에 대한 감정 레이블을 포함하고 있다.

3. **메트릭**: 성능 평가는 Precision (P), Recall (R), F1-score를 사용하여 이루어졌다. F1-score는 모델의 전반적인 성능을 평가하는 데 중요한 지표로 사용되었다.

4. **비교 결과**: SIGQA는 모든 경쟁 모델에 대해 우수한 성능을 보였다. 예를 들어, SIGQA는 Restaurant, Laptop, Phone 데이터셋에서 각각 0.6674, 0.4682, 0.5437의 F1-score를 기록하였다. 이는 기존의 최고 성능 모델들보다 유의미하게 높은 수치이다(𝑝 < .05). 특히, SIGQA는 생성된 감정 이미지를 통해 텍스트 기반 모델의 성능을 크게 향상시켰으며, 이는 감정 분석에서 시각적 신호의 중요성을 강조한다.

5. **이미지 품질 평가**: SIGQA의 이미지 품질 평가(IQA) 구성 요소는 세분화된 지역 평가를 통해 생성된 이미지의 품질을 효과적으로 측정하였다. IQA는 기존의 전체 평가 방법보다 더 나은 성능을 보였으며, 이는 ABSA와 같은 세밀한 작업에서 지역 수준의 평가가 중요함을 보여준다.

6. **데이터 효율성**: SIGQA는 생성된 감정 이미지를 통해 데이터 효율성을 높였다. 제한된 훈련 데이터에서 시각적 신호를 활용함으로써, 모델의 성능이 향상되었고, 이는 특히 데이터가 부족한 상황에서 더욱 두드러졌다.

이러한 결과들은 SIGQA가 텍스트 기반 감정 분석의 성능을 향상시키는 데 있어 효과적이고 실용적인 방법임을 입증하며, 향후 연구에서 시각적 신호를 활용한 데이터 증강의 가능성을 제시한다.

---




This paper proposes a novel method called Sentimental Image Generation with Image Quality Assessment (SIGQA) to enhance performance in text-based sentiment analysis (Aspect-Based Sentiment Analysis, ABSA). SIGQA consists of two main components: generating sentiment-aligned images and assessing the quality of the generated images to select the optimal one.

#### Summary of Results

1. **Competing Models**: SIGQA was compared with several state-of-the-art models, including TAS-BERT, Extract-Classify, GAS, Paraphrase, DLO, MvP, MUL, ChatGPT, and LLaMA. These models employed various methodologies, including traditional classification-based approaches and generative models.

2. **Test Data**: Experiments were conducted using the ABSA-ACOS and en-Phone datasets. These datasets contain various reviews and their corresponding sentiment labels for sentiment analysis.

3. **Metrics**: Performance evaluation was conducted using Precision (P), Recall (R), and F1-score. The F1-score was particularly important as it serves as a key indicator of the model's overall performance.

4. **Comparison Results**: SIGQA demonstrated superior performance over all competing models. For instance, SIGQA achieved F1-scores of 0.6674, 0.4682, and 0.5437 on the Restaurant, Laptop, and Phone datasets, respectively. These scores are significantly higher than those of existing top-performing models (𝑝 < .05). Notably, SIGQA greatly enhanced the performance of text-based models through the use of generated sentiment images, emphasizing the importance of visual signals in sentiment analysis.

5. **Image Quality Assessment**: The Image Quality Assessment (IQA) component of SIGQA effectively measured the quality of generated images through fine-grained regional evaluations. IQA outperformed traditional overall assessment methods, highlighting the significance of local-level assessments in tasks like ABSA.

6. **Data Efficiency**: SIGQA improved data efficiency by utilizing generated sentiment images. By leveraging visual signals, the model's performance was enhanced, particularly in scenarios with limited training data.

These results demonstrate that SIGQA is an effective and practical method for improving performance in text-based sentiment analysis, suggesting future research avenues for utilizing visual signals in data augmentation.


<br/>
# 예제



이 논문에서는 감정 기반 이미지 생성(Sentimental Image Generation)과 이미지 품질 평가(Image Quality Assessment)를 결합한 새로운 방법인 SIGQA(Sentimental Image Generation with Image Quality Assessment)를 제안합니다. 이 방법은 텍스트 기반의 감정 분석(Aspect-Based Sentiment Analysis, ABSA)에서 성능을 향상시키기 위해 설계되었습니다.

#### 1. 데이터셋 및 태스크
- **트레이닝 데이터**: ABSA-ACOS 및 en-Phone 데이터셋을 사용합니다. 이 데이터셋은 고객 리뷰와 해당 리뷰에 대한 감정 요소(예: 측면, 의견, 감정)를 포함하고 있습니다.
- **테스트 데이터**: 동일한 데이터셋에서 분할된 테스트 세트를 사용하여 모델의 성능을 평가합니다.

#### 2. 입력 및 출력
- **입력**: 고객 리뷰 텍스트(예: "이 식당의 서비스는 매우 느리지만 음식은 훌륭하다.")와 함께, 이 리뷰에 대한 감정 요소(예: 측면: 서비스, 의견: 느리다, 감정: 부정적)를 포함합니다.
- **출력**: 모델은 감정 요소를 기반으로 한 감정 쌍(Aspect-Category-Opinion-Sentiment Quadruple)을 생성합니다. 예를 들어, 출력은 다음과 같을 수 있습니다:
  - 측면: 서비스
  - 의견: 느리다
  - 감정: 부정적

#### 3. 이미지 생성 과정
1. **감정 강조 예측**: LLM(대형 언어 모델)을 사용하여 리뷰에서 감정 요소를 예측합니다.
2. **장면 설명 작성**: 예측된 감정 요소를 바탕으로 구체적인 장면 설명을 작성합니다. 예를 들어, "고객이 느린 서비스에 불만을 표출하는 장면"과 같은 설명이 생성됩니다.
3. **이미지 생성**: 장면 설명을 텍스트-이미지 모델에 입력하여 후보 이미지 풀을 생성합니다.

#### 4. 이미지 품질 평가
- 생성된 이미지의 품질을 평가하기 위해, 이미지 품질 평가 모델을 사용하여 각 이미지의 지역별 품질을 평가합니다. 이 과정에서 이미지가 리뷰와 얼마나 잘 연관되어 있는지를 판단합니다.

#### 5. 최종 출력
- 최종적으로, 가장 높은 품질 점수를 받은 이미지를 선택하여 ABSA 태스크에 사용합니다. 이 이미지는 텍스트와 함께 모델에 입력되어 감정 요소의 추출을 강화합니다.




This paper proposes a novel method called SIGQA (Sentimental Image Generation with Image Quality Assessment), which combines sentimental image generation and image quality assessment to enhance performance in text-based sentiment analysis (Aspect-Based Sentiment Analysis, ABSA).

#### 1. Dataset and Task
- **Training Data**: The ABSA-ACOS and en-Phone datasets are used. These datasets contain customer reviews along with sentiment elements (e.g., aspect, opinion, sentiment) related to those reviews.
- **Test Data**: A split test set from the same datasets is used to evaluate the model's performance.

#### 2. Input and Output
- **Input**: Customer review text (e.g., "The service at this restaurant is very slow, but the food is excellent.") along with sentiment elements for that review (e.g., aspect: service, opinion: slow, sentiment: negative).
- **Output**: The model generates sentiment pairs (Aspect-Category-Opinion-Sentiment Quadruple) based on the sentiment elements. For example, the output could be:
  - Aspect: Service
  - Opinion: Slow
  - Sentiment: Negative

#### 3. Image Generation Process
1. **Emphasis Prediction**: A Large Language Model (LLM) is used to predict sentiment elements from the review.
2. **Scene Description Writing**: A concrete scene description is created based on the predicted sentiment elements. For example, a description like "A customer expressing dissatisfaction with slow service" is generated.
3. **Image Generation**: The scene description is input into a text-to-image model to produce a candidate pool of images.

#### 4. Image Quality Assessment
- To evaluate the quality of the generated images, an image quality assessment model is used to assess the quality of each image at a regional level. This process determines how well the image relates to the review.

#### 5. Final Output
- Ultimately, the image with the highest quality score is selected for use in the ABSA task. This image is input into the model along with the text to enhance the extraction of sentiment elements.

<br/>
# 요약

이 논문에서는 감정 기반 이미지 생성을 위한 방법인 SIGQA(Sentimental Image Generation with Image Quality Assessment)를 제안합니다. 이 방법은 텍스트 리뷰를 기반으로 감정적으로 적합한 이미지를 생성하고, 생성된 이미지의 품질을 평가하여 최적의 이미지를 선택함으로써 텍스트 기반 감정 분석의 성능을 향상시킵니다. 실험 결과, SIGQA는 여러 벤치마크 데이터셋에서 최첨단 성능을 달성하며, 생성된 이미지가 감정 분석의 정확성을 높이는 데 기여함을 보여줍니다.

---

This paper proposes a method called SIGQA (Sentimental Image Generation with Image Quality Assessment) for generating sentiment-based images. The method generates emotionally appropriate images based on text reviews and evaluates the quality of the generated images to select the optimal one, thereby enhancing the performance of text-based sentiment analysis. Experimental results demonstrate that SIGQA achieves state-of-the-art performance on multiple benchmark datasets, showing that the generated images contribute to improving the accuracy of sentiment analysis.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Sentimental Image Generation Pipeline (Fig. 2)**: 이 다이어그램은 감정적 이미지를 생성하는 과정의 세 단계를 보여줍니다. 첫 번째 단계는 감정 요소의 예측, 두 번째는 장면 설명의 재작성, 세 번째는 텍스트-이미지 모델을 통한 이미지 생성입니다. 이 과정은 감정적 이미지가 텍스트의 의미를 보다 잘 반영하도록 돕습니다.
   - **Image Quality Assessment Pipeline (Fig. 3)**: 이 다이어그램은 이미지 품질 평가의 세부 단계를 설명합니다. 이미지가 세분화되고 각 세그먼트에 대해 평가가 이루어지며, 최종적으로 가장 적합한 이미지를 선택하는 과정을 보여줍니다. 이는 감정적 이미지의 품질을 정밀하게 평가할 수 있는 방법을 제시합니다.
   - **Vision-Language Model Integration (Fig. 4)**: 이 피규어는 텍스트와 이미지를 통합하는 비전-언어 모델의 구조를 설명합니다. 텍스트와 이미지의 정보를 결합하여 감정 분석을 수행하는 방법을 시각적으로 나타냅니다.

2. **테이블**
   - **성능 비교 (Table 2)**: 다양한 모델과의 성능 비교 결과를 보여줍니다. SIGQA 모델은 기존의 분류 기반 및 생성 모델보다 우수한 성능을 보이며, 감정적 이미지 생성이 텍스트 기반 감정 분석에 긍정적인 영향을 미친다는 것을 입증합니다.
   - **강조 방법의 기여도 (Table 3)**: 다양한 강조 방법을 사용했을 때의 F1 점수를 비교합니다. 감정적 이미지를 포함했을 때 성능이 크게 향상되며, 특히 Pair Emphasis가 가장 높은 성능을 보입니다. 이는 강조 방법이 감정적 이미지의 효과를 극대화하는 데 중요한 역할을 한다는 것을 나타냅니다.
   - **이미지 품질 평가 방법 비교 (Table 4)**: 제안된 IQA 방법이 기존의 전반적인 평가 방법보다 우수한 성능을 보임을 보여줍니다. 이는 세분화된 지역 수준의 평가가 감정 분석과 같은 세밀한 작업에 더 적합하다는 것을 시사합니다.

3. **어펜딕스**
   - 어펜딕스에는 실험 설정, 데이터셋, 모델 아키텍처 등 연구의 세부 사항이 포함되어 있습니다. 이는 연구의 재현성을 높이고, 다른 연구자들이 이 방법을 기반으로 추가 연구를 수행할 수 있도록 돕습니다.

---

### Insights and Results from Other Components (Diagrams, Figures, Tables, Appendices)

1. **Diagrams and Figures**
   - **Sentimental Image Generation Pipeline (Fig. 2)**: This diagram illustrates the three-step process of generating sentimental images. The first step involves predicting sentiment elements, the second step is rewriting scene descriptions, and the third step is generating images through a text-to-image model. This process helps ensure that the generated sentimental images better reflect the meaning of the text.
   - **Image Quality Assessment Pipeline (Fig. 3)**: This diagram details the steps involved in assessing image quality. It shows how images are segmented, evaluated for each segment, and ultimately how the best-fitting image is selected. This provides a method for precisely evaluating the quality of sentimental images.
   - **Vision-Language Model Integration (Fig. 4)**: This figure explains the structure of the vision-language model that integrates text and images. It visually represents how the information from text and images is combined to perform sentiment analysis.

2. **Tables**
   - **Performance Comparison (Table 2)**: This table presents the performance comparison results with various models. The SIGQA model outperforms existing classification-based and generative models, demonstrating that generating sentimental images positively impacts text-based sentiment analysis.
   - **Contribution of Emphasis Methods (Table 3)**: This table compares the F1 scores when using various emphasis methods. The inclusion of generated sentimental images significantly enhances performance, with Pair Emphasis yielding the highest scores. This indicates that emphasis methods play a crucial role in maximizing the effectiveness of sentimental images.
   - **Comparison of Image Quality Assessment Methods (Table 4)**: This table shows that the proposed IQA method outperforms traditional overall assessment methods. It suggests that fine-grained regional assessments are more suitable for tasks like sentiment analysis, which require detailed evaluations.

3. **Appendices**
   - The appendices include details on experimental settings, datasets, and model architectures, enhancing the reproducibility of the research and enabling other researchers to build upon this work for further studies.

<br/>
# refer format:



```bibtex
@article{bao2026sentimental,
  title={Sentimental image generation with image quality assessment},
  author={Bao, Xiaoyi and Gu, Jinghang and Wang, Zhongqing and Huang, Chu-Ren},
  journal={Pattern Recognition},
  volume={177},
  pages={113269},
  year={2026},
  publisher={Elsevier},
  doi={10.1016/j.patcog.2026.113269}
}
```




Xiaoyi Bao, Jinghang Gu, Zhongqing Wang, and Chu-Ren Huang. "Sentimental Image Generation with Image Quality Assessment." *Pattern Recognition* 177 (2026): 113269. https://doi.org/10.1016/j.patcog.2026.113269.
