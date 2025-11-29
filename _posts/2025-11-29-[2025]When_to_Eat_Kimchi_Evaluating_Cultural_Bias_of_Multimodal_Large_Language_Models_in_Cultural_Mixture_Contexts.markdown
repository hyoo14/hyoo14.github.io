---
layout: post
title:  "[2025]When to Eat Kimchi: Evaluating Cultural Bias of Multimodal Large Language Models in Cultural Mixture Contexts"
date:   2025-11-29 00:35:29 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 연구에서는 MIXCUBE라는 데이터셋을 사용하여 다문화 대형 언어 모델(MLLMs)의 문화적 편향을 평가




짧은 요약(Abstract) :


이 논문은 다문화적 입력에 대한 다중 모달 대형 언어 모델(MLLMs)의 문화적 편향을 평가하는 연구입니다. 글로벌화가 진행됨에 따라, MLLMs는 다양한 문화 요소를 인식하고 올바르게 반응하는 것이 중요합니다. 예를 들어, 모델은 아시아 여성이 김치를 먹고 있는 이미지와 아프리카 남성이 김치를 먹고 있는 이미지를 모두 올바르게 인식해야 합니다. 그러나 현재의 MLLMs는 사람의 시각적 특징에 과도하게 의존하여 잘못된 분류를 초래하고 있습니다. 이 연구에서는 MIXCUBE라는 교차 문화 편향 벤치마크를 도입하고, 다섯 개 국가와 네 개 민족의 요소를 연구하여 MLLMs의 강건성을 평가합니다. 연구 결과, MLLMs는 고자원 문화에서 더 높은 정확도와 낮은 민감도를 보이는 반면, 저자원 문화에서는 그렇지 않음을 발견했습니다. GPT-4o 모델은 저자원 문화에서 원본과 변형된 문화 설정 간에 최대 58%의 정확도 차이를 보였습니다.



This paper evaluates the cultural bias of multimodal large language models (MLLMs) in the context of cultural mixture. In a highly globalized world, it is important for MLLMs to recognize and respond correctly to mixed cultural inputs. For instance, a model should correctly identify kimchi when an Asian woman is eating it, as well as when an African man is eating it. However, current MLLMs show an over-reliance on the visual features of the person, leading to misclassification. To examine the robustness of MLLMs across different ethnicities, we introduce MIXCUBE, a cross-cultural bias benchmark, and study elements from five countries and four ethnicities. Our findings reveal that MLLMs achieve higher accuracy and lower sensitivity to such perturbations for high-resource cultures, but not for low-resource cultures. The GPT-4o model shows up to a 58% difference in accuracy between the original and perturbed cultural settings in low-resource cultures.


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



이 연구에서는 다문화 대형 언어 모델(MLLMs)의 문화적 편향을 평가하기 위해 MIXCUBE라는 새로운 벤치마크 데이터셋을 도입했습니다. 이 데이터셋은 2,500개의 이미지로 구성되어 있으며, 다섯 개의 문화(아제르바이잔, 한국, 미얀마, 영국, 미국)와 세 가지 카테고리(음식, 축제, 의류)를 포함합니다. 각 이미지는 원본 이미지와 함께, 인종을 변경한 합성 이미지로 구성되어 있습니다. 이 과정에서 사용된 주요 방법론은 다음과 같습니다.

1. **이미지 수집**: 원본 이미지는 자동 웹 스크래핑 도구와 수동 검색 절차를 통해 수집되었습니다. 수집 기준은 문화적 마커가 명확하고, 해당 문화의 원주율을 반영하도록 설정되었습니다.

2. **이미지 합성**: 원본 이미지에서 사람의 얼굴을 자동으로 마스킹한 후, 인페인팅 기법을 사용하여 다른 인종의 사람으로 대체하는 합성 이미지를 생성했습니다. 이 과정에서 Segment Anything Model(SAM)과 Stability REST API를 활용했습니다. 인종은 아프리카, 백인, 동아시아, 남아시아로 구분하였으며, 각 인종의 특징을 반영하여 합성 이미지를 생성했습니다.

3. **모델 평가**: MLLMs의 성능을 평가하기 위해 두 가지 작업을 수행했습니다. 첫 번째는 국가 식별 작업으로, 주어진 문화 마커의 출처 국가를 식별하는 것이며, 두 번째는 문화 마커 식별 작업으로, 음식의 이름을 식별하는 것입니다. 이 과정에서 GLM-4-Plus라는 평가 모델을 사용하여 MLLMs의 응답이 정확한지 평가했습니다.

4. **정확도 측정**: 각 작업의 정확도는 MLLMs의 성능을 정량적으로 평가하는 데 사용되었습니다. 원본 이미지와 합성 이미지의 정확도를 비교하여 인종 변경이 MLLMs의 성능에 미치는 영향을 분석했습니다.

이 연구는 MLLMs가 고자원 문화에 비해 저자원 문화에 대한 인식이 낮다는 것을 발견하였으며, 이는 더 다양한 문화적 데이터를 통해 개선할 필요가 있음을 강조합니다.




In this study, we introduced a new benchmark dataset called MIXCUBE to evaluate the cultural bias of multimodal large language models (MLLMs). This dataset consists of 2,500 images spanning five cultures (Azerbaijan, South Korea, Myanmar, the UK, and the US) and three categories (food, festivals, and clothing). Each image is accompanied by synthesized images where the ethnicity of the person is altered. The key methodologies employed in this process are as follows:

1. **Image Collection**: Original images were collected using an automatic web scraping tool and a manual search procedure. The collection criteria were established to ensure that cultural markers were clear and representative of the respective culture.

2. **Image Synthesis**: After automatically masking the faces in the original images, we generated synthesized images by replacing the person with someone of a different ethnicity using inpainting techniques. This process utilized the Segment Anything Model (SAM) and the Stability REST API. Ethnicities were categorized as African, Caucasian, East Asian, and South Asian, and the synthesis reflected the characteristics of each ethnicity.

3. **Model Evaluation**: We evaluated the performance of MLLMs through two tasks. The first task was Country Identification, which involved identifying the source country of a given cultural marker, while the second task was Cultural Marker Identification, which focused on identifying the name of a food item. For this, we used an evaluator model called GLM-4-Plus to assess whether the responses from MLLMs were accurate.

4. **Accuracy Measurement**: The accuracy of each task was used to quantitatively assess the performance of the MLLMs. We compared the accuracy of original images with that of synthesized images to analyze the impact of ethnicity alteration on MLLM performance.

The study found that MLLMs exhibit lower awareness of low-resource cultures compared to high-resource cultures, highlighting the need for more diverse cultural data to improve their cultural understanding.


<br/>
# Results



이 연구에서는 다문화 대형 언어 모델(MLLMs)의 문화적 편향을 평가하기 위해 MIXCUBE라는 새로운 벤치마크를 도입했습니다. 이 벤치마크는 아제르바이잔, 미얀마, 한국, 영국, 미국의 다섯 가지 문화에서 수집된 2,500개의 이미지를 포함하고 있으며, 각 이미지는 음식, 축제, 의류와 같은 세 가지 문화적 마커 카테고리로 분류됩니다. 연구의 주요 목표는 MLLMs가 다양한 민족적 배경을 가진 사람들과 함께 있는 문화적 요소를 얼마나 잘 인식하는지를 평가하는 것입니다.

#### 경쟁 모델
연구에서는 세 가지 MLLM 모델을 평가했습니다:
1. **GPT-4o**: OpenAI에서 개발한 모델로, 전반적으로 가장 높은 성능을 보였습니다.
2. **GLM-4v-Plus**: ZhipuAI에서 개발한 모델로, 다양한 문화적 마커 인식에서 안정적인 성능을 보였습니다.
3. **InternVL2.5**: OpenGVLab에서 개발한 모델로, 상대적으로 낮은 정확도를 보였지만 일관성을 유지했습니다.

#### 테스트 데이터
MIXCUBE 데이터셋은 각 문화에서 33개의 원본 이미지를 수집하고, 각 이미지에 대해 네 가지 민족적 배경(아프리카, 백인, 동아시아, 남아시아)으로 대체한 합성 이미지를 생성하여 총 2,500개의 이미지를 구성했습니다. 각 이미지는 문화적 출처와 관련된 레이블이 부여되었습니다.

#### 메트릭
모델의 성능은 두 가지 주요 작업을 통해 평가되었습니다:
1. **국가 식별 정확도**: 주어진 문화적 마커의 출처 국가를 식별하는 능력.
2. **문화적 마커 식별 정확도**: 이미지에서 음식의 이름을 식별하는 능력.

정확도는 각 작업에서 모델의 성능을 정량화하는 데 사용되었습니다. 원본 이미지에 대한 정확도는 일반적으로 합성 이미지보다 높았으며, 특히 저자원 문화(예: 아제르바이잔, 미얀마)에서 더 큰 성능 저하가 관찰되었습니다.

#### 비교
연구 결과, MLLMs는 고자원 문화(예: 한국, 영국, 미국)에서 더 높은 정확도를 보였으며, 저자원 문화에서는 정확도가 낮고 민족적 배경에 따라 성능이 크게 달라지는 경향이 있었습니다. 예를 들어, GPT-4o는 저자원 문화에서 원본 이미지와 합성 이미지 간의 정확도 차이가 최대 58%에 달하는 것으로 나타났습니다. 반면, 고자원 문화에서는 민족적 배경에 따른 정확도 차이가 상대적으로 적었습니다.

이 연구는 MLLMs가 문화적 요소를 인식하는 데 있어 민족적 배경에 따라 편향이 존재함을 보여주며, 더 다양한 문화적 데이터를 통해 이러한 편향을 줄일 필요성을 강조합니다.

---



This study introduces a new benchmark called MIXCUBE to evaluate the cultural bias of multimodal large language models (MLLMs). The benchmark consists of 2,500 images collected from five cultures: Azerbaijan, Myanmar, South Korea, the UK, and the US, categorized into three cultural markers: food, festivals, and clothing. The primary goal of the research is to assess how well MLLMs recognize cultural elements when presented with individuals from diverse ethnic backgrounds.

#### Competing Models
The study evaluated three MLLM models:
1. **GPT-4o**: Developed by OpenAI, it demonstrated the highest overall performance.
2. **GLM-4v-Plus**: Developed by ZhipuAI, it showed stable performance in recognizing various cultural markers.
3. **InternVL2.5**: Developed by OpenGVLab, it exhibited relatively lower accuracy but maintained consistency.

#### Test Data
The MIXCUBE dataset was constructed by collecting 33 original images from each culture and generating synthetic images by replacing each image with four different ethnic backgrounds (African, Caucasian, East Asian, South Asian), resulting in a total of 2,500 images. Each image was labeled with its cultural origin.

#### Metrics
The performance of the models was evaluated through two main tasks:
1. **Country Identification Accuracy**: The ability to identify the country of origin for a given cultural marker.
2. **Cultural Marker Identification Accuracy**: The ability to identify the name of food in the image.

Accuracy was used to quantify the models' performance in both tasks. Generally, the accuracy for original images was higher than for synthetic images, with a more significant performance drop observed in low-resource cultures (e.g., Azerbaijan, Myanmar).

#### Comparison
The results indicated that MLLMs achieved higher accuracy for high-resource cultures (e.g., South Korea, the UK, the US) while showing lower accuracy and greater variability in performance based on ethnic background for low-resource cultures. For instance, GPT-4o showed up to a 58% difference in accuracy between original and synthetic images in low-resource cultures. In contrast, high-resource cultures exhibited relatively minor accuracy differences based on ethnic backgrounds.

This study highlights the existence of biases in MLLMs' recognition of cultural elements based on ethnic backgrounds and emphasizes the need for more diverse cultural data to mitigate these biases.


<br/>
# 예제



이 논문에서는 다문화 대형 언어 모델(MLLMs)의 문화적 편향을 평가하기 위해 MIXCUBE라는 데이터셋을 소개합니다. 이 데이터셋은 5개 문화(아제르바이잔, 한국, 미얀마, 영국, 미국)와 3개 카테고리(음식, 축제, 의류)에 걸쳐 2,500개의 이미지를 포함하고 있습니다. 각 이미지는 원본 이미지와 함께, 인종을 변경한 4개의 합성 이미지로 구성되어 있습니다. 이 연구의 주요 목표는 MLLMs가 다양한 인종의 사람들과 함께 있는 문화적 요소를 얼마나 잘 인식하는지를 평가하는 것입니다.

#### 예시

1. **트레이닝 데이터**:
   - **원본 이미지**: 한국의 김치를 먹고 있는 아시아 여성의 사진.
   - **합성 이미지**: 같은 이미지에서 아시아 여성을 아프리카 남성으로 변경한 사진.
   - **라벨**: 원본 이미지의 경우 "김치, 한국"으로 라벨링, 합성 이미지의 경우 "김치, 한국"으로 라벨링.

2. **테스트 데이터**:
   - **원본 이미지**: 아제르바이잔의 전통 음식을 먹고 있는 아제르바이잔 남성의 사진.
   - **합성 이미지**: 같은 이미지에서 아제르바이잔 남성을 남아시아 남성으로 변경한 사진.
   - **라벨**: 원본 이미지의 경우 "아제르바이잔 음식", 합성 이미지의 경우 "아제르바이잔 음식"으로 라벨링.

3. **테스크**:
   - **국가 식별**: MLLM에게 주어진 이미지에서 음식의 출처 국가를 식별하도록 요청합니다. 예를 들어, "이 이미지의 음식은 어떤 나라와 가장 관련이 있습니까?"라는 질문을 통해 모델의 응답을 평가합니다.
   - **문화적 마커 식별**: MLLM에게 주어진 이미지에서 음식의 이름을 식별하도록 요청합니다. 예를 들어, "이 이미지의 음식 이름은 무엇입니까?"라는 질문을 통해 모델의 응답을 평가합니다.

이러한 방식으로, 연구자들은 MLLMs가 다양한 인종적 배경을 가진 사람들과 함께 있는 문화적 요소를 얼마나 잘 인식하는지를 평가하고, 저자원 문화와 고자원 문화 간의 성능 차이를 분석합니다.

---




This paper introduces a dataset called MIXCUBE to evaluate the cultural bias of multimodal large language models (MLLMs). The dataset consists of 2,500 images spanning five cultures (Azerbaijan, South Korea, Myanmar, the UK, and the US) and three categories (food, festivals, clothing). Each image is accompanied by four synthesized images where the ethnicity of the person is altered. The main goal of this research is to assess how well MLLMs recognize cultural elements when presented with people of different ethnic backgrounds.

#### Example

1. **Training Data**:
   - **Original Image**: A photo of an Asian woman eating kimchi.
   - **Synthesized Image**: The same image with the Asian woman replaced by an African man.
   - **Label**: For the original image, labeled as "Kimchi, South Korea"; for the synthesized image, also labeled as "Kimchi, South Korea".

2. **Test Data**:
   - **Original Image**: A photo of an Azerbaijani man eating traditional Azerbaijani food.
   - **Synthesized Image**: The same image with the Azerbaijani man replaced by a South Asian man.
   - **Label**: For the original image, labeled as "Azerbaijani food"; for the synthesized image, also labeled as "Azerbaijani food".

3. **Tasks**:
   - **Country Identification**: The MLLM is asked to identify the country of origin of the food in the given image. For example, a question like "Which country is the food in this image most associated with?" is posed to evaluate the model's response.
   - **Cultural Marker Identification**: The MLLM is asked to identify the name of the food in the given image. For example, a question like "What is the name of the food in this image?" is posed to evaluate the model's response.

Through this approach, the researchers assess how well MLLMs recognize cultural elements in the context of different ethnic backgrounds and analyze performance differences between low-resource and high-resource cultures.

<br/>
# 요약

이 연구에서는 MIXCUBE라는 데이터셋을 사용하여 다문화 대형 언어 모델(MLLMs)의 문화적 편향을 평가하였다. 실험 결과, MLLMs는 고자원 문화에서 높은 정확도를 보였으나 저자원 문화에서는 성능 저하가 두드러졌다. 예를 들어, 아제르바이잔과 미얀마의 경우, 인종을 변경했을 때 정확도가 40% 이상 감소하는 경향을 보였다.



This study evaluated the cultural bias of multimodal large language models (MLLMs) using a dataset called MIXCUBE. The results showed that MLLMs achieved higher accuracy in high-resource cultures, while significant performance drops were observed in low-resource cultures. For instance, in the cases of Azerbaijan and Myanmar, accuracy decreased by over 40% when the ethnicity was altered.

<br/>
# 기타



#### 다이어그램 및 피규어
1. **Figure 1**: 실험 예시로, 원본 이미지와 인종이 변경된 합성 이미지가 나열되어 있습니다. 이 피규어는 MLLM이 인종 변경에 따라 문화적 요소를 인식하는 데 어떤 영향을 받는지를 보여줍니다.

2. **Figure 2**: MIXCUBE 데이터셋의 이미지 합성 과정이 설명되어 있습니다. 원본 이미지에서 인종을 변경하여 생성된 이미지의 과정을 시각적으로 나타내고 있습니다.

3. **Figure 3**: 원본 이미지와 합성 이미지의 국가 식별 정확도를 비교한 그래프입니다. 합성 이미지의 정확도가 원본 이미지보다 평균 7.64% 낮다는 것을 보여줍니다.

4. **Figure 4**: 국가 식별 정확도의 차이를 나타내는 히트맵입니다. 합성된 인종이 원본 문화의 인구 통계와 유사할 때 정확도가 더 높다는 것을 강조합니다.

5. **Figure 5**: 문화적 마커 식별 정확도를 보여주는 그래프입니다. 합성 이미지에서 인종 변경이 정확도에 미치는 영향을 나타냅니다.

6. **Figure 6**: MIXCUBE 데이터셋의 전체 구성 파이프라인을 보여줍니다. 데이터 수집, 합성, 품질 보증 과정을 시각적으로 설명합니다.

#### 테이블
1. **Table 1**: 국가 ISO 코드와 문화 마커 카테고리의 약어를 정리한 표입니다. 각 문화와 카테고리에 대한 명확한 참조를 제공합니다.

2. **Table 2**: 데이터셋의 구성 요소를 나타내는 표입니다. 각 문화와 카테고리별로 수집된 이미지 수를 보여줍니다.

3. **Table 3**: 실험에 사용된 주요 하이퍼파라미터 값을 정리한 표입니다. 각 모델의 설정을 명확히 합니다.

4. **Table 7**: 국가 식별 정확도 결과를 보여주는 표입니다. 각 모델의 성능을 비교할 수 있습니다.

5. **Table 8**: 문화적 마커 식별 정확도 결과를 보여주는 표입니다. 각 모델의 성능을 비교할 수 있습니다.

#### 어펜딕스
- **Appendix A**: 데이터셋 구성의 세부 사항을 설명합니다. 이미지 수집 기준, 마스킹 절차, 레이블링 방법 등을 포함합니다.
- **Appendix B**: 이미지 합성을 위한 프롬프트와 MLLM 평가를 위한 프롬프트를 제공합니다.
- **Appendix C**: 실험 설정 및 모델에 대한 추가 정보를 제공합니다.




#### Diagrams and Figures
1. **Figure 1**: An example of the experiment showing original and synthesized images where the ethnicity is altered. This figure illustrates how MLLMs are affected by changes in ethnicity when recognizing cultural elements.

2. **Figure 2**: Depicts the image synthesis process for the MIXCUBE dataset. It visually represents the steps taken to generate images by altering the ethnicity from the original images.

3. **Figure 3**: A graph comparing the country identification accuracy between original and synthesized images. It shows that the accuracy of synthesized images is, on average, 7.64% lower than that of original images.

4. **Figure 4**: A heatmap illustrating the differences in country identification accuracy. It highlights that accuracy is higher when the synthesized ethnicity closely resembles the demographic of the original culture.

5. **Figure 5**: A graph showing the cultural marker identification accuracy. It indicates the impact of ethnicity changes on accuracy in synthesized images.

6. **Figure 6**: Visualizes the overall construction pipeline of the MIXCUBE dataset. It explains the processes of data collection, synthesis, and quality assurance.

#### Tables
1. **Table 1**: A table summarizing the ISO codes for countries and abbreviations for cultural marker categories. It provides a clear reference for each culture and category.

2. **Table 2**: A table showing the composition of the dataset. It lists the number of images collected for each culture and category.

3. **Table 3**: A table detailing the key hyperparameters used in the experiments. It clarifies the settings for each model.

4. **Table 7**: A table presenting the results of country identification accuracy. It allows for comparison of performance across models.

5. **Table 8**: A table showing the results of cultural marker identification accuracy. It facilitates comparison of performance across models.

#### Appendix
- **Appendix A**: Describes the details of dataset construction, including image collection criteria, masking procedures, and labeling methods.
- **Appendix B**: Provides prompts for image synthesis and evaluation of MLLMs.
- **Appendix C**: Offers additional information on experimental settings and models.

<br/>
# refer format:
### BibTeX 

```bibtex
@inproceedings{kim2025when,
  title={When to Eat Kimchi: Evaluating Cultural Bias of Multimodal Large Language Models in Cultural Mixture Contexts},
  author={Jun Seong Kim and Kyaw Ye Thu and Javad Ismayilzada and Junyeong Park and Eunsu Kim and Huzama Ahmad and Na Min An and James Thorne and Alice Oh},
  booktitle={Proceedings of the 3rd Workshop on Cross-Cultural Considerations in NLP (C3NLP 2025)},
  pages={143--154},
  year={2025},
  publisher={Association for Computational Linguistics},
  url={https://huggingface.co/datasets/kyawyethu/MixCuBe}
}
```

### 시카고 스타일

Jun Seong Kim, Kyaw Ye Thu, Javad Ismayilzada, Junyeong Park, Eunsu Kim, Huzama Ahmad, Na Min An, James Thorne, and Alice Oh. "When to Eat Kimchi: Evaluating Cultural Bias of Multimodal Large Language Models in Cultural Mixture Contexts." In *Proceedings of the 3rd Workshop on Cross-Cultural Considerations in NLP (C3NLP 2025)*, 143–154. Association for Computational Linguistics, 2025. https://huggingface.co/datasets/kyawyethu/MixCuBe.
