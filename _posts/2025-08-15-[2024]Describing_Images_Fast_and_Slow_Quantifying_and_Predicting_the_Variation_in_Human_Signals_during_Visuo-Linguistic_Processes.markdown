---
layout: post
title:  "[2024]Describing Images Fast and Slow: Quantifying and Predicting the Variation in Human Signals during Visuo-Linguistic Processes"
date:   2025-08-15 15:45:04 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

이 논문에서는 네덜란드어 이미지 설명 데이터셋과 동시 수집된 시선 추적 데이터를 사용하여 인간의 시각-언어적 신호의 변화를 정량화하고 예측하는 방법을 제안  

모델이 인간의 복잡한 자극에 대한 편향이 부족하다는 것을 시사




짧은 요약(Abstract) :

이 논문은 이미지의 특성과 인간이 이미지를 설명하는 방식 간의 복잡한 관계를 탐구합니다. 인간의 행동은 눈 움직임과 설명 시작 시점과 같은 신호에서 나타나는 다양한 변화를 통해 드러나며, 이러한 신호는 현재의 사전 훈련된 모델에서 거의 무시되고 있습니다. 저자들은 네덜란드어 이미지 설명과 동시에 수집된 눈 추적 데이터를 사용하여 이러한 신호의 변화를 분석하고, 이미지의 특성이 이러한 변화를 부분적으로 유발한다고 가정합니다. 연구 결과, 사전 훈련된 모델이 이러한 변화를 약한 정도로 포착할 수 있음을 보여주며, 이는 모델이 인간의 복잡한 자극에 대한 편향이 부족하다는 것을 시사합니다.


This paper explores the intricate relationship between the properties of an image and how humans behave while describing the image. This behavior shows ample variation, as manifested in human signals such as eye movements and when humans start to describe the image. Despite the value of such signals of visuo-linguistic variation, they are virtually disregarded in the training of current pretrained models. The authors use a corpus of Dutch image descriptions with concurrently collected eye-tracking data to investigate the nature of the variation in visuo-linguistic signals and hypothesize that this variation partly stems from the properties of the images. The results indicate that pretrained models capture such variation to a weak degree, suggesting that the models lack biases about what makes a stimulus complex for humans.


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


이 논문에서는 이미지 설명과 관련된 인간의 시각-언어적 신호의 변화를 정량화하고 예측하기 위해 다양한 방법론을 사용합니다. 연구의 주요 목표는 이미지의 특성이 인간의 언어적 반응에 미치는 영향을 이해하고, 이를 통해 사전 훈련된 비전 인코더가 이러한 변화를 얼마나 잘 캡처할 수 있는지를 평가하는 것입니다.

1. **데이터셋**: 연구에 사용된 데이터셋은 '네덜란드 이미지 설명 및 시선 추적 코퍼스(DIDEC)'입니다. 이 코퍼스는 참가자들이 실제 장면을 묘사하는 동안 수집된 시선 추적 데이터와 음성 데이터를 포함하고 있습니다. 총 307개의 이미지에 대해 4604개의 설명이 수집되었습니다.

2. **신호의 변화를 정량화하는 방법**:
   - **발화 시작 시간(Speech Onsets)**: 각 설명의 발화 시작 시간을 측정하여 평균 및 표준편차를 계산합니다. 이를 통해 이미지에 따라 발화 시작 시간이 어떻게 달라지는지를 분석합니다.
   - **시작점(Starting Points)**: 각 설명에서 첫 번째로 언급된 명사를 분석하여, 이미지에 따라 얼마나 다양한 시작점이 사용되는지를 측정합니다.
   - **설명의 변이(Variation in Descriptions)**: BLEU 점수를 사용하여 설명 간의 유사성을 측정합니다. 이 점수는 생성된 문장과 참조 문장 간의 n-그램 기반 정밀도를 계산하여 언어적 변이를 정량화합니다.
   - **시선 변이(Variation in Gaze)**: 시선 추적 데이터를 기반으로 시선의 변이를 측정합니다. 각 이미지에 대해 시선의 경로를 비교하고, 이 경로 간의 유사성을 기반으로 변이를 정량화합니다.

3. **상관관계 분석**: 각 신호의 변이 간의 상관관계를 분석하여, 이미지의 특성이 이러한 변이에 미치는 영향을 평가합니다. Spearman 상관계수를 사용하여 신호 간의 관계를 분석합니다.

4. **유사성 기반 예측(Similarity-based Prediction)**: 사전 훈련된 비전 인코더(예: CLIP, ViT)를 사용하여 이미지의 특성을 인코딩하고, 이 인코딩된 특성을 기반으로 인간 신호의 변이를 예측합니다. k-최근접 이웃 알고리즘을 사용하여 가장 유사한 이미지를 찾고, 이들 이미지의 변이를 가중 평균하여 목표 이미지의 변이를 예측합니다.

5. **모델 평가**: CLIP와 ViT 모델을 사용하여 예측의 정확성을 평가하고, RNDCLIP(무작위 초기화된 CLIP 모델)와 비교하여 사전 훈련된 모델이 인간 신호의 변이를 얼마나 잘 캡처하는지를 분석합니다.

이러한 방법론을 통해 연구자들은 이미지의 특성이 인간의 언어적 반응에 미치는 영향을 정량화하고, 사전 훈련된 모델이 이러한 변이를 얼마나 잘 반영하는지를 평가할 수 있었습니다.

---



In this paper, various methodologies are employed to quantify and predict the variation in human visuo-linguistic signals related to image descriptions. The primary goal of the study is to understand how the properties of images influence human linguistic responses and to evaluate how well pretrained vision encoders can capture these variations.

1. **Dataset**: The dataset used in the study is the 'Dutch Image Description and Eye-tracking Corpus (DIDEC)'. This corpus includes eye-tracking data and speech data collected while participants described real-life scenes. A total of 307 images were described, resulting in 4604 descriptions.

2. **Methods for Quantifying Signal Variation**:
   - **Speech Onsets**: The onset times of each description are measured, and the mean and standard deviation are calculated. This analysis helps to understand how the onset times vary depending on the image.
   - **Starting Points**: The first nouns mentioned in each description are analyzed to measure how diverse the starting points are for different images.
   - **Variation in Descriptions**: BLEU scores are used to measure the similarity between descriptions. This score calculates n-gram-based precision between generated sentences and reference sentences to quantify linguistic variation.
   - **Variation in Gaze**: The eye-tracking data is used to measure the variation in gaze. The gaze paths for each image are compared, and the variation is quantified based on the similarity of these paths.

3. **Correlation Analysis**: The correlations between the variations of each signal are analyzed to assess the impact of image properties on these variations. Spearman's correlation coefficient is used to analyze the relationships between signals.

4. **Similarity-based Prediction**: Pretrained vision encoders (e.g., CLIP, ViT) are used to encode the properties of images, and this encoded information is used to predict the variation in human signals. A k-nearest neighbors algorithm is employed to find the most similar images, and the variation of these images is averaged with weights to predict the variation for the target image.

5. **Model Evaluation**: The predictions' accuracy is evaluated using the CLIP and ViT models, and comparisons are made with RNDCLIP (a randomly initialized version of CLIP) to analyze how well pretrained models capture the variation in human signals.

Through these methodologies, the researchers were able to quantify the influence of image properties on human linguistic responses and assess how well pretrained models reflect these variations.


<br/>
# Results


이 연구에서는 이미지 설명과 관련된 인간 신호의 변화를 정량화하고 예측하기 위해 Dutch Image Description and Eye-tracking Corpus(DIDEC)를 사용했습니다. 연구의 주요 결과는 다음과 같습니다.

1. **변화의 정량화**: 
   - **발화 시작 시간**: 평균 발화 시작 시간은 3.42초였으며, 참가자 간의 변동성이 있었습니다. 발화 시작 시간의 평균은 1.69초에서 7.07초 사이로 나타났습니다.
   - **시작점의 다양성**: 각 이미지에 대해 고유한 시작점의 수를 계산한 결과, 평균 6.45개의 시작점이 나타났습니다. 이는 이미지에 따라 다르게 나타났습니다.
   - **전체 설명의 다양성**: BLEU-2 메트릭을 사용하여 설명의 다양성을 측정한 결과, 평균 BLEU-2 점수는 0.53으로 나타났습니다. 이는 설명 간의 유사성을 나타내며, 점수가 높을수록 설명이 유사하다는 것을 의미합니다.
   - **시선의 다양성**: 시선의 변화를 측정하기 위해 제안된 거리 메트릭을 사용하여 평균 24.00의 시선 다양성 점수를 얻었습니다.

2. **상관관계 분석**: 
   - 발화 시작 시간과 설명의 BLEU-2 점수 간에는 유의미한 음의 상관관계(ρ=-0.391)가 발견되었습니다. 이는 설명이 더 유사할수록 발화가 더 빨리 시작된다는 것을 의미합니다.
   - 시선의 다양성과 발화 시작 시간 간에는 유의미한 양의 상관관계(ρ=0.455)가 발견되었습니다. 이는 시선의 변화가 클수록 발화 시작 시간이 길어진다는 것을 나타냅니다.

3. **유사성 기반 예측**: 
   - CLIP 및 ViT와 같은 사전 훈련된 비전 인코더를 사용하여 이미지 특성을 기반으로 인간 신호의 변화를 예측했습니다. CLIP을 사용한 경우, 설명의 변화 예측에서 0.3380의 상관계수를 얻었고, ViT는 0.3135의 상관계수를 기록했습니다. RNDCLIP은 유의미한 상관관계를 보이지 않았습니다.
   - 발화 시작 시간 예측에서도 CLIP을 사용했을 때 0.2981의 상관계수를 얻었으며, ViT는 0.2428의 상관계수를 기록했습니다. RNDCLIP은 유의미한 상관관계를 보이지 않았습니다.
   - 시작점 예측에서는 CLIP이 13.00%의 정확도를 보였고, ViT는 26.47%의 정확도를 기록했습니다.

이 연구는 이미지 설명과 관련된 인간의 인지 과정을 이해하는 데 기여하며, 사전 훈련된 모델이 이러한 과정을 얼마나 잘 캡처할 수 있는지를 보여줍니다.

---



This study utilized the Dutch Image Description and Eye-tracking Corpus (DIDEC) to quantify and predict the variation in human signals related to image descriptions. The main findings of the study are as follows:

1. **Quantification of Variation**:
   - **Speech Onsets**: The average speech onset time was 3.42 seconds, with variability among participants. The mean onset time ranged from 1.69 seconds to 7.07 seconds.
   - **Variation in Starting Points**: The number of unique starting points per image was calculated, resulting in an average of 6.45 starting points. This varied depending on the image.
   - **Variation in Full Descriptions**: Using the BLEU-2 metric, the linguistic variation in descriptions was measured, yielding an average BLEU-2 score of 0.53. A higher score indicates greater similarity between descriptions.
   - **Variation in Gaze**: A proposed distance metric for measuring gaze variation yielded an average gaze variation score of 24.00.

2. **Correlation Analysis**:
   - A significant negative correlation (ρ=-0.391) was found between speech onset times and BLEU-2 scores of descriptions, indicating that more similar descriptions lead to earlier speech onset.
   - A significant positive correlation (ρ=0.455) was found between gaze variation and speech onset times, suggesting that greater variation in gaze is associated with longer speech onset times.

3. **Similarity-based Prediction**:
   - Pretrained vision encoders such as CLIP and ViT were used to predict changes in human signals based on image features. CLIP achieved a correlation coefficient of 0.3380 for predicting variation in descriptions, while ViT recorded 0.3135. RNDCLIP showed no significant correlation.
   - For predicting speech onset times, CLIP yielded a correlation coefficient of 0.2981, while ViT achieved 0.2428. RNDCLIP did not show significant correlation.
   - In predicting starting points, CLIP achieved an accuracy of 13.00%, while ViT recorded an accuracy of 26.47%.

This study contributes to understanding the cognitive processes involved in image descriptions and demonstrates how well pretrained models can capture these processes.


<br/>
# 예제


이 논문에서는 이미지 설명과 관련된 인간의 신호 변화를 정량화하고 예측하는 연구를 수행했습니다. 연구에 사용된 데이터셋은 네덜란드어로 된 이미지 설명과 동시에 수집된 시선 추적 데이터로 구성된 DIDEC(Dutch Image Description and Eye-tracking Corpus)입니다. 이 데이터셋은 307개의 실제 이미지에 대한 4604개의 설명을 포함하고 있으며, 각 설명은 45명의 참가자에 의해 생성되었습니다.

#### 트레이닝 데이터와 테스트 데이터

1. **트레이닝 데이터**:
   - **입력**: 이미지의 CLIP(Contrastive Language-Image Pretraining) 인코더를 통해 생성된 이미지 표현 벡터.
   - **출력**: 각 이미지에 대한 평균 발화 시작 시간, 시작 단어의 변동성, 설명의 언어적 변동성, 시선의 변동성.
   - **예시**: 특정 이미지에 대해, 참가자들이 설명을 시작하는 평균 시간(예: 3.42초), 첫 번째 단어의 변동성(예: 6.45개의 고유 시작 단어), 설명의 BLEU-2 점수(예: 0.53), 시선의 변동성 점수(예: 24.00).

2. **테스트 데이터**:
   - **입력**: 트레이닝 데이터와 유사한 방식으로 인코딩된 이미지 표현 벡터.
   - **출력**: 예측된 평균 발화 시작 시간, 시작 단어, 설명의 변동성, 시선의 변동성.
   - **예시**: 테스트 이미지에 대해 예측된 평균 발화 시작 시간(예: 3.50초), 가장 자주 사용된 첫 번째 단어(예: "man"), 설명의 변동성 점수(예: 0.40), 시선의 변동성 점수(예: 22.00).

#### 구체적인 작업(Task)

- **작업 1**: 이미지에 대한 설명을 생성하는 과정에서 발화 시작 시간의 변동성을 예측합니다. 이 작업은 이미지의 복잡성과 관련된 인지적 요구를 반영합니다.
- **작업 2**: 참가자들이 설명을 시작할 때 사용하는 첫 번째 단어의 변동성을 예측합니다. 이는 이미지의 시각적 특성과 관련이 있습니다.
- **작업 3**: 이미지 설명의 언어적 변동성을 BLEU 점수를 통해 측정합니다. 이 점수는 생성된 설명과 참조 설명 간의 유사성을 기반으로 합니다.
- **작업 4**: 시선 추적 데이터를 통해 시선의 변동성을 측정하고 예측합니다. 이는 참가자들이 이미지에서 주목하는 영역의 다양성을 반영합니다.

이러한 작업들은 이미지의 시각적 특성이 인간의 언어적 반응에 미치는 영향을 이해하는 데 기여하며, 모델이 이러한 변동성을 얼마나 잘 캡처할 수 있는지를 평가합니다.

---



This paper conducts research on quantifying and predicting variations in human signals related to image descriptions. The dataset used in the study is the DIDEC (Dutch Image Description and Eye-tracking Corpus), which consists of Dutch image descriptions collected concurrently with eye-tracking data. This dataset includes 4604 descriptions for 307 real-life images, generated by 45 participants.

#### Training Data and Test Data

1. **Training Data**:
   - **Input**: Image representation vectors generated by the CLIP (Contrastive Language-Image Pretraining) encoder.
   - **Output**: Mean speech onset time, variation in starting words, linguistic variation of descriptions, and gaze variation for each image.
   - **Example**: For a specific image, the average time participants start describing (e.g., 3.42 seconds), the variation in first words (e.g., 6.45 unique starting words), the BLEU-2 score of descriptions (e.g., 0.53), and the gaze variation score (e.g., 24.00).

2. **Test Data**:
   - **Input**: Image representation vectors encoded in a similar manner to the training data.
   - **Output**: Predicted mean speech onset time, starting word, variation in descriptions, and variation in gaze.
   - **Example**: For a test image, the predicted mean speech onset time (e.g., 3.50 seconds), the most frequently used first word (e.g., "man"), the variation score of descriptions (e.g., 0.40), and the gaze variation score (e.g., 22.00).

#### Specific Tasks

- **Task 1**: Predict the variation in speech onset times when generating descriptions for images. This task reflects the cognitive demands associated with the complexity of the images.
- **Task 2**: Predict the variation in the first word used by participants when starting their descriptions. This is related to the visual characteristics of the images.
- **Task 3**: Measure the linguistic variation of image descriptions using BLEU scores, which are based on the similarity between generated descriptions and reference descriptions.
- **Task 4**: Measure and predict gaze variation using eye-tracking data, reflecting the diversity of areas participants focus on in the images.

These tasks contribute to understanding how the visual properties of images influence human linguistic responses and evaluate how well models can capture such variations.

<br/>
# 요약
이 논문에서는 네덜란드어 이미지 설명 데이터셋과 동시 수집된 시선 추적 데이터를 사용하여 인간의 시각-언어적 신호의 변화를 정량화하고 예측하는 방법을 제안하였다. 결과적으로, 이미지의 특성이 언어적 설명의 변동성과 시선의 변동성에 영향을 미친다는 것을 발견하였으며, 사전 훈련된 비전 인코더가 이러한 변화를 약한 정도로 포착할 수 있음을 보여주었다. 예를 들어, 이미지의 복잡성이 높을수록 설명 시작 시점이 늦어지고, 다양한 설명이 생성되는 경향이 있었다.

---

This paper proposes a method to quantify and predict the variation in human visuo-linguistic signals using a Dutch image description dataset and concurrently collected eye-tracking data. The results reveal that the properties of images influence the variability in linguistic descriptions and gaze, and pretrained vision encoders can capture this variation to a weak extent. For instance, images with higher complexity tend to elicit later speech onset and a greater diversity of descriptions.

<br/>
# 기타


1. **다이어그램 및 피규어**
   - **Figure 1**: 이 그림은 데이터셋에서 최소 및 최대 평균 발화 시작 시간을 가진 이미지를 보여줍니다. 최대 발화 시작 시간을 가진 이미지는 설명의 첫 번째 명사에서 가장 높은 변동성을 유도합니다. 이는 이미지의 상대적 인지 복잡성을 나타내며, 복잡한 이미지가 더 긴 발화 시작 시간을 유도할 수 있음을 시사합니다.
   - **Figure 2**: 이 그림은 특정 이미지에 대한 변동성 점수를 시각적으로 나타내며, 각 설명의 예시와 함께 제공됩니다. 이는 이미지 설명의 다양성을 정량화하는 데 유용합니다.
   - **Figure 3**: 이 그림은 평균 발화 시작 시간, 시작점의 변동성, 전체 설명의 BLEU-2 기반 변동성, 시선의 변동성 간의 스피어만 상관 계수를 보여줍니다. 모든 상관관계는 유의미하며, 이는 이미지의 특성이 언어적 및 시각적 변동성에 영향을 미친다는 것을 나타냅니다.

2. **테이블**
   - **Table 1**: 이 테이블은 유사성 기반 접근 방식을 사용하여 설명의 변동성을 예측한 결과를 보여줍니다. CLIP과 ViT 모델이 약한 상관관계를 보였으며, RNDCLIP은 유의미한 상관관계를 보이지 않았습니다. 이는 CLIP이 언어와 시각 데이터 간의 정렬 목표에 따라 더 나은 성능을 보일 수 있음을 시사합니다.
   - **Table 2**: 평균 발화 시작 시간을 예측한 결과를 보여줍니다. CLIP을 사용할 때 예측이 약한 상관관계를 보였으며, 이는 이미지의 복잡성이 발화 시작 시간에 영향을 미친다는 것을 나타냅니다.
   - **Table 3**: 설명의 첫 번째 명사를 예측한 결과를 보여줍니다. CLIP과 ViT 모델이 무작위 예측보다 더 나은 성능을 보였으며, 이는 이미지의 시각적 특성이 언어적 선택에 영향을 미친다는 것을 나타냅니다.
   - **Table 4**: 시선의 변동성을 예측한 결과를 보여줍니다. CLIP 모델이 가장 높은 상관관계를 보였으며, 이는 이미지 특성이 시선 변동성과 관련이 있음을 시사합니다.

3. **어펜딕스**
   - **Appendix A**: 데이터 전처리 과정에서 사용된 spaCy의 성능을 보여줍니다. 대형 모델이 가장 적은 오류를 보였으며, 이는 데이터의 신뢰성을 높이는 데 기여했습니다.
   - **Appendix B**: 발화 시작 시간의 분포를 보여주는 히스토그램을 포함하고 있으며, 비정규 분포를 나타냅니다. 이는 이미지에 따라 발화 시작 시간이 다양하게 나타날 수 있음을 시사합니다.
   - **Appendix C**: 참가자 기반의 상관관계 분석 결과를 보여줍니다. 모든 참가자가 발화 시작 시간이 낮은 이미지에서 더 적은 언어적 변동성을 보였음을 나타냅니다.
   - **Appendix D**: BERTje 기반의 설명 변동성 분석 결과를 보여줍니다. BLEU-2 기반 변동성과의 상관관계가 약한 부정적 상관관계를 보였습니다.
   - **Appendix E**: BLEU-2 기반 변동성과 BERTje 기반 변동성을 결합한 결과를 보여줍니다. 이 조합은 설명의 변동성을 정량화하는 데 유용할 수 있습니다.





1. **Diagrams and Figures**
   - **Figure 1**: This figure shows images with the minimum and maximum mean speech onset times in the dataset. The image with the maximum onset also elicits the highest variation in the first nouns of the descriptions, indicating the relative cognitive complexity of the images.
   - **Figure 2**: This figure visually represents the variation scores for a specific image, along with examples of its descriptions. It is useful for quantifying the diversity of image descriptions.
   - **Figure 3**: This figure displays Spearman's correlation coefficients between mean onsets, variation in starting points, BLEU-2 based variation in full descriptions, and variation in gaze. All correlations are significant, suggesting that image features influence linguistic and visual variability.

2. **Tables**
   - **Table 1**: This table shows the results of predicting variation in descriptions using a similarity-based approach. Both CLIP and ViT models exhibited weak correlations, while RNDCLIP showed no meaningful correlation, indicating that CLIP may perform better due to its alignment with language and visual data.
   - **Table 2**: This table presents results for predicting mean speech onsets. Predictions using CLIP showed weak correlations, suggesting that image complexity affects speech onset times.
   - **Table 3**: This table shows results for predicting the first uttered nouns in descriptions. Both CLIP and ViT outperformed random predictions, indicating that visual features of images influence linguistic choices.
   - **Table 4**: This table presents results for predicting gaze variation. The CLIP model showed the highest correlation, suggesting a link between image features and gaze variability.

3. **Appendices**
   - **Appendix A**: This appendix shows the performance of spaCy in the data preprocessing steps. The large model had the fewest errors, contributing to the reliability of the data.
   - **Appendix B**: This appendix includes histograms showing the distribution of speech onset times, indicating a non-normal distribution. This suggests variability in speech onset times depending on the images.
   - **Appendix C**: This appendix presents participant-based correlation analysis results, indicating that all participants tended to start describing images earlier when those images elicited less linguistic variation.
   - **Appendix D**: This appendix shows results of variation analysis based on BERTje, revealing a weak negative correlation with BLEU-2 based variation.
   - **Appendix E**: This appendix presents results combining BLEU-2 and BERTje based variation scores, which may be useful for quantifying description variability.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{takmaz2024describing,
  author    = {Ece Takmaz and Sandro Pezzelle and Raquel Fernández},
  title     = {Describing Images Fast and Slow: Quantifying and Predicting the Variation in Human Signals during Visuo-Linguistic Processes},
  booktitle = {Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics, Volume 1: Long Papers},
  pages     = {2072--2087},
  year      = {2024},
  month     = {March},
  publisher = {Association for Computational Linguistics},
  address   = {Dublin, Ireland},
  url       = {https://github.com/ecekt/visuolinguistic_signal_variation}
}
```

### 시카고 스타일

Takmaz, Ece, Sandro Pezzelle, and Raquel Fernández. "Describing Images Fast and Slow: Quantifying and Predicting the Variation in Human Signals during Visuo-Linguistic Processes." In *Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics, Volume 1: Long Papers*, 2072–2087. Dublin, Ireland: Association for Computational Linguistics, 2024. https://github.com/ecekt/visuolinguistic_signal_variation.
