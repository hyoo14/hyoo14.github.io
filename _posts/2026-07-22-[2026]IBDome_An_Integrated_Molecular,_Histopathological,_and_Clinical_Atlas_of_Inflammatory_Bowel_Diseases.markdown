---
layout: post
title:  "[2026]IBDome: An Integrated Molecular, Histopathological, and Clinical Atlas of Inflammatory Bowel Diseases"
date:   2026-07-22 15:52:38 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 연구에서는 1002명의 염증성 장 질환(IBD) 환자와 비-IBD 대조군의 다중 오믹스 및 다중 모달 분석을 수행하여, 혈청 단백질 기반의 염증 활동 지표(IBD-IPSS)를 개발하고, 인공지능 모델을 통해 조직 이미지에서 병리학적 질병 활동 점수를 예측하였다.


짧은 요약(Abstract) :



IBDome은 염증성 장 질환(IBD)에 대한 통합된 분자, 조직병리학적, 임상 아틀라스를 제공합니다. 이 연구는 1002명의 임상적으로 주석이 달린 IBD 환자와 비-IBD 대조군을 대상으로 한 다중 오믹스 및 다중 모드 분석을 포함합니다. 정상 및 염증이 있는 장 조직의 전체 엑솜 및 RNA 시퀀싱, 혈청 단백질체학, H&E 염색된 조직 절편의 이미지에서의 조직병리학적 평가가 포함되었습니다. 연구 결과, 정상 및 염증 조직의 전사체 프로파일은 크론병과 궤양성 대장염에서 뚜렷한 부위별 염증 서명을 드러냈습니다. 혈청 단백질체학을 활용하여 장내 분자 염증을 반영하는 염증 단백질 중증도 서명이 개발되었습니다. 또한, 기초 모델 기반의 딥러닝이 조직병리학적 질병 활동 점수를 정확하게 예측하고 H&E 염색된 장 조직 이미지에서 크론병과 궤양성 대장염을 분류할 수 있음을 보여주었습니다. 이 통합적이고 공개적으로 이용 가능한 다중 오믹스 자원은 IBD 연구의 잠재력을 강조하며, 다중 오믹스와 고급 계산 접근 방식을 결합하여 IBD의 이해와 관리 개선에 기여할 수 있습니다.




IBDome provides an integrated molecular, histopathological, and clinical atlas of inflammatory bowel diseases (IBD). This study includes a multiomic and multimodal analysis of 1002 clinically annotated patients with IBD and non-IBD controls, incorporating whole-exome and RNA sequencing of normal and inflamed gut tissues, serum proteomics, and histopathological assessments from images of H&E-stained tissue sections. The results revealed distinct site-specific inflammatory signatures in Crohn's disease and ulcerative colitis through transcriptomic profiles of normal and inflamed tissues. Leveraging serum proteomics, an inflammatory protein severity signature reflecting underlying intestinal molecular inflammation was developed. Furthermore, foundation model-based deep learning accurately predicted histologic disease activity scores and enabled classification of Crohn's disease vs ulcerative colitis from images of H&E-stained intestinal tissue sections. This integrative, publicly available multiomics resource highlights the potential of combining multiomics and advanced computational approaches to improve understanding and management of IBD.


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



IBDome 연구에서는 다중 오믹스(multiomics) 및 다중 모달(multi-modal) 접근 방식을 통해 염증성 장 질환(IBD)의 포괄적인 아틀라스를 구축했습니다. 이 연구의 방법론은 다음과 같은 주요 요소로 구성됩니다.

1. **연구 설계 및 코호트**: 이 연구는 다기관, 다중 모달 분석으로 설계되었으며, 독일의 베를린과 에를랑겐에 위치한 2개의 주요 센터에서 IBD 환자와 비-IBD 대조군을 모집했습니다. 총 1002명의 환자 데이터가 수집되었으며, 이들은 임상, 내시경, 조직병리학, 분자 및 이미징 데이터를 포함합니다.

2. **임상 및 내시경 평가**: 표준화된 의료 질문지를 통해 환자의 기본 정보와 질병 활동 점수를 수집했습니다. UC(궤양성 대장염)와 CD(크론병)에 대해 각각 Partial Mayo Score와 Harvey-Bradshaw Index를 사용하여 임상 질병 활동을 기록했습니다. 내시경 질병 활동은 UCEIS(궤양성 대장염 내시경 중증도 지수)와 SES-CD(크론병 단순 내시경 점수)를 사용하여 평가되었습니다.

3. **조직병리학 및 질병 활동 점수**: 포르말린 고정된 장 조직 샘플을 H&E 염색하여 조직병리학적 평가를 수행했습니다. 조직병리학적 질병 활동 점수는 UC에 대해 수정된 Naini Cortina 점수, CD에 대해 수정된 Riley 점수를 사용하여 평가되었습니다.

4. **RNA 시퀀싱 및 전사체 분석**: 내시경 중 채취한 생검 또는 절제된 조직에서 RNA를 분리하고 RNA 시퀀싱을 통해 전사체 분석을 수행했습니다. 이를 통해 염증성 장 질환의 분자적 특징을 파악했습니다.

5. **단백질체 분석**: 혈청 샘플을 사용하여 Olink Target 96 염증 패널을 통해 단백질의 풍부함을 측정했습니다. 이 데이터는 질병 활동을 모니터링하는 데 사용되는 IBD-IPSS(염증성 단백질 중증도 지수)를 개발하는 데 기여했습니다.

6. **AI 기반 분석**: H&E 염색된 조직 이미지에서 병리학적 질병 활동 점수를 예측하기 위해 여러 기초 모델을 적용했습니다. 이 과정에서 주의 기반 다중 사례 학습 모델을 사용하여 이미지에서 중요한 특징을 추출하고, 이를 통해 질병의 하위 유형을 분류했습니다.

7. **데이터 통합 및 분석**: 수집된 모든 데이터를 통합하여 IBDome Explorer라는 웹 플랫폼을 통해 공개적으로 접근할 수 있도록 하였습니다. 이 플랫폼은 연구자들이 IBD의 생물학을 탐구하고, 데이터 분석 및 검증을 수행할 수 있도록 지원합니다.

이러한 방법론은 IBD의 복잡한 생물학적 메커니즘을 이해하고, 개인 맞춤형 치료 전략을 개발하는 데 중요한 기초 자료를 제공합니다.

---




The IBDome study constructed a comprehensive atlas of inflammatory bowel disease (IBD) using a multiomics and multimodal approach. The methodology of this study consists of the following key components:

1. **Study Design and Cohorts**: This study was designed as a multicenter, multimodal analysis, recruiting IBD patients and non-IBD controls from two major centers located in Berlin and Erlangen, Germany. A total of 1002 patient data were collected, encompassing clinical, endoscopic, histopathological, molecular, and imaging data.

2. **Clinical and Endoscopic Assessment**: A standardized medical questionnaire was implemented to collect basic patient information and disease activity scores. Clinical disease activity was recorded using the Partial Mayo Score for ulcerative colitis (UC) and the Harvey-Bradshaw Index for Crohn's disease (CD). Endoscopic disease activity was assessed using the Ulcerative Colitis Endoscopic Index of Severity (UCEIS) and the Simple Endoscopic Score for Crohn’s Disease (SES-CD).

3. **Histopathology and Disease Activity Scoring**: Formalin-fixed paraffin-embedded intestinal tissue samples were processed and stained with H&E for histopathological evaluation. Histologic disease activity scores were assessed using the modified Naini Cortina score for UC and the modified Riley score for CD.

4. **RNA Sequencing and Transcriptomic Analysis**: RNA was isolated from biopsies collected during endoscopy or from resected tissue, followed by RNA sequencing to perform transcriptomic analysis. This helped identify the molecular features of inflammatory bowel disease.

5. **Proteomic Profiling**: Serum samples were analyzed using the Olink Target 96 Inflammation panel to measure protein abundance. This data contributed to the development of the IBD-IPSS (Inflammatory Protein Severity Signature), which is used for monitoring disease activity.

6. **AI-Based Analysis**: Foundation models were applied to predict histologic disease activity scores directly from pathology images of H&E-stained tissues. An attention-based multiple instance learning model was utilized to extract important features from the images, enabling the classification of disease subtypes.

7. **Data Integration and Analysis**: All collected data were integrated and made publicly accessible through a web platform called IBDome Explorer. This platform supports researchers in exploring IBD biology and conducting data analysis and validation.

These methodologies provide crucial foundational data for understanding the complex biological mechanisms of IBD and developing personalized treatment strategies.


<br/>
# Results



이 연구에서는 인공지능(AI) 기반의 모델을 사용하여 염증성 장질환(IBD)의 조직학적 질병 활동 점수를 예측하는 성능을 평가했습니다. 연구에 사용된 데이터는 두 개의 주요 코호트에서 수집된 H&E 염색된 조직 이미지로 구성되어 있으며, 총 1212개의 이미지를 사용했습니다. 이 이미지는 수정된 Naini Cortina 점수와 수정된 Riley 점수에 따라 분류되었습니다.

#### 경쟁 모델
연구에서는 네 가지 다른 기초 모델(CHIEF, UNI2, Virchow2, H-optimus-1)을 사용하여 예측 성능을 비교했습니다. 각 모델은 5겹 교차 검증을 통해 훈련 및 내부 검증을 수행했습니다.

#### 테스트 데이터
테스트 데이터는 두 개의 코호트에서 수집된 H&E 이미지로 구성되어 있으며, Berlin 코호트에서 699개의 이미지(514개는 수정된 Naini Cortina 점수, 185개는 수정된 Riley 점수)와 Erlangen 코호트에서 556개의 이미지(472개는 수정된 Riley 점수, 84개는 수정된 Naini Cortina 점수)가 포함되었습니다.

#### 메트릭
모델의 성능은 실제 점수와 예측 점수 간의 피어슨 상관계수를 사용하여 평가되었습니다. Virchow2 모델은 수정된 Riley 점수 예측에서 r ≈ 0.933의 상관관계를 보였고, UNI2 모델은 수정된 Naini Cortina 점수 예측에서 r ≈ 0.801의 상관관계를 보였습니다.

#### 비교
Erlangen 코호트에 모델을 배포한 결과, 두 점수 모두에서 강력한 성능을 보였습니다. 또한 Innsbruck 코호트에서도 높은 일치를 보였으며, 다수의 전문가 합의 조직학적 질병 활동 점수와의 상관관계가 확인되었습니다. AI 기반 점수는 원래 점수와의 상관관계가 일치하거나 초과하는 성능을 보여주었으며, 이는 AI 기반 조직학적 점수가 질병 활동의 보다 표준화되고 객관적인 평가를 지원할 수 있음을 시사합니다.

이 연구의 결과는 AI 모델이 IBD의 조직학적 평가에서 신뢰할 수 있는 도구가 될 수 있음을 보여주며, 이는 임상에서의 활용 가능성을 높입니다.

---




In this study, the performance of artificial intelligence (AI)-based models in predicting histologic disease activity scores for inflammatory bowel disease (IBD) was evaluated. The data used in the study consisted of H&E stained tissue images collected from two main cohorts, totaling 1212 images. These images were categorized according to modified Naini Cortina scores and modified Riley scores.

#### Competing Models
The study utilized four different foundation models (CHIEF, UNI2, Virchow2, H-optimus-1) to compare predictive performance. Each model underwent 5-fold cross-validation for training and internal validation.

#### Test Data
The test data comprised H&E images collected from two cohorts, including 699 images from the Berlin cohort (514 images for modified Naini Cortina scores and 185 images for modified Riley scores) and 556 images from the Erlangen cohort (472 images for modified Riley scores and 84 images for modified Naini Cortina scores).

#### Metrics
The performance of the models was assessed based on the Pearson correlation coefficient between true scores and predicted scores. The Virchow2 model achieved a correlation of r ≈ 0.933 in predicting the modified Riley score, while the UNI2 model showed a correlation of r ≈ 0.801 for the modified Naini Cortina score.

#### Comparison
When deployed to the Erlangen cohort, the models demonstrated strong performance for both scores. Additionally, high agreement with multiobserver consensus histologic disease activity scores was observed in the Innsbruck cohort. The AI-based scores matched or even surpassed the original scores, suggesting that AI-based histologic scoring could help achieve a more standardized and objective assessment of disease activity, thereby improving objective disease monitoring in IBD.

The results of this study indicate that AI models can serve as reliable tools for histologic evaluation in IBD, enhancing their potential for clinical application.


<br/>
# 예제



이 연구에서는 인플라메이토리 장 질환(IBD)의 병리학적 이미지를 분석하기 위해 인공지능(AI) 기반의 딥러닝 모델을 사용했습니다. 이 모델은 H&E(헤마톡실린-에오신) 염색된 조직 슬라이드 이미지를 입력으로 받아, 해당 이미지에서 병리학적 질병 활동 점수를 예측하는 작업을 수행했습니다.

#### 데이터셋 구성
1. **트레이닝 데이터**: 
   - **입력**: H&E 염색된 조직 슬라이드 이미지 (예: 514개의 이미지가 베를린에서 수집됨, 185개의 이미지가 에를랑겐에서 수집됨)
   - **출력**: 각 이미지에 대한 수정된 나이니 코르티나 점수(Modified Naini Cortina Score) 또는 수정된 라일리 점수(Modified Riley Score). 예를 들어, 특정 이미지가 "3"이라는 점수를 받을 수 있습니다.

2. **테스트 데이터**:
   - **입력**: H&E 염색된 조직 슬라이드 이미지 (예: 556개의 이미지)
   - **출력**: 모델이 예측한 점수. 예를 들어, 모델이 특정 이미지를 분석한 결과 "2.5"라는 점수를 예측할 수 있습니다.

#### 작업 흐름
1. **데이터 수집**: 연구자들은 여러 병원에서 IBD 환자들의 조직 샘플을 수집하고, 이를 H&E 염색하여 슬라이드 이미지를 생성했습니다.
2. **모델 훈련**: 수집된 이미지를 사용하여 딥러닝 모델을 훈련시킵니다. 이 과정에서 모델은 입력 이미지와 해당 이미지의 실제 점수 간의 관계를 학습합니다.
3. **모델 평가**: 훈련된 모델은 테스트 데이터셋을 사용하여 성능을 평가합니다. 이때, 모델의 예측 점수와 실제 점수를 비교하여 정확도를 측정합니다.
4. **결과 해석**: 모델의 예측 결과를 바탕으로, 병리학적 활동의 정도를 평가하고, 이를 통해 IBD 환자의 상태를 모니터링하는 데 활용할 수 있습니다.

이러한 방식으로, 연구자들은 AI 모델이 병리학적 이미지를 분석하여 IBD의 질병 활동을 정확하게 예측할 수 있음을 입증했습니다.

---




In this study, an artificial intelligence (AI)-based deep learning model was utilized to analyze pathological images of inflammatory bowel disease (IBD). The model takes H&E (Hematoxylin and Eosin) stained tissue slide images as input and performs the task of predicting histologic disease activity scores.

#### Dataset Composition
1. **Training Data**: 
   - **Input**: H&E stained tissue slide images (e.g., 514 images collected from Berlin, 185 images from Erlangen)
   - **Output**: Histologic disease activity scores, such as the Modified Naini Cortina Score or Modified Riley Score. For example, a specific image might receive a score of "3".

2. **Test Data**:
   - **Input**: H&E stained tissue slide images (e.g., 556 images)
   - **Output**: Predicted scores by the model. For instance, the model might predict a score of "2.5" for a specific image.

#### Workflow
1. **Data Collection**: Researchers collected tissue samples from IBD patients at various hospitals, stained them with H&E, and created slide images.
2. **Model Training**: The collected images were used to train the deep learning model. During this process, the model learns the relationship between input images and their corresponding actual scores.
3. **Model Evaluation**: The trained model is evaluated using the test dataset. The model's predicted scores are compared to the actual scores to measure accuracy.
4. **Result Interpretation**: Based on the model's predictions, the degree of histologic activity is assessed, which can be used to monitor the condition of IBD patients.

Through this approach, the researchers demonstrated that the AI model could accurately predict disease activity in IBD by analyzing pathological images.

<br/>
# 요약


이 연구에서는 1002명의 염증성 장 질환(IBD) 환자와 비-IBD 대조군의 다중 오믹스 및 다중 모달 분석을 수행하여, 혈청 단백질 기반의 염증 활동 지표(IBD-IPSS)를 개발하고, 인공지능 모델을 통해 조직 이미지에서 병리학적 질병 활동 점수를 예측하였다. 결과적으로, IBD-IPSS는 기존의 내시경 점수와 강한 상관관계를 보였으며, AI 모델은 조직 이미지에서 높은 정확도로 IBD 아형을 분류할 수 있었다. 이 연구는 IBD의 생물학적 이해를 심화시키고, 개인 맞춤형 치료 전략 개발에 기여할 수 있는 중요한 자원으로 자리잡았다.

---

In this study, a multi-omic and multimodal analysis of 1002 patients with inflammatory bowel disease (IBD) and non-IBD controls was conducted to develop a serum protein-based inflammatory activity index (IBD-IPSS) and to predict histologic disease activity scores from tissue images using an artificial intelligence model. The results showed that IBD-IPSS strongly correlated with established endoscopic scores, and the AI model accurately classified IBD subtypes from histologic images. This research serves as a significant resource for deepening the biological understanding of IBD and contributing to the development of personalized treatment strategies.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: IBDome 아틀라스의 데이터셋과 샘플 수를 보여주는 다이어그램으로, 1002명의 환자에서 수집된 임상, 유전자, 단백질, 조직학적 데이터의 통합을 강조합니다. 이 아틀라스는 IBD의 면역병리학적 특성을 이해하는 데 중요한 자원입니다.
   - **Figure 2**: IBD-IPSS(Inflammatory Protein Severity Signature)의 개발을 보여주는 화산도입니다. 이 피규어는 염증이 있는 IBD 샘플과 비염증 샘플 간의 단백질 차이를 시각적으로 나타내며, 특정 단백질이 염증의 지표로 작용할 수 있음을 시사합니다.
   - **Figure 3**: 조직 질병 특이적 염증 유전자 서명을 보여주는 주성분 분석 결과입니다. 이 피규어는 CD와 UC 간의 유전자 발현 차이를 강조하며, 조직 유형이 염증의 변동성에 미치는 영향을 보여줍니다.
   - **Figure 4**: 염증 질병 위치에 따른 혈청 단백질 바이오마커를 식별하는 결과를 보여주는 화산도입니다. 이 피규어는 CD와 UC의 혈청 단백질 차이를 강조하며, 비침습적인 질병 모니터링의 가능성을 제시합니다.
   - **Figure 5**: 병리 이미지에서 조직학적 질병 활동 점수를 예측하는 AI 기반 분석 파이프라인을 보여줍니다. 이 피규어는 AI 모델이 병리학적 이미지를 통해 질병 활동을 정확하게 예측할 수 있음을 나타냅니다.
   - **Figure 6**: IBD 아형 분류를 위한 AI 모델의 성능을 보여주는 결과입니다. 이 피규어는 AI 모델이 CD와 UC를 효과적으로 구분할 수 있음을 강조합니다.

2. **테이블**
   - **Table 1**: 수정된 Naini Cortina 및 수정된 Riley 점수의 최대 조직병리학적 점수를 보여주는 표입니다. 이 표는 각 질병 유형 및 샘플링 방법에 따른 최대 점수를 정리하여, 조직학적 평가의 표준화를 지원합니다.

3. **어펜딕스**
   - 어펜딕스에는 연구에 사용된 방법론, 데이터 수집 및 분석 절차에 대한 자세한 설명이 포함되어 있습니다. 이는 연구의 재현성을 높이고, 다른 연구자들이 이 데이터를 활용할 수 있도록 돕습니다.





1. **Diagrams and Figures**
   - **Figure 1**: A diagram showing the datasets and sample sizes of the IBDome atlas, emphasizing the integration of clinical, genomic, proteomic, and histological data collected from 1002 patients. This atlas serves as a crucial resource for understanding the immunopathological characteristics of IBD.
   - **Figure 2**: A volcano plot illustrating the development of the IBD-IPSS (Inflammatory Protein Severity Signature). This figure visually represents the differences in proteins between inflamed IBD samples and non-inflamed samples, suggesting that specific proteins may act as indicators of inflammation.
   - **Figure 3**: Results from principal component analysis showing tissue disease-specific inflammatory gene signatures. This figure highlights the differences in gene expression between CD and UC, demonstrating the impact of tissue type on the variability of inflammation.
   - **Figure 4**: A volcano plot identifying serum protein biomarkers based on inflammatory disease localization. This figure emphasizes the differences in serum proteins between CD and UC, suggesting the potential for non-invasive disease monitoring.
   - **Figure 5**: An overview of the AI-based analysis pipeline for predicting histologic disease activity from pathology images. This figure indicates that AI models can accurately predict disease activity through pathological images.
   - **Figure 6**: Results showing the performance of AI models for classifying IBD subtypes. This figure highlights the ability of AI models to effectively distinguish between CD and UC.

2. **Tables**
   - **Table 1**: A table displaying the maximum histopathological scores for modified Naini Cortina and modified Riley scores. This table organizes the maximum scores by disease type and sampling method, supporting the standardization of histopathological assessments.

3. **Appendix**
   - The appendix includes detailed descriptions of the methodologies used in the study, data collection, and analysis procedures. This enhances the reproducibility of the research and assists other researchers in utilizing this data.

<br/>
# refer format:

### BibTeX format  

```bibtex
@article{Plattner2026,
  author = {Plattner, Christina Q. and Sturm, Gregor and Kühl, Anja A. and Atreya, Raja and Carollo, Sandro and Rieder, Dietmar and Gronauer, Raphael and Günther, Michael and Ormanns, Steffen and Manzl, Claudia and Wirtz, Stefan and Meneghetti, Asier Rabasco and Hegazy, Ahmed N. and Patankar, Jay V. and Carrero, Zunamys I. and Grabherr, Felix and Meyer, Moritz and Adolph, Timon E. and Tilg, Herbert and Neurath, Markus F. and Kather, Jakob Nikolas and Becker, Christoph and Siegmund, Britta and Trajanoski, Zlatko},
  title = {IBDome: An Integrated Molecular, Histopathological, and Clinical Atlas of Inflammatory Bowel Diseases},
  journal = {Gastroenterology},
  year = {2026},
  pages = {1--17},
  doi = {10.1053/j.gastro.2026.05.023},
  publisher = {Elsevier Inc. on behalf of American Gastroenterological Association Institute}
}
```

### Chicago style format  

Plattner, Christina Q., Gregor Sturm, Anja A. Kühl, Raja Atreya, Sandro Carollo, Dietmar Rieder, Raphael Gronauer, Michael Günther, Steffen Ormanns, Claudia Manzl, Stefan Wirtz, Asier Rabasco Meneghetti, Ahmed N. Hegazy, Jay V. Patankar, Zunamys I. Carrero, Felix Grabherr, Moritz Meyer, Timon E. Adolph, Herbert Tilg, Markus F. Neurath, Jakob Nikolas Kather, Christoph Becker, Britta Siegmund, and Zlatko Trajanoski. "IBDome: An Integrated Molecular, Histopathological, and Clinical Atlas of Inflammatory Bowel Diseases." *Gastroenterology* (2026): 1–17. https://doi.org/10.1053/j.gastro.2026.05.023.
