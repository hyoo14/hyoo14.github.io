---
layout: post
title:  "[2025]Forensic Self-Descriptions Are All You Need for Zero-Shot Detection, Open-Set Source Attribution, and Clustering of AI-generated Images"
date:   2026-02-03 18:57:36 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 AI 생성 이미지의 탐지 및 출처 귀속을 위한 새로운 접근법인 '포렌식 자기 설명'을 제안합니다.


짧은 요약(Abstract) :

이 논문의 초록에서는 고급 AI 기반 도구가 생성한 사실적인 이미지가 포렌식 탐지 및 출처 추적에 상당한 도전을 제기하고 있다는 점을 강조합니다. 기존의 방법들은 훈련 중에 알려진 소스에 특정한 특징에 의존하기 때문에 보지 못한 생성기에 대해 일반화하는 데 실패하는 경우가 많습니다. 이를 해결하기 위해, 저자들은 이미지 생성 과정에서 고유한 미세 구조를 명시적으로 모델링하는 새로운 접근 방식을 제안합니다. 이 방법은 실제 이미지만을 사용하여 다양한 예측 필터를 학습하고, 이러한 필터를 통해 미세 구조의 잔여물을 추출합니다. 이 잔여물들을 여러 스케일에서 공동 모델링하여 각 이미지에 대한 고유한 포렌식 자기 설명을 생성합니다. 이 자기 설명을 통해 저자들은 합성 이미지의 제로샷 탐지, 개방형 출처 추적, 그리고 사전 지식 없이 이미지 소스의 클러스터링을 수행할 수 있습니다. 실험 결과는 이 방법이 경쟁 기술에 비해 우수한 정확성과 적응성을 달성함을 보여줍니다.



The abstract of this paper emphasizes that the emergence of advanced AI-based tools for generating realistic images poses significant challenges for forensic detection and source attribution. Traditional methods often fail to generalize to unseen generators due to their reliance on features specific to known sources during training. To address this issue, the authors propose a novel approach that explicitly models forensic microstructures—subtle patterns unique to the image creation process. Using only real images in a self-supervised manner, they learn a set of diverse predictive filters to extract residuals that capture different aspects of these microstructures. By jointly modeling these residuals across multiple scales, they obtain a unique forensic self-description for each image. This self-description enables them to perform zero-shot detection of synthetic images, open-set source attribution, and clustering of image sources without prior knowledge. Extensive experiments demonstrate that their method achieves superior accuracy and adaptability compared to competing techniques.


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



이 논문에서 제안하는 방법은 인공지능(AI) 생성 이미지의 탐지 및 출처 귀속을 위한 새로운 접근 방식을 제시합니다. 이 방법은 '법의학적 자기 설명(forensic self-description)'이라는 개념을 기반으로 하며, 이는 이미지의 생성 과정에서 나타나는 미세한 패턴인 '법의학적 미세구조(forensic microstructures)'를 모델링하여 이루어집니다.

#### 1. 법의학적 미세구조 추출
이 방법의 첫 번째 단계는 실제 이미지를 사용하여 장면 내용을 근사화하는 다양한 예측 필터를 학습하는 것입니다. 이 필터들은 이미지의 픽셀 주변 이웃을 기반으로 하여 각 픽셀의 값을 예측합니다. 이 과정에서 생성된 잔여물(residuals)은 법의학적 미세구조를 포함하고 있으며, 이러한 잔여물은 이미지의 고유한 특성을 포착합니다.

#### 2. 다중 스케일 모델링
학습된 필터를 사용하여 단일 이미지에서 여러 개의 잔여물을 추출한 후, 이 잔여물들을 다중 스케일에서 공동 모델링합니다. 이를 통해 각 이미지에 대한 고유한 법의학적 자기 설명을 생성합니다. 이 자기 설명은 이미지의 내재적 법의학적 특성을 효과적으로 캡슐화하여, 다양한 작업을 수행할 수 있게 합니다.

#### 3. 제로샷 탐지
법의학적 자기 설명을 활용하여 제로샷 탐지를 수행합니다. 이는 특정 생성 모델에 대한 사전 노출 없이 이미지가 실제인지 AI 생성인지 판단하는 작업입니다. 이를 위해, 실제 이미지에서 얻은 자기 설명의 분포를 모델링하고, 새로운 이미지의 자기 설명이 이 분포에서 벗어나는 정도를 평가하여 탐지를 수행합니다.

#### 4. 오픈셋 출처 귀속
법의학적 자기 설명을 사용하여 이미지의 출처를 식별하는 오픈셋 출처 귀속 작업을 수행합니다. 이는 알려진 출처 집합 중에서 이미지를 가장 가능성이 높은 출처에 귀속시키거나, 출처가 알려지지 않은 경우 이를 식별하는 작업입니다. 각 출처의 자기 설명 분포를 모델링하고, 새로운 이미지의 자기 설명이 가장 높은 가능성을 가진 출처에 귀속됩니다.

#### 5. 비지도 클러스터링
마지막으로, 법의학적 자기 설명을 사용하여 비지도 클러스터링을 수행합니다. 이는 레이블이 없는 이미지 데이터셋에서 공통 출처를 식별하는 작업으로, K-평균 클러스터링 알고리즘을 사용하여 이미지들을 유사성에 따라 그룹화합니다.

이러한 방법론은 AI 생성 이미지의 탐지 및 출처 귀속에서 높은 정확도와 적응성을 보여주며, 다양한 생성 모델에 대해 강력한 성능을 발휘합니다.

---



The method proposed in this paper presents a novel approach for the detection and source attribution of AI-generated images. This approach is based on the concept of 'forensic self-descriptions,' which models the subtle patterns known as 'forensic microstructures' that emerge during the image creation process.

#### 1. Forensic Microstructure Extraction
The first step of this method involves learning a diverse set of predictive filters using only real images to approximate scene content. These filters predict the value of each pixel based on its surrounding neighbors. The residuals generated in this process contain forensic microstructures, capturing the unique characteristics of the image.

#### 2. Multi-Scale Modeling
After extracting multiple residuals from a single image using the learned filters, these residuals are jointly modeled across multiple scales. This results in a unique forensic self-description for each image. This self-description effectively encapsulates the intrinsic forensic properties of the image, enabling the performance of various tasks.

#### 3. Zero-Shot Detection
Using forensic self-descriptions, zero-shot detection is performed. This refers to the task of determining whether an image is real or AI-generated without prior exposure to specific generative models. To achieve this, the distribution of self-descriptions obtained from real images is modeled, and the likelihood of a new image's self-description deviating from this distribution is evaluated for detection.

#### 4. Open-Set Source Attribution
Forensic self-descriptions are utilized to perform open-set source attribution, which involves identifying the source of an image among a set of known sources or determining if the image originates from an unknown source. The distribution of self-descriptions for each known source is modeled, and the self-description of a new image is attributed to the source with the highest likelihood.

#### 5. Unsupervised Clustering
Finally, unsupervised clustering is performed using forensic self-descriptions. This task involves identifying common sources in an unlabeled image dataset, employing a K-means clustering algorithm to group images based on the similarity of their descriptions.

This methodology demonstrates high accuracy and adaptability in detecting and attributing AI-generated images, showcasing robust performance across various generative models.


<br/>
# Results



이 논문에서는 AI 생성 이미지의 탐지 및 출처 귀속을 위한 새로운 접근 방식을 제안하고, 이를 통해 제안된 방법의 성능을 기존의 경쟁 모델들과 비교합니다. 연구의 주요 결과는 다음과 같습니다.

1. **경쟁 모델**: 제안된 방법은 CNNDet, PatchFor, LGrad, UFD, DE-FAKE, Aeroblade, ZED, NPR 등 여러 기존의 탐지 방법과 비교되었습니다. 이들 모델은 주로 감독 학습 기반의 방법으로, 특정 생성 모델에 대한 훈련을 통해 성능을 발휘합니다.

2. **테스트 데이터**: 실험은 COCO2017, ImageNet-1k, ImageNet-22k, MIDB와 같은 다양한 실제 이미지 데이터셋과 ProGAN, StyleGAN, GigaGAN, Stable Diffusion, DALLE 등 여러 생성 모델에서 생성된 합성 이미지 데이터셋을 사용하여 수행되었습니다. 이 데이터셋은 다양한 생성 모델을 포함하여, 제안된 방법의 일반화 능력을 평가하는 데 중요한 역할을 했습니다.

3. **메트릭**: 성능 평가는 주로 AUC(Area Under the ROC Curve)로 측정되었습니다. AUC는 탐지 성능을 정량적으로 평가하는 데 유용한 지표로, 0.5는 무작위 추측을 의미하고 1.0은 완벽한 탐지를 의미합니다.

4. **비교 결과**: 제안된 방법은 모든 데이터셋에서 평균 AUC 0.960을 기록하며, 경쟁 모델들보다 우수한 성능을 보였습니다. 특히, NPR 모델과 비교했을 때, 제안된 방법은 0.892의 최악의 경우 AUC를 기록하며, 이는 다른 모델들보다 현저히 높은 수치입니다. ZED와 DE-FAKE는 특정 생성 모델에 대해 성능이 저조한 반면, 제안된 방법은 다양한 생성 모델에 대해 일관된 성능을 유지했습니다.

5. **오픈 세트 출처 귀속**: 오픈 세트 출처 귀속 실험에서도 제안된 방법은 0.933의 AU-CRR(Area Under the Correct Rejection Rate Curve)와 0.913의 AU-OSCR(Area Under the Open Set Classification Rate Curve)를 기록하며, 기존의 방법들보다 높은 성능을 보였습니다. 이는 제안된 방법이 알려지지 않은 출처의 이미지를 효과적으로 탐지하고 귀속할 수 있음을 나타냅니다.

6. **비지도 클러스터링**: 비지도 클러스터링 실험에서도 제안된 방법은 높은 정확도와 순도, NMI(Normalized Mutual Information)를 기록하며, 다른 방법들보다 우수한 성능을 보였습니다. 이는 제안된 방법이 이미지의 출처를 효과적으로 클러스터링할 수 있음을 보여줍니다.

이러한 결과들은 제안된 방법이 AI 생성 이미지 탐지 및 출처 귀속에 있어 강력하고 유연한 도구임을 입증합니다.

---




This paper proposes a novel approach for the detection and source attribution of AI-generated images and compares the performance of this method against existing competitive models. The main results of the study are as follows:

1. **Competing Models**: The proposed method was compared with several existing detection methods, including CNNDet, PatchFor, LGrad, UFD, DE-FAKE, Aeroblade, ZED, and NPR. These models primarily rely on supervised learning techniques, performing well by training on specific generative models.

2. **Test Data**: The experiments were conducted using various real image datasets such as COCO2017, ImageNet-1k, ImageNet-22k, and MIDB, along with synthetic image datasets generated from multiple models like ProGAN, StyleGAN, GigaGAN, Stable Diffusion, and DALLE. This diverse dataset played a crucial role in evaluating the generalization capability of the proposed method.

3. **Metrics**: Performance evaluation was primarily measured using AUC (Area Under the ROC Curve). AUC is a useful metric for quantitatively assessing detection performance, where 0.5 indicates random guessing and 1.0 indicates perfect detection.

4. **Comparison Results**: The proposed method achieved an average AUC of 0.960 across all datasets, outperforming the competing models. Notably, it recorded a worst-case AUC of 0.892 compared to NPR, which is significantly higher than other models. While ZED and DE-FAKE struggled with specific generative models, the proposed method maintained consistent performance across various generators.

5. **Open-Set Source Attribution**: In the open-set source attribution experiments, the proposed method achieved an AU-CRR (Area Under the Correct Rejection Rate Curve) of 0.933 and an AU-OSCR (Area Under the Open Set Classification Rate Curve) of 0.913, demonstrating superior performance compared to existing methods. This indicates that the proposed method effectively detects and attributes images from unknown sources.

6. **Unsupervised Clustering**: In the unsupervised clustering experiments, the proposed method also recorded high accuracy, purity, and NMI (Normalized Mutual Information), outperforming other methods. This shows that the proposed method can effectively cluster images based on their sources.

These results demonstrate that the proposed method is a powerful and flexible tool for detecting and attributing AI-generated images.


<br/>
# 예제



이 논문에서는 AI 생성 이미지의 탐지 및 출처 귀속을 위한 새로운 접근 방식을 제안합니다. 이 방법은 "포렌식 자기 설명(forensic self-description)"을 사용하여 이미지의 고유한 포렌식 마이크로구조를 모델링합니다. 이 과정에서 사용되는 데이터는 실제 이미지로, 이 이미지를 통해 다양한 예시를 설정할 수 있습니다.

#### 트레이닝 데이터와 테스트 데이터

1. **트레이닝 데이터**:
   - **입력**: 실제 이미지 데이터셋 (예: COCO2017, ImageNet-1k, ImageNet-22k, MIDB 등)
   - **출력**: 각 이미지에 대한 포렌식 자기 설명 (forensic self-description) 생성

   이 과정에서, 연구자들은 K개의 선형 예측 필터를 학습하여 이미지의 장면 내용을 근사하고, 이를 통해 포렌식 마이크로구조를 포함하는 잔여(residual)를 추출합니다. 이 잔여는 이미지의 고유한 특성을 나타내며, 이를 통해 포렌식 자기 설명이 생성됩니다.

2. **테스트 데이터**:
   - **입력**: 새로운 AI 생성 이미지 (예: ProGAN, StyleGAN3, Stable Diffusion 등)
   - **출력**: 이미지가 실제인지 AI 생성인지의 여부, 출처 귀속, 클러스터링 결과

   테스트 과정에서는 Gaussian Mixture Model (GMM)을 사용하여 트레이닝 데이터에서 학습한 포렌식 자기 설명의 분포를 모델링합니다. 이후, 새로운 이미지의 포렌식 자기 설명이 이 분포와 얼마나 일치하는지를 평가하여, 이미지가 실제인지 AI 생성인지 판단합니다. 또한, 이미지의 출처를 귀속시키고, 클러스터링을 통해 유사한 출처의 이미지를 그룹화합니다.

#### 구체적인 작업 예시

- **제로샷 탐지 (Zero-Shot Detection)**: 
  - **작업**: 새로운 AI 생성 이미지가 주어졌을 때, 이 이미지가 실제인지 AI 생성인지 판단합니다.
  - **입력**: 새로운 AI 생성 이미지
  - **출력**: 이미지가 실제인지 AI 생성인지의 확률

- **오픈셋 출처 귀속 (Open-Set Source Attribution)**:
  - **작업**: 주어진 이미지의 출처가 알려진 출처 중 하나인지, 아니면 알려지지 않은 출처인지 판단합니다.
  - **입력**: 새로운 이미지
  - **출력**: 이미지의 출처 (알려진 출처 또는 '알 수 없음')

- **비지도 클러스터링 (Unsupervised Clustering)**:
  - **작업**: 주어진 이미지 데이터셋에서 유사한 출처의 이미지를 그룹화합니다.
  - **입력**: 여러 이미지
  - **출력**: 각 이미지의 클러스터 레이블

이러한 작업을 통해, 연구자들은 AI 생성 이미지의 탐지 및 출처 귀속의 정확성을 높이고, 새로운 생성 모델에 대한 적응력을 향상시킬 수 있습니다.

---




This paper proposes a novel approach for detecting and attributing the source of AI-generated images using "forensic self-descriptions" to model the unique forensic microstructures of images. The data used in this process consists of real images, which can be structured into various examples.

#### Training Data and Test Data

1. **Training Data**:
   - **Input**: Real image datasets (e.g., COCO2017, ImageNet-1k, ImageNet-22k, MIDB, etc.)
   - **Output**: Forensic self-descriptions generated for each image

   In this process, researchers learn a set of K linear predictive filters to approximate the scene content of the images, extracting residuals that contain the forensic microstructures. These residuals represent the unique characteristics of the images, leading to the creation of forensic self-descriptions.

2. **Test Data**:
   - **Input**: New AI-generated images (e.g., ProGAN, StyleGAN3, Stable Diffusion, etc.)
   - **Output**: Whether the image is real or AI-generated, source attribution, clustering results

   During testing, a Gaussian Mixture Model (GMM) is used to model the distribution of forensic self-descriptions learned from the training data. The likelihood of the new image's forensic self-description being real is evaluated against this distribution to determine if the image is real or AI-generated. Additionally, the source of the image is attributed, and clustering is performed to group similar sources.

#### Specific Task Examples

- **Zero-Shot Detection**:
  - **Task**: Determine whether a given new AI-generated image is real or synthetic.
  - **Input**: New AI-generated image
  - **Output**: Probability of the image being real or synthetic

- **Open-Set Source Attribution**:
  - **Task**: Identify whether the source of a given image is one of the known sources or an unknown source.
  - **Input**: New image
  - **Output**: Source of the image (known source or 'unknown')

- **Unsupervised Clustering**:
  - **Task**: Group similar sources in a given image dataset.
  - **Input**: Multiple images
  - **Output**: Cluster labels for each image

Through these tasks, researchers can enhance the accuracy of detecting AI-generated images and attributing their sources, improving adaptability to new generative models.

<br/>
# 요약


이 논문에서는 AI 생성 이미지의 탐지 및 출처 귀속을 위한 새로운 접근법인 '포렌식 자기 설명'을 제안합니다. 이 방법은 실제 이미지를 사용하여 포렌식 마이크로구조를 모델링하고, 이를 통해 제로샷 탐지, 오픈셋 출처 귀속 및 클러스터링을 수행합니다. 실험 결과, 제안된 방법은 기존 기술들보다 높은 정확도와 적응성을 보여주었습니다.

---

This paper proposes a novel approach called 'forensic self-descriptions' for detecting and attributing AI-generated images. The method models forensic microstructures using real images, enabling zero-shot detection, open-set source attribution, and clustering. Experimental results demonstrate that the proposed method achieves higher accuracy and adaptability compared to existing techniques.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: 이 그림은 제안된 방법의 개요를 보여줍니다. 각 이미지에서 포렌식 마이크로구조의 자기 설명을 추출하고 이를 통해 제로샷 탐지, 오픈셋 소스 귀속, 클러스터링을 수행하는 과정을 시각적으로 설명합니다. 이 그림은 제안된 방법이 어떻게 다양한 작업을 수행할 수 있는지를 명확하게 보여줍니다.
   - **Figure 2**: 실제 이미지와 합성 이미지의 시각화를 통해 데이터셋의 다양성을 보여줍니다. 이는 연구에서 사용된 다양한 생성 모델의 결과를 비교하는 데 유용합니다.
   - **Figure 4**: 포렌식 자기 설명 필터의 평균 파워 스펙트럼을 시각화하여, 실제 이미지와 합성 이미지 간의 차이를 강조합니다. 이 그림은 각 생성 모델이 고유한 스펙트럼 특성을 가지고 있음을 보여주며, 이는 탐지 성능에 중요한 역할을 합니다.
   - **Figure 5**: t-SNE 플롯을 통해 실제 이미지와 합성 이미지 간의 분포를 시각적으로 나타냅니다. 이 그림은 각 생성 모델이 서로 다른 클러스터를 형성하여, 포렌식 자기 설명이 효과적으로 마이크로구조를 캡처하고 있음을 보여줍니다.

2. **테이블**
   - **Table 1**: 제로샷 합성 이미지 탐지 성능을 비교한 결과를 보여줍니다. 제안된 방법이 다른 기존 방법들보다 높은 AUC(Area Under the Curve) 점수를 기록하여, 제로샷 탐지에서의 우수성을 입증합니다.
   - **Table 2**: 최악의 경우 제로샷 탐지 성능을 비교한 결과로, 제안된 방법이 다른 방법들에 비해 일관되게 높은 성능을 보임을 나타냅니다. 이는 포렌식 자기 설명이 다양한 생성 모델에 대해 강력한 탐지 능력을 제공함을 시사합니다.
   - **Table 3**: 오픈셋 소스 귀속 성능 비교 결과를 보여줍니다. 제안된 방법이 다른 최신 방법들보다 높은 정확도와 AU-CRR, AU-OSCR을 기록하여, 소스 귀속에서의 우수성을 입증합니다.
   - **Table 4**: 클러스터링 성능 비교 결과로, 제안된 방법이 모든 메트릭에서 가장 높은 성능을 기록하여, 포렌식 자기 설명이 효과적인 클러스터링을 가능하게 함을 보여줍니다.
   - **Table 5**: 제안된 방법의 다양한 설계 선택이 제로샷 탐지 성능에 미치는 영향을 분석한 결과로, 포렌식 자기 설명의 효과성을 강조합니다.

3. **어펜딕스**
   - 어펜딕스에는 실험에 사용된 데이터셋, 구현 세부사항, 추가 실험 결과 등이 포함되어 있습니다. 이는 연구의 재현성을 높이고, 제안된 방법의 신뢰성을 강화하는 데 기여합니다.

---

### Results and Insights from Other Components (Diagrams, Figures, Tables, Appendices)

1. **Diagrams and Figures**
   - **Figure 1**: This figure provides an overview of the proposed method. It visually explains the process of extracting the forensic microstructure's self-description from each image and how it enables zero-shot detection, open-set source attribution, and clustering. This figure clearly illustrates how the proposed method can perform various tasks.
   - **Figure 2**: It visualizes real and synthetic images to demonstrate the diversity of the datasets. This is useful for comparing the results of various generative models used in the study.
   - **Figure 4**: This figure visualizes the average power spectrum of the forensic self-description filters, highlighting the differences between real and synthetic images. It shows that each generative model has unique spectral characteristics, which play a crucial role in detection performance.
   - **Figure 5**: The t-SNE plot visually represents the distribution between real and synthetic images. This figure shows that each generative model forms distinct clusters, indicating that the forensic self-descriptions effectively capture the microstructures.

2. **Tables**
   - **Table 1**: It presents the performance comparison of zero-shot synthetic image detection. The proposed method achieves a higher AUC (Area Under the Curve) score than other existing methods, demonstrating its superiority in zero-shot detection.
   - **Table 2**: This table compares the worst-case performance of zero-shot detection, showing that the proposed method consistently outperforms others. This suggests that forensic self-descriptions provide strong detection capabilities across various generative models.
   - **Table 3**: It shows the performance comparison for open-set source attribution. The proposed method achieves higher accuracy, AU-CRR, and AU-OSCR than other state-of-the-art methods, proving its effectiveness in source attribution.
   - **Table 4**: This table compares clustering performance, indicating that the proposed method achieves the highest performance across all metrics, demonstrating that forensic self-descriptions enable effective clustering.
   - **Table 5**: It analyzes the impact of various design choices on zero-shot detection performance, emphasizing the effectiveness of forensic self-descriptions.

3. **Appendices**
   - The appendices include details about the datasets used in the experiments, implementation specifics, and additional experimental results. This enhances the reproducibility of the research and strengthens the reliability of the proposed method.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{nguyen2025forensic,
  title={Forensic Self-Descriptions Are All You Need for Zero-Shot Detection, Open-Set Source Attribution, and Clustering of AI-generated Images},
  author={Nguyen, Tai D. and Azizpour, Aref and Stamm, Matthew C.},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025},
  organization={Computer Vision Foundation},
}
```

### 시카고 스타일

Nguyen, Tai D., Aref Azizpour, and Matthew C. Stamm. 2025. "Forensic Self-Descriptions Are All You Need for Zero-Shot Detection, Open-Set Source Attribution, and Clustering of AI-generated Images." In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). USA: Computer Vision Foundation. 




<br/>
# 질문


* t-SNE에서 분리가 잘 보인다고 했는데(t-SNE는 시각화 트릭일 수도 있음), 원공간에서의 분리(예: inter/intra distance)로도 설명 가능한가?  
(“보기 좋다”를 “정량 근거”로 바꾸게 만드는 질문.)  


* 최신 closed-source 모델 (예: GPT-4o image, DALL·E 최신, Imagen 최신, nano banana) 같은 것들은 왜 없냐?  



* partial generation (inpainting, editing)에 대한 한계   

Your benchmark focuses on fully generated images. How does the method behave when only part of an image is synthetic (e.g., inpainting, object replacement, local edits)?  

이건 실세계 forensic에서 가장 중요한 질문임:   

fake image ≠ fully synthetic  
 
대부분 hybrid   


* 8. Why worst-case IN-22k fails?


The worst-case AUC on IN-22k is substantially lower (0.714). Is this due to resolution, content diversity, or preprocessing differences? Does this indicate sensitivity to natural image statistics?



* Complementarity with VLMs

Have you explored combining forensic self-descriptions with vision-language model embeddings? Do the two capture orthogonal signals (microstructure vs semantics), and can an ensemble improve worst-case detection?




