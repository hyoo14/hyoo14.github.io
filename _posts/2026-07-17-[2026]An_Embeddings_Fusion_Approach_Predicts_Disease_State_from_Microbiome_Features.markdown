---
layout: post
title:  "[2026]An Embeddings Fusion Approach Predicts Disease State from Microbiome Features"
date:   2026-07-17 21:10:44 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 연구에서는 13,534개의 공공 메타게놈 샘플을 사용하여 미생물 군집의 시각적 표현을 기반으로 한 융합 모델을 개발하였으며, 이 모델은 질병 상태, 질병 유형 및 지리적 위치를 예측하는 데 높은 정확도를 보였습니다.

microbiome map은 여기서는 그냥 갖다 쓴거임.. 참고로 재스퍼툴로 생성되는데 이거는 23년 논문..
재스퍼툴은 긴 abundance 벡터를 Hilbert curve를 이용해 2차원 이미지로 접어 넣는 것. 딥러닝 임베딩이나 이미지 생성 모델은 아님  

힐버트 커브는 미생물 카테고리들을 1차원 순서대로 따라가면서, 정사각형 안의 가까운 칸들에 배치하는 길.. 각 칸의 위치는 카테고리별로 고정되고, abundance는 그 칸의 밝기나 색 진하기만 바꿔.  

결론적으로 여기서는 이 맵을 통해 더 나은 성능 보였다 이말..  


짧은 요약(Abstract) :


이 연구는 미생물군집의 특성을 기반으로 질병 상태를 예측하는 새로운 접근 방식을 제안합니다. 연구자들은 고차원 데이터를 처리하는 데 효과적인 딥 뉴럴 네트워크를 활용하여, NCBI의 분류 체계와 미생물 조성을 이미지로 인코딩하여 임베딩을 생성합니다. 이 방법은 질병 상태, 유형 및 지리적 위치와 같은 다양한 요인을 포착할 수 있는 임베딩을 생성합니다. 연구 결과, 13,534개의 공공 인간 메타게놈을 분석한 결과, 건강한 상태와 질병 상태를 구분하는 평균 분류 성능이 84%에 달하며, 질병 유형에 대해서는 87%, 신체 부위에 대해서는 99%, 지리적 위치에 대해서는 88%의 정확도를 기록했습니다. 이 연구는 미생물 샘플에서 파생된 메타게놈을 효율적으로 맥락화할 수 있는 임베딩 접근법의 유용성을 강조합니다.



This research introduces a novel approach to predicting disease states based on microbiome features. The researchers utilize deep neural networks, which are effective for processing high-dimensional data, to generate embeddings by encoding NCBI's taxonomic tree and microbial compositions as images. This method enables the creation of embeddings that capture various factors such as disease status, type, and geographical location. The results show that, when profiling 13,534 public human metagenomes, the model achieves an average classification performance of 84% in distinguishing healthy from disease conditions, 87% for disease types, 99% for body sites, and 88% for geographical locations. This work highlights the utility of an embeddings approach for efficiently contextualizing metagenomes derived from microbiome samples.


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



이 연구에서는 미생물군집의 풍부도 프로파일을 기반으로 질병 상태를 예측하기 위해 새로운 기계 학습 모델을 개발했습니다. 이 모델은 "Microbiome Maps"라는 시각적 표현을 사용하여 미생물의 조성과 질병 상태, 유형, 지리적 위치 등의 정보를 인코딩합니다. 모델의 주요 구성 요소는 다음과 같습니다.

1. **데이터 수집 및 전처리**: 연구팀은 13,534개의 공공 메타게놈 데이터를 수집하였으며, 이 데이터는 85개의 연구, 24개의 질병 유형, 35개 국가, 31,756개의 미생물 종을 포함합니다. 데이터는 NCBI의 뉴클레오타이드 데이터베이스를 사용하여 프로파일링되었으며, 각 샘플의 미생물 조성을 나타내는 풍부도 프로파일이 생성되었습니다. 이 프로파일은 CLR(Centered Log-Ratio) 변환을 통해 정규화되었습니다.

2. **Microbiome Maps 생성**: 각 풍부도 프로파일은 Jasper 도구를 사용하여 이미지로 변환됩니다. 이 이미지에서 각 픽셀은 특정 미생물을 나타내며, 색상은 해당 미생물의 상대적 풍부도를 나타냅니다. 이러한 시각적 표현은 미생물의 조성 패턴을 해석하는 데 유용합니다.

3. **모델 아키텍처**: 연구팀은 CNN(Convolutional Neural Network)과 ViT(Vision Transformer)를 결합한 퓨전 모델을 설계했습니다. CNN은 지역적 패턴을 학습하는 데 강점을 가지며, ViT는 전역적 맥락을 모델링하는 데 효과적입니다. 이 두 모델은 각각의 특징을 추출한 후, 최종적으로 결합하여 통합된 특징 벡터를 생성합니다.

4. **훈련 및 평가**: 모델은 다중 레이블 분류 작업을 통해 훈련되었습니다. 훈련 과정에서 모델은 질병 상태, 질병 유형, 신체 부위, 지리적 위치를 동시에 예측할 수 있도록 설계되었습니다. 모델의 성능은 AUC(Area Under the Curve) 메트릭을 사용하여 평가되었으며, 훈련된 모델은 높은 정확도로 질병 상태를 분류할 수 있음을 보여주었습니다.

5. **임베딩 생성**: 퓨전 모델은 각 미생물군집의 풍부도 프로파일에 대한 임베딩을 생성합니다. 이 임베딩은 질병 상태, 질병 유형, 지리적 위치 등의 정보를 포함하며, 새로운 샘플에 대한 임베딩을 생성하고 기존 클러스터에 적합하게 조정할 수 있습니다.

이 연구는 미생물군집 데이터를 활용하여 질병 상태를 예측하는 데 있어 컴퓨터 비전 기술을 적용한 최초의 사례로, 향후 다양한 데이터 모달리티와 통합하여 예측 및 특성화 모델을 개발하는 데 기초가 될 수 있습니다.

---



In this study, a novel machine learning model was developed to predict disease states based on microbiome abundance profiles. The model utilizes visual representations known as "Microbiome Maps" to encode information about microbial composition, disease status, type, and geographical location. The key components of the model are as follows:

1. **Data Collection and Preprocessing**: The research team collected 13,534 public metagenomic datasets, which included 85 studies, 24 disease types, 35 countries, and 31,756 microbial species. The data was profiled using NCBI's nucleotide database, generating abundance profiles that represent the microbial composition of each sample. These profiles were normalized using a Centered Log-Ratio (CLR) transformation.

2. **Microbiome Maps Generation**: Each abundance profile was converted into an image using the Jasper tool. In these images, each pixel represents a specific microbe, and the color indicates the relative abundance of that microbe. This visual representation is useful for interpreting microbial compositional patterns.

3. **Model Architecture**: The research team designed a fusion model that combines Convolutional Neural Networks (CNN) and Vision Transformers (ViT). CNNs excel at learning local patterns, while ViTs are effective at modeling global context. These two models extract features independently and then combine them to create a unified feature vector.

4. **Training and Evaluation**: The model was trained using a multi-label classification approach. During training, the model was designed to predict disease status, disease type, body site, and geographical location simultaneously. The model's performance was evaluated using the Area Under the Curve (AUC) metric, demonstrating high accuracy in classifying disease states.

5. **Embedding Generation**: The fusion model generates embeddings for each microbiome abundance profile. These embeddings contain information about disease status, disease type, and geographical location, allowing for the creation of embeddings for new samples that can be quickly fitted to existing clusters.

This study represents the first application of computer vision techniques to predict disease states from microbiome data, providing a foundation for future models that integrate various data modalities for prediction and characterization via the human gut microbiome.


<br/>
# Results



이 연구에서는 13,534개의 공공 메타게놈 샘플을 사용하여 질병 상태를 예측하기 위한 새로운 머신러닝 모델을 개발했습니다. 이 모델은 비주얼 표현을 활용하여 미생물 군집의 풍부함을 나타내는 "Microbiome Maps"를 생성하고, 이를 통해 다양한 질병 상태, 질병 유형, 신체 부위 및 지리적 위치를 분류하는 데 성공했습니다.

#### 경쟁 모델
연구팀은 기존의 여러 공개 모델을 평가하여 성능을 비교했습니다. 이 모델들은 TorchVision에서 제공하는 CNN 및 ViT 모델로, ImageNet 데이터셋으로 사전 훈련된 모델들이었습니다. 그러나 이들 모델은 미생물 맵의 특성과 맞지 않아 낮은 성능을 보였습니다. 예를 들어, CNN 모델은 질병 상태 분류에서 평균 AUC(Area Under the Curve) 0.78을 기록했으며, ViT 모델은 0.78을 기록했습니다. 반면, 연구팀이 개발한 퓨전 모델은 훈련 가능한 버전에서 0.85의 AUC를 달성했습니다.

#### 테스트 데이터
모델은 10%의 테스트 데이터(홀드아웃 세트)를 사용하여 평가되었습니다. 이 데이터는 1,350개의 샘플로 구성되어 있으며, 다양한 질병 상태와 유형을 포함하고 있습니다. 연구팀은 이 데이터를 통해 모델의 일반화 능력을 평가했습니다.

#### 메트릭
모델의 성능은 여러 메트릭을 통해 평가되었습니다. 주요 메트릭으로는 AUC, 정확도, F1 점수, 해밍 손실(Hamming Loss), Jaccard 점수 등이 있습니다. 퓨전 모델은 다중 레이블 분류 작업에서 가장 높은 성능을 보였으며, 훈련 가능한 퓨전 모델은 0.965의 마이크로 정밀도(micro precision)와 0.801의 마이크로 재현율(micro recall)을 기록했습니다. 이는 모델이 클래스 불균형을 잘 처리하고 있다는 것을 나타냅니다.

#### 비교
퓨전 모델은 기존의 CNN 및 ViT 모델보다 전반적으로 우수한 성능을 보였으며, 특히 다중 레이블 분류 작업에서 가장 높은 정확도를 기록했습니다. 연구팀은 퓨전 모델이 지역적 및 전역적 패턴을 모두 학습할 수 있도록 CNN과 ViT를 결합하여 성능을 극대화했다고 설명했습니다. 또한, 모델은 새로운 샘플에 대한 임베딩을 생성하고 기존 클러스터에 적합하게 조정할 수 있는 능력을 보여주었습니다.

이러한 결과는 미생물 군집의 복잡한 데이터에서 질병 상태를 예측하는 데 있어 퓨전 모델이 효과적임을 입증하며, 향후 임상 환경에서의 활용 가능성을 제시합니다.

---




In this study, a novel machine learning model was developed to predict disease states using 13,534 public metagenomic samples. The model successfully utilized visual representations to create "Microbiome Maps" that classify various disease states, disease types, body sites, and geographical locations.

#### Competing Models
The research team evaluated several existing public models to compare performance. These models included CNN and ViT architectures provided by TorchVision, which were pretrained on the ImageNet dataset. However, these models performed poorly due to a fundamental domain mismatch with the characteristics of microbiome maps. For instance, the CNN model achieved an average AUC (Area Under the Curve) of 0.78 for disease status classification, while the ViT model also recorded 0.78. In contrast, the fusion model developed by the research team achieved an AUC of 0.85 in its trainable version.

#### Test Data
The model was evaluated using 10% of the test data (holdout set), consisting of 1,350 samples that included various disease states and types. The research team used this data to assess the model's generalization capability.

#### Metrics
The model's performance was evaluated using several metrics, including AUC, accuracy, F1 score, Hamming loss, and Jaccard score. The fusion model demonstrated the highest performance in multi-label classification tasks, with the trainable fusion model achieving a micro precision of 0.965 and a micro recall of 0.801. This indicates that the model effectively handles class imbalance.

#### Comparison
The fusion model outperformed the existing CNN and ViT models overall, particularly in multi-label classification tasks where it recorded the highest accuracy. The research team explained that the fusion model maximized performance by combining CNN and ViT to learn both local and global patterns. Additionally, the model demonstrated the ability to generate embeddings for new samples and fit them into existing clusters.

These results demonstrate the effectiveness of the fusion model in predicting disease states from complex microbiome data and suggest its potential for future applications in clinical settings.


<br/>
# 예제



이 논문에서는 미생물군집의 특성을 기반으로 질병 상태를 예측하기 위한 새로운 기계 학습 모델을 제안합니다. 이 모델은 "Microbiome Maps"라는 시각적 표현을 사용하여 미생물의 조성을 이미지로 변환하고, 이를 통해 임베딩을 생성합니다. 이 임베딩은 질병 상태, 질병 유형, 신체 부위 및 지리적 위치와 같은 여러 요인을 인코딩합니다.

#### 트레이닝 데이터와 테스트 데이터

1. **트레이닝 데이터**:
   - **데이터 수**: 13,534개의 공공 미생물군집 메타게놈 샘플
   - **연구 수**: 85개의 연구에서 수집된 데이터
   - **질병 유형**: 24가지 질병 유형 (예: 자가면역 질환, 암, 심혈관 질환 등)
   - **신체 부위**: 4개의 신체 부위 (예: 장, 구강 등)
   - **지리적 위치**: 35개 국가에서 수집된 데이터
   - **입력**: 각 샘플의 미생물 조성을 나타내는 이미지 (Microbiome Map)
   - **출력**: 각 샘플의 질병 상태 (건강/질병), 질병 유형, 신체 부위, 지리적 위치

2. **테스트 데이터**:
   - **데이터 수**: 1,350개의 샘플 (10%의 홀드아웃 세트)
   - **입력**: 테스트 샘플의 Microbiome Map 이미지
   - **출력**: 예측된 질병 상태, 질병 유형, 신체 부위, 지리적 위치

#### 구체적인 테스크

- **이진 분류**: 건강한 샘플과 질병 샘플을 구분하는 작업
- **다중 클래스 분류**: 24가지 질병 유형을 구분하는 작업
- **다중 레이블 분류**: 질병 상태, 질병 유형, 신체 부위 및 지리적 위치를 동시에 예측하는 작업

모델은 CNN(합성곱 신경망)과 ViT(비전 트랜스포머)를 결합한 퓨전 아키텍처를 사용하여 각 샘플의 임베딩을 생성하고, 이를 통해 다양한 질병 상태를 높은 정확도로 예측합니다.

---




This paper proposes a novel machine learning model to predict disease states based on microbiome features. The model utilizes visual representations called "Microbiome Maps" to convert microbial compositions into images, enabling the creation of embeddings that capture multiple factors such as disease status, disease type, body site, and geographical location.

#### Training Data and Test Data

1. **Training Data**:
   - **Number of Samples**: 13,534 public human metagenome samples
   - **Number of Studies**: Data collected from 85 studies
   - **Disease Types**: 24 different disease types (e.g., autoimmune diseases, cancer, cardiovascular diseases, etc.)
   - **Body Sites**: 4 body sites (e.g., gut, oral cavity, etc.)
   - **Geographical Locations**: Data from 35 countries
   - **Input**: Images representing the microbial composition of each sample (Microbiome Map)
   - **Output**: Disease status (healthy/diseased), disease type, body site, geographical location for each sample

2. **Test Data**:
   - **Number of Samples**: 1,350 samples (10% holdout set)
   - **Input**: Microbiome Map images of test samples
   - **Output**: Predicted disease status, disease type, body site, geographical location

#### Specific Tasks

- **Binary Classification**: Distinguishing between healthy and diseased samples
- **Multi-Class Classification**: Classifying among 24 disease types
- **Multi-Label Classification**: Simultaneously predicting disease status, disease type, body site, and geographical location

The model employs a fusion architecture combining CNN (Convolutional Neural Network) and ViT (Vision Transformer) to generate embeddings for each sample, allowing for high-accuracy predictions of various disease states.

<br/>
# 요약


이 연구에서는 13,534개의 공공 메타게놈 샘플을 사용하여 미생물 군집의 시각적 표현을 기반으로 한 융합 모델을 개발하였으며, 이 모델은 질병 상태, 질병 유형 및 지리적 위치를 예측하는 데 높은 정확도를 보였습니다. 모델은 CNN과 ViT를 결합하여 미생물 맵 이미지를 처리하고, 다중 레이블 분류 작업에서 평균 84%의 정확도를 달성했습니다. 이 접근법은 새로운 샘플에 대한 임베딩을 생성하고 기존 클러스터에 적합시켜 미지의 미생물 샘플을 특성화하는 데 활용될 수 있습니다.

---

This study developed a fusion model based on visual representations of microbial communities using 13,534 public metagenomic samples, achieving high accuracy in predicting disease status, disease type, and geographical location. The model combines CNN and ViT to process microbiome map images, achieving an average accuracy of 84% in multi-label classification tasks. This approach allows for the generation of embeddings for new samples that can be fitted to existing clusters, enabling the characterization of unknown microbiome samples.

<br/>
# 기타



#### 1. 다이어그램 및 피규어
- **Microbiome Maps**: 이 피규어는 미생물 군집의 조성을 시각적으로 표현한 것으로, 각 픽셀은 특정 미생물의 상대적 풍부함을 나타냅니다. 이러한 시각적 표현은 미생물의 조성 패턴을 이해하는 데 도움을 주며, 질병 상태와의 연관성을 파악하는 데 유용합니다.
  
- **Fusion Model Architecture**: 이 다이어그램은 CNN과 ViT 모델을 결합한 퓨전 모델의 구조를 보여줍니다. CNN은 지역적 패턴을 학습하고, ViT는 전역적 맥락을 모델링하여 미생물 맵의 공간적 세부사항을 잘 포착합니다. 이 구조는 질병 상태 예측의 정확성을 높이는 데 기여합니다.

- **Attention and Activation Maps**: 이 피규어는 모델이 특정 미생물 군집에 집중하는 영역을 시각화합니다. 이는 모델이 어떤 미생물 패턴에 주목하는지를 보여주며, 해석 가능성을 높입니다.

#### 2. 테이블
- **Table 1**: 각 질병 카테고리와 조건에 따른 샘플 수를 나타냅니다. 이 데이터는 연구의 다양성과 샘플의 균형을 보여주며, 다양한 질병 상태를 예측하는 데 필요한 데이터의 범위를 제공합니다.

- **Table 2**: 모델의 성능을 비교한 결과로, CNN, ViT, 그리고 퓨전 모델의 AUC(Area Under Curve) 값을 보여줍니다. 퓨전 모델이 모든 평가 지표에서 가장 높은 성능을 보였으며, 이는 다양한 질병 상태를 효과적으로 분류할 수 있는 능력을 나타냅니다.

#### 3. 어펜딕스
- **Supplementary Materials**: 추가 자료는 모델 훈련 및 평가에 대한 세부 정보를 제공합니다. 여기에는 데이터 전처리, 모델 아키텍처, 성능 평가 방법 등이 포함되어 있어 연구의 재현성을 높입니다.

---

### Summary of Results and Insights

#### 1. Diagrams and Figures
- **Microbiome Maps**: This figure visually represents the composition of microbial communities, where each pixel indicates the relative abundance of a specific microbe. Such visual representations aid in understanding microbial composition patterns and their associations with disease states.

- **Fusion Model Architecture**: This diagram illustrates the structure of the fusion model that combines CNN and ViT. The CNN learns local patterns, while the ViT models global context, effectively capturing the spatial details of microbiome maps. This structure contributes to improved accuracy in predicting disease states.

- **Attention and Activation Maps**: This figure visualizes the areas where the model focuses on specific microbial communities. It shows which microbial patterns the model pays attention to, enhancing interpretability.

#### 2. Tables
- **Table 1**: Displays the number of samples in each disease category and condition. This data highlights the diversity of the study and the balance of samples, providing a range necessary for predicting various disease states.

- **Table 2**: Compares the performance of models, showing AUC (Area Under Curve) values for CNN, ViT, and the fusion model. The fusion model achieved the highest performance across all metrics, indicating its ability to effectively classify various disease states.

#### 3. Appendix
- **Supplementary Materials**: Additional materials provide detailed information on model training and evaluation. This includes data preprocessing, model architecture, and performance evaluation methods, enhancing the reproducibility of the research.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@article{Valdes2026,
  author = {Camilo Valdes and Andre Goncalves and Haonan Zhu and Boya Zhang and Hiran Ranganathan and James Thissen and Jose Manuel Marti and Car Reen Kok and Nisha Mulakken and Crystal Jaing and Nicholas Be},
  title = {An Embeddings Fusion Approach Predicts Disease State from Microbiome Features},
  journal = {bioRxiv},
  year = {2026},
  month = {April},
  doi = {10.64898/2026.04.20.719747},
  url = {https://doi.org/10.64898/2026.04.20.719747}
}
```

### 시카고 스타일

Valdes, Camilo, Andre Goncalves, Haonan Zhu, Boya Zhang, Hiran Ranganathan, James Thissen, Jose Manuel Marti, Car Reen Kok, Nisha Mulakken, Crystal Jaing, and Nicholas Be. "An Embeddings Fusion Approach Predicts Disease State from Microbiome Features." *bioRxiv*, April 2026. https://doi.org/10.64898/2026.04.20.719747.
