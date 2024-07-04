---
layout: post
title:  "[2021]Learning Transferable Visual Models From Natural Language Supervision"  
date:   2024-07-04 16:44:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

짧은 요약(Abstract) :    


최신 컴퓨터 비전 시스템은 미리 정해진 객체 카테고리 집합을 예측하도록 훈련됩니다. 이러한 제한된 형태의 감독은 일반성과 사용성을 제한하여 다른 시각적 개념을 지정하려면 추가로 레이블이 지정된 데이터가 필요합니다. 이미지에 대한 원시 텍스트로부터 직접 학습하는 것은 훨씬 더 광범위한 감독 소스를 활용하는 유망한 대안입니다. 우리는 캡션과 이미지의 일치 여부를 예측하는 간단한 사전 훈련 작업이 인터넷에서 수집된 4억 쌍의 (이미지, 텍스트) 데이터셋에서 처음부터 SOTA 이미지 표현을 효율적이고 확장 가능하게 학습하는 방법을 보여줍니다. 사전 훈련 후 자연어를 사용하여 학습된 시각적 개념을 참조하거나 새로운 개념을 설명하여 모델을 후속 작업으로 제로샷 전이할 수 있습니다. 우리는 OCR, 비디오에서의 액션 인식, 지리적 위치 확인, 다양한 유형의 세밀한 객체 분류와 같은 30개 이상의 다양한 컴퓨터 비전 데이터셋에서 성능을 연구합니다. 이 모델은 대부분의 작업에 비트리비얼하게 전이되며 데이터셋 특정 훈련 없이도 종종 완전 감독된 기준과 경쟁할 수 있습니다. 예를 들어, 우리는 원래의 ResNet50이 ImageNet에서 훈련한 1.28백만 개의 예제를 사용하지 않고도 ImageNet 제로샷에서 원래 ResNet50의 정확도를 일치시킵니다. 우리는 우리의 코드와 사전 훈련된 모델 가중치를 [여기](https://github.com/OpenAI/CLIP)에서 공개합니다.


State-of-the-art (SOTA) computer vision systems are trained to predict a fixed set of predetermined object categories. This restricted form of supervision limits their generality and usability since additional labeled data is needed to specify any other visual concept. Learning directly from raw text about images is a promising alternative which leverages a much broader source of supervision. We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet. After pre-training, natural language is used to reference learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks. We study performance on over 30 different computer vision datasets, spanning tasks such as OCR, action recognition in videos, geo-localization, and many types of fine-grained object classification. The model transfers non-trivially to most tasks and is often competitive with a fully supervised baseline without the need for any dataset specific training. For instance, we match the accuracy of the original ResNet50 on ImageNet zero-shot without needing to use any of the 1.28 million training examples it was trained on. We release our code and pre-trained model weights at [here](https://github.com/OpenAI/CLIP).




* Useful sentences :  


{% endhighlight %}  

<br/>

[Paper link](https://drive.google.com/drive/folders/1lYInbcFeFyP_ejZKqxzTkcbtn851FIow?usp=sharing)  
[~~Lecture link~~]()   

<br/>

# 단어정리  
*  
 
<br/>
# Methodology    


#### 데이터셋 구축

기존 연구는 주로 MS-COCO, Visual Genome, YFCC100M 세 가지 데이터셋을 사용해 왔습니다. MS-COCO와 Visual Genome은 높은 품질의 크라우드 소싱된 데이터셋이지만 각각 약 10만 장의 사진만 포함하고 있어 현대의 기준으로는 작습니다. 반면 다른 컴퓨터 비전 시스템은 최대 35억 장의 인스타그램 사진으로 훈련됩니다. YFCC100M은 1억 장의 사진을 포함하고 있지만 각 이미지의 메타데이터가 부족하고 품질이 다양합니다. 이러한 한계를 극복하기 위해 우리는 인터넷에서 수집된 4억 쌍의 (이미지, 텍스트) 데이터를 사용해 새로운 데이터셋을 구축했습니다.

#### 효율적인 사전 훈련 방법 선택

우리의 초기 접근법은 VirTex와 유사하게 이미지 CNN과 텍스트 변환기를 공동으로 훈련하여 이미지의 캡션을 예측하는 것이었습니다. 그러나 이 방법을 효율적으로 확장하는 데 어려움을 겪었습니다. 최근의 대조적 표현 학습 연구는 대조적 목표가 예측 목표를 능가할 수 있음을 발견했습니다. 이를 바탕으로, 우리는 CLIP(Contrastive Language-Image Pre-training)를 소개하여 이미지와 텍스트 인코더를 공동으로 훈련시켜 대조적 임베딩 공간을 최대화하는 방식으로 학습했습니다.

#### 모델 선택 및 확장

이미지 인코더로는 ResNet50과 Vision Transformer를 사용했습니다. ResNet50을 기반으로 여러 가지 수정 사항을 적용하고, Vision Transformer는 최근 도입된 ViT를 사용했습니다. 텍스트 인코더로는 변형된 Transformer를 사용했습니다. 모델의 확장은 Tan & Le의 접근 방식을 채택하여 너비, 깊이, 해상도를 동시에 확장했습니다.

#### 사전 훈련

다양한 크기의 5개의 ResNet과 3개의 Vision Transformer를 훈련했습니다. 가장 큰 ResNet 모델인 RN50x64는 592개의 V100 GPU에서 18일 동안 훈련되었고, 가장 큰 Vision Transformer는 256개의 V100 GPU에서 12일 동안 훈련되었습니다. 

#### CLIP 사용하기

CLIP는 이미지와 텍스트 조각이 함께 묶인 경우를 예측하도록 사전 훈련되었습니다. 이를 다운스트림 작업에 적용하기 위해, 우리는 CLIP의 제로샷 전이 성능을 다양한 컴퓨터 비전 데이터셋에서 평가했습니다. 이를 통해 CLIP가 대부분의 작업에서 비트리비얼하게 전이되며, 종종 완전 감독된 기준과 경쟁할 수 있음을 확인했습니다.



#### Creating a Sufficiently Large Dataset

Existing work has mainly used three datasets: MS-COCO, Visual Genome, and YFCC100M. While MS-COCO and Visual Genome are high-quality crowd-labeled datasets, they are small by modern standards with approximately 100,000 training photos each. By comparison, other computer vision systems are trained on up to 3.5 billion Instagram photos. YFCC100M, at 100 million photos, is a possible alternative but the metadata for each image is sparse and of varying quality. To overcome these limitations, we constructed a new dataset of 400 million (image, text) pairs collected from a variety of publicly available sources on the Internet.

#### Selecting an Efficient Pre-Training Method

Our initial approach, similar to VirTex, jointly trained an image CNN and text transformer from scratch to predict the caption of an image. However, we encountered difficulties efficiently scaling this method. Recent work in contrastive representation learning has found that contrastive objectives can outperform the equivalent predictive objective. Based on this finding, we introduce CLIP (Contrastive Language-Image Pre-training) which jointly trains an image and text encoder to maximize the cosine similarity of the image and text embeddings.

#### Choosing and Scaling a Model

We used ResNet50 and Vision Transformer as image encoders. The ResNet50 was modified with several improvements and the Vision Transformer followed the recently introduced ViT. The text encoder was a modified Transformer. Model scaling followed the approach of Tan & Le, allocating additional compute across width, depth, and resolution.

#### Pre-Training

We trained a series of five ResNets and three Vision Transformers of varying sizes. The largest ResNet model, RN50x64, was trained for 18 days on 592 V100 GPUs, while the largest Vision Transformer was trained for 12 days on 256 V100 GPUs.

#### Using CLIP

CLIP is pre-trained to predict if an image and a text snippet are paired together. To apply CLIP to downstream tasks, we evaluated its zero-shot transfer performance on various computer vision datasets. We found that CLIP transfers non-trivially to most tasks and is often competitive with a fully supervised baseline.




<br/>
# Results  



#### 초기 비교

Visual N-Grams (Li et al., 2017)이 제시한 방법과 비교해보면, CLIP는 ImageNet의 정확도를 11.5%에서 76.2%로 향상시켰고, 1.28백만 개의 크라우드 레이블이 지정된 훈련 예제를 사용하지 않고도 원래 ResNet50의 성능을 맞췄습니다. 또한 CLIP 모델의 Top-5 정확도는 Inception-V4와 일치하는 95%를 기록했습니다. 이는 강력한 완전 감독된 기준을 제로샷 설정에서 맞춘다는 점에서 CLIP가 유연하고 실용적인 제로샷 컴퓨터 비전 분류기로서 중요한 진전을 이뤘음을 시사합니다.

#### 다양한 데이터셋에서의 성능

CLIP는 총 27개의 데이터셋에서 성능을 평가받았습니다. 이 데이터셋에는 STL10, Stanford Cars, Food101, Flowers102, FGVCAircraft, ImageNet, CIFAR10, PascalVOC2007, Kinetics700, UCF101 등이 포함됩니다. 이 평가에서 CLIP는 대부분의 데이터셋에서 높은 성능을 보였고, 특히 STL10에서는 새로운 SOTA를 달성했습니다. 일반 객체 분류 데이터셋에서는 제로샷 CLIP가 약간의 우위를 보였고, 동영상에서의 액션 인식에서는 ResNet50을 능가했습니다.

#### 자연 분포 이동에 대한 강인성

제로샷 CLIP 모델은 자연 분포 이동에 대해 매우 강인한 성능을 보였습니다. 모든 제로샷 CLIP 모델은 ImageNet 정확도와 분포 이동 후 정확도 사이의 격차를 최대 75%까지 줄였습니다. 이는 대규모 작업 및 데이터셋 비특이적 사전 훈련이 시스템의 강인성을 향상시킨다는 것을 시사합니다.

#### 데이터 중복 분석

큰 인터넷 데이터셋에서 사전 훈련을 수행하는 경우, 다운스트림 평가와의 의도치 않은 중복이 우려됩니다. 35개의 데이터셋 중 9개는 중복이 전혀 없었고, 중복율의 중앙값은 2.2%, 평균은 3.2%였습니다. 중복으로 인한 전체 정확도의 변화는 거의 없었고, 최대 0.6%의 향상만이 통계적으로 유의미했습니다.



#### Initial Comparison

Comparing to Visual N-Grams (Li et al., 2017), CLIP improves accuracy on ImageNet from a proof of concept 11.5% to 76.2% and matches the performance of the original ResNet50 despite using none of the 1.28 million crowd-labeled training examples. Additionally, the top-5 accuracy of CLIP models is noticeably higher, matching Inception-V4 with a 95% top-5 accuracy. This ability to match the performance of a strong, fully supervised baseline in a zero-shot setting suggests CLIP is a significant step towards flexible and practical zero-shot computer vision classifiers.

#### Performance Across Various Datasets

CLIP was evaluated on a total of 27 datasets, including STL10, Stanford Cars, Food101, Flowers102, FGVCAircraft, ImageNet, CIFAR10, PascalVOC2007, Kinetics700, and UCF101. CLIP showed high performance on most datasets, achieving a new SOTA on STL10. On general object classification datasets, zero-shot CLIP had a slight advantage, and on action recognition in videos, it outperformed ResNet50.

#### Robustness to Natural Distribution Shift

Zero-shot CLIP models demonstrated significant robustness to natural distribution shifts. All zero-shot CLIP models reduced the gap between ImageNet accuracy and accuracy under distribution shift by up to 75%. This suggests that large-scale task and dataset agnostic pre-training promotes the development of more robust systems.

#### Data Overlap Analysis

Pre-training on a large internet dataset raises concerns of unintentional overlap with downstream evaluations. Out of 35 datasets studied, 9 had no detected overlap, with a median overlap of 2.2% and an average of 3.2%. The overall accuracy was rarely shifted by more than 0.1%, with only 7 datasets above this threshold, and only 2 being statistically significant after Bonferroni correction, with a maximum detected improvement of only 0.6%.

 


<br/>
# 예시  



#### 예시 1: STL10 데이터셋에서의 성능
CLIP 모델은 STL10 데이터셋에서 새로운 SOTA(최고 성능)를 달성했습니다. 이 데이터셋은 제한된 수의 라벨이 지정된 예제를 포함하고 있어 비지도 학습을 권장합니다. 제로샷 CLIP는 훈련 예제를 전혀 사용하지 않고 99.3%의 정확도를 달성했습니다. 이는 모델이 STL10에서 최고 성능을 기록한 것입니다.

#### 예시 2: 다양한 데이터셋에서의 성능
CLIP 모델은 다양한 데이터셋에서 테스트되었으며, 특히 Stanford Cars와 Food101에서는 로지스틱 회귀 모델을 각각 20% 이상 초과하는 성능을 보였습니다. 그러나 Flowers102와 FGVCAircraft 데이터셋에서는 10% 이상 낮은 성능을 보였습니다. 이는 WIT와 ImageNet 간의 각 작업에 대한 감독 양의 차이 때문으로 추정됩니다.

#### 예시 3: 비디오에서의 액션 인식
CLIP는 비디오에서의 액션 인식 데이터셋인 Kinetics700과 UCF101에서도 탁월한 성능을 보였습니다. Kinetics700에서는 ResNet50보다 14.5% 더 높은 성능을 보였고, UCF101에서는 7.7% 더 높은 성능을 보였습니다. 이는 동사와 관련된 시각적 개념에 대해 자연어가 더 넓은 감독을 제공하기 때문으로 보입니다.

#### 예시 4: 특화된 작업에서의 성능 저하
제로샷 CLIP는 위성 이미지 분류(예: EuroSAT, RESISC45), 림프절 종양 감지(PatchCamelyon), 합성 장면에서의 객체 수 세기(CLEVRCounts), 자율 주행 관련 작업(예: 독일 교통 표지 인식(GTSRB), 가장 가까운 차량까지의 거리 인식(KITTI Distance))과 같은 특화된, 복잡하거나 추상적인 작업에서 약한 성능을 보였습니다. 이는 제로샷 CLIP가 더 복잡한 작업에 대한 성능이 낮음을 강조합니다.



#### Example 1: Performance on the STL10 Dataset
The CLIP model achieved a new state-of-the-art (SOTA) on the STL10 dataset, which is designed to encourage unsupervised learning by containing only a limited number of labeled examples. Zero-shot CLIP, without using any training examples, achieved 99.3% on this dataset, setting a new SOTA.

#### Example 2: Performance Across Various Datasets
The CLIP model was evaluated on various datasets and showed outstanding performance, especially on Stanford Cars and Food101, where it outperformed logistic regression models by over 20%. However, on Flowers102 and FGVCAircraft datasets, zero-shot CLIP underperformed by over 10%. These differences are suspected to be due to varying amounts of per-task supervision between WIT and ImageNet.

#### Example 3: Action Recognition in Videos
CLIP also demonstrated excellent performance on video action recognition datasets such as Kinetics700 and UCF101. On Kinetics700, CLIP outperformed ResNet50 by 14.5%, and on UCF101, it outperformed ResNet50's features by 7.7%. This is speculated to be due to natural language providing wider supervision for visual concepts involving verbs compared to the noun-centric object supervision in ImageNet.

#### Example 4: Poor Performance on Specialized Tasks
Zero-shot CLIP showed weak performance on several specialized, complex, or abstract tasks such as satellite image classification (e.g., EuroSAT, RESISC45), lymph node tumor detection (PatchCamelyon), counting objects in synthetic scenes (CLEVRCounts), and self-driving related tasks (e.g., German traffic sign recognition (GTSRB), recognizing distance to the nearest car (KITTI Distance)). These results highlight the poor capability of zero-shot CLIP on more complex tasks.


<br/>  
# 요약 


CLIP 모델은 인터넷에서 수집한 4억 쌍의 (이미지, 텍스트) 데이터를 사용하여 학습되었습니다. 이미지와 텍스트 인코더를 공동으로 훈련시켜 대조적 임베딩 공간을 최대화하는 방식으로 학습했습니다. 이미지 인코더로는 ResNet50과 Vision Transformer를 사용했으며, 텍스트 인코더로는 변형된 Transformer를 사용했습니다. 모델의 확장은 너비, 깊이, 해상도를 동시에 확장하는 방식으로 이루어졌습니다. 제로샷 CLIP는 다양한 데이터셋에서 테스트되었으며, 특히 자연 분포 이동에 대해 높은 강인성을 보였습니다.



The CLIP model was trained using 400 million (image, text) pairs collected from the internet. It employed joint training of image and text encoders to maximize the contrastive embedding space. The image encoders used were ResNet50 and Vision Transformer, while the text encoder was a modified Transformer. Model scaling involved simultaneous expansion of width, depth, and resolution. Zero-shot CLIP was tested across various datasets, demonstrating high robustness to natural distribution shifts.

# 기타  

CLIP 모델은 인터넷에서 수집한 4억 쌍의 (이미지, 텍스트) 데이터를 사용하여 학습되었습니다. 이미지와 텍스트 인코더를 공동으로 훈련시켜 대조적 임베딩 공간을 최대화하는 방식으로 학습했습니다. 이미지 인코더로는 ResNet50과 Vision Transformer를 사용했으며, 텍스트 인코더로는 변형된 Transformer를 사용했습니다.

사실 computer vision부분을 잘 모르기도 하지만 내 사전지식은 이 논문 이전에 멈춰있었던 갓 같다. 자연어를 사용한 CLIP을 통해 fine grained된.. 그러니까 레이블이 없거나 더 세세하게 나눠지는 분류와 같은 부분들도 더 모델이 잘 처리하게 된 것 같다. 굉장히 흥미롭고.. 이러한 논문들을 읽을 기회를 얻었기에 이 과목을 듣길 잘 했따는 생각이 든다.  


CLIP model was trained using 400 million image and text pairs collected from the internet. It employed joint training of image and text encoders to maximize the contrastive embedding space. The image encoders used were ResNet50 and Vision Transformer, while the text encoder was a modified Transformer.

I still don't know much about computer vision field, and it seems like my prior knowledge stopped at the point before this paper. CLIP uses natural language and handles fine-grained classifications such as those without labels or those divided into more detailed categories better. It's very interesting, and I'm glad I took this course because it gave me the opportunity to read such papers.

<br/>
# refer format:     
Radford, Alec, Kim, Jong Wook, Hallacy, Chris, Ramesh, Aditya, Goh, Gabriel, Agarwal, Sandhini, Sastry, Girish, Askell, Amanda, Mishkin, Pamela, Clark, Jack, Krueger, Gretchen, & Sutskever, Ilya. (2021). Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning (ICML 2021).


@inproceedings{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and Krueger, Gretchen and Sutskever, Ilya},
  booktitle={Proceedings of the 38th International Conference on Machine Learning (ICML 2021)},
  year={2021}
}

