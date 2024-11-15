---
layout: post
title:  "[2024]Beyond Deepfake Images Detecting AI-Generated Videos"  
date:   2024-11-15 01:23:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 


이 논문은 **Forensic 흔적(비디오 생성 과정에서 남은 흔적) 학습**과 **Few-shot 학습(적은 데이터로 파인튜닝)**을 결합하여, 기존의 탐지 모델이 새로운 비디오 생성기에도 적응할 수 있는 메서드를 제안합니다. Few-shot 학습은 기존의 지식을 바탕으로 최소한의 데이터를 사용해 새로운 forensic 패턴을 학습함으로써, 빠르고 효율적인 적응을 가능하게 합니다.


짧은 요약(Abstract) :    




최근 생성형 AI의 발전으로 인해 시각적으로 현실적인 합성 비디오를 생성하는 기술이 발전했습니다. 하지만 기존의 합성 이미지 탐지기는 합성 비디오를 제대로 탐지하지 못합니다. 이는 비디오 생성기가 이미지 생성기와 다른 흔적을 남기기 때문입니다. 본 연구에서는 합성 비디오에서 남겨진 독특한 흔적을 학습하여 H.264 재압축 후에도 신뢰할 수 있는 탐지 및 생성기 소스 추적을 수행할 수 있음을 보여줍니다. 또한, 새로운 생성기를 탐지하기 위해 제로샷 전이 학습은 어려움을 겪지만, 소량의 데이터로 학습하는 few-shot 학습을 통해 높은 정확도를 달성할 수 있음을 입증합니다.



Recent advances in generative AI have led to the development of techniques to generate visually realistic synthetic video. While a number of techniques have been developed to detect AI-generated synthetic images, this paper shows that synthetic image detectors are unable to detect synthetic videos. This is because synthetic video generators introduce substantially different traces than those left by image generators. Despite this, the study demonstrates that synthetic video traces can be learned and used for reliable synthetic video detection or generator source attribution even after H.264 re-compression. Furthermore, while detecting videos from new generators through zero-shot transferability is challenging, accurate detection of videos from a new generator can be achieved through few-shot learning.



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




이 논문의 메서드는 비디오 생성기의 **forensic 흔적**을 학습하고 이를 통해 합성 비디오를 탐지하며, 새로운 생성기에도 빠르게 적응할 수 있는 Few-shot 학습 방법론을 제안합니다. 주요 내용은 다음과 같습니다:

---

#### 1. **Forensic 흔적**
   - **Forensic 흔적**은 비디오 생성 과정에서 생성기가 남기는 **고유한 디지털 신호나 패턴**을 의미합니다. 
   - 예를 들어:
     - 픽셀 간의 색상 왜곡, 패턴 반복, 특정 주파수 대역에서 나타나는 이상 현상 등.
     - H.264와 같은 압축 환경에서도 변하지 않는 특징.
   - 이 논문은 비디오 생성기가 이미지 생성기와 다른 forensic 흔적을 남긴다는 점을 발견하고, 이를 학습해 비디오 탐지와 생성기 소스 추적에 활용합니다.

---

#### 2. **Few-shot 학습**
   - 기존의 제로샷 학습은 새로운 비디오 생성기를 탐지하는 데 한계가 있습니다. 이를 극복하기 위해, 논문은 소량의 데이터를 사용한 Few-shot 학습을 도입했습니다.
   - **Few-shot 학습의 주요 과정**:
     1. **사전 학습된 모델 활용**:
        - 기존 합성 비디오 탐지 모델의 가중치를 초기화 값으로 사용하여 학습 속도를 높이고 기존 forensic 흔적의 일부를 재활용.
     2. **소량의 라벨링 데이터로 미세 조정(Fine-tuning)**:
        - 새로운 생성기로부터 얻은 비디오 샘플 몇 개를 모델에 추가 학습시켜 새로운 forensic 흔적을 학습.
     3. **데이터 증강(Data Augmentation)**:
        - Few-shot 데이터의 부족 문제를 해결하기 위해, 크기 변환, 노이즈 추가, 밝기 조정 등 다양한 데이터 증강 기법을 활용.
     4. **전이 학습(Transfer Learning)**:
        - 기존 forensic 흔적 학습 가중치를 기반으로 새로운 생성기에 적응.

---

#### 3. **비디오-레벨 탐지**
   - 기존에는 프레임 단위 탐지로 각 이미지 프레임에서 결과를 도출했습니다.
   - 이 논문에서는 여러 프레임의 forensic 흔적을 통합해 비디오 전체에서 생성기 특성을 분석하는 **비디오-레벨 탐지 방법**을 제안했습니다.
   - 이 방법은 개별 프레임에서 놓칠 수 있는 비디오 전반의 생성기 특성을 포착하며 탐지 성능을 크게 향상시켰습니다.

---

#### 4. **데이터셋 및 평가**
   - **데이터셋 구축**:
     - 최신 비디오 생성기(Luma, VideoCrafter 등)에서 생성된 합성 비디오와 Moments in Time(MiT), Video-ACID 같은 실제 비디오를 포함.
     - 비디오 데이터는 H.264 압축을 적용해 다양한 품질 조건에서의 모델 학습을 가능하게 설계.
   - **평가 방법**:
     - AUC와 정확도를 주요 지표로 사용해 프레임 수준 및 비디오 수준에서 성능을 비교 평가.

---



This paper proposes a methodology that focuses on learning the **forensic traces** of video generators to detect synthetic videos and adapt to new generators using **Few-shot learning**. The main points are:

---

#### 1. **Forensic Traces**
   - **Forensic traces** are unique digital signals or patterns left during the video generation process.
   - Examples include:
     - Pixel-level color distortions, repetitive patterns, or anomalies in specific frequency bands.
     - Features that persist even after H.264 compression.
   - The paper identifies that video generators leave distinct forensic traces compared to image generators and uses these traces for video detection and source attribution.

---

#### 2. **Few-Shot Learning**
   - To address the limitations of zero-shot learning in detecting new video generators, the paper introduces **Few-shot learning**.
   - **Few-shot learning process**:
     1. **Pre-trained Model Initialization**:
        - Weights from existing synthetic video detection models are used to initialize the learning process, leveraging prior knowledge of forensic traces.
     2. **Fine-tuning with Minimal Data**:
        - A small amount of labeled video samples from new generators is used to fine-tune the model, enabling it to learn new forensic traces.
     3. **Data Augmentation**:
        - Techniques like resizing, noise addition, and brightness adjustment are applied to enrich the limited dataset and improve generalization.
     4. **Transfer Learning**:
        - Builds on previously learned forensic traces to adapt to the new generator’s characteristics.

---

#### 3. **Video-Level Detection**
   - Instead of analyzing individual frames, the paper aggregates forensic traces across multiple frames to perform **video-level detection**.
   - This method captures generator-specific patterns that may not be evident in frame-level analysis, significantly enhancing detection performance.

---

#### 4. **Dataset and Evaluation**
   - **Dataset Construction**:
     - Synthetic videos generated using state-of-the-art tools like Luma and VideoCrafter, along with real videos from Moments in Time (MiT) and Video-ACID datasets.
     - H.264 compression was applied to ensure robust performance under varying quality conditions.
   - **Evaluation**:
     - Performance was assessed using AUC and accuracy, comparing results at both frame and video levels.

---



   
 
<br/>
# Results  




논문의 결과는 제안된 모델이 기존의 이미지 탐지 모델보다 합성 비디오 탐지 성능에서 우수함을 보여줍니다. 주요 비교 모델, 사용 데이터셋, 성능 증가 결과는 다음과 같습니다:

---

#### 1. **비교 모델**
   - **기존 이미지 탐지 모델**:
     - **ResNet-50**: 잔차 학습을 사용하는 이미지 분류 모델.
     - **Swin-Transformer**: 트랜스포머 기반의 이미지 분석 모델.
     - **Xception**: 효율적인 깊이별 컨볼루션을 사용하는 이미지 분류 모델.
     - **DenseNet**: 피처를 조밀하게 연결하여 학습하는 모델.
   - **MISLnet**: 비디오 수준에서 forensic 흔적을 종합적으로 학습하는 모델로 제안된 모델과 가장 직접적인 비교 대상.
   - 이러한 모델들은 모두 프레임 단위로 작동하며, 비디오 전체를 분석하지 못함.

---

#### 2. **사용 데이터셋**
   - **합성 비디오 데이터셋**:
     - **Luma**, **VideoCrafter**, **Stable Video Diffusion**, **CogVideo** 등 최신 비디오 생성기로 생성된 비디오 데이터.
     - 텍스트 프롬프트를 활용하여 다양한 합성 비디오를 생성.
   - **실제 비디오 데이터셋**:
     - **Moments in Time (MiT)**: 다양한 실제 비디오가 포함된 대규모 데이터셋.
     - **Video-ACID**: 품질 높은 실제 비디오 데이터셋.
   - **H.264 압축 데이터**:
     - 다양한 품질 조건에서 모델의 성능을 테스트하기 위해 모든 비디오 데이터에 H.264 압축을 적용.

---

#### 3. **평가 메트릭**
   - **AUC (Area Under Curve)**:
     - 합성 비디오 탐지 성능 평가의 주요 지표로 사용.
   - **탐지 정확도(Detection Accuracy)**:
     - 실제와 합성 비디오를 구분하는 정확도를 측정.

---

#### 4. **주요 결과**
   - **AUC 증가**:
     - MISLnet과의 비교에서 제안된 모델은 AUC 점수가 평균 **6~8%** 향상.
     - 특히 H.264 압축 데이터에서 성능이 더 뚜렷하게 증가.
   - **Few-shot 학습**:
     - 새로운 생성기(Luma 등)에 대해 Few-shot 학습을 수행했을 때, **제로샷 학습 대비 탐지 정확도가 12~15% 향상**.
   - **비디오-레벨 분석**:
     - 프레임 단위 탐지만 수행했을 때보다 비디오-레벨 탐지에서 AUC와 정확도가 각각 **10% 이상 향상**.

---



The results demonstrate that the proposed model outperforms existing image detection models in synthetic video detection. Key comparisons, datasets, and performance improvements are summarized below:

---

#### 1. **Comparison Models**
   - **Existing Image Detection Models**:
     - **ResNet-50**: A residual learning-based image classification model.
     - **Swin-Transformer**: A transformer-based image analysis model.
     - **Xception**: An efficient image classification model using depth-wise convolutions.
     - **DenseNet**: A model that densely connects features during learning.
   - **MISLnet**: A direct comparison model capable of aggregating frame-level forensic traces for video-level analysis.
   - All models operate on a frame-by-frame basis, lacking comprehensive video-level analysis.

---

#### 2. **Datasets Used**
   - **Synthetic Video Datasets**:
     - Videos generated using state-of-the-art generators like **Luma**, **VideoCrafter**, **Stable Video Diffusion**, and **CogVideo**.
     - Diverse synthetic videos were created using various text prompts.
   - **Real Video Datasets**:
     - **Moments in Time (MiT)**: A large-scale dataset with diverse real-world videos.
     - **Video-ACID**: A high-quality dataset of real videos.
   - **H.264 Compressed Data**:
     - To evaluate model robustness, all videos were processed with H.264 compression.

---

#### 3. **Evaluation Metrics**
   - **AUC (Area Under Curve)**:
     - Used as the primary metric for evaluating synthetic video detection performance.
   - **Detection Accuracy**:
     - Measures the ability to distinguish between real and synthetic videos.

---

#### 4. **Key Results**
   - **AUC Improvements**:
     - Compared to MISLnet, the proposed model achieved an average AUC improvement of **6–8%**, especially pronounced on H.264 compressed data.
   - **Few-shot Learning**:
     - For new generators (e.g., Luma), few-shot learning improved detection accuracy by **12–15%** over zero-shot learning.
   - **Video-Level Analysis**:
     - Video-level detection enhanced AUC and accuracy by **over 10%** compared to frame-level analysis alone. 

---



<br/>
# 예제  



제안된 모델은 Luma 생성기로 만든 합성 비디오에서 H.264 압축 후에도 픽셀 간 색상 왜곡 흔적을 정확히 탐지했으나, 비교 모델인 ResNet-50은 이러한 패턴을 제대로 탐지하지 못했다.

---


The proposed model accurately detected pixel-level color distortion traces in synthetic videos generated by Luma, even after H.264 compression, whereas the comparison model, ResNet-50, failed to identify these patterns.

<br/>  
# 요약   



이 논문은 비디오 생성기의 forensic 흔적을 학습하여 합성 비디오를 탐지하고, 새로운 생성기에도 적응할 수 있는 Few-shot 학습 방법론을 제안합니다. 제안된 모델은 Luma, VideoCrafter와 같은 최신 합성 비디오 생성기에서 생성된 데이터에서 압축(H.264) 후에도 높은 탐지 성능을 보였습니다. 특히, Few-shot 학습을 통해 새로운 생성기에 대해 제로샷 학습보다 탐지 정확도가 12~15% 향상되었습니다. 예를 들어, 제안된 모델은 Luma 생성 비디오에서 색상 왜곡 흔적을 정확히 탐지했지만, ResNet-50과 같은 기존 모델은 이를 탐지하지 못했습니다. 제안된 모델은 AUC 기준으로 MISLnet 대비 평균 6~8% 높은 성능을 기록했습니다.

---


This paper proposes a methodology to learn forensic traces of video generators to detect synthetic videos and adapt to new generators using Few-shot learning. The proposed model demonstrated high detection performance on datasets generated by state-of-the-art generators like Luma and VideoCrafter, even after H.264 compression. Few-shot learning improved detection accuracy for new generators by 12–15% compared to zero-shot learning. For instance, the proposed model accurately detected color distortion traces in Luma-generated videos, whereas existing models like ResNet-50 failed to do so. The model achieved an average AUC improvement of 6–8% over MISLnet.


<br/>  
# 기타  


<br/>
# refer format:     

@inproceedings{vahdati2024beyond,
  title={Beyond Deepfake Images: Detecting AI-Generated Videos},
  author={Vahdati, Danial Samadi and Nguyen, Tai D. and Azizpour, Aref and Stamm, Matthew C.},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2024},
  organization={IEEE},
  url={https://example.com}, % replace with the actual URL if available
}



Vahdati, Danial Samadi, Tai D. Nguyen, Aref Azizpour, and Matthew C. Stamm. "Beyond Deepfake Images: Detecting AI-Generated Videos." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2024.    