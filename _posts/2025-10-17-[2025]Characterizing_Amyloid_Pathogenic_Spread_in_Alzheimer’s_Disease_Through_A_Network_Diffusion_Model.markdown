---
layout: post
title:  "[2025]Characterizing Amyloid Pathogenic Spread in Alzheimer’s Disease Through A Network Diffusion Model"
date:   2025-10-17 16:31:00 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 연구에서는 알츠하이머병(AD)에서 아밀로이드 병리의 확산을 모델링하기 위해 백질 구조 뇌 네트워크를 기반으로 한 네트워크 확산 모델을 설계하였다.


짧은 요약(Abstract) :

(근데 이미지 디퓨전(생성) 모델은 아니었음..)  
이 연구에서는 알츠하이머병(AD)의 아밀로이드 병리 확산을 백질 뇌 네트워크를 통해 시뮬레이션하는 네트워크 확산 모델을 설계했습니다. 이 모델은 건강한 대조군(HC), 경도 인지 장애(MCI), 그리고 AD 진단 하에 있는 하위 집단의 아밀로이드 확산을 성공적으로 모델링하였으며, 18F-플로르베타피르 양전자 방출 단층촬영(PET)에서 관찰된 아밀로이드의 지역 분포를 재현했습니다. 최적의 확산 시간(t)을 조정한 결과, HC 그룹이 가장 낮은 시간(107.22±16.67)을 보였고, MCI(122.78±19.63)와 AD(136.20±24.47) 그룹이 뒤를 이었습니다. 모든 세 그룹에서 최적의 시작 시드는 뇌간으로 나타났으며, HC와 MCI에서는 외측 안와전두엽이, AD에서는 혀회선이 뒤따랐습니다. 이 연구 결과는 아밀로이드가 주로 신피질과 연관 피질에서 시작된다는 아밀로이드 단계 연구의 증거를 뒷받침합니다. 백질 구조 네트워크의 확산 과정에서의 중요성은 AD에서 아밀로이드의 전시냅스 확산 가설에 대한 증거를 제공합니다. 결론적으로, 이 연구는 AD에서 아밀로이드의 병인학과 그 확산에 대한 새로운 통찰을 제공합니다.


In this study, we designed a network diffusion model to simulate the spread of amyloid pathology through white matter brain networks of diagnostic subpopulations of healthy control (HC), mild cognitive impairment (MCI), and Alzheimer's disease (AD). Our network diffusion model successfully modeled the spread of amyloid, recapturing regional distributions of amyloid observed in 18F-florbetapir positron emission tomography (PET). When tuning the optimal parameters, we found that the optimal diffusion time (t) provided a notion of temporal progression, where the HC group had the lowest time (107.22±16.67), followed by MCI (122.78±19.63), and lastly AD (136.20±24.47). The optimal starting seeds were the brainstem in all three diagnostic groups, followed by the lateral orbitofrontal lobes for HC and MCI and the lingual gyri in AD. Our findings corroborate evidence from amyloid staging studies where amyloid starts in the primary neocortex and associative cortex. The significance of the white matter structural network in the diffusion process provides evidence for the trans-synaptic spread hypothesis of amyloid in AD. In conclusion, our study provides novel insights into the pathogenesis of amyloid in AD and its subsequent propagation throughout the brain.


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



이 연구에서는 알츠하이머병(AD)에서 아밀로이드 병리의 전파를 모델링하기 위해 네트워크 확산 모델을 설계했습니다. 이 모델은 백질 뇌 네트워크를 통해 아밀로이드 병리가 어떻게 퍼지는지를 시뮬레이션합니다. 연구는 건강한 대조군(HC), 경도 인지 장애(MCI), 그리고 알츠하이머병(AD) 진단 하에 있는 세 가지 하위 집단을 대상으로 진행되었습니다.

#### 1. 데이터 수집
연구에 사용된 데이터는 알츠하이머병 신경영상 이니셔티브(ADNI) 데이터베이스에서 수집되었습니다. 이 데이터에는 확산 텐서 이미징(DTI) 스캔, T1 가중 구조 MRI(sMRI), 그리고 인구 통계학적 데이터가 포함되어 있습니다. 각 피험자는 83개의 피질 및 피질 하부 뇌 영역(ROI)으로 나누어져, 이들 간의 섬유 밀도를 기반으로 한 네트워크가 구성되었습니다.

#### 2. 네트워크 구축
DTI 데이터는 전처리 과정을 거쳐 노이즈 제거, 모션 보정, 왜곡 보정을 수행했습니다. 그런 다음, 섬유 할당 연속 추적(Fiber Assignment by Continuous Tracking, FACT) 알고리즘을 사용하여 섬유 밀도 네트워크를 구축했습니다. 각 네트워크의 인접 행렬은 두 ROI 간의 섬유 밀도로 가중치가 부여되었습니다.

#### 3. 아밀로이드 PET 데이터 전처리
18F-florbetapir(PET) 이미징 데이터는 ADNI 데이터베이스에서 수집되었으며, 이 데이터는 고해상도 MRI와 일치하도록 조정되었습니다. PET 데이터는 각 하위 집단의 평균 아밀로이드 분포를 계산하는 데 사용되었습니다.

#### 4. 네트워크 확산 모델
아밀로이드의 전파를 모델링하기 위해 열 확산 이론을 적용했습니다. 이론에 따르면, 아밀로이드는 연결된 뇌 영역 간에 전파됩니다. 모델은 다음의 열 방정식을 사용하여 아밀로이드의 분포를 설명합니다:

\[
\frac{dx(t)}{dt} = -\beta H x(t)
\]

여기서 \(x(t)\)는 시간 \(t\)에서 각 영역의 병리 분포를 나타내며, \(\beta\)는 확산 상수, \(H\)는 연결망의 라플라시안입니다. 초기 병리 분포 \(x(0)\)는 특정 시드 영역에 1의 값을 부여하고 나머지 영역에는 0을 부여하여 설정합니다.

#### 5. 최적화 및 평가
모델의 성능을 평가하기 위해, 예측된 분포와 실제 PET 데이터 간의 제곱 오차 합(SSE)을 최소화하는 최적의 확산 시간 \(t\)와 시드 영역을 찾았습니다. 세 가지 진단 그룹 간의 최적 확산 시간의 차이를 평가하기 위해 ANOVA를 사용했습니다.

#### 6. 부트스트래핑 및 널 모델 테스트
부트스트래핑을 통해 각 하위 집단의 차이를 검증하고, 무작위 네트워크 널 모델을 사용하여 구조적 뇌 네트워크의 중요성을 평가했습니다. 이 과정에서 무작위 네트워크는 구조적 네트워크의 엣지를 무작위로 섞어 생성되었습니다.

이 연구는 아밀로이드의 전파 메커니즘을 이해하는 데 중요한 통찰을 제공하며, 백질 구조적 네트워크가 아밀로이드 전파의 기초가 됨을 보여줍니다.

---
 

In this study, we designed a network diffusion model to simulate the spread of amyloid pathology in Alzheimer's disease (AD). This model simulates how amyloid pathology propagates through white matter brain networks. The research was conducted on three diagnostic subpopulations: healthy controls (HC), mild cognitive impairment (MCI), and Alzheimer's disease (AD).

#### 1. Data Acquisition
The data used in this study were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database. This data included diffusion tensor imaging (DTI) scans, T1-weighted structural MRI (sMRI), and demographic data. Each subject was divided into 83 cortical and subcortical brain regions of interest (ROIs), and a network was constructed based on the fiber density between these regions.

#### 2. Network Construction
The DTI data underwent preprocessing, including denoising, motion correction, and distortion correction. Subsequently, fiber assignment by continuous tracking (FACT) algorithm was used to construct fiber density networks. The adjacency matrix of each network was weighted by the fiber density between pairs of ROIs.

#### 3. Preprocessed Amyloid PET Data
The 18F-florbetapir (FBP) PET imaging data were collected from the ADNI database and aligned with high-resolution MRI. The PET data were used to compute the average amyloid distribution for each subpopulation.

#### 4. Network Diffusion Model
To model the propagation of amyloid, we applied the theory of heat diffusion. According to this theory, amyloid propagates between connected brain regions. The model uses the following heat equation to describe the distribution of amyloid:

\[
\frac{dx(t)}{dt} = -\beta H x(t)
\]

where \(x(t)\) represents the distribution of pathology in each region at time \(t\), \(\beta\) is the diffusivity constant, and \(H\) is the Laplacian of the connectome. The initial distribution of pathology \(x(0)\) is seeded with a value of 1 at a specified seed region and 0 elsewhere.

#### 5. Optimization and Evaluation
To evaluate the model's performance, we sought to minimize the sum of squared errors (SSE) between the predicted distribution and the actual PET data to find the optimal diffusion time \(t\) and seed region. ANOVA was used to assess differences in optimal diffusion time among the three diagnostic groups.

#### 6. Bootstrapping and Null Model Testing
Bootstrapping was conducted to verify differences among subgroups, and a random network null model was used to assess the significance of the structural brain network. In this process, the edges of the structural network were randomly shuffled to create a randomized network.

This study provides important insights into the mechanisms of amyloid propagation and demonstrates that the white matter structural network serves as the backbone for amyloid spread.


<br/>
# Results


이 연구에서는 알츠하이머병(AD)에서 아밀로이드 병리의 확산을 모델링하기 위해 네트워크 확산 모델을 사용하였습니다. 연구의 주요 결과는 다음과 같습니다.

1. **모델 성능**: 아밀로이드 PET 분포를 예측한 결과, 세 가지 진단 그룹(건강한 대조군(HC), 경도 인지 장애(MCI), 알츠하이머병(AD)) 모두에서 통계적으로 유의미한 적합성을 보였습니다. Pearson 상관계수는 HC에서 0.451, MCI에서 0.449, AD에서 0.471로 나타났으며, 모든 그룹에서 p-값은 0.01 미만으로 유의미했습니다.

2. **최적 확산 시간(t)**: 최적 확산 시간은 질병의 중증도에 따라 증가하는 경향을 보였습니다. HC 그룹의 최적 확산 시간은 98초, MCI는 104초, AD는 119초로 나타났습니다. 이는 아밀로이드의 확산이 질병의 진행과 함께 느려진다는 것을 시사합니다.

3. **최적 시드(seed) 지역**: 모든 진단 그룹에서 최적 시드는 뇌간(brainstem)으로 나타났습니다. 이는 아밀로이드 병리가 뇌의 특정 지역에서 시작된다는 것을 나타내며, 이전의 병리학적 연구와 일치합니다. 그러나 부트스트랩 분석에서는 HC와 MCI 그룹에서 좌우 외측 안와 전두엽(lateral orbitofrontal lobe)도 두 번째 및 세 번째로 자주 선택된 시드 지역으로 나타났습니다.

4. **부트스트랩 통계**: 1000회의 부트스트랩을 통해 각 그룹 간의 최적 확산 시간의 차이가 통계적으로 유의미하다는 것을 확인했습니다. ANOVA 분석 결과, F 통계량은 471.9로, p-값은 10^-5 미만으로 나타났습니다. 이는 질병의 중증도에 따라 확산 시간이 증가한다는 것을 다시 한번 확인해줍니다.

5. **무작위 네트워크 비교**: 무작위 네트워크 모델을 사용하여 실험을 반복한 결과, 구조적 뇌 네트워크에서 얻은 결과와 비교했을 때 무작위 네트워크의 적합성은 유의미하게 낮았습니다. Pearson 상관계수는 HC에서 0.20, MCI에서 0.21, AD에서 0.20으로 나타났으며, p-값은 10^-5 미만으로 유의미했습니다. 이는 구조적 뇌 네트워크가 아밀로이드 확산의 메커니즘을 이해하는 데 중요한 역할을 한다는 것을 시사합니다.

이러한 결과들은 아밀로이드 병리의 전파 메커니즘을 이해하는 데 중요한 통찰을 제공하며, 향후 연구에서 아밀로이드의 초기 병리학적 변화와 관련된 뇌의 구조적 연결성을 탐구하는 데 기여할 수 있습니다.

---




In this study, a network diffusion model was employed to model the spread of amyloid pathology in Alzheimer's disease (AD). The main results of the study are as follows:

1. **Model Performance**: The predictions of amyloid PET distributions showed statistically significant fits across all three diagnostic groups (Healthy Control (HC), Mild Cognitive Impairment (MCI), and Alzheimer's Disease (AD)). The Pearson correlation coefficients were 0.451 for HC, 0.449 for MCI, and 0.471 for AD, with all groups having p-values less than 0.01, indicating significant results.

2. **Optimal Diffusion Time (t)**: The optimal diffusion time showed a trend of increasing with disease severity. The optimal diffusion time for the HC group was 98 seconds, for MCI it was 104 seconds, and for AD it was 119 seconds. This suggests that the spread of amyloid slows down as the disease progresses.

3. **Optimal Seed Region**: The optimal seed identified for all diagnostic groups was the brainstem. This indicates that amyloid pathology may initiate in specific regions of the brain, aligning with previous pathological studies. However, bootstrap analysis revealed that the left and right lateral orbitofrontal lobes were also frequently chosen as secondary and tertiary seed regions in the HC and MCI groups.

4. **Bootstrap Statistics**: A bootstrapping test with 1000 iterations confirmed that the differences in optimal diffusion times among the groups were statistically significant. ANOVA analysis yielded an F-statistic of 471.9 with a p-value of less than 10^-5, reaffirming that diffusion time increases with disease severity.

5. **Random Network Comparison**: When the diffusion procedure was repeated using randomized network models, the fit performance was significantly lower compared to the structural brain networks. The Pearson correlation coefficients were 0.20 for HC, 0.21 for MCI, and 0.20 for AD, with p-values less than 10^-5. This suggests that the structural brain network plays a crucial role in understanding the mechanisms of amyloid spread.

These results provide important insights into the mechanisms of amyloid pathology propagation and may contribute to future research exploring the structural connectivity of the brain related to early pathological changes in amyloid.


<br/>
# 예제



이 논문에서는 알츠하이머병(AD)에서 아밀로이드 병리의 전파를 네트워크 확산 모델을 통해 모델링하고 분석합니다. 연구의 주요 목표는 아밀로이드의 전파 메커니즘을 이해하고, 이를 통해 질병의 진행을 예측하는 것입니다. 

#### 데이터셋
1. **트레이닝 데이터**: 
   - **입력**: 
     - DTI(확산 텐서 이미징) 스캔, T1 가중치 구조 MRI, 인구 통계 데이터
     - 세 가지 진단 그룹: 건강한 대조군(HC), 경도 인지 장애(MCI), 알츠하이머병(AD)
   - **출력**: 
     - 아밀로이드 PET(SUVR) 값의 지역별 평균

2. **테스트 데이터**: 
   - **입력**: 
     - 동일한 DTI 및 MRI 데이터
     - 아밀로이드 PET 데이터
   - **출력**: 
     - 네트워크 확산 모델을 통해 예측된 아밀로이드 분포

#### 구체적인 작업
- **모델링**: 
  - 아밀로이드의 전파를 네트워크 확산 모델을 사용하여 시뮬레이션합니다. 
  - 초기 시드(seed) 지역에서 아밀로이드가 어떻게 확산되는지를 모델링합니다.
  
- **최적화**: 
  - 모델의 두 가지 주요 매개변수(확산 시간(t) 및 초기 시드)를 조정하여 예측된 아밀로이드 분포와 실제 PET 데이터 간의 오차를 최소화합니다.
  
- **검증**: 
  - 부트스트랩 방법을 사용하여 각 진단 그룹의 최적 확산 시간과 초기 시드의 차이를 검증합니다.
  - 무작위 네트워크 모델과 비교하여 구조적 뇌 네트워크의 중요성을 평가합니다.

이러한 과정을 통해 연구자들은 아밀로이드의 전파 메커니즘을 이해하고, 알츠하이머병의 진행을 예측하는 데 기여할 수 있습니다.

---




This paper models and analyzes the propagation of amyloid pathology in Alzheimer's disease (AD) through a network diffusion model. The primary goal of the study is to understand the mechanisms of amyloid propagation and to predict the progression of the disease.

#### Dataset
1. **Training Data**: 
   - **Input**: 
     - DTI (Diffusion Tensor Imaging) scans, T1-weighted structural MRI, demographic data
     - Three diagnostic groups: Healthy Controls (HC), Mild Cognitive Impairment (MCI), Alzheimer's Disease (AD)
   - **Output**: 
     - Regional average of amyloid PET (SUVR) values

2. **Test Data**: 
   - **Input**: 
     - The same DTI and MRI data
     - Amyloid PET data
   - **Output**: 
     - Predicted amyloid distribution from the network diffusion model

#### Specific Tasks
- **Modeling**: 
  - Simulate the propagation of amyloid using a network diffusion model. 
  - Model how amyloid spreads from an initial seed region.

- **Optimization**: 
  - Adjust two key parameters of the model (diffusion time (t) and initial seed) to minimize the error between the predicted amyloid distribution and the actual PET data.

- **Validation**: 
  - Use bootstrapping methods to validate the differences in optimal diffusion time and initial seed among the diagnostic groups.
  - Compare with random network models to assess the importance of the structural brain network.

Through these processes, researchers can gain insights into the mechanisms of amyloid propagation and contribute to predicting the progression of Alzheimer's disease.

<br/>
# 요약

이 연구에서는 알츠하이머병(AD)에서 아밀로이드 병리의 확산을 모델링하기 위해 백질 구조 뇌 네트워크를 기반으로 한 네트워크 확산 모델을 설계하였다. 결과적으로, 이 모델은 건강한 대조군(HC), 경도 인지 장애(MCI), AD 환자 그룹에서 아밀로이드의 분포를 성공적으로 재현하였으며, 아밀로이드의 확산 시간은 질병의 중증도에 따라 증가하는 경향을 보였다. 최적의 시작 지점은 모든 그룹에서 뇌간으로 나타났으며, 이는 아밀로이드 병리의 전파 메커니즘에 대한 새로운 통찰을 제공한다.

---

In this study, a network diffusion model based on white matter structural brain networks was designed to model the spread of amyloid pathology in Alzheimer's disease (AD). The model successfully recaptured the distribution of amyloid in healthy controls (HC), mild cognitive impairment (MCI), and AD patient groups, showing that the diffusion time of amyloid increased with disease severity. The optimal starting point was identified as the brainstem across all groups, providing new insights into the propagation mechanism of amyloid pathology.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **주요 방법론 파이프라인**: 연구의 방법론을 시각적으로 나타내며, ADNI 데이터에서 시작하여 백질 섬유 밀도 연결망을 구축하고, 네트워크 확산 모델을 통해 병리의 확산을 시뮬레이션하는 과정을 보여줍니다. 이 과정은 초기 병리의 시드 주입, 확산 시간 최적화, 그리고 예측된 분포와 실제 PET 데이터 간의 비교를 포함합니다.
   - **PET 분포 비교**: 예측된 PET 분포와 각 진단 그룹의 평균 실제 분포를 비교하는 그래프가 포함되어 있습니다. 이 그래프는 모델이 실제 데이터와 얼마나 잘 일치하는지를 보여주며, 예측된 분포가 실제 데이터와 유사한 패턴을 보임을 나타냅니다.

2. **테이블**
   - **인구 통계학적 데이터**: ADNI GO-2 코호트의 인구 통계학적 특성을 보여주는 테이블이 포함되어 있습니다. 이 테이블은 건강한 대조군(HC), 경도 인지 장애(MCI), 알츠하이머병(AD) 그룹 간의 평균 나이, 교육 연수, 성별 분포 등을 비교합니다. 이 데이터는 각 그룹 간의 차이가 통계적으로 유의미하지 않음을 보여줍니다.
   - **네트워크 확산 결과**: 각 진단 그룹의 최적 확산 시간(t)과 상관 계수(Pearson's R)를 보여주는 테이블이 있습니다. 이 테이블은 AD의 진행 정도에 따라 확산 시간이 증가하는 경향을 나타내며, 이는 병리의 진행을 시사합니다.

3. **어펜딕스**
   - 연구의 방법론 및 데이터 처리 과정에 대한 추가적인 세부사항이 포함되어 있습니다. 여기에는 DTI 스캔의 전처리, 네트워크 구축 방법, 확산 모델의 수학적 기초 등이 설명되어 있습니다. 이러한 정보는 연구의 재현 가능성을 높이고, 다른 연구자들이 유사한 방법론을 적용할 수 있도록 돕습니다.




1. **Diagrams and Figures**
   - **Main Methodology Pipeline**: This visually represents the methodology of the study, starting from ADNI data, constructing white matter fiber density connectomes, and simulating the spread of pathology through a network diffusion model. The process includes initial seed injection, optimization of diffusion time, and comparison of predicted distributions with actual PET data.
   - **Comparison of PET Distributions**: A graph comparing predicted PET distributions with the average actual distributions for each diagnostic group. This graph illustrates how well the model aligns with real data, indicating that the predicted distributions exhibit similar patterns to the actual data.

2. **Tables**
   - **Demographic Data**: A table showing the demographic characteristics of the ADNI GO-2 cohort. It compares average age, years of education, and gender distribution across healthy controls (HC), mild cognitive impairment (MCI), and Alzheimer's disease (AD) groups. This data indicates that there are no statistically significant differences among the groups.
   - **Network Diffusion Results**: A table presenting the optimal diffusion time (t) and Pearson's correlation (R) for each diagnostic group. This table shows a trend of increasing diffusion time with the severity of AD, suggesting a progression of pathology.

3. **Appendices**
   - Additional details on the methodology and data processing are included. This encompasses the preprocessing of DTI scans, methods for network construction, and the mathematical foundations of the diffusion model. Such information enhances the reproducibility of the study and assists other researchers in applying similar methodologies.

<br/>
# refer format:


```bibtex
@inproceedings{Xu2025,
  author = {Frederick H. Xu and Duy Duong-Tran and Heng Huang and Andrew J. Saykin and Paul Thompson and Christos Davatzikos and Yize Zhao and Li Shen},
  title = {Characterizing Amyloid Pathogenic Spread in Alzheimer’s Disease Through A Network Diffusion Model},
  booktitle = {Proceedings of the 16th ACM International Conference on Bioinformatics, Computational Biology, and Health Informatics (BCB '25)},
  year = {2025},
  month = {October},
  location = {Philadelphia, PA, USA},
  publisher = {ACM},
  pages = {1--9},
  doi = {10.1145/3765612.3767223}
}
```

### 시카고 스타일

Xu, Frederick H., Duy Duong-Tran, Heng Huang, Andrew J. Saykin, Paul Thompson, Christos Davatzikos, Yize Zhao, and Li Shen. 2025. "Characterizing Amyloid Pathogenic Spread in Alzheimer’s Disease Through A Network Diffusion Model." In *Proceedings of the 16th ACM International Conference on Bioinformatics, Computational Biology, and Health Informatics (BCB '25)*, 1-9. Philadelphia, PA, USA: ACM. https://doi.org/10.1145/3765612.3767223.
