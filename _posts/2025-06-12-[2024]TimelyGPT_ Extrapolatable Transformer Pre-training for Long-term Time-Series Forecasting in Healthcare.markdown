---
layout: post
title:  "[2024]TimelyGPT: Extrapolatable Transformer Pre-training for Long-term Time-Series Forecasting in Healthcare"  
date:   2025-06-12 01:45:40 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 


xPos 임베딩(추세(trend)와 주기성(periodic patterns)을 캡처할 수 있도록 설계된 위치 임베딩), Retention 모듈(Transformer의 self-attention을 대체하여 전역 시계열 의존성을 모델링 (Recurrent 구조)), Temporal Convolution 모듈(지역적(local) 상호작용을 포착하는 1D depth-wise separable convolution)을 통해 시계열 테스크 수행 가능한 헬스케어 시계열 모델 구현  




짧은 요약(Abstract) :    





최근 BERT나 GPT 같은 대규모 사전학습 모델(PTM)이 자연어처리(NLP)와 컴퓨터 비전(CV)에서 큰 성과를 거두었지만, 의료 시계열 데이터에 대한 PTM 개발은 상대적으로 뒤처져 있습니다. 이 논문에서는 이를 해결하기 위해 **TimelyGPT**라는 새로운 Transformer 기반의 사전학습 모델을 제안합니다. 이 모델은 \*\*xPos(Extrapolatable Position Embedding)\*\*을 활용해 시간 시계열 데이터의 \*\*추세(trend)\*\*와 \*\*주기성(periodic pattern)\*\*을 효과적으로 표현하며, **Recurrent Attention**과 **Temporal Convolution** 모듈을 결합하여 장기적인 시간 의존성과 지역적인 상호작용을 모두 포착할 수 있습니다. Sleep-EDF와 PopHR이라는 대규모 헬스케어 시계열 데이터셋에서 TimelyGPT는 짧은 입력(2000 timestep)만으로도 최대 6000 timestep의 예측을 정확하게 수행하며, 초기 진단 정보로부터 향후 질병을 예측하는 등 **장기적인 예측과 불규칙 시계열 처리**에서 뛰어난 성능을 입증합니다. 이 모델은 환자 건강 예측 및 위험 추적에 유용하게 활용될 수 있습니다.



> **Motivation**: Large-scale pre-trained models (PTMs) such as BERT and GPT have recently achieved great success in Natural Language Processing and Computer Vision domains. However, the development of PTMs on healthcare time-series data is lagging behind. This underscores the limitations of the existing transformer-based architectures, particularly their scalability to handle large-scale time series and ability to capture long-term temporal dependencies.
> **Methods**: In this study, we present Timely Generative Pre-trained Transformer (TimelyGPT). TimelyGPT employs an extrapolatable position (xPos) embedding to encode trend and periodic patterns into time-series representations. It also integrates recurrent attention and temporal convolution modules to effectively capture global-local temporal dependencies.
> **Materials**: We evaluated TimelyGPT on two large-scale healthcare time series datasets corresponding to continuous biosignals and irregularly-sampled time series, respectively: (1) the Sleep EDF dataset consisting of over 1.2 billion timesteps; (2) the longitudinal healthcare administrative database PopHR, comprising 489,000 patients randomly sampled from the Montreal population.
> **Results**: In forecasting continuous biosignals, TimelyGPT achieves accurate extrapolation up to 6,000 timesteps of body temperature during the sleep stage transition, given a short look-up window (i.e., prompt) containing only 2,000 timesteps. For irregularly-sampled time series, TimelyGPT with a proposed time-specific inference demonstrates high top recall scores in predicting future diagnoses using early diagnostic records, effectively handling irregular intervals between clinical records.
> **Conclusion**: Together, we envision TimelyGPT to be useful in various health domains, including long-term patient health state forecasting and patient risk trajectory prediction.
> **Availability**: The open-sourced code is available at Github.






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





**1. 전체 아키텍처 개요**

TimelyGPT는 긴 시계열 데이터를 다룰 수 있도록 설계된 **생성형 사전학습 Transformer**입니다. 구조적으로는 다음 3가지 핵심 모듈로 구성되어 있습니다:

* **xPos 임베딩**: 추세(trend)와 주기성(periodic patterns)을 캡처할 수 있도록 설계된 위치 임베딩
* **Retention 모듈**: Transformer의 self-attention을 대체하여 **전역 시계열 의존성**을 모델링 (Recurrent 구조 포함)
* **Temporal Convolution 모듈**: **지역적(local)** 상호작용을 포착하는 1D depth-wise separable convolution

---

**2. xPos (Extrapolatable Position Embedding)**

* 기존 Transformer 위치 임베딩의 한계를 극복하기 위해 **xPos**를 도입.
* 두 토큰 $n, m$ 간 거리 $n - m$를 기반으로 다음과 같이 임베딩에 \*\*회전(rotation)\*\*과 \*\*지수 감쇠(exponential decay)\*\*를 적용함:

$$
\tilde{Q}_n \tilde{K}_m = \gamma^{n-m} \hat{Q}_n \hat{K}_m
$$

$$
\hat{Q}_n = X_n W_Q e^{i\theta_n}, \quad \hat{K}_m = X_m W_K e^{-i\theta_m}
$$

* 여기서 $\gamma^{n-m}$: 과거 정보의 영향력을 점차 감소시키는 decay factor
* $e^{i\theta}$: 주기성 캡처 (Fourier 기반)

---

**3. Retention (RNN 기반 Attention 모듈)**

* Retention은 기존 Attention과 달리 **선형 시간 복잡도**와 **순차적 정보 처리**를 모두 만족시킴.
* 시간 차이를 반영하여 decay를 적용하는 방식:

$$
D_{nm} =
\begin{cases}
\gamma^{n-m} & n \geq m \\
0 & n < m
\end{cases}
\quad \Rightarrow \quad
\text{Retention}(X) = (\hat{Q} \hat{K}^\top \odot D) V
$$

* 불규칙 시계열 처리 시, 시간 간격 $\Delta t$을 decay factor에 직접 반영:

$$
S_n = \gamma^{\Delta t_{n,n-1}} S_{n-1} + K_n^\top V_n, \quad \text{Retention}(X_n) = Q_n S_n
$$

---

**4. Temporal Convolution 모듈**

* 시계열 내 지역적 상호작용을 캡처하기 위해 **1D Convolution Tokenizer** 사용.
* 두 개의 kernel size 3, stride 2인 depthwise separable convolution을 사용해 시퀀스를 1/4 길이로 줄임.
* 이후 normalization + activation(Swish) + pointwise convolution으로 정제함.

---

**5. 학습 방식**

* **Pre-training**: unlabeled biosignal 데이터 또는 진단 코드를 기반으로 **next-token prediction** 방식 학습

  * 연속값(MSE)과 범주값(Cross Entropy) 둘 다 적용
* **Fine-tuning**: 다운스트림 예측 태스크에 맞게 전체 모델을 end-to-end 방식으로 미세조정

---

**1. Overview**

TimelyGPT is a **generative Transformer model** designed for long-range time-series forecasting. It is composed of three key components:

* **Extrapolatable position embedding (xPos)** to encode trend and periodicity
* **Retention module** to model global dependencies using recurrent-like attention
* **Temporal convolution module** to capture local interactions in time

---

**2. xPos Embedding**

To capture trends and seasonal patterns, xPos embeds the relative position between tokens using both **rotational encoding** and **exponential decay**:

$$
\tilde{Q}_n \tilde{K}_m = \gamma^{n-m} \hat{Q}_n \hat{K}_m
$$

$$
\hat{Q}_n = X_n W_Q e^{i\theta_n}, \quad \hat{K}_m = X_m W_K e^{-i\theta_m}
$$

* $\gamma^{n-m}$: decay factor to attenuate influence from distant tokens
* $e^{i\theta}$: captures periodicity (oscillations)

---

**3. Retention Module**

The Retention mechanism serves as a **recurrent alternative** to self-attention with **linear complexity**. It uses chunk-wise processing and is reformulated as:

$$
D_{nm} =
\begin{cases}
\gamma^{n-m} & n \geq m \\
0 & n < m
\end{cases}
\quad \Rightarrow \quad
\text{Retention}(X) = (\hat{Q} \hat{K}^\top \odot D) V
$$

For irregular time intervals, time gaps $\Delta t$ are explicitly used:

$$
S_n = \gamma^{\Delta t_{n,n-1}} S_{n-1} + K_n^\top V_n, \quad \text{Retention}(X_n) = Q_n S_n
$$

---

**4. Temporal Convolution Module**

To model local dependencies, a **1D convolutional tokenizer** is applied:

* Two 1D conv layers with kernel=3 and stride=2 (reduces sequence length to 1/4)
* Followed by batch normalization, Swish activation, and pointwise convolution

This captures multi-scale patterns when stacked over decoder layers.

---

**5. Training Strategy**

* **Pre-training**: next-token prediction task on unlabeled data (MSE for continuous, CE for categorical)
* **Fine-tuning**: end-to-end tuning on task-specific forecasting datasets (biosignals, diagnoses)


   
 
<br/>
# Results  


	
**1. 사용된 테스트 데이터셋**

* **Sleep-EDF**:

  * 1.2 billion timestep 규모의 **연속적인 생체 신호 데이터** (EEG, EOG, EMG, 체온 등 7가지 바이오신호 포함)
  * 2,000 timestep의 입력으로 최대 **6,000 timestep 예측** 수행

* **PopHR**:

  * 캐나다 몬트리올 인구 기반의 **불규칙 간격 EHR(전자건강기록)** 데이터
  * 489,000명 환자에 대한 **ICD-9 기반 진단 코드(PheCodes)** 예측 태스크

---

**2. 비교된 경쟁 아키텍처**

* **연속 시계열 예측**:

  * Informer, Autoformer, Fedformer, PatchTST, TimesNet, DLinear, TS2Vec 등
* **불규칙 시계열 예측**:

  * mTAND, SeFT (irregular time series 특화 모델)

---

**3. 주요 태스크**

* **연속 바이오신호 예측 (Sleep-EDF)**:

  * MAE (Mean Absolute Error) 및 Cross-Correlation을 평가 지표로 사용
  * forecasting window: 720 / 2000 / 6000 timesteps
* **불규칙 진단 코드 예측 (PopHR)**:

  * Top-K Recall\@5, 10, 15 (정답 코드가 상위 K개 예측에 포함될 확률)

---

**4. 주요 결과 요약**

* **Sleep-EDF (연속 시계열)**:

  * TimelyGPT는 긴 시계열(6,000 timestep)까지도 정확하게 예측
  * PatchTST는 720에서는 MAE가 가장 낮았지만, 긴 예측에서는 급격히 성능 저하
  * TimelyGPT는 2000, 6000 window 모두에서 **MAE 및 상관계수 최고 성능**
  * MAE 예시:

    * 2000 window: TimelyGPT = 0.567, PatchTST = 0.768, Informer = 1.013

* **PopHR (불규칙 시계열)**:

  * TimelyGPT(time-specific inference)가 Recall\@5 = 58.65%, @10 = 70.83%로 최고
  * Trajectory-based 방식보다 time-specific 방식이 **시계열 불규칙성에 강건**
  * mTAND는 일부 구간에서 유사 성능을 보였지만 장기 예측에선 성능 하락

---

**1. Datasets Used**

* **Sleep-EDF**:

  * Large-scale continuous biosignal dataset with 1.2 billion timesteps
  * Includes EEG, EOG, EMG, temperature, etc.
  * Forecasting task: up to 6,000 timesteps based on 2,000-timestep input

* **PopHR**:

  * Irregularly-sampled EHR dataset (phenotype-level diagnosis codes from 489,000 patients)
  * Converted from ICD-9 to PheCodes
  * Long-term prediction of diagnostic trajectories

---

**2. Baselines Compared**

* **Continuous signals**: Informer, Autoformer, Fedformer, PatchTST, TimesNet, DLinear, TS2Vec
* **Irregular signals**: mTAND, SeFT (designed for irregular time series)

---

**3. Tasks and Metrics**

* **Task 1: Continuous biosignal forecasting**

  * Metrics: Mean Absolute Error (MAE), Cross-Correlation
  * Forecast windows: 720, 2000, 6000 timesteps

* **Task 2: Diagnostic code forecasting**

  * Metrics: Top-K Recall\@5, @10, @15
  * Assesses whether correct diagnosis appears in top K predicted codes

---

**4. Main Results**

* **Sleep-EDF Results**:

  * TimelyGPT outperformed all baselines in long-range forecasting (2000, 6000 steps)
  * PatchTST performed best for 720-step forecast but degraded beyond training length
  * Example MAE at 2000-timestep forecast:

    * TimelyGPT: **0.567**
    * PatchTST: 0.768
    * Informer: 1.013

* **PopHR Results**:

  * TimelyGPT (time-specific inference) achieved best recall:

    * Recall\@5 = **58.65%**, Recall\@10 = **70.83%**
  * Time-specific inference was better than trajectory-based inference, especially in long-term irregular forecasting
  * mTAND followed closely at short horizons but fell off on long-term predictions





<br/>
# 예제  




**1. Sleep-EDF 바이오신호 예측 시각화**

* 입력: 2,000 timestep 동안 측정된 **직장 체온**
* 목표: 이후 6,000 timestep까지의 체온 예측
* 실제 체온은 **약간 하강 후 재상승**하는 추세(trend)를 가짐
* 결과:

  * TimelyGPT는 **소폭 하강 후 상승하는 추세를 정확히 포착**
  * 특히 7,000 timestep 부근의 체온 상승을 유일하게 예측함
  * 다른 모델 (GPT, PatchTST, DLinear 등)은 이 상승을 놓침
* **의미**: xPos 기반의 추세 인코딩이 유효함을 시각적으로 입증

---

**2. PopHR 환자 건강 trajectory 예시**

* 사례: **신장 질환과 종양 관련 질병**을 가진 실제 환자
* 입력: 50개 진단 코드 시퀀스 (PheCodes)
* 목표: 이후 100 timestep 동안 발생할 가능성이 높은 질병 예측
* 결과:

  * TimelyGPT는 Pyelonephritis (PheCode 590.0)를 **3회 정확히 예측**
  * 과거 기록에 존재했던 질병(PheCode 740.9)은 다시 발생할 것이라 예측하여 적중
  * Recall\@5가 \*\*85.7%\*\*로 매우 높음
* **의미**: TimelyGPT는 시간 간격(Δt)을 반영한 **time-specific 예측**을 통해 불규칙한 진단 코드 예측에도 강력함을 보여줌

---

**1. Forecasting Biosignals in Sleep-EDF**

* **Input**: 2,000 timesteps of **rectal body temperature** during sleep
* **Goal**: Forecast up to 6,000 future timesteps
* **True Pattern**: Slight dip followed by a noticeable upward trend around timestep 7,000
* **Prediction Outcome**:

  * **TimelyGPT** captured the rising temperature accurately
  * **Other models** (PatchTST, DLinear, GPT) failed to capture the upward trend
  * Visual comparison shows TimelyGPT’s curve aligned with the red groundtruth line
* **Insight**: Demonstrates TimelyGPT’s extrapolation ability using xPos-encoded trends

---

**2. Case Study: Diagnostic Trajectory of a Patient in PopHR**

* **Patient**: Diagnosed with **neoplasm and genitourinary conditions**
* **Input**: Sequence of 50 historical diagnosis codes (PheCodes)
* **Prediction Window**: Next 100 timesteps
* **Prediction Outcome**:

  * TimelyGPT predicted **Pyelonephritis (PheCode 590.0)** three times accurately around age 61
  * Also predicted **PheCode 740.9**, which had appeared earlier in the patient’s history
  * Achieved **Top-5 Recall of 85.7%** for this patient
* **Insight**: Shows that TimelyGPT can utilize past diagnostic patterns and irregular time intervals for precise health trajectory forecasting





<br/>  
# 요약   



TimelyGPT는 긴 시계열 예측을 위해 xPos 임베딩, Retention 모듈, Temporal Convolution을 결합한 Transformer 기반 사전학습 모델이다. 이 모델은 Sleep-EDF와 PopHR 데이터셋에서 기존 Transformer 계열 모델보다 뛰어난 장기 예측 성능을 보였으며, 특히 불규칙 시계열에서도 높은 정확도를 유지했다. 실제 체온 예측과 진단 코드 예측 예시에서 TimelyGPT는 뚜렷한 추세와 주기성을 정확히 포착하여 우수한 결과를 도출하였다.



TimelyGPT is a Transformer-based pre-trained model that combines xPos embeddings, Retention modules, and Temporal Convolution for long-term time-series forecasting. It outperformed prior Transformer models on both continuous (Sleep-EDF) and irregular (PopHR) healthcare datasets, maintaining high accuracy in long-horizon predictions. In concrete examples, TimelyGPT accurately captured trend shifts in body temperature and future diagnoses, demonstrating strong extrapolation and robustness to irregular sampling.



<br/>  
# 기타  



####  **Figure 1: TimelyGPT 아키텍처 개요**

* a. 전체 구조: Conv-subsampling tokenizer → L개의 decoder layer
* b. xPos가 주기성과 추세를 임베딩에 반영
* c. Chunk-wise Retention: 긴 시계열을 효율적으로 처리
* d. Temporal Convolution: 지역적 상호작용 감지

**➡ 인사이트**: TimelyGPT는 전역/지역적 시간 패턴을 동시에 잡을 수 있는 하이브리드 구조

---

####  **Figure 4: MAE 및 상관계수 비교 (Sleep-EDF)**

* 8개 모델 대상 3가지 forecast window (720, 2000, 6000)
* TimelyGPT는 **MAE, Cross-Correlation 모두에서 상위 성능**
* 긴 예측 구간(6000)일수록 타 모델 대비 격차가 커짐

**➡ 인사이트**: TimelyGPT는 학습 길이를 초과한 extrapolation에서도 가장 안정적인 성능을 보임

---

####  **Figure 5: 6,000 timestep 예측 시각화**

* 체온 예측 곡선 비교 (Groundtruth vs. TimelyGPT vs. PatchTST 등)
* TimelyGPT만이 **7,000 timestep 부근의 온도 상승**을 포착

**➡ 인사이트**: xPos 기반의 장기 추세 학습 능력이 실제 예측 정확도에 기여

---

####  **Figure 6: Top-5 진단 예측 Recall 분포 (PopHR)**

* TimelyGPT(time-specific) vs PatchTST vs mTAND
* 예측 구간이 길어질수록 TimelyGPT의 우세가 뚜렷

**➡ 인사이트**: 시간 차(decay)를 반영한 Retention이 irregular data에 유리

---

####  **Table 1: Sleep-EDF 성능 비교 (MAE, 상관계수)**

| Model     | MAE (2000) | MAE (6000) | Corr (2000) |
| --------- | ---------- | ---------- | ----------- |
| TimelyGPT | **0.567**  | **0.575**  | **0.628**   |
| PatchTST  | 0.768      | 0.824      | 0.512       |
| Informer  | 1.013      | 1.256      | 0.256       |

**➡ 인사이트**: 일반적인 Transformer보다 훨씬 더 정확하고 안정적인 장기 예측 가능

---

####  **Table 2: PopHR 진단 예측 Recall\@K**

| Model             | Recall\@5 | Recall\@10 | Recall\@15 |
| ----------------- | --------- | ---------- | ---------- |
| TimelyGPT (time)  | **58.65** | **70.83**  | 82.69      |
| TimelyGPT (traj.) | 52.30     | 64.35      | 77.12      |
| PatchTST          | 48.17     | 65.55      | 73.31      |

**➡ 인사이트**: TimelyGPT는 정확도뿐 아니라 다양한 K값에서도 안정적인 성능 유지

---

####  **Table 3: Ablation Study**

* RoPE 제거 시 MAE 급등: 1.072 → xPos의 중요성 확인
* Pre-training 제거 시 성능 하락 → 사전학습 효과 입증

**➡ 인사이트**: xPos + Pre-training + Convolution의 조합이 전체 성능에 핵심적으로 작용

---

#### **Figure 1: TimelyGPT Architecture Overview**

* a. Full pipeline: convolutional tokenizer → L decoder layers
* b. xPos embeds periodic and trend signals
* c. Chunk-wise Retention for long-sequence efficiency
* d. Temporal convolution for local interactions

**➡ Insight**: Combines global and local modeling for robust forecasting

---

####  **Figure 4: MAE and Cross-Correlation vs. Forecast Window (Sleep-EDF)**

* Compared 8 models over 3 windows: 720, 2000, 6000 timesteps
* TimelyGPT had lowest MAE and highest correlation across all
* Performance gap widened at longer horizons

**➡ Insight**: TimelyGPT handles extrapolation beyond training length better than others

---

####  **Figure 5: 6,000-timestep Forecast Visualization**

* Plots rectal temperature: TimelyGPT vs PatchTST vs others
* TimelyGPT alone captured the late-stage upward trend (\~t=7000)

**➡ Insight**: Strong trend modeling with xPos aids accurate long-horizon forecasts

---

####  **Figure 6: Top-5 Recall Distribution for Diagnosis Prediction**

* Compared time-specific vs. trajectory-based TimelyGPT, PatchTST, mTAND
* Time-specific inference showed superior and consistent performance over longer windows

**➡ Insight**: Retention with time-decay is well-suited for irregular clinical data

---

####  **Table 1: Sleep-EDF Model Comparison**

| Model     | MAE (2000) | MAE (6000) | Corr (2000) |
| --------- | ---------- | ---------- | ----------- |
| TimelyGPT | **0.567**  | **0.575**  | **0.628**   |
| PatchTST  | 0.768      | 0.824      | 0.512       |

**➡ Insight**: TimelyGPT delivers top-tier performance even on very long prediction horizons

---

####  **Table 2: PopHR Diagnosis Prediction Results**

| Model             | Recall\@5 | Recall\@10 | Recall\@15 |
| ----------------- | --------- | ---------- | ---------- |
| TimelyGPT (time)  | **58.65** | **70.83**  | 82.69      |
| TimelyGPT (traj.) | 52.30     | 64.35      | 77.12      |
| PatchTST          | 48.17     | 65.55      | 73.31      |

**➡ Insight**: Time-specific inference excels in handling irregular time gaps in clinical trajectories

---

####  **Table 3: Ablation Study**

* Removing RoPE increases MAE to 1.072 → validates xPos’s importance
* Without pre-training, performance drops significantly

**➡ Insight**: All three components—xPos, pre-training, convolution—are crucial for strong results




<br/>
# refer format:     


@inproceedings{song2024timelygpt,
  author    = {Ziyang Song and Qincheng Lu and Hao Xu and He Zhu and David Buckeridge and Yue Li},
  title     = {TimelyGPT: Extrapolatable Transformer Pre-training for Long-term Time-Series Forecasting in Healthcare},
  booktitle = {Proceedings of the 15th ACM Conference on Bioinformatics, Computational Biology, and Health Informatics (ACM BCB ’24)},
  year      = {2024},
  pages     = {1--16},
  publisher = {ACM},
  address   = {Shenzhen, Guangdong, PR China},
  doi       = {10.1145/3698587.3701364},
  url       = {https://doi.org/10.1145/3698587.3701364}
}




Song, Ziyang, Qincheng Lu, Hao Xu, He Zhu, David Buckeridge, and Yue Li. 2024. “TimelyGPT: Extrapolatable Transformer Pre-training for Long-term Time-Series Forecasting in Healthcare.” In Proceedings of the 15th ACM Conference on Bioinformatics, Computational Biology, and Health Informatics (ACM BCB ’24), Shenzhen, Guangdong, PR China, November 22–25. https://doi.org/10.1145/3698587.3701364.  



