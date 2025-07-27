---
layout: post
title:  "[2025]Scaling Collapse Reveals Universal Dynamics in Compute-Optimally Trained Neural Networks"  
date:   2025-07-27 02:13:40 +0200
categories: study
---

{% highlight ruby %}


한줄 요약: 


다양한 크기의 신경망들이 적절히 정규화되면 훈련 중 loss 곡선이 하나의 보편적인 곡선으로 수렴한다는 현상(supercollapse)을 발견하고, 이를 이론적(파워 법칙 기반의 신경망 스케일링 법칙-더 많은 학습을 하고 더 큰 모델을 쓰면 loss는 감소, SGD 노이즈 다이내믹스 이론-학습률이 작아질수록 노이즈가 억제)으로 설명   


짧은 요약(abstract):   





대규모 신경망 학습의 동적 특성을 이해하는 것은 여전히 중요한 난제로 남아 있습니다. 하지만 이 논문은 **계산량에 최적화된(compute-optimal)** 방식으로 훈련된 모델들이 **공통적인 규칙적인 패턴**을 따르고 있음을 발견합니다. 다양한 크기의 모델들이 훈련 동안 경험하는 \*\*loss 곡선이 특정 방식으로 정규화되면 하나의 보편적인(universal) 곡선으로 모이게 되는 현상(collapse)\*\*이 발생합니다. 특히, 학습률 감소(learning rate decay)를 적용할 경우 이 곡선들의 차이는 랜덤 시드에 의한 개별 모델의 잡음 수준보다 작아지는 극단적으로 정밀한 수렴 현상, 즉 "**supercollapse**"가 관측됩니다. 이 현상은 트랜스포머, 다양한 데이터셋, 다양한 학습률 스케줄에 걸쳐 관측되며, 적절하지 않은 하이퍼파라미터 설정 시 이 수렴 현상이 깨집니다. 이러한 현상을 설명하기 위해, 저자들은 파워 법칙(power-law)에 기반한 신경망 스케일링 법칙과 확률적 경사 하강법(SGD) 노이즈 모델을 분석하고, 학습률 스케줄이 어떻게 loss 곡선을 변형시키는지 설명합니다. 그 결과, 이 프레임워크는 모델 간 비교뿐 아니라, 잘못된 스케일링 설정을 진단하는 도구로도 활용될 수 있습니다.

---



> Understanding neural network training dynamics at scale is an important open problem. Although realistic model architectures, optimizers, and data interact in complex ways that make predictive theory challenging, we show that compute-optimally trained models exhibit remarkably precise collective regularities. Specifically, loss curves from models of varying sizes collapse onto a single universal curve when training compute and loss are normalized to unity at the end of training. With learning rate decay, discrepancies between normalized curves fall below the noise floor of individual models’ loss curves across random seeds, yielding an exceptionally tight collapse we term “supercollapse.” We observe supercollapse across learning rate schedules, datasets, and architectures, including transformers trained on next-token prediction. This collapse breaks down when hyperparameters are scaled suboptimally, providing a practical indicator of proper scaling. We explain these phenomena by connecting collapse to the power-law structure in typical neural scaling laws, and analyzing a simple but effective model of SGD noise dynamics that accurately captures how learning rate schedules deform loss curves away from power laws while preserving universality, and why learning rate decay suppresses variance to enable supercollapse.

---





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





1. **모델 아키텍처**

   * **Transformer (트랜스포머)**: Decoder-only 구조, 깊이는 3층으로 고정하고 너비(embedding dimension)는 768부터 2048까지 확장하여 다양한 크기의 모델 실험을 수행하였습니다.
   * **MLP (다층 퍼셉트론)**: synthetic regression task를 위해 사용되었으며, 입력 특징은 power-law Fourier spectrum을 갖도록 설계되었습니다. 너비는 512\~4096으로 스케일링.

2. **학습 데이터**

   * **CIFAR-5M**: CIFAR-10과 유사한 이미지 600만 개로 구성된 데이터셋을 그레이스케일로 변환하고 flatten하여 토큰 시퀀스로 사용.
   * **Lichess**: 체스 게임의 알파벳 기보(Algebraic notation)를 기반으로 한 문자 수준의 토큰화된 텍스트 데이터셋.
   * **Synthetic Power-Law Data**: 특정 주파수 분포를 따르도록 생성된 합성 데이터로 MLP 모델 학습에 사용됨.

3. **학습 방식과 기법**

   * **Compute-Optimal Training**: 모델 크기에 따라 계산량(compute budget)을 달리하며, 파레토 최적선을 기반으로 각 모델의 최적 학습 스텝 수를 추정함.
   * **µP (mu-parameterization)**: 안정적인 스케일링을 위해 가중치 초기화 및 학습률을 모델 너비에 따라 조정하는 파라미터화 기법.
   * **Adam Optimizer** 사용, 학습률은 일정 또는 감소 스케줄(linear decay, cosine decay 등)을 실험함.
   * **Supercollapse 현상 분석**: loss 곡선을 정규화한 후 다양한 모델 크기에서 하나의 보편 곡선으로 수렴(collapse)하는 현상 분석. 특히 학습률 감소 시 variance가 억제되어 noise floor 이하로 수렴하는 supercollapse가 발생함.

---



1. **Model Architectures**

   * **Transformer**: Decoder-only transformers with a fixed depth of 3 layers and varying widths (embedding dimensions from 768 to 2048). Model sizes ranged from 12M to 79M parameters.
   * **MLPs**: Used for synthetic regression tasks. The target functions were constructed with a power-law Fourier spectrum to elicit neural scaling laws. Model widths ranged from 512 to 4096.

2. **Training Data**

   * **CIFAR-5M**: A dataset of 6 million CIFAR-like grayscale images, flattened into sequences of 1024 tokens for autoregressive prediction.
   * **Lichess**: A dataset of chess games encoded in algebraic notation, used for character-level next-token prediction.
   * **Synthetic Power-Law Data**: Constructed to follow a known power-law spectrum, used to test MLP behavior.

3. **Training Setup and Techniques**

   * **Compute-Optimal Training**: Each model was trained to a compute-optimal horizon determined by the Pareto frontier between compute and final loss.
   * **µP (mu-parameterization)**: Used to scale weight initialization and learning rates with model width, ensuring stable training across scales.
   * **Optimizer**: Adam was used across experiments with various learning rate schedules including constant and decaying (e.g., linear, cosine).
   * **Supercollapse Analysis**: Training loss curves were normalized and analyzed across model sizes. With learning rate decay, the normalized curves collapsed more tightly than the seed-level noise, exhibiting a phenomenon termed “supercollapse”.




   
 
<br/>
# Results  





1. **스케일링 콜랩스(Scaling Collapse)의 발견**

   * 서로 다른 크기의 모델들이 \*\*계산량(Compute)\*\*과 **Loss**를 정규화하면, \*\*하나의 보편적인 곡선(universal loss curve)\*\*으로 거의 정확하게 수렴함.
   * 이 현상은 Transformer, MLP 아키텍처 모두에서 나타났으며, **CIFAR-5M**, **Lichess**, **합성 회귀 데이터** 등 다양한 데이터셋에서 관측됨.

2. **슈퍼콜랩스(Supercollapse)**

   * 학습률 감소(Learning Rate Decay)를 적용하면, 정규화된 loss 곡선들 간의 차이가 \*\*개별 모델의 시드별 변동 수준(noise floor)\*\*보다 작아지는 극도로 정밀한 수렴이 발생.
   * 이 현상은 학습률이 훈련 후반에 충분히 감소할 때 주로 나타나며, 다양한 learning rate 스케줄(선형, 코사인 등)에서도 유지됨.

3. **하이퍼파라미터의 민감도 측정**

   * 하이퍼파라미터를 적절하게 스케일하지 않으면 collapse가 깨짐 → collapse는 **스케일링 적절성의 민감하고 실용적인 진단 지표**로 사용 가능.
   * 예: µP 대신 일정 학습률을 사용하거나, 데이터 지수 γ를 compute-optimal 값보다 크게 하면 collapse가 사라짐.

4. **메트릭과 시각화**

   * collapse의 정량적 평가는 두 가지 지표로 수행됨:

     * **Collapse tolerance (Δ)**: 서로 다른 모델들 간 loss 곡선의 상대적 편차
     * **Noise floor (σ)**: 같은 모델에서 시드만 바꿔서 생기는 편차
   * Δ < σ일 때 supercollapse로 정의됨 (Figure 1 참조)

---



1. **Discovery of Scaling Collapse**

   * When training compute and loss are normalized, loss curves from models of varying sizes align nearly perfectly onto a **single universal curve**.
   * This collapse is observed across different architectures (Transformers, MLPs), datasets (CIFAR-5M, Lichess), and learning rate schedules.

2. **Supercollapse Phenomenon**

   * With **learning rate decay**, the discrepancy between normalized loss curves becomes **smaller than the noise floor** of individual models across random seeds.
   * This ultra-consistent alignment, termed **“supercollapse”**, emerges prominently when the learning rate decays sufficiently near the end of training and holds across schedules like linear or cosine decay.

3. **Sensitivity to Hyperparameter Scaling**

   * Improper scaling of hyperparameters disrupts the collapse, making it a **sensitive and practical diagnostic tool** for verifying compute-optimal setups.
   * Examples include replacing µP parameterization with constant learning rates, or deviating from the compute-optimal data exponent γ.

4. **Evaluation Metrics and Visualizations**

   * Collapse quality is measured using:

     * **Collapse tolerance (Δ)**: relative deviation across normalized loss curves
     * **Noise floor (σ)**: variation across seeds for a single model
   * If Δ < σ, the regime is defined as **supercollapse** (as shown in Figure 1).




<br/>
# 예제  





1. **Transformer 모델: CIFAR-5M 데이터 (이미지 → 토큰 예측)**

   * **입력**: CIFAR-5M 데이터셋은 CIFAR-10 스타일의 이미지 600만 개로 구성됨. 각 컬러 이미지를 **그레이스케일**로 변환한 후 **32×32 픽셀**을 **flatten하여 1024 길이의 시퀀스**로 만듦.
   * **출력**: 각 시점에서 다음 픽셀의 intensity 값(0\~255)을 **다음 토큰으로 예측**하는 **next-token prediction** 과제.
   * **목표**: autoregressive 방식으로 **픽셀 순서대로 loss 최소화**.

2. **Transformer 모델: Lichess 데이터 (텍스트 → 다음 문자 예측)**

   * **입력**: 온라인 체스 게임 데이터를 \*\*알파벳 체스 표기법(Algebraic notation)\*\*으로 표현. 이를 \*\*문자 단위(character-level)\*\*로 토크나이즈하여 시퀀스를 구성.
   * **출력**: 주어진 문자열 시퀀스에서 **다음 문자를 예측**하는 next-token prediction.
   * **목표**: 언어 모델링처럼 다음 문자의 확률 분포를 예측하여 loss를 최소화.

3. **MLP 모델: 합성 회귀 데이터 (연속 입력 → 연속 출력)**

   * **입력**: 8차원 연속값 벡터 `x ∈ [-0.5, 0.5]^8`. 이 입력에 대해 Fourier 기반의 주기 함수(파워 법칙 스펙트럼을 따름)가 타겟으로 정의됨.
   * **출력**: 연속값 스칼라 `y = φ(x)` (고주파 성분 포함)
   * **목표**: **회귀 문제**로서, MLP가 이 비선형 주기 함수를 근사할 수 있도록 학습함.

---



1. **Transformer with CIFAR-5M (Image → Token Prediction)**

   * **Input**: The CIFAR-5M dataset consists of 6 million CIFAR-like RGB images. Each image is converted to grayscale and **flattened into a 1024-token sequence** (32×32 pixels).
   * **Output**: The task is **next-token prediction** of pixel intensity values (integers from 0 to 255) in raster scan order.
   * **Objective**: Autoregressively predict the next pixel value and minimize the cross-entropy loss over the entire sequence.

2. **Transformer with Lichess Dataset (Text → Next Character Prediction)**

   * **Input**: Chess games represented in **algebraic notation** are tokenized at the **character level**, producing sequences of chess move strings.
   * **Output**: Predict the **next character** in the sequence, similar to standard language modeling tasks.
   * **Objective**: Minimize the next-token prediction loss for the sequence of chess moves.

3. **MLP with Synthetic Power-Law Regression Data (Vector → Scalar Regression)**

   * **Input**: A continuous input vector `x ∈ [-0.5, 0.5]^8`. The target function is a periodic function with a **power-law Fourier spectrum**, designed to elicit scaling behavior.
   * **Output**: A scalar `y = φ(x)` representing the function output.
   * **Objective**: A **regression task**, where the MLP learns to approximate this structured nonlinear function over continuous inputs.





<br/>  
# 요약   




이 논문은 다양한 크기의 Transformer 및 MLP 모델을 CIFAR-5M, Lichess, 합성 파워법칙 데이터에 대해 compute-optimal 설정으로 훈련시키고, 정규화된 loss 곡선들이 하나의 보편 곡선으로 수렴하는 현상을 분석하였다.
그 결과, 학습률 감소가 적용된 경우 개별 모델의 잡음 수준보다 낮은 오차로 loss 곡선이 수렴하는 **supercollapse** 현상이 나타났으며, 이는 스케일링 설정이 잘못되었을 때 깨지는 민감한 지표로도 활용 가능함이 밝혀졌다.
예를 들어 CIFAR-5M에서는 그레이스케일 이미지의 픽셀 시퀀스를 입력으로 받아 다음 픽셀 값을 예측하는 과제를 수행하였고, Lichess에서는 체스 기보의 다음 문자를 예측하는 next-token prediction을 수행하였다.

---


This paper trains various-sized Transformer and MLP models on CIFAR-5M, Lichess, and synthetic power-law datasets under compute-optimal settings, analyzing how their normalized loss curves collapse onto a single universal trajectory.
The key finding is a phenomenon called **supercollapse**, where learning rate decay leads to normalized loss curves aligning with precision beyond the noise level of individual models—making it a sensitive indicator of proper scaling.
For instance, in CIFAR-5M, models predict the next pixel intensity from grayscale image sequences, while in Lichess, they predict the next character in chess game notation through next-token prediction tasks.




<br/>  
# 기타  



####  Figure 1

* **내용**: 다양한 Transformer 모델의 loss 곡선을 compute-optimal 기준으로 정규화한 뒤 시각화함.
* **인사이트**: 모든 모델이 하나의 보편 곡선으로 수렴하며, 특히 학습률 감소가 적용된 경우에는 noise floor보다도 낮은 오차를 가지는 **supercollapse** 현상이 뚜렷하게 나타남.

####  Figure 2

* **내용**: 정규화 과정에서 loss에서 빼는 기준 값 $\hat{L}$에 따라 collapse 품질이 어떻게 달라지는지를 비교함.
* **인사이트**: collapse가 제대로 일어나려면 irreducible loss $L_0$을 정확히 추정해야 하며, 다른 값을 사용하면 collapse가 깨짐 → 적절한 기준 설정이 핵심임을 강조.

####  Figure 3

* **내용**: 다양한 학습률 스케줄(예: linear decay, cosine decay 등)에서의 collapse 여부 시각화.
* **인사이트**: 스케줄이 달라도 compute-optimal 훈련 조건을 맞추면 collapse가 유지됨 → collapse는 일정한 학습 곡선 구조의 보편성(universality)을 반영함.

####  Figure 4

* **내용**: µP 대신 일정 학습률 사용, 또는 잘못된 데이터 지수 γ 사용 시 collapse가 무너지는 사례.
* **인사이트**: collapse는 단순한 성능이 아닌 **동적 학습 구조의 정합성**을 평가하는 민감한 도구가 될 수 있음.

####  Figure 5

* **내용**: loss가 power-law 형태 $L(t, p) = L_0 + t^{-\mu} + p^{-\nu}$를 따를 때 collapse가 정확히 일어남을 수식 및 시뮬레이션으로 확인.
* **인사이트**: 이론적으로도 collapse와 compute-optimal 훈련 조건이 일치함 → collapse는 단순한 경험적 현상이 아닌 수학적 근거가 있는 현상임.

####  Appendix A–F

* **내용**: 아키텍처 세부 설정, compute-optimal 조건 추정법, 학습률 스케줄에 따른 noise 모델 해석 등 상세 기술.
* **인사이트**: 실험의 재현 가능성과 이론적 해석을 높이기 위한 **정교한 분석 도구**와 **정량 평가 기준**을 제공함.

---


####  Figure 1

* **Content**: Shows normalized loss curves across various Transformer models under compute-optimal training.
* **Insight**: All curves collapse onto a **universal trajectory**, and with learning rate decay, the alignment becomes tighter than the **individual model's noise floor**, indicating **supercollapse**.

####  Figure 2

* **Content**: Compares the effect of different offset values $\hat{L}$ subtracted during normalization.
* **Insight**: The collapse only occurs when the **irreducible loss $L_0$** is correctly estimated—highlighting the importance of proper offset selection.

####  Figure 3

* **Content**: Evaluates loss curve collapse across various learning rate schedules (e.g., linear, cosine).
* **Insight**: Despite different schedules, if training is compute-optimal, collapse persists—demonstrating **universality** of loss curve structure.

####  Figure 4

* **Content**: Shows breakdown of collapse when µP is replaced with a constant learning rate or when the data exponent γ is incorrectly set.
* **Insight**: Collapse serves as a **sensitive diagnostic** for the consistency of training dynamics, not just final performance.

####  Figure 5

* **Content**: Simulates collapse using power-law loss functions $L(t, p) = L_0 + t^{-\mu} + p^{-\nu}$.
* **Insight**: Confirms that **compute-optimality mathematically implies collapse**, indicating that the phenomenon is theoretically grounded.

####  Appendix A–F

* **Content**: Includes architectural details, methods to estimate compute-optimal schedules, and theoretical analysis of noise dynamics under varying learning rate schedules.
* **Insight**: These sections ensure **experimental reproducibility** and provide a **rigorous theoretical foundation** for the observed phenomena.




<br/>
# refer format:     



@inproceedings{qiu2025scaling,
  title     = {Scaling Collapse Reveals Universal Dynamics in Compute-Optimally Trained Neural Networks},
  author    = {Shikai Qiu and Lechao Xiao and Andrew Gordon Wilson and Jeffrey Pennington and Atish Agarwala},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025},
  volume    = {267},
  publisher = {PMLR},
  url       = {https://github.com/shikaiqiu/supercollapse.git}
}





Qiu, Shikai, Lechao Xiao, Andrew Gordon Wilson, Jeffrey Pennington, and Atish Agarwala.
“Scaling Collapse Reveals Universal Dynamics in Compute-Optimally Trained Neural Networks.”
In Proceedings of the 42nd International Conference on Machine Learning (ICML), Vol. 267, 2025. PMLR.
Available at: https://github.com/shikaiqiu/supercollapse.git


