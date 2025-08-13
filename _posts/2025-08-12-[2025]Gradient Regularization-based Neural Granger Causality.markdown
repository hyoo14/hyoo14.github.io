---
layout: post
title:  "[2025]Gradient Regularization-based Neural Granger Causality"  
date:   2025-08-12 07:38:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 


시계열 예측 (forecasting), 예측 모델의 gradient를 이용해 인과성 추론, 희소성(L1 정규화)을 적용해 인과성 네트워크 구조 복원을 위한 Gradient Regularization-based Neural Granger Causality (GRNGC) 제안(모든 시계열 변수를 하나의 예측 모델로 처리, Kolmogorov-Arnold Network (KAN) 사용(다른 MLP등도 가능), 모델의 **입력과 출력 간 gradient**를 계산하여 변수 간 인과성을 파악, L1 정규화**를 적용하여 불필요한 연결을 제거하고 해석 가능성을 높임)




짧은 요약(Abstract) :    



이 논문은 기존 신경망 기반 Granger 인과성(Granger causality) 모델들이 가진 한계를 해결하기 위해 \*\*Gradient Regularization-based Neural Granger Causality (GRNGC)\*\*라는 새로운 방법을 제안합니다. 기존 방법들은 주로 각 시계열 변수마다 별도의 모델을 학습해야 하는 *component-wise* 구조를 사용해 연산량이 많고, 첫 번째 계층 가중치에 희소성 패널티를 부여해 인과성을 추론하는데, 이는 복잡한 상호작용을 포착하는 데 한계가 있습니다.
GRNGC는 단일 예측 모델만으로 모든 시계열 변수를 처리하고, 모델 입력과 출력 간의 **gradient**에 L1 정규화를 적용해 인과성을 추론합니다. 이 방법은 특정 예측 모델에 종속되지 않고 KAN, MLP, LSTM 등 다양한 구조에 적용 가능해 유연성이 높습니다. DREAM, Lorenz-96, fMRI BOLD, CausalTime 등의 시뮬레이션 데이터셋에서 기존 기법보다 우수한 성능을 보였으며, DNA, Yeast, HeLa, 방광 요로 상피암(bladder urothelial carcinoma)과 같은 실제 생물학 데이터에서도 유전자 조절 네트워크 복원 성능을 입증했습니다.

---

This paper introduces **Gradient Regularization-based Neural Granger Causality (GRNGC)** to address the limitations of existing neural network-based Granger causality models. Traditional approaches often adopt a *component-wise* architecture that requires training a separate model for each time series, leading to high computational costs, and they impose sparsity-inducing penalties on the first-layer weights to infer causality, which limits the ability to capture complex interactions.
GRNGC uses only a single prediction model for all time series variables and applies **L1 regularization** to the gradient between the model’s inputs and outputs to infer Granger causality. The method is model-agnostic, allowing integration with various forecasting architectures such as KAN, MLP, and LSTM, offering high flexibility. Numerical simulations on datasets like DREAM, Lorenz-96, fMRI BOLD, and CausalTime show that GRNGC outperforms existing baselines while significantly reducing computational overhead. Furthermore, experiments on real-world datasets—DNA, Yeast, HeLa, and bladder urothelial carcinoma—demonstrate its effectiveness in reconstructing gene regulatory networks.




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





이 논문에서 제안하는 \*\*GRNGC(Gradient Regularization-based Neural Granger Causality)\*\*는 다음과 같은 핵심 설계 요소로 구성됩니다.

1. **단일 모델 기반 구조**

   * 기존 *component-wise* 구조와 달리, 모든 시계열 변수를 하나의 예측 모델로 처리합니다.
   * 이렇게 하면 변수마다 별도의 모델을 학습할 필요가 없어 계산량이 크게 줄어듭니다.

2. **예측 모델 아키텍처**

   * 기본적으로 \*\*Kolmogorov-Arnold Network (KAN)\*\*를 사용하지만, MLP, LSTM, TCN 등 다양한 예측 모델로 대체 가능합니다.
   * KAN은 Kolmogorov–Arnold 표현 정리에 기반해 다변수 함수를 단일 변수 함수들의 합으로 표현하며, 이를 확장해 깊고 넓은 신경망 구조로 적용했습니다.

3. **Granger 인과성 추론 방식**

   * 기존 방식처럼 1계층 가중치만 보는 대신, 모델의 **입력과 출력 간 gradient**를 계산하여 변수 간 인과성을 파악합니다.
   * 각 타임스텝 예측값을 합한 스칼라 함수에 대해 입력 시계열 변수별 gradient를 구하고, 이를 시간 평균하여 Granger 인과성 행렬을 생성합니다.

4. **L1 정규화 기반 희소성 부여**

   * 계산된 gradient 기반 인과성 행렬에 **L1 정규화**를 적용하여 불필요한 연결을 제거하고 해석 가능성을 높입니다.
   * 최종 손실 함수는 예측 손실(`Lp`)과 희소성 손실(`Ls`)을 합한 형태로 학습됩니다.

5. **데이터 및 학습**

   * 모델은 DREAM, Lorenz-96, fMRI BOLD, CausalTime 등의 시뮬레이션 데이터뿐만 아니라 DNA, Yeast, HeLa, 방광 요로 상피암 유전자 발현 데이터에 적용되었습니다.
   * 학습 시 각 데이터셋 특성에 맞춘 하이퍼파라미터 튜닝을 수행했고, 일부 경우에는 KAN 대신 MLP/LSTM/TCN을 사용해 성능 비교도 진행했습니다.

---



The proposed **GRNGC (Gradient Regularization-based Neural Granger Causality)** consists of the following key components:

1. **Single-Model Architecture**

   * Unlike the traditional *component-wise* architecture, GRNGC uses a single prediction model to handle all time series variables.
   * This eliminates the need to train separate models for each variable, significantly reducing computational cost.

2. **Prediction Model Architecture**

   * The default choice is the **Kolmogorov-Arnold Network (KAN)**, but it can be replaced with MLP, LSTM, or TCN.
   * KAN is based on the Kolmogorov–Arnold representation theorem, which expresses multivariate functions as sums of univariate functions, adapted here into a deep and wide neural network suitable for time series forecasting.

3. **Granger Causality Inference**

   * Instead of relying solely on first-layer weights, GRNGC computes the **gradient between the model’s inputs and outputs** to infer variable dependencies.
   * It sums the predicted values across time into a scalar function, calculates the gradient with respect to each input variable, and averages over time to form the Granger causality matrix.

4. **L1 Regularization for Sparsity**

   * An **L1 regularization** term is applied to the gradient-based causality matrix to remove irrelevant connections and enhance interpretability.
   * The final loss function combines the prediction loss (`Lp`) with the sparsity-inducing loss (`Ls`).

5. **Data and Training**

   * GRNGC is evaluated on both simulated datasets (DREAM, Lorenz-96, fMRI BOLD, CausalTime) and real-world gene expression datasets (DNA, Yeast, HeLa, bladder urothelial carcinoma).
   * Hyperparameters are tuned for each dataset, and in some cases, KAN is replaced with MLP, LSTM, or TCN to assess robustness across different forecasting architectures.



   
 
<br/>
# Results  





이 논문에서 제안한 **GRNGC**는 다양한 시뮬레이션 데이터와 실제 생물학 데이터에서 기존 최신 기법들과 비교 평가되었습니다.

1. **경쟁 모델**

   * 비교 대상에는 cMLP, cLSTM, TCDF, eSRU, GVAR, PCMCI, JGC, CR-VAE, CUTS+ 등 비선형 시계열 Granger 인과성 추론 모델과, 일부 경우에는 유전자 조절 네트워크 전용 기법(GENIE3, NIMEFI, TIGRESS, NARROMI, CVP 등)이 포함되었습니다.

2. **테스트 데이터셋**

   * **시뮬레이션 데이터**: DREAM3/4 (유전자 네트워크 시뮬레이션), Lorenz-96 (비선형 동역학), fMRI BOLD (뇌 연결성), CausalTime (AQI, Traffic, Medical 시계열).
   * **실제 데이터**: SOS DNA 복구 네트워크, Yeast 세포 주기, HeLa 세포, 방광 요로 상피암(BLCA) 유전자 발현 데이터.

3. **평가 지표**

   * **AUROC** (Area Under ROC Curve)와 **AUPRC** (Area Under Precision-Recall Curve)를 사용.
   * 유전자 네트워크 데이터의 경우 자기-인과(self-causality)는 제외하고 평가.

4. **비교 결과 요약**

   * **DREAM3/4**: 대부분의 서브데이터셋에서 AUROC 최고 성능 달성. 특히 DREAM4에서는 5개 전부 1위를 기록.
   * **Lorenz-96**: 복잡도가 높아질수록(F=20, 40)에도 최고 AUROC·AUPRC 유지, 혼돈 상태에서도 강인함 입증.
   * **fMRI BOLD**: AUROC는 JGC보다 약간 낮지만 AUPRC 최고, 전반적으로 안정적 성능.
   * **CausalTime**: Traffic·Medical 데이터에서 모든 모델 중 최고, Medical에서는 유일하게 AUROC/AUPRC 모두 0.9 이상.
   * **실제 유전자 네트워크**: SOS, Yeast, HeLa, BLCA 모두에서 기존 특화 모델보다 높은 AUROC·AUPRC를 기록하거나 동등 성능. 특히 Yeast·BLCA에서 2위 모델 대비 AUROC 약 10%p 우위.

이 결과는 GRNGC가 **계산 효율성**, **복잡한 비선형 상호작용 처리 능력**, **실제 데이터 적용성** 모두에서 경쟁 모델을 능가함을 보여줍니다.

---



The proposed **GRNGC** was benchmarked against state-of-the-art models on both simulated and real-world datasets.

1. **Competing Models**

   * Comparisons were made with neural Granger causality methods such as cMLP, cLSTM, TCDF, eSRU, GVAR, PCMCI, JGC, CR-VAE, and CUTS+, as well as gene regulatory network inference methods (GENIE3, NIMEFI, TIGRESS, NARROMI, CVP, etc.) in biological datasets.

2. **Test Datasets**

   * **Simulated datasets**: DREAM3/4 (gene network simulations), Lorenz-96 (nonlinear dynamics), fMRI BOLD (brain connectivity), and CausalTime (AQI, Traffic, Medical time series).
   * **Real datasets**: SOS DNA repair network, Yeast cell cycle, HeLa cell cycle, and Bladder Urothelial Carcinoma (BLCA) gene expression profiles.

3. **Evaluation Metrics**

   * **AUROC** (Area Under the ROC Curve) and **AUPRC** (Area Under the Precision-Recall Curve) were used.
   * For gene network datasets, self-causality was excluded from evaluation.

4. **Key Findings**

   * **DREAM3/4**: Achieved the highest AUROC in most sub-datasets; in DREAM4, ranked first in all five sub-datasets.
   * **Lorenz-96**: Maintained top AUROC and AUPRC even at higher complexity levels (F=20, 40), demonstrating robustness under chaotic dynamics.
   * **fMRI BOLD**: Slightly lower AUROC than JGC but highest AUPRC, showing balanced and stable performance.
   * **CausalTime**: Best performance on Traffic and Medical datasets; in Medical, GRNGC was the only model with both AUROC and AUPRC above 0.9.
   * **Real gene regulatory networks**: Outperformed or matched specialized methods in SOS, Yeast, HeLa, and BLCA datasets. Notably, in Yeast and BLCA, GRNGC exceeded the second-best AUROC by around 10 percentage points.

Overall, these results show that GRNGC surpasses competitors in **computational efficiency**, **ability to capture complex nonlinear interactions**, and **applicability to real-world datasets**.





<br/>
# 예제    





이 논문에서 다룬 **트레이닝 데이터와 테스트 데이터의 구체적인 입력·출력 형태**와 **테스크**는 다음과 같습니다.

1. **입력 (Input)**

   * 모든 데이터셋은 **다변량 시계열(multivariate time series)** 형태입니다.
   * 각 시계열 변수는 시간 축으로 정렬된 값들의 집합이며, 길이(`T`)와 차원(`p`)이 데이터셋마다 다릅니다.
   * 예시:

     * DREAM4: `p = 100`개의 유전자 발현량, `T = 210` 타임포인트
     * Lorenz-96: `p = 100`개의 시계열 변수, `T = 1000`
     * fMRI BOLD: `p = 15`개의 뇌 영역 시그널, `T = 200`
     * CausalTime Medical: `p = 40`, `T = 19200`
   * 모델 입력은 각 시계열의 **이전 k개 시점**(`xt-k : xt-1`)이며, 모든 변수의 과거값이 함께 들어갑니다.

2. **출력 (Output)**

   * 다음 시점(`xt`)에서 **모든 변수의 예측값**을 동시에 산출합니다.
   * 예측 결과와 실제값의 차이를 예측 손실(`Lp`)로 계산합니다.
   * 학습이 끝난 후, 모델의 **입출력 gradient**를 이용해 \*\*Granger 인과성 행렬(변수 간 인과 연결)\*\*을 도출합니다.

3. **테스크 (Task)**

   * **주요 목적**: 어떤 변수의 과거값이 다른 변수의 미래값 예측에 유의미하게 기여하는지 식별 (Granger causality 추론)
   * **세부 작업**:

     1. 시계열 예측 (forecasting)
     2. 예측 모델의 gradient를 이용해 인과성 추론
     3. 희소성(L1 정규화)을 적용해 인과성 네트워크 구조 복원
   * **평가 방식**: 예측된 인과성 네트워크와 실제 네트워크(ground truth)의 일치도를 AUROC, AUPRC로 측정.

---



In this study, the **training and testing data inputs/outputs** and **tasks** are as follows:

1. **Input**

   * All datasets are in the form of **multivariate time series**.
   * Each variable is a sequence of values ordered over time, with sequence length (`T`) and dimensionality (`p`) depending on the dataset.
   * Examples:

     * DREAM4: `p = 100` gene expression variables, `T = 210` time points
     * Lorenz-96: `p = 100` variables, `T = 1000`
     * fMRI BOLD: `p = 15` brain region signals, `T = 200`
     * CausalTime Medical: `p = 40`, `T = 19200`
   * The model input consists of the **past k time steps** (`xt-k : xt-1`) for all variables together.

2. **Output**

   * The model predicts **all variables at the next time step** (`xt`) simultaneously.
   * The prediction loss (`Lp`) is computed from the difference between predicted and true values.
   * After training, **gradients between inputs and outputs** are used to derive a **Granger causality matrix** representing directional dependencies between variables.

3. **Task**

   * **Main objective**: Identify whether the past values of one variable significantly improve the prediction of another variable’s future values (Granger causality inference).
   * **Detailed procedure**:

     1. Forecast next time-step values for all variables
     2. Use the model’s gradients to infer causal relationships
     3. Apply L1 regularization to produce a sparse causality network structure
   * **Evaluation**: Compare the predicted causality network to the ground-truth network using AUROC and AUPRC scores.






<br/>  
# 요약   




GRNGC는 단일 예측 모델과 입력–출력 gradient의 L1 정규화를 활용해 다양한 아키텍처(KAN, MLP, LSTM)에서 Granger 인과성을 추론하는 방법을 제안한다.
DREAM, Lorenz-96, fMRI BOLD, CausalTime 및 실제 유전자 네트워크 데이터에서 기존 모델 대비 높은 AUROC·AUPRC와 낮은 계산 비용을 달성했다.
입력은 다변량 시계열의 과거 k개 시점이며, 출력은 다음 시점의 모든 변수 예측값과 이를 기반으로 한 인과성 네트워크다.


GRNGC introduces a single-model framework that infers Granger causality by applying L1 regularization to input–output gradients, compatible with various architectures (KAN, MLP, LSTM).
It achieves superior AUROC/AUPRC and lower computational cost compared to baselines on DREAM, Lorenz-96, fMRI BOLD, CausalTime, and real gene network datasets.
The input is the past k time steps of multivariate time series, and the output includes predictions for all variables at the next step along with a derived causality network.



<br/>  
# 기타  


<br/>
# refer format:     


@article{liu2025grngc,
  title={Gradient Regularization-based Neural Granger Causality},
  author={Liu, Meiliang and Wen, Donghui and Yang, Xiaoxiao and Xu, Yunfang and Li, Zijin and Si, Zhengye and Yang, Xinyue and Zhao, Zhiwen},
  journal={arXiv preprint arXiv:2507.11178},
  year={2025},
  url={https://arxiv.org/abs/2507.11178}
}



Liu, Meiliang, Donghui Wen, Xiaoxiao Yang, Yunfang Xu, Zijin Li, Zhengye Si, Xinyue Yang, and Zhiwen Zhao. 2025. “Gradient Regularization-based Neural Granger Causality.” arXiv preprint arXiv:2507.11178. https://arxiv.org/abs/2507.11178.   




