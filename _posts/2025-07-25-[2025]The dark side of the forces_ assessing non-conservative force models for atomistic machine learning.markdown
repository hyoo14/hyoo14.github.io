---
layout: post
title:  "[2025]The dark side of the forces: assessing non-conservative force models for atomistic machine learning"  
date:   2025-07-25 14:30:40 +0900
categories: study
---

{% highlight ruby %}


한줄 요약: 


물리학 힘을 머신러닝으로 추정  
비보존 모델이 실제와 괴리가 생김을 밝힘   
보존모델을 메인으로, 비보존은 보조로 사용하는 하이브리드 제안   


짧은 요약(Abstract) :    


이 논문은 머신러닝을 활용해 원자 집단의 에너지와 그 에너지를 안정적인 상태로 이끄는 힘(포스)을 추정하는 기술에 대한 연구입니다. 전통적으로는 물리 법칙(특히 에너지 보존)을 지키기 위해 힘을 에너지의 도함수로 계산했지만, 최근에는 에너지 보존을 고려하지 않고 직접적으로 힘을 예측하는 모델들이 등장하고 있습니다. 이 논문은 이러한 비보존(non-conservative) 모델이 실제 시뮬레이션에서 문제가 될 수 있음을 밝혔습니다. 예를 들어, 구조 최적화의 수렴 실패, 분자동역학의 불안정성 등 다양한 문제가 발생합니다. 회전 대칭성과는 달리 에너지 보존은 학습하거나 모니터링하기가 어렵기 때문에, 저자들은 직접 예측한 힘을 보조적으로 사용하되, 주된 힘은 여전히 보존적인(conservative) 모델로부터 얻는 하이브리드 방식이 최적이라고 제안합니다.



The use of machine learning to estimate the energy of a group of atoms, and the forces that drive them to more stable configurations, has revolutionized the fields of computational chemistry and materials discovery. In this domain, rigorous enforcement of symmetry and conservation laws has traditionally been considered essential. For this reason, interatomic forces are usually computed as the derivatives of the potential energy, ensuring energy conservation. Several recent works have questioned this physically constrained approach, suggesting that directly predicting the forces yields a better trade-off between accuracy and computational efficiency – and that energy conservation can be learned during training. This work investigates the applicability of such non-conservative models in microscopic simulations. We identify and demonstrate several fundamental issues, from ill-defined convergence of geometry optimization to instability in various types of molecular dynamics. Contrary to the case of rotational symmetry, energy conservation is hard to learn, monitor, and correct for. The best approach to exploit the acceleration afforded by direct force prediction might be to use it in tandem with a conservative model, reducing – rather than eliminating – the additional cost of backpropagation, but avoiding the pathological behavior associated with non-conservative forces.




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



이 논문에서는 원자 수준 시뮬레이션에서 사용되는 보존적(conservative) 모델과 비보존적(non-conservative) 모델을 비교합니다.
대표적으로 사용된 모델은 **PET 아키텍처(Pozdnyakov & Ceriotti, 2023)**로, 이 모델은 회전 대칭을 강제하지 않는 구조로 설계되어 있으며, 다음과 같은 다양한 학습 방식으로 실험되었습니다:

PET (보존적 모델): 에너지 
𝑉
V와 그라디언트로부터 힘 
𝑓
f을 유도

PET-NC (비보존적 모델): 힘을 직접 예측 (에너지와의 연관 없음)

PET-M (혼합형 모델): 에너지로부터 유도된 힘과 직접 예측한 힘을 모두 포함

추가로, ORB나 Equiformer 등의 최신 비보존적 GNN 기반 모델들도 비교 대상으로 포함

학습 데이터는 bulk water dataset이며, 물의 액체 상태에서의 원자 위치와 에너지/힘을 포함한 고품질 양자역학적 데이터입니다. 모델은 에너지와 힘을 함께 학습하거나 단일한 목표만을 학습하는 다양한 설정으로 학습됩니다.

또한 Jacobian 비대칭성 측정(λ 지표), 닫힌 경로에서의 일(work), 에너지 드리프트, NVE/NVT 시뮬레이션의 안정성 평가 등 다양한 실험을 통해 보존성과 비보존성의 차이를 수학적·물리적으로 평가합니다.




This study compares conservative and non-conservative force models for atomistic machine learning using the PET architecture (Pozdnyakov & Ceriotti, 2023), which is rotationally unconstrained. The models evaluated include:

PET (Conservative): Forces derived via gradients from a predicted potential energy

PET-NC (Non-Conservative): Forces directly predicted without enforcing energy conservation

PET-M (Hybrid): Includes both gradient-based and direct force predictions

Additional models like ORB and Equiformer (non-conservative GNN-based models) are also assessed.

The training dataset is a bulk water dataset, consisting of atomistic configurations with energies and forces computed from quantum mechanical simulations. Models are trained under different supervision schemes: with energies only, forces only, or both.

The authors also introduce metrics to quantify non-conservativeness (e.g., Jacobian antisymmetry ratio λ), and evaluate models via geometry optimization, molecular dynamics simulations (NVE/NVT), and force-path integrals to examine their physical fidelity.




   
 
<br/>
# Results  



이 논문에서는 원자 수준 시뮬레이션에서 사용되는 보존적(conservative) 모델과 비보존적(non-conservative) 모델을 비교합니다.
대표적으로 사용된 모델은 **PET 아키텍처(Pozdnyakov & Ceriotti, 2023)**로, 이 모델은 회전 대칭을 강제하지 않는 구조로 설계되어 있으며, 다음과 같은 다양한 학습 방식으로 실험되었습니다:

PET (보존적 모델): 에너지 
𝑉
V와 그라디언트로부터 힘 
𝑓
f을 유도

PET-NC (비보존적 모델): 힘을 직접 예측 (에너지와의 연관 없음)

PET-M (혼합형 모델): 에너지로부터 유도된 힘과 직접 예측한 힘을 모두 포함

추가로, ORB나 Equiformer 등의 최신 비보존적 GNN 기반 모델들도 비교 대상으로 포함

학습 데이터는 bulk water dataset이며, 물의 액체 상태에서의 원자 위치와 에너지/힘을 포함한 고품질 양자역학적 데이터입니다. 모델은 에너지와 힘을 함께 학습하거나 단일한 목표만을 학습하는 다양한 설정으로 학습됩니다.

또한 Jacobian 비대칭성 측정(λ 지표), 닫힌 경로에서의 일(work), 에너지 드리프트, NVE/NVT 시뮬레이션의 안정성 평가 등 다양한 실험을 통해 보존성과 비보존성의 차이를 수학적·물리적으로 평가합니다.




This study compares conservative and non-conservative force models for atomistic machine learning using the PET architecture (Pozdnyakov & Ceriotti, 2023), which is rotationally unconstrained. The models evaluated include:

PET (Conservative): Forces derived via gradients from a predicted potential energy

PET-NC (Non-Conservative): Forces directly predicted without enforcing energy conservation

PET-M (Hybrid): Includes both gradient-based and direct force predictions

Additional models like ORB and Equiformer (non-conservative GNN-based models) are also assessed.

The training dataset is a bulk water dataset, consisting of atomistic configurations with energies and forces computed from quantum mechanical simulations. Models are trained under different supervision schemes: with energies only, forces only, or both.

The authors also introduce metrics to quantify non-conservativeness (e.g., Jacobian antisymmetry ratio λ), and evaluate models via geometry optimization, molecular dynamics simulations (NVE/NVT), and force-path integrals to examine their physical fidelity.






<br/>
# 예제  





 입력 및 출력 데이터:
트레이닝 데이터: 논문에서는 주로 bulk water dataset을 사용합니다.
이 데이터는 액체 상태의 물에 대한 양자역학적 계산을 통해 얻은 것으로, 각 프레임은 다음을 포함합니다:

입력: 각 원자의 위치 (3D 좌표), 원자 종류 (산소, 수소 등)

출력(label): 총 시스템 에너지, 각 원자에 작용하는 힘(3차원 벡터)

일부 비교 실험에서는 다양한 물질 데이터셋(OC20, OC22 등) 또는 pre-trained foundation models (e.g., MACE, ORB, EquiformerV2)을 사용해 물이나 금속 구조에서의 일반화 성능도 테스트합니다.



 테스크 종류:
힘 예측 (Force Prediction): 주어진 원자 구조에 대해 각 원자에 작용하는 힘을 예측합니다.

에너지 예측 (Energy Prediction): 전체 시스템의 잠재 에너지를 예측합니다.

기하학 최적화 (Geometry Optimization): 초기 원자 구조에서 출발하여 에너지가 최소화되는 구조로 수렴하도록 힘을 반복적으로 적용합니다.

분자동역학 시뮬레이션 (Molecular Dynamics): 시간에 따라 시스템의 움직임을 시뮬레이션하며 온도, 확산도, 진동 스펙트럼 등을 측정합니다.



 대표적 실험 예시:
PET-NC 모델로 NVE 시뮬레이션을 수행했을 때, 입력은 초기 물 구조와 속도이며, 출력은 각 스텝에서의 힘 → 위치 갱신 → 결과적인 온도 변화입니다.

이 실험에서 PET-NC는 비보존적 힘을 사용했기 때문에, 온도가 수천 켈빈으로 급격히 상승하는 물리적으로 잘못된 결과가 나타납니다. 반면, PET-C(보존적 모델)은 안정적인 에너지 유지와 정상적인 진동 스펙트럼을 보입니다.




 Input and Output Data:
Training Data: The main dataset used is the bulk water dataset, derived from quantum mechanical simulations of liquid water.
Each sample includes:

Input: 3D atomic positions and atom types (e.g., O, H)

Output (label): Total potential energy of the system, and per-atom force vectors (3D)

Additional evaluations include diverse datasets (e.g., OC20, OC22) and general-purpose pre-trained models such as MACE, ORB, and EquiformerV2 to assess generalizability across materials.



 Task Types:
Force Prediction: Predict the force vectors on each atom given an atomic configuration.

Energy Prediction: Predict the total potential energy of the atomic system.

Geometry Optimization: Iteratively update atomic positions to minimize energy using predicted forces.

Molecular Dynamics Simulation: Evolve the system over time and measure properties like temperature, diffusion, and vibrational spectra.



 Representative Experiment:
In one key experiment, the PET-NC model is used to run an NVE molecular dynamics simulation.

Input: Initial atomic structure and velocities of a water system

Process: At each step, forces are predicted and positions updated accordingly

Output: The trajectory, including changes in kinetic temperature

In this case, because PET-NC does not conserve energy, the simulation exhibits a dramatic and unphysical rise in temperature (e.g., thousands of Kelvin), while PET-C maintains stable thermodynamic behavior.




<br/>  
# 요약   


물리 법칙(에너지 보존)을 따르는 보존적 모델과, 힘을 직접 예측하는 비보존적 모델을 비교하기 위해 다양한 머신러닝 아키텍처(PET, PET-NC 등)를 훈련시켰다.
실험 결과, 비보존적 모델은 힘 예측 정확도는 준수하지만, 에너지 폭주 및 분자동역학 시뮬레이션의 불안정성과 같은 심각한 물리적 문제를 유발하였다.
이에 따라, 보존적 모델에 비보존적 예측을 보조적으로 결합한 하이브리드 모델(PET-M)이 가장 안정적이고 효율적인 방법으로 제안되었다.

To compare physically grounded conservative models and direct-force non-conservative models, various machine learning architectures (e.g., PET, PET-NC) were trained.
Results showed that while non-conservative models yield decent force accuracy, they cause severe physical issues such as energy drift and unstable molecular dynamics.
Therefore, a hybrid approach (PET-M) that supplements conservative forces with direct predictions was proposed as the most robust and efficient solution.



<br/>  
# 기타  




 Table 1: PET 모델의 에너지 및 힘 예측 정확도
**보존적 모델(PET-C)**은 에너지 예측 MAE 0.55 meV/atom, 힘 MAE 19.4 meV/Å로 가장 뛰어난 성능을 보임.

**비보존적 모델(PET-NC)**은 힘 MAE가 24.8로 약 30% 더 큼.

PET-M(보존+비보존): 두 방식의 장점을 절충하며, 실제 시뮬레이션에 적합.



 Figure 1: 원자쌍 거리별 Jacobian 비대칭성 시각화
**비보존적 힘의 비대칭성(λij)**은 원자 간 거리가 멀수록 커짐 → 장거리 상호작용에서 비물리적 효과 발생 가능성 시사.

이는 집단적 구조 변화가 중요한 시스템(예: 고체 확산, 물 클러스터)에 큰 영향을 줄 수 있음.



 Figure 2: NVE 시뮬레이션 중 온도 상승 시계열
PET-NC나 ORB 같은 비보존 모델은 시간 경과에 따라 온도가 비정상적으로 상승, 물리적으로 무의미한 시뮬레이션이 됨.

PET-M 모델은 conservative force를 주기적으로 적용해 이 현상을 완화함 → 다중 타임스텝(MTS) 전략 효과 입증.



 Figure 3: 속도 상관 함수 스펙트럼 (cvv(ω))
Langevin thermostat을 강하게 적용할수록 시간적 특성이 왜곡됨 (확산도 ↓, 진동 모드 둔화).

반면, **global thermostat (SVR)**은 구조/다이내믹 정확도 유지하면서 온도도 안정적으로 제어 → 적절한 thermostat 선택 중요.



 Appendix:
Appendix G: 다양한 재료(고체, 표면 등)에서의 실험 → 비보존 모델이 일반화에 취약하며, 에너지 비보존으로 인한 부작용이 일관되게 나타남.

Appendix H: 비보존 모델을 pretraining으로 활용 후 에너지 헤드만 fine-tuning → 학습 시간 단축 + 보존성 확보.

Appendix I: MTS 시뮬레이션 설정과 결과 비교 → PET-M은 conservative force만 사용하는 것과 거의 동일한 안정성 확보.





Table 1: Energy and Force Accuracy of PET Models
The conservative model (PET-C) achieved the lowest errors: 0.55 meV/atom for energy and 19.4 meV/Å for force.

The non-conservative PET-NC showed ~30% higher force error (24.8).

The hybrid PET-M balances both approaches and is suited for practical simulation use.



 Figure 1: Jacobian Asymmetry vs Interatomic Distance
As interatomic distance increases, the Jacobian asymmetry (λij) also grows, suggesting greater non-physical effects in long-range interactions.

This highlights the risk of using non-conservative models for systems with collective motion or delocalized interactions.



 Figure 2: Temperature Drift in NVE Simulations
PET-NC and ORB cause runaway heating in NVE simulations, making the trajectories physically meaningless.

PET-M, with periodic conservative corrections via Multiple Time Stepping (MTS), stabilizes temperature and restores physical plausibility.


 Figure 3: Velocity Correlation Spectrum (cvv(ω))
Strong Langevin thermostats distort dynamical features (e.g., suppress diffusion, broaden peaks).

Stochastic velocity rescaling (SVR) preserves both thermal stability and correct dynamical behavior, indicating the importance of thermostat choice.


 Appendix:
Appendix G: Additional tests on solids, surfaces, and general materials confirm that non-conservative models often fail to generalize and exhibit consistent unphysical drift.

Appendix H: Shows that using non-conservative models for pretraining and fine-tuning the energy head yields fast training with energy conservation.

Appendix I: Validates MTS simulation design, showing that PET-M offers near-identical stability and accuracy compared to fully conservative forces.




<br/>
# refer format:     



@inproceedings{bigi2025dark,
  title     = {The dark side of the forces: assessing non-conservative force models for atomistic machine learning},
  author    = {Filippo Bigi and Marcel F. Langer and Michele Ceriotti},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025},
  series    = {Proceedings of Machine Learning Research},
  volume    = {267},
  pages     = {1--10},
  address   = {Vancouver, Canada},
  publisher = {PMLR},
  url       = {https://zenodo.org/records/14778891}
}



Bigi, Filippo, Marcel F. Langer, and Michele Ceriotti. 2025.
“The Dark Side of the Forces: Assessing Non-Conservative Force Models for Atomistic Machine Learning.”
In Proceedings of the 42nd International Conference on Machine Learning (ICML), 1–10.
Vancouver, Canada: PMLR. https://zenodo.org/records/14778891.






