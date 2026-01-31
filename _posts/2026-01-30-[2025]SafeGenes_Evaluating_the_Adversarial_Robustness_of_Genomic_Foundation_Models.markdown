---
layout: post
title:  "[2025]SafeGenes: Evaluating the Adversarial Robustness of Genomic Foundation Models"
date:   2026-01-30 16:04:27 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 SafeGenes라는 프레임워크를 통해 유전체 기초 모델(GFM)의 적대적 강인성을 평가하였다.


짧은 요약(Abstract) :



이 연구에서는 유전자 변이 효과 예측에 있어 강력한 도구로 자리 잡고 있는 유전자 기초 모델(Genomic Foundation Models, GFMs)의 적대적 강인성을 평가하기 위한 프레임워크인 SafeGenes를 제안합니다. GFMs는 변이 효과 예측에서 상당한 성공을 거두었지만, 이들의 적대적 강인성은 아직 충분히 탐구되지 않았습니다. SafeGenes는 설계된 유사 적대적 유전자와 임베딩 공간 조작에 대한 강인성을 평가하기 위해 적대적 공격을 활용합니다. 우리는 Fast Gradient Sign Method (FGSM)와 소프트 프롬프트 공격을 사용하여 GFMs의 적대적 취약성을 평가했습니다. FGSM은 입력 시퀀스에 최소한의 변화를 주고, 소프트 프롬프트 공격은 입력 토큰을 수정하지 않고 모델 예측을 조작하기 위해 연속 임베딩을 최적화합니다. 이 연구는 GFMs의 적대적 조작에 대한 취약성을 포괄적으로 평가하며, 특히 의료 맥락에서의 결정에 직접적인 영향을 미칠 수 있는 심각한 취약성을 드러냅니다.




In this study, we propose SafeGenes, a framework for evaluating the adversarial robustness of Genomic Foundation Models (GFMs), which have demonstrated significant success in variant effect prediction. However, their adversarial robustness remains largely unexplored. SafeGenes leverages adversarial attacks to assess robustness against engineered near-identical adversarial genes and embedding-space manipulations. We evaluate the adversarial vulnerabilities of GFMs using two approaches: the Fast Gradient Sign Method (FGSM), which introduces minimal perturbations to input sequences, and a soft prompt attack that optimizes continuous embeddings to manipulate model predictions without modifying the input tokens. This study provides a comprehensive assessment of GFM susceptibility to adversarial manipulation, revealing critical vulnerabilities that could directly impact decisions in medical contexts.


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



이 연구에서는 유전자 변이 효과 예측을 위한 두 가지 주요 공격 전략을 구현하여 유전자 기초 모델(GFMs)의 적대적 취약성을 평가합니다. 이 공격 전략은 **Fast Gradient Sign Method (FGSM)**와 **소프트 프롬프트 공격**입니다. 각 방법의 세부 사항은 다음과 같습니다.

1. **Fast Gradient Sign Method (FGSM)**:
   FGSM은 입력 시퀀스에 대한 손실 함수의 그래디언트를 활용하여 적대적 입력을 생성하는 널리 사용되는 공격 방법입니다. 주어진 원래 입력 시퀀스 \( s \)와 해당 레이블 \( y \)에 대해, FGSM은 다음 공식을 사용하여 적대적 입력 \( s_{adv} \)를 생성합니다:
   \[
   s_{adv} = s + \epsilon \cdot \text{sign}(\nabla_s L(s, y))
   \]
   여기서 \( L(s, y) \)는 분류 손실을 나타내며, \( \nabla_s L \)은 입력 시퀀스 임베딩에 대한 손실의 그래디언트입니다. \( \epsilon \)은 작은 스칼라로, perturbation의 크기를 조절합니다. 이 방법은 모델의 입력 시퀀스에 미세한 변화를 주어 예측의 안정성을 저하시킬 수 있는지를 평가하는 데 사용됩니다.

2. **소프트 프롬프트 공격**:
   소프트 프롬프트 공격은 학습 가능한 임베딩 시퀀스를 원래 입력 시퀀스 앞에 추가하여 모델의 내부 표현 공간에서 작동하는 공격입니다. 이 공격은 두 가지 변형이 있습니다:
   - **신뢰도 탈취(Confidence Hijack)**: 이 공격은 모델이 잘못된 예측에 대한 신뢰도를 높이도록 유도합니다. 이를 위해, 우리는 pseudo-log-likelihood ratio (PLLR) 계산을 유지하고, 레이블 \( y \)에 따라 손실을 최대화하는 목표를 설정합니다.
   - **타겟 소프트 프롬프트 공격(Targeted Soft Prompt Attack)**: 이 공격은 benign 변이를 pathogenic으로 잘못 분류하도록 유도하는 데 초점을 맞춥니다. benign 예제에 대한 마스크를 설정하고, 이들에 대해 높은 PLLR 값을 출력하도록 모델을 강제합니다.

이 연구에서는 ESM1b, ESM2와 같은 다양한 GFMs을 사용하여 이 공격들을 평가하였으며, 각 모델의 성능 저하를 ROC AUC, AUPR 및 임계값 기반 메트릭을 통해 측정하였습니다. 이러한 방법론은 모델의 적대적 강인성을 평가하고, 임상 유전체학에서의 신뢰성을 보장하기 위한 중요한 기초를 제공합니다.

---



In this study, we implement two main attack strategies to evaluate the adversarial vulnerabilities of Genomic Foundation Models (GFMs) for variant effect prediction. These attack strategies are **Fast Gradient Sign Method (FGSM)** and **Soft Prompt Attack**. The details of each method are as follows:

1. **Fast Gradient Sign Method (FGSM)**:
   FGSM is a widely used attack method that generates adversarial inputs by leveraging the gradient of the loss function with respect to the input sequence. Given an original input sequence \( s \) and its corresponding label \( y \), FGSM constructs an adversarial input \( s_{adv} \) using the following formula:
   \[
   s_{adv} = s + \epsilon \cdot \text{sign}(\nabla_s L(s, y))
   \]
   Here, \( L(s, y) \) represents the classification loss, and \( \nabla_s L \) denotes the gradient of the loss with respect to the input sequence embeddings. \( \epsilon \) is a small scalar that controls the magnitude of the perturbation. This method is used to assess whether subtle changes to the model's input sequences can undermine prediction stability.

2. **Soft Prompt Attack**:
   The soft prompt attack operates in the embedding space by prepending a trainable embedding sequence to the original input sequences. This attack includes two variants:
   - **Confidence Hijack**: This attack aims to mislead the model by increasing its confidence in incorrect predictions. Specifically, we retain the pseudo-log-likelihood ratio (PLLR) computation and define an adversarial objective that flips the labels and maximizes the binary cross-entropy loss.
   - **Targeted Soft Prompt Attack**: This attack focuses on misclassifying benign variants as pathogenic. A mask is applied over benign examples, and the attack loss is defined to force the model to output high PLLR values for benign variants, mimicking confident pathogenic predictions.

In this study, we evaluate these attacks using various GFMs, such as ESM1b and ESM2, and measure the degradation in model performance using metrics like ROC AUC, AUPR, and threshold-based metrics. This methodology provides a crucial foundation for assessing the adversarial robustness of models and ensuring reliability in clinical genomics applications.


<br/>
# Results


이 연구에서는 다양한 유전자 기초 모델(GFMs)의 적대적 강건성을 평가하기 위해 여러 실험을 수행했습니다. 주요 결과는 다음과 같습니다:

1. **모델 성능 저하**: 모든 평가된 GFM은 FGSM(빠른 그래디언트 부호 방법) 및 소프트 프롬프트 공격에 대해 일관된 성능 저하를 보였습니다. 특히, AUC(곡선 아래 면적)와 AUPR(정밀도-재현율 곡선 아래 면적) 메트릭에서 모두 감소가 관찰되었습니다. 예를 들어, ESM2-650M 모델은 FGSM 공격을 받을 경우 AUC가 0.740에서 0.710으로 감소했습니다.

2. **소프트 프롬프트 공격의 효과**: 소프트 프롬프트 공격은 특히 작은 모델에서 가장 큰 성능 저하를 초래했습니다. ESM2-150M 모델은 AUC가 0.630에서 0.550으로 감소하는 등, 소프트 프롬프트 공격이 모델의 신뢰성을 크게 저하시킬 수 있음을 보여주었습니다.

3. **모델 크기에 따른 강건성 차이**: ESM1b 및 ESM1v와 같은 대형 모델은 상대적으로 더 나은 성능을 보였지만, 여전히 적대적 공격에 취약했습니다. 예를 들어, ESM1b 모델은 FGSM 공격 후 AUC가 0.810에서 0.780으로 감소했습니다.

4. **임상 유용성**: 이러한 결과는 임상 유전자 해석에서의 신뢰성 문제를 강조합니다. FGSM 및 소프트 프롬프트 공격 모두 모델의 결정 경계를 약화시켜, 잘못된 질병 분류를 초래할 수 있습니다. 특히, benign(양성) 변이가 pathogenic(병원성)으로 잘못 분류될 위험이 높아졌습니다.

5. **비교 모델**: ProteinBERT와 같은 다른 모델과 비교했을 때, ESM 모델들은 더 높은 성능을 보였지만, 적대적 공격에 대한 취약성은 여전히 존재했습니다. 예를 들어, ProteinBERT는 FGSM 공격을 받을 경우 AUC가 0.762에서 0.429로 감소했습니다.

이 연구는 GFMs의 적대적 강건성을 평가하는 데 있어 중요한 통찰을 제공하며, 향후 임상 유전자 모델의 신뢰성을 높이기 위한 방안으로 적대적 공격에 대한 평가가 필수적임을 강조합니다.

---



In this study, various genomic foundation models (GFMs) were evaluated for their adversarial robustness through a series of experiments. The key findings are as follows:

1. **Performance Degradation**: All evaluated GFMs exhibited consistent performance degradation under both FGSM (Fast Gradient Sign Method) and soft prompt attacks. Significant drops in AUC (Area Under the Curve) and AUPR (Area Under the Precision-Recall Curve) metrics were observed. For instance, the ESM2-650M model's AUC decreased from 0.740 to 0.710 when subjected to FGSM attacks.

2. **Effect of Soft Prompt Attacks**: The soft prompt attack caused the most significant performance degradation, particularly in smaller models. The ESM2-150M model's AUC dropped from 0.630 to 0.550, indicating that soft prompt attacks can severely undermine model reliability.

3. **Robustness Differences by Model Size**: Larger models, such as ESM1b and ESM1v, demonstrated relatively better baseline performance but remained vulnerable to adversarial attacks. For example, the AUC of the ESM1b model decreased from 0.810 to 0.780 after FGSM attacks.

4. **Clinical Relevance**: These results highlight the reliability issues in clinical genomic interpretation. Both FGSM and soft prompt attacks weakened the model's decision boundaries, increasing the risk of misclassifying benign variants as pathogenic. This poses a significant threat to patient care.

5. **Comparative Models**: Compared to other models like ProteinBERT, ESM models showed higher performance but still exhibited vulnerabilities to adversarial attacks. For example, ProteinBERT's AUC dropped from 0.762 to 0.429 under FGSM attacks.

This study provides critical insights into the adversarial robustness of GFMs and emphasizes the necessity of evaluating adversarial attacks to enhance the reliability of clinical genomic models in the future.


<br/>
# 예제


이 논문에서는 유전적 변이의 효과를 예측하기 위해 두 가지 데이터셋을 사용합니다: 심근병증(Cardiomyopathies, CM)과 부정맥(Arrhythmias, ARM)입니다. 각 데이터셋은 병리적(병원성) 변이와 비병리적(양성) 변이를 포함하고 있으며, 이 변이들은 유전적 질병의 진단 및 치료에 중요한 역할을 합니다.

#### 데이터셋 구성
1. **심근병증(CM) 데이터셋**
   - **훈련 데이터**: 
     - 병리적 변이: 238개
     - 비병리적 변이: 202개
     - 총합: 440개
   - **테스트 데이터**: 
     - 병리적 변이: 118개
     - 비병리적 변이: 100개
     - 총합: 218개

2. **부정맥(ARM) 데이터셋**
   - **훈련 데이터**: 
     - 병리적 변이: 168개
     - 비병리적 변이: 158개
     - 총합: 326개
   - **테스트 데이터**: 
     - 병리적 변이: 84개
     - 비병리적 변이: 79개
     - 총합: 163개

#### 입력 및 출력
- **입력**: 각 변이는 DNA 또는 단백질 서열로 표현됩니다. 예를 들어, 특정 아미노산의 변형이 포함된 서열이 입력으로 사용됩니다. 이 서열은 모델에 의해 처리되어 변이의 병리적 여부를 예측하는 데 사용됩니다.
  
- **출력**: 모델의 출력은 각 변이에 대한 병리적 확률을 나타내는 점수입니다. 이 점수는 변이가 병리적일 확률을 나타내며, 0과 1 사이의 값으로 표현됩니다. 예를 들어, 0.8의 점수는 해당 변이가 병리적일 확률이 80%임을 의미합니다.

#### 태스크
모델의 주요 태스크는 주어진 변이에 대해 병리적(1) 또는 비병리적(0)로 분류하는 것입니다. 이를 위해 모델은 훈련 데이터에서 학습한 패턴을 바탕으로 테스트 데이터의 변이를 평가하고, 각 변이에 대한 예측을 수행합니다.



This paper utilizes two datasets for predicting the effects of genetic variants: Cardiomyopathies (CM) and Arrhythmias (ARM). Each dataset includes pathogenic (disease-causing) and benign (non-disease-causing) variants, which play a crucial role in the diagnosis and treatment of genetic diseases.

#### Dataset Composition
1. **Cardiomyopathies (CM) Dataset**
   - **Training Data**: 
     - Pathogenic variants: 238
     - Benign variants: 202
     - Total: 440
   - **Test Data**: 
     - Pathogenic variants: 118
     - Benign variants: 100
     - Total: 218

2. **Arrhythmias (ARM) Dataset**
   - **Training Data**: 
     - Pathogenic variants: 168
     - Benign variants: 158
     - Total: 326
   - **Test Data**: 
     - Pathogenic variants: 84
     - Benign variants: 79
     - Total: 163

#### Input and Output
- **Input**: Each variant is represented as a DNA or protein sequence. For example, a sequence containing a specific amino acid mutation is used as input. This sequence is processed by the model to predict the pathogenicity of the variant.
  
- **Output**: The model's output is a score representing the pathogenic probability for each variant. This score indicates the likelihood that the variant is pathogenic, expressed as a value between 0 and 1. For instance, a score of 0.8 means there is an 80% probability that the variant is pathogenic.

#### Task
The primary task of the model is to classify each given variant as pathogenic (1) or benign (0). To achieve this, the model evaluates the variants in the test data based on patterns learned from the training data and performs predictions for each variant.

<br/>
# 요약


이 논문에서는 SafeGenes라는 프레임워크를 통해 유전체 기초 모델(GFM)의 적대적 강인성을 평가하였다. Fast Gradient Sign Method(FGSM)와 소프트 프롬프트 공격을 사용하여 모델의 취약점을 분석한 결과, 모든 GFM이 적대적 공격에 취약하며, 특히 소프트 프롬프트 공격이 더 심각한 성능 저하를 초래함을 발견하였다. 이러한 결과는 임상 유전체학에서의 신뢰성과 안전성을 보장하기 위해 모델의 강인성 평가가 필수적임을 강조한다.




In this paper, the authors evaluated the adversarial robustness of genomic foundation models (GFMs) using a framework called SafeGenes. By employing the Fast Gradient Sign Method (FGSM) and soft prompt attacks, they found that all GFMs exhibited vulnerabilities to adversarial attacks, with soft prompt attacks causing more severe performance degradation. These findings underscore the necessity of robustness evaluation for ensuring reliability and safety in clinical genomics.

<br/>
# 기타


1. **다이어그램 및 피규어**
   - **Figure 1**: 이 다이어그램은 GFM의 변이 효과 예측에서의 적대적 민감도를 보여줍니다. 특히, 병원성 변이와 정상 변이의 임베딩 간의 거리 조정이 어떻게 모델의 예측에 영향을 미치는지를 설명합니다. 이는 모델이 변이의 병원성을 예측하는 데 있어 결정 경계가 얼마나 불안정한지를 강조합니다.
   - **Figure 2**: CM 데이터셋에서 FGSM 공격에 대한 모델의 강건성을 평가한 결과를 보여줍니다. AUC와 AUPR의 일관된 감소는 적대적 입력이 모델의 분류 성능에 미치는 영향을 강조합니다. 이는 모델이 적대적 공격에 얼마나 취약한지를 나타냅니다.
   - **Figure 3**: ARM 데이터셋에서 FGSM 공격의 영향을 보여줍니다. ROC 곡선의 경미한 감소는 입력 변형에 대한 모델의 민감성을 나타냅니다. 이는 모델의 신뢰성을 평가하는 데 중요한 정보를 제공합니다.
   - **Figure 4**: CM 및 ARM 데이터셋에서의 목표 소프트 프롬프트 공격 결과를 보여줍니다. 이 공격은 정상 변이를 병원성으로 잘못 분류하도록 유도하며, 이는 모델의 신뢰성을 저하시킵니다.

2. **테이블**
   - **Table 1 & 2**: CM 및 ARM 데이터셋에서 다양한 공격 전략에 대한 AUC 점수를 보여줍니다. FGSM, DeepFool, C&W, PGD와 같은 공격 방법이 모델 성능에 미치는 영향을 비교합니다. 이 결과는 각 공격 방법의 효과를 정량화하고, 모델의 강건성을 평가하는 데 중요한 통찰을 제공합니다.
   - **Table 5**: PGD 공격 하에서의 모델 신뢰성 지표를 보여줍니다. Brier 점수와 ECE 값의 상승은 모델의 신뢰도가 심각하게 저하되었음을 나타냅니다. 이는 임상적 결정에 대한 신뢰성을 평가하는 데 중요한 정보를 제공합니다.

3. **어펜딕스**
   - 어펜딕스에는 연구에서 사용된 데이터셋의 통계와 코드 접근 방법이 포함되어 있습니다. 이는 연구의 재현성을 높이고, 다른 연구자들이 이 연구를 기반으로 추가 연구를 수행할 수 있도록 돕습니다.

---



1. **Diagrams and Figures**
   - **Figure 1**: This diagram illustrates the adversarial sensitivity in GFM for variant effect prediction. It explains how the adjustment of distances between pathogenic and benign variant embeddings affects the model's predictions, highlighting the instability of decision boundaries in predicting variant pathogenicity.
   - **Figure 2**: This figure shows the robustness analysis of the model against FGSM attacks on the CM dataset. The consistent drop in AUC and AUPR emphasizes the impact of adversarial inputs on the model's classification performance, indicating the model's vulnerability to adversarial attacks.
   - **Figure 3**: This figure presents the effects of FGSM attacks on the ARM dataset. The slight drop in the ROC curve indicates the model's sensitivity to input perturbations, providing critical insights into the model's reliability.
   - **Figure 4**: This figure displays the results of targeted soft prompt attacks on both CM and ARM datasets. The attack misclassifies benign variants as pathogenic, highlighting the degradation of the model's trustworthiness.

2. **Tables**
   - **Table 1 & 2**: These tables present AUC scores under various attack strategies for the CM and ARM datasets. They compare the effects of attack methods like FGSM, DeepFool, C&W, and PGD on model performance, providing quantitative insights into the effectiveness of each attack method and assessing model robustness.
   - **Table 5**: This table shows calibration and decision reliability metrics under PGD attacks. The increase in Brier scores and ECE values indicates a severe degradation in model reliability, which is crucial for evaluating the trustworthiness of clinical decisions.

3. **Appendix**
   - The appendix includes statistics of the datasets used in the study and access to the code. This enhances the reproducibility of the research and allows other researchers to build upon this work for further studies.

<br/>
# refer format:


### BibTeX 형식

```bibtex
@article{zhan2025safegenes,
  title={SafeGenes: Evaluating the Adversarial Robustness of Genomic Foundation Models},
  author={Zhan, Huixin and Barbour, Clovis and Moore, Jason H.},
  journal={arXiv preprint arXiv:2506.00821v2},
  year={2025},
  url={https://arxiv.org/abs/2506.00821v2},
  note={Accessed: 2025-12-02}
}
```

### 시카고 스타일

Huixin Zhan, Clovis Barbour, and Jason H. Moore. 2025. "SafeGenes: Evaluating the Adversarial Robustness of Genomic Foundation Models." arXiv preprint arXiv:2506.00821v2. Accessed December 2, 2025. https://arxiv.org/abs/2506.00821v2.




# 리뷰   


요약 (Summary)     
본 원고는 변이 효과 예측(variant effect prediction)을 위한 유전체 파운데이션 모델(Genomic Foundation Models, GFMs)의 적대적 강건성(adversarial robustness)을 평가하기 위한 프레임워크인 SafeGenes를 제안한다. 저자들은 gradient 기반 공격(FGSM/PGD)과 임베딩 공간 기반 공격(soft prompt attack)을 모두 도입하고, 여러 모델과 데이터셋에 걸쳐 성능이 체계적으로 저하됨을 보여준다. 본 연구는 임상 유전체학(clinical genomics)에서 GFMs의 신뢰성과 안전성이라는 중요하고 시의적절한 문제를 다룬다는 점에서 의미가 있다.



강점    
본 연구 주제는 기계학습 강건성과 임상적 동기가 있는 유전체 예측을 결합하고 있어 독자층과 매우 높은 관련성을 가진다.

유전체 파운데이션 모델 맥락에서 soft prompt attack을 적용한 시도는 참신하며, 임베딩 공간(embedding-space)의 취약성을 조명하는 흥미로운 관점을 제공한다.

FGSM, PGD, C&W, DeepFool, Boundary attack 등 다양한 공격 전략과 ESM 계열 모델 및 ProteinBERT 등 여러 모델을 포함하는 폭넓은 실험 구성이 장점이다.

Brier score, ECE, FPR@TPR 등 보정(calibration) 및 임상적 강건성 지표를 포함함으로써, 실제 의사결정 상황과의 연결성을 강화하였다.





주요 우려 사항   
1. Soft prompt attack의 위협 모델(threat model)과 실제 적용 가능성이 불명확함

Soft prompt attack은 기술적으로 흥미롭지만, 그 위협 모델이 실제 환경에서 얼마나 현실적인지에 대한 정당화가 충분하지 않다.
입력 서열이나 모델 가중치를 변경하지 않고, 공격자가 연속적인 prompt 임베딩을 삽입하거나 최적화할 수 있는 실제 배포 시나리오가 무엇인지 명확하지 않다.

본 원고는
(i) 진단적 stress-test로서의 평가 도구인지,
(ii) 실제 적대적 공격 시나리오를 가정한 분석인지
를 보다 명확히 구분할 필요가 있다. 이러한 구분이 부족할 경우, 본 연구의 핵심 기여는 임상적으로 의미 있는 보안 분석이라기보다 개념적 데모(conceptual demonstration)로 인식될 위험이 있다.

2. 적대적 변형(adversarial perturbation)의 생물학적 타당성에 대한 근거 부족

본 논문은 “near-identical adversarial genes” 및 “minimal perturbations”를 강조하지만, 생성된 적대적 예제가 실제 생물학적으로 타당한 변이인지에 대한 검증이 충분히 이루어지지 않았다.

기존 연구들이 코돈 보존(codon-level preservation)이나 모티프 보존(motif conservation)과 같은 생물학적 제약 조건을 명시적으로 적용한 것과 달리, 본 연구의 공격은 주로 임베딩 공간에서 수행된다. 이는 생성된 적대적 예제가 생물학적으로 의미 있는 변이라기보다 추상적인 표현 공간상의 교란에 불과할 가능성을 제기한다.

3. Soft prompt 결과에 대한 과도한 의존

FGSM 및 기타 gradient 기반 공격은 비교적 제한적인 성능 저하만을 유발하는 반면, soft prompt attack이 논문의 핵심 결과와 서사를 거의 지배하고 있다.

이러한 불균형은, 관찰된 취약성이 일반적인 적대적 민감성이라기보다는 특정하고 특수한 공격 설정에 의해 주도된다는 인상을 줄 수 있다.
토큰 수준(token-level) 교란이 상대적으로 효과가 낮은 이유와, 이것이 실제 환경에서의 강건성 해석에 어떤 의미를 갖는지에 대한 보다 균형 잡힌 논의가 필요하다.

4. 완화(mitigation) 전략에 대한 구체적 제안 부족

Discussion에서는 강건성 평가의 중요성을 강조하지만, 이러한 취약점을 어떻게 완화할 수 있는지에 대한 실질적인 통찰은 제한적이다.

임상적 적용을 전제로 한다면, 독자들은 최소한 다음과 같은 방향에 대한 구체적인 가설이나 초기 실험을 기대할 수 있다:

adversarial training

임베딩 정규화(embedding regularization)

calibration 개선 전략

현재 형태에서는 진단(diagnosis)에 머무르고 있으며, 대응 전략이 충분히 제시되지 못하고 있다.

부차적 우려 사항 (Minor Concerns)

Methods 섹션은 다소 장황하며, 여러 수식과 목적 함수 설명은 더 간결하게 정리될 수 있다.

prompt 길이(prompt token 수)에 대한 ablation 실험 그림은(예: 100 prompt tokens) 오히려 공격 설정의 인위성을 강조하는 효과를 낳을 수 있다.



종합    
본 논문은 유전체 파운데이션 모델의 강건성을 평가하는 데 있어 중요한 공백을 메우며, 임베딩 공간 기반 공격이라는 새로운 패러다임을 제시한다는 점에서 가치가 있다. 그러나 현재의 서술은 soft prompt attack의 실제적 의미와 영향력을 충분히 현실적인 생물학적 및 배포 환경에 근거하여 설명하지 못하고 있다.

위협 모델을 보다 명확히 정립하고, 생물학적 타당성 검증을 추가하며, 본 프레임워크를 “진단적 벤치마크”로서 위치시키는 방향으로 서술을 정제한다면, 본 원고는 훨씬 더 설득력 있고 완성도 높은 논문이 될 것이다.

