---
layout: post
title:  "[2025]Distribution-Level Feature Distancing for Machine Unlearning"  
date:   2025-03-21 19:16:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 





짧은 요약(Abstract) :    




딥러닝의 활용이 증가함에 따라 개인 정보 보호에 대한 요구도 높아지고 있습니다. 특히, “잊혀질 권리”에 따라 특정 데이터(예: 얼굴 이미지)를 학습에 사용한 모델에서 제거하는 것이 중요해졌습니다. 하지만 기존의 머신 언러닝 기법은 데이터를 잊게 하는 과정에서 모델의 성능이 전반적으로 저하되는 “상관관계 붕괴(correlation collapse)” 문제가 발생합니다. 이 논문은 이러한 문제를 해결하기 위해 **Distribution-Level Feature Distancing (DLFD)**라는 새로운 방법을 제안합니다. DLFD는 ‘잊어야 하는 데이터’와 ‘유지할 데이터’의 특성 분포를 멀어지게 하여 효율적으로 언러닝을 수행하면서도, 원래의 분류 성능을 보존합니다. 얼굴 인식 데이터셋을 이용한 실험을 통해 DLFD가 기존 최신 기법들보다 더 나은 언러닝 성능과 모델 유지 능력을 보여줍니다.

⸻


With growing concerns over privacy in deep learning, the “right to be forgotten” has become increasingly important—requiring AI systems to remove specific data (e.g., personal face images) upon request. Existing machine unlearning methods often cause correlation collapse, where meaningful connections between features and labels are disrupted, reducing model performance. To address this, the authors propose a novel method called Distribution-Level Feature Distancing (DLFD). DLFD alters the feature distribution of retained data to differ from that of the data to be forgotten, ensuring the model forgets specific instances without degrading its overall performance. Experiments on facial recognition datasets show that DLFD significantly outperforms previous state-of-the-art unlearning methods in both forgetting and utility preservation.



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




1. 제안 방법 개요 – DLFD (Distribution-Level Feature Distancing)
DLFD는 기존 머신 언러닝 기법의 한계인 “상관관계 붕괴(Correlation Collapse)” 문제를 해결하기 위한 새로운 프레임워크입니다. 상관관계 붕괴란 모델이 데이터를 잊으려다 중요한 feature-label 관계까지 훼손되는 현상입니다. 이를 방지하면서도 특정 데이터를 효과적으로 잊게 하기 위해 DLFD는 세 가지 핵심 구성요소를 포함합니다:
	•	(1) Feature Distribution Optimization:
잊어야 할 데이터(Forget set)와 유지할 데이터(Retain set)의 feature 분포를 최적 수송 거리(Optimal Transport, OT)를 이용해 서로 멀어지도록 조정합니다. 이렇게 하면 모델이 데이터를 잊으면서도 기존의 유용한 분류 성능을 유지할 수 있습니다.
	•	(2) Classification Loss Preservation:
잊지 않아야 할 데이터의 분류 정확도를 유지하기 위해, perturbation 과정 중에 분류 손실(classification loss)을 같이 고려합니다. 이 손실은 시간이 지남에 따라 점차 중요성이 증가하는 선형 가중치를 적용해 모델 유틸리티를 지켜줍니다.
	•	(3) Dynamic Forgetting Strategy:
학습 중 실시간으로 언러닝 점수를 측정하여, 일정 기준 이상 잊었을 경우에는 OT 손실을 중단하고 분류 손실 중심으로만 fine-tuning을 진행하여 효율성과 성능을 동시에 확보합니다.

⸻

2. 아키텍처 및 구현 구조
DLFD는 기본적으로 기존 모델(예: ResNet18, DenseNet121, EfficientNetB0)을 사용하며, 언러닝 과정에서는 해당 모델의 파라미터를 그대로 유지한 채 추가적인 perturbation과 fine-tuning을 통해 성능을 조정합니다. 이 프레임워크는 다양한 비전 모델에 적용 가능하도록 설계되어 있습니다.

⸻

3. 트레이닝 데이터
DLFD는 얼굴 이미지 기반의 다양한 데이터셋으로 학습되며 다음과 같은 세 가지 주요 데이터셋을 사용합니다:
	•	MUFAC: 8개 연령 그룹으로 분류된 13,068개의 얼굴 이미지.
	•	RAF-DB: 7개 감정 클래스를 가진 15,000장의 얼굴 이미지.
	•	MUCAC (from CelebA): 성별, 나이, 표정의 세 가지 속성이 포함된 30,000장의 얼굴 이미지.

각 데이터셋은 retain set, forget set, unseen test set으로 나뉘어 언러닝 성능과 모델 유틸리티를 동시에 평가합니다.

⸻


1. Proposed Method – DLFD (Distribution-Level Feature Distancing)
DLFD is a novel machine unlearning framework designed to mitigate correlation collapse, a phenomenon where useful feature-label relationships degrade during the forgetting process. DLFD consists of three main components:
	•	(1) Feature Distribution Optimization:
The method uses Optimal Transport (OT) to shift the feature distribution of the retained data away from that of the forget data. This helps the model forget specific instances while preserving overall classification utility.
	•	(2) Classification Loss Preservation:
To retain task-relevant information, classification loss is incorporated into the perturbation process. A linear weight gradually increases during training, balancing between maximizing OT distance and maintaining model utility.
	•	(3) Dynamic Forgetting Strategy:
The method dynamically monitors the forgetting score during training. Once the score reaches a threshold, the OT-based perturbation stops, and training continues with only classification loss, reducing overhead while preserving task performance.

⸻

2. Architecture and Implementation
DLFD is implemented on top of widely-used CNN architectures like ResNet18, DenseNet121, and EfficientNetB0. It does not require structural modifications to the original model, making it compatible with various image classification models.

⸻

3. Training Data
Three facial datasets are used to evaluate DLFD under various classification tasks:
	•	MUFAC: 13,068 facial images categorized into 8 age groups.
	•	RAF-DB: 15,000 facial images labeled with 7 emotion classes.
	•	MUCAC (from CelebA): 30,000 facial images with binary labels for gender, age, and expression.

Each dataset is split into Dretain (to keep), Dforget (to forget), and Dunseen (used for evaluation only), enabling rigorous testing of both forgetting performance and task accuracy.

⸻






       
 
<br/>
# Results  






⸻




1. 테스크(Task)
DLFD는 총 4가지 얼굴 인식 관련 분류 과제에서 평가됩니다:
	•	연령 분류 (8-class)
	•	감정 인식 (7-class)
	•	성별 분류 (2-class)
	•	멀티 속성 분류 (3개의 이진 라벨: 성별, 나이, 표정)

이러한 다양한 분류 과제를 통해 제안된 모델의 일반성과 안정성을 평가했습니다.

⸻

2. 테스트 데이터(Test Set)
각 실험은 다음과 같은 세 종류의 데이터셋으로 나뉘어 구성됩니다:
	•	Dretain: 학습에 계속 사용하는 데이터
	•	Dforget: 모델이 잊어야 하는 대상 데이터
	•	Dunseen: 학습에는 사용되지 않고 평가에만 사용되는 새로운 데이터

이러한 분리로 인해 언러닝 성능과 모델 일반화 성능을 동시에 정확하게 평가할 수 있습니다.

⸻

3. 평가 지표(Metrics)
	•	Test Accuracy: 모델이 본래 분류 과제에서 얼마나 잘 작동하는지를 평가 (모델 유틸리티)
	•	Forgetting Score: Membership Inference Attack (MIA)를 통해 얼마나 데이터를 잘 “잊었는지” 평가.
	•	수식: Forgetting Score = |MIA Acc. − 0.5| × 2
	•	완전히 잊은 경우 0.0 (무작위 분류 수준)
	•	NoMUS (Normalized Machine Unlearning Score): 모델 유틸리티와 언러닝 성능을 함께 고려한 종합 지표
	•	수식: NoMUS = 0.5 × (정확도 + (1 - Forgetting Score))
	•	값이 1에 가까울수록 유틸리티와 언러닝 모두 우수함

⸻

4. 비교 방법(Comparison)
DLFD는 기존의 여러 최신 언러닝 기법들과 비교됩니다:
	•	Fine-tuning
	•	NegGrad (2020)
	•	CF-k, EU-k (2022)
	•	UNSIR (2023)
	•	BadTeaching (2023)
	•	SCRUB (2023)

실험 결과, DLFD는 대부분의 경우에서 가장 높은 NoMUS 점수를 기록하며, 모델 성능 유지와 언러닝 성능 모두에서 뛰어난 결과를 보였습니다. 특히 연령 및 감정 인식과 같은 다중 클래스 과제에서 성능 차이가 두드러졌습니다.

⸻



1. Tasks
DLFD is evaluated across four facial classification tasks:
	•	Age Classification (8 classes)
	•	Emotion Recognition (7 classes)
	•	Gender Classification (binary)
	•	Multi-Attribute Classification (3 binary labels: gender, age, expression)

These tasks allow for a thorough evaluation of the model’s generalizability and robustness.

⸻

2. Test Data
The experiments utilize three distinct datasets:
	•	Dretain: data to be retained for continued training
	•	Dforget: data that the model is expected to unlearn
	•	Dunseen: unseen data used solely for evaluation

This structure enables clear assessment of both forgetting capability and task generalization.

⸻

3. Evaluation Metrics
	•	Test Accuracy: Measures the model’s performance on the original task (model utility)
	•	Forgetting Score: Evaluated using Membership Inference Attack (MIA) to determine how well the model has forgotten specific data
	•	Formula: Forgetting Score = |MIA Accuracy − 0.5| × 2
	•	Perfect forgetting yields a score of 0.0 (random guess)
	•	NoMUS (Normalized Machine Unlearning Score): A unified metric balancing utility and forgetting
	•	Formula: NoMUS = 0.5 × (Accuracy + (1 - Forgetting Score))
	•	Higher values indicate better overall performance

⸻

4. Comparisons
DLFD is compared against several state-of-the-art unlearning methods:
	•	Fine-tuning
	•	NegGrad (2020)
	•	CF-k, EU-k (2022)
	•	UNSIR (2023)
	•	BadTeaching (2023)
	•	SCRUB (2023)

Across nearly all tasks, DLFD achieved the highest NoMUS, demonstrating superior performance in both forgetting and task utility. The advantage was especially prominent in complex tasks like age and emotion classification, where feature entanglement is higher.

⸻




















<br/>
# 예제  




⸻



DLFD는 얼굴 이미지를 입력으로 받아 분류 과제를 수행하는 구조이기 때문에, 각 데이터는 이미지와 레이블로 구성됩니다. 다음은 각 데이터셋에서 사용되는 예시 형태입니다.

⸻

1. 트레이닝 데이터 (Dtrain = Dretain + Dforget)
	•	입력 이미지: 128×128 크기의 정규화된 얼굴 이미지 (예: RGB 3채널 이미지)
	•	레이블 (예시):
	•	연령 분류: y = 0 (010세), y = 1 (1120세), …, y = 7 (71세 이상)
	•	감정 분류: y = 0 (기쁨), y = 1 (슬픔), …, y = 6 (놀람)
	•	성별 분류: y = 0 (남성), y = 1 (여성)
	•	멀티 속성 분류: y = [1, 0, 1] (여성, 젊음, 웃는 얼굴)

예시:

# 연령 분류
x = image_128x128_tensor
y = 3  # 예: 31~40세 그룹

# 멀티 속성 분류
x = image_128x128_tensor
y = [1, 0, 1]  # (여성, 나이 젊음, 웃는 표정)



⸻

2. 테스트 데이터 (Dtest = Dunseen)
	•	학습에 사용되지 않은 새로운 얼굴 이미지로 구성됨
	•	각 이미지와 정답 레이블이 존재하며, **모델 유틸리티(정확도)와 언러닝 평가(MIA 점수)**를 위한 기준으로 사용됨

예시:

# 감정 인식 테스트 샘플
x = unseen_image_tensor
y = 2  # 예: 분노



⸻

3. 언러닝 평가용 데이터 (Dforget)
	•	사용자가 삭제 요청한 개인정보 이미지
	•	모델이 이 데이터를 기억하지 않아야 함
	•	Forgetting Score 평가에서 핵심적으로 사용됨

예시:

# 삭제 요청된 샘플
x = user_face_image_tensor
y = 1  # 예: 여성



⸻

DLFD operates on facial images for classification tasks. Each data point consists of an image and its corresponding label. Here’s what the actual data looks like:

⸻

1. Training Data (Dtrain = Dretain + Dforget)
	•	Input image: A normalized facial image of size 128×128 pixels (RGB format)
	•	Labels (examples):
	•	Age classification: y = 0 (0–10 years), y = 1 (11–20), …, y = 7 (71+)
	•	Emotion recognition: y = 0 (happy), y = 1 (sad), …, y = 6 (surprised)
	•	Gender classification: y = 0 (male), y = 1 (female)
	•	Multi-attribute classification: y = [1, 0, 1] (female, young, smiling)

Example:

# Age classification
x = image_128x128_tensor
y = 3  # e.g., age group 31–40

# Multi-attribute classification
x = image_128x128_tensor
y = [1, 0, 1]  # (female, young, smiling)



⸻

2. Test Data (Dtest = Dunseen)
	•	Unseen facial images not used during training
	•	Each sample has a ground-truth label and is used to evaluate model utility (accuracy) and forgetting performance (MIA score)

Example:

# Emotion recognition test sample
x = unseen_image_tensor
y = 2  # e.g., angry



⸻

3. Forgetting Data (Dforget)
	•	Contains personal images requested for deletion
	•	The model is expected to “forget” these instances
	•	Used to compute the Forgetting Score

Example:

# User-requested deletion sample
x = user_face_image_tensor
y = 1  # e.g., female



⸻

더 구체적인 포맷이 필요하거나 코드로 구성된 예제를 원하시면 말씀해주세요!


<br/>  
# 요약   




DLFD는 잊혀야 할 데이터와 유지할 데이터를 분포 수준에서 명확히 구분하여, 기존의 상관관계 붕괴 문제를 해결하고 모델 성능을 유지하는 머신 언러닝 기법이다. 실험 결과, 나이, 감정, 성별 분류 등 다양한 과제에서 기존 최신 기법보다 높은 정확도와 낮은 포겟팅 스코어를 기록했다. 실제 데이터는 128×128 얼굴 이미지와 분류 레이블로 구성되며, 멀티 속성 분류의 경우 [성별, 나이, 표정] 형식의 이진 벡터를 사용한다.

⸻



DLFD is a machine unlearning method that addresses correlation collapse by clearly separating the feature distributions of forget and retain data. Experimental results show that DLFD achieves higher accuracy and lower forgetting scores than previous methods across various tasks, including age, emotion, and gender classification. The input data consists of 128×128 facial images with labels, and multi-attribute classification uses binary vectors in the format [gender, age, expression].


<br/>  
# 기타  




⸻

[국문 설명] 기타 요소들

1. 주요 피규어(Figures)
	•	Figure 1: 상관관계 붕괴(Correlation Collapse)의 개념적 설명. 잘못된 방향의 언러닝이 태스크 관련 feature까지 훼손할 수 있음을 시각적으로 보여줌.
	•	Figure 2 & 4: 피처 공간에서의 분포 시각화. DLFD는 클래스 간 경계를 잘 유지하지만 기존 방법은 경계가 무너지는 현상을 보임.
	•	Figure 3: DLFD의 핵심 구조를 단계별로 설명한 흐름도. OT 손실을 기반으로 데이터 분포를 최적화하는 과정 포함.
	•	Figure 5: Forget 데이터와 Unseen 데이터의 손실 분포 비교. DLFD의 결과가 Retrained 모델과 유사한 분포를 형성함.
	•	Figure 6: 기존 방법의 에러-최대화 기법이 오히려 Forget 데이터의 손실 값을 비정상적으로 높이는 현상을 경고함.

⸻

2. 테이블(Tables)
	•	Table 1: ResNet18 모델 기준으로 다양한 언러닝 기법을 비교. DLFD가 가장 높은 NoMUS를 기록함.
	•	Table 2: 여러 아키텍처(ResNet, DenseNet, EfficientNet)에서의 DLFD 성능을 비교. 모든 모델에서 우수한 결과.
	•	Table 3: DLFD 구성요소별 성능 변화에 대한 ablation study 결과. 각 구성요소가 성능 향상에 기여함을 보여줌.

⸻

3. 어펜딕스(부록)
본 논문에는 어펜딕스가 별도로 명시되지는 않았지만, 실험 세팅, 추가 실험, 또는 확장 논의에 해당하는 내용이 본문 내 “Discussion”과 “Ablation Study” 섹션에서 다뤄집니다. 특히 정보 누출 위험성과 태스크 복잡도에 따른 trade-off에 대한 고찰이 실용적인 인사이트를 제공합니다.

⸻


1. Key Figures
	•	Figure 1: Conceptual illustration of correlation collapse, showing how misguided unlearning can harm task-relevant features.
	•	Figures 2 & 4: Visualizations of feature space. DLFD maintains clear class separation, unlike previous methods which show degraded boundaries.
	•	Figure 3: Workflow diagram of the DLFD method, detailing the optimization process using OT loss.
	•	Figure 5: Loss distribution comparisons between forget and unseen data. DLFD produces distributions similar to the ground-truth retrained model.
	•	Figure 6: Demonstrates the danger of naive error-maximization methods, which can abnormally increase forget data loss.

⸻

2. Tables
	•	Table 1: Performance comparison of unlearning methods using the ResNet18 model. DLFD achieves the highest NoMUS score.
	•	Table 2: Evaluation of DLFD across different architectures (ResNet, DenseNet, EfficientNet), showing consistently superior performance.
	•	Table 3: Ablation study showing the performance contributions of each DLFD component.

⸻

3. Appendix
Although the paper does not include a formal appendix section, additional discussions and analyses—such as experimental settings and observations about information leakage and task complexity—are provided in the Discussion and Ablation Study sections. These offer valuable practical insights into real-world unlearning challenges.

⸻




<br/>
# refer format:     



@inproceedings{choi2025distribution,
  title={Distribution-Level Feature Distancing for Machine Unlearning: Towards a Better Trade-off Between Model Utility and Forgetting},
  author={Choi, Dasol and Na, Dongbin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025},
  organization={Association for the Advancement of Artificial Intelligence}
}


Choi, Dasol, and Dongbin Na. 2025. “Distribution-Level Feature Distancing for Machine Unlearning: Towards a Better Trade-off Between Model Utility and Forgetting.” Proceedings of the AAAI Conference on Artificial Intelligence. Association for the Advancement of Artificial Intelligence.





