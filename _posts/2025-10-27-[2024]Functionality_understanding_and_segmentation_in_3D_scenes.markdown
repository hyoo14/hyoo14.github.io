---
layout: post
title:  "[2024]Functionality understanding and segmentation in 3D scenes"
date:   2025-10-27 20:38:11 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 3D 장면에서 기능 이해 및 분할을 위한 Fun3DU라는 새로운 방법을 제안합니다.  (LLM + VLM, -> LLama3.1 9B + Molmo(clip+vicuna) )  
추가적으로 sementation을 위한 모델들 사용  


짧은 요약(Abstract) :

이 논문의 초록에서는 3D 장면에서의 기능 이해와 세분화에 대한 내용을 다루고 있습니다. 기능 이해는 자연어 설명을 해석하여 3D 환경에서 상호작용 가능한 기능적 객체(예: 손잡이, 버튼 등)를 찾는 과정을 포함합니다. 이러한 기능 이해는 세계 지식과 공간 인식을 모두 요구하기 때문에 매우 도전적입니다. 예를 들어, "천장 불을 켜라"라는 작업이 주어졌을 때, AI 에이전트는 작업 설명에 명시되지 않은 스위치를 찾아야 합니다. 현재까지 이 문제를 해결하기 위한 전용 방법은 개발되지 않았습니다. 본 논문에서는 Fun3DU라는 첫 번째 접근 방식을 소개하며, 이는 사전 훈련된 비전 및 언어 모델을 활용하여 자연어 설명을 해석하고 기능적 객체를 세분화하는 방법입니다. Fun3DU는 훈련이 필요 없으며, SceneFun3D라는 데이터셋에서 평가하여 기존의 3D 세분화 방법보다 우수한 성능을 보였습니다.


The abstract of this paper discusses functionality understanding and segmentation in 3D scenes. Functionality understanding involves interpreting natural language descriptions to locate functional interactive objects (such as handles and buttons) in a 3D environment. This task is highly challenging as it requires both world knowledge and spatial perception. For instance, given a task like "turn on the ceiling light," an AI agent must locate the switch, which is not explicitly mentioned in the task description. To date, no dedicated methods have been developed for this problem. In this paper, we introduce Fun3DU, the first approach designed for functionality understanding in 3D scenes. Fun3DU leverages pre-trained vision and language models to interpret task descriptions and segment functional objects without requiring training. We evaluate Fun3DU on SceneFun3D, the only dataset available for benchmarking this task, and demonstrate that it significantly outperforms existing 3D segmentation methods.


짧은 질문(Questions) :  

Training-free 접근의 한계에 대해 어떻게 생각하시나요?
— 완전 사전학습 모델에 의존하는 만큼, unseen scene이나 domain shift 환경에서의 일반화 성능은 어떻게 보장하나요?

LLM이 functional object와 contextual object를 구분하는 과정에서 오류가 생기면, 파이프라인 전체에 어떤 영향을 주나요?
— 예를 들어 잘못된 reasoning이 segmentation 단계까지 전파될 수 있지 않나요?

Molmo + SAM 조합을 선택한 이유가 구체적으로 뭔가요?
— LLaVA나 GroundingDINO 같은 다른 VLM을 써봤을 때 차이가 있었나요?

View-selection의 KL-divergence 기반 점수 설계는 어떤 실험적 근거로 선택했나요?
— 단순히 mask confidence 기반보다 실제로 얼마나 개선됐는지 수치적으로 설명해주실 수 있나요?

Fun3DU가 LLM reasoning과 VLM segmentation을 결합했다고 했는데, 두 모듈 간의 불일치(conflict)는 어떻게 조정하나요?
— 예를 들어 LLM이 ‘handle’을 예측했지만 VLM이 이를 인식하지 못할 때 어떻게 처리하나요?



What are the main limitations of adopting a completely training-free approach?
— Since Fun3DU fully relies on pre-trained models, how can it generalize to unseen scenes or domains with different object distributions?

How does an error in the LLM’s reasoning affect the overall pipeline?
— For instance, if the LLM misidentifies the functional or contextual object, does that error propagate to the segmentation stage?

Why did you specifically choose the Molmo + SAM combination?
— Did you compare it with other VLMs such as LLaVA or Grounding DINO, and if so, what motivated this choice?

What motivated the use of KL-divergence for the view-selection score?
— How much quantitative improvement did it bring compared to simpler metrics like mask confidence or entropy?

How do you resolve conflicts between the LLM’s predictions and the VLM’s segmentation outputs?
— For example, when the LLM predicts a “handle” but the VLM fails to detect it, how is this disagreement handled?
  


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


**메서드: Fun3DU**

Fun3DU는 3D 장면에서 기능 이해 및 세분화를 위한 최초의 방법으로, 자연어 설명을 해석하여 실제 3D 환경에서 기능적 상호작용 객체를 세분화하는 데 중점을 둡니다. 이 방법은 세계 지식과 비전 인식 능력을 활용하여 사전 훈련된 비전 및 언어 모델을 기반으로 하며, 특정 작업에 대한 세부 조정 없이도 작동합니다.

#### 1. 아키텍처 구성 요소
Fun3DU는 네 가지 주요 모듈로 구성됩니다:

1. **작업 설명 이해 모듈**: 이 모듈은 대형 언어 모델(LLM)을 사용하여 작업 설명을 해석하고, 기능적 목표를 이해합니다. 예를 들어, "천장 조명을 켜라"라는 명령이 주어지면, LLM은 조명 스위치를 찾아야 한다는 것을 추론합니다. 이 과정에서 Chain-of-Thought 추론을 통해 작업을 수행하는 데 필요한 객체를 식별합니다.

2. **맥락적 객체 세분화 모듈**: 이 모듈은 각 뷰에서 맥락적 객체를 세분화합니다. 이를 위해 오픈 어휘 세분화 기법을 사용하여 정확성과 효율성을 높입니다. 또한, 시각적 정보의 가시성을 기반으로 한 뷰 선택 알고리즘을 통해 수천 개의 뷰 중에서 유용한 뷰만 선택합니다.

3. **기능적 객체 세분화 모듈**: 선택된 뷰에서 기능적 객체를 세분화합니다. 이 과정에서는 2D 비전 및 언어 모델(VLM)을 사용하여 세분화된 마스크를 생성합니다.

4. **다중 뷰 합의 모듈**: 이 모듈은 2D 마스크를 3D 포인트 클라우드로 변환하고, 여러 뷰에서의 결과를 통합하여 최종 3D 세분화 마스크를 생성합니다. 이 과정에서 포인트-픽셀 대응 관계를 활용하여 정확성을 높입니다.

#### 2. 훈련 데이터 및 기법
Fun3DU는 SceneFun3D라는 데이터셋을 사용하여 평가됩니다. 이 데이터셋은 3D 장면에서의 기능적 세분화를 위한 고해상도 포인트 클라우드와 다중 뷰 이미지를 포함하고 있으며, 3000개 이상의 작업 설명이 주어집니다. Fun3DU는 훈련이 필요 없는 방법으로 설계되어, 사전 훈련된 모델을 활용하여 작업 설명을 해석하고 기능적 객체를 세분화합니다.

#### 3. 성능
Fun3DU는 기존의 오픈 어휘 3D 세분화 방법들과 비교하여 상당한 성능 향상을 보여줍니다. 특히, 기능적 객체를 정확하게 세분화하는 데 필요한 깊은 추론 능력을 요구하는 작업에서 우수한 결과를 나타냅니다.




**Method: Fun3DU**

Fun3DU is the first method designed for functionality understanding and segmentation in 3D scenes, focusing on interpreting natural language descriptions to segment functional interactive objects in real-world 3D environments. This method leverages world knowledge and vision perception capabilities of pre-trained vision and language models, operating without the need for task-specific fine-tuning.

#### 1. Architecture Components
Fun3DU consists of four main modules:

1. **Task Description Understanding Module**: This module utilizes a Large Language Model (LLM) to interpret the task description and understand the functional goal. For instance, given the command "turn on the ceiling light," the LLM infers that it needs to locate the light switch. This process employs Chain-of-Thought reasoning to identify the objects necessary for completing the task.

2. **Contextual Object Segmentation Module**: This module segments contextual objects in each view using open-vocabulary segmentation techniques to enhance accuracy and efficiency. Additionally, a visibility-based view selection algorithm is employed to reduce the number of views from thousands to a few informative ones.

3. **Functional Object Segmentation Module**: This module segments the functional objects in the selected views using a 2D Vision and Language Model (VLM) to generate the segmented masks.

4. **Multi-View Agreement Module**: This module lifts the 2D masks into 3D using the camera poses for each view and aggregates the results into the final 3D segmentation masks. It utilizes point-to-pixel correspondences to enhance accuracy.

#### 2. Training Data and Techniques
Fun3DU is evaluated on the SceneFun3D dataset, which includes high-resolution point clouds and multi-view images for functionality segmentation in 3D scenes, along with over 3000 task descriptions. Fun3DU is designed as a training-free method, relying entirely on pre-trained models to interpret task descriptions and segment functional objects.

#### 3. Performance
Fun3DU demonstrates significant performance improvements compared to existing open-vocabulary 3D segmentation methods. It excels in tasks that require deeper reasoning capabilities necessary for accurately segmenting functional objects.


<br/>
# Results



이 논문에서는 Fun3DU라는 새로운 방법을 소개하며, 3D 장면에서의 기능 이해 및 세분화 문제를 다룹니다. 이 방법은 SceneFun3D라는 데이터셋을 사용하여 평가되었으며, 이 데이터셋은 230개의 장면과 3000개 이상의 작업 설명으로 구성되어 있습니다. 

#### 경쟁 모델
Fun3DU는 OpenMask3D, OpenIns3D, LERF와 같은 최신 오픈 어휘 3D 세분화 방법들과 비교되었습니다. 이들 모델은 기능적 객체를 세분화하는 데 있어 한계가 있으며, 특히 세부적인 기능적 객체를 정확하게 식별하는 데 어려움을 겪었습니다.

#### 테스트 데이터
SceneFun3D 데이터셋은 고해상도 RGBD 이미지와 3D 포인트 클라우드를 포함하고 있으며, 각 장면에 대해 평균 15개의 작업 설명이 제공됩니다. 이 데이터셋은 기능적 객체를 세분화하는 데 필요한 세밀한 기하학적 세부정보를 포함하고 있습니다.

#### 메트릭
평가 메트릭으로는 평균 정밀도(mAP), IoU(Intersection over Union), 평균 재현율(mAR) 등이 사용되었습니다. 특히, mAP는 IoU 임계값 0.25와 0.5에서 측정되며, 0.5에서 0.95까지의 IoU에 대한 평균을 계산하여 mAP를 도출합니다.

#### 비교 결과
Fun3DU는 두 개의 데이터셋(split0과 split1)에서 모두 우수한 성능을 보였습니다. 예를 들어, split0에서 Fun3DU는 mAP 7.6, AP 25 33.3, mIoU 15.2를 기록했습니다. 반면, OpenMask3D는 mAP 0.2, AP 25 0.4, mIoU 0.2에 불과했습니다. Fun3DU는 OpenIns3D와 LERF보다도 현저히 높은 성능을 보였으며, 특히 기능적 객체를 세분화하는 데 있어 더 나은 정밀도를 달성했습니다. 

결과적으로, Fun3DU는 기존의 오픈 어휘 3D 세분화 방법들보다 기능적 객체를 더 정확하게 세분화할 수 있는 능력을 보여주었으며, 이는 복잡한 3D 환경에서의 기능 이해에 있어 중요한 진전을 의미합니다.

---




This paper introduces a new method called Fun3DU, addressing the problem of functionality understanding and segmentation in 3D scenes. The method is evaluated using the SceneFun3D dataset, which consists of 230 scenes and over 3000 task descriptions.

#### Competing Models
Fun3DU is compared against state-of-the-art open-vocabulary 3D segmentation methods such as OpenMask3D, OpenIns3D, and LERF. These models have limitations in accurately segmenting functional objects, particularly struggling with fine-grained identification of functional elements.

#### Test Data
The SceneFun3D dataset includes high-resolution RGBD images and 3D point clouds, with an average of 15 task descriptions provided for each scene. This dataset contains detailed geometric information necessary for the precise localization of small functional objects.

#### Metrics
Evaluation metrics include mean Average Precision (mAP), Intersection over Union (IoU), and mean Average Recall (mAR). Specifically, mAP is measured at IoU thresholds of 0.25 and 0.5, and the mean is calculated over IoUs from 0.5 to 0.95.

#### Comparison Results
Fun3DU demonstrated superior performance on both splits of the dataset (split0 and split1). For instance, on split0, Fun3DU achieved an mAP of 7.6, an AP 25 of 33.3, and an mIoU of 15.2. In contrast, OpenMask3D only reached an mAP of 0.2, an AP 25 of 0.4, and an mIoU of 0.2. Fun3DU significantly outperformed both OpenIns3D and LERF, particularly in achieving higher precision in segmenting functional objects.

In conclusion, Fun3DU shows a remarkable ability to accurately segment functional objects compared to existing open-vocabulary 3D segmentation methods, marking a significant advancement in functionality understanding in complex 3D environments.


<br/>
# 예제



이 논문에서는 3D 장면에서 기능 이해 및 세분화를 위한 새로운 방법인 Fun3DU를 소개합니다. 이 방법은 자연어 설명을 해석하여 3D 환경에서 기능적 상호작용 객체(예: 손잡이, 버튼 등)를 찾아내는 것을 목표로 합니다. 

#### 트레이닝 데이터와 테스트 데이터

**트레이닝 데이터**: 
- **SceneFun3D 데이터셋**: 이 데이터셋은 3D 실내 환경의 고해상도 스캔을 포함하고 있으며, 각 장면에 대해 평균 15개의 작업 설명이 제공됩니다. 각 작업 설명은 특정 기능적 객체를 포함하고 있으며, 이 객체에 대한 마스크가 주어집니다. 예를 들어, "TV 위의 캐비닛의 아래 서랍을 열기"라는 작업 설명이 있을 수 있습니다. 이 경우, 서랍의 손잡이가 기능적 객체로 간주됩니다.

**테스트 데이터**: 
- 테스트 데이터는 SceneFun3D의 두 개의 분할(split0, split1)로 나뉘어 있으며, 각 분할은 다양한 장면과 작업 설명을 포함하고 있습니다. 예를 들어, "TV 옆의 캐비닛의 아래 서랍을 열기"라는 작업 설명이 주어질 수 있습니다. 이 경우, 로봇은 캐비닛을 찾고, 아래 서랍을 식별한 후, 서랍의 손잡이를 회전시켜야 합니다.

#### 구체적인 테스크 예시

1. **작업 설명**: "TV 위의 캐비닛의 아래 서랍을 열기"
   - **단계**:
     1. 캐비닛을 찾기
     2. 캐비닛의 아래 서랍을 식별하기
     3. 서랍에 접근하기
     4. 서랍의 손잡이를 회전시키기
   - **작용하는 객체**: 손잡이
   - **객체 계층**: ["캐비닛", "서랍"]

이러한 방식으로 Fun3DU는 자연어 설명을 기반으로 3D 장면에서 기능적 객체를 세분화하고, 로봇이 물리적 세계와 상호작용할 수 있도록 돕습니다.

---




This paper introduces a new method for functionality understanding and segmentation in 3D scenes called Fun3DU. The goal of this method is to interpret natural language descriptions to locate functional interactive objects (e.g., handles, buttons) in a 3D environment.

#### Training Data and Test Data

**Training Data**: 
- **SceneFun3D Dataset**: This dataset includes high-resolution scans of 3D indoor environments, with an average of 15 task descriptions provided for each scene. Each task description corresponds to specific functional objects, and masks for these objects are provided. For example, a task description might be "Open the bottom drawer of the cabinet with the TV on top." In this case, the handle of the drawer is considered the functional object.

**Test Data**: 
- The test data is divided into two splits (split0, split1) of SceneFun3D, each containing various scenes and task descriptions. For instance, a task description could be "Open the bottom drawer of the cabinet next to the TV." In this case, the robot needs to locate the cabinet, identify the bottom drawer, and then rotate the handle of the drawer.

#### Specific Task Example

1. **Task Description**: "Open the bottom drawer of the cabinet with the TV on top."
   - **Steps**:
     1. Locate the cabinet.
     2. Identify the bottom drawer of the cabinet.
     3. Approach the bottom drawer.
     4. Rotate the handle of the bottom drawer.
   - **Acted-on Object**: Handle
   - **Object Hierarchy**: ["Cabinet", "Drawer"]

In this way, Fun3DU helps segment functional objects in 3D scenes based on natural language descriptions, enabling robots to interact with the physical world.

<br/>
# 요약
이 논문에서는 3D 장면에서 기능 이해 및 분할을 위한 Fun3DU라는 새로운 방법을 제안합니다. 이 방법은 자연어 설명을 해석하여 기능적 상호작용 객체를 식별하고, 여러 뷰에서 세분화하여 3D 포인트 클라우드로 집계합니다. 실험 결과, Fun3DU는 기존의 최첨단 방법들보다 유의미하게 높은 성능을 보였습니다.

---

This paper introduces Fun3DU, a novel method for functionality understanding and segmentation in 3D scenes. The approach interprets natural language descriptions to identify functional interactive objects and segments them across multiple views, aggregating the results into a 3D point cloud. Experimental results show that Fun3DU significantly outperforms existing state-of-the-art methods.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: Fun3DU의 개념을 설명하는 다이어그램으로, 자연어 설명을 해석하여 3D 환경에서 기능적 객체를 세분화하는 과정을 보여줍니다. 이 다이어그램은 Fun3DU의 전반적인 구조와 작동 방식을 시각적으로 이해하는 데 도움을 줍니다.
   - **Figure 2**: Fun3DU의 네 가지 주요 모듈을 설명하는 다이어그램입니다. 각 모듈의 역할과 상호작용을 시각적으로 나타내어, 기능적 객체를 세분화하는 과정에서의 각 단계의 중요성을 강조합니다.
   - **Figure 4**: 시각적 객체의 가시성을 평가하는 방법을 보여주는 예시입니다. 이 피규어는 객체의 위치와 각도 분포를 분석하여 가시성을 평가하는 방법을 설명합니다.
   - **Figure 5**: Fun3DU와 다른 방법들의 예측 결과를 비교하는 시각적 결과입니다. 이 피규어는 Fun3DU가 기능적 객체를 정확하게 세분화하는 능력을 보여주며, 다른 방법들이 주로 맥락적 객체에 집중하는 경향이 있음을 강조합니다.

2. **테이블**
   - **Table 1 & 2**: Fun3DU와 여러 기준선 방법의 성능을 비교한 결과를 보여줍니다. Fun3DU는 mAP, AP 25, mIoU 등에서 다른 방법들보다 우수한 성능을 보이며, 특히 기능적 객체 세분화에서의 정확성을 강조합니다. 이 결과는 Fun3DU가 기능적 이해를 위한 새로운 접근 방식을 제공함을 나타냅니다.
   - **Table 3**: Fun3DU의 아키텍처 설계 선택이 최종 성능에 미치는 영향을 분석한 결과입니다. 각 구성 요소의 중요성을 보여주며, VLM을 사용하는 것이 성능 향상에 기여함을 나타냅니다.
   - **Table 4**: 뷰 선택 하이퍼파라미터의 영향을 평가한 결과입니다. 이 테이블은 뷰의 품질이 기능적 객체 세분화의 정확성에 미치는 영향을 강조합니다.

3. **어펜딕스**
   - 어펜딕스에는 추가적인 실험 결과와 방법론에 대한 세부 정보가 포함되어 있습니다. 이는 연구의 신뢰성을 높이고, 다른 연구자들이 이 방법을 재현할 수 있도록 돕습니다.

### Insights and Results from Figures, Tables, and Appendices

1. **Diagrams and Figures**
   - **Figure 1**: This diagram illustrates the concept of Fun3DU, showing how it interprets natural language descriptions to segment functional objects in a 3D environment. It helps in visually understanding the overall structure and operation of Fun3DU.
   - **Figure 2**: This figure describes the four main modules of Fun3DU. It visually represents the roles and interactions of each module, emphasizing the importance of each step in the process of segmenting functional objects.
   - **Figure 4**: This figure shows examples of how to evaluate the visibility of visual objects. It explains the method of analyzing the position and angle distributions of objects to assess visibility.
   - **Figure 5**: This visual result compares the predictions of Fun3DU with other methods. It highlights Fun3DU's ability to accurately segment functional objects while showing that other methods tend to focus on contextual objects.

2. **Tables**
   - **Table 1 & 2**: These tables present the performance comparison between Fun3DU and various baseline methods. Fun3DU outperforms others in metrics like mAP, AP 25, and mIoU, particularly emphasizing its accuracy in functional object segmentation. This indicates that Fun3DU offers a novel approach for functionality understanding.
   - **Table 3**: This table analyzes the impact of architectural design choices on the final performance of Fun3DU. It shows the importance of using a VLM, which contributes to performance improvement.
   - **Table 4**: This table assesses the impact of view selection hyperparameters. It emphasizes that the quality of views significantly affects the accuracy of functional object segmentation.

3. **Appendices**
   - The appendices contain additional experimental results and detailed methodology. This enhances the credibility of the research and aids other researchers in reproducing the method.

<br/>
# refer format:


### BibTeX 형식

```bibtex
@inproceedings{corsetti2024functionality,
  title={Functionality understanding and segmentation in 3D scenes},
  author={Corsetti, Jaime and Giuliari, Francesco and Fasoli, Alice and Boscaini, Davide and Poiesi, Fabio},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024},
  organization={Computer Vision Foundation}
}
```

### 시카고 스타일

Jaime Corsetti, Francesco Giuliari, Alice Fasoli, Davide Boscaini, and Fabio Poiesi. "Functionality Understanding and Segmentation in 3D Scenes." In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2024. Computer Vision Foundation.
