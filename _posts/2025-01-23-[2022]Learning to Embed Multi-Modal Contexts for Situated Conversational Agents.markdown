---
layout: post
title:  "[2022]Learning to Embed Multi-Modal Contexts for Situated Conversational Agents"  
date:   2025-01-23 21:25:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

이미지와 텍스트를 모두 입력받는 멀티모달에서 ResNet-34과 BART를 통합(비전에서 단순히 얻은 object 이름뿐만 아니라 feature도 결합)하여 대화의 흐름을 이해하고 적절한 응답을 생성(BART가 잘 통합해서)    
(CLIP이전의 접근방식으로 볼 수 있음)   

짧은 요약(Abstract) :    




이 연구는 복합적인 멀티모달 입력을 처리할 수 있는 가상 쇼핑 도우미를 목표로 하는 SIMMC 2.0을 중심으로 합니다. 이 모델은 객체의 시각적 외형과 사용자 발화를 포함하는 멀티모달 입력을 기반으로 작동합니다. 연구는 다음과 같은 4가지 하위 작업을 포함합니다: 멀티모달 비모호화, 멀티모달 코리퍼런스 해결, 멀티모달 대화 상태 추적, 그리고 응답 검색 및 생성. 대부분의 작업 지향 대화 시스템은 이들 하위 작업을 개별적으로 처리하지만, 본 연구는 시각적 입력을 통합하고 모든 하위 작업을 한 번에 수행하는 통합 모델을 제안합니다. 이 접근법은 DSTC10에서 MM-Coref와 응답 검색 부문에서 우승했으며, 다른 하위 작업에서는 준우승을 기록했습니다. 이 연구는 멀티모달 작업 지향 대화 시스템의 새로운 기준을 제시합니다.

---



The Situated Interactive Multi-Modal Conversations (SIMMC) 2.0 aims to create virtual shopping assistants that can accept complex multi-modal inputs, i.e., visual appearances of objects and user utterances. It consists of four subtasks: multi-modal disambiguation (MM-Disamb), multi-modal coreference resolution (MM-Coref), multi-modal dialog state tracking (MM-DST), and response retrieval and generation. While many task-oriented dialog systems usually tackle each subtask separately, we propose a jointly learned multi-modal encoder-decoder that incorporates visual inputs and performs all four subtasks at once for efficiency. This approach won the MM-Coref and response retrieval subtasks and was nominated runner-up for the remaining subtasks using a single unified model at the 10th Dialog Systems Technology Challenge (DSTC10), setting a high bar for the novel task of multi-modal task-oriented dialog systems.



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




이 연구에서는 **SIMMC 2.0** 데이터셋을 사용하여 멀티모달 작업 지향 대화 시스템을 개발했습니다. 이 데이터셋은 가상현실(VR) 환경에서 사용자가 특정 작업을 수행하기 위해 대화하는 시뮬레이션 데이터를 포함하며, 각 대화는 시각적 및 비시각적 객체 속성을 포함합니다. 모델은 BART(Bidirectional and Auto-Regressive Transformers)를 기반으로 한 인코더-디코더 구조를 채택했으며, 시각적 및 언어적 모달리티를 통합하여 모든 하위 작업을 동시에 처리할 수 있는 통합 시스템을 제안했습니다.

훈련 데이터로는 VR 장면에서의 시뮬레이션 대화 데이터가 사용되었으며, 이 데이터는 4가지 주요 하위 작업을 포함합니다:
1. **멀티모달 비모호화(MM-Disamb)**: 대화 내에서 언급된 모호한 객체를 해소하는 작업.
2. **멀티모달 코리퍼런스 해결(MM-Coref)**: 대화에서 언급된 객체와 해당 장면의 객체를 매핑하는 작업.
3. **멀티모달 대화 상태 추적(MM-DST)**: 사용자의 의도를 추적하고 관련 정보를 기록.
4. **응답 검색 및 생성(Response Retrieval & Generation)**: 적절한 시스템 응답을 검색하거나 생성.

모델은 ResNet-34로 추출한 객체의 시각적 특징과 BART로 처리된 언어적 특징을 결합하여 멀티모달 표현을 생성합니다. 각 하위 작업은 공통 인코더를 공유하며, 특정 작업에 따라 분류 헤드나 언어 모델링 손실로 훈련됩니다. **하드 파라미터 공유(hard parameter sharing)** 방식으로 훈련해 모델 복잡도를 줄이고 과적합 위험을 낮췄습니다. 또한, 객체 속성(classification)과 비코리퍼런스(empty coreference) 예측과 같은 보조 과제를 추가로 학습하여 모델 성능을 강화했습니다.

최종적으로, 이 모델은 DSTC10 대회에서 여러 작업에서 우수한 성능을 기록하며 멀티모달 대화 시스템 개발을 위한 새로운 기준을 제시했습니다.

---


This study developed a multi-modal task-oriented dialogue system using the **SIMMC 2.0** dataset, which simulates conversations in a virtual reality (VR) setting where users interact to complete specific tasks. The dataset includes multimodal information, such as visual and non-visual attributes of objects. The model employs a BART (Bidirectional and Auto-Regressive Transformers)-based encoder-decoder architecture that integrates visual and linguistic modalities, enabling simultaneous handling of all subtasks in a unified framework.

The training data comprises simulated dialogue data from VR scenes, encompassing four main subtasks:
1. **Multi-modal Disambiguation (MM-Disamb)**: Resolving ambiguous references to objects within dialogues.
2. **Multi-modal Coreference Resolution (MM-Coref)**: Mapping mentioned objects in dialogues to corresponding scene objects.
3. **Multi-modal Dialogue State Tracking (MM-DST)**: Tracking user intents and recording relevant information.
4. **Response Retrieval and Generation**: Retrieving or generating appropriate system responses.

The model combines visual features extracted by ResNet-34 with linguistic features processed by BART to generate multi-modal representations. All subtasks share a common encoder, with task-specific heads for classification or language modeling losses. The training employs **hard parameter sharing** to reduce complexity and mitigate overfitting risks. Auxiliary tasks, such as object attribute classification and empty coreference prediction, were added to further enhance the model's performance.

This approach demonstrated outstanding performance across multiple subtasks in the DSTC10 competition, setting a new benchmark for developing multi-modal dialogue systems.


   
 
<br/>
# Results  




이 연구는 **SIMMC 2.0** 데이터셋의 테스트 세트를 사용하여 모델의 성능을 평가했습니다. 테스트 데이터는 15%의 데이터셋으로 구성되었으며, 4개의 주요 하위 작업(MM-Disamb, MM-Coref, MM-DST, Response Retrieval & Generation)을 평가하기 위한 다양한 메트릭이 사용되었습니다. 주된 메트릭은 다음과 같습니다:
- **MM-Disamb**: 정확도(Accuracy)
- **MM-Coref**: 객체 F1(Object F1)
- **MM-DST**: 슬롯 F1(Slot F1), 행위 F1(Action F1)
- **응답 검색**: MRR(Mean Reciprocal Rank), R@1, R@5, R@10, 평균 순위(Mean Rank)
- **응답 생성**: BLEU-4 점수

**BART 기반 모델**은 GPT-2 및 MTN(Multi-modal Transformer Networks) 등 기존 모델들과 비교되었으며, 모든 작업에서 뛰어난 성능을 보였습니다. 특히, 이 모델은 다음과 같은 성과를 기록했습니다:
- **MM-Coref**: 75.8% 객체 F1 (1위)
- **응답 검색**: MRR 81.5%, R@1 71.2%, R@5 95.0%, R@10 98.2%, 평균 순위 1.9 (1위)
- **MM-Disamb**: 93.8% 정확도 (준우승)
- **MM-DST**: 슬롯 F1 90.3%, 행위 F1 95.9% (준우승)

BART 모델은 단일 모델로 모든 하위 작업을 동시에 처리하면서 높은 성능을 달성했으며, 특히 MM-Coref 및 응답 검색 작업에서 다른 모델보다 월등한 결과를 보여줬습니다. 또한, ResNet-34 기반의 시각적 특징과 언어적 특징을 결합한 멀티모달 표현이 모델 성능에 크게 기여했습니다. 이 모델은 기존의 GPT-2와 MTN보다 최대 39% 이상의 객체 F1 향상을 기록하며 멀티모달 대화 시스템의 새로운 기준을 제시했습니다.

---



The study evaluated the model's performance using the **SIMMC 2.0** test dataset, which comprises 15% of the total data. Various metrics were employed to assess the model across the four main subtasks (MM-Disamb, MM-Coref, MM-DST, Response Retrieval & Generation):
- **MM-Disamb**: Accuracy
- **MM-Coref**: Object F1
- **MM-DST**: Slot F1, Action F1
- **Response Retrieval**: MRR (Mean Reciprocal Rank), R@1, R@5, R@10, Mean Rank
- **Response Generation**: BLEU-4 score

The **BART-based model** outperformed existing models such as GPT-2 and Multi-modal Transformer Networks (MTN) across all tasks, achieving the following results:
- **MM-Coref**: 75.8% Object F1 (Ranked 1st)
- **Response Retrieval**: MRR 81.5%, R@1 71.2%, R@5 95.0%, R@10 98.2%, Mean Rank 1.9 (Ranked 1st)
- **MM-Disamb**: 93.8% Accuracy (Runner-up)
- **MM-DST**: Slot F1 90.3%, Action F1 95.9% (Runner-up)

The BART model demonstrated superior performance, especially in the MM-Coref and response retrieval tasks, where it significantly outperformed other models. The integration of ResNet-34 visual features with linguistic features to create multi-modal representations contributed heavily to the model's success. Compared to GPT-2 and MTN, the BART model showed up to a 39% improvement in Object F1, setting a new benchmark for multi-modal conversational systems.



<br/>
# 예제  





**테스트 데이터 예제**:  
사용자는 "저렴한 빨간색과 흰색 블라우스가 있는지 보여줄 수 있나요?"라고 묻습니다.  
테스트 장면에는 여러 의류 객체가 있으며, 빨간색과 흰색 블라우스가 왼쪽 벽에 존재하지만, 유사한 색상의 옷들도 포함되어 있습니다.

**제안된 BART 기반 모델의 출력**:
- **MM-Disamb**: 사용자의 요청에서 모호성을 탐지하고, 특정 객체를 정확히 식별해야 함을 판단합니다.
- **MM-Coref**: 사용자의 언급(빨간색과 흰색 블라우스)을 정확히 장면의 특정 객체(객체 ID 7번)와 매핑합니다.
- **MM-DST**: 사용자가 요청한 블라우스의 속성(색상: 빨간색과 흰색, 가격: 저렴함)을 추적합니다.
- **응답 생성**: "왼쪽 벽에 빨간색과 흰색 블라우스가 하나 있습니다. 이 옵션이 마음에 드시나요?"와 같은 적절한 응답을 생성합니다.

**기존 GPT-2 모델의 출력**:
- **MM-Disamb**: 모호성을 탐지하지 못하고, 관련 없는 객체도 포함하여 응답을 생성합니다.
- **MM-Coref**: 객체 매핑에서 실패하거나 여러 객체를 잘못 매핑합니다.
- **MM-DST**: 요청한 속성(저렴한 가격)을 추적하지 못하고 일반적인 응답을 생성합니다.
- **응답 생성**: "여기 몇 가지 옵션이 있습니다. 어떤 것을 원하시나요?"와 같은 모호한 응답을 생성합니다.

**결과 비교**:  
BART 기반 모델은 테스트 데이터의 멀티모달 정보를 정확히 해석하여 사용자가 요청한 객체를 명확히 식별하고, 속성과 위치를 포함한 적절한 정보를 제공합니다. 반면, GPT-2 모델은 모호한 결과를 생성하여 사용자의 의도를 완전히 만족시키지 못합니다.

---



**Test Data Example**:  
The user asks, "Can you show me a cheap red and white blouse?"  
The test scene includes multiple clothing objects, with a red and white blouse on the left wall, but other similarly colored clothes are also present.

**Proposed BART-based Model Output**:
- **MM-Disamb**: Detects ambiguity in the user's request and determines that a specific object needs to be identified.
- **MM-Coref**: Correctly maps the user's reference ("red and white blouse") to a specific object in the scene (Object ID 7).
- **MM-DST**: Tracks the attributes of the requested blouse (color: red and white, price: cheap).
- **Response Generation**: Produces an appropriate response, such as "There is a red and white blouse on the left wall. Does this option suit you?"

**Baseline GPT-2 Model Output**:
- **MM-Disamb**: Fails to detect ambiguity and includes irrelevant objects in the response.
- **MM-Coref**: Fails to map the reference or incorrectly maps it to multiple objects.
- **MM-DST**: Does not accurately track the requested attributes (e.g., cheap price) and generates a generic response.
- **Response Generation**: Produces a vague response, such as "Here are some options. Which one do you like?"

**Comparison of Results**:  
The BART-based model effectively interprets the multimodal information in the test data, clearly identifying the requested object and providing accurate information, including its attributes and location. In contrast, the GPT-2 model generates ambiguous outputs, failing to fully satisfy the user's intent.


<br/>  
# 요약   



이 연구는 SIMMC 2.0 데이터셋을 사용하여 멀티모달 대화 시스템의 성능을 개선하고자 BART 기반 모델을 제안했습니다. 이 모델은 객체의 시각적 속성과 언어적 표현을 통합하여 멀티모달 비모호화, 코리퍼런스 해결, 대화 상태 추적, 응답 생성 등 네 가지 하위 작업을 동시에 수행합니다. 테스트 데이터에서 "저렴한 빨간색과 흰색 블라우스를 보여달라"는 요청에 대해, 제안된 모델은 객체의 색상, 가격, 위치를 정확히 식별하고 관련 정보를 제공했습니다. 반면, 기존 GPT-2 모델은 모호한 객체 매핑과 부정확한 응답을 생성하여 사용자의 요구를 충족시키지 못했습니다. 결과적으로, 제안된 모델은 명확한 객체 식별과 속성 추적을 통해 멀티모달 대화 시스템의 새로운 기준을 제시했습니다.

---



This study proposed a BART-based model to enhance the performance of multimodal conversational systems using the SIMMC 2.0 dataset. The model integrates visual attributes and linguistic expressions to simultaneously perform four subtasks: multimodal disambiguation, coreference resolution, dialog state tracking, and response generation. On test data, for a request like "Show me a cheap red and white blouse," the proposed model accurately identified the color, price, and location of the object and provided relevant information. In contrast, the baseline GPT-2 model failed to map the object correctly and generated vague responses, failing to meet user expectations. Consequently, the proposed model sets a new standard for multimodal conversational systems through precise object identification and attribute tracking.

<br/>  
# 기타  



1. **Figure 1**: SIMMC 2.0 데이터셋에서 대화와 장면의 예를 보여줍니다. 사용자가 파란색 후드 재킷과 베이지색 재킷 중 하나를 선택해야 하는 시나리오로, 시스템은 다중 모달 컨텍스트를 활용하여 대화의 맥락과 장면의 객체를 연결합니다.

2. **Table 1**: 모델의 입력 표현 예시를 보여줍니다. 대화의 이전 기록(히스토리), 사용자 발화, 장면 컨텍스트가 포함되며, 이를 통해 시스템이 객체 ID와 시각적 특징을 포함한 다중 모달 입력을 처리할 수 있도록 구성됩니다.

3. **Figure 2**: 제안된 다중 작업 통합 BART 모델의 구조를 시각적으로 나타냅니다. 장면 객체는 시각적 특징과 함께 표준화된 객체 ID로 표현되며, MM-Coref 및 속성 분류를 통해 대화 상태 추적과 응답 생성이 가능합니다.

4. **Table 2 및 3**: 개발 및 테스트 데이터 세트에 대한 결과를 요약한 테이블입니다. Table 2는 베이스라인 모델과 비교하여 제안된 모델의 성능을 보여주며, MM-Coref에서 75.8%의 객체 F1, 응답 검색에서 R@1 71.2%를 달성했습니다. Table 3은 DSTC10 경쟁의 공식 리더보드 결과로, 제안된 모델이 주요 작업에서 1위를 차지했음을 보여줍니다.

5. **Figure 3**: 대화 발화와 장면 객체 ID 간의 어텐션 맵을 시각화합니다. 모델이 특정 발화에서 참조된 객체를 올바르게 연결했는지 보여주며, 예를 들어 보라색 드레스와 같은 객체를 정확히 참조합니다.

---



1. **Figure 1**: Illustrates an example dialog and scene from the SIMMC 2.0 dataset. The scenario involves the user choosing between a blue hoodie jacket and a beige jacket, with the system leveraging multi-modal context to connect dialog references with scene objects.

2. **Table 1**: Displays an example of input representations for the model. It includes dialog history, user utterances, and scene context, allowing the system to process multi-modal inputs such as object IDs and visual features.

3. **Figure 2**: Shows the architecture of the proposed joint multi-tasking BART model. Scene objects are represented by their visual features and standardized object IDs, enabling MM-Coref, attribute classification, dialog state tracking, and response generation.

4. **Tables 2 and 3**: Summarize results on the development and test datasets. Table 2 compares the proposed model with baseline models, achieving 75.8% object F1 in MM-Coref and R@1 of 71.2% in response retrieval. Table 3 presents the official leaderboard from the DSTC10 competition, showing that the proposed model ranked first in major tasks.

5. **Figure 3**: Visualizes attention maps between dialog utterances and scene object IDs. It demonstrates how the model correctly links objects, such as identifying a purple dress from the given dialog reference.



<br/>
# 구조 및 구성:  


논문의 서술 구조는 전형적인 학술 논문의 구성 방식을 따르고 있으며, 다음과 같이 크게 6개의 주요 섹션으로 나뉩니다:

1. **서론 (Introduction)**  
   이 섹션에서는 연구의 배경, 동기, 그리고 SIMMC 2.0의 중요성을 강조합니다. 기존 작업 지향 대화 시스템의 한계를 짚고, 멀티모달 입력을 활용한 통합 시스템 개발의 필요성을 설명하며, 제안된 모델의 기여도를 간략히 소개합니다.

2. **관련 연구 (Related Work)**  
   기존의 대화 시스템, 멀티모달 학습 방법론, 그리고 SIMMC 데이터셋의 발전 과정에 대해 서술합니다. 이를 통해 현재 연구가 기존 연구와 어떻게 차별화되는지를 명확히 제시합니다.

3. **방법론 (Methodology)**  
   제안된 모델의 구조와 학습 방법을 상세히 설명합니다. SIMMC 2.0 데이터셋의 구성, 멀티모달 입력 표현 방식, 그리고 통합 BART 기반 모델의 아키텍처와 학습 절차가 포함되어 있습니다. 특히 각 하위 작업을 어떻게 모델이 처리하는지에 대한 설명이 중심을 이룹니다.

4. **실험 및 결과 (Experiments and Results)**  
   실험 설계와 평가 방법론을 설명하며, 테스트 데이터와 사용된 메트릭에 기반한 모델 성능 결과를 제시합니다. 기존 GPT-2 및 MTN 모델과의 비교를 통해 제안된 모델의 우월성을 입증하며, 주요 성과 지표와 함께 상세한 분석을 제공합니다.

5. **논의 (Discussion)**  
   실험 결과를 바탕으로 제안된 모델의 강점과 한계를 분석하며, 실제 적용 가능성 및 향후 연구 방향을 제시합니다. 특히, 멀티모달 대화 시스템 개발의 가능성과 남은 도전 과제에 대해 언급합니다.

6. **결론 (Conclusion)**  
   연구의 주요 성과를 요약하며, 본 연구가 멀티모달 대화 시스템 연구에 기여한 점을 강조합니다. 또한 향후 연구에서 확장될 수 있는 잠재적 방향을 간략히 언급하며 논문을 마무리합니다.

각 섹션은 명확한 문제 제기와 해결 방안을 중심으로 구성되어 있으며, 논리적인 흐름을 통해 독자가 연구의 중요성과 기여도를 쉽게 이해할 수 있도록 설계되었습니다.   



<br/>
# refer format:     



@inproceedings{lee2022learning,
  title={Learning to Embed Multi-Modal Contexts for Situated Conversational Agents},
  author={Lee, Haeju and Kwon, Oh Joon and Choi, Yunseon and Park, Minho and Han, Ran and Kim, Yoonhyung and Kim, Jinhyeon and Lee, Youngjune and Shin, Haebin and Lee, Kangwook and Kim, Kee-Eung},
  booktitle={Findings of the Association for Computational Linguistics: NAACL 2022},
  pages={813--830},
  year={2022},
  organization={Association for Computational Linguistics}
}






Lee, Haeju, Oh Joon Kwon, Yunseon Choi, Minho Park, Ran Han, Yoonhyung Kim, Jinhyeon Kim, Youngjune Lee, Haebin Shin, Kangwook Lee, and Kee-Eung Kim. "Learning to Embed Multi-Modal Contexts for Situated Conversational Agents." In Findings of the Association for Computational Linguistics: NAACL 2022, 813–830. Association for Computational Linguistics, 2022.










