---
layout: post
title:  "[2024]OPENPI2.0 An Improved Dataset for Entity Tracking in Texts"  
date:   2024-07-24 15:29:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

짧은 요약(Abstract) :    

텍스트는 변화하는 세상을 묘사하는 경우가 많습니다(예: 절차, 이야기, 뉴스 기사). 이를 이해하려면 엔티티가 어떻게 변하는지 추적하는 것이 필요합니다. 이전 데이터셋인 OPENPI는 텍스트에서 엔티티 상태 변화를 크라우드소싱한 주석으로 제공했습니다. 그러나 이 주석들은 자유 형식이었고 중요한 변화를 식별하지 못해 모델 평가에 어려움이 있었습니다. 이러한 한계를 극복하기 위해, 우리는 엔티티와 속성이 완전히 정형화되고 추가적인 엔티티 중요도 주석이 포함된 개선된 데이터셋인 OPENPI2.0을 소개합니다. 보다 공정한 평가 환경에서, 현재 최첨단 언어 모델들이 여전히 충분히 유능하지 않음을 발견했습니다. 또한 중요한 엔티티의 상태 변화를 사슬 생각(chain-of-thought) 프롬프트로 사용하는 것이 질문 응답 및 고전적 계획 같은 작업에서 성능을 향상시킴을 보여주었습니다. 우리는 텍스트에서 엔티티의 동역학을 이해할 수 있는 모델 개발을 위해 OPENPI2.0을 제공합니다.


Much text describes a changing world (e.g., procedures, stories, newswires), and understanding them requires tracking how entities change. An earlier dataset, OPENPI, provided crowdsourced annotations of entity state changes in text. However, a major limitation was that those annotations were free-form and did not identify salient changes, hampering model evaluation. To overcome these limitations, we present an improved dataset, OPENPI2.0, where entities and attributes are fully canonicalized and additional entity salience annotations are added. On our fairer evaluation setting, we find that current state-of-the-art language models are far from competent. We also show that using state changes of salient entities as a chain-of-thought prompt, downstream performance is improved on tasks such as question answering and classical planning, outperforming the setting involving all related entities indiscriminately. We offer OPENPI2.0 for the continued development of models that can understand the dynamics of entities in text.

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

3.2 유틸리티: 엔티티 추적 평가
원래의 OPENPI 평가 세트와 마찬가지로, OPENPI2.0은 엔티티 추적에 대한 모델을 벤치마크하기 위해 설계되었습니다. 절차의 단계를 제공하여 엔티티와 속성이 겪는 상태 변화를 예측하는 것입니다. OPENPI2.0의 엔티티와 속성이 완전히 정형화되어, 평가가 더 공정하게 이루어질 수 있습니다. 시작으로, 우리는 Tandon et al. (2020)을 따라 모델이 완전한 문장을 예측하게 합니다: "엔티티의 속성은 전 상태에서 후 상태로 변화합니다", 그런 다음 이 문장을 실제 데이터의 문장과 비교합니다 (표 4). 우리는 또한 평가를 두 가지 하위 작업으로 더 세분화합니다: i. 주어진 단계에서 엔티티와 해당 속성을 예측하는 스키마타 예측 (예: "오븐을 켜다"는 단계에서, 랙의 온도가 상태 변화를 겪음), ii. 단계, 엔티티, 속성을 제공하여 상태 변화를 예측하는 작업 (예: 이전 정보가 주어지면 상태 변화는 차가움에서 뜨거움으로 변화함).


3.2 Utility: Evaluation of Entity Tracking
Just as the original evaluation set of OPENPI, OPENPI2.0 is meant to benchmark models on entity tracking – given a step in a procedure, predicting the state changes that entities and their attributes undergo. With the entities and attributes in OPENPI2.0 now fully canonicalized, evaluation can be done more fairly. To start with, we follow Tandon et al. (2020) and have models predict one complete sentence: “attribute of entity is pre-state before and post-state afterwards”, which is then compared to such sentences in the ground-truth data (Table 4). We further make the evaluation more fine-grained by formulating two sub-tasks: i. predicting schemata, namely the entities and their corresponding attributes given a step (e.g., given “turn on the oven”, the temperature of the rack undergo state changes), and ii. predicting the change of states given a step, an entity and an attribute (e.g., given the previous information, the state change is from cool to hot).


<br/>
# Results  

5. 결과 데이터셋: OPENPI2.0
OPENPI 데이터셋의 평가 세트에 엔티티와 속성의 정형화 및 엔티티 중요도를 추가함으로써, 우리는 이제 OPENPI2.0을 완전히 제공합니다. 절차와 엔티티 상태 주석은 변경되지 않았으므로 OPENPI2.0에는 여전히 55개의 절차가 평균 5.0단계를 가지고 있습니다. 이 절차들은 wikiHow에서 수집되었으며, 주제는 일상 활동입니다. OPENPI2.0은 크라우드소싱된 주석에 의해 원래 엔티티-속성-상태 변화를 계승합니다. 정형화 후, 개발 세트에는 평균적으로 7.6개의 고유 언급과 5.5개의 확장 언급을 가진 356개의 정형 엔티티, 3.0개의 고유 언급과 3.3개의 확장 언급을 가진 3240개의 정형 속성, 그리고 1193개의 전후 상태가 있습니다. 클러스터링 및 확장의 품질은 §3.1에서 입증될 수 있습니다. 중요도 레이블(1에서 5까지의 척도)과 관련하여, 엔티티의 글로벌 중요도의 평균은 3.5이고 표준 편차는 1.4입니다; 엔티티의 로컬 중요도의 평균은 3.4이고 표준 편차는 1.5입니다.


5 Resulting Dataset: OPENPI2.0
By adding canonicalization of entities and attributes as well as salience of entities to the evaluation set of the OPENPI dataset, we now fully present OPENPI2.0. As the procedures and entity state annotations have not changed, OPENPI2.0 still has 55 procedures with 5.0 steps on average. These procedures are collected from wikiHow and their topics are everyday activities. OPENPI2.0 also inherits the original entity-attribute-state changes annotated by crowd workers. After canonicalization, there are 356 canon entities each with 7.6 unique mentions and 5.5 expanded mentions on average, 3240 canon attributes, each with 3.0 unique mentions and 3.3 expanded mentions on average, and 1193 before-after states in the development set. The quality of clustering and expansion can be evidenced in §3.1. Regarding salience labels (on a scale of 1 to 5), the global salience of entities has a mean of 3.5 and standard deviation of 1.4; the local salience of entities has a mean of 3.4 and standard deviation of 1.5.

<br/>
# 예시  



텍스트에서 엔티티 추적의 효용성을 평가하기 위해 OPENPI2.0 데이터셋은 엔티티와 속성이 겪는 상태 변화를 예측하는 작업을 벤치마크로 사용합니다. 예를 들어, 어떤 사람이 케이크를 굽는 것을 목표로 하는 절차를 고려해봅시다. 각 단계에 대해 변화가 발생하는 엔티티와 속성을 나열합니다. 예를 들어, '오븐을 예열한다'는 단계에서는 랙(온도)이 맞지만 오븐(색깔)은 틀립니다.

다음은 더 구체적인 예제입니다:

단계: 달걀을 밀가루와 섞는다.
엔티티와 속성: 달걀(모양), 밀가루(색깔, 위치), 믹싱 볼(내용물, 무게)
이 예제에서, 모델은 달걀, 밀가루, 믹싱 볼과 같은 엔티티와 이들의 속성 변화를 예측합니다. 예를 들어, 달걀의 모양은 고체에서 액체로 변할 수 있으며, 밀가루는 색깔과 위치가 변할 수 있고, 믹싱 볼의 내용물과 무게도 변할 수 있습니다.

다음은 상태 예측 작업의 예입니다:

단계: 달걀을 밀가루와 섞는다.
엔티티와 속성: 달걀(모양: 고체 -> 액체), 밀가루(색깔: 흰색 -> 노란색, 위치: 상자 안 -> 믹싱 볼 안), 믹싱 볼(내용물: 빈 -> 채워짐, 무게: 가벼움 -> 무거움)
이러한 예제는 모델이 텍스트에서 엔티티의 동역학을 이해하고 예측할 수 있도록 돕기 위해 제공됩니다.



To evaluate the utility of entity tracking in text, the OPENPI2.0 dataset benchmarks models on the task of predicting state changes that entities and their attributes undergo. For example, consider a procedure where a person's goal is to bake a cake. For each step, list the entities and attributes that undergo any change. For example, for the step 'heat the oven', rack (temperature) is correct, while oven (color) is wrong.

Here is a more detailed example:

Step: Mix the eggs with flour.
Entities and attributes: eggs (shape), flour (color, location), mixing bowl (content, weight)
In this example, the model predicts the changes in entities such as eggs, flour, and mixing bowl and their attributes. For instance, the shape of the eggs may change from solid to liquid, the color and location of the flour may change, and the content and weight of the mixing bowl may also change.

Here is an example for the state prediction task:

Step: Mix the eggs with flour.
Entities and attributes: eggs (shape: solid -> liquid), flour (color: white -> yellow, location: in the box -> in the mixing bowl), mixing bowl (content: empty -> filled, weight: light -> heavy)
These examples are provided to help models understand and predict the dynamics of entities in text.

<br/>  
# 요약 

OPENPI2.0은 엔티티와 속성의 정형화 및 엔티티 중요도 주석을 포함한 개선된 데이터셋입니다.
현재 최첨단 언어 모델들은 공정한 평가 환경에서도 충분히 유능하지 않음을 발견했습니다.
중요한 엔티티의 상태 변화를 사슬 생각 프롬프트로 사용하면 다운스트림 작업 성능이 향상됩니다.
OPENPI2.0은 엔티티 추적을 위한 모델 개발을 위해 제공됩니다.
엔티티 추적 평가에서 엔티티와 속성의 변화를 예측하는 작업을 사용합니다.


OPENPI2.0 is an improved dataset that includes canonicalization of entities and attributes, and entity salience annotations.
Current state-of-the-art language models are found to be insufficiently competent even in a fair evaluation setting.
Using state changes of salient entities as a chain-of-thought prompt improves downstream task performance.
OPENPI2.0 is offered for the continued development of models for entity tracking.
The evaluation of entity tracking involves the task of predicting changes in entities and attributes.

# 기타  

OPENPI2.0은 크라우드소싱된 주석을 기반으로 엔티티 상태 변화를 추적하는 데이터셋입니다.
엔티티와 속성의 정형화로 인해 평가가 더 공정하게 이루어질 수 있습니다.
중요한 엔티티의 상태 변화를 예측함으로써 다운스트림 작업에서 비용을 절감할 수 있습니다.
이 데이터셋은 wikiHow에서 일상 활동에 대한 절차를 수집하여 구성되었습니다.
엔티티 중요도는 절차의 목표 달성에 대한 중요도와 각 단계의 중요도로 나뉘어 평가됩니다.


OPENPI2.0 is a dataset that tracks changes in entity states based on crowdsourced annotations.
The canonicalization of entities and attributes allows for fairer evaluation.
Predicting changes in salient entities can reduce costs in downstream tasks.
The dataset is composed of procedures collected from wikiHow on everyday activities.
Entity salience is evaluated based on both the importance to achieving the procedure's goal and the importance in each step.


<br/>
# refer format:     
Zhang, Li, Xu, Hainiu, Kommula, Abhinav, Callison-Burch, Chris, & Tandon, Niket. (2024). "OPENPI2.0: An Improved Dataset for Entity Tracking in Texts." arXiv preprint arXiv:2305.14603v2.

@article{zhang2024openpi2,
  title={OPENPI2.0: An Improved Dataset for Entity Tracking in Texts},
  author={Zhang, Li and Xu, Hainiu and Kommula, Abhinav and Callison-Burch, Chris and Tandon, Niket},
  journal={arXiv preprint arXiv:2305.14603v2},
  year={2024}
}



