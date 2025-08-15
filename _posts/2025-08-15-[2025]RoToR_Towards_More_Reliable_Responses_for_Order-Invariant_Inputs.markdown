---
layout: post
title:  "[2025]RoToR: Towards More Reliable Responses for Order-Invariant Inputs"
date:   2025-08-15 19:33:01 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

위치 ID 수정으로 진정한 순서 불변 입력을 위한 제로샷 언어 모델 RoToR 제안( 각 세그먼트에 대해 고정된 전역 순서를 사용하여 포지셔널 ID를 할당, 이는 각 쿼리에 대해 포지셔널 ID를 재할당하는 대신, 모든 쿼리에 대해 동일한 순서를 유지함으로써 모델의 일관성을 높임, 이 방법은 모델이 훈련 중에 학습한 포지셔널 정보를 유지하면서도, 입력의 순서에 대한 의존성을 줄임)  
이를 통해 리스트형 입력에 대한 순서 불변성을 개선  
Selective Routing을 통해 순서에 민감한 입력 선별하여 따로 처리  



즉, 새로 포지션 지정이 아니라 미리 절대 포지션 지정하는것..  



짧은 요약(Abstract) :

이 논문에서는 리스트형 입력에 대한 언어 모델의 위치 편향 문제를 해결하기 위한 새로운 접근법인 RoToR(순서 불변 입력에 대한 더 신뢰할 수 있는 응답을 위한 모델)를 제안합니다. 기존의 제로샷 순서 불변 언어 모델은 실제 리스트형 문제에서 제한된 성과를 보였으며, 이 논문에서는 두 가지 주요 한계를 식별하고 극복합니다. 첫째, 위치 ID 할당을 수정하여 불변성을 강제할 때 발생하는 훈련 및 추론 분포 불일치 문제를 해결하고, 둘째, 실제 리스트형 문제에서 순서에 민감한 입력과 불변 입력의 혼합에 적응하지 못하는 문제를 해결합니다. RoToR는 최소한의 위치 ID 수정으로 진정한 순서 불변 입력을 위한 제로샷 언어 모델을 제공하며, Selective Routing이라는 적응형 프레임워크를 통해 순서 불변 및 순서 민감 입력을 모두 처리할 수 있습니다. 실험 결과, RoToR와 Selective Routing이 실제 리스트형 입력 작업을 효과적으로 처리할 수 있음을 보여줍니다.


This paper proposes a new approach called RoToR (Reliable Responses for Order-Invariant Inputs) to address the positional bias problem of language models for listwise inputs. Existing zero-shot order-invariant language models have shown limited success on practical listwise problems, and this paper identifies and overcomes two key limitations. First, it addresses the training and inference distribution mismatch that arises from modifying positional ID assignments to enforce invariance. Second, it tackles the failure to adapt to a mixture of order-invariant and order-sensitive inputs in practical listwise problems. RoToR provides a zero-shot language model for genuinely order-invariant inputs with minimal modifications of positional IDs, and it introduces an adaptive framework called Selective Routing that can handle both order-invariant and order-sensitive inputs. Experimental results demonstrate that RoToR with Selective Routing can effectively manage practical listwise input tasks.


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


이 논문에서는 RoToR(Robust to Order)라는 새로운 언어 모델을 제안합니다. RoToR는 리스트형 입력에 대한 순서 불변성을 보장하기 위해 설계되었습니다. 기존의 언어 모델들은 입력의 순서에 민감하여, 같은 정보라도 순서에 따라 다른 결과를 도출하는 경향이 있습니다. 이러한 문제를 해결하기 위해 RoToR는 두 가지 주요 기법을 사용합니다.

1. **전역 정렬(Global Sorting)**: RoToR는 입력의 각 세그먼트에 대해 고정된 전역 순서를 사용하여 포지셔널 ID를 할당합니다. 이는 각 쿼리에 대해 포지셔널 ID를 재할당하는 대신, 모든 쿼리에 대해 동일한 순서를 유지함으로써 모델의 일관성을 높입니다. 이 방법은 모델이 훈련 중에 학습한 포지셔널 정보를 유지하면서도, 입력의 순서에 대한 의존성을 줄입니다.

2. **선택적 라우팅(Selective Routing)**: RoToR는 입력이 순서에 민감한 경우와 그렇지 않은 경우를 구분하여 처리할 수 있는 메커니즘을 제공합니다. 이 방법은 두 개의 모델(원래 모델과 RoToR 모델)에서 각각의 예측 확률을 계산한 후, 더 높은 확률을 가진 모델의 출력을 선택합니다. 이를 통해 RoToR는 다양한 입력 유형에 대해 더 나은 성능을 발휘할 수 있습니다.

RoToR는 Lost in the Middle (LitM), Knowledge Graph QA (KGQA), 그리고 MMLU와 같은 여러 벤치마크에서 실험을 통해 그 효과성을 입증하였습니다. 실험 결과, RoToR는 기존의 모델들보다 더 높은 정확도와 안정성을 보여주었으며, 특히 입력의 순서가 변경되었을 때에도 성능 저하가 적었습니다.




This paper proposes a new language model called RoToR (Robust to Order), designed to ensure order invariance for listwise inputs. Existing language models tend to be sensitive to the order of inputs, leading to different outcomes based on the same information presented in varying sequences. To address this issue, RoToR employs two main techniques:

1. **Global Sorting**: RoToR assigns positional IDs to each segment of the input based on a fixed global order. Instead of reassigning positional IDs for each query, it maintains the same order across all queries, enhancing the model's consistency. This approach reduces the model's dependency on the order of inputs while preserving the positional information learned during training.

2. **Selective Routing**: RoToR provides a mechanism to differentiate between order-sensitive and order-invariant inputs. It calculates the prediction probabilities from two models (the original model and the RoToR model) and selects the output from the model with the higher probability. This allows RoToR to perform better across various types of inputs.

RoToR has been experimentally validated on several benchmarks, including Lost in the Middle (LitM), Knowledge Graph QA (KGQA), and MMLU. The results demonstrate that RoToR achieves higher accuracy and stability compared to existing models, particularly when the order of inputs is altered, showing minimal performance degradation.


<br/>
# Results


이 논문에서는 RoToR(Reliable Responses for Order-Invariant Inputs)라는 새로운 모델을 제안하여 언어 모델의 순서 불변 입력에 대한 신뢰성을 높이고자 하였습니다. 기존의 언어 모델들은 입력의 순서에 민감하여, 특히 리스트 형식의 입력에서 성능 저하를 보이는 경향이 있습니다. 이 문제를 해결하기 위해 RoToR는 두 가지 주요 기여를 통해 성능을 개선하였습니다.

1. **모델 성능 비교**: RoToR는 기존의 여러 모델들과 비교하여 성능을 평가하였습니다. 특히, PINE, PCW, Set-Based Prompting과 같은 기존의 순서 불변 모델들과 비교하였으며, 다양한 벤치마크 데이터셋인 Lost in the Middle (LitM), Knowledge Graph QA (KGQA), MMLU에서 실험을 진행하였습니다. RoToR는 모든 설정에서 가장 높은 성능을 기록하였으며, 특히 LitM 벤치마크에서는 30개의 문서에서 61.4%의 정확도를 달성하였습니다. 반면, PINE은 58.6%의 정확도를 기록하여 RoToR에 비해 낮은 성능을 보였습니다.

2. **테스트 데이터 및 메트릭**: RoToR는 다양한 테스트 데이터셋에서 평가되었습니다. LitM에서는 문서의 순서를 무작위로 섞었을 때도 RoToR는 성능 저하가 없었으며, 이는 모델의 강건성을 나타냅니다. KGQA에서는 RoToR가 PINE보다 낮은 표준 편차를 기록하여, 입력 순서에 대한 안정성을 보여주었습니다. MMLU에서는 RoToR가 Selective Routing을 통해 순서 민감한 입력을 효과적으로 처리하여, 원래 모델과 비슷한 성능을 유지하였습니다.

3. **비교 결과**: RoToR는 기존 모델들에 비해 계산 비용이 낮고, 실행 시간도 단축되었습니다. 예를 들어, LitM에서 RoToR는 30개의 문서에 대해 43%의 실행 시간 감소를 기록하였습니다. 또한, RoToR는 PINE에 비해 더 적은 수의 충돌을 발생시켜, 입력의 순서 불변성을 유지하는 데 성공하였습니다.

결론적으로, RoToR는 순서 불변 입력을 처리하는 데 있어 기존 모델들보다 더 신뢰할 수 있는 성능을 보여주며, 다양한 입력 형식에 대해 강건한 결과를 제공합니다.

---



In this paper, a new model called RoToR (Reliable Responses for Order-Invariant Inputs) is proposed to enhance the reliability of language models for order-invariant inputs. Existing language models tend to be sensitive to the order of inputs, particularly in listwise formats, leading to performance degradation. To address this issue, RoToR improves performance through two main contributions.

1. **Model Performance Comparison**: RoToR was evaluated against several existing models, including PINE, PCW, and Set-Based Prompting, across various benchmark datasets such as Lost in the Middle (LitM), Knowledge Graph QA (KGQA), and MMLU. RoToR achieved the highest performance across all setups, recording an accuracy of 61.4% on the LitM benchmark with 30 documents. In contrast, PINE recorded an accuracy of 58.6%, demonstrating lower performance compared to RoToR.

2. **Test Data and Metrics**: RoToR was evaluated on diverse test datasets. In LitM, even when the order of documents was shuffled, RoToR showed no significant performance drop, indicating the model's robustness. In KGQA, RoToR exhibited lower standard deviation compared to PINE, showcasing stability against input order variations. In MMLU, RoToR effectively handled order-sensitive inputs through Selective Routing, maintaining performance similar to the original model.

3. **Comparison Results**: RoToR demonstrated lower computational costs and reduced execution times compared to existing models. For instance, in LitM, RoToR achieved a 43% reduction in total runtime with 30 documents. Additionally, RoToR resulted in fewer collisions than PINE, successfully maintaining the invariance of input order.

In conclusion, RoToR provides a more reliable performance in processing order-invariant inputs compared to existing models, delivering robust results across various input formats.


<br/>
# 예제


이 논문에서는 RoToR(Robust to Order)라는 새로운 언어 모델을 제안하고, 이를 통해 리스트형 입력에 대한 순서 불변성을 개선하는 방법을 다룹니다. 이 모델은 특히 "Lost in the Middle" (LitM), "Knowledge Graph QA" (KGQA), 그리고 "Massive Multitask Language Understanding" (MMLU)와 같은 벤치마크에서 성능을 평가합니다.

#### 예시 1: Lost in the Middle (LitM)
- **입력**: 여러 개의 문서가 주어지고, 그 중에서 특정 질문에 대한 답변을 찾아야 합니다. 예를 들어, "누가 첫 번째 노벨 물리학상을 받았나요?"라는 질문이 주어질 수 있습니다. 이 질문에 대한 답변은 여러 문서에서 찾아야 하며, 문서의 순서는 중요하지 않습니다.
- **출력**: 모델은 "알베르트 아인슈타인"과 같은 정확한 답변을 JSON 형식으로 출력합니다. 예를 들어, `{"Answer": "알베르트 아인슈타인"}`.

#### 예시 2: Knowledge Graph QA (KGQA)
- **입력**: 지식 그래프의 사실들이 주어집니다. 예를 들어, (슈퍼볼 XLII, 우승팀, 뉴욕 자이언츠)와 같은 형식으로 주어질 수 있습니다. 질문은 "슈퍼볼 XLII MVP는 어떤 팀에서 뛰었나요?"와 같은 형태일 수 있습니다.
- **출력**: 모델은 JSON 형식으로 답변을 출력합니다. 예를 들어, `{"Answer": "뉴욕 자이언츠"}`.

#### 예시 3: MMLU
- **입력**: 다수의 선택지가 있는 질문이 주어집니다. 예를 들어, "노르크로스는 어떤 주장을 하나?"라는 질문과 함께 선택지 A, B, C, D가 주어질 수 있습니다.
- **출력**: 모델은 가장 높은 확률을 가진 선택지를 선택하여 출력합니다. 예를 들어, `{"Answer": "C"}`.

이러한 예시들은 RoToR 모델이 다양한 리스트형 입력을 처리하는 데 있어 어떻게 작동하는지를 보여줍니다. RoToR는 입력의 순서에 영향을 받지 않도록 설계되어, 모델이 더 안정적이고 신뢰할 수 있는 답변을 제공할 수 있도록 합니다.

---



This paper proposes a new language model called RoToR (Robust to Order) and discusses how to improve the order invariance for listwise inputs. The model is evaluated on benchmarks such as "Lost in the Middle" (LitM), "Knowledge Graph QA" (KGQA), and "Massive Multitask Language Understanding" (MMLU).

#### Example 1: Lost in the Middle (LitM)
- **Input**: A set of documents is provided, and the task is to find an answer to a specific question. For instance, the question could be "Who received the first Nobel Prize in Physics?" The answer must be found across multiple documents, and the order of the documents is irrelevant.
- **Output**: The model outputs the correct answer in JSON format, such as `{"Answer": "Albert Einstein"}`.

#### Example 2: Knowledge Graph QA (KGQA)
- **Input**: Facts in the form of a knowledge graph are given. For example, (Super Bowl XLII, winner, New York Giants) could be one of the facts. The question might be "Which team did the Super Bowl XLII MVP play for?"
- **Output**: The model outputs the answer in JSON format, such as `{"Answer": "New York Giants"}`.

#### Example 3: MMLU
- **Input**: A multiple-choice question is presented. For example, "What does Norcross argue?" along with options A, B, C, and D.
- **Output**: The model selects the option with the highest probability and outputs it, such as `{"Answer": "C"}`.

These examples illustrate how the RoToR model operates in processing various listwise inputs. RoToR is designed to be unaffected by the order of inputs, allowing the model to provide more stable and reliable answers.

<br/>
# 요약
이 논문에서는 RoToR라는 새로운 언어 모델을 제안하여 리스트형 입력에 대한 순서 불변성을 개선하고, Selective Routing을 통해 순서에 민감한 입력을 효과적으로 처리하는 방법을 소개합니다. 실험 결과, RoToR는 기존 모델들보다 더 높은 성능을 보였으며, 특히 입력 순서가 변경되었을 때도 안정적인 결과를 유지했습니다. 이 연구는 리스트형 입력 처리의 신뢰성을 높이는 데 기여하며, 다양한 실제 문제에 적용 가능성을 보여줍니다.

---

This paper introduces a new language model called RoToR, which improves order invariance for listwise inputs and effectively handles order-sensitive inputs through Selective Routing. Experimental results show that RoToR outperforms existing models and maintains stable performance even when the input order is changed. This research contributes to enhancing the reliability of listwise input processing and demonstrates applicability to various real-world problems.

<br/>
# 기타

1. **다이어그램 및 피규어**:
   - **Figure 1**: RoToR의 자기 주의 메커니즘을 설명하며, 기존 모델과의 차별점을 보여줍니다. RoToR는 순환 배치를 통해 포지션 ID를 최소한으로 수정하여 분포 불일치를 줄입니다.
   - **Figure 3**: RoToR의 순환 배치 방식을 시각적으로 설명합니다. 이는 각 세그먼트가 고정된 순서로 배치되도록 하여, 쿼리와 관계없이 일관된 포지션 ID를 유지합니다.
   - **Figure 4**: 선택적 라우팅 메커니즘을 설명하며, 두 모델(원본 및 RoToR) 간의 신뢰도 점수를 비교하여 더 높은 신뢰도를 가진 모델의 출력을 선택하는 과정을 보여줍니다.

2. **테이블**:
   - **Table 1**: Lost in the Middle (LitM) 벤치마크에서 RoToR의 성능을 보여줍니다. RoToR는 모든 설정에서 가장 높은 성능을 기록하며, 특히 인덱스 편향이 제거된 경우에 안정적인 성능을 보입니다.
   - **Table 2**: KGQA 벤치마크에서 RoToR의 성능을 비교합니다. RoToR는 PINE보다 낮은 표준 편차로 더 일관된 성능을 보여줍니다.
   - **Table 3**: MMLU 벤치마크에서 선택적 라우팅을 적용한 RoToR의 성능을 보여줍니다. 선택적 라우팅을 통해 RoToR는 원본 모델과 경쟁할 수 있는 성능을 발휘합니다.
   - **Table 4**: PINE과 RoToR의 효율성을 비교합니다. RoToR는 FLOPs와 지연 시간에서 PINE보다 우수한 성능을 보이며, 특히 세그먼트 수가 증가할수록 효율성이 향상됩니다.

3. **어펜딕스**:
   - 어펜딕스에서는 다양한 실험 세부사항과 추가적인 결과를 제공합니다. 예를 들어, 선택적 라우팅의 하이퍼파라미터 조정 결과와 다양한 글로벌 정렬 전략의 성능을 비교합니다. 이는 RoToR의 유연성과 강력한 성능을 강조합니다.




1. **Diagrams and Figures**:
   - **Figure 1**: Illustrates the self-attention mechanism of RoToR, highlighting its differences from existing models. RoToR minimizes distribution mismatch through circular arrangements with minimal modifications to positional IDs.
   - **Figure 3**: Visually explains the circular arrangement method of RoToR, ensuring that each segment is placed in a fixed order, maintaining consistent positional IDs regardless of the query.
   - **Figure 4**: Depicts the Selective Routing mechanism, comparing confidence scores between the two models (original and RoToR) to select the output from the model with higher confidence.

2. **Tables**:
   - **Table 1**: Shows the performance of RoToR on the Lost in the Middle (LitM) benchmark. RoToR consistently achieves the highest performance across all setups, particularly demonstrating stability when index bias is removed.
   - **Table 2**: Compares RoToR's performance on the KGQA benchmark. RoToR exhibits lower standard deviation, indicating more consistent performance compared to PINE.
   - **Table 3**: Displays the performance of RoToR with Selective Routing on the MMLU benchmark. Selective Routing allows RoToR to compete effectively with the original model.
   - **Table 4**: Compares the efficiency of PINE and RoToR. RoToR outperforms PINE in terms of FLOPs and latency, with efficiency gains increasing as the number of segments increases.

3. **Appendix**:
   - The appendix provides various experimental details and additional results. For instance, it includes the results of hyperparameter tuning for Selective Routing and compares the performance of different global sorting strategies, emphasizing RoToR's flexibility and robust performance.

<br/>
# refer format:


### BibTeX 형식
```bibtex
@inproceedings{yoon2025rotor,
  title={RoToR: Towards More Reliable Responses for Order-Invariant Inputs},
  author={Soyoung Yoon and Dongha Ahn and Youngwon Lee and Minkyu Jung and HyungJoo Jang and Seung-won Hwang},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={18739--18760},
  year={2025},
  month={July},
  publisher={Association for Computational Linguistics},

}
```

### 시카고 스타일 인용
Yoon, Soyoung, Dongha Ahn, Youngwon Lee, Minkyu Jung, HyungJoo Jang, and Seung-won Hwang. "RoToR: Towards More Reliable Responses for Order-Invariant Inputs." In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 18739–18760. Association for Computational Linguistics, July 2025.
