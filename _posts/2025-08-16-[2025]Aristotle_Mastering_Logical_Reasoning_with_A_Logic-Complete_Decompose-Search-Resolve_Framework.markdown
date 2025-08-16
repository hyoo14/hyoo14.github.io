---
layout: post
title:  "[2025]Aristotle: Mastering Logical Reasoning with A Logic-Complete Decompose-Search-Resolve Framework"
date:   2025-08-16 21:30:36 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 


시스템이 추론 과정에서 문제를 분해하고, 검색하며, 해결하는 데 있어 논리적 작업의 고유한 구조를 충분히 활용하지 못해서 
논리 분해기(Logical Decomposer), 논리 검색 라우터(Logical Search Router), 논리 해결기(Logical Resolver) 제안  
기호 표현과 논리 규칙을 전체 추론 과정에 통합하여, 하위 작업의 복잡성을 줄이고, 검색 오류를 최소화하며, 논리적 모순을 해결  

짧은 요약(Abstract) :


현재의 대형 언어 모델(LLMs)은 다양한 추론 작업에서 인상적인 발전을 이루었지만, 논리적 추론 작업에서는 여전히 효율성과 효과성에서 주요한 도전 과제가 남아 있습니다. 이는 이러한 시스템이 추론 과정에서 문제를 분해하고, 검색하며, 해결하는 데 있어 논리적 작업의 고유한 구조를 충분히 활용하지 못하기 때문입니다. 이를 해결하기 위해, 우리는 'Aristotle'이라는 논리 완전 추론 프레임워크를 제안합니다. 이 프레임워크는 세 가지 주요 구성 요소인 논리 분해기(Logical Decomposer), 논리 검색 라우터(Logical Search Router), 논리 해결기(Logical Resolver)를 포함합니다. 우리의 프레임워크는 기호 표현과 논리 규칙을 전체 추론 과정에 통합하여, 하위 작업의 복잡성을 줄이고, 검색 오류를 최소화하며, 논리적 모순을 해결하는 데 기여합니다. 여러 데이터셋에 대한 실험 결과는 Aristotle이 정확성과 효율성 모두에서 최첨단 추론 프레임워크를 지속적으로 초월함을 보여줍니다. 특히 복잡한 논리적 추론 시나리오에서 두드러진 성과를 보였습니다.



Current advanced reasoning methods in large language models (LLMs) have made impressive strides in various reasoning tasks. However, significant challenges remain in both efficacy and efficiency when it comes to logical reasoning tasks. This is rooted in the fact that these systems fail to fully leverage the inherent structure of logical tasks throughout the reasoning processes, such as decomposition, search, and resolution. To address this, we propose a logic-complete reasoning framework, Aristotle, which consists of three key components: Logical Decomposer, Logical Search Router, and Logical Resolver. Our framework comprehensively integrates symbolic expressions and logical rules into the entire reasoning process, significantly alleviating the bottlenecks of logical reasoning, i.e., reducing sub-task complexity, minimizing search errors, and resolving logical contradictions. Experimental results on several datasets demonstrate that Aristotle consistently outperforms state-of-the-art reasoning frameworks in both accuracy and efficiency, particularly excelling in complex logical reasoning scenarios.


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



**Aristotle: 논리적 추론을 위한 완전한 프레임워크**

Aristotle은 논리적 추론을 위한 새로운 프레임워크로, 세 가지 주요 구성 요소인 논리 분해기(Logical Decomposer), 논리 검색 라우터(Logical Search Router), 논리 해결기(Logical Resolver)를 통합하여 설계되었습니다. 이 프레임워크는 기존의 대형 언어 모델(LLM)이 논리적 작업을 수행할 때 직면하는 효율성과 효과성의 문제를 해결하기 위해 고안되었습니다.

1. **논리 분해기 (Logical Decomposer)**: 이 모듈은 주어진 문제를 더 작고 단순한 구성 요소로 분해합니다. 이를 통해 복잡한 논리적 작업의 복잡성을 줄이고, 각 구성 요소가 명확하게 정의된 논리적 구조를 따르도록 합니다. 이 과정에서 LLM을 사용하여 원래의 전제와 질문을 기호적 형식으로 변환하고, 이를 정규화(Normalization) 및 스코렐화(Skolemization)하여 정규형으로 변환합니다.

2. **논리 검색 라우터 (Logical Search Router)**: 이 모듈은 모순 증명을 기반으로 하여 논리적 불일치를 직접 검색합니다. 이를 통해 신뢰할 수 없는 평가자에 의한 검색 오류를 줄이고, 기존 방법보다 적은 단계로 문제를 해결할 수 있도록 합니다. 이 과정에서 현재의 절차와 상충되는 조항을 찾고, 이를 통해 논리적 모순을 탐지합니다.

3. **논리 해결기 (Logical Resolver)**: 이 모듈은 각 추론 단계에서 논리적 모순을 엄격하게 해결합니다. 검색 라우터가 제공한 정보를 바탕으로, 해결기는 상충되는 용어를 제거하고 나머지 용어를 연결하여 새로운 절차를 생성합니다. 이 과정에서 LLM을 사용하여 논리적 결론을 도출합니다.

이러한 구성 요소들은 서로 긴밀하게 연결되어 있으며, 각 단계에서 기호적 표현과 논리적 규칙을 통합하여 논리적 추론의 일관성을 높입니다. 실험 결과, Aristotle은 여러 논리적 추론 벤치마크에서 기존의 최첨단 프레임워크보다 4.5%에서 5.4% 더 높은 정확도를 기록하며, 특히 복잡한 논리적 시나리오에서 두드러진 성능 향상을 보였습니다.




**Aristotle: A Logic-Complete Framework for Logical Reasoning**

Aristotle is a novel framework designed for logical reasoning, integrating three key components: the Logical Decomposer, Logical Search Router, and Logical Resolver. This framework aims to address the efficacy and efficiency challenges faced by existing large language models (LLMs) when performing logical tasks.

1. **Logical Decomposer**: This module breaks down the given problem into smaller and simpler components. By doing so, it reduces the complexity of logical tasks and ensures that each component adheres to a clearly defined logical structure. During this process, an LLM is employed to translate the original premises and questions into a symbolic format, which is then normalized and skolemized to convert them into a standard form.

2. **Logical Search Router**: This module utilizes proof by contradiction to directly search for logical inconsistencies. By doing so, it minimizes search errors caused by unreliable evaluators and allows for problem resolution in fewer steps compared to existing methods. In this process, it identifies clauses that conflict with the current reasoning path, thereby detecting logical contradictions.

3. **Logical Resolver**: This module rigorously resolves logical contradictions at each reasoning step. Guided by the information provided by the Search Router, the Resolver cancels out conflicting terms and connects the remaining terms to produce new clauses. An LLM is used in this process to derive logical conclusions.

These components are tightly interconnected, fully integrating symbolic expressions and logical rules into each stage of the reasoning process, thereby enhancing the coherence of logical reasoning. Experimental results demonstrate that Aristotle consistently outperforms state-of-the-art frameworks by 4.5% to 5.4% in accuracy across various logical reasoning benchmarks, particularly excelling in complex logical scenarios.


<br/>
# Results



이 논문에서는 "Aristotle"이라는 새로운 논리적 추론 프레임워크를 제안하고, 이를 기존의 여러 경쟁 모델들과 비교하여 성능을 평가했습니다. 실험은 세 가지 주요 데이터셋인 ProntoQA, ProofWriter, LogicNLI에서 수행되었습니다. 각 데이터셋은 서로 다른 난이도를 가지고 있으며, ProntoQA는 기본적인 추론을 요구하는 반면, ProofWriter는 더 복잡한 구조를 포함하고, LogicNLI는 가장 도전적인 논리적 관계를 다룹니다.

#### 경쟁 모델
1. **Naive Prompting**: 기본적인 프롬프트 기법으로, 모델이 질문에 대한 직접적인 답변을 생성합니다.
2. **Chain-of-Thought (CoT)**: 모델이 중간 추론 단계를 생성하도록 유도하여 더 나은 성능을 발휘합니다.
3. **Chain-of-Thought with Self-Consistency (CoT-SC)**: 여러 번의 추론을 통해 결과를 집계하여 최종 답변을 도출합니다.
4. **Cumulative Reasoning (CR)**: 여러 단계에 걸쳐 추론을 발전시킵니다.
5. **Tree-of-Thought (ToT)**: 트리 구조를 사용하여 여러 경로를 탐색합니다.
6. **SymbCoT**: 기호 표현을 통합하여 추론을 수행합니다.
7. **Logic-LM**: 자연어 입력을 기호 형식으로 변환하여 외부 논리 엔진을 사용합니다.

#### 테스트 데이터
- **ProntoQA**: 500개의 질문으로 구성되어 있으며, 기본적인 논리적 관계를 평가합니다.
- **ProofWriter**: 600개의 질문으로, 다양한 깊이의 논리적 추론을 요구합니다.
- **LogicNLI**: 300개의 질문으로, 복잡한 논리적 관계를 다룹니다.

#### 메트릭
성능 평가는 정확도(Accuracy)로 측정되었습니다. 각 모델의 성능은 다음과 같은 방식으로 비교되었습니다:
- **ProntoQA**: CoT-SC, ToT, CR, SymbCoT와 비교하여 평균 11.6%의 성능 향상을 보였습니다.
- **ProofWriter**: GPT-4와 GPT-4o에서 각각 4.3% 및 6.2%의 성능 향상을 기록했습니다.
- **LogicNLI**: 가장 도전적인 데이터셋에서 6.3% 및 6.4%의 성능 향상을 보였습니다.

#### 결과
Aristotle는 모든 데이터셋에서 기존의 최첨단 모델들보다 일관되게 높은 성능을 보였으며, 특히 복잡한 논리적 구조를 가진 문제에서 더욱 두드러진 성과를 나타냈습니다. 이 연구는 LLM 기반의 추론 프레임워크에서 기호 논리 표현을 완전히 통합한 첫 번째 사례로, LLM이 기호 구조에 대한 완전한 논리적 추론을 수행할 수 있음을 입증했습니다.

---




In this paper, a new logical reasoning framework called "Aristotle" is proposed, and its performance is evaluated against several competing models. The experiments were conducted on three main datasets: ProntoQA, ProofWriter, and LogicNLI. Each dataset has different levels of difficulty, with ProntoQA requiring basic reasoning, ProofWriter incorporating more complex structures, and LogicNLI addressing the most challenging logical relationships.

#### Competing Models
1. **Naive Prompting**: A basic prompting technique where the model generates a direct answer to the question.
2. **Chain-of-Thought (CoT)**: Encourages the model to generate intermediate reasoning steps, leading to better performance.
3. **Chain-of-Thought with Self-Consistency (CoT-SC)**: Runs the reasoning process multiple times and aggregates the results to determine the final answer.
4. **Cumulative Reasoning (CR)**: Builds reasoning over successive iterations.
5. **Tree-of-Thought (ToT)**: Uses a tree structure to explore multiple reasoning pathways.
6. **SymbCoT**: Integrates symbolic expressions into the reasoning process.
7. **Logic-LM**: Translates natural language input into a symbolic format and uses an external logic engine.

#### Test Data
- **ProntoQA**: Composed of 500 questions, assessing basic logical relationships.
- **ProofWriter**: Contains 600 questions requiring various depths of logical reasoning.
- **LogicNLI**: Comprises 300 questions dealing with complex logical relationships.

#### Metrics
Performance is measured in terms of accuracy. The performance of each model was compared as follows:
- **ProntoQA**: Showed an average improvement of 11.6% over CoT-SC, ToT, CR, and SymbCoT.
- **ProofWriter**: Achieved improvements of 4.3% and 6.2% with GPT-4 and GPT-4o, respectively.
- **LogicNLI**: Demonstrated improvements of 6.3% and 6.4% on the most challenging dataset.

#### Results
Aristotle consistently outperformed all state-of-the-art models across all datasets, particularly excelling in complex logical structures. This work marks the first successful complete integration of symbolic logic expressions into an LLM-based reasoning framework, demonstrating that LLMs can perform complete logical reasoning over symbolic structures.


<br/>
# 예제



이 논문에서는 "Aristotle"이라는 논리적 추론 프레임워크를 제안하고, 이를 통해 다양한 논리적 추론 작업을 수행하는 방법을 설명합니다. 이 프레임워크는 세 가지 주요 구성 요소로 이루어져 있습니다: 논리 분해기(Logical Decomposer), 논리 검색 라우터(Logical Search Router), 그리고 논리 해결기(Logical Resolver)입니다. 이 구성 요소들은 각각의 역할을 수행하여 복잡한 논리적 문제를 해결하는 데 기여합니다.

#### 예시: ProntoQA 데이터셋

1. **입력 데이터 (Premises P)**:
   - "각 점프스는 과일이다."
   - "모든 점프스는 웜퍼스이다."
   - "모든 웜퍼스는 투명하지 않다."
   - "웜퍼스는 텀퍼스이다."
   - "텀퍼스는 mean이다."
   - "텀퍼스는 vumpus이다."
   - "모든 vumpus는 차갑다."
   - "각 vumpus는 yumpus이다."
   - "yumpus는 오렌지색이다."
   - "yumpus는 numpus이다."
   - "numpus는 둔하다."
   - "각 numpus는 dumpus이다."
   - "모든 dumpus는 수줍지 않다."
   - "impuses는 수줍다."
   - "dumpus는 rompuses이다."
   - "각 rompus는 액체이다."
   - "rompuses는 zumpuses이다."
   - "Alex는 텀퍼스이다."

2. **질문 (Statement S)**:
   - "Alex는 수줍지 않다. 이 문장이 참인지, 거짓인지, 알 수 없는지 판단하시오."

3. **출력 데이터 (Query St)**:
   - "Shy(Alex, False)" ::: Alex는 수줍지 않다.

#### 처리 과정

1. **번역 (Translation)**:
   - 주어진 전제와 질문을 논리적 형식으로 변환합니다. 각 전제는 사실과 규칙으로 나뉘어 표현됩니다.

2. **분해 (Decomposition)**:
   - 변환된 전제와 질문을 정규화(Normalization) 및 스코렘화(Skolemization)하여 간단한 논리 형태로 분해합니다.

3. **해결 (Resolution)**:
   - 서로 모순되는 용어를 확인하고, 발견된 모순을 해결하기 위해 해결 규칙을 적용합니다. 만약 모순이 발견되면 "Contradiction"을 출력하고, 그렇지 않으면 새로운 절을 출력합니다.

이러한 과정을 통해 "Aristotle" 프레임워크는 주어진 전제에 기반하여 질문에 대한 답을 도출합니다.

---




In this paper, a logical reasoning framework called "Aristotle" is proposed, which describes how to perform various logical reasoning tasks. This framework consists of three main components: Logical Decomposer, Logical Search Router, and Logical Resolver. Each of these components plays a specific role in solving complex logical problems.

#### Example: ProntoQA Dataset

1. **Input Data (Premises P)**:
   - "Each jompus is fruity."
   - "Every jompus is a wumpus."
   - "Every wumpus is not transparent."
   - "Wumpuses are tumpuses."
   - "Tumpuses are mean."
   - "Tumpuses are vumpuses."
   - "Every vumpus is cold."
   - "Each vumpus is a yumpus."
   - "Yumpuses are orange."
   - "Yumpuses are numpuses."
   - "Numpuses are dull."
   - "Each numpus is a dumpus."
   - "Every dumpus is not shy."
   - "Impuses are shy."
   - "Dumpuses are rompuses."
   - "Each rompus is liquid."
   - "Rompuses are zumpuses."
   - "Alex is a tumpus."

2. **Question (Statement S)**:
   - "Is the following statement true, false, or unknown? Alex is not shy."

3. **Output Data (Query St)**:
   - "Shy(Alex, False)" ::: Alex is not shy.

#### Processing Steps

1. **Translation**:
   - The given premises and question are translated into a logical format. Each premise is expressed in terms of facts and rules.

2. **Decomposition**:
   - The translated premises and question are decomposed into a standardized logical form through normalization and skolemization.

3. **Resolution**:
   - The system checks for complementary or contradictory terms. If contradictions are found, the resolution rule is applied. If a contradiction is found, it outputs "Contradiction"; otherwise, it outputs the new clause.

Through this process, the "Aristotle" framework derives answers to the questions based on the given premises.

<br/>
# 요약


이 논문에서는 논리적 추론을 위한 새로운 프레임워크인 Aristotle을 제안하며, 이는 세 가지 주요 구성 요소(논리 분해기, 검색 라우터, 논리 해결기)를 통해 복잡한 논리적 문제를 효과적으로 해결한다. 실험 결과, Aristotle은 기존의 최첨단 방법들보다 정확성과 효율성에서 일관되게 우수한 성능을 보였다. 예를 들어, ProofWriter와 LogicNLI 데이터셋에서 평균적으로 4.5%의 성능 향상을 달성하였다.



This paper proposes a new framework for logical reasoning called Aristotle, which effectively addresses complex logical problems through three key components: a Logical Decomposer, a Search Router, and a Logical Resolver. Experimental results show that Aristotle consistently outperforms existing state-of-the-art methods in both accuracy and efficiency. For instance, it achieved an average performance improvement of 4.5% on the ProofWriter and LogicNLI datasets.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: 이 피규어는 기존의 추론 프레임워크와 Aristotle의 성능을 비교합니다. 특히, Search Error (SE)와 Reasoning Error (RE)에서 Aristotle이 기존 방법들보다 우수한 성능을 보임을 나타냅니다. 이는 Aristotle이 더 효율적이고 정확한 추론을 가능하게 한다는 것을 시사합니다.
   - **Figure 2**: Aristotle의 전체적인 구조를 보여주는 다이어그램으로, 각 모듈(Translator, Decomposer, Search Router, Resolver)의 역할을 명확히 설명합니다. 이 구조는 각 단계에서 논리적 규칙과 기호 표현을 통합하여 추론 과정을 최적화하는 방법을 강조합니다.

2. **테이블**
   - **Table 1**: 다양한 데이터셋에서 Aristotle의 성능을 다른 최신 방법들과 비교한 결과를 보여줍니다. 특히, Aristotle이 모든 데이터셋에서 평균적으로 4.5%에서 11.6%의 성능 향상을 보였음을 나타냅니다. 이는 복잡한 논리적 구조를 처리하는 데 있어 Aristotle의 우수성을 강조합니다.
   - **Table 2**: Claude 및 LLaMA와 같은 다른 모델에서의 성능을 비교하여, Aristotle의 일반화 가능성을 보여줍니다. 이는 다양한 모델에서도 일관된 성능 향상을 달성할 수 있음을 시사합니다.

3. **어펜딕스**
   - **Appendix D**: 데이터셋의 세부 사항을 설명하며, 각 데이터셋의 난이도와 특성을 명확히 합니다. ProntoQA, ProofWriter, LogicNLI의 난이도 차이를 통해 Aristotle의 성능이 복잡한 문제에서 더욱 두드러진다는 점을 강조합니다.
   - **Appendix F**: 오류 분석을 통해 Aristotle의 한계와 개선 가능성을 제시합니다. 특히, 데이터셋의 구성 문제로 인한 오류와 LLM의 번역 및 해석 과정에서 발생하는 오류를 분석하여, 향후 연구 방향을 제시합니다.

### Insights
- **효율성과 정확성**: Aristotle은 기존의 방법들보다 더 높은 정확성과 효율성을 보여주며, 이는 복잡한 논리적 문제를 해결하는 데 있어 중요한 성과입니다.
- **모듈화된 접근**: 각 모듈의 역할이 명확히 정의되어 있어, 문제 해결 과정에서의 오류를 줄이고, 더 나은 성능을 발휘할 수 있도록 합니다.
- **일반화 가능성**: 다양한 모델에서의 성능 향상은 Aristotle의 접근 방식이 특정 모델에 국한되지 않고, 널리 적용 가능하다는 것을 보여줍니다.

---




1. **Diagrams and Figures**
   - **Figure 1**: This figure compares the performance of existing reasoning frameworks with Aristotle. It particularly highlights that Aristotle outperforms other methods in terms of Search Error (SE) and Reasoning Error (RE), suggesting that Aristotle enables more efficient and accurate reasoning.
   - **Figure 2**: A diagram illustrating the overall structure of Aristotle, clearly explaining the roles of each module (Translator, Decomposer, Search Router, Resolver). This structure emphasizes how integrating logical rules and symbolic representations at each stage optimizes the reasoning process.

2. **Tables**
   - **Table 1**: Shows the performance of Aristotle compared to other state-of-the-art methods across various datasets. It indicates that Aristotle achieves an average performance improvement of 4.5% to 11.6% across all datasets, highlighting its superiority in handling complex logical structures.
   - **Table 2**: Compares performance across different models, such as Claude and LLaMA, demonstrating the generalizability of Aristotle. This suggests that consistent performance improvements can be achieved across various models.

3. **Appendices**
   - **Appendix D**: Details the specifications of the datasets, clarifying the differences in difficulty and characteristics of each dataset. The varying levels of difficulty in ProntoQA, ProofWriter, and LogicNLI emphasize Aristotle's performance in more complex problems.
   - **Appendix F**: Provides an error analysis that outlines the limitations and potential improvements for Aristotle. It analyzes errors arising from dataset construction issues and the translation and interpretation processes of LLMs, suggesting future research directions.

### Insights
- **Efficiency and Accuracy**: Aristotle demonstrates higher accuracy and efficiency compared to existing methods, marking a significant achievement in solving complex logical problems.
- **Modular Approach**: The clear definition of each module's role reduces errors in the problem-solving process and allows for better performance.
- **Generalizability**: Performance improvements across various models indicate that Aristotle's approach is not limited to specific models but can be widely applied.

<br/>
# refer format:



### BibTeX 형식
```bibtex
@inproceedings{xu2025aristotle,
  author    = {Jundong Xu and Hao Fei and Meng Luo and Qian Liu and Liangming Pan and William Yang Wang and Preslav Nakov and Mong-Li Lee and Wynne Hsu},
  title     = {Aristotle: Mastering Logical Reasoning with A Logic-Complete Decompose-Search-Resolve Framework},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {3052--3075},
  year      = {2025},
  month     = {July},
  publisher = {Association for Computational Linguistics},
  address   = {Bangkok, Thailand},
}
```

### 시카고 스타일 인용
Xu, Jundong, Hao Fei, Meng Luo, Qian Liu, Liangming Pan, William Yang Wang, Preslav Nakov, Mong-Li Lee, and Wynne Hsu. "Aristotle: Mastering Logical Reasoning with A Logic-Complete Decompose-Search-Resolve Framework." In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 3052–3075. Bangkok, Thailand: Association for Computational Linguistics, 2025.
