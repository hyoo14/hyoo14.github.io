---
layout: post
title:  "[2025]LLM-SRBench: A New Benchmark for Scientific Equation Discovery with Large Language Models"  
date:   2025-07-25 13:59:40 +0900
categories: study
---

{% highlight ruby %}


한줄 요약: 


추론으로 방정식 도출하게하는 밴치마크 데이터셋 제안(기존은 암기위주)  


짧은 요약(Abstract) :    



이 논문은 과학 방정식 발견(task of scientific equation discovery)을 위한 새로운 벤치마크 LLM-SRBench를 제안합니다. 기존 벤치마크들은 유명한 방정식을 사용해 대형 언어 모델(LLMs)이 단순히 암기한 내용을 반복하는 것이지 진정한 ‘발견’을 하는지 판단하기 어렵습니다. 이를 해결하기 위해, 저자들은 총 239개의 문제로 구성된 두 가지 유형의 과제를 포함하는 새로운 벤치마크를 설계했습니다:

LSR-Transform: 잘 알려진 물리 모델을 변형하여 드문 수학적 표현으로 바꾸고, LLM이 암기가 아닌 추론을 통해 문제를 해결하는지 평가합니다.

LSR-Synth: 인공적으로 생성된, 참신하고 발견 지향적인 문제로 구성되어 있으며, 데이터를 바탕으로 새로운 수식을 찾아야 합니다.

실험 결과, 가장 잘 수행된 시스템조차도 상징적 정확도(symbolic accuracy)가 **31.5%**에 그쳤습니다. 이는 이 과제가 매우 어렵고, 제안된 벤치마크가 LLM 기반 방정식 발견의 한계를 평가하고 향후 연구를 이끄는 데 유용함을 보여줍니다.


Abstract
Scientific equation discovery has long been a cornerstone of scientific progress, enabling the derivation of laws governing natural phenomena. Recently, Large Language Models (LLMs) have gained interest for this task due to their potential to leverage embedded scientific knowledge for hypothesis generation. However, it is difficult to assess the true discovery capabilities of these methods because existing benchmarks often use well-known equations. This makes them vulnerable to memorization by LLMs and results in inflated performance metrics that do not reflect genuine discovery. In this paper, we introduce LLM-SRBench, a comprehensive benchmark with 239 challenging problems across four scientific domains specifically designed to evaluate LLM-based scientific equation discovery methods while preventing trivial memorization. Our benchmark comprises two main categories:

LSR-Transform, which transforms common physical models into less common mathematical representations to test reasoning beyond memorized forms, and

LSR-Synth, which introduces synthetic, discovery-driven problems requiring data-driven reasoning.
Through extensive evaluation of several state-of-the-art methods, using both open and closed LLMs, we find that the best-performing system so far achieves only 31.5% symbolic accuracy. These findings highlight the challenges of scientific equation discovery, positioning LLM-SRBench as a valuable resource for future research.




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



이 논문은 **LLM(대형 언어 모델)**을 활용한 과학 방정식 발견(equation discovery)의 능력을 평가하기 위해 특별히 설계된 벤치마크 LLM-SRBench를 제안합니다. LLM-SRBench는 두 가지 주요 문제 유형으로 구성됩니다:

LSR-Transform
기존의 유명 물리 방정식(Feynman equations)을 변형하여 잘 알려지지 않은 수학적 표현으로 바꾸고, 입력-출력 변수의 역할을 바꾸어 새로운 문제를 생성합니다. 이 과정은 파이썬의 SymPy 라이브러리를 사용해 수식을 기호적으로 재정의하고, 새로운 문제 설명은 GPT-4o를 활용해 자연어로 생성합니다. 이 데이터는 모델이 암기한 공식이 아닌, 낯선 수식 형태에 대해 추론할 수 있는지 확인하는 데 목적이 있습니다.

LSR-Synth
과학 분야별 대표 문제(예: 화학 반응 속도, 생물학적 개체 수 증가 등)에 대해 기존에 알려진 수학적 항과 LLM이 생성한 참신한 synthetic term을 조합하여 새로운 방정식을 만듭니다. 이후 **수치 해석 기법(numerical solver)**을 이용해 문제의 해 존재성을 검증하고, GPT-4o를 이용한 novelty 평가와 전문가 검증을 통해 문제의 난이도와 신뢰성을 확보합니다.

이 벤치마크는 특별한 아키텍처보다는 다양한 LLM 기반 방법들의 비교 실험으로 구성되어 있습니다. 사용된 대표적인 모델/기법들은 다음과 같습니다:

Direct Prompting (DataBlind): 데이터를 사용하지 않고 문제 설명만으로 수식을 생성하는 베이스라인.

LLM-SR (Shojaee et al., 2024b): LLM이 생성한 수식 skeleton을 파이썬 함수 형태로 표현하고, 진화 알고리즘 기반 피드백 루프를 통해 수정.

LaSR (Grayeli et al., 2024): 수식 개념을 추출하고 이를 다시 LLM + PySR 기반 진화 탐색으로 확장.

SGA (Ma et al., 2024): LLM으로 discrete 수식 구조를 생성하고 PyTorch 시뮬레이션으로 연속적 파라미터를 최적화하는 bilevel optimization 구조.

실험에는 Llama-3.1-8B-Instruct, GPT-3.5-turbo, GPT-4o-mini 등의 백본이 사용되며, 각 모델은 동일한 조건(문제당 LLM 호출 1000회 이내)에서 평가됩니다.




This paper introduces LLM-SRBench, a new benchmark designed to evaluate LLM-based scientific equation discovery. The benchmark consists of two major problem types:

LSR-Transform
This component takes existing equations from the Feynman dataset and transforms them into less familiar mathematical forms by switching input-output variables. Symbolic transformations are performed using the SymPy Python library, and new problem descriptions are generated using GPT-4o. This tests whether LLMs can reason beyond memorized equation formats.

LSR-Synth
This component generates novel synthetic equations by combining known scientific terms (e.g., from reaction kinetics or population dynamics) with novel symbolic terms generated by an LLM. The solvability of each equation is verified using numerical solvers, and the novelty of the expression is validated using GPT-4o and domain experts.

The benchmark does not introduce a single new model architecture, but rather evaluates various LLM-based equation discovery methods, including:

Direct Prompting (DataBlind): Generates equations purely from the prompt without using any data.

LLM-SR (Shojaee et al., 2024b): Generates equation skeletons as Python functions and refines them through a multi-island evolutionary loop guided by LLM feedback.

LaSR (Grayeli et al., 2024): Abstracts symbolic relations into concepts and uses a hybrid evolutionary + LLM-guided approach to evolve new hypotheses.

SGA (Ma et al., 2024): Implements a bilevel optimization strategy, where LLMs propose symbolic expressions, and PyTorch simulations optimize the parameters.

Experiments are conducted with Llama-3.1-8B-Instruct, GPT-3.5-turbo, and GPT-4o-mini, with all methods limited to 1000 LLM calls per problem to ensure fair comparison.




   
 
<br/>
# Results  



논문은 LLM 기반 방정식 발견 방법들을 **세 가지 LLM 백본(GPT-4o-mini, GPT-3.5-turbo, Llama-3.1-8B)**을 활용하여 비교합니다. 평가 대상은 네 가지 방법입니다:

Direct Prompting (DataBlind): 데이터 없이 문제 설명만으로 방정식을 생성

SGA (Ma et al., 2024): LLM + 시뮬레이션 기반 bilevel 최적화

LaSR (Grayeli et al., 2024): 수식 개념 학습 후 진화 탐색

LLM-SR (Shojaee et al., 2024b): 파이썬 함수 기반 skeleton + 진화 전략

모든 방법은 LLM-SRBench의 두 가지 하위 데이터셋에서 평가됩니다:

LSR-Transform (총 111개 문제): 잘 알려진 물리 방정식을 변형한 버전

LSR-Synth (총 128개 문제): 합성된 참신한 수식 포함, 4개 과학 도메인(화학, 생물, 물리, 소재과학)



사용된 평가 척도는 다음과 같습니다:

Symbolic Accuracy (SA): 예측 수식과 정답 수식의 기호적 유사성 평가 (GPT-4o를 이용한 평가자)

Acc₀.₁: 예측값이 정답과 10% 이내 오차 범위에 있을 확률

NMSE (Normalized Mean Squared Error): 정규화된 평균제곱오차



핵심 결과 요약:

전체적으로 LLM-SR + GPT-4o-mini 조합이 **상징 정확도 31.5% (LSR-Transform 기준)**으로 최고 성능을 보임

LaSR + GPT-4o-mini는 수치 정밀도(Acc₀.₁, NMSE) 면에서 가장 뛰어남

Direct Prompting 방식은 거의 모든 지표에서 가장 낮은 성능을 보이며, 데이터 없는 추론의 한계를 보여줌

LSR-Synth 데이터셋은 기존 문제를 변형한 LSR-Transform보다 훨씬 어려워, 전반적으로 낮은 성능 기록됨

과학 분야별로도 성능 편차 존재: 화학/생물 분야에서 OOD 일반화 성능이 더 낮음



추가 분석에서는 기호 정확도와 OOD 일반화 성능(Acc₀.₁, NMSE) 간에 높은 상관관계를 확인하여, 기호적 정답성이 수치적 일반화력의 지표가 될 수 있음을 시사함.





Direct Prompting (DataBlind) – Generates equations without access to data

SGA (Ma et al., 2024) – A bilevel optimization strategy using LLMs and simulations

LaSR (Grayeli et al., 2024) – Learns equation concepts and performs hybrid evolution

LLM-SR (Shojaee et al., 2024b) – Generates Python skeleton functions refined via evolutionary search

These methods are evaluated on the two parts of LLM-SRBench:

LSR-Transform (111 tasks): Transformed variants of known physics equations

LSR-Synth (128 tasks): Synthetic, novel equation discovery tasks spanning chemistry, biology, physics, and materials science



Metrics used:

Symbolic Accuracy (SA): Measures symbolic similarity to the ground truth using GPT-4o as an evaluator

Acc₀.₁: Proportion of predictions within 10% relative error

NMSE (Normalized Mean Squared Error): Measures numerical precision



Key findings:

LLM-SR with GPT-4o-mini achieves the highest symbolic accuracy (31.5%) on LSR-Transform

LaSR with GPT-4o-mini consistently performs best in numeric precision (highest Acc₀.₁, lowest NMSE)

Direct Prompting shows poor performance across all metrics, highlighting the limits of memorization without data

LSR-Synth tasks are significantly harder than LSR-Transform, with lower scores across the board

There are domain-specific differences, with chemistry and biology exhibiting worse OOD generalization than physics or materials science



Correlation analysis shows strong alignment between symbolic accuracy and numeric generalization (Acc₀.₁ and NMSE), suggesting that symbolic correctness is a reliable indicator of generalization ability.




<br/>
# 예제  



논문에서 제시된 **과학 방정식 발견(task)**은 다음과 같은 형태입니다:

목표: 주어진 입력 변수들과 과학적 맥락, 수치 데이터로부터 출력 변수와의 관계를 나타내는 수학적 방정식을 발견하라.



예시 1: 고전역학 (LSR-Transform에서 발췌)
과학적 맥락: 에너지를 저장하는 진동 시스템의 질량(m)을 추정

입력 변수들:

평균 저장 에너지 
𝐸
𝑛
E 
n
​
 

구동 주파수 
𝜔
ω

고유 주파수 
𝜔
0
ω 
0
​
 

진폭 
𝑥
x

출력 변수: 질량 
𝑚
m

정답 방정식 예시:

𝑚
=
4
𝐸
𝑛
𝑥
2
(
𝜔
2
+
𝜔
0
2
)
m= 
x 
2
 (ω 
2
 +ω 
0
2
​
 )
4E 
n
​
 
​
 
입력 데이터 예시:

𝐸
𝑛
E 
n
​
 	
𝜔
ω	
𝜔
0
ω 
0
​
 	
𝑥
x	
𝑚
m (출력)
4.7	1.2	2.3	1.5	1.2
3.4	2.7	2.7	3.1	0.1
2.8	1.5	3.6	1.4	0.4



예시 2: 화학 반응 속도 (LSR-Synth에서 발췌)
과학적 맥락: 농도 
𝐴
(
𝑡
)
A(t)에 따른 반응 속도 
𝑑
𝐴
𝑑
𝑡
dt
dA
​
  예측

입력 변수들: 시간 
𝑡
t, 농도 
𝐴
(
𝑡
)
A(t)

출력 변수: 
𝑑
𝐴
𝑑
𝑡
dt
dA
​
 

정답 방정식 예시 (합성된 식):

𝑑
𝐴
𝑑
𝑡
=
−
𝑘
𝐴
(
𝑡
)
2
−
𝑘
𝐴
(
𝑡
)
exp
⁡
(
−
𝑘
𝑠
𝑡
)
+
𝑘
𝑝
sin
⁡
(
𝜔
𝐴
(
𝑡
)
𝑡
)
dt
dA
​
 =−kA(t) 
2
 −kA(t)exp(−kst)+kpsin(ωA(t)t)
이 방정식은 일반적인 항(예: 
−
𝑘
𝐴
2
−kA 
2
 )과 참신한 항(예: 
sin
⁡
(
𝜔
𝐴
𝑡
)
sin(ωAt))을 조합하여, 모델이 단순 암기 대신 데이터 기반 추론을 요구하도록 설계되어 있습니다.

🔬 English Version (Examples: Input/Output Structure, Task Description)
The core task in the benchmark is:

Goal: Given input features, scientific context, and a dataset of numerical observations, discover a symbolic equation describing the relationship to the target variable.



Example 1: Classical Mechanics (from LSR-Transform)
Scientific Task: Predict the mass 
𝑚
m of an oscillating system storing energy

Input Variables:

Mean stored energy 
𝐸
𝑛
E 
n
​
 

Driving frequency 
𝜔
ω

Natural frequency 
𝜔
0
ω 
0
​
 

Amplitude 
𝑥
x

Target Variable: Mass 
𝑚
m

Ground-Truth Equation:

𝑚
=
4
𝐸
𝑛
𝑥
2
(
𝜔
2
+
𝜔
0
2
)
m= 
x 
2
 (ω 
2
 +ω 
0
2
​
 )
4E 
n
​
 
​
 
Sample Input Data:

𝐸
𝑛
E 
n
​
 	
𝜔
ω	
𝜔
0
ω 
0
​
 	
𝑥
x	
𝑚
m (output)
4.7	1.2	2.3	1.5	1.2
3.4	2.7	2.7	3.1	0.1
2.8	1.5	3.6	1.4	0.4



Example 2: Chemical Reaction Kinetics (from LSR-Synth)
Scientific Task: Predict the rate of change 
𝑑
𝐴
𝑑
𝑡
dt
dA
​
  for chemical concentration 
𝐴
(
𝑡
)
A(t)

Input Variables: Time 
𝑡
t, concentration 
𝐴
(
𝑡
)
A(t)

Target Variable: 
𝑑
𝐴
𝑑
𝑡
dt
dA
​
 

Synthetic Ground-Truth Equation:

𝑑
𝐴
𝑑
𝑡
=
−
𝑘
𝐴
(
𝑡
)
2
−
𝑘
𝐴
(
𝑡
)
exp
⁡
(
−
𝑘
𝑠
𝑡
)
+
𝑘
𝑝
sin
⁡
(
𝜔
𝐴
(
𝑡
)
𝑡
)
dt
dA
​
 =−kA(t) 
2
 −kA(t)exp(−kst)+kpsin(ωA(t)t)
This equation combines known terms (e.g., 
−
𝑘
𝐴
2
−kA 
2
 ) and novel terms (e.g., 
sin
⁡
(
𝜔
𝐴
𝑡
)
sin(ωAt)) to encourage data-driven discovery rather than memorization.





<br/>  
# 요약   


이 논문은 과학 방정식 발견을 위해 기존 물리 공식을 변형하거나 합성 수식을 포함한 새로운 문제를 생성한 LLM-SRBench를 제안한다. 실험 결과, 가장 성능이 좋은 LLM-SR(GPT-4o-mini 기반) 모델도 상징 정확도 31.5%에 그쳐 이 과제가 여전히 어렵다는 점을 보여준다




This paper proposes LLM-SRBench, a benchmark for scientific equation discovery that includes both transformed versions of known physics equations and novel synthetic problems. Experiments show that even the best-performing model, LLM-SR with GPT-4o-mini, achieves only 31.5% symbolic accuracy, highlighting the difficulty of the task.



<br/>  
# 기타  





Figure 1: Error 분석 (Feynman vs. LSR-Transform/Synth)
LLM이 기존 Feynman 문제에서는 수치 오차와 기호 오차가 매우 낮아, **단순 암기(recitation)**에 의한 성능임을 시사함.

반면 LSR-SRBench에서는 오차가 크고 감소 속도도 느려, 추론 기반 탐색이 필요함을 보여줌.



Figure 2 & 3: 데이터 생성 파이프라인
Figure 2는 전체 discovery 프로세스를 단계별로 보여주며, 과학 지식 기반 추론과 데이터 기반 탐색의 결합 구조를 시각화함.

Figure 3은 LSR-Transform과 LSR-Synth가 어떻게 생성되는지를 상세히 설명하며, 각각의 문제 생성, 변환, 검증 과정이 구조적으로 정리되어 있음.



Table 1: 성능 비교표
각 모델-LLM 조합에 대한 Symbolic Accuracy, Acc₀.₁, NMSE 지표가 정리되어 있으며,
GPT-4o-mini 기반 LLM-SR이 기호 정확도에서 가장 우수하고, LaSR이 수치 오차 최소화에 강점을 보임.

**도메인별 성능 편차(예: 소재 과학에서의 높은 정확도)**도 확인 가능함.



Figure 4: 문제 복잡도별 성능 (Feynman vs. LSR-Transform)
동일한 복잡도에서도 LSR-Transform 문제에서 성능이 더 낮게 나타나며, 이는 LLM이 **낯선 표현(unfamiliar forms)**에서 어려움을 겪는다는 것을 시사함.



Figure 5: ID vs. OOD 일반화 성능
모든 모델에서 OOD 성능이 ID보다 낮으며, 일반화에 어려움이 존재함.

특히 화학/생물 분야에서는 OOD 오차가 더 크게 증가해 도메인 편향 문제를 시사함.



Figure 6: 기호 정확도와 OOD 정밀도의 상관관계
Symbolic Accuracy가 높을수록 Acc₀.₁은 높고 NMSE는 낮아지며,
이는 기호 정답성이 일반화 가능성의 지표로 활용될 수 있음을 보여줌.



Appendix A: 방정식 복잡도 분포와 생성 예시
Figure 8, 9, 10 등을 통해 LSR-SRBench의 수식 복잡도는 Feynman보다 높지만, 기하학적 표현의 균형을 고려하여 설계됨.

생성된 수식 예시(예: 
𝑑
𝐴
𝑑
𝑡
=
−
𝑘
𝐴
2
+
sin
⁡
(
𝐴
)
dt
dA
​
 =−kA 
2
 +sin( 
A
​
 ))는 모델이 단순 공식 기억에 의존하지 않고 정교한 추론을 해야 함을 보여줌.



Figure 1: Error Analysis (Feynman vs. LSR)
Shows much lower symbolic and numeric error for Feynman problems, indicating LLMs are memorizing these solutions.

In contrast, higher and more gradual error on LSR problems suggests the need for genuine reasoning and exploration.



Figure 2 & 3: Data Generation Pipelines
Figure 2 visualizes the full discovery workflow, emphasizing the synergy between scientific priors and data-driven refinement.

Figure 3 details how LSR-Transform and LSR-Synth datasets are constructed and validated, highlighting multi-step filtering, transformation, and novelty checks.



Table 1: Performance Comparison
Summarizes symbolic accuracy (SA), Acc₀.₁, and NMSE across models and domains.

LLM-SR + GPT-4o-mini achieves the best SA, while LaSR excels in numeric precision, with notable performance variation across domains.



Figure 4: Performance vs. Equation Complexity
Even at similar complexity levels, LSR-Transform is harder than Feynman, implying that LLMs struggle with unfamiliar forms, not just long equations.



Figure 5: In-Domain (ID) vs. Out-of-Domain (OOD)
Performance consistently drops on OOD data, revealing generalization challenges.

Chemistry and biology tasks show the greatest OOD degradation, suggesting domain-dependent vulnerabilities.



Figure 6: Symbolic Accuracy vs. OOD Generalization
Symbolic accuracy positively correlates with Acc₀.₁ and negatively with NMSE, indicating that better symbolic discovery leads to stronger generalization.



Appendix A: Expression Complexity and Examples
Figures 8–10 compare complexity distributions and provide generation examples.

They demonstrate that LSR-SRBench contains non-trivial, interpretable, and solvable equations, pushing models beyond rote memorization.




<br/>
# refer format:     



@inproceedings{shojaee2025llmsrbench,
  title     = {{LLM-SRBench: A New Benchmark for Scientific Equation Discovery with Large Language Models}},
  author    = {Parshin Shojaee and Ngoc-Hieu Nguyen and Kazem Meidani and Amir Barati Farimani and Khoa D Doan and Chandan K Reddy},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025},
  volume    = {267},
  publisher = {PMLR},
  address   = {Vancouver, Canada},
  url       = {https://github.com/deep-symbolic-mathematics/llm-srbench}
}





Shojaee, Parshin, Ngoc-Hieu Nguyen, Kazem Meidani, Amir Barati Farimani, Khoa D. Doan, and Chandan K. Reddy. "LLM-SRBench: A New Benchmark for Scientific Equation Discovery with Large Language Models." In Proceedings of the 42nd International Conference on Machine Learning (ICML), vol. 267. Vancouver, Canada: PMLR, 2025. https://github.com/deep-symbolic-mathematics/llm-srbench.






