---
layout: post
title:  "[2025]Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions"  
date:   2025-07-27 01:33:40 +0200
categories: study
---

{% highlight ruby %}


한줄 요약: 

디퓨전 마스킹 모델(오토리그레시브한)인데 추론 때 더 쉬운 디코딩 전략 써서 극적 성능 향상  



짧은 요약(Abstract) :    


이 논문은 Masked Diffusion Models (MDMs) 의 학습 복잡성과 추론 유연성 간의 균형을 분석합니다.
MDM은 Autoregressive Models (ARMs) 에 비해 학습 단계에서는 훨씬 더 많은 마스킹(infilling) 문제를 해결해야 하므로 계산적으로 어렵지만, 추론 단계에서는 토큰을 임의 순서로 디코딩할 수 있는 유연성을 가집니다.
저자들은 MDM이 실제로 학습 시 계산적으로 어려운 하위 문제를 포함함을 이론적, 실험적으로 보여주고, 반대로 추론 단계에서 디코딩 순서를 적절히 조절하면 성능이 크게 향상될 수 있음을 입증합니다.
예를 들어 Sudoku 퍼즐에서, 단순한 추론 전략을 적용한 것만으로 정확도가 **7% 미만 → 약 90%**까지 향상되었으며, 이는 교사 강제 학습(teacher forcing)된 ARM보다도 더 나은 성능임을 보여줍니다.



This paper analyzes the trade-off between training complexity and inference flexibility in Masked Diffusion Models (MDMs).
While MDMs must solve exponentially many infilling tasks during training — making training computationally harder than for Autoregressive Models (ARMs) — they offer much more flexibility during inference by allowing token decoding in arbitrary order.
The authors theoretically and empirically demonstrate that MDMs indeed face computationally intractable subproblems during training.
However, they also show that by adaptively selecting the decoding order during inference, MDMs can significantly improve performance, bypassing difficult problems encountered during training.
On tasks like Sudoku, this adaptive inference boosts performance from less than 7% to about 90%, even surpassing ARMs trained via teacher forcing with significantly more parameters.




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



이 논문에서는 Masked Diffusion Models (MDMs) 의 학습 및 추론 과정을 이론적, 실험적으로 분석합니다. 주요 메서드는 다음과 같습니다:

모델 구조 및 학습 방식:

MDM은 주어진 문장에 무작위로 노이즈를 추가(마스킹)하고, 이 마스킹된 상태에서 원래의 토큰을 예측하는 방식으로 학습됩니다.

MDM은 모든 가능한 마스크 조합에 대해 토큰을 복원하도록 훈련되기 때문에, 총 2^L개의 infilling 문제를 풀어야 합니다. 이를 “순서 무관 학습(order-agnostic training)”이라고 합니다.

이에 반해, 전통적인 Autoregressive Model (ARM)은 순차적으로 한 토큰씩 예측하며, L개의 순차적 예측만 수행합니다.

학습 난이도 이론 증명:

저자들은 일부 조건에서 MDM이 해결해야 하는 마스킹 문제가 계산적으로 NP-난해함을 수학적으로 증명했습니다 (예: L&O distribution 기반 증명).

추론 전략 – Adaptive Inference:

MDM의 특징인 자유로운 디코딩 순서를 활용하여, 추론 시 더 쉬운 위치의 토큰부터 해제하도록 설계된 “적응형 추론(adaptive inference)” 전략을 제안합니다.

Top-K와 Top-K probability margin이라는 두 가지 oracle 방식으로, 현재 마스킹된 위치들 중에서 예측이 쉬운 순서부터 디코딩합니다.

훈련 없이 추론 성능 향상:

이 adaptive strategy는 MDM을 재학습하지 않고도, 추론 단계에서 성능을 획기적으로 향상시키는 방법입니다.




This work examines the training and inference properties of Masked Diffusion Models (MDMs), with a focus on the impact of token ordering. The core methodology includes:

Model Structure and Training Objective:

MDMs learn to reverse a random masking process applied to sequences by predicting masked tokens based on the unmasked context.

They are trained to solve all possible masking combinations (2^L subproblems), which is referred to as order-agnostic training.

In contrast, Autoregressive Models (ARMs) solve only L left-to-right prediction problems during training.

Theoretical Analysis of Training Hardness:

The authors prove that many of the subproblems MDMs face are computationally intractable, even under benign data distributions (e.g., L&O distributions).

Inference Strategy – Adaptive Inference:

Leveraging MDMs’ flexibility at inference time, they propose an adaptive decoding strategy that selects easier-to-predict token positions first.

Two oracle strategies are introduced: Top-K (based on max probability) and Top-K probability margin (based on the difference between top probabilities), which guide which tokens to unmask at each step.

Training-Free Performance Gains:

These adaptive inference methods lead to dramatic improvements in performance during inference without modifying the training procedure.




   
 
<br/>
# Results  



이 논문은 MDM과 ARM을 다양한 작업에서 비교하며, 특히 추론 단계에서 디코딩 순서를 조절했을 때 MDM이 얼마나 성능이 향상되는지를 보여줍니다.

Logic Puzzle (Sudoku, Zebra Puzzle):

Sudoku 퍼즐에서, vanilla MDM은 **정확도 6.88%**에 불과했지만, 제안된 adaptive inference (Top-K probability margin) 전략을 사용하면 정확도가 **89.49%**로 급상승했습니다.

이 성능은 교사 강제 학습(teacher forcing)으로 순서를 학습한 ARM (87.18%)보다도 더 우수했습니다.

Zebra Puzzle에서도 vanilla MDM은 76.9%, Top-K margin은 98.3% 정확도를 기록했습니다.

Hard Sudoku (일반 학습보다 더 어려운 퍼즐):

adaptive MDM은 어려운 퍼즐에서도 49.88% 정확도로, ARM(32.57%)보다 훨씬 높은 성능을 보였습니다.

이는 적응형 추론 전략이 보이지 않는 데이터 분포 변화에도 강건함을 보여줍니다.

텍스트 생성 (HumanEval, MMLU 등):

8B 규모의 LLaDA 모델을 대상으로, adaptive inference가 텍스트 생성에서 generative perplexity 감소 및 정확도 향상을 가져왔습니다.

예: HumanEval-Split task에서 vanilla (14.2%) → Top-K margin (22.3%)로 상승.

실험 메트릭 및 비교 모델:

주요 평가 지표로는 정확도 (Accuracy), 생성 퍼플렉서티 (GenPPL), 엔트로피, log-likelihood 등을 사용.

비교 모델:

ARM (w/ and w/o decoding order)

Vanilla MDM

MDM + Top-K

MDM + Top-K probability margin




The paper reports substantial empirical findings demonstrating the effectiveness of adaptive inference for Masked Diffusion Models (MDMs), particularly in comparison to Autoregressive Models (ARMs):

Logic Puzzles (Sudoku, Zebra Puzzle):

On the Sudoku task, vanilla MDM achieves only 6.88% accuracy, whereas using adaptive inference with Top-K probability margin boosts accuracy to 89.49%.

This even outperforms ARMs trained with teacher-forced decoding order (87.18%).

In the Zebra puzzle, vanilla MDM scores 76.9%, while Top-K margin strategy achieves 98.3%.

Hard Sudoku Puzzles:

On unseen and more difficult Sudoku instances, adaptive MDM maintains robustness, achieving 49.88% accuracy versus 32.57% for ARM, demonstrating its strength under distribution shifts.

Text Generation Tasks (HumanEval, MMLU, etc.):

Using the 8B LLaDA model, adaptive MDM inference consistently reduced generative perplexity and improved accuracy.

Example: On the HumanEval-Split Line task, accuracy increased from 14.2% (vanilla) to 22.3% (Top-K margin).

Evaluation Metrics and Baselines:

Metrics: Accuracy, Generative Perplexity (GenPPL), Entropy, and Log-likelihood.

Baselines:

ARM (with and without decoding order)

Vanilla MDM

MDM + Top-K

MDM + Top-K probability margin




<br/>
# 예제  


이 논문에서는 MDM의 훈련과 추론 성능을 분석하기 위해 세 가지 주요 태스크를 사용했습니다:

1.  Logic Puzzle – Sudoku & Zebra Puzzle
입력 (Training Input): 부분적으로 마스킹된 퍼즐 상태 (예: 빈 칸이 있는 Sudoku 보드).

출력 (Training Output): 정답 숫자나 논리 퍼즐의 해답.

테스크 설명:

각 퍼즐은 토큰 시퀀스로 표현되며, 일부 위치는 마스킹됨.

MDM은 마스킹된 위치의 정답을 예측하는 방식으로 훈련됨.

목표: 전체 퍼즐을 정확히 푸는 것 (정확히 채운 보드).

2.  텍스트 생성 – HumanEval, MMLU, Math 등
입력: 마스킹된 문장 혹은 코드 조각 (e.g., 함수 이름이나 변수 일부가 가려짐).

출력: 마스킹된 부분의 복원된 텍스트 (ex. 적절한 단어, 코드 라인).

테스크 설명:

LLaDA 모델을 사용하여 문장이나 코드를 생성하는 작업.

다양한 난이도의 자연어/수학 문제도 포함됨.

3.  Synthetic Logic Task – L&O-NAE-SAT
입력: 관찰값(observations)만 주어진 상황에서, 숨겨진 잠재 토큰(latent tokens)을 예측.

출력: 관찰값을 만족시키는 잠재 변수들의 값.

테스크 설명:

논리식을 구성하는 숨겨진 토큰을 추론해야 하는 난해한 태스크.

마스킹된 위치를 예측하는 MDM의 성능을 정밀하게 측정함.



The paper evaluates MDMs using three key tasks with distinct input-output structures:

1.  Logic Puzzles – Sudoku & Zebra Puzzle
Input (Training): Partially masked logic puzzle (e.g., a Sudoku grid with missing entries).

Output (Training): The correct values for masked cells.

Task Description:

Each puzzle is represented as a token sequence with certain positions masked.

MDMs are trained to predict the correct tokens in masked positions.

Goal: Fully and correctly solve the puzzle.

2.  Text Generation – HumanEval, MMLU, Math Benchmarks
Input: Sentences or code snippets with masked tokens (e.g., missing variable names, functions).

Output: The correct completions of masked parts (e.g., valid word or code lines).

Task Description:

Tasks involve natural language and code generation using the LLaDA model.

Includes a range of tasks with increasing difficulty levels.

3.  Synthetic Logic Task – L&O-NAE-SAT
Input: Only observation tokens provided; latent tokens are masked.

Output: Predict the values of latent tokens that satisfy logical constraints.

Task Description:

A logic-based synthetic task where MDM must infer hidden variables.

Designed to evaluate how well MDM handles complex subproblems during inference.





<br/>  
# 요약   


이 논문은 Masked Diffusion Model(MDM)이 순서에 구애받지 않고 마스킹된 토큰을 예측하도록 학습하며, 학습 시 복잡한 문제를 다루는 대신 추론 시 디코딩 순서를 자유롭게 선택할 수 있다는 특징을 갖는다고 분석한다.
Sudoku, Zebra Puzzle, HumanEval 등의 태스크에서 adaptive inference 전략(Top-K probability margin 등)을 사용하면 MDM의 성능이 vanilla 버전보다 현저히 향상되며, ARM보다도 높은 정확도를 기록하였다.
특히 Sudoku에서는 정확도가 6.88%에서 89.49%로 크게 상승하며, 추론 시 어려운 위치를 피하는 전략만으로도 재학습 없이 성능을 극적으로 끌어올릴 수 있음을 입증하였다.


This paper analyzes how Masked Diffusion Models (MDMs) are trained to predict masked tokens in an order-agnostic manner, trading off training complexity for flexible inference that allows arbitrary decoding orders.
On tasks like Sudoku, Zebra puzzles, and HumanEval, adaptive inference strategies (e.g., Top-K probability margin) significantly improve MDM performance, surpassing even Autoregressive Models (ARMs) trained with ordering supervision.
For instance, in the Sudoku task, accuracy jumps from 6.88% to 89.49%, showing that adaptively avoiding difficult subproblems during inference leads to dramatic gains without retraining.



<br/>  
# 기타  





 Figure 1
내용: MDM의 학습은 난이도 불균형(intractable vs. easy subproblems)이 존재하며, 추론 시 adaptive strategy를 적용하면 어려운 문제를 회피할 수 있다는 개념도.

인사이트: 단순한 디코딩 순서 조절만으로도 학습 시의 계산 복잡성을 보완 가능함을 시각적으로 제시.



 Figure 2
왼쪽: MDM은 ARM보다 더 많은 subproblem을 학습하지만, 대부분의 마스킹 문제는 더 어렵고 likelihood가 낮음 (log(FLOPs) vs. -log p).

오른쪽: 텍스트와 논리 퍼즐 데이터에서, MDM은 관측 위치(observed positions)는 잘 예측하지만, 잠재 위치(latent positions)에서는 큰 오류 발생.

인사이트: MDM은 일부 마스킹 유형에 취약하며, inference 시 이러한 어려운 위치를 피하는 것이 중요.



 Table 1 (L&O-NAE-SAT Task Accuracy)
결과: Adaptive inference가 vanilla보다 10~30% 높은 정확도를 기록.

인사이트: Oracle 없이도 학습된 MDM 자체가 유용한 순서 정보를 내포하고 있음을 의미.



 Table 2 & 3 (Sudoku, Zebra Puzzle Accuracy)
결과: Top-K probability margin 기반 adaptive inference는 vanilla MDM 및 ARM(without ordering)보다 월등한 성능을 보임.

인사이트: 논리 추론 문제에서 디코딩 순서가 중요하며, 이를 adaptive하게 조절하는 것이 기존 모델을 뛰어넘을 수 있는 핵심임.



 Table 4 (Text Tasks – LLaDA 8B 모델)
결과: HumanEval, Math 등에서 vanilla → Top-K margin 전략으로 최대 8% 정확도 향상.

인사이트: MDM은 논리 퍼즐 뿐 아니라 일반 텍스트 생성에도 적용 가능하며, 추론 전략이 성능을 좌우함.



 Table 5 (Hard Sudoku Generalization)
결과: 학습보다 어려운 퍼즐에서도 ARM보다 adaptive MDM이 더 높은 정확도(49.88%) 기록.

인사이트: 적응형 추론 전략은 distribution shift 상황에서도 강건함.



 Appendix B
내용: L&O 분포에서 마스킹 문제가 계산적으로 어려움(NP-hard)을 이론적으로 증명함.

인사이트: 학습 시 어려운 하위 문제들이 실제로 존재하며, inference에서 이를 회피하는 것이 실질적으로 중요함.





 Figure 1
Content: Illustrates that MDM training involves a spectrum of subproblem difficulty, but adaptive inference helps skip hard ones.

Insight: Visualization of how adaptive token selection during inference can overcome training-time complexity.



 Figure 2
Left: Shows that MDM subproblems are more computationally intensive (lower log-likelihood) than ARM’s.

Right: Demonstrates performance imbalance—MDMs perform well on observed positions but poorly on latent ones.

Insight: Highlights the need to avoid latent positions during inference to boost reliability.



 Table 1 (L&O-NAE-SAT Task Accuracy)
Result: Adaptive inference boosts accuracy by 10–30% over vanilla MDM.

Insight: Even without retraining, MDM logits encode useful cues for guiding decoding order.



 Table 2 & 3 (Sudoku & Zebra Puzzle Accuracy)
Result: Top-K probability margin inference dramatically outperforms vanilla MDM and ARMs without order supervision.

Insight: Logical reasoning tasks benefit significantly from dynamic, sequence-dependent decoding.



 Table 4 (LLaDA 8B – Text Benchmarks)
Result: Accuracy improvements of up to 8% in HumanEval and Math tasks via Top-K margin.

Insight: Adaptive MDM inference generalizes well to real-world generation tasks beyond puzzles.



 Table 5 (Hard Sudoku – Generalization)
Result: On unseen, harder Sudoku puzzles, adaptive MDM achieves 49.88% vs. ARM’s 32.57%.

Insight: Suggests robustness of adaptive inference under distribution shift and task difficulty escalation.



 Appendix B
Content: Theoretical proof that many MDM subproblems (e.g., in L&O distributions) are NP-hard.

Insight: Motivates the need for inference-time strategies to avoid such intractable subproblems.




<br/>
# refer format:     



@inproceedings{kim2025train,
  title     = {Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions},
  author    = {Jaeyeon Kim and Kulin Shah and Vasilis Kontonis and Sham Kakade and Sitan Chen},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025},
  series    = {Proceedings of Machine Learning Research},
  volume    = {267},
  address   = {Vancouver, Canada},
  publisher = {PMLR}
}




Jaeyeon Kim, Kulin Shah, Vasilis Kontonis, Sham Kakade, and Sitan Chen. “Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions.” In Proceedings of the 42nd International Conference on Machine Learning (ICML), vol. 267, Vancouver, Canada: PMLR, 2025.

