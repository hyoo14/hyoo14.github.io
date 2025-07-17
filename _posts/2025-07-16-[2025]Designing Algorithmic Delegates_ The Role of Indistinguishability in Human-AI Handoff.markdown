---
layout: post
title:  "[2025]Designing Algorithmic Delegates: The Role of Indistinguishability in Human-AI Handoff"  
date:   2025-07-16 18:52:40 -0800
categories: study
---

{% highlight ruby %}


한줄 요약: 



짧은 요약(Abstract) :    



AI 기술이 발전하면서 사람들은 점점 더 AI 에이전트에게 의사결정을 위임하게 됩니다. 이 논문은 사람들이 ‘관찰 가능한 특징’만으로 판단해서 유사한 상황들을 하나의 범주(category)로 처리한다는 인지적 한계를 고려해, 인간-AI 협업에서 최적의 알고리즘적 대리인(delegate)을 설계하는 문제를 정의합니다.

핵심적으로, 인간은 자신이 관찰할 수 있는 정보만을 기준으로 AI에게 업무를 위임할지 직접 처리할지를 결정하고, AI 역시 그가 볼 수 있는 정보만으로 행동합니다. 이러한 정보의 제약 속에서 인간과 AI가 각각 구분하는 **범주(category)**가 서로 다를 수 있고, 이를 기반으로 **최적의 위임 정책(delegate design)**을 만드는 것이 이 연구의 주된 목표입니다.

저자들은 이 최적 대리인 설계 문제가 단순히 가장 정확한 AI를 만드는 것보다 훨씬 복잡하며, 조합 최적화(combinatorial optimization) 성격을 지닌다고 보여줍니다. 일반적으로 이 문제는 계산적으로 어려운 문제(NP-hard)이지만, 특정 조건에서는 효율적으로 해결할 수 있는 알고리즘도 제안합니다. 마지막으로, 시뮬레이션 실험을 통해 실사용 시 AI가 어떻게 점진적으로 개선되는지도 분석합니다.




As AI technologies improve, people are increasingly willing to delegate tasks to AI agents. Often, humans decide whether to delegate based on observable properties of a decision-making instance, treating instances with the same observable features as identical—a process rooted in cognitive categorization. In this paper, we define the problem of designing the optimal algorithmic delegate in the presence of such human and machine-defined categories.

This design task is crucial because the optimal delegate can be significantly more effective than the best standalone algorithm. However, the solution is non-obvious: the problem is fundamentally combinatorial, and even in simple settings, optimal designs depend on the intricate structure of the task. While finding the optimal delegate is computationally hard in general, the authors identify efficient algorithms for specific cases, such as when actions can be decomposed based on separately observable features.

They also simulate a process where a designer updates a delegate over time based on user adoption patterns. Though this does not guarantee global optimality, it often yields high-performing solutions. Overall, this work highlights the importance of tailoring AI designs not only to accuracy, but also to the way humans categorize and delegate tasks.


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






1. **인간-AI 위임 모델 정의**:

   * 세 개의 행위자(agent)가 존재: **인간**, **기계(AI)**, 그리고 **기계 설계자**.
   * 각각이 관찰할 수 있는 특징(feature)들이 제한되어 있으며, 이로 인해 동일한 의사결정 인스턴스를 서로 다르게 구분합니다.
   * 이를 통해 \*\*"인간 범주(Human Categories)"\*\*와 \*\*"기계 범주(Machine Categories)"\*\*가 정의됩니다.

2. **결정 흐름 모델링**:

   * 인간은 자신이 관찰한 범주(HC)에 따라 해당 인스턴스를 AI에게 위임할지 직접 처리할지 결정합니다.
   * 위임 시 AI는 자신이 관찰한 범주(MC)를 기반으로 사전 정의된 정책(policy)에 따라 행동합니다.
   * 인간은 항상 예측 손실이 더 적은 쪽(AI 또는 자기 자신)을 선택합니다.

3. **최적 대리인(delegate) 정의 및 공식화**:

   * 전체 인스턴스 공간은 이산적인 상태들(state)로 구성되어 있고, 각 상태에 대해 최적 행동 $f^*(x)$와 손실 함수 $(a - f^*(x))^2$가 주어집니다.
   * 기계의 행동 정책 $f_M$은 각 기계 범주에 대해 어떤 행동을 취할지를 정의합니다.
   * \*\*목표는 전체 팀 손실을 최소화하는 $f_M$\*\*을 찾는 것.

4. **조합 최적화로 변환**:

   * 최적 대리인 설계 문제는 특정한 인간 범주 집합 $R$을 “기계가 담당할 범주”로 선택하는 문제로 환원됩니다.
   * 이 문제는 \*\*범주 내 분산(variance)\*\*과 **기계가 사용하는 상태들의 조건부 분산**에 기반한 목적함수를 최소화하는 문제로 정식화됩니다.

5. **계산 복잡도 및 알고리즘**:

   * 일반적으로는 \*\*NP-난해(NP-hard)\*\*한 문제이지만, 다음과 같은 경우에는 **효율적인 알고리즘**이 존재함:

     * 인간과 기계의 특징이 독립적으로 관찰되며, 최적 행동이 선형(linear) 함수일 때.
     * 인간이나 기계가 관찰하는 특징 수가 제한적일 때.

6. **모의 실험(시뮬레이션)**:

   * 기계가 인간에게 채택(adopt)된 범주를 바탕으로 점진적으로 자신을 개선하는 시나리오를 모의 실험으로 분석.
   * 이는 최적 해에 수렴하지는 않지만 꽤 높은 성능의 위임 정책을 생성함.

---



The method proposed in this paper involves a formal framework for designing **optimal algorithmic delegates** in human-AI collaboration settings. The main components are:

1. **Delegation Model Setup**:

   * The system includes three actors: a **human**, a **machine (AI agent)**, and a **machine designer**.
   * Each agent observes only a subset of features, leading to **Human Categories (HC)** and **Machine Categories (MC)**, i.e., indistinguishable sets of instances from each perspective.

2. **Decision-Making Process**:

   * The human observes an HC and decides whether to delegate or act themselves based on expected loss.
   * If delegated, the machine observes an MC and takes action based on a predefined policy.
   * The human always delegates only if the machine has lower expected loss in that category.

3. **Formalization of Optimal Delegation**:

   * Each instance is defined by a feature vector, with a ground-truth action $f^*(x)$ and squared loss function $(a - f^*(x))^2$.
   * The machine’s action function $f_M$ assigns an action to each MC.
   * The **objective** is to find the machine policy $f_M$ that minimizes total **team loss**.

4. **Reformulation as Combinatorial Optimization**:

   * The key insight is that optimal design reduces to selecting a subset $R$ of human categories to “retain” for delegation.
   * The loss function becomes a sum of within-category variances and cross-category variances over machine categories.

5. **Computational Tractability & Algorithms**:

   * The delegation problem is **NP-hard** in general.
   * However, **efficient algorithms** are derived for cases where:

     * The ground-truth function is **separable or linear**, and feature distributions are independent.
     * The number of features observed by human or machine is small.

6. **Simulated Iterative Design**:

   * The authors simulate an iterative process where the machine is continually refined based on actual user delegation patterns.
   * Although not globally optimal, this adaptive method often leads to strong-performing delegates.




   
 
<br/>
# Results  




1. **경쟁 모델**

   * 주요 비교 대상은 \*\*"oblivious delegate" (비적응형 일반 기계)\*\*입니다. 이 기계는 사용자가 실제로 위임할지 여부와 무관하게 **모든 범주에 대해 평균적인 행동**을 하도록 설계된 것입니다.
   * 또 하나의 비교 대상은 **iteratively trained delegate**로, 실제 사용자의 위임 패턴을 관찰하여 반복적으로 재설계된 모델입니다.

2. **평가 척도 (Metric)**

   * 팀 성능은 \*\*예측 손실(team loss)\*\*로 평가되며, 이는 주어진 상태에서 인간 또는 기계가 선택한 행동과 실제 최적 행동 간의 \*\*제곱 오차 (squared loss)\*\*로 계산됩니다.
   * 팀 손실은 각 인간 범주 $C$에 대해 인간과 기계 중 더 낮은 손실을 선택한 후 전체에 대해 가중 평균을 취해 산출합니다:

     $$
     \ell(f_H, f_M) = \sum_C P(C) \cdot \min\{\ell_H(C), \ell_M(C)\}
     $$

3. **테스트 시나리오**

   * 인위적으로 구성된 예제들과 확률 분포에 기반한 **시뮬레이션 설정**을 사용함.
   * 특히 두 개의 특징만 사용하는 **이진 피처 공간**과, 인간과 기계가 각각 하나의 피처만 관찰할 수 있는 간단한 구조에서도, 최적 대리인은 복잡하고 비직관적인 정책 구조를 갖게 됨.

4. **주요 결과**

   * \*\*최적 대리인(delegate)\*\*은 oblivious delegate에 비해 팀 성능을 **임의의 수준까지 향상시킬 수 있음**. 즉, oblivious delegate가 갖는 팀 손실은 무한히 커질 수 있는 반면, 최적 대리인은 일정 수준 이하의 손실을 유지함.
   * 반복적(delegate adoption 기반) 학습 방식은 일반적으로 꽤 좋은 성능을 내지만, \*\*전역 최적(global optimum)\*\*에는 도달하지 못하며 \*\*지역 최적(local optimum)\*\*에 머무름.
   * 실험에서, 사람이 쉽게 처리할 수 있는 범주는 AI에게 위임하지 않는 것이 오히려 전체 성능을 높이는 데 도움이 됨을 보여줌.
   * 문제는 일반적으로 **NP-hard**이지만, **선형 함수 또는 feature 수가 적을 경우 tractable**한 구조를 가지며, 효율적 알고리즘으로 최적 대리인을 찾을 수 있음.

---



1. **Baselines**

   * The main baseline is the **oblivious delegate**, a machine that takes average actions across machine-observable categories, regardless of whether users actually delegate.
   * Another comparison is the **iteratively trained delegate**, which adapts based on observed user adoption patterns over multiple iterations.

2. **Evaluation Metric**

   * The primary metric is **team loss**, computed as the squared difference between the chosen action (by human or machine) and the ground-truth optimal action.
   * In each human category $C$, the team selects the agent (human or machine) with the **lower expected loss**, and computes a weighted average across categories:

     $$
     \ell(f_H, f_M) = \sum_C P(C) \cdot \min\{\ell_H(C), \ell_M(C)\}
     $$

3. **Test Scenarios**

   * The authors use both **theoretical constructions** and **simulated setups** with binary features and known distributions.
   * Even in the simplest setup — with two binary features, where the human and machine each observe only one — the structure of the optimal delegate turns out to be surprisingly complex and non-linear.

4. **Key Findings**

   * The **optimal delegate** can **arbitrarily outperform** the oblivious machine: the team loss for the oblivious machine can grow unbounded, while the optimal delegate maintains bounded loss.
   * The **iterative refinement** strategy (based on user delegation feedback) typically performs well, but it does **not guarantee convergence to the global optimum** — it can converge to suboptimal local minima.
   * Delegating only in categories where the human performs poorly improves overall team performance.
   * While the general problem is **NP-hard**, it is tractable in specific settings such as when the ground truth is **linear** or when either agent observes **few features**.





<br/>
# 예제  





1. **인풋 구성**

   * 피처 수: 2개 (이진 변수 $x_1, x_2 \in \{0,1\}$)
   * 가능한 상태 수(state): 총 4개 → (0,0), (0,1), (1,0), (1,1)
   * 인간은 $x_1$만 관찰하고, 기계는 $x_2$만 관찰함
     → 인간 범주는 $x_1$에 따라 두 개 (C₁: x₁=0, C₂: x₁=1)
     → 기계 범주는 $x_2$에 따라 두 개 (K₁: x₂=0, K₂: x₂=1)

2. **출력 (Ground truth 함수)**

   * 각 상태에서의 이상적 행동 값(즉, 최적 가격, 판단 등)을 아래처럼 정의:

     ```
     f*(x11) = 0  (x₁=0, x₂=0)
     f*(x12) = 1  (x₁=0, x₂=1)
     f*(x21) = a  (x₁=1, x₂=0)
     f*(x22) = b  (x₁=1, x₂=1)
     ```

     * 여기서 a와 b는 다양한 실험을 위해 임의로 설정됨 (예: a=0.3, b=0.9 등)

3. **예시 시나리오:**

   * 사람이 x₁=1인 상태 (C₂)에서는 평균적으로 좋은 행동을 선택할 수 있기 때문에, 이 범주는 사람이 처리함.
   * x₁=0인 상태 (C₁)에서는 기계의 예측이 더 나을 수 있으므로 위임하게 됨.
   * 기계는 K₁ (x₂=0)과 K₂ (x₂=1)에 대해 평균적으로 어떤 값을 선택할지를 학습하거나 설계됨.

4. **팀 행동 예시**

   * C₁에서 인간이 기계에 위임함 → 기계는 K₁, K₂를 보고 행동함.
   * C₂에서는 인간이 직접 행동함 → 평균적으로 $a$와 $b$에 기반해 결정함.
   * 그 결과로 전체 팀 손실이 계산됨.

---



1. **Input Setup**

   * Number of features: 2 binary variables $x_1, x_2 \in \{0,1\}$
   * Total number of states: 4 → (0,0), (0,1), (1,0), (1,1)
   * The **human** observes only $x_1$, and the **machine** observes only $x_2$:
     → Human categories: C₁ for $x_1 = 0$, C₂ for $x_1 = 1$
     → Machine categories: K₁ for $x_2 = 0$, K₂ for $x_2 = 1$

2. **Output (Ground Truth Function)**

   * The ideal action in each state is defined as:

     ```
     f*(x11) = 0  (x₁=0, x₂=0)
     f*(x12) = 1  (x₁=0, x₂=1)
     f*(x21) = a  (x₁=1, x₂=0)
     f*(x22) = b  (x₁=1, x₂=1)
     ```

     * $a$ and $b$ are variables used to explore different behaviors (e.g., a=0.3, b=0.9)

3. **Scenario Example**

   * In category C₂ (where $x_1 = 1$), the human performs well and chooses to act.
   * In category C₁ (where $x_1 = 0$), the human delegates to the machine.
   * The machine uses its observed category (K₁ or K₂) to determine the action.

4. **Team Behavior Example**

   * When in C₁: the machine takes action based on its category (K₁ or K₂).
   * When in C₂: the human acts based on the average of $a$ and $b$.
   * Overall, the system computes team loss based on who acted and the squared difference from $f^*$.

---



<br/>  
# 요약   


이 논문은 인간과 AI가 각기 다른 정보를 관찰하며 의사결정할 때, 위임(delegate)이 최적으로 이루어지도록 설계된 알고리즘 대리인을 만드는 모델을 제안한다. 저자들은 이 문제를 조합 최적화로 정식화하고, 일반적으로는 NP-hard이지만 특정 조건에서는 효율적인 해법이 가능함을 보였다. 예제로는 2개의 이진 피처만을 사용하는 설정에서, 인간과 기계가 각자 하나의 피처만 관찰할 때도 최적 대리인이 복잡한 구조를 가진다는 것을 수치 실험으로 확인했다.




This paper proposes a model for designing algorithmic delegates that enable optimal delegation when humans and AI observe different subsets of information. The authors formulate this as a combinatorial optimization problem and show that while it is NP-hard in general, efficient solutions exist in specific cases. Through simulations using two binary features, they demonstrate that even simple setups can lead to complex optimal delegation strategies.







<br/>  
# 기타  




####  Figure 1: 인간 vs 기계 범주의 개념도

* 인간은 관찰 가능한 피처만을 기반으로 인스턴스를 분류하며, AI도 자신만의 범주로 상태를 구분한다.
* **인사이트**: 같은 인스턴스라도 인간과 AI가 전혀 다르게 구분하고 판단할 수 있음을 시각적으로 표현.
  → 인간의 위임 결정은 인간 범주(HC), 기계의 행동은 기계 범주(MC)에 따라 달라진다.

---

####  Figure 4: 최적 위임 구조 예시

* 격자 형태로 모든 상태(x₁, x₂ 조합)를 시각화하고, 각 칸에는 최적 행동 $f^*(x)$이 표시됨.
* (b)는 인간이 직접 행동할 때의 평균 손실, (c)는 기계가 처리할 때의 평균 손실, (d)-(e)는 전체 팀 손실 비교를 보여줌.
* **인사이트**: 기계가 전체 범주를 다룰 때보다, 실제로 위임받은 범주만 고려해 행동을 설계했을 때 훨씬 더 낮은 팀 손실을 달성할 수 있다.

---

####  Figure 7: 2D 예시에서 $f^*(x)$에 따른 최적 위임 전략 변화

* (a): 각 $a, b$ 조합에 따른 팀 손실을 컬러맵으로 시각화
* (b): 어떤 범주(C₁ 또는 C₂)를 기계가 처리해야 최적인지 구간별로 시각화
* **인사이트**: 단 4개의 상태만 있어도 최적 위임 전략이 비선형적이고 비연결적인 복잡한 구조를 보임 → 위임 설계는 단순 평균으로 해결되지 않는다.

---

####  Figure 8: 반복적(delegate-feedback 기반) 학습 vs 최적 위임 비교

* 반복적으로 사용자 위임 데이터를 보고 기계를 재설계하는 과정과 최적 대리인과의 차이를 시각화
* **인사이트**: 반복 설계는 국소 최적에 수렴할 수 있지만, **전역 최적과는 전혀 다른 행동**을 선택하게 될 수 있음.

---

####  Appendix A & B

* A: oblivious delegate와 최적 delegate의 성능 차이를 이론적으로 정리하며, oblivious가 임의로 나쁠 수 있음을 증명
* B: 반복 업데이트로 수렴한 delegate의 성능을 분석하여, 실제로는 꽤 괜찮은 성능을 내지만 항상 최적은 아님을 실험적으로 보여줌
* **인사이트**: 단순한 설계나 학습 루틴은 인간의 위임 행동을 충분히 반영하지 못할 수 있으며, 이를 고려한 설계가 중요하다.

---


####  Figure 1: Human vs. Machine Categories

* Visualizes how humans and machines partition the same decision-making space using different observable features.
* **Insight**: Humans and machines may perceive the same situation differently, leading to distinct delegation and action strategies.

---

####  Figure 4: Delegation Strategy Example

* A grid shows each (x₁, x₂) state with the corresponding optimal action $f^*(x)$.
* Panels (b) and (c) compute expected losses when the human or machine acts alone; (d) and (e) compare team performance under different delegates.
* **Insight**: A machine that is designed specifically for the human-delegated categories (not all categories) yields significantly lower team loss.

---

####  Figure 7: Optimal Strategy by Ground Truth $f^*(x)$

* (a): Heatmap of team loss for all combinations of $a, b$ (two unknown values of $f^*(x)$)
* (b): Which human category (C₁ or C₂) the machine should retain, shown as regions
* **Insight**: Even in a simple 2-feature world, the optimal delegation strategy is non-linear and fragmented—delegation decisions require more than averaging.

---

####  Figure 8: Iterative vs. Optimal Delegate

* Visualizes how an iteratively updated machine changes based on human adoption, compared to the globally optimal delegate
* **Insight**: The iterative method may converge to a completely different policy than the optimal one, highlighting the risk of relying only on behavioral feedback.

---

####  Appendix A & B

* A: Theoretically demonstrates that oblivious delegates can perform arbitrarily worse than optimal ones
* B: Empirically shows that iteratively trained delegates often perform well, but may not reach the optimal team performance
* **Insight**: Realistic machine designs must account for human delegation behavior—naive or uniform training may miss key performance opportunities.




<br/>
# refer format:     



@inproceedings{greenwood2025delegates,
  author    = {Sophie Greenwood and Karen Levy and Solon Barocas and Hoda Heidari and Jon Kleinberg},
  title     = {Designing Algorithmic Delegates: The Role of Indistinguishability in Human-AI Handoff},
  booktitle = {Proceedings of the 26th ACM Conference on Economics and Computation (EC ’25)},
  year      = {2025},
  pages     = {306--336},
  publisher = {ACM},
  address   = {Stanford, CA, USA},
  doi       = {10.1145/3736252.3742533},
  url       = {https://doi.org/10.1145/3736252.3742533}
}




Sophie Greenwood, Karen Levy, Solon Barocas, Hoda Heidari, and Jon Kleinberg.
“Designing Algorithmic Delegates: The Role of Indistinguishability in Human-AI Handoff.”
In Proceedings of the 26th ACM Conference on Economics and Computation (EC ’25), 306–336. Stanford, CA, USA: ACM, 2025. https://doi.org/10.1145/3736252.3742533.









