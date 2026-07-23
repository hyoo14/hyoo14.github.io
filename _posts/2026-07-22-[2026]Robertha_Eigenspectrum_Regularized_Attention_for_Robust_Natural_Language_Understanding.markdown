---
layout: post
title:  "[2026]Robertha: Eigenspectrum Regularized Attention for Robust Natural Language Understanding"
date:   2026-07-22 17:54:05 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

Robertha는 Modern Hopfield Attention과 Eigenspectrum Regularization을 결합해 손상된 임베딩을 안정적인 의미 표현으로 복원하는 강건한 NLU 모델   


짧은 요약(Abstract) :



Encoder-based language models are vulnerable to embedding corruption, where fixed-magnitude perturbations disproportionately affect low-norm embeddings and degrade semantic representations. This vulnerability is particularly severe for function words that play a critical role in preserving grammatical and semantic structure. Existing robustness methods mainly rely on adversarial training or data augmentation, often sacrificing clean accuracy or failing to generalize under stronger corruption.

We propose Robertha, a robustness framework that integrates Modern Hopfield attention with Eigenspectrum Regularization (ESR) to recover corrupted semantic representations. Our iterative Hopfield retrieval refines noisy embeddings toward stable semantic attractors, while ESR regularizes the attention eigenspectrum to form well-separated attractor basins with improved recovery capability. Extensive experiments on GLUE and SuperGLUE benchmarks demonstrate that Robertha consistently improves robustness under embedding perturbations while maintaining competitive clean-task performance. Additional analyses show that the proposed framework effectively reduces the asymmetric vulnerability of low-norm embeddings and provides more stable representation learning under noisy conditions.

⸻



Encoder 기반 언어모델은 임베딩에 동일한 크기의 잡음이 가해질 경우, 크기가 작은(low-norm) 임베딩이 상대적으로 더 큰 영향을 받아 의미 표현이 쉽게 손상되는 문제를 가진다. 이러한 취약성은 문법 및 의미 구조를 유지하는 데 중요한 기능어에서 더욱 두드러진다. 기존의 강건성 향상 기법은 주로 적대적 학습이나 데이터 증강에 의존하며, 깨끗한 데이터에서의 성능 저하 또는 강한 잡음 환경에서의 일반화 한계를 보인다.

본 논문에서는 Modern Hopfield Attention과 **Eigenspectrum Regularization (ESR)**을 결합한 강건성 향상 프레임워크인 Robertha를 제안한다. 제안 방법은 반복적인 Hopfield retrieval을 통해 손상된 의미 표현을 안정적인 semantic attractor 방향으로 복원하고, ESR을 이용해 attention의 eigenspectrum을 정규화함으로써 분리된 attractor basin을 형성하여 복원 능력을 향상시킨다. GLUE 및 SuperGLUE 벤치마크에 대한 실험 결과, 제안 방법은 임베딩 교란 환경에서 기존 방법보다 우수한 강건성을 달성하면서도 깨끗한 데이터에서 경쟁력 있는 성능을 유지하였다. 또한 추가 분석을 통해 low-norm 임베딩의 비대칭적 취약성을 완화하고, 잡음 환경에서 더욱 안정적인 표현 학습이 가능함을 확인하였다.

⸻


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




They propose Robertha, a robustness-oriented encoder architecture that mitigates semantic degradation caused by embedding perturbations through iterative semantic retrieval and spectral regularization.

Embedding Corruption

To evaluate robustness, Gaussian perturbations are injected into the input embedding space:

\tilde{\mathbf{e}}=\mathbf{e}+\boldsymbol{\epsilon},
\qquad
\boldsymbol{\epsilon}\sim\mathcal{N}(0,\sigma^2I),

where \mathbf{e} denotes the original token embedding and \tilde{\mathbf{e}} represents the corrupted embedding. Since low-norm embeddings receive proportionally larger perturbations, semantic information encoded by function words becomes particularly vulnerable.

Iterative Modern Hopfield Retrieval

Instead of directly processing corrupted embeddings, Robertha employs iterative Modern Hopfield attention to progressively refine noisy semantic representations.

Given the current representation, attention retrieves the most relevant semantic memories stored in the key-value space and updates the representation through repeated retrieval. This iterative refinement gradually moves corrupted embeddings toward stable semantic attractors, enabling recovery from perturbations while preserving contextual information.

Eigenspectrum Regularization

To improve the stability of semantic retrieval, we introduce Eigenspectrum Regularization (ESR) on the attention key representations.

ESR encourages a compact eigenspectrum by reducing eigenvalue entropy, producing well-separated attractor basins in the representation space. Consequently, corrupted representations are more likely to converge to their corresponding semantic attractors rather than drifting toward unstable regions.

Training Objective

The proposed model is optimized using the standard task loss together with the ESR objective,

\mathcal{L}
=
\mathcal{L}_{task}
+
\lambda
\mathcal{L}_{ESR},

where \lambda controls the contribution of spectral regularization. During inference, iterative Hopfield retrieval improves robustness without requiring adversarial training or additional data augmentation.

⸻



3. 제안 방법

본 논문에서는 임베딩 교란으로 인해 손상되는 의미 표현을 복원하기 위해 Modern Hopfield Attention과 **Eigenspectrum Regularization (ESR)**을 결합한 강건성 프레임워크인 Robertha를 제안한다.

임베딩 교란 모델

모델의 강건성을 평가하기 위해 입력 임베딩에 Gaussian 잡음을 추가한다.

\tilde{\mathbf{e}}
=
\mathbf{e}
+
\boldsymbol{\epsilon},
\qquad
\boldsymbol{\epsilon}
\sim
\mathcal{N}(0,\sigma^2I),

여기서 \mathbf{e}는 원래 토큰 임베딩이며, \tilde{\mathbf{e}}는 잡음이 추가된 임베딩을 의미한다. 동일한 크기의 잡음이 추가될 경우 low-norm 임베딩은 상대적으로 더 큰 영향을 받아 의미 표현이 쉽게 손상된다.

반복적인 Modern Hopfield Retrieval

손상된 임베딩을 그대로 사용하는 대신, 제안 모델은 반복적인 Modern Hopfield attention을 이용하여 의미 표현을 단계적으로 복원한다.

현재 표현을 query로 사용하여 attention memory에서 가장 유사한 semantic pattern을 반복적으로 검색하고, 이를 이용해 표현을 지속적으로 갱신한다. 이러한 iterative retrieval 과정은 손상된 표현을 안정적인 semantic attractor 방향으로 이동시키며, 잡음으로 인해 왜곡된 의미 정보를 점진적으로 복원한다.

Eigenspectrum Regularization

보다 안정적인 semantic retrieval을 위해 attention key representation에 **Eigenspectrum Regularization (ESR)**을 적용한다.

ESR은 attention representation의 eigenvalue entropy를 감소시켜 compact한 eigenspectrum을 형성하며, 표현 공간에서 서로 명확히 분리된 semantic attractor basin을 생성한다. 이를 통해 잡음이 포함된 표현이 불안정한 영역으로 이동하는 대신 올바른 의미 상태로 수렴하도록 유도한다.

학습 목표

최종 학습은 task loss와 ESR loss를 동시에 최적화한다.

\mathcal{L}
=
\mathcal{L}_{task}
+
\lambda
\mathcal{L}_{ESR},

여기서 \lambda는 regularization의 영향을 조절하는 하이퍼파라미터이다. 제안 방법은 별도의 adversarial training이나 데이터 증강 없이도 embedding perturbation 환경에서 높은 강건성을 확보할 수 있다.

⸻



Gaussian embedding perturbation → Iterative Hopfield retrieval → ESR로 semantic attractor를 강화 → 손상된 embedding을 원래 의미 표현으로 복원.



<br/>
# Results





4 Experimental Setup

Benchmarks and Evaluation Protocol

We evaluate Robertha on 13 natural language understanding tasks drawn from GLUE and SuperGLUE. The reported GLUE and SuperGLUE results are aggregate normalized scores on a 0–100 scale.

Robustness is evaluated by injecting isotropic Gaussian noise into input token embeddings at five corruption levels:

\sigma \in \{0,\;0.5,\;1.0,\;2.0,\;5.0\},

corresponding to clean, low, moderate, high, and extreme corruption, respectively. All experiments are repeated using five random seeds, and results are reported as mean ± standard deviation.

Comparison Models

We compare Robertha with the following baselines:

* TinyBERT: the standard encoder model trained without an explicit robustness mechanism.
* Data Augmentation: Gaussian noise is injected into embeddings during training.
* FreeLB: an adversarial training method that optimizes worst-case embedding perturbations.
* Robertha: the proposed architectural defense using iterative Modern Hopfield attention and Eigenspectrum Regularization.

Thus, the comparison covers standard training, noise-based training augmentation, adversarial training, and architectural robustness.

⸻

5 Main Results

Aggregate GLUE Results

Model	Clean	Low	Moderate	High	Extreme
TinyBERT	64.7 ± 0.4	50.2 ± 1.2	49.6 ± 1.2	49.1 ± 1.2	48.9 ± 1.1
FreeLB	64.7 ± 0.3	50.0 ± 1.0	49.2 ± 1.0	48.9 ± 1.0	48.7 ± 1.0
Data Augmentation	64.3 ± 0.2	50.3 ± 1.0	49.1 ± 0.8	48.7 ± 0.7	48.5 ± 0.6
Robertha	61.9 ± 0.4	61.4 ± 0.4	60.4 ± 0.5	58.1 ± 0.7	54.0 ± 0.5

Robertha incurs a 2.8-point clean-performance reduction relative to TinyBERT, but substantially improves performance at every corruption level. Under low corruption, Robertha achieves 61.4, compared with 50.3 for Data Augmentation and 50.0 for FreeLB. At high corruption, it reaches 58.1, outperforming Data Augmentation and FreeLB by 9.4 and 9.2 points, respectively.

Even under extreme corruption, Robertha retains a score of 54.0, whereas all training-based baselines fall below 49. This indicates that the proposed architectural mechanism degrades more gradually as the perturbation magnitude increases.

Aggregate SuperGLUE Results

Model	Clean	Low	Moderate	High	Extreme
TinyBERT	60.5 ± 0.7	51.9 ± 1.9	51.2 ± 2.0	51.1 ± 1.3	51.0 ± 1.3
FreeLB	58.5 ± 0.5	51.8 ± 1.3	51.4 ± 1.2	51.1 ± 1.1	51.0 ± 1.0
Data Augmentation	58.6 ± 0.3	52.5 ± 1.6	52.0 ± 1.5	51.8 ± 1.4	51.7 ± 1.3
Robertha	60.1 ± 0.5	59.3 ± 0.6	59.0 ± 1.3	57.4 ± 1.8	54.4 ± 1.4

On SuperGLUE, Robertha nearly matches TinyBERT on clean data, with only a 0.4-point difference. Its robustness advantage is more pronounced under corruption. At low corruption, it obtains 59.3, exceeding Data Augmentation by 6.8 points and FreeLB by 7.5 points. At high corruption, it reaches 57.4, compared with 51.8 for Data Augmentation and 51.1 for FreeLB.

These results suggest that the proposed method is particularly effective on reasoning-oriented tasks, where conventional adversarial training and noise augmentation provide limited protection.

⸻

5.1 Robustness–Clean Accuracy Trade-off

The results reveal a clear trade-off. Robertha sacrifices some clean GLUE accuracy in exchange for substantially improved robustness. The clean penalty is relatively small on SuperGLUE, but more noticeable on GLUE.

Consequently, the method should not be interpreted as uniformly superior under all conditions. Its main benefit appears when embedding corruption is expected, while a standard TinyBERT model remains preferable when only clean accuracy is considered.

⸻

5.2 Ablation Study

The paper evaluates the contributions of iterative retrieval and ESR separately.

Model Variant	GLUE Clean	GLUE Extreme	SuperGLUE Clean	SuperGLUE Extreme
Single-pass Hopfield, no ESR	62.2 ± 0.2	49.2 ± 1.2	59.8 ± 0.7	52.1 ± 1.5
Iterative Hopfield, no ESR	61.8 ± 0.3	52.8 ± 1.2	60.0 ± 0.5	51.8 ± 2.4
ESR replaced with Dropout	61.6 ± 0.7	53.6 ± 1.0	60.6 ± 0.5	52.4 ± 1.5
ESR replaced with Weight Decay	61.3 ± 0.8	50.9 ± 0.8	59.0 ± 1.0	53.0 ± 1.4
Full Robertha	61.9 ± 0.4	54.0 ± 0.5	60.1 ± 0.5	54.4 ± 1.4

Iteration alone improves GLUE robustness but does not consistently help SuperGLUE. Replacing ESR with Dropout or Weight Decay also gives inconsistent gains across the two benchmarks. The complete model achieves the best extreme-corruption result on both GLUE and SuperGLUE, supporting the authors’ claim that ESR contributes beyond generic regularization.

However, the ablation also shows that a meaningful portion of the GLUE improvement can already be obtained using iteration or Dropout. Therefore, ESR appears most convincing as a mechanism for consistency across task families rather than as the sole source of robustness.

⸻

5.3 Additional Robustness Evaluation

The authors additionally evaluate the model on AdvGLUE, PAWS, TextBugger, and HotFlip. Robertha reportedly outperforms the baselines on eight out of nine AdvGLUE and PAWS tasks and remains competitive under character-level perturbations.

The paper also reports that the effective rank of the attention key representations is strongly correlated with performance degradation as corruption increases. This analysis is used to support the proposed connection between eigenspectrum concentration and robustness.

⸻

5.4 Scalability and Computational Cost

A 110M-parameter Robertha variant is compared with BERT-base under high corruption on four tasks:

Model	CoLA	MRPC	WNLI	BoolQ
BERT-base	50.6	29.3	54.9	53.3
Robertha-large	54.9	74.8	63.4	62.2
Improvement	+4.3	+45.5	+8.5	+8.9

The method therefore appears to retain its robustness advantage at a larger encoder scale. Nevertheless, iterative retrieval increases inference latency. Depending on the corruption level and iteration budget, the paper reports approximately 1.25× to 3.54× higher latency.

⸻

5.5 Overall Evaluation

The empirical evidence supports the following conclusion:

Robertha is substantially more robust than TinyBERT, FreeLB, and Gaussian-noise data augmentation under the specific setting of additive embedding corruption.

However, the comparison does not establish superiority for general-purpose NLP robustness. The strongest evidence is limited to encoder-based classification models and embedding-space Gaussian corruption. The experiments do not directly demonstrate robustness on decoder-only language models, real quantization errors, prompt attacks, or natural text perturbations at contemporary LLM scale.

⸻



4. 실험 설정

데이터셋 및 평가 방식

Robertha는 GLUE와 SuperGLUE에 포함된 총 13개의 자연어 이해 태스크에서 평가되었다. 논문에서 보고하는 GLUE 및 SuperGLUE 결과는 각 태스크의 평가 점수를 0–100 범위로 정규화한 후 집계한 점수이다.

강건성 평가는 입력 토큰 임베딩에 isotropic Gaussian noise를 추가하는 방식으로 수행된다. 잡음의 세기는 다음 다섯 단계로 설정된다.

\sigma \in \{0,\;0.5,\;1.0,\;2.0,\;5.0\},

각각 clean, low, moderate, high, extreme corruption에 해당한다. 모든 실험은 서로 다른 5개의 random seed로 반복되며, 평균과 표준편차를 함께 보고한다.

비교 모델

비교 대상은 다음과 같다.

* TinyBERT: 별도의 강건성 기법 없이 일반적인 방식으로 학습한 기준 모델
* Data Augmentation: 학습 과정에서 임베딩에 Gaussian noise를 추가하는 방법
* FreeLB: 임베딩 공간의 worst-case perturbation을 이용하는 adversarial training 방법
* Robertha: 반복적인 Modern Hopfield attention과 ESR을 사용하는 제안 모델

따라서 비교 실험은 일반 학습, 잡음 기반 데이터 증강, 적대적 학습, 구조적 강건성 기법을 포함한다.

⸻

5. 주요 결과

GLUE 종합 결과

모델	Clean	Low	Moderate	High	Extreme
TinyBERT	64.7 ± 0.4	50.2 ± 1.2	49.6 ± 1.2	49.1 ± 1.2	48.9 ± 1.1
FreeLB	64.7 ± 0.3	50.0 ± 1.0	49.2 ± 1.0	48.9 ± 1.0	48.7 ± 1.0
Data Augmentation	64.3 ± 0.2	50.3 ± 1.0	49.1 ± 0.8	48.7 ± 0.7	48.5 ± 0.6
Robertha	61.9 ± 0.4	61.4 ± 0.4	60.4 ± 0.5	58.1 ± 0.7	54.0 ± 0.5

Robertha의 clean 성능은 TinyBERT보다 약 2.8점 낮다. 그러나 잡음이 추가되면 모든 corruption 수준에서 큰 성능 우위를 보인다.

Low corruption에서 Robertha는 61.4점을 기록하여 Data Augmentation보다 11.1점, FreeLB보다 11.4점 높다. High corruption에서는 58.1점으로 Data Augmentation보다 9.4점, FreeLB보다 9.2점 높은 성능을 보인다.

Extreme corruption에서도 Robertha는 54.0점을 유지하지만, 나머지 방법은 모두 49점 이하로 감소한다. 이는 잡음의 크기가 증가할 때 제안 모델의 성능 저하가 상대적으로 완만하다는 것을 보여준다.

SuperGLUE 종합 결과

모델	Clean	Low	Moderate	High	Extreme
TinyBERT	60.5 ± 0.7	51.9 ± 1.9	51.2 ± 2.0	51.1 ± 1.3	51.0 ± 1.3
FreeLB	58.5 ± 0.5	51.8 ± 1.3	51.4 ± 1.2	51.1 ± 1.1	51.0 ± 1.0
Data Augmentation	58.6 ± 0.3	52.5 ± 1.6	52.0 ± 1.5	51.8 ± 1.4	51.7 ± 1.3
Robertha	60.1 ± 0.5	59.3 ± 0.6	59.0 ± 1.3	57.4 ± 1.8	54.4 ± 1.4

SuperGLUE의 clean 환경에서는 Robertha가 TinyBERT보다 0.4점 낮아 사실상 유사한 성능을 유지한다.

Low corruption에서는 Robertha가 59.3점을 기록하여 Data Augmentation보다 6.8점, FreeLB보다 7.5점 높다. High corruption에서는 57.4점으로 Data Augmentation보다 5.6점, FreeLB보다 6.3점 높다.

따라서 reasoning 비중이 높은 SuperGLUE에서도 일반적인 adversarial training이나 noise augmentation보다 제안 구조가 더 안정적으로 작동한다고 저자들은 해석한다.

⸻

5.1 Clean 성능과 강건성의 Trade-off

결과에는 명확한 trade-off가 존재한다. Robertha는 강건성을 높이는 대신 GLUE clean 성능을 일부 희생한다. SuperGLUE에서는 clean 성능 저하가 0.4점에 불과하지만, GLUE에서는 약 2.8점의 손실이 발생한다.

따라서 Robertha가 모든 조건에서 우수하다고 해석하기는 어렵다. 잡음이 존재하는 환경에서는 상당한 이점이 있지만, clean accuracy만 중요하다면 기존 TinyBERT가 더 좋은 선택일 수 있다.

⸻

5.2 Ablation 결과

논문은 iterative retrieval과 ESR이 각각 성능에 미치는 영향을 분석한다.

모델 변형	GLUE Clean	GLUE Extreme	SuperGLUE Clean	SuperGLUE Extreme
Single-pass Hopfield, ESR 없음	62.2 ± 0.2	49.2 ± 1.2	59.8 ± 0.7	52.1 ± 1.5
Iterative Hopfield, ESR 없음	61.8 ± 0.3	52.8 ± 1.2	60.0 ± 0.5	51.8 ± 2.4
ESR 대신 Dropout	61.6 ± 0.7	53.6 ± 1.0	60.6 ± 0.5	52.4 ± 1.5
ESR 대신 Weight Decay	61.3 ± 0.8	50.9 ± 0.8	59.0 ± 1.0	53.0 ± 1.4
전체 Robertha	61.9 ± 0.4	54.0 ± 0.5	60.1 ± 0.5	54.4 ± 1.4

Iteration만 적용해도 GLUE에서는 어느 정도 강건성이 증가한다. 그러나 SuperGLUE에서는 iteration만으로 일관된 개선이 나타나지 않는다. ESR을 Dropout이나 Weight Decay로 대체한 경우에도 한쪽 벤치마크에서는 개선되지만 다른 쪽에서는 약한 결과가 나타난다.

전체 Robertha는 GLUE와 SuperGLUE의 extreme corruption 모두에서 가장 높은 성능을 기록한다. 이는 ESR이 단순한 일반 정규화와는 다른 역할을 한다는 저자들의 주장을 일정 부분 뒷받침한다.

다만 GLUE 결과만 보면 iteration이나 Dropout으로도 상당 부분의 개선이 발생한다. 따라서 ESR의 가장 강한 증거는 “모든 성능 향상의 유일한 원인”이라기보다, 서로 다른 유형의 태스크에서 성능을 일관되게 유지한다는 점에 있다.

⸻

5.3 추가 강건성 평가

논문은 AdvGLUE, PAWS, TextBugger, HotFlip에 대한 추가 실험도 수행한다. 저자들에 따르면 Robertha는 AdvGLUE와 PAWS의 9개 태스크 중 8개에서 비교 모델보다 높은 성능을 보이며, character-level corruption에서도 경쟁력 있는 성능을 나타낸다.

또한 corruption이 증가할수록 attention key representation의 effective rank가 증가하고 성능이 감소하는 현상 사이에 강한 상관관계가 보고된다. 저자들은 이를 ESR이 eigenspectrum을 집중시키고 attractor를 안정화한다는 설명의 근거로 사용한다.

⸻

5.4 대규모 모델 및 계산 비용

논문은 약 110M 파라미터 규모의 Robertha-large를 BERT-base와 비교한다. High corruption에서의 결과는 다음과 같다.

모델	CoLA	MRPC	WNLI	BoolQ
BERT-base	50.6	29.3	54.9	53.3
Robertha-large	54.9	74.8	63.4	62.2
성능 차이	+4.3	+45.5	+8.5	+8.9

따라서 저자들은 Robertha의 강건성 효과가 TinyBERT 수준의 소형 모델에만 국한되지 않는다고 주장한다.

하지만 반복 retrieval로 인해 추론 비용이 증가한다. 논문에서 보고된 latency는 corruption 수준과 iteration 수에 따라 기존 모델 대비 약 1.25배에서 3.54배까지 증가한다.

⸻

5.5 종합 평가

실험 결과가 직접적으로 뒷받침하는 결론은 다음과 같다.

Robertha는 additive Gaussian embedding corruption이라는 설정에서 TinyBERT, FreeLB 및 noise-based data augmentation보다 상당히 높은 강건성을 보인다.

그러나 이를 일반적인 NLP robustness 전체로 확대하기는 어렵다. 실험의 핵심 증거는 encoder 기반 분류 모델과 embedding-space Gaussian noise에 집중되어 있다. Decoder-only LLM, 실제 quantization error, prompt 공격, 자연스러운 텍스트 교란에 대해서까지 우수하다고 결론 내릴 수 있는 결과는 아니다.



<br/>
# 예제




The examples below are simplified illustrations. The exact probabilities are not reported in the paper.

1. Training Example

Input

Sentence: The service was not satisfactory.
Label: Negative

After tokenization:

[CLS], The, service, was, not, satisfactory, ., [SEP]

The model predicts:

P(Negative) = 0.89
P(Positive) = 0.11
Prediction  = Negative

Robertha is trained using the task loss together with Eigenspectrum Regularization:

\mathcal{L}
=
\mathcal{L}_{task}
+
\lambda\mathcal{L}_{ESR}.

⸻

2. Clean Test Example

Input

The restaurant was not expensive.

Expected output

Label: Positive

A clean prediction may be:

P(Negative) = 0.18
P(Positive) = 0.82
Prediction  = Positive

⸻

3. Corrupted Test Example

The visible sentence remains unchanged:

The restaurant was not expensive.

However, Gaussian noise is added to the internal token embeddings:

\tilde{\mathbf e}_t
=
\mathbf e_t+\boldsymbol\epsilon_t,
\qquad
\boldsymbol\epsilon_t\sim\mathcal N(0,\sigma^2I).

A standard model may lose the effect of the word not:

TinyBERT:
P(Negative) = 0.64
P(Positive) = 0.36
Prediction  = Negative

Robertha iteratively refines the corrupted representation:

Robertha:
P(Negative) = 0.24
P(Positive) = 0.76
Prediction  = Positive

⸻

4. Input–Output Pipeline

Sentence
   ↓
Tokenization
   ↓
Token embeddings
   ↓
Gaussian embedding corruption at test time
   ↓
Iterative Hopfield retrieval + ESR
   ↓
Classification probabilities
   ↓
Predicted label

The key point is that the attack does not visibly modify the sentence. It changes the model’s internal embedding vectors, while Robertha attempts to recover a stable semantic representation before producing the final prediction.


이 논문의 Gaussian 공격은 문장 자체를 바꾸는 공격이 아니라, 토큰화 이후의 내부 임베딩을 바꾸는 공격이야. 따라서 사용자에게 보이는 테스트 문장은 동일하고, 모델 내부 입력만 달라진다.

아래 수치와 개별 예측은 논문에 그대로 수록된 사례가 아니라, 학습·추론 과정을 이해하기 위한 재구성 예시:

⸻

1. 전체 입출력 구조

예를 들어 SST-2 감성 분류에서는 다음과 같다.

외부에서 보이는 입력

Sentence: The movie was not interesting.

정답 출력

Label: Negative

토큰화하면:

[CLS], The, movie, was, not, interesting, ., [SEP]

각 토큰은 임베딩 벡터로 변환된다.

X=[e_{\text{[CLS]}},e_{\text{The}},e_{\text{movie}},\ldots,e_{\text{not}},e_{\text{interesting}}]

테스트 시 Gaussian corruption을 적용하면:

\tilde e_t=e_t+\epsilon_t,
\qquad
\epsilon_t\sim\mathcal N(0,\sigma^2I)

즉 문장은 여전히 The movie was not interesting.이지만, 모델이 받는 내부 벡터는 달라진다.

⸻

2. Training 데이터 예시

예시 A: SST-2 감성 분류

학습 샘플 1

입력

The movie was surprisingly enjoyable.

정답

Positive

모델 출력 형식

P(Negative) = 0.08
P(Positive) = 0.92
Prediction  = Positive

학습 샘플 2

입력

The movie was not enjoyable.

정답

Negative

모델 출력 형식

P(Negative) = 0.88
P(Positive) = 0.12
Prediction  = Negative

학습 손실은 일반적으로 정답 라벨에 대한 cross-entropy와 ESR 정규화 항을 결합한다.

\mathcal L
=
\mathcal L_{\text{classification}}
+
\lambda\mathcal L_{\text{ESR}}

Robertha의 핵심은 학습 문장의 라벨 형식이 특별한 것이 아니라, 모델 내부 attention 구조와 정규화 방식이 다르다는 점이다.

⸻

예시 B: MRPC 문장쌍 분류

MRPC는 두 문장이 같은 의미인지 판별한다.

학습 샘플

입력

Sentence 1: The company reported a decline in revenue.
Sentence 2: The firm's revenue decreased.

정답

Paraphrase

출력

P(Not paraphrase) = 0.09
P(Paraphrase)     = 0.91
Prediction        = Paraphrase

두 문장이 결합되어 입력된다.

[CLS] The company reported ... [SEP]
The firm's revenue decreased . [SEP]

⸻

예시 C: BoolQ 질의응답 분류

BoolQ는 passage와 질문을 읽고 Yes 또는 No를 예측한다.

학습 샘플

입력

Passage:
Penguins are birds, but they cannot fly. They use their wings
to swim through the water.
Question:
Can penguins fly?

정답

No

출력

P(No)  = 0.96
P(Yes) = 0.04
Prediction = No

여기서는 cannot이 정답을 결정하는 핵심 표현이다.

⸻

3. Clean test 예시

학습에 사용되지 않은 테스트 문장을 넣는다고 하자.

SST-2 테스트 샘플

입력

The plot was not particularly original, but the acting was excellent.

정답

Positive

문장에는 부정 표현이 있지만, 전체 평가는 긍정적이다.

TinyBERT의 clean 예측 예시

P(Negative) = 0.22
P(Positive) = 0.78
Prediction  = Positive

Robertha의 clean 예측 예시

P(Negative) = 0.27
P(Positive) = 0.73
Prediction  = Positive

두 모델 모두 정답을 맞힌다. Robertha는 clean 환경에서 항상 더 높은 confidence를 갖는 것이 아니며, 논문 결과에서도 GLUE clean 성능은 TinyBERT보다 다소 낮다.

⸻

4. Corrupted test 예시

같은 문장에 embedding noise를 추가한다.

외부 입력

The plot was not particularly original, but the acting was excellent.

외부 문장은 전혀 바뀌지 않는다.

내부 임베딩

Clean 상태:

e_{\text{not}}
=
[0.12,-0.07,0.03,\ldots]

Corrupted 상태:

\tilde e_{\text{not}}
=
e_{\text{not}}+\epsilon
=
[0.74,-0.51,0.39,\ldots]

숫자는 설명용 예시다.

TinyBERT의 corrupted 예측 예시

P(Negative) = 0.58
P(Positive) = 0.42
Prediction  = Negative

잡음 때문에 but, not, excellent 사이의 관계를 제대로 처리하지 못해 예측이 뒤집힌 상황이다.

Robertha의 corrupted 예측 예시

Hopfield retrieval step 1:
P(Negative) = 0.46
P(Positive) = 0.54
Hopfield retrieval step 2:
P(Negative) = 0.31
P(Positive) = 0.69
Final prediction:
Positive

Robertha는 반복 retrieval을 통해 손상된 표현을 기존에 학습된 의미 패턴에 가까운 방향으로 이동시킨다고 설명한다.

⸻

5. 논문의 not 사례를 입출력으로 표현하면

원래 문장

Who can help?

의미:

도울 수 있는 사람을 묻는 질문

부정 문장

Who cannot help?

의미:

도울 수 없는 사람을 묻는 질문

실제 공격에서는 not을 삭제하지 않는다.

테스트 입력

Who cannot help?

정답 의미

cannot-help

Clean 모델 출력 예시

P(can-help)     = 0.06
P(cannot-help)  = 0.94
Prediction      = cannot-help

not 임베딩이 손상된 baseline 출력 예시

P(can-help)     = 0.71
P(cannot-help)  = 0.29
Prediction      = can-help

즉 모델이 실제로 본 텍스트는 여전히 Who cannot help?지만, 내부적으로는 not의 의미 기능을 충분히 사용하지 못해 마치 Who can help?를 처리한 것처럼 행동한다.

Robertha 출력 예시

Initial corrupted representation:
P(can-help)     = 0.55
P(cannot-help)  = 0.45
After iterative retrieval:
P(can-help)     = 0.18
P(cannot-help)  = 0.82
Prediction      = cannot-help

⸻

6. Training 방식별 차이

동일한 학습 샘플을 사용하더라도 비교 모델마다 처리 방식이 다르다.

모델	학습 입력	추가 처리	학습 목표
TinyBERT	Clean 문장	없음	Task loss
Data Augmentation	Clean 문장	임베딩에 무작위 noise 추가	Task loss
FreeLB	Clean 문장	worst-case perturbation을 반복 계산	Adversarial task loss
Robertha	Clean 문장	Hopfield attention 및 ESR 적용	Task loss + ESR

예를 들어 다음 학습 문장이 있다고 하자.

The service was not satisfactory.
Label: Negative

TinyBERT

텍스트 → 임베딩 → Transformer → Negative

Data Augmentation

텍스트 → 임베딩 + 무작위 noise → Transformer → Negative

FreeLB

텍스트 → 임베딩
      → 모델 손실을 크게 만드는 방향의 perturbation 탐색
      → perturbed embedding으로 학습
      → Negative

Robertha

텍스트 → 임베딩
      → iterative Hopfield attention
      → ESR이 적용된 representation
      → Negative

Robertha는 기본 설정에서 매 학습 샘플에 공격을 생성해 학습하는 방식이라기보다, 표현 공간 자체가 손상된 표현을 안정적인 상태로 복원하도록 구조를 변경하는 방법으로 이해하는 것이 적절하다.

⸻

7. Test 데이터 처리 전체 예시

[Test sentence]
The restaurant was not expensive.
        ↓ Tokenization
[CLS], The, restaurant, was, not, expensive, ., [SEP]
        ↓ Embedding
e_CLS, e_The, e_restaurant, e_was, e_not, ...
        ↓ Test-time corruption
e_t + ε_t,    ε_t ~ N(0, σ²I)
        ↓ Model
TinyBERT:
standard self-attention
→ 예측이 쉽게 불안정해질 수 있음
Robertha:
iterative Hopfield retrieval
+ eigenspectrum-regularized attention
→ 손상된 표현 복원 시도
        ↓ Output
Label: Not expensive / Affordable
또는 태스크에 따라 Positive / Negative

핵심은 다음과 같다.

Training 데이터와 test 데이터는 일반적인 문장·라벨 형태이고, 공격은 test 문장을 눈에 보이게 수정하는 것이 아니라 모델 내부의 token embedding을 수정한다.
Robertha의 출력 형식도 일반 분류 모델과 동일하며, 차이는 출력 이전의 표현 복원 과정에 있다.


<br/>
# 요약





Robertha는 Modern Hopfield Attention과 Eigenspectrum Regularization을 결합해, 노이즈로 손상된 토큰 임베딩을 안정적인 의미 표현으로 복원하는 NLU 모델이다.
GLUE와 SuperGLUE 실험에서 TinyBERT, FreeLB, 데이터 증강보다 임베딩 노이즈 환경에서 높은 강건성을 보였다.
다만 깨끗한 데이터에서의 일부 성능 저하와 반복 검색에 따른 추론 비용 증가는 주요 한계로 남는다.




Robertha is a robust NLU model that combines Modern Hopfield Attention with Eigenspectrum Regularization to recover stable semantic representations from corrupted token embeddings.
Across GLUE and SuperGLUE, it outperforms TinyBERT, FreeLB, and data augmentation under embedding noise.
However, its improved robustness comes with modest clean-performance degradation and increased inference cost from iterative retrieval.


<br/>
# 기타











Figure 1. Robertha Architecture

Description

* TinyBERT encoder
* Iterative Modern Hopfield attention
* Eigenspectrum Regularization (ESR)

Key takeaway

The model progressively restores corrupted semantic representations through iterative retrieval.

⸻

Figure 2. Embedding Corruption

Description

* Gaussian noise is added to token embeddings.
* Low-norm embeddings receive proportionally larger perturbations.

Key takeaway

Small embedding vectors are more vulnerable to corruption.

⸻

Figure 3. Eigenspectrum Analysis

Description

* Comparison of eigenvalue distributions
* With and without ESR

Key takeaway

ESR produces a more compact eigenspectrum and more stable semantic attractors.

⸻

Table 1. Main Results

Description

* GLUE
* SuperGLUE
* Multiple corruption levels

Key takeaway

Robertha consistently achieves higher robustness under embedding corruption.

⸻

Table 2. Ablation Study

Description

* Remove iterative retrieval
* Remove ESR
* Replace ESR with Dropout
* Replace ESR with Weight Decay

Key takeaway

Both iterative retrieval and ESR contribute to the final performance.

⸻

Table 3. Large-scale Evaluation

Description

* Evaluation on BERT-base (110M)

Key takeaway

The robustness improvement also generalizes to a larger encoder model.

⸻

Appendix

A. Additional Benchmarks

* AdvGLUE
* PAWS
* TextBugger
* HotFlip

→ Evaluates robustness under additional adversarial settings.

⸻

B. Hyperparameters

* Learning rate
* Batch size
* Epochs
* Noise level
* Number of retrieval iterations

→ Provides full experimental settings for reproducibility.

⸻

C. Additional Ablation

* Retrieval iteration analysis
* ESR coefficient analysis

→ Examines the contribution of each design component.

⸻

D. Efficiency

* Number of parameters
* FLOPs
* Inference latency

→ Robustness improves at the cost of approximately 1.25×–3.54× higher inference latency.

⸻


Figure 1. Robertha Architecture

내용

* TinyBERT encoder 위에 Modern Hopfield Attention을 적용
* Iterative retrieval을 반복 수행
* Attention key에 ESR(Eigenspectrum Regularization) 적용

핵심

손상된 embedding을 semantic attractor 방향으로 반복적으로 복원하는 전체 구조를 보여준다.

⸻

Figure 2. Embedding Corruption

내용

* 입력 embedding에 Gaussian noise 추가
* Low-norm embedding이 더 크게 영향을 받음

핵심

동일한 noise라도 작은 embedding일수록 의미 손상이 심해진다는 점을 설명한다.

⸻

Figure 3. Eigenspectrum Analysis

내용

* ESR 적용 전후 eigenvalue 분포 비교
* ESR 적용 시 spectrum이 더 compact해짐

핵심

Representation이 안정적인 attractor basin을 형성함을 시각적으로 보여준다.

⸻

Table 1. Main Results

내용

* GLUE
* SuperGLUE
* 다양한 noise level 비교

핵심

대부분 corruption 환경에서 Robertha가 가장 높은 robustness를 기록하였다.

⸻

Table 2. Ablation Study

내용

* Iteration 제거
* ESR 제거
* Dropout 대체
* Weight Decay 대체

핵심

Iteration과 ESR이 모두 성능 향상에 기여하며, 전체 모델이 가장 좋은 결과를 보인다.

⸻

Table 3. Large-scale Evaluation

내용

* TinyBERT뿐 아니라
* BERT-base(110M)에서도 평가

핵심

큰 encoder에서도 robustness 향상이 유지됨을 보여준다.

⸻

Appendix

A. Additional Benchmarks

* AdvGLUE
* PAWS
* TextBugger
* HotFlip

→ 추가 공격 환경에서도 성능을 평가한다.

⸻

B. Hyperparameters

* learning rate
* batch size
* epoch
* noise level
* iteration 수

→ 모든 실험 설정을 공개하여 재현성을 제공한다.

⸻

C. Additional Ablation

* retrieval iteration 변화
* ESR coefficient 변화

→ 각 구성요소의 영향을 추가 분석한다.

⸻

D. Efficiency

* Parameter 수
* FLOPs
* Inference latency

→ Robustness는 증가하지만 추론 시간이 약 1.25–3.54배 증가한다.

⸻


<br/>
# refer format:


@inproceedings{podasca-das-2026-robertha,
  title     = {Robertha: Eigenspectrum Regularized Attention for Robust Natural Language Understanding},
  author    = {Podasca, Andreia and Das, Anup},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  editor    = {Liakata, Maria and Moreira, Viviane P. and Zhang, Jiajun and Jurgens, David},
  year      = {2026},
  month     = jul,
  address   = {San Diego, California, United States},
  publisher = {Association for Computational Linguistics},
  pages     = {15229--15264},
  url       = {https://aclanthology.org/2026.acl-long.696/},
  isbn      = {979-8-89176-390-6}
}




Podasca, Andreia, and Anup Das. “Robertha: Eigenspectrum Regularized Attention for Robust Natural Language Understanding.” In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics, Volume 1: Long Papers, edited by Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and David Jurgens, 15229–15264. San Diego, CA: Association for Computational Linguistics, 2026.