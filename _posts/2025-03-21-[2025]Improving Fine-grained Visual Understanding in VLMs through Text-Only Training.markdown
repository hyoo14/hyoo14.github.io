---
layout: post
title:  "[2025]Improving Fine-grained Visual Understanding in VLMs through Text-Only Training"  
date:   2025-03-21 21:20:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 


vlm을 언어로만 추가 학습  


짧은 요약(Abstract) :    




⸻



이 논문은 **시각-언어 모델(VLMs)**의 정밀한 시각적 이해 능력을 **이미지 없이, 텍스트만으로 학습하는 방식(text-only training)**으로 향상시킬 수 있는지를 탐구합니다. 전통적인 VLM 학습은 이미지와 텍스트의 쌍을 수집하고 처리하는 데 많은 자원이 필요하지만, 이 논문에서는 언어만으로도 충분히 시각 개념을 학습할 수 있다는 가설을 세웁니다.

이들은 인간이 시각 개념을 언어 설명을 통해 학습하는 방식에서 영감을 받아, 텍스트 기반의 학습이 시각 인식 능력 향상에 도움이 된다고 주장합니다. 나비 종 분류와 한국 문화 이해라는 두 가지 도메인에서 실험을 진행했으며, 그 결과 텍스트만으로 학습한 모델도 기존 방식과 비슷하거나 더 나은 성능을 보이면서, 계산 자원도 훨씬 적게 소모함을 확인했습니다. 이 방식은 특히 자원이 제한된 환경에서 매우 유용한 대안이 될 수 있다고 강조합니다.

⸻



Visual-Language Models (VLMs) have become a powerful tool for bridging the gap between visual and linguistic understanding. However, the conventional learning approaches for VLMs often suffer from limitations, such as the high resource requirements of collecting and training image-text paired data. Recent research has suggested that language understanding plays a crucial role in the performance of VLMs, potentially indicating that text-only training could be a viable approach. In this work, we investigate the feasibility of enhancing fine-grained visual understanding in VLMs through text-only training. Inspired by how humans develop visual concept understanding, where rich textual descriptions can guide visual recognition, we hypothesize that VLMs can also benefit from leveraging text-based representations to improve their visual recognition abilities. We conduct comprehensive experiments on two distinct domains: fine-grained species classification and cultural visual understanding tasks. Our findings demonstrate that text-only training can be comparable to conventional image-text training while significantly reducing computational costs. This suggests a more efficient and cost-effective pathway for advancing VLM capabilities, particularly valuable in resource-constrained environments.

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




⸻



1. 제안 방식 개요
이 논문은 Vision-Language Model(VLM)을 **이미지 없이 텍스트만으로 학습(text-only training)**해서도 정밀한 시각 인식을 수행할 수 있는지를 실험합니다. 특히 사람의 시각 개념 학습이 언어를 통해 이루어지는 점에서 착안해, 모델도 텍스트만으로 시각적 개념을 학습할 수 있다고 가정합니다.

⸻

2. 아키텍처 (사용한 모델 구조)
실험에는 7B 규모의 오픈소스 VLM 두 가지가 사용되었습니다:
	•	LLaVA-1.6-7B: CLIP 기반의 비전 인코더와 Vicuna 기반의 언어 모델로 구성되며, 인스트럭션 튜닝을 통해 시각적 추론 능력을 강화함.
	•	Qwen2-VL-7B: Qwen2 언어 모델에 비전 인식 기능을 추가한 구조로, Vision Transformer와 projection layer를 통해 이미지-언어 정렬을 수행함.

두 모델 모두 비전 인코더 + 프로젝션 레이어 + 언어 모델의 구조를 따르며, 실험에서는 **비전 인코더는 고정(freeze)**하고 언어 모델과 프로젝터만 fine-tuning 하였습니다.

⸻

3. 트레이닝 데이터 구성 방식
두 가지 도메인에서 데이터를 구성하였습니다:
	•	나비 종 분류 (BUTTERFLY)
각 종마다 하나의 이미지를 선택하고, 이를 기반으로 GPT-4o를 이용해 날개 색, 형태, 생태 특성 등 시각적·생물학적 특징을 자세히 묘사한 텍스트를 생성함.
	•	한국 문화 시각 이해 (K-VISCUIT)
주어진 키워드마다 인터넷에서 이미지를 수집하고, GPT-4o를 활용해 시각적 특징, 역사적 의미, 지역적 차이 등을 포함한 텍스트 설명을 생성함.

각 도메인에 대해 두 가지 버전의 학습 데이터셋을 만듦:
(1) 이미지 + 텍스트 버전,
(2) 텍스트만 포함된 버전.

샘플 수는 BUTTERFLY는 100개, K-VISCUIT은 237개로, 비교 실험에 적합하도록 설계되었습니다.

⸻



1. Overview
This paper explores the feasibility of training Vision-Language Models (VLMs) using text-only training to enhance fine-grained visual understanding. Inspired by how humans learn visual concepts through language, the authors hypothesize that VLMs can similarly benefit from rich textual descriptions.

⸻

2. Architecture
Two open-source 7B-scale VLMs were used:
	•	LLaVA-1.6-7B: Combines a CLIP-based vision encoder and a Vicuna language model, enhanced with instruction tuning for visual reasoning.
	•	Qwen2-VL-7B: Extends the Qwen2 language model with a Vision Transformer and a projection layer to align visual and language features.

Both models follow a modular architecture: vision encoder + projector + language model. In fine-tuning, the vision encoder is frozen, and only the language model and projection layers are updated.

⸻

3. Training Data
Training datasets were built for two domains:
	•	Butterfly species classification (BUTTERFLY)
For each species, one image was selected, and GPT-4o was used to generate detailed textual descriptions covering visual traits (e.g., wing patterns, color) and biological aspects (e.g., habitat, behavior).
	•	Korean cultural visual understanding (K-VISCUIT)
One image per keyword was collected from the internet, and GPT-4o generated rich descriptions including visual, historical, and cultural elements.

Each domain has two dataset versions:
(1) image-text pairs, and
(2) text-only descriptions.

The number of training samples is 100 for BUTTERFLY and 237 for K-VISCUIT, allowing direct comparison between training methods.

⸻






       
 
<br/>
# Results  



⸻


1. 테스크(Task)

두 가지 테스크로 모델 성능을 평가했습니다:
	•	Type 1: 시각 인식(Visual Recognition)
이미지를 보고 객체(예: 나비 종, 문화유산 이름 등)를 맞추는 다지선다형 문제
	•	Type 2: 복합 추론(Complex Reasoning)
이미지에 기반한 생태/문화적 정보를 추론하는 고차원 질문
(예: 나비의 번식 시기, 유물의 역사적 용도 등)

⸻

2. 테스트 데이터(Test Dataset)
	•	BUTTERFLY
나비 100종에 대한 질문 400개 (Type 1: 200개 / Type 2: 200개)
	•	K-VISCUIT
한국 문화 시각 이해 데이터셋
Type 1: 237개, Type 2: 420개 총 657개 문제

⸻

3. 평가 메트릭(Metric)
	•	정확도(Accuracy):
각 유형별 문제에 대해 정답을 맞춘 비율로 평가

⸻

4. 결과 및 비교(Results & Comparison)
	•	텍스트만으로 훈련한 모델도 기존 이미지+텍스트 학습과 비슷하거나 더 좋은 성능을 보임
	•	특히 LLaVA-1.6-7B는 낮은 초기 성능에도 불구하고 텍스트-only 방식으로 +5%p 이상 성능 향상
	•	Qwen2-VL-7B는 본래 성능이 높지만, 여전히 text-only 훈련에서 더 좋은 결과를 보이기도 함

예시:

모델	데이터셋	Type 1	Type 2	전체 성능
Qwen2-VL-7B (텍스트-only)	K-VISCUIT	74.26%	69.76%	71.39%
LLaVA-1.6-7B (텍스트-only)	BUTTERFLY	30.5%	54.5%	42.5%

	•	이미지 없이 테스트해도 성능이 떨어지는 것으로 보아, 단순한 암기 효과가 아님이 확인됨
	•	리소스 측면에서도 효율적임: 학습 시간, GPU 메모리 사용, 에너지 소비 모두 크게 절감

⸻


1. Tasks

Two types of evaluation tasks were used:
	•	Type 1: Visual Recognition
Multiple-choice questions requiring object recognition from images (e.g., butterfly species, cultural items)
	•	Type 2: Complex Reasoning
Higher-order multiple-choice questions involving ecological or cultural understanding (e.g., breeding cycles, historical use)

⸻

2. Test Datasets
	•	BUTTERFLY:
400 total questions (Type 1: 200, Type 2: 200)
	•	K-VISCUIT:
657 total questions (Type 1: 237, Type 2: 420)

⸻

3. Metrics
	•	Accuracy was used as the evaluation metric—percentage of correct answers.

⸻

4. Results & Comparison
	•	Text-only training achieved comparable or even superior performance to image-text training.
	•	For LLaVA-1.6-7B, text-only training led to over +5% improvement from the baseline.
	•	For Qwen2-VL-7B, which had higher initial accuracy, text-only training still performed slightly better in many cases.

Example summary:

Model	Dataset	Type 1	Type 2	Overall Accuracy
Qwen2-VL-7B (Text-only)	K-VISCUIT	74.26%	69.76%	71.39%
LLaVA-1.6-7B (Text-only)	BUTTERFLY	30.5%	54.5%	42.5%

	•	Ablation test (evaluating without image input) showed significant performance drop, confirming the gains are not due to memorization.
	•	Resource usage (training time, GPU memory, energy) was also substantially lower in text-only training, showing greater efficiency and sustainability.

⸻





















<br/>
# 예제  




⸻



1. 트레이닝 데이터 예시

논문에서는 이미지 없이도 시각적 개념을 학습할 수 있도록, GPT-4o를 사용해 시각적 세부묘사와 배경지식을 포함한 텍스트를 생성했습니다. 이미지가 포함된 경우와 이미지 없이 텍스트만 있는 경우 두 가지 버전으로 구성했습니다.

(1) 나비 분류(BUTTERFLY) 텍스트-only 예시

텍스트 설명 예시:

이 나비는 선명하고 생기 있는 파란색 날개를 가지고 있으며, 빛을 받을 때 은은하게 반짝이는 광택이 돋보입니다. 수컷의 앞날개는 하늘색에 검은색 테두리가 선명하게 둘러져 있고, 뒷날개에는 검은 점과 흰색 윤곽선이 있습니다. 주요 서식지는 온난한 초원과 들판이며, 봄철에 주로 번식하고, 애벌레는 특정 식물의 잎을 먹습니다.

(2) 한국 문화 유산(K-VISCUIT) 텍스트-only 예시

‘갓’에 대한 설명:

갓은 조선 시대 양반 남성들이 착용하던 전통 모자로, 상징성과 기능성을 동시에 지녔습니다. 통정(갓의 몸통)은 말총이나 대나무로 만들어지며, 양태(챙)는 얇게 쪼갠 대나무에 흑칠과 비단을 덧입혀 만듭니다. 흑립, 백립, 방립 등 다양한 종류가 있으며, 착용자의 신분이나 상황에 따라 달라졌습니다. 현재는 전통 행사나 공연에서 주로 착용되며, 현대 패션에도 영감을 주고 있습니다.



⸻

2. 테스트 데이터 예시

(1) 나비 테스트 Type 1 (시각 인식 문제)

질문: 이 이미지에 나타난 나비의 종 이름은 무엇인가요?
보기:
A) ADONIS
B) COMMON BLUE
C) MONARCH
D) PALE CLOUDED YELLOW
정답: A) ADONIS

(2) 나비 테스트 Type 2 (생태적/생물학적 추론 문제)

질문: 이 나비의 번식 시기와 그 시기에 영향을 주는 요소는 무엇인가요?
보기:
A) 늦봄, 온도와 숙주 식물에 의해 영향
B) 여름, 일조량에 따라 결정됨
C) 가을, 지역 강수량이 결정 요인
D) 겨울, 먹이의 유무에 따라 조절
정답: A

⸻


1. Training Data Examples

To simulate visual understanding without actual images, the authors generated rich textual descriptions using GPT-4o, focusing on visual features, behavior, and context. There are two versions: with image-text pairs and text-only.

(1) Butterfly Dataset (Text-only training example)

Example Description:

This butterfly has brilliant, vibrant blue wings with a subtle iridescent shimmer that catches the light beautifully. The male's upper wings are a striking sky blue with thin, jet-black borders, and the hindwings have delicate black spots outlined in white. It typically inhabits warm grasslands and fields. It breeds in late spring, and the caterpillars feed on specific host plant leaves.

(2) Korean Cultural Dataset (Text-only training example)

Example on ‘Gat’ (traditional Korean hat):

The gat is a traditional Korean hat worn by noblemen during the Joseon Dynasty. The cylindrical crown (tongjeong) is made from horsehair or bamboo, while the brim (yangtae) is crafted from thin bamboo, coated with lacquer and black silk. Variations include heungnip, baeknip, and bangnip, used based on social status or occasion. Though no longer used daily, it appears in cultural ceremonies and influences modern fashion.



⸻

2. Test Data Examples

(1) Butterfly Dataset - Type 1 (Visual Recognition)

Question: What is the name of the butterfly species shown in this image?
Options:
A) ADONIS
B) COMMON BLUE
C) MONARCH
D) PALE CLOUDED YELLOW
Answer: A) ADONIS

(2) Butterfly Dataset - Type 2 (Ecological/Biological Reasoning)

Question: What is the typical breeding season for this butterfly, and what affects its timing?
Options:
A) Late spring, affected by temperature and host plant availability
B) Summer, determined by sunlight duration
C) Fall, depending on regional rainfall
D) Winter, influenced by food presence
Answer: A

⸻





<br/>  
# 요약   



⸻



이 논문은 이미지 없이 텍스트만으로 Vision-Language Model(VLM)을 학습시켜도 시각적 개념을 효과적으로 이해할 수 있는지를 실험했다. 나비 분류와 한국 문화 인식 테스크에서 텍스트-only 학습은 기존 이미지-텍스트 학습과 유사하거나 더 나은 정확도를 보였으며, 학습 시간과 자원도 절감되었다. GPT-4o로 생성된 시각적 설명문이 학습과 평가에 활용되었고, 테스트 질문은 시각 인식과 생태적 추론을 포함한다.

⸻



This study investigates whether Vision-Language Models (VLMs) can effectively learn visual concepts through text-only training without image inputs. In tasks like butterfly classification and Korean cultural understanding, text-only models achieved comparable or even superior accuracy while reducing training time and computational cost. Visual descriptions generated by GPT-4o were used for training, and the evaluation included both recognition and ecological reasoning questions.

⸻




<br/>  
# 기타  



⸻



Figure 1
	•	텍스트-only 학습과 이미지 기반 학습을 비교하는 개념도.
	•	사람이 언어를 통해 시각 개념을 학습하는 과정을 비유적으로 보여줌.

Table 1
	•	각 모델(Qwen2-VL-7B, LLaVA-1.6-7B)의 성능을 Type 1(시각 인식)과 Type 2(복합 추론) 기준으로 정리.
	•	텍스트-only 학습이 전반적으로 기존 학습 방식보다 성능이 높거나 유사함을 수치로 보여줌.

Table 2
	•	이미지 입력 유무에 따른 성능 차이 비교.
	•	이미지 없이 테스트했을 때 성능이 눈에 띄게 하락해, 단순 텍스트 암기가 아니라는 점을 입증.

Figure 2
	•	이미지+텍스트 학습과 텍스트-only 학습의 리소스 소비 비교.
	•	학습 시간, GPU 메모리 사용량, 에너지 소비 모두 텍스트-only가 더 효율적임을 시각적으로 표현.

Appendix A: 모델 세부 구조 및 설정
	•	LLaVA와 Qwen2-VL의 구조, 파라미터 구성, 학습 방식 등의 세부사항 제공.

Appendix B: 데이터 생성 예시 및 프롬프트
	•	나비와 한국 문화 데이터셋의 텍스트 설명을 생성한 프롬프트 및 생성 결과 예시 포함.

Appendix C: 향후 연구 방향
	•	대형 모델과 다양한 데이터셋 확장, 하이브리드 학습 접근법 등 미래 연구 방향 제안.

⸻


Figure 1
	•	A conceptual diagram comparing text-only training with image-based training.
	•	Illustrates how humans learn visual concepts through language, serving as an analogy.

Table 1
	•	Presents performance scores for Qwen2-VL-7B and LLaVA-1.6-7B across Type 1 (recognition) and Type 2 (reasoning) tasks.
	•	Shows that text-only training performs similarly or even better than traditional image-text training.

Table 2
	•	Compares performance with and without image input.
	•	Accuracy drops significantly without images, indicating that models did not rely on text memorization.

Figure 2
	•	Visualizes resource efficiency between image-text and text-only training.
	•	Text-only training reduces training time, GPU memory usage, and energy consumption.

Appendix A: Model Architecture and Training Settings
	•	Details the structures and configurations of LLaVA and Qwen2-VL, including frozen components and fine-tuning settings.

Appendix B: Data Generation Prompts and Examples
	•	Provides GPT-4o prompts and example outputs used to generate training data for butterfly and cultural datasets.

Appendix C: Future Work
	•	Suggests directions such as scaling to larger models and datasets, hybrid training methods, and real-world applications.

⸻





<br/>
# refer format:     


@inproceedings{choi2025textonly,
  title     = {Improving Fine-grained Visual Understanding in VLMs through Text-Only Training},
  author    = {Choi, Dasol and Son, Guijin and Kim, Soo Yong and Paik, Gio and Hong, Seunghyeok},
  booktitle = {Proceedings of the AAAI 2025 Workshop on Vision-Language Models},
  year      = {2025},
  organization = {Association for the Advancement of Artificial Intelligence},
  address   = {Palo Alto, CA},
  note      = {To appear}
}


Choi, Dasol, Guijin Son, Soo Yong Kim, Gio Paik, and Seunghyeok Hong. “Improving Fine-grained Visual Understanding in VLMs through Text-Only Training.” Paper presented at the AAAI 2025 Workshop on Vision-Language Models, Palo Alto, CA. To appear.



