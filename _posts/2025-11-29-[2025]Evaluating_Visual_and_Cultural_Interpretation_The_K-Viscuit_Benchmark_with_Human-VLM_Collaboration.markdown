---
layout: post
title:  "[2025]Evaluating Visual and Cultural Interpretation: The K-Viscuit Benchmark with Human-VLM Collaboration"
date:   2025-11-29 00:33:18 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 한국 문화를 중심으로 한 비전-언어 모델(VLM)의 문화적 이해를 평가하기 위한 K-Viscuit 벤치마크를 제안   


짧은 요약(Abstract) :


이 논문에서는 문화적으로 포괄적인 비전-언어 모델(VLM)을 개발하기 위해, 문화적으로 관련된 질문을 평가하는 벤치마크의 필요성을 강조합니다. 기존의 접근 방식은 주로 인간 주석자에게 의존하여 노동 집약적이며 다양한 질문을 생성하는 데 인지적 부담을 초래합니다. 이를 해결하기 위해, 저자들은 인간-VLM 협업을 통해 질문을 생성하고 원주율을 검증하는 반자동 프레임워크를 제안합니다. 이 프레임워크를 통해 한국 문화를 중심으로 한 K-Viscuit 데이터셋을 생성하였으며, 실험 결과는 오픈 소스 모델이 한국 문화를 이해하는 데 있어 상용 모델에 비해 뒤처진다는 것을 보여줍니다. 또한, 외부 지식 통합 및 다중 선택 QA를 넘어서는 평가를 포함한 추가 분석을 제시합니다.


This paper emphasizes the need for a benchmark that evaluates the ability of culturally inclusive vision-language models (VLMs) to address culturally relevant questions. Existing approaches typically rely on human annotators, making the process labor-intensive and creating a cognitive burden in generating diverse questions. To address this, the authors propose a semi-automated framework for constructing cultural VLM benchmarks through human-VLM collaboration, where questions are generated and verified. Using this framework, they created the K-Viscuit dataset focused on Korean culture, and experiments reveal that open-source models lag behind proprietary ones in understanding Korean culture. The paper also presents further analyses, including human evaluation, augmenting VLMs with external knowledge, and evaluation beyond multiple-choice QA.


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



이 논문에서는 문화적으로 포괄적인 비전-언어 모델(VLM)을 평가하기 위한 벤치마크인 K-Viscuit를 개발하는 방법론을 제안합니다. 이 방법론은 인간과 VLM의 협업을 통해 문화적 질문을 생성하고 검증하는 반자동 프레임워크를 포함합니다. 

1. **모델 선택**: 연구팀은 강력한 비전-언어 모델인 GPT-4-Turbo를 사용하여 질문과 답변을 생성합니다. 이 모델은 이미지와 관련된 문화적 지식을 바탕으로 질문을 생성하는 데 도움을 줍니다.

2. **데이터 수집**: K-Viscuit 데이터셋은 한국 문화를 중심으로 구성되며, 10개의 핵심 개념(음식, 음료, 게임, 축제, 종교, 도구, 의복, 유산, 건축, 농업)을 포함합니다. 각 개념에 대해 다양한 이미지를 수집하고, 각 이미지에 대해 질문을 생성합니다.

3. **질문 생성**: 질문은 두 가지 유형으로 나뉩니다. 첫 번째는 시각적 인식을 평가하는 TYPE 1 질문이며, 두 번째는 문화적 지식 응용을 평가하는 TYPE 2 질문입니다. TYPE 1 질문은 기본적인 시각적 정보(예: 객체 식별)를 평가하고, TYPE 2 질문은 문화적 맥락에 대한 깊은 이해를 요구합니다.

4. **인간 검증**: 생성된 질문과 답변은 한국어 원어민에 의해 검증됩니다. 이 과정에서 질문의 문화적 적합성과 품질을 보장하기 위해 검토 및 수정이 이루어집니다.

5. **실험 및 평가**: K-Viscuit 데이터셋을 사용하여 다양한 비전-언어 모델의 성능을 평가합니다. 실험 결과는 오픈 소스 모델과 상용 모델 간의 성능 차이를 보여주며, 한국 문화에 대한 이해의 한계를 드러냅니다.

이러한 방법론은 문화적 다양성을 고려한 모델 평가의 중요성을 강조하며, 향후 연구에서 더 포괄적인 VLM을 개발하는 데 기여할 수 있습니다.

---




This paper proposes a methodology for developing a benchmark called K-Viscuit to evaluate culturally inclusive vision-language models (VLMs). The methodology includes a semi-automated framework for generating and verifying cultural questions through human-VLM collaboration.

1. **Model Selection**: The research team employs a powerful vision-language model, GPT-4-Turbo, to assist in generating questions and answers. This model helps create questions based on cultural knowledge related to the images.

2. **Data Collection**: The K-Viscuit dataset is centered around Korean culture and encompasses ten core concepts (Food, Beverage, Game, Celebrations, Religion, Tool, Clothes, Heritage, Architecture, Agriculture). Various images are collected for each concept, and questions are generated for each image.

3. **Question Generation**: Questions are divided into two types. The first type, TYPE 1, assesses visual recognition, while the second type, TYPE 2, evaluates cultural knowledge application. TYPE 1 questions assess basic visual information (e.g., object identification), while TYPE 2 questions require a deeper understanding of cultural context.

4. **Human Verification**: The generated questions and answers are verified by native Korean speakers. This process involves reviewing and revising to ensure cultural relevance and quality of the questions.

5. **Experiments and Evaluation**: The performance of various vision-language models is evaluated using the K-Viscuit dataset. The experimental results reveal performance gaps between open-source and proprietary models, highlighting limitations in understanding Korean culture.

This methodology emphasizes the importance of evaluating models with cultural diversity in mind and can contribute to the development of more inclusive VLMs in future research.


<br/>
# Results



이 논문에서는 K-Viscuit라는 한국 문화 중심의 비주얼 질문 응답(VQA) 벤치마크 데이터셋을 개발하고, 이를 통해 다양한 비전-언어 모델(VLM)의 성능을 평가하였다. K-Viscuit 데이터셋은 657개의 예시로 구성되어 있으며, 10개의 개념 카테고리(음식, 음료, 게임, 축제, 종교, 도구, 의복, 유산, 건축, 농업)를 포함하고 있다. 각 예시는 이미지, 질문, 그리고 네 개의 선택지로 구성되어 있으며, 질문은 시각적 인식과 문화적 지식 응용을 평가하는 두 가지 유형으로 나뉜다.

모델 성능 평가는 여러 개의 오픈 소스 및 상용 VLM을 사용하여 진행되었으며, 각 모델의 정확도를 측정하기 위해 다중 선택 질문 형식을 사용하였다. 실험 결과, 상용 모델들이 오픈 소스 모델들보다 한국 문화에 대한 이해에서 더 높은 정확도를 보였다. 예를 들어, GPT-4o 모델은 전체 정확도에서 89.50%를 기록하였고, Gemini-1.5-Pro는 81.58%의 정확도를 보였다. 반면, LLaVA-1.6-13B와 같은 오픈 소스 모델들은 상대적으로 낮은 성능을 보였으며, Llama-3.2-11B는 68.04%의 정확도를 기록하였다.

이 연구는 VLM의 문화적 이해 능력을 평가하기 위한 새로운 벤치마크를 제안하며, 한국 문화에 대한 VLM의 성능 차이를 분석함으로써 향후 개선 방향을 제시하고 있다. 또한, 질문의 유형에 따라 모델의 성능 차이를 분석한 결과, TYPE 2 질문(문화적 지식 응용)에서 모델들이 더 높은 정확도를 보이는 경향이 있음을 발견하였다. 이는 시각적 인식이 다양한 문화적 맥락에서 도전 과제가 될 수 있음을 시사한다.




This paper develops a benchmark dataset called K-Viscuit, focused on Korean culture, and evaluates the performance of various vision-language models (VLMs) using this dataset. The K-Viscuit dataset consists of 657 examples and includes 10 concept categories (Food, Beverage, Game, Celebrations, Religion, Tool, Clothes, Heritage, Architecture, Agriculture). Each example comprises an image, a question, and four options, with questions categorized into two types: visual recognition and cultural knowledge application.

Model performance evaluation was conducted using several open-source and proprietary VLMs, employing a multiple-choice question format to measure accuracy. The experimental results showed that proprietary models achieved higher accuracy in understanding Korean culture compared to open-source models. For instance, the GPT-4o model recorded an overall accuracy of 89.50%, while Gemini-1.5-Pro achieved 81.58%. In contrast, open-source models like LLaVA-1.6-13B exhibited relatively lower performance, with Llama-3.2-11B achieving an accuracy of 68.04%.

This study proposes a new benchmark for evaluating the cultural understanding capabilities of VLMs and highlights performance disparities in understanding Korean culture, suggesting directions for future improvements. Additionally, the analysis of performance differences based on question types revealed that models tended to perform better on TYPE 2 questions (cultural knowledge application), indicating that visual recognition poses inherent challenges in diverse cultural contexts.


<br/>
# 예제



이 논문에서는 한국 문화를 중심으로 한 비전-언어 모델(VLM)의 평가를 위한 K-Viscuit 벤치마크를 구축하는 방법을 제안합니다. 이 과정은 크게 네 가지 단계로 나뉩니다: 개념 선택, 이미지 선택, 질문 및 옵션 주석 달기, 그리고 인간 검증입니다.

1. **개념 선택**: 연구자들은 한국 문화와 관련된 다양한 개념을 정의합니다. 이 개념들은 음식, 음료, 게임, 축제, 종교, 도구, 의복, 유산, 건축, 농업 등으로 나뉘며, 각 개념은 일상 생활에서 자주 접할 수 있는 요소들로 구성됩니다.

2. **이미지 선택**: 각 개념에 맞는 이미지를 웹에서 수집합니다. 이때, 각 개념에 대해 최대 두 개의 이미지만 선택하여 다양성을 확보합니다. 이미지는 주로 위키미디어 공용에서 CC 라이센스를 가진 자료를 사용합니다.

3. **질문 및 옵션 주석 달기**: 선택된 이미지에 대해 VLM을 활용하여 질문과 그에 대한 정답 및 오답을 생성합니다. 질문은 두 가지 유형으로 나뉘며, TYPE 1은 기본적인 시각 정보 인식을 평가하고, TYPE 2는 문화적 지식 응용을 평가합니다. VLM은 인간이 제공한 예시와 지침을 바탕으로 질문을 생성합니다.

4. **인간 검증**: 생성된 질문과 옵션은 한국어 원어민에 의해 검증됩니다. 이 단계에서는 질문이 문화적으로 적절하고, 고유한 문화적 요소를 반영하는지를 확인합니다. 검증을 통해 부적절한 질문은 제외하고, 최종적으로 데이터셋에 포함될 질문을 선정합니다.

이러한 과정을 통해 K-Viscuit 데이터셋은 한국 문화에 대한 VLM의 이해도를 평가하는 데 사용됩니다. 데이터셋은 총 657개의 예시로 구성되며, 각 예시는 이미지, 질문, 그리고 네 개의 선택지로 이루어져 있습니다. 이 데이터셋은 VLM의 문화적 이해 능력을 평가하는 데 중요한 역할을 합니다.




This paper proposes a method for constructing the K-Viscuit benchmark to evaluate vision-language models (VLMs) focused on Korean culture. The process is divided into four main stages: concept selection, image selection, question and option annotation, and human verification.

1. **Concept Selection**: Researchers define various concepts related to Korean culture. These concepts are categorized into food, beverage, game, celebration, religion, tool, clothing, heritage, architecture, and agriculture, each consisting of elements commonly encountered in daily life.

2. **Image Selection**: Images corresponding to each concept are collected from the web. In this step, a maximum of two images per concept is selected to ensure diversity. The images are primarily sourced from Wikimedia Commons under CC licenses.

3. **Question and Option Annotation**: For the selected images, questions, correct answers, and distractors are generated using a VLM. The questions are categorized into two types: TYPE 1 assesses basic visual information recognition, while TYPE 2 evaluates cultural knowledge application. The VLM generates questions based on examples and guidelines provided by humans.

4. **Human Verification**: The generated questions and options are verified by native Korean speakers. This step ensures that the questions are culturally appropriate and reflect unique cultural elements. Inappropriate questions are excluded, and the final questions to be included in the dataset are selected.

Through this process, the K-Viscuit dataset is used to evaluate VLMs' understanding of Korean culture. The dataset consists of a total of 657 examples, each comprising an image, a question, and four options. This dataset plays a crucial role in assessing the cultural understanding capabilities of VLMs.

<br/>
# 요약

이 논문에서는 한국 문화를 중심으로 한 비전-언어 모델(VLM)의 문화적 이해를 평가하기 위한 K-Viscuit 벤치마크를 제안합니다. 연구진은 인간과 VLM의 협업을 통해 질문을 생성하고, 원주율을 통해 생성된 질문의 품질을 검증하여 데이터셋을 구축했습니다. 실험 결과, 상용 모델이 오픈 소스 모델보다 한국 문화 이해에서 더 높은 성능을 보였으며, 이는 문화적 다양성을 고려한 모델 평가의 중요성을 강조합니다.




This paper proposes the K-Viscuit benchmark to evaluate the cultural understanding of vision-language models (VLMs) focused on Korean culture. The researchers constructed the dataset by generating questions through human-VLM collaboration and verifying the quality of the generated questions via human review. Experimental results showed that proprietary models outperformed open-source models in understanding Korean culture, highlighting the importance of culturally diverse model evaluation.

<br/>
# 기타



이 논문에서는 K-Viscuit라는 한국 문화에 초점을 맞춘 비주얼-언어 모델(VLM) 벤치마크를 제안하고, 이를 통해 VLM의 문화적 이해 능력을 평가하는 방법을 제시합니다. K-Viscuit는 두 가지 질문 유형(시각 인식 및 문화적 지식 응용)을 포함하며, 각 질문은 이미지와 관련된 다중 선택 질문 형식으로 구성됩니다. 

#### 다이어그램 및 피규어
1. **프레임워크 개요 (Figure 2)**: 이 다이어그램은 K-Viscuit 데이터셋을 생성하는 과정의 단계(개념 선택, 이미지 선택, 질문 및 옵션 주석, 인간 검증)를 보여줍니다. VLM이 질문을 생성하고, 원주율이 검증하는 과정을 통해 데이터셋의 품질을 보장합니다.

2. **데이터셋 예시 (Figure 1)**: 이 피규어는 K-Viscuit 데이터셋의 예시를 보여줍니다. 각 이미지와 관련된 질문 및 선택지를 통해 VLM이 한국 문화를 얼마나 잘 이해하는지를 평가할 수 있습니다.

3. **개념 분포 (Figure 3)**: K-Viscuit 데이터셋의 각 개념(음식, 음료, 게임 등)의 분포를 나타내며, 데이터셋이 다양한 문화적 요소를 포괄하고 있음을 보여줍니다.

4. **모델 성능 비교 (Table 2)**: 다양한 VLM 모델의 성능을 비교한 표로, 개방형 소스 모델과 상용 모델 간의 성능 차이를 강조합니다. 상용 모델이 한국 문화 이해에서 더 높은 정확도를 보이는 경향이 있음을 보여줍니다.

5. **인간 평가 결과 (Figure 5)**: 한국인과 비한국인 참가자 간의 정확도를 비교한 그래프입니다. 한국 문화에 대한 이해도가 높은 참가자가 더 높은 정확도를 보이는 경향이 있음을 나타냅니다.

6. **시각 의존성 분석 (Figure 7)**: 실제 이미지와 무작위로 생성된 노이즈 이미지에서 모델의 성능을 비교하여 K-Viscuit 질문이 시각적 정보에 얼마나 의존하는지를 평가합니다.

#### 테이블
- **데이터셋 통계 (Table 1)**: K-Viscuit 데이터셋의 통계 정보를 제공하며, 총 657개의 예시와 10개의 개념 카테고리를 포함하고 있음을 보여줍니다.
- **모델 평가 결과 (Table 2)**: 각 모델의 정확도를 보여주며, 상용 모델이 개방형 소스 모델보다 더 나은 성능을 보이는 경향이 있음을 나타냅니다.
- **질문 유형별 성능 (Table 3)**: 시각 인식 질문과 문화적 지식 응용 질문에 대한 모델의 성능을 비교합니다.
- **회수 기반 생성 결과 (Table 7)**: 외부 지식을 활용한 모델 성능 향상을 보여주는 표로, 회수된 문서가 모델의 성능에 긍정적인 영향을 미친다는 것을 나타냅니다.




This paper proposes a benchmark called K-Viscuit, focused on Korean culture, to evaluate the cultural understanding capabilities of vision-language models (VLMs). K-Viscuit includes two types of questions (visual recognition and cultural knowledge application), with each question structured in a multiple-choice format related to images.

#### Diagrams and Figures
1. **Framework Overview (Figure 2)**: This diagram illustrates the stages of creating the K-Viscuit dataset (concept selection, image selection, question and option annotation, human verification). It shows how the VLM generates questions and how native speakers verify them to ensure dataset quality.

2. **Dataset Examples (Figure 1)**: This figure presents examples from the K-Viscuit dataset, demonstrating how questions and options related to each image can assess the VLM's understanding of Korean culture.

3. **Concept Distribution (Figure 3)**: It shows the distribution of various concepts (food, beverage, game, etc.) in the K-Viscuit dataset, indicating that the dataset encompasses a wide range of cultural elements.

4. **Model Performance Comparison (Table 2)**: A table comparing the performance of various VLM models, highlighting the performance gap between open-source and proprietary models in understanding Korean culture.

5. **Human Evaluation Results (Figure 5)**: A graph comparing the accuracy between Korean and non-Korean participants, indicating that those with a better understanding of Korean culture tend to achieve higher accuracy.

6. **Visual Dependency Analysis (Figure 7)**: This figure compares model performance using actual images versus randomly generated noise images, assessing how much K-Viscuit questions rely on visual information.

#### Tables
- **Dataset Statistics (Table 1)**: Provides statistical information about the K-Viscuit dataset, indicating it contains a total of 657 examples across 10 concept categories.
- **Model Evaluation Results (Table 2)**: Shows the accuracy of each model, indicating that proprietary models tend to perform better than open-source models.
- **Performance by Question Type (Table 3)**: Compares model performance on visual recognition questions versus cultural knowledge application questions.
- **Retrieval-Augmented Generation Results (Table 7)**: A table showing how external knowledge retrieval positively impacts model performance, indicating that retrieved documents can enhance cultural knowledge in VQA tasks.

<br/>
# refer format:

### BibTeX 

```bibtex
@inproceedings{Park2025,
  author = {ChaeHun Park and Yujin Baek and Jaeseok Kim and Yu-Jung Heo and Du-Seong Chang and Jaegul Choo},
  title = {Evaluating Visual and Cultural Interpretation: The K-Viscuit Benchmark with Human-VLM Collaboration},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages = {21960--21974},
  year = {2025},
  month = {July},
  publisher = {Association for Computational Linguistics},
  url = {https://huggingface.co/datasets/ddehun/k-viscuit}
}
```

### 시카고 스타일

ChaeHun Park, Yujin Baek, Jaeseok Kim, Yu-Jung Heo, Du-Seong Chang, and Jaegul Choo. "Evaluating Visual and Cultural Interpretation: The K-Viscuit Benchmark with Human-VLM Collaboration." In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 21960–21974. July 2025. Association for Computational Linguistics. https://huggingface.co/datasets/ddehun/k-viscuit.
