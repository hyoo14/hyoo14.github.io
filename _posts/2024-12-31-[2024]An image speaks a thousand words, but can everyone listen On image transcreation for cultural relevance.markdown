---
layout: post
title:  "[2024]An image speaks a thousand words, but can everyone listen? On image transcreation for cultural relevance"  
date:   2024-12-31 02:20:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 



짧은 요약(Abstract) :    


멀티미디어 콘텐츠의 증가로 인해 번역자는 단어뿐만 아니라 이미지 같은 다른 요소들도 문화적으로 적합하게 변환하는 역할을 맡게 되었습니다. 그러나 기계 번역 시스템은 여전히 텍스트와 음성 언어 처리에만 국한되어 있습니다. 이 연구에서는 이미지 번역 작업을 통해 이미지를 문화적으로 관련성 있게 만드는 새로운 작업을 제안합니다. 이를 위해 최첨단 생성 모델로 구성된 세 가지 파이프라인을 구축하고, 문화적 관련성과 의미 보존을 평가하기 위해 개념과 실제 응용 프로그램 두 부분으로 이루어진 데이터셋을 생성했습니다. 인간 평가를 통해 현재 이미지 편집 모델은 이 작업을 수행하는 데 실패했으며, 대형 언어 모델(LLM)과 검색기를 결합하면 성능을 개선할 수 있음을 발견했습니다. 하지만, 개념 데이터셋에서는 일부 국가에서 이미지를 번역할 성공률이 5%에 불과하고, 응용 프로그램 데이터셋에서는 성공 사례가 전혀 없는 등 이 작업이 매우 어렵다는 점이 강조됩니다.


With the rise of multimedia content, translators increasingly adapt not only words but also elements like images to suit cultural contexts. However, machine translation systems remain confined to speech and text processing. This research introduces a new task of translating images to enhance cultural relevance. Three state-of-the-art generative model pipelines are proposed, and a dataset comprising conceptual and real-world application parts was created for evaluation. Human evaluations revealed that current image-editing models fail to address this task, but combining large language models (LLMs) and retrievers can improve performance. Despite this, the best pipelines only succeed in translating 5% of images for some countries in the conceptual dataset, and no successes were observed in the application dataset, highlighting the task's challenges.


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


이 논문은 이미지를 문화적으로 번역(트랜스크리에이션)하는 새로운 작업을 제안하며, 이를 위해 다음과 같은 3가지 파이프라인을 설계했습니다:

1. **e2e-instruct (End-to-End Instruction 기반 편집)**  
   - **사용 모델**: InstructPix2Pix  
   - **핵심 내용**: 자연어 지침을 입력으로 받아 한 번의 단계에서 이미지를 편집하는 방식입니다. 모델은 텍스트 레이블이나 분할 마스크 없이 자연어 설명만으로 이미지를 편집할 수 있습니다.  
   - **한계**: 문화적 맥락을 이해하거나 세부적인 편집을 수행하는 데 있어 한계를 보입니다.

2. **cap-edit (Caption → 텍스트 편집 → 이미지 편집)**  
   - **사용 모델**: InstructBLIP-FlanT5-XXL (이미지 캡션 생성), GPT-3.5 (캡션 변환), PlugnPlay (이미지 편집)  
   - **핵심 내용**: 이미지를 캡션화한 후, 해당 캡션을 대형 언어 모델(LLM)을 이용해 문화적으로 수정합니다. 이후, 수정된 캡션을 기반으로 이미지를 편집합니다.  
   - **장점**: 이미지 편집 모델이 직접 처리하기 어려운 문화적 이해 부분을 LLM이 처리하도록 분담함으로써 성능이 향상됩니다.

3. **cap-retrieve (Caption → 텍스트 편집 → 이미지 검색)**  
   - **사용 모델**: LLM 편집(GPT-3.5), LAION 데이터셋 기반 이미지 검색  
   - **핵심 내용**: 캡션을 수정한 후, 해당 캡션을 사용해 국가별 이미지 데이터베이스에서 적절한 이미지를 검색합니다.  
   - **한계**: 검색된 이미지가 원본과 시각적 유사성이 부족하거나 문화적으로 관련성이 떨어지는 경우가 종종 발생합니다.

---

### 트레이닝 데이터
- **개념 데이터셋**: 브라질, 일본, 인도 등 7개국에서 수집된 약 600개의 이미지로 구성. 각 이미지에는 음식, 의류, 축제 등의 보편적인 개념을 반영.  
- **응용 데이터셋**: 실제 세계 응용 프로그램(교육 및 어린이 문학)에서 수집된 100개의 이미지로 구성. 예를 들어, 수학 문제지에서 지역적 맥락에 맞는 물체나 이미지를 사용하는 사례 포함.  
- **데이터 수집 과정**: 각 국가의 문화적 맥락을 반영하기 위해 현지인 평가자가 데이터를 수집하고 검토.

---


This paper proposes a novel task of translating images for cultural relevance and designs three pipelines:

1. **e2e-instruct (End-to-End Instruction-based Editing)**  
   - **Model Used**: InstructPix2Pix  
   - **Key Idea**: This approach edits images in a single step by taking natural language instructions as input. The model does not require text labels, segmentation masks, or example output images.  
   - **Limitations**: The model struggles to perform culturally nuanced edits and lacks a deep understanding of context.

2. **cap-edit (Caption → Text Edit → Image Edit)**  
   - **Models Used**: InstructBLIP-FlanT5-XXL (image captioning), GPT-3.5 (caption transformation), PlugnPlay (image editing)  
   - **Key Idea**: This modular pipeline captions the image, modifies the caption using an LLM for cultural relevance, and then edits the image based on the transformed caption.  
   - **Advantages**: By offloading cultural understanding to LLMs, the pipeline achieves better results than direct image-editing models.

3. **cap-retrieve (Caption → Text Edit → Image Retrieval)**  
   - **Models Used**: LLM editing (GPT-3.5), image retrieval from the LAION dataset  
   - **Key Idea**: The culturally modified caption is used to retrieve an appropriate image from a country-specific image database.  
   - **Limitations**: Retrieved images often lack visual similarity to the source or cultural relevance.

---

### Training Data
- **Concept Dataset**: Contains ~600 images from 7 countries (e.g., Brazil, Japan, India). Each image reflects universal concepts like food, clothing, and festivals.  
- **Application Dataset**: Includes 100 images from real-world applications like educational worksheets and children's literature. Examples include replacing Christmas trees with local decorations.  
- **Data Collection Process**: Native annotators ensure cultural relevance by collecting and verifying data from their respective regions.



   
 
<br/>
# Results  





**테스트 데이터**
- **개념 데이터셋**: 브라질, 일본, 미국 등 7개국에서 약 600개의 이미지를 포함한 데이터셋. 각 이미지는 음식, 의류, 축제 등 보편적인 개념을 나타냄.
- **응용 데이터셋**: 수학 문제지나 어린이 문학과 같은 실제 응용 분야에서 수집된 약 100개의 이미지. 각 이미지는 특정 문화적 맥락에 맞춰 사용되도록 설계됨.

**비교 모델 및 파이프라인**
- **테스트된 파이프라인**:  
  1. e2e-instruct  
  2. cap-edit  
  3. cap-retrieve  
- **평가 기준**: 시각적 변화(C0), 의미적 동일성(C1), 문화적 관련성(C3), 자연스러움(C4) 등 인간 평가를 통해 성능 평가.

**결과**
1. **개념 데이터셋에서의 성능**:
   - 가장 높은 성능은 **cap-retrieve** 파이프라인에서 관찰되었으며, 일부 국가에서는 최대 30%의 이미지를 성공적으로 변환(일본)했지만, 일부 국가(나이지리아)에서는 5%에 그침.
   - **cap-edit**는 전반적으로 안정적인 성능을 보였으나, e2e-instruct는 편집하지 않은 이미지가 많아 상대적으로 낮은 점수를 기록.

2. **응용 데이터셋에서의 성능**:
   - 교육용 이미지에서는 **cap-edit**가 이미지의 문화적 관련성을 높이며 원본 학습 목표를 유지하는 데 가장 성공적이었음.
   - 그러나 응용 데이터셋에서 작업이 더 복잡했기 때문에 모든 파이프라인이 일부 국가(포르투갈)에서 실패를 기록.

**향상된 점**
- 대형 언어 모델(LLM)을 활용한 **cap-edit**와 **cap-retrieve**는 단순히 이미지를 편집하는 **e2e-instruct**에 비해 문화적 맥락 이해 및 자연스러운 편집에서 더 높은 성능을 달성.
- LLM과 검색기를 결합함으로써 일부 국가에서 성공률이 5%에서 30%로 향상.

---



**Test Data**
- **Concept Dataset**: Contains ~600 images across 7 countries, each representing universal categories such as food, clothing, and festivals.
- **Application Dataset**: Consists of ~100 images from real-world applications like math worksheets and children's literature, designed to fit specific cultural contexts.

**Comparison Models and Pipelines**
- **Tested Pipelines**:  
  1. e2e-instruct  
  2. cap-edit  
  3. cap-retrieve  
- **Evaluation Metrics**: Performance was assessed via human evaluation based on visual change (C0), semantic equivalence (C1), cultural relevance (C3), and naturalness (C4).

**Results**
1. **Performance on the Concept Dataset**:
   - **cap-retrieve** performed the best, achieving up to 30% successful transcreations for some countries (Japan), but only 5% for others (Nigeria).
   - **cap-edit** showed consistent performance, while e2e-instruct had a lower success rate due to many unedited images.

2. **Performance on the Application Dataset**:
   - For educational images, **cap-edit** successfully improved cultural relevance while maintaining the original educational goals.
   - The task complexity in the application dataset resulted in all pipelines failing for some countries (Portugal).

**Improvements**
- Pipelines leveraging large language models (LLMs), such as **cap-edit** and **cap-retrieve**, outperformed the simple **e2e-instruct** in understanding cultural contexts and producing natural edits.
- By incorporating LLMs and retrievers, success rates improved from 5% to 30% for certain countries.


<br/>
# 예제  


데이터셋 예시

이미지: 나이지리아 전통 음식인 "아말라와 에웨두(amala-and-ewedu)"가 담긴 접시.
목표 국가: 미국
원본 캡션: "나이지리아 전통 음식 아말라와 에웨두를 담은 접시."
목표 캡션 (LLM 편집 후): "미국 전통 음식 비프와 채소를 담은 접시."


Dataset Example

Image: A plate featuring the Nigerian traditional dish "amala and ewedu."
Target Country: United States
Original Caption: "A plate of Nigerian traditional food, amala and ewedu."
Target Caption (after LLM edit): "A plate of American traditional food, beef and vegetables."

<br/>  
# 요약   


이 연구는 이미지를 문화적으로 번역하는 새로운 작업을 제안하며, InstructPix2Pix, LLM(예: GPT-3.5) 기반 편집 및 검색 파이프라인(cap-edit, cap-retrieve)을 설계했습니다. 개념 데이터셋(7개국, 약 600개 이미지)과 응용 데이터셋(교육 자료 등 100개 이미지)을 사용해 평가한 결과, cap-retrieve는 문화적 관련성에서 가장 높은 성능을 보였으며 일부 국가에서는 성공률이 30%에 도달했습니다. 예를 들어, 나이지리아 전통 음식을 미국 전통 음식으로 변환할 때, cap-edit는 음식만 변경하며 원본 이미지를 유지해 높은 점수를 얻었고, cap-retrieve는 적합한 대체 이미지를 검색해 문화적 관련성을 극대화했으나 레이아웃 유사성은 낮았습니다.

---


This study proposes a novel task of culturally translating images and designs pipelines such as InstructPix2Pix, LLM-based editing (e.g., GPT-3.5), and retrieval pipelines (cap-edit, cap-retrieve). Using a concept dataset (~600 images across 7 countries) and an application dataset (100 educational images), evaluations showed that cap-retrieve achieved the highest cultural relevance, with success rates reaching 30% for some countries. For example, when translating Nigerian traditional food to American cuisine, cap-edit preserved the original structure while changing the dish, achieving high scores, whereas cap-retrieve maximized cultural relevance by retrieving suitable replacement images but lacked layout similarity.


<br/>  
# 기타  


Figure 1:
연구의 전체 파이프라인을 시각화한 다이어그램으로, e2e-instruct, cap-edit, cap-retrieve의 세 가지 주요 파이프라인 구성 요소를 보여줍니다. 각 단계는 입력 이미지에서 시작하여 최종 결과 이미지로 이어지며, 각 파이프라인의 주요 프로세스를 설명합니다.

Figure 2:
파이프라인 성능 비교 그래프로, 각 국가(브라질, 일본, 미국 등)별로 e2e-instruct, cap-edit, cap-retrieve의 성공률을 보여줍니다. cap-retrieve가 대부분의 국가에서 가장 높은 성공률을 기록했으며, 일부 국가(나이지리아 등)에서는 모든 파이프라인의 성능이 낮게 나타났습니다.

Figure 3:
예제 이미지를 포함한 사례 연구로, 원본 이미지와 각 파이프라인(e2e-instruct, cap-edit, cap-retrieve) 결과를 비교합니다. 예를 들어, 나이지리아 음식을 미국 음식으로 변환한 결과를 보여주며, 각 파이프라인의 시각적 변화와 문화적 관련성을 평가합니다.

Figure 4:
응용 데이터셋 결과를 시각적으로 보여주는 그래프와 이미지 예제입니다. 수학 문제지에서 지역적 요소(예: 나무나 의류)를 대체하는 과정을 다루며, cap-edit가 가장 자연스럽고 목표 맥락을 잘 반영한 결과를 생성했음을 강조합니다.


Figure 1:
A diagram illustrating the overall research pipeline, showcasing the three main pipelines: e2e-instruct, cap-edit, and cap-retrieve. Each step is detailed, starting from the input image and progressing to the final output, highlighting the core processes of each pipeline.

Figure 2:
A performance comparison graph showing success rates of e2e-instruct, cap-edit, and cap-retrieve across different countries (e.g., Brazil, Japan, USA). Cap-retrieve achieved the highest success rates in most countries, while all pipelines showed low performance in some countries like Nigeria.

Figure 3:
Case studies with example images comparing original images and results from each pipeline (e2e-instruct, cap-edit, cap-retrieve). For instance, it illustrates the transformation of Nigerian food into American food, evaluating the visual changes and cultural relevance of each pipeline.

Figure 4:
Visual results and examples from the application dataset, including changes to contextual elements (e.g., trees or clothing) in math worksheets. The figure highlights how cap-edit produced the most natural and contextually appropriate results.

<br/>
# refer format:     


@inproceedings{khanuja2024image_transcreation,
  title={An image speaks a thousand words, but can everyone listen? On image transcreation for cultural relevance},
  author={Khanuja, Simran and Ramamoorthy, Sathyanarayanan and Song, Yueqi and Neubig, Graham},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  pages={10258--10279},
  year={2024},
  organization={Association for Computational Linguistics},
  address={Singapore},
  month={November},
  doi={10.18653/v1/2024.emnlp-main.573},
  url={https://machine-transcreation.github.io/image-transcreation/}
}



Khanuja, Simran, Sathyanarayanan Ramamoorthy, Yueqi Song, and Graham Neubig. “An Image Speaks a Thousand Words, but Can Everyone Listen? On Image Transcreation for Cultural Relevance.” Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, November 12–16, 2024, 10258–10279. Singapore: Association for Computational Linguistics. https://machine-transcreation.github.io/image-transcreation/.

