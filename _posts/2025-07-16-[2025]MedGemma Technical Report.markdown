---
layout: post
title:  "[2025]MedGemma Technical Report"  
date:   2025-07-16 18:49:40 -0800
categories: study
---

{% highlight ruby %}


한줄 요약: 



짧은 요약(Abstract) :    



의료 분야에서 인공지능(AI)은 매우 큰 잠재력을 지니고 있지만, 데이터의 다양성과 복잡성, 개인정보 보호 요구 등으로 인해 학습과 배포에 어려움이 따릅니다. 이를 해결하기 위해 이 보고서는 Gemma 3 모델을 기반으로 개발된 MedGemma라는 새로운 의료 비전-언어 파운데이션 모델 모음을 소개합니다 (모델 크기: 4B 및 27B). MedGemma는 의료 이미지와 텍스트에 대한 고차원의 이해 및 추론 능력을 보여주며, 동급 크기의 생성 모델보다 훨씬 뛰어난 성능을 보이고, 특정 작업에 최적화된 모델 수준에 근접합니다. 특히, 훈련에 사용되지 않은 데이터(Out-of-distribution)에 대해서도 의료 멀티모달 QA에서 2.6–10%, 흉부 엑스레이 분류에서 15.5–18.1%, 에이전트 환경에서 10.8% 향상된 성능을 기록했습니다. 추가 파인튜닝을 통해 특정 하위 도메인에서도 성능이 더욱 향상되며, 예를 들어 전자의무기록 검색 오류를 50% 줄였고, 기흉 분류나 병리학 패치 분류에서도 최첨단 모델과 유사한 성능을 달성했습니다. 이와 함께, MedGemma는 시각 인식 기능을 담당하는 의료 특화 비전 인코더인 MedSigLIP도 소개하며, 기존 특화 이미지 인코더와 대등하거나 우수한 성능을 보입니다. 이 모델들은 의료 연구 및 응용 개발을 크게 가속할 수 있는 강력한 기반을 제공합니다.


Artificial intelligence (AI) holds significant promise for healthcare, but faces challenges in training and deployment due to the diversity of medical data, complexity of tasks, and privacy constraints. This report introduces MedGemma, a new collection of medical vision–language foundation models built on Gemma 3 (4B and 27B). MedGemma shows advanced understanding and reasoning over medical images and text, outperforming similarly-sized generative models and approaching task-specific performance levels while preserving the generality of the Gemma 3 base. On out-of-distribution tasks, MedGemma improves medical multimodal QA by 2.6–10%, chest X-ray classification by 15.5–18.1%, and agentic evaluations by 10.8%. Fine-tuning MedGemma further enhances performance in subdomains, reducing electronic health record retrieval errors by 50%, and achieving performance comparable to specialized SOTA methods for pneumothorax and histopathology classification. Additionally, MedSigLIP, a medically tuned vision encoder derived from SigLIP, enables strong visual capabilities and matches or exceeds specialized medical image encoders. Together, the MedGemma collection offers a robust foundation for accelerating medical research and downstream AI applications.



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





**1. 모델 아키텍처**
MedGemma는 Google의 **Gemma 3** 언어 모델(4B, 27B 크기)을 기반으로 하며, 이미지 인코더로는 **SigLIP-400M**을 수정한 **MedSigLIP**을 사용합니다. 입력 이미지 해상도는 896×896이며, 텍스트는 SentencePiece 토크나이저(262K vocab)를 사용합니다. 멀티모달 처리를 위해 이미지-텍스트 교차구성이 가능하고, 최대 128K 토큰 길이의 컨텍스트를 처리할 수 있습니다. 모델은 TPUv4/v5e/v5p에서 훈련되었습니다.

**2. 트레이닝 데이터**

* **텍스트 데이터**는 MedQA, MedMCQA, PubMedQA, AfriMed-QA, HealthSearchQA 등 다양한 의료 QA 데이터셋에서 **instruction-tuned teacher model**을 이용해 응답 및 로짓을 생성하여 사용하였고, 약 **20만 개의 합성 질문**도 포함되었습니다.
* **이미지 데이터**는 X-ray, CT, MRI, 병리 조직(patch), 피부질환, 안과(망막) 등 총 3300만 개 이상의 이미지-텍스트 페어를 포함합니다.
* **데이터 전처리**는 Gemma 3 방식을 따라 이미지를 리사이즈(896×896) 및 정규화하고, CT 이미지는 RGB 3채널에 각각 다른 window/level 조합으로 인코딩됩니다.

**3. 학습 단계**

* **Vision Encoder 사전학습**: SigLIP을 2% 비율로 의료 데이터와 혼합하여 기존 성능을 유지하면서 의료 표현 학습을 강화.
* **멀티모달 사전학습 (Pretraining)**: 기존 Gemma 3 체크포인트를 기반으로, 의료 이미지-텍스트 데이터를 10% 비율로 섞어 5 에폭 학습.
* **후처리 학습 (Post-training)**:

  * **Distillation**: 대형 instruction-tuned 모델에서 생성된 의료 응답을 사용해 텍스트 학습
  * **Reinforcement Learning (RL)**: 이미지-텍스트 페어를 이용한 멀티모달 RL 수행 (supervised보다 더 일반화 잘 됨)

**4. MedSigLIP**
MedSigLIP은 SigLIP-400M을 의료 이미지에 특화되도록 미세조정한 인코더이며, 효율적인 학습을 위해 448×448 해상도 버전도 제공됩니다(포지셔널 임베딩만 downsample).

---



**1. Model Architecture**
MedGemma builds on the **Gemma 3 language model** (available in 4B and 27B sizes) with a vision encoder derived from **SigLIP-400M**, called **MedSigLIP**. Inputs use 896×896 resolution images and SentencePiece tokenizer with a 262K vocabulary. The architecture supports multimodal input with flexible image-text interleaving and a long context length of up to 128k tokens. Training was conducted on TPUv4, TPUv5e, and TPUv5p.

**2. Training Data**

* **Text data** came from multiple medical QA datasets (MedQA, MedMCQA, PubMedQA, AfriMed-QA, HealthSearchQA, etc.) with responses and logits sampled from a large **instruction-tuned teacher model**, including **200,000 synthetic QA pairs**.
* **Image data** included over **33 million image-text pairs** from radiology (X-rays, CTs, MRIs), histopathology (patches), dermatology (skin lesion images), and ophthalmology (fundus images).
* **Preprocessing** followed Gemma 3 settings, with images resized to 896×896. CT images were RGB encoded using different window/level settings to highlight anatomical features.

**3. Training Stages**

* **Vision Encoder Enhancement**: Fine-tuned SigLIP with 2% medical data mixed into original training corpus to retain general performance.
* **Multimodal Pretraining**: Continued training from Gemma 3 checkpoints with 10% medical image-text data for 5 epochs, selected checkpoint based on chest X-ray and VQA validation performance.
* **Post-training**:

  * **Distillation** from instruction-tuned teacher on medical text
  * **Reinforcement Learning (RL)** using image-text pairs to enhance multimodal generalization

**4. MedSigLIP**
MedSigLIP is a specialized vision encoder tuned for medical images. It retains the structure of SigLIP-400M and is released in a **448×448 resolution variant** for efficient usage by the community, differing only in positional embeddings.





   
 
<br/>
# Results  




**1. 평가 테스크 및 데이터셋**
MedGemma는 다음 5가지 주요 의료 테스크와 일반 벤치마크에서 평가되었습니다:

* **의료 텍스트 질의응답**: MedQA, MedMCQA, PubMedQA, MMLU(Medical), AfriMed-QA, MedXpertQA(OOD)
* **의료 이미지 분류**: MIMIC-CXR, CheXpert, CXR14(흉부 X-ray), US-Derm, PathMCQA, EyePACS
* **의료 시각 질의응답 (VQA)**: SLAKE, VQA-RAD, MedXpertQA
* **흉부 X-ray 리포트 생성**: MIMIC-CXR (RadGraph F1, 전문가 평가)
* **에이전트 행동 평가**: AgentClinic (MedQA 기반, MIMIC-IV 기반)
* **일반 도메인 평가**: MMLU Pro, Global MMLU Lite, MMMU(val)

**2. 성능 요약**

* **텍스트 QA**

  * MedGemma 27B는 기존 Gemma 3 27B 대비 최대 **+12\~20% 정확도 향상**
  * OOD인 MedXpertQA에서도 27B 모델은 **25.7% → 67.5%**(멀티모달)
  * 경쟁 모델 GPT-4o, Gemini 2.5와 유사한 성능

* **이미지 분류**

  * MedGemma 4B는 CheXpert, CXR14에서 **15\~20% 이상의 macro F1 향상**
  * 피부질환(US-Derm): 71.8% (Gemma 3 4B는 52.5%)
  * 망막병증(EyePACS): 64.9% (Gemma 3 4B는 14.4%)

* **VQA**

  * SLAKE token F1: 72.3% (Gemma 3 4B는 40.2%)
  * VQA-RAD accuracy: 69.1% (Gemma 3 4B는 48.7%)

* **리포트 생성 (MIMIC-CXR)**

  * RadGraph F1: **29.5%** (MedVersa 30.0과 유사)
  * 전문가 평가에서 **81%의 생성 리포트가 원본보다 임상적으로 우수하거나 동등**

* **에이전트 행동 (AgentClinic)**

  * MedGemma 27B: MedQA 기반에서 **56.2%**, MIMIC-IV 기반에서 **46.0%**, 인간의사보다 높은 성능 기록 (54%)

* **일반 도메인 성능 유지**

  * MMLU Pro: 60.2% (Gemma 3 27B: 67.5%)
  * Multi-modal MMMU: 47.3% (Gemma 3 4B: 48.8%)
  * 크기 대비 일반 모델 대비 성능 저하 거의 없음

**3. 미세조정 결과**

* MIMIC-CXR 리포트 생성: RadGraph F1 → 29.5 → **30.3 (SOTA 달성)**
* Pneumothorax 분류: F1 → **71.5** (SOTA 72.5 근접)
* CRC100k 병리 패치 분류: Weighted F1 → **94.5** (SOTA 97.3에 근접)
* EHR QA (synthetic): RL fine-tune 후 **84.2 → 93.6%**

**4. MedSigLIP 성능**

* ELIXR 대비 CheXpert zero-shot AUC **+2.0%**, fracture AUC **+7.1%** 향상
* Dermatology, Histopathology에서도 HAI-DEF 모델보다 높은 linear probe 성능

---



**1. Evaluation Tasks and Datasets**
MedGemma was evaluated across five core medical tasks and several general-purpose benchmarks:

* **Medical Text QA**: MedQA, MedMCQA, PubMedQA, MMLU-Med, AfriMed-QA, MedXpertQA (OOD)
* **Medical Image Classification**: MIMIC-CXR, CheXpert, CXR14 (chest X-rays), US-Derm, PathMCQA, EyePACS
* **Medical Visual Question Answering (VQA)**: SLAKE, VQA-RAD, MedXpertQA
* **Chest X-ray Report Generation**: MIMIC-CXR (RadGraph F1 and human expert review)
* **Agentic Behavior Evaluation**: AgentClinic (MedQA and MIMIC-IV-based)
* **General Purpose Tasks**: MMLU Pro, Global MMLU Lite, MMMU (val)

**2. Performance Highlights**

* **Text QA**

  * MedGemma 27B outperforms Gemma 3 27B by **+12–20%**
  * MedXpertQA (OOD): 25.7% (text-only) → **67.5%** (multimodal)
  * Comparable to GPT-4o and Gemini 2.5 Pro

* **Image Classification**

  * Chest X-rays: **+15–20% macro F1 improvement**
  * US-Derm: 71.8% (vs. 52.5%)
  * EyePACS: 64.9% (vs. 14.4%)

* **VQA**

  * SLAKE token F1: **72.3%** (vs. 40.2%)
  * VQA-RAD accuracy: **69.1%** (vs. 48.7%)

* **CXR Report Generation**

  * RadGraph F1: **29.5%**, matching SOTA models like MedVersa
  * Expert review: **81% of reports** were equal or superior in clinical decision quality

* **Agentic Behavior (AgentClinic)**

  * MedGemma 27B: 56.2% (MedQA) and 46.0% (MIMIC-IV), **exceeding human physicians (54%)**

* **General Purpose Tasks**

  * MMLU Pro: 60.2% (vs. 67.5%)
  * MMMU (multi-modal): 47.3% (vs. 48.8%)
  * Minimal trade-off despite specialization

**3. Fine-tuning Results**

* CXR Report Generation (RadGraph F1): 29.5 → **30.3 (SOTA)**
* Pneumothorax Classification (SIIM-ACR): F1 = **71.5**
* Histopathology (CRC100k): Weighted F1 = **94.5**
* EHR QA: Accuracy improved from 84.2% to **93.6%** after RL fine-tuning

**4. MedSigLIP Performance**

* On CheXpert, zero-shot AUC **+2.0%**, fracture AUC **+7.1%** vs. ELIXR
* Outperforms HAI-DEF models in dermatology and histopathology on linear probe tasks

---




<br/>
# 예제  





**1. 의료 질의응답 (Text QA) 예시**

* **입력**:
  질문: “A 35-year-old man presents with fever, cough, and chest pain. Chest X-ray shows lobar consolidation. What is the most likely diagnosis?”

* **출력 (MedGemma 27B)**:
  “Community-acquired pneumonia”

* **특징**: 의료시험 스타일의 질문에 대해 요약된 정확한 진단을 출력.

---

**2. 의료 시각 질의응답 (Visual QA) 예시**

* **입력**:

  * 이미지: 흉부 X-ray
  * 질문: “What abnormality is shown in the left lung?”

* **출력 (MedGemma 4B)**:
  “Large pneumothorax with a chest tube in place”

* **추가 정보**: 모델은 X-ray의 위치 및 병변 상태를 언어로 설명 가능.

---

**3. 병리 이미지 분류 (Histopathology QA)**

* **입력**: 병리 조직 슬라이드 이미지 + 질문: “What is the most likely diagnosis?”

* **출력**: “High-grade invasive carcinoma”

* **참고**: 조직 이미지의 특징을 추출하여, 병리 등급 및 암의 종류를 추론.

---

**4. 피부 질환 예측 (Dermatology)**

* **입력**: 피부 병변 이미지

* **출력**: “Basal Cell Carcinoma (BCC)”

* **질문 확장 예시**:

  * “What further information would you need from the patient?”
  * 모델 출력:
    “When did it start? Is it itchy? Any similar lesions?”

---

**5. EHR 정보 추론 (Fine-tuned 모델)**

* **입력**: FHIR 포맷의 환자 기록 요약 (수천 개 항목 포함)

  * 예: "Patient is on metformin and insulin. Last HbA1c was 9.2%."
  * 질문: “Is the patient's diabetes controlled?”
* **출력**: “No. The HbA1c level is elevated, indicating poor glycemic control.”

---



**1. Medical Text QA Example**

* **Input**:
  “A 35-year-old man presents with fever, cough, and chest pain. Chest X-ray shows lobar consolidation. What is the most likely diagnosis?”

* **Output (MedGemma 27B)**:
  “Community-acquired pneumonia”

* **Note**: Model gives precise clinical diagnosis in USMLE-style question format.

---

**2. Medical Visual Question Answering**

* **Input**:

  * Image: Chest X-ray
  * Question: “What abnormality is shown in the left lung?”

* **Output (MedGemma 4B)**:
  “Large pneumothorax with a chest tube in place”

* **Note**: Open-ended, descriptive answers showing spatial understanding.

---

**3. Histopathology Classification**

* **Input**: Patch from pathology slide + Question: “What is the most likely diagnosis?”

* **Output**: “High-grade invasive carcinoma”

* **Note**: Inference based on image morphology and cellular structure.

---

**4. Dermatology Image Classification**

* **Input**: Skin lesion image

* **Output**: “Basal Cell Carcinoma (BCC)”

* **Extended Interaction**:

  * “What further information would you need from the patient?”
  * Output:
    “When did it start? Is it itchy? Any similar lesions?”

---

**5. EHR Reasoning (Fine-tuned model)**

* **Input**: Condensed FHIR-format patient record

  * “Patient is on metformin and insulin. Last HbA1c was 9.2%.”
  * Question: “Is the patient’s diabetes controlled?”
* **Output**: “No. The HbA1c level is elevated, indicating poor glycemic control.”

---




<br/>  
# 요약   


MedGemma는 Gemma 3 모델 기반의 의료 비전-언어 파운데이션 모델로, 3300만 개 이상의 의료 이미지-텍스트 페어와 다양한 QA 데이터를 활용해 멀티모달 학습과 강화학습으로 훈련되었습니다.
이 모델은 MedQA, MIMIC-CXR, AgentClinic 등 다양한 벤치마크에서 동급 크기의 일반 모델보다 최대 20% 이상의 성능 향상을 보이며, 흉부 X-ray 리포트 생성이나 병리 진단에서도 SOTA 성능에 근접했습니다.
예를 들어, “이 환자의 가장 가능성 높은 진단은?”이라는 질문과 병리 이미지를 입력하면 MedGemma는 “고등급 침습성 암”과 같이 정확하고 요약된 의학적 답변을 생성합니다.


MedGemma is a medical vision-language foundation model built on Gemma 3, trained with over 33 million medical image–text pairs and various QA datasets using multimodal pretraining and reinforcement learning.
It outperforms similarly sized base models by up to 20% on benchmarks like MedQA, MIMIC-CXR, and AgentClinic, and approaches state-of-the-art performance in tasks like chest X-ray report generation and histopathology classification.
For instance, given a pathology image and a prompt like “What is the most likely diagnosis?”, MedGemma generates concise and accurate responses such as “High-grade invasive carcinoma.”


<br/>  
# 기타  




** 표 3\~4: 의료 QA 성능 비교**

* MedGemma 27B는 MedQA에서 **87.7%**, PubMedQA에서 **76.8%** 정확도를 달성하며, 동일 규모의 Gemma 3보다 10\~20% 향상됨.
* \*\*MedXpertQA (OOD)\*\*에서도 성능이 크게 향상되어, 멀티모달 설정에서는 67.5%까지 도달해 대형 API 모델과 유사한 성능을 보임.
* **인사이트**: 의료 QA에서도 특화 파운데이션 모델이 일반 모델보다 훨씬 우수한 추론력을 갖춤.

---

** 표 7\~8: 의료 이미지 분류 성능 (제로샷)**

* 흉부 X-ray (MIMIC-CXR, CheXpert, CXR14)에서 **macro F1 88.9%** 등 뛰어난 제로샷 성능 기록.
* 피부/망막/병리 이미지를 대상으로 한 MCQA 평가에서도 Gemma 3 대비 20\~50% 정확도 향상.
* **인사이트**: 훈련 없이도 이미지 진단 작업에 강력한 성능을 발휘할 수 있어, 의료현장 적용 가능성 ↑

---

** 그림 2\~3: VQA 및 병리 이미지 자유응답 예시**

* 흉부 X-ray에 대한 “가장 가능성 있는 이상 소견은?” 질문에 대해 MedGemma는 “좌측 기흉과 chest tube”와 같이 자연스럽고 임상적으로 정확한 답변 생성.
* 병리 이미지에 대해 “이 조직의 가장 가능성 높은 진단은?” 질문에 대해 “Colorectal adenocarcinoma”처럼 전문가 수준의 진단 응답 제공.
* **인사이트**: 의료 이미지 + 자연어 자유응답을 결합한 상호작용이 가능함을 시각적으로 보여줌.

---

** 표 13\~14: 파인튜닝 실험 결과**

* MIMIC-CXR 보고서 생성은 RadGraph F1 기준 **29.5 → 30.3**, 병리 분류는 **94.5%**, 전자의무기록 QA는 \*\*93.6%\*\*까지 향상됨.
* **인사이트**: 의료 특화 모델은 적은 데이터와 파인튜닝만으로도 SOTA 성능에 근접 가능.

---

** MedSigLIP (표 15\~16, 그림 5)**

* CheXpert, CXR14에서 **HAI-DEF 대비 +2% AUC** 개선, fracture 항목은 **+7.1%** 향상.
* Dermatology, Histopathology에서도 기존 전문 인코더보다 높은 linear probe 성능 기록.
* **인사이트**: MedSigLIP은 다양한 의료 이미지 도메인에서 통합적이고 강력한 비전 인코더 역할 수행.

---


** Tables 3–4: Medical QA Performance**

* MedGemma 27B achieves **87.7% on MedQA**, **76.8% on PubMedQA**, with 10–20% gains over Gemma 3.
* On **MedXpertQA (OOD)**, performance reaches **67.5%** (multimodal), comparable to large proprietary models.
* **Insight**: Medically-tuned models greatly surpass general models in reasoning accuracy on medical QA tasks.

---

** Tables 7–8: Medical Image Classification (Zero-shot)**

* Achieves **macro F1 of 88.9%** on MIMIC-CXR and high scores across CheXpert, CXR14, dermatology, and pathology datasets.
* Up to **20–50% accuracy gains** over Gemma 3 in dermatology and histopathology MCQA.
* **Insight**: MedGemma exhibits strong diagnostic potential with zero-shot inference, promising for clinical deployment.

---

** Figures 2–3: VQA and Histopathology Freeform Examples**

* For chest X-rays, MedGemma outputs responses like “Large pneumothorax with chest tube,” aligned with radiologist interpretations.
* For pathology slides, it identifies diagnoses like “Colorectal adenocarcinoma” or “High-grade invasive carcinoma.”
* **Insight**: Demonstrates interactive, natural-language understanding of complex medical images.

---

**Tables 13–14: Fine-tuning Experiments**

* Report generation improved (RadGraph F1 **29.5 → 30.3**), histopathology reached **94.5%**, and EHR QA improved to **93.6%** accuracy.
* **Insight**: With minimal fine-tuning, MedGemma can approach or surpass SOTA in specialized tasks.

---

** MedSigLIP (Tables 15–16, Figure 5)**

* Outperforms HAI-DEF models on CheXpert (+2.0% AUC, +7.1% for fracture), dermatology, and pathology.
* **Insight**: MedSigLIP serves as a unified and powerful vision encoder across multiple medical domains.

---



<br/>
# refer format:     



@misc{sellergren2025medgemma,
  title={MedGemma Technical Report},
  author={Andrew Sellergren and Sahar Kazemzadeh and Tiam Jaroensri and Atilla Kiraly and Madeleine Traverse and Timo Kohlberger and Shawn Xu and Fayaz Jamil and C{\'i}an Hughes and Charles Lau and Justin Chen and Fereshteh Mahvar and Liron Yatziv and Tiffany Chen and Bram Sterling and Stefanie Anna Baby and Susanna Maria Baby and Jeremy Lai and Samuel Schmidgall and Lu Yang and Kejia Chen and Per Bjornsson and Shashir Reddy and Ryan Brush and Kenneth Philbrick and Mercy Asiedu and Ines Mezerreg and Howard Hu and Howard Yang and Richa Tiwari and Sunny Jansen and Preeti Singh and Yun Liu and Shekoofeh Azizi and Aishwarya Kamath and Johan Ferret and Shreya Pathak and Nino Vieillard and Ramona Merhej and Sarah Perrin and Tatiana Matejovicova and Alexandre Ram{\'e} and Morgane Riviere and Louis Rouillard and Thomas Mesnard and Geoffrey Cideron and Jean-bastien Grill and Sabela Ramos and Edouard Yvinec and Michelle Casbon and Elena Buchatskaya and Jean-Baptiste Alayrac and Dmitry Lepikhin and Vlad Feinberg and Sebastian Borgeaud and Alek Andreev and Cassidy Hardin and Robert Dadashi and L{\'e}onard Hussenot and Armand Joulin and Olivier Bachem and Yossi Matias and Katherine Chou and Avinatan Hassidim and Kavi Goel and Clement Farabet and Joelle Barral and Tris Warkentin and Jonathon Shlens and David Fleet and Victor Cotruta and Omar Sanseviero and Gus Martins and Phoebe Kirk and Anand Rao and Shravya Shetty and David F. Steiner and Can Kirmizibayrak and Rory Pilgrim and Daniel Golden and Lin Yang},
  year={2025},
  eprint={2507.05201},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://doi.org/10.48550/arXiv.2507.05201}
} 







Sellergren, Andrew, Sahar Kazemzadeh, Tiam Jaroensri, Atilla Kiraly, Madeleine Traverse, Timo Kohlberger, Shawn Xu, et al. "MedGemma Technical Report." arXiv preprint arXiv:2507.05201, July 12, 2025. https://doi.org/10.48550/arXiv.2507.05201.  





