---
layout: post
title:  "[2024]MiDRED: An Annotated Corpus for Microbiome Knowledge Base Construction"  
date:   2025-03-15 11:24:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

미생물-질병 관계 데이터셋 제안  



짧은 요약(Abstract) :    




미생물과 질병 간의 상호작용은 최근 저비용, 고정확도의 시퀀싱 기술 발전에 따라 중요한 연구 분야로 부각되고 있다. 이를 체계적으로 정리하기 위해 전문가들은 학술 논문에서 미생물-질병 연관성을 수작업으로 추출하여 지식 베이스를 구축하지만, 연구 논문의 급격한 증가 속도를 따라가기 어렵다. 기존의 관계 추출(Relation Extraction) 기술은 다양한 분야에서 성공적으로 활용되고 있지만, 미생물 연구 분야에서는 이를 지원하는 데이터셋이 부족한 실정이다.  
이에 우리는 **MiDRED(Microbe-Disease Relation Extraction Dataset)**을 소개한다. MiDRED는 3,116개의 미생물-질병 간 관계를 포함하는 인간이 직접 주석한 데이터셋으로, 미생물과 질병 간의 정밀한 관계를 나타낸다. 또한, 이 데이터셋은 기존의 주요 미생물-질병 지식 베이스(KB)와의 호환성을 고려하여 설계되었으며, 모델 학습 시 긍정 편향(positive bias)을 방지하기 위해 ‘연관 없음(no relation)’과 같은 부정적 사례도 포함하고 있다. MiDRED는 생명과학 및 자연어 처리(NLP) 연구자들이 미생물 관련 텍스트 마이닝 솔루션을 개발하고, 미생물 지식 베이스를 자동으로 구축 및 유지하는 데 도움을 줄 것으로 기대된다.

---


The interplay between microbiota and diseases has emerged as a significant area of research facilitated by the proliferation of cost-effective and precise sequencing technologies. To keep track of the many findings, domain experts manually review publications to extract reported microbe-disease associations and compile them into knowledge bases. However, manual curation efforts struggle to keep up with the pace of publications.  
Relation extraction has demonstrated remarkable success in other domains, yet the availability of datasets supporting such methods within the domain of microbiome research remains limited. To bridge this gap, we introduce the **Microbe-Disease Relation Extraction Dataset (MiDRED)**; a human-annotated dataset containing 3,116 annotations of fine-grained relationships between microbes and diseases. We hope this dataset will help address the scarcity of data in this crucial domain and facilitate the development of advanced text-mining solutions to automate the creation and maintenance of microbiome knowledge bases.



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






MiDRED 데이터셋을 구축하기 위해 다음과 같은 방법론이 사용되었다.

#### **1. 데이터 수집 및 정규화 (Data Collection and Entity Normalization)**
MiDRED는 **PubMed**에서 제공하는 논문 초록을 기반으로 생성되었으며, PubTator 도구를 활용하여 관련성이 높은 문서를 선별하였다. 특히, **Disbiome** 데이터베이스에 등록된 문헌을 우선적으로 고려하여 미생물 연구에 적합한 논문을 선택했다.  
- **미생물 엔티티 정규화**: LPSN (*List of Prokaryotic Names with Standing in Nomenclature*)을 사용하여 표준 미생물 명칭으로 정규화.  
- **질병 엔티티 정규화**: CTD (*Comparative Toxicogenomics Database*)를 활용하여 표준 질병 명칭으로 변환.  
이러한 과정을 통해 데이터의 일관성을 유지하고, 기존의 주요 생물학적 지식 베이스와의 호환성을 높였다.

#### **2. 관계 주석 (Relation Annotation)**
MiDRED는 주요 미생물-질병 지식 베이스(KBs)와의 호환성을 고려하여 **네 가지 관계 클래스**를 정의하였다.  
- **Connecting**: 미생물이 질병과 연관되거나 위험 요인으로 작용하는 경우.  
- **Contrasting**: 미생물이 질병의 개선에 기여하는 경우.  
- **Pathogen**: 미생물이 특정 질병의 원인 병원체로 작용하는 경우.  
- **No Relation**: 미생물과 질병 간에 관계가 없는 경우.  

각 관계 유형의 신뢰도를 높이기 위해 **이중 주석(double annotation)**을 적용하고, 주석 불일치가 발생한 경우 **세 번째 검토(third round annotation)**를 통해 조정했다. 결과적으로 **Fleiss’ Kappa 값이 0.710**으로 나타나 높은 주석 일관성을 확보하였다.

#### **3. 데이터 분할 및 모델 평가 (Data Splitting and Model Evaluation)**
MiDRED는 **1,655개 문서에서 5,590개의 미생물 엔티티, 6,437개의 질병 엔티티, 총 3,116개의 관계**를 포함하며, 다음과 같은 비율로 데이터셋을 분할하였다.
- **Train set**: 80% (2,169 관계)
- **Development set**: 10% (447 관계)
- **Test set**: 10% (500 관계)

데이터셋은 **사실(triple) 단위**로 나누어 훈련, 검증, 테스트 간에 중복되지 않도록 설계되었다.

#### **4. 베이스라인 모델 및 실험 (Baseline Models and Experiments)**
MiDRED의 성능을 검증하기 위해 **개체명 인식(NER)과 관계 추출(RE) 실험**을 수행했다.

- **Named Entity Recognition (NER) 모델**  
  - BiLSTM-CRF  
  - BioBERT-CRF  
  - PubMedBERT-CRF  
  - **결과**: BioBERT-CRF가 F1-score 0.969로 가장 높은 성능을 보임.

- **Relation Extraction (RE) 모델**  
  - **Fine-tuned Biomedical Language Models**  
    - BioLinkBERT  
    - PubMedBERT  
  - **Large Language Models (LLMs)**  
    - GPT-3.5  
    - GPT-4  
  - **결과**: BioLinkBERT가 F1-score 0.905로 가장 높은 성능을 기록했으며, LLM 기반 모델(GPT-3.5, GPT-4)은 상대적으로 성능이 낮았다.


MiDRED는 미생물-질병 관계 분석을 위한 **최초의 대규모 데이터셋**으로, 실험 결과 최신 바이오메디컬 언어 모델들이 높은 성능을 보였다. 그러나 드문 미생물-질병 관계에 대한 일반화 성능이 낮다는 한계를 지니며, 향후 연구에서는 **희소 클래스(sparse class) 문제 해결**과 **추가적인 데이터 확장**이 필요할 것으로 보인다.

---



#### **1. Data Collection and Entity Normalization**
The MiDRED dataset was constructed by collecting abstracts from **PubMed**, utilizing the **PubTator** tool to prioritize relevant studies, especially those indexed in the **Disbiome database**.  
- **Microbe Entity Normalization**: Standardized using the **List of Prokaryotic Names with Standing in Nomenclature (LPSN)**.  
- **Disease Entity Normalization**: Standardized using the **Comparative Toxicogenomics Database (CTD)**.  
This approach ensures consistency in entity representation and enhances compatibility with existing microbiome knowledge bases.

#### **2. Relation Annotation**
To align MiDRED with major microbiome knowledge bases, the dataset defines **four relation classes**:  
- **Connecting**: The microbe is associated with or acts as a risk factor for the disease.  
- **Contrasting**: The microbe contributes to improving or mitigating the disease.  
- **Pathogen**: The microbe is a pathogenic agent of the disease.  
- **No Relation**: No known relationship between the microbe and the disease.  

Annotations were performed using a **double-annotation process**, with a third round for conflict resolution. The dataset achieved a **high inter-annotator agreement (Fleiss’ Kappa = 0.710)**, indicating strong annotation consistency.

#### **3. Data Splitting and Model Evaluation**
MiDRED consists of **1,655 documents, 5,590 microbial entities, 6,437 disease entities, and 3,116 annotated relations**. The dataset was split as follows:
- **Training set**: 80% (2,169 relations)
- **Development set**: 10% (447 relations)
- **Test set**: 10% (500 relations)

The dataset was **split by unique fact triples** to prevent data leakage between training, validation, and test sets.

#### **4. Baseline Models and Experiments**
Experiments were conducted on **Named Entity Recognition (NER) and Relation Extraction (RE) tasks**.

- **NER Models**  
  - BiLSTM-CRF  
  - BioBERT-CRF  
  - PubMedBERT-CRF  
  - **Results**: BioBERT-CRF achieved the highest F1-score of **0.969**.

- **RE Models**  
  - **Fine-tuned Biomedical Language Models**  
    - BioLinkBERT  
    - PubMedBERT  
  - **Large Language Models (LLMs)**  
    - GPT-3.5  
    - GPT-4  
  - **Results**: BioLinkBERT achieved the highest F1-score of **0.905**, while LLM-based models (GPT-3.5, GPT-4) underperformed.


MiDRED represents the **first large-scale dataset** dedicated to **microbe-disease relation extraction**. While state-of-the-art biomedical language models demonstrated strong performance, challenges remain in generalizing to **rare microbe-disease associations**. Future work should focus on **addressing class imbalance** and **expanding the dataset** for improved performance in long-tail distributions.



   
 
<br/>
# Results  




MiDRED의 성능을 평가하기 위해 다양한 **경쟁 모델(Baseline Models)**을 활용하여 **개체명 인식(NER)과 관계 추출(RE) 실험**을 수행하였다. 실험은 **테스트 데이터셋**을 기반으로 수행되었으며, **Precision (정밀도), Recall (재현율), F1-score** 등의 평가 메트릭을 활용하여 성능을 비교하였다.

---

### **1. 개체명 인식 (NER) 실험 결과**
#### **① 테스트 데이터**
MiDRED의 테스트 데이터셋에는 **1,590개의 개체명(721개 미생물, 869개 질병 엔티티)**가 포함되어 있으며, 모델은 이 데이터를 기반으로 개체명을 예측하였다.

#### **② 평가 메트릭**
NER 모델의 성능은 **Precision (정밀도), Recall (재현율), F1-score**를 기준으로 평가되었다.

#### **③ 경쟁 모델 및 성능 비교**
- **BiLSTM-CRF**  
  - Precision: **0.877**  
  - Recall: **0.891**  
  - F1-score: **0.884**  
- **PubMedBERT-CRF**  
  - Precision: **0.947**  
  - Recall: **0.972**  
  - F1-score: **0.959**  
- **BioBERT-CRF**  
  - Precision: **0.957**  
  - Recall: **0.981**  
  - F1-score: **0.969**  

#### **④ 분석**
- **BioBERT-CRF** 모델이 **F1-score 0.969**로 가장 높은 성능을 기록.
- PubMedBERT-CRF 모델도 유사한 성능을 보였으며, 기존의 BiLSTM-CRF보다 **NER 성능이 크게 향상됨**.

---

### **2. 관계 추출 (RE) 실험 결과**
#### **① 테스트 데이터**
MiDRED의 테스트 데이터셋에는 **500개의 관계 인스턴스(Connecting: 272, Contrasting: 32, Pathogen: 153, No relation: 43)**가 포함되어 있다.

#### **② 평가 메트릭**
RE 모델의 성능은 **Precision (정밀도), Recall (재현율), F1-score**로 평가되었으며, 특히 **상위 25% 빈도수 내 미생물과 비교적 드문 미생물 간의 관계 추출 성능 차이**도 분석되었다.

#### **③ 경쟁 모델 및 성능 비교**
- **Fine-tuned Biomedical Language Models**  
  - **PubMedBERT-base**  
    - Precision: **0.867**  
    - Recall: **0.855**  
    - F1-score: **0.861**  
  - **BioLinkBERT-large**  
    - Precision: **0.907**  
    - Recall: **0.904**  
    - F1-score: **0.905**  

- **Large Language Models (LLMs)**  
  - **GPT-3.5**  
    - Precision: **0.542**  
    - Recall: **0.562**  
    - F1-score: **0.552**  
  - **GPT-4**  
    - Precision: **0.716**  
    - Recall: **0.725**  
    - F1-score: **0.721**  

#### **④ 분석**
- **BioLinkBERT-large 모델이 F1-score 0.905로 가장 높은 성능**을 보였으며, **PubMedBERT-base보다 우수한 결과**를 나타냄.
- **GPT 기반 LLM 모델(GPT-3.5, GPT-4)은 상대적으로 낮은 성능을 기록**, 특히 GPT-3.5의 F1-score는 0.552로 미미한 수준.
- **미생물 빈도수에 따른 성능 차이**를 분석한 결과, BiLSTM-CRF 및 BioLinkBERT 모델은 **상위 25% 빈도수 내 미생물 데이터에서 더 높은 성능을 보였으나, 드문 미생물 데이터에서는 성능이 감소**하는 패턴을 나타냄.

---

### **3. 결론 및 향후 연구 방향**
- **최신 바이오메디컬 언어 모델(BioBERT, BioLinkBERT)이 전통적인 BiLSTM-CRF보다 높은 성능을 기록**하여, 미생물-질병 관계 추출에 효과적인 것으로 나타남.
- **LLM 기반 모델(GPT-3.5, GPT-4)은 비교적 낮은 성능을 보였으며, 현 시점에서는 BioLinkBERT 같은 특화 모델이 더 적합**.
- **드문 미생물-질병 관계에 대한 일반화 성능이 부족**하므로, 향후 연구에서는 **데이터 확장 및 불균형 해소 전략**이 필요할 것으로 예상됨.

---



To evaluate the performance of MiDRED, we conducted **Named Entity Recognition (NER) and Relation Extraction (RE) experiments** using various **baseline models**. The models were tested on the **test dataset**, and performance was measured using **Precision, Recall, and F1-score**.

---

### **1. Named Entity Recognition (NER) Results**
#### **① Test Dataset**
The MiDRED test set includes **1,590 entity mentions (721 microbes, 869 diseases)**.

#### **② Evaluation Metrics**
NER models were evaluated using **Precision, Recall, and F1-score**.

#### **③ Baseline Models & Performance**
- **BiLSTM-CRF**  
  - Precision: **0.877**  
  - Recall: **0.891**  
  - F1-score: **0.884**  
- **PubMedBERT-CRF**  
  - Precision: **0.947**  
  - Recall: **0.972**  
  - F1-score: **0.959**  
- **BioBERT-CRF**  
  - Precision: **0.957**  
  - Recall: **0.981**  
  - F1-score: **0.969**  

#### **④ Analysis**
- **BioBERT-CRF achieved the highest F1-score of 0.969**.
- PubMedBERT-CRF also performed well, outperforming BiLSTM-CRF by a significant margin.

---

### **2. Relation Extraction (RE) Results**
#### **① Test Dataset**
The MiDRED test set includes **500 relation instances** across four categories:  
- Connecting: **272**  
- Contrasting: **32**  
- Pathogen: **153**  
- No relation: **43**  

#### **② Evaluation Metrics**
RE models were evaluated based on **Precision, Recall, and F1-score**, with additional analysis on **performance across frequent vs. rare microbes**.

#### **③ Baseline Models & Performance**
- **Fine-tuned Biomedical Language Models**  
  - **PubMedBERT-base**  
    - Precision: **0.867**  
    - Recall: **0.855**  
    - F1-score: **0.861**  
  - **BioLinkBERT-large**  
    - Precision: **0.907**  
    - Recall: **0.904**  
    - F1-score: **0.905**  

- **Large Language Models (LLMs)**  
  - **GPT-3.5**  
    - Precision: **0.542**  
    - Recall: **0.562**  
    - F1-score: **0.552**  
  - **GPT-4**  
    - Precision: **0.716**  
    - Recall: **0.725**  
    - F1-score: **0.721**  

#### **④ Analysis**
- **BioLinkBERT-large achieved the highest F1-score (0.905)**, outperforming **PubMedBERT-base**.
- **GPT-based LLMs (GPT-3.5, GPT-4) showed relatively poor performance**, with **GPT-3.5 achieving only 0.552 F1-score**.
- **Performance varied based on microbe frequency**, with rare microbes leading to **lower accuracy in RE tasks**.

---

### **3. Conclusion & Future Work**
- **Biomedical models (BioBERT, BioLinkBERT) significantly outperformed BiLSTM-CRF and LLMs in microbe-disease relation extraction**.
- **LLMs (GPT-3.5, GPT-4) underperformed**, suggesting that **domain-specific models are more suitable** for this task.
- **Future research should focus on mitigating class imbalance** and **expanding data coverage for rare microbe-disease associations**.






<br/>
# 예제  




MiDRED는 **관계 추출(Relation Extraction, RE)**과 **개체명 인식(Named Entity Recognition, NER)**을 포함한 다양한 생명과학 NLP 작업을 지원하는 데이터셋이다. 여기에서는 **트레이닝 데이터와 테스트 데이터의 구조**, 그리고 **각 태스크(Task)의 입력 및 출력 예제**를 설명한다.

---

### **1. 트레이닝 데이터 (Training Data)**
MiDRED의 트레이닝 데이터는 PubMed 논문의 문장에서 **미생물(Microbe)과 질병(Disease)의 관계를 주석**한 것이다. 각 문장은 다음과 같은 정보로 구성된다.

#### **트레이닝 데이터 예제**
```json
{
  "sentence": "Helicobacter pylori infection is strongly associated with gastric cancer.",
  "head_entity": "Helicobacter pylori",
  "tail_entity": "gastric cancer",
  "relation": "Connecting"
}
```
- **sentence**: 원문 문장  
- **head_entity**: 문장에서의 미생물 개체명  
- **tail_entity**: 문장에서의 질병 개체명  
- **relation**: 미생물과 질병 간의 관계 클래스 (Connecting, Contrasting, Pathogen, No relation 중 하나)

---

### **2. 테스트 데이터 (Test Data)**
테스트 데이터는 트레이닝 데이터와 동일한 구조로 이루어지며, 모델이 학습되지 않은 새로운 문장에 대해 관계를 예측하는 데 사용된다.

#### **테스트 데이터 예제**
```json
{
  "sentence": "Lactobacillus reuteri was shown to reduce inflammation in patients with colitis.",
  "head_entity": "Lactobacillus reuteri",
  "tail_entity": "colitis",
  "relation": "Contrasting"
}
```
- 이 경우, **Lactobacillus reuteri가 colitis(대장염)를 완화하는 효과가 있음**을 나타내므로 관계 클래스는 **Contrasting**이다.

---

### **3. 태스크(Task) 및 입력/출력 예제**
MiDRED는 **두 가지 주요 NLP 태스크**를 지원한다.

#### **① 개체명 인식 (NER: Named Entity Recognition)**
**입력(Input):**  
```text
"Recent studies show that Helicobacter pylori infection can cause gastric ulcers."
```
**출력(Output):**  
```json
[
  {"entity": "Helicobacter pylori", "type": "Microbe", "start": 23, "end": 43},
  {"entity": "gastric ulcers", "type": "Disease", "start": 65, "end": 79}
]
```
- 모델은 **문장에서 미생물(Microbe)과 질병(Disease) 개체명을 정확히 추출**해야 한다.

---

#### **② 관계 추출 (RE: Relation Extraction)**
**입력(Input):**  
```json
{
  "sentence": "Recent studies show that Helicobacter pylori infection can cause gastric ulcers.",
  "head_entity": "Helicobacter pylori",
  "tail_entity": "gastric ulcers"
}
```
**출력(Output):**  
```json
{"relation": "Pathogen"}
```
- 이 경우, **Helicobacter pylori가 gastric ulcers(위궤양)의 원인이므로 관계는 "Pathogen"으로 분류**된다.

---

### **4. 태스크별 평가 기준 (Evaluation Metrics)**
각 태스크의 성능은 다음 평가 지표를 활용하여 측정된다.

| 태스크 | 평가 메트릭 |
|--------|------------|
| 개체명 인식 (NER) | Precision, Recall, F1-score |
| 관계 추출 (RE) | Precision, Recall, F1-score |

NER 태스크에서는 **개체명을 정확히 찾아내는 성능(Precision, Recall, F1-score)**이 중요하며, RE 태스크에서는 **미생물과 질병 간 관계를 얼마나 정확하게 예측하는지**가 중요하다.

---


- **MiDRED는 미생물-질병 관계 분석을 위한 정제된 데이터셋**으로, **생명과학 NLP 연구자들이 활용할 수 있도록 관계 주석이 명확히 포함됨**.
- **NER 태스크는 미생물과 질병 개체명을 정확히 인식하는 것**이 목표이며, **RE 태스크는 미생물과 질병 간의 관계를 분류하는 것**이 목표임.
- 모델의 성능은 **Precision, Recall, F1-score**를 기준으로 평가되며, **NER와 RE 태스크 모두 높은 성능을 보이는 모델이 필요**.

---



MiDRED is a **biomedical NLP dataset** designed for **Relation Extraction (RE) and Named Entity Recognition (NER)**. Below, we describe the **training data, test data, task structure, and input-output examples**.

---

### **1. Training Data**
The MiDRED training dataset consists of sentences annotated with **microbe-disease relations** extracted from PubMed articles.

#### **Training Data Example**
```json
{
  "sentence": "Helicobacter pylori infection is strongly associated with gastric cancer.",
  "head_entity": "Helicobacter pylori",
  "tail_entity": "gastric cancer",
  "relation": "Connecting"
}
```
- **sentence**: The original text sentence  
- **head_entity**: The microbe entity in the sentence  
- **tail_entity**: The disease entity in the sentence  
- **relation**: The relationship class (one of Connecting, Contrasting, Pathogen, No relation)

---

### **2. Test Data**
Test data follows the same format as the training set but contains unseen sentences for model evaluation.

#### **Test Data Example**
```json
{
  "sentence": "Lactobacillus reuteri was shown to reduce inflammation in patients with colitis.",
  "head_entity": "Lactobacillus reuteri",
  "tail_entity": "colitis",
  "relation": "Contrasting"
}
```
- Here, **Lactobacillus reuteri helps reduce inflammation in colitis**, making the relation **Contrasting**.

---

### **3. Tasks & Input/Output Examples**
MiDRED supports **two primary NLP tasks**.

#### **① Named Entity Recognition (NER)**
**Input:**  
```text
"Recent studies show that Helicobacter pylori infection can cause gastric ulcers."
```
**Output:**  
```json
[
  {"entity": "Helicobacter pylori", "type": "Microbe", "start": 23, "end": 43},
  {"entity": "gastric ulcers", "type": "Disease", "start": 65, "end": 79}
]
```
- The model must **correctly extract microbe and disease entities from the text**.

---

#### **② Relation Extraction (RE)**
**Input:**  
```json
{
  "sentence": "Recent studies show that Helicobacter pylori infection can cause gastric ulcers.",
  "head_entity": "Helicobacter pylori",
  "tail_entity": "gastric ulcers"
}
```
**Output:**  
```json
{"relation": "Pathogen"}
```
- In this case, **Helicobacter pylori is a known pathogen causing gastric ulcers**, so the relation is classified as **"Pathogen"**.

---

### **4. Task Evaluation Metrics**
Each task's performance is measured using the following evaluation metrics:

| Task | Evaluation Metric |
|------|------------------|
| Named Entity Recognition (NER) | Precision, Recall, F1-score |
| Relation Extraction (RE) | Precision, Recall, F1-score |

NER performance is judged based on **how accurately entities are extracted**, while RE performance is measured by **how well relationships between microbes and diseases are predicted**.

---


- **MiDRED is a high-quality dataset for microbe-disease relation extraction**, providing clear annotations for NLP research.  
- **The NER task focuses on entity detection, while the RE task classifies microbe-disease relationships**.  
- **Performance is evaluated using Precision, Recall, and F1-score, emphasizing high accuracy in both tasks**.



<br/>  
# 요약   




MiDRED는 미생물-질병 관계 추출을 위해 PubMed 논문에서 개체명을 정규화하고, 관계 유형(Connecting, Contrasting, Pathogen, No relation)을 주석하여 3,116개의 관계 데이터를 구축하였다. 실험 결과, **BioLinkBERT-large가 F1-score 0.905로 가장 높은 성능**을 기록했으며, **GPT-3.5 및 GPT-4와 같은 LLM 모델은 상대적으로 낮은 성능**을 보였다. 테스트 데이터에서 개체명 인식(NER)과 관계 추출(RE) 태스크를 수행한 결과, **BioBERT-CRF가 NER에서 F1-score 0.969, BioLinkBERT가 RE에서 0.905를 기록**하여 특화된 바이오메디컬 모델이 효과적임을 확인했다.

---


MiDRED was constructed by normalizing entity names from PubMed articles and annotating relationship types (Connecting, Contrasting, Pathogen, No relation), resulting in **3,116 microbe-disease relation instances**. Experimental results showed that **BioLinkBERT-large achieved the highest F1-score of 0.905**, while **LLM models like GPT-3.5 and GPT-4 underperformed**. On the test dataset, **BioBERT-CRF achieved an F1-score of 0.969 for NER, and BioLinkBERT reached 0.905 for RE**, demonstrating the effectiveness of specialized biomedical models.


<br/>  
# 기타  





MiDRED 논문에는 **모델 아키텍처, 데이터셋 구조, 실험 결과를 시각적으로 표현한 여러 피규어(Figure), 다이어그램(Diagram), 테이블(Table), 그리고 어펜딕스(Appendix)**가 포함되어 있다.

---

### **1. 주요 피규어 및 다이어그램**
#### **① MiDRED 데이터셋 구축 과정 (Figure 1)**
- PubMed 논문에서 **미생물(Microbe)과 질병(Disease) 개체를 정규화하고, 관계(Relation)를 주석**하는 과정을 도식화.
- 개체명 인식(NER) 및 관계 추출(RE)을 위한 **텍스트 주석(annotation) 단계**가 포함됨.

#### **② 관계 추출(Relation Extraction) 모델 아키텍처 (Figure 2)**
- BioBERT, BioLinkBERT, PubMedBERT와 같은 **바이오메디컬 언어 모델의 학습 구조**를 보여줌.
- 모델 입력(Input), 인코딩(Encoding), 관계 예측(Relation Classification)의 **단계별 데이터 흐름**이 포함됨.

---

### **2. 주요 테이블**
#### **① MiDRED 데이터셋 통계 (Table 1)**
- MiDRED 데이터셋의 **문서 수, 개체명(Entity) 개수, 관계(Relation) 개수**를 정리.
- 데이터셋이 학습(train), 개발(dev), 테스트(test) 세트로 **어떻게 분할**되었는지를 표로 제시.

#### **② 개체명 인식(NER) 성능 비교 (Table 2)**
- **BiLSTM-CRF, BioBERT-CRF, PubMedBERT-CRF** 등 여러 모델의 **Precision, Recall, F1-score**를 비교.
- BioBERT-CRF가 **F1-score 0.969**로 가장 높은 성능을 기록.

#### **③ 관계 추출(RE) 성능 비교 (Table 3)**
- **BioLinkBERT-large, PubMedBERT-base, GPT-3.5, GPT-4** 등의 모델 성능을 정리.
- BioLinkBERT-large가 **F1-score 0.905로 최고 성능을 기록**, GPT 모델들은 상대적으로 낮은 성능을 보임.

---

### **3. 어펜딕스 (Appendices)**
- **데이터셋 구축 세부 사항**: 미생물 및 질병 개체의 정규화 과정, 주석 가이드라인.
- **추가 실험 결과**: RE 모델의 세부적인 하이퍼파라미터 튜닝, 오류 분석.
- **데이터 예제**: MiDRED의 문장별 관계 주석 예시를 포함.

---


MiDRED 논문에는 **데이터셋 구축, 모델 성능, 추가 분석을 위한 다양한 테이블과 다이어그램이 포함**되어 있으며, 어펜딕스에서는 데이터 주석 과정 및 실험 세부 내용을 제공하여 연구자들이 MiDRED를 활용할 수 있도록 지원한다.

---



The MiDRED paper includes **figures, diagrams, tables, and appendices** that visually represent the **model architecture, dataset structure, and experimental results**.

---

### **1. Key Figures and Diagrams**
#### **① MiDRED Dataset Construction Process (Figure 1)**
- A visual representation of how **microbe and disease entities are normalized and annotated** from PubMed articles.
- Includes the annotation process for **Named Entity Recognition (NER) and Relation Extraction (RE)**.

#### **② Relation Extraction Model Architecture (Figure 2)**
- Shows the **training structure of biomedical language models** such as BioBERT, BioLinkBERT, and PubMedBERT.
- Illustrates **data flow through input, encoding, and relation classification stages**.

---

### **2. Key Tables**
#### **① MiDRED Dataset Statistics (Table 1)**
- Summarizes the **number of documents, entities, and relations** in the dataset.
- Provides a breakdown of how the dataset is **split into training, development, and test sets**.

#### **② NER Performance Comparison (Table 2)**
- Compares **Precision, Recall, and F1-score** for **BiLSTM-CRF, BioBERT-CRF, and PubMedBERT-CRF**.
- **BioBERT-CRF achieves the highest F1-score of 0.969**.

#### **③ RE Performance Comparison (Table 3)**
- Summarizes model performance for **BioLinkBERT-large, PubMedBERT-base, GPT-3.5, and GPT-4**.
- **BioLinkBERT-large achieves the highest F1-score of 0.905**, while **GPT models perform relatively poorly**.

---

### **3. Appendices**
- **Dataset Construction Details**: Normalization of microbe and disease entities, annotation guidelines.
- **Additional Experimental Results**: Hyperparameter tuning and error analysis for RE models.
- **Dataset Examples**: Sentence-level relation annotation samples from MiDRED.

---


The MiDRED paper **includes various tables and diagrams** to illustrate **dataset construction, model performance, and additional analysis**, while the appendices provide **detailed dataset annotations and experimental insights** for researchers using MiDRED.


<br/>
# refer format:     


@inproceedings{Hogan2024MiDRED,
  author    = {William Hogan and Andrew Bartko and Jingbo Shang and Chun-Nan Hsu},
  title     = {MiDRED: An Annotated Corpus for Microbiome Knowledge Base Construction},
  booktitle = {Proceedings of the 23rd Workshop on Biomedical Language Processing},
  pages     = {398--408},
  year      = {2024},
  publisher = {Association for Computational Linguistics},
  address   = {La Jolla, CA},
  month     = {August},
  url       = {https://huggingface.co/datasets/shangdatalab-ucsd/midred}
}



William Hogan, Andrew Bartko, Jingbo Shang, and Chun-Nan Hsu. "MiDRED: An Annotated Corpus for Microbiome Knowledge Base Construction." In Proceedings of the 23rd Workshop on Biomedical Language Processing, 398–408. La Jolla, CA: Association for Computational Linguistics, August 2024.





