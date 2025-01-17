---
layout: post
title:  "[2024]Mitigating Social Biases in Text-to-Image Diffusion Models via Linguistic-Aligned Attention Guidance"  
date:   2025-01-17 17:27:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

디퓨전 모델류들의 편향 경감을 위해 프롬프팅을 사전에 조정하도록 한 듯?  

짧은 요약(Abstract) :    




이 논문은 텍스트에서 이미지를 생성하는 모델이 가지는 사회적 편향 문제를 완화하는 새로운 방법론을 제안합니다. 텍스트-이미지 생성 모델은 강력한 생성 능력을 보여주지만, 종종 성별과 인종 등에서 사회적 편향을 드러냅니다. 특히, 다수 인물이 포함된 이미지를 생성할 때 단일한 인종이나 성별로 구성되는 문제가 발생하며, 이는 사회적 다양성을 해칠 수 있습니다.

이를 해결하기 위해, 저자들은 **Linguistic-aligned Attention Guidance(언어 정렬 주의 지침)**이라는 모듈과 **공정한 추론(Fair Inference)** 방법을 제안했습니다. 이 방법은 생성된 이미지와 텍스트 프롬프트 간의 의미적 연관성을 활용하여 편향의 원인을 식별하고 이를 완화합니다. 텍스트 프롬프트 내 특정 단어(예: "faces", "doctors")가 어떤 사회적 그룹과 연관될 수 있는지를 분석한 후, 해당 단어와 관련된 의미적 영역을 찾아 공정한 속성을 부여하는 방식으로 작업합니다.

결과적으로, 이 방법은 구조적 및 의미적 정보를 보존하면서도 공정한 속성을 생성할 수 있으며, 기존의 방법들과 비교했을 때 여러 상황에서 우수한 성과를 보였습니다.

---


Recent advancements in text-to-image generative models have showcased remarkable capabilities across various tasks. However, these powerful models have revealed the inherent risks of social biases, such as propagating distorted real-world perspectives and unforeseen prejudice. Existing debiasing methods are primarily designed for scenarios with a single individual in the image, often producing homogenous race or gender attributes in multi-individual scenarios, thereby harming social group diversity.

To address this issue, the authors propose a novel method leveraging **Linguistic-aligned Attention Guidance** and **Fair Inference**. This approach identifies semantic regions associated with biased tokens (e.g., "faces," "doctors") and applies fair attribute generation within these regions. By aligning linguistic and attention-based cues, the method mitigates biases while preserving the original structural and semantic integrity of the generated images.

Extensive experiments and analyses demonstrate that this method outperforms existing debiasing techniques in generating fair and diverse images, particularly in scenarios involving multiple individuals.



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



### 메서드 (Proposed Method)
이 논문은 **언어 정렬 주의 지침(Linguistic-aligned Attention Guidance)**과 **공정한 추론(Fair Inference)**이라는 두 가지 주요 구성 요소를 제안합니다.

1. **Linguistic-aligned Attention Guidance**:  
   - 편향이 발생하는 의미 영역을 정확히 찾아내기 위해, 텍스트 프롬프트에서 특정 단어(예: "faces", "doctors")와 연관된 주의 영역을 분석합니다.  
   - 블록 투표(Block Voting)와 언어적 정렬(Linguistic Alignment)을 통해 의미 영역을 정확히 추출합니다.

2. **Fair Inference**:  
   - 편향이 발견된 의미 영역에 대해 공정한 속성을 생성합니다.  
   - 예를 들어, 성별 편향이 있을 경우 "남성"과 "여성" 속성을 균등하게 적용해 공정한 분포를 만듭니다.  
   - 원래 이미지의 구조적 및 의미적 정보를 보존하면서도 공정한 속성을 추가합니다.

---

### 제안하는 데이터셋
- **사용 데이터셋**:  
  논문은 Stable Diffusion 모델(Stable Diffusion 1.4)을 기반으로 작업하며, 다음과 같은 데이터셋을 사용합니다:  
  - **CelebA**: 얼굴 속성을 포함한 데이터셋.  
  - **FairFace**: 성별과 인종에 따른 공정성을 측정하기 위한 데이터셋.  
  - **LAION-Aesthetics**: 다양한 스타일 및 활동과 관련된 프롬프트를 제공하는 대규모 텍스트-이미지 데이터셋.

---

### 비교하는 모델
다양한 기존 모델들과의 성능 비교를 통해 제안된 메서드의 효과를 입증합니다:
1. **Stable Diffusion with Ethical Interventions (SD-EI)**: 텍스트 프롬프트를 통해 윤리적 개입을 적용.
2. **FairDiffusion**: 공정한 프롬프트 테이블과 의미적 안내를 사용하는 방법.
3. **Unified Concept Editing (UCE)**: 교차 주의 레이어를 업데이트해 편향을 완화.
4. **Fine-tune**: 모델의 특정 컴포넌트를 미세 조정하여 편향 완화.

---

### 제안하는 아키텍처
Stable Diffusion 1.4 아키텍처를 기반으로 하며, 다음과 같은 기술적 향상을 제안:
- **블록 투표(Block Voting)**를 통해 교차 주의 맵을 정렬해 보다 정확한 의미 영역을 추출.
- **언어적 정렬(Linguistic Alignment)**으로 의미적 영역의 안정성과 정확성을 보장.

---



### Method (Proposed Method)
This paper introduces two main components:
1. **Linguistic-aligned Attention Guidance**:  
   - Identifies semantically biased regions in text-to-image prompts (e.g., "faces," "doctors").  
   - Employs **Block Voting** to align cross-attention maps with meaningful regions and **Linguistic Alignment** to ensure robust semantic region extraction.

2. **Fair Inference**:  
   - Mitigates biases in detected semantic regions by introducing fair attributes while preserving original structural and semantic information.  
   - For instance, in gender debiasing, it ensures "male" and "female" attributes are equally represented.

---

### Proposed Dataset
The paper relies on existing datasets to test and validate the method:
- **CelebA**: A dataset containing facial attributes.
- **FairFace**: Used to measure fairness across gender and racial attributes.
- **LAION-Aesthetics**: Provides diverse prompts for styles and activities.

---

### Comparison Models
The proposed method is compared against several existing approaches:
1. **Stable Diffusion with Ethical Interventions (SD-EI)**: Adds ethical interventions via prompts.
2. **FairDiffusion**: Utilizes fair prompt tables and semantic guidance.
3. **Unified Concept Editing (UCE)**: Updates cross-attention layers to mitigate biases.
4. **Fine-tune**: Fine-tunes specific components of the model for debiasing.

---

### Proposed Architecture
Based on Stable Diffusion 1.4, the paper incorporates:
- **Block Voting** to align cross-attention maps with semantic regions.
- **Linguistic Alignment** for robust and stable extraction of semantic regions.



   
 
<br/>
# Results  



### 실험 결과 및 성능 비교

#### 1. **평가 메트릭**  
- **Bias Metrics**: 
  - Bias-W: 생성된 전체 결과에서 편향 측정.
  - Bias-P: 단일 이미지 내 편향 측정.
- **Structural Consistency**: 생성된 이미지의 구조적 일관성을 평가.
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index)
  - LPIPS (Learned Perceptual Image Patch Similarity)
- **Semantic Consistency**: 생성된 이미지의 의미적 일관성을 평가.
  - CLIP-I: CLIP 모델 임베딩의 코사인 유사도.
  - DINO-I: DINO 임베딩의 코사인 유사도.

---

#### 2. **결과 요약**  

**(1) Bias-W 및 Bias-P (편향 완화 성능):**  
- **성별 편향(Gender Bias):**  
  - 제안된 방법(Ours)은 Bias-W에서 0.080, Bias-P에서 0.100으로 모든 비교 모델보다 우수한 성능을 보임.  
  - FairDiffusion과 SD-EI는 일부 성능을 보였으나, UCE와 Fine-tune은 단일 성별 혹은 단일 인종에 치우침.  

- **인종 편향(Race Bias):**  
  - 제안된 방법은 Bias-W에서 0.177, Bias-P에서 0.225로, FairDiffusion(0.243, 0.294)보다 낮은 편향 수치를 기록.

- **교차 편향(Gender × Race Bias):**  
  - 제안된 방법은 Bias-W에서 0.145, Bias-P에서 0.137로 가장 낮은 값 기록.

---

**(2) 구조적 및 의미적 일관성:**  
- **구조적 일관성(PSNR, SSIM, LPIPS):**
  - 제안된 방법은 PSNR 17.929, SSIM 0.686, LPIPS 0.254로 모든 모델 중 최고 성능을 보임.
  - Fine-tune과 SD-EI는 구조적 왜곡이 더 크게 나타남.

- **의미적 일관성(CLIP-I, DINO-I):**
  - 제안된 방법은 CLIP-I 0.897, DINO-I 0.908로 가장 높은 의미적 일관성을 기록.  
  - FairDiffusion이 그다음으로 높은 값을 기록.

---

#### 3. **결론**
- 제안된 방법은 모든 편향 완화 메트릭(Bias-W, Bias-P)에서 최고의 성능을 기록했으며, 구조적 및 의미적 일관성도 유지.
- 기존 방법들은 편향 완화 성능이 제한적이거나 일부 속성에서 과도한 집중 현상을 보임.

---



### Experiment Results and Performance

#### 1. **Evaluation Metrics**  
- **Bias Metrics**: 
  - Bias-W: Measures bias across all generated results.
  - Bias-P: Measures bias within a single image.
- **Structural Consistency**: Evaluates structural integrity of generated images:
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index)
  - LPIPS (Learned Perceptual Image Patch Similarity)
- **Semantic Consistency**: Evaluates semantic integrity:
  - CLIP-I: Cosine similarity of CLIP embeddings.
  - DINO-I: Cosine similarity of DINO embeddings.

---

#### 2. **Summary of Results**  

**(1) Bias-W and Bias-P (Bias Mitigation Performance):**  
- **Gender Bias:**  
  - Proposed method (Ours): Bias-W 0.080, Bias-P 0.100, outperforming all comparison models.  
  - FairDiffusion and SD-EI showed some effectiveness, but UCE and Fine-tune exhibited overrepresentation of a single gender or race.

- **Race Bias:**  
  - Proposed method: Bias-W 0.177, Bias-P 0.225, lower than FairDiffusion (0.243, 0.294).  

- **Intersectional Bias (Gender × Race):**  
  - Proposed method: Bias-W 0.145, Bias-P 0.137, recording the lowest values.

---

**(2) Structural and Semantic Consistency:**  
- **Structural Consistency (PSNR, SSIM, LPIPS):**
  - Proposed method achieved PSNR 17.929, SSIM 0.686, and LPIPS 0.254, the best among all models.
  - Fine-tune and SD-EI showed greater structural distortions.

- **Semantic Consistency (CLIP-I, DINO-I):**
  - Proposed method recorded the highest scores: CLIP-I 0.897, DINO-I 0.908.  
  - FairDiffusion followed as the next best performer.

---

#### 3. **Conclusion**
- The proposed method achieved the best performance across all bias mitigation metrics (Bias-W, Bias-P) while maintaining structural and semantic integrity.  
- Existing methods showed limited effectiveness in bias mitigation or overrepresentation of certain attributes.




<br/>
# 예제  



### 예제 비교
논문에서는 다양한 모델이 생성한 "의사(doctors)"라는 직업 프롬프트에 대한 결과를 비교하였습니다. 각 모델이 성별(Gender)과 인종(Race)의 편향을 어떻게 처리하는지 시각적으로 비교했습니다.

---

#### 1. **Stable Diffusion (원본 모델)**
- **결과:**  
  - 대부분의 이미지에서 "백인 남성(White Male)"으로 구성된 결과를 생성.  
  - 성별 및 인종의 다양성이 부족.  
- **문제점:**  
  - 특정 그룹(백인 남성)이 과도하게 대표되며, 이는 편향된 결과를 초래.

---

#### 2. **Stable Diffusion with Ethical Interventions (SD-EI)**
- **결과:**  
  - 성별 다양성은 증가했으나, "아시아 남성(Asian Male)"과 "아시아 여성(Asian Female)"에 치우친 결과를 생성.  
- **문제점:**  
  - 이미지 내 모든 인물이 동일한 인종/성별로 표현됨.  

---

#### 3. **FairDiffusion**
- **결과:**  
  - 구조적 및 의미적 일관성을 잘 유지하며, 성별과 인종 간 다양성을 향상.  
  - 그러나 단일 이미지 내에서 동일한 인종 및 성별로만 구성되는 경향이 존재.  

---

#### 4. **Unified Concept Editing (UCE)**
- **결과:**  
  - 대부분의 이미지에서 "백인 여성(White Female)"으로 구성.  
  - 특정 성별 및 인종에 과도하게 집중.  

---

#### 5. **Fine-tune**
- **결과:**  
  - "인도 남성(Indian Male)"과 "인도 여성(Indian Female)"에 편향된 결과.  
  - 이미지 내 모든 인물이 동일한 얼굴 특징을 공유.  

---

#### 6. **Proposed Method (제안된 방법)**
- **결과:**  
  - 다양한 성별과 인종을 포함한 이미지 생성.  
  - 단일 이미지 내에서도 각 인물이 다양한 속성을 가지며, 구조적 및 의미적 정보가 잘 보존됨.  

---

### 시각적 비교
- **도구:**  
  연구에서는 CelebA 및 FairFace 데이터셋으로 훈련된 분류기를 사용해 이미지 내 성별 및 인종을 식별.  
- **결과:**  
  제안된 방법은 "백인, 흑인, 아시아인, 인도인" 및 "남성, 여성"의 속성을 골고루 포함하며, 기존 모델 대비 가장 공정한 결과를 보여줌.

---



### Example Comparison
The paper compares the results of various models on the prompt "doctors" to evaluate how each model handles gender and racial biases.

---

#### 1. **Stable Diffusion (Original Model)**
- **Results:**  
  - Most images featured "White Male" individuals.  
  - Lacked diversity in both gender and race.  
- **Issues:**  
  - Overrepresentation of a specific group (White Male) led to biased results.

---

#### 2. **Stable Diffusion with Ethical Interventions (SD-EI)**
- **Results:**  
  - Improved gender diversity but overrepresented "Asian Male" and "Asian Female."  
- **Issues:**  
  - Individuals in a single image were all from the same race/gender group.  

---

#### 3. **FairDiffusion**
- **Results:**  
  - Maintained good structural and semantic consistency while improving gender and racial diversity.  
  - However, tended to homogenize attributes within a single image.  

---

#### 4. **Unified Concept Editing (UCE)**
- **Results:**  
  - Most images predominantly featured "White Female."  
  - Overfocused on a specific gender/race.  

---

#### 5. **Fine-tune**
- **Results:**  
  - Images were biased toward "Indian Male" and "Indian Female."  
  - All individuals in an image shared the same facial features.  

---

#### 6. **Proposed Method**
- **Results:**  
  - Generated images with diverse genders and races.  
  - Preserved structural and semantic information while ensuring individuals within a single image had varied attributes.  

---

### Visual Comparison
- **Tools:**  
  The study used classifiers trained on CelebA and FairFace datasets to identify gender and race within generated images.  
- **Results:**  
  The proposed method evenly included attributes like "White, Black, Asian, Indian" and "Male, Female," showing the most equitable results compared to existing models.


<br/>  
# 요약 




이 논문은 텍스트-이미지 생성 모델의 사회적 편향 문제를 해결하기 위해 **언어 정렬 주의 지침(Linguistic-aligned Attention Guidance)**과 **공정한 추론(Fair Inference)**이라는 새로운 방법론을 제안합니다. 이 접근법은 텍스트 프롬프트에서 편향을 유발하는 특정 단어와 연관된 의미 영역을 찾아 편향을 완화하며, 다수의 인물이 포함된 이미지에서도 다양한 속성을 보장합니다. 제안된 방법은 CelebA, FairFace, 그리고 LAION-Aesthetics 데이터셋을 사용하여 다양한 시나리오에서 테스트되었으며, 기존의 모델(FairDiffusion, UCE 등)과 비교했을 때 구조적 및 의미적 일관성을 유지하면서도 성별 및 인종 편향 완화에서 우수한 성능을 보였습니다. 특히, "의사(doctors)"라는 프롬프트를 기반으로 한 실험에서 제안된 방법은 단일 이미지 내에서도 다양한 성별과 인종을 포함한 결과를 생성하며, 기존 모델이 보여준 단일 속성 편향을 극복했습니다. 따라서 이 논문은 텍스트-이미지 생성 모델에서 공정성을 보장하는 데 중요한 기여를 합니다.

---



This paper proposes a novel approach to mitigate social biases in text-to-image generation models through **Linguistic-aligned Attention Guidance** and **Fair Inference**. The method identifies semantic regions associated with bias-inducing tokens in text prompts and mitigates these biases while ensuring diverse attributes in multi-individual images. Using datasets like CelebA, FairFace, and LAION-Aesthetics, the proposed method was evaluated across various scenarios, demonstrating superior performance in bias mitigation compared to existing models like FairDiffusion and UCE, while preserving structural and semantic consistency. In experiments with the prompt "doctors," the proposed method effectively generated images that included diverse genders and races within a single image, overcoming the single-attribute biases of prior models. This work makes a significant contribution to ensuring fairness in text-to-image generation models.  


<br/>  
# 기타  





### 논문 속 다이어그램 설명:

1. **Figure 1: Occupation별 생성된 이미지의 성별 및 인종 빈도 분포**
   - Stable Diffusion 모델이 생성한 직업별 이미지에서 특정 그룹("백인 남성")이 과도하게 대표되는 경향을 보여줍니다.
   - "housekeeper" 직업은 "여성"으로, 나머지 직업은 대부분 "남성"으로 치우친 편향이 관찰됩니다.

2. **Figure 2: 다양한 편향 완화 기법의 비교 결과**
   - 여러 기존 방법(FairDiffusion, UCE 등)과 제안된 방법의 결과 비교를 시각화한 그림입니다.
   - 기존 방법들은 단일 이미지 내 모든 인물이 동일한 성별과 인종을 갖는 경향이 있는 반면, 제안된 방법은 다양한 속성을 가진 인물들을 생성합니다.

3. **Figure 3: 제안된 방법의 파이프라인**
   - **1단계:** Linguistic-aligned Attention Guidance 모듈을 통해 텍스트 프롬프트의 편향이 발생하는 의미적 영역을 식별합니다.
   - **2단계:** Fair Inference를 사용하여 해당 영역에 공정한 속성을 부여하면서 구조적 및 의미적 정보를 보존합니다.

4. **Figure 4: Block Voting 및 Linguistic Alignment의 효과**
   - (a) Block Voting을 적용하기 전후의 교차 주의 맵 비교.
   - (b) Cross-attention 블록 및 스텝에 따른 효과성 평가.
   - (c) Semantic region의 정확도와 threshold 값의 상관관계.
   - (d) 제안된 모듈의 적용 후, 더 정확하고 안정적인 의미 영역이 생성됨을 보여줍니다.

5. **Figure 5: 다양한 기법으로 생성된 "의사" 이미지 비교**
   - Stable Diffusion과 비교 방법(UCE, FairDiffusion 등) 및 제안된 방법의 생성 결과를 비교합니다.
   - 제안된 방법은 더 다양한 성별 및 인종을 포함하며, 단일 이미지 내에서도 속성의 다양성이 돋보입니다.

6. **Figure 6: Ablation Study 및 Age Distribution**
   - (a) Ablation Study 결과: Fair Inference와 Linguistic-aligned Attention Guidance의 기여도를 평가.
   - (b) Stable Diffusion과 제안된 방법의 직업별 연령 분포 비교. 제안된 방법은 더 넓은 연령 분포를 포함함을 보여줍니다.

7. **Figure 7: 다양한 시나리오에서의 결과**
   - 개인 묘사(예: "smiling"), 일상 활동(예: "flowers 옆에 서 있는 사람들"), 스타일 프롬프트 등에서 제안된 방법이 더 공정한 결과를 생성함을 보여줍니다.

---



### Diagrams in the Paper:

1. **Figure 1: Frequency Distribution of Gender and Race Across Occupations**
   - Visualizes the bias in Stable Diffusion-generated images, showing overrepresentation of specific groups (e.g., "White Male") in most occupations.
   - Certain occupations like "housekeeper" are heavily skewed toward "Female."

2. **Figure 2: Comparison of Bias Mitigation Techniques**
   - Compares the results of existing methods (e.g., FairDiffusion, UCE) and the proposed method.
   - While prior methods homogenize attributes within a single image, the proposed method generates individuals with diverse attributes.

3. **Figure 3: Proposed Method Pipeline**
   - **Step 1:** Linguistic-aligned Attention Guidance identifies biased semantic regions in text prompts.
   - **Step 2:** Fair Inference applies fair attributes to these regions while preserving structural and semantic information.

4. **Figure 4: Effectiveness of Block Voting and Linguistic Alignment**
   - (a) Comparison of cross-attention maps before and after applying Block Voting.
   - (b) Evaluation of cross-attention effectiveness across blocks and steps.
   - (c) Correlation between semantic region accuracy and threshold values.
   - (d) Demonstrates that the proposed modules result in more accurate and stable semantic regions.

5. **Figure 5: Comparison of Generated "Doctors" Images**
   - Compares results from Stable Diffusion, other methods (e.g., UCE, FairDiffusion), and the proposed method.
   - The proposed method shows more diverse genders and races, even within a single image.

6. **Figure 6: Ablation Study and Age Distribution**
   - (a) Ablation Study results: Evaluates the contributions of Fair Inference and Linguistic-aligned Attention Guidance.
   - (b) Age distributions for occupations in Stable Diffusion vs. the proposed method. The latter includes broader age diversity.

7. **Figure 7: Results Across Scenarios**
   - Illustrates fairer results from the proposed method in scenarios like personal descriptors (e.g., "smiling"), daily activities (e.g., "standing next to flowers"), and style prompts.


<br/>
# refer format:     


@inproceedings{jiang2024mitigating,
  author    = {Yue Jiang and Yueming Lyu and Ziwen He and Bo Peng and Jing Dong},
  title     = {Mitigating Social Biases in Text-to-Image Diffusion Models via Linguistic-Aligned Attention Guidance},
  booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia (MM '24)},
  year      = {2024},
  month     = {October 28-November 1},
  address   = {Melbourne, VIC, Australia},
  publisher = {ACM},
  doi       = {10.1145/3664647.3680748},
  isbn      = {979-8-4007-0686-8},
  pages     = {3391--3400}
}




Yue Jiang, Yueming Lyu, Ziwen He, Bo Peng, and Jing Dong. "Mitigating Social Biases in Text-to-Image Diffusion Models via Linguistic-Aligned Attention Guidance." In Proceedings of the 32nd ACM International Conference on Multimedia (MM '24), 3391–3400. Melbourne, VIC, Australia, October 28–November 1, 2024. ACM. https://doi.org/10.1145/3664647.3680748.

