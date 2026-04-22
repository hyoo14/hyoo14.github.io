---
layout: post
title:  "[2024]BIOSCAN-5M: A Multimodal Dataset for Insect Biodiversity"
date:   2026-04-22 13:18:45 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: BIOSCAN-5M 데이터셋은 5백만 개 이상의 곤충 표본에 대한 다중 모달 정보를 포함하며, DNA 바코드, 이미지 및 분류 정보를 활용하여 세 가지 벤치마크 실험을 수행하였다.


짧은 요약(Abstract) :


이 논문의 초록에서는 BIOSCAN-5M이라는 곤충 생물 다양성 데이터셋을 소개하고 있습니다. 이 데이터셋은 500만 개 이상의 곤충 표본에 대한 다중 모드 정보를 포함하고 있으며, 기존의 이미지 기반 생물학 데이터셋을 크게 확장합니다. 데이터셋에는 분류학적 레이블, 원시 뉴클레오타이드 바코드 서열, 바코드 인덱스 번호, 지리적 정보 및 크기 정보가 포함되어 있습니다. 저자들은 이 데이터셋을 활용하여 분류 및 클러스터링 정확도에 대한 다중 모드 데이터 유형의 영향을 보여주는 세 가지 벤치마크 실험을 제안합니다. 첫 번째로, DNA 바코드 서열에 대해 마스크 언어 모델을 사전 훈련하고, 이 대규모 참조 라이브러리를 사용하여 종 및 속 수준의 분류 성능에 미치는 영향을 입증합니다. 두 번째로, 이미지와 DNA 바코드를 사용한 제로샷 전이 학습 작업을 제안하여 의미 있는 클러스터가 생성될 수 있는지를 조사합니다. 마지막으로, DNA 바코드, 이미지 데이터 및 분류학적 정보를 사용하여 대조 학습을 수행하여 다중 모드의 벤치마크를 설정합니다. 이 데이터셋의 코드 저장소는 제공됩니다.



The abstract of this paper introduces the BIOSCAN-5M dataset for insect biodiversity, which contains multimodal information for over 5 million insect specimens. It significantly expands existing image-based biological datasets by including taxonomic labels, raw nucleotide barcode sequences, barcode index numbers, geographical information, and size information. The authors propose three benchmark experiments to demonstrate the impact of multimodal data types on classification and clustering accuracy. First, they pretrain a masked language model on the DNA barcode sequences and demonstrate the impact of using this large reference library on species- and genus-level classification performance. Second, they propose a zero-shot transfer learning task applied to images and DNA barcodes to investigate whether meaningful clusters can be derived from these representations. Finally, they benchmark multimodality by performing contrastive learning on DNA barcodes, image data, and taxonomic information. The code repository for this dataset is provided.


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



BIOSCAN-5M 데이터셋은 5백만 개 이상의 곤충 표본에 대한 다중 모드 정보를 포함하는 포괄적인 데이터셋으로, 생물 다양성 연구와 기계 학습 커뮤니티에 기여하기 위해 개발되었습니다. 이 데이터셋은 DNA 바코드, 이미지, 지리적 정보 및 분류학적 레이블을 포함하여 생물체의 분류 및 클러스터링 정확도를 높이는 데 사용됩니다.

#### 모델 및 아키텍처
BIOSCAN-5M 데이터셋의 주요 모델은 **BarcodeBERT**라는 변환기 기반의 언어 모델입니다. 이 모델은 DNA 바코드 시퀀스를 입력으로 받아들이고, 마스킹된 언어 모델링(Masked Language Modeling) 기법을 사용하여 사전 훈련됩니다. 이 과정에서 모델은 DNA 바코드의 패턴을 학습하고, 이를 통해 종 및 속 수준의 분류 성능을 향상시킵니다.

#### 훈련 데이터
훈련 데이터는 BIOSCAN-5M 데이터셋에서 제공되는 2,283,900개의 고유 DNA 시퀀스와 41,232개의 다른 보류 시퀀스를 포함하여 총 2,325,132개의 샘플로 구성됩니다. 이 데이터는 비표기된 샘플을 사용하여 자가 감독 학습(self-supervised learning)을 통해 사전 훈련된 후, 고품질 레이블이 있는 훈련 세트에서 미세 조정(fine-tuning)됩니다.

#### 특별한 기법
1. **제로샷 클러스터링(Zero-shot Clustering)**: 이 기법은 사전 훈련된 인코더를 사용하여 보지 못한 데이터셋을 클러스터링하는 방법입니다. 이 과정에서 DNA 바코드와 이미지 데이터를 결합하여 의미 있는 클러스터를 생성합니다.
   
2. **대조 학습(Contrastive Learning)**: 이 방법은 서로 다른 모드(이미지, DNA, 텍스트) 간의 임베딩을 정렬하여 공유 임베딩 공간을 학습합니다. 이를 통해 다양한 샘플을 훈련에 포함시켜 불완전한 분류 레이블을 사용하더라도 분류 성능을 향상시킬 수 있습니다.

3. **다중 모드 학습(Multimodal Learning)**: 이 접근법은 이미지, DNA 바코드, 텍스트 레이블을 통합하여 분류 작업을 수행합니다. 각 모드의 정보를 활용하여 더 나은 성능을 달성합니다.

이러한 방법론을 통해 BIOSCAN-5M 데이터셋은 생물 다양성 모니터링 및 기계 학습 연구에 중요한 기여를 할 것으로 기대됩니다.

---




The BIOSCAN-5M dataset is a comprehensive resource containing multimodal information on over 5 million insect specimens, developed to contribute to biodiversity research and the machine learning community. This dataset includes DNA barcodes, images, geographical information, and taxonomic labels, which are utilized to enhance classification and clustering accuracy of organisms.

#### Model and Architecture
The primary model used with the BIOSCAN-5M dataset is **BarcodeBERT**, a transformer-based language model. This model takes DNA barcode sequences as input and employs a Masked Language Modeling technique for pre-training. During this process, the model learns patterns in DNA barcodes, improving species and genus-level classification performance.

#### Training Data
The training data consists of 2,283,900 unique DNA sequences from the BIOSCAN-5M dataset and 41,232 additional sequences from the other held-out partition, totaling 2,325,132 samples. This data is used for self-supervised learning on unlabelled samples, followed by fine-tuning on a training set with high-quality labels.

#### Special Techniques
1. **Zero-shot Clustering**: This technique uses pretrained encoders to cluster unseen datasets. In this process, DNA barcodes and image data are combined to create meaningful clusters.

2. **Contrastive Learning**: This method aligns embeddings across different modalities (images, DNA, text) to learn a shared embedding space. This allows for the inclusion of diverse samples in training, enhancing classification performance even with incomplete taxonomic labels.

3. **Multimodal Learning**: This approach integrates images, DNA barcodes, and text labels to perform classification tasks. By leveraging information from each modality, it achieves better performance.

Through these methodologies, the BIOSCAN-5M dataset is expected to make significant contributions to biodiversity monitoring and machine learning research.


<br/>
# Results



BIOSCAN-5M 데이터셋을 사용한 실험 결과는 여러 경쟁 모델과 비교하여 다양한 테스트 데이터와 메트릭을 통해 평가되었습니다. 이 연구에서는 DNA 바코드, 이미지 데이터, 그리고 분류 정보를 포함한 다중 모달리티를 활용하여 생물 분류의 정확성을 높이는 방법을 제안했습니다.

1. **경쟁 모델**: 실험에서는 BarcodeBERT, DNABERT-2, DNABERT-S, HyenaDNA, 그리고 CNN 기반 모델과 같은 여러 경쟁 모델이 사용되었습니다. 각 모델은 DNA 바코드와 이미지 데이터를 기반으로 한 분류 작업에서 성능을 비교했습니다.

2. **테스트 데이터**: 데이터는 'seen', 'unseen', 'heldout', 'unknown'의 네 가지 카테고리로 나뉘어 평가되었습니다. 'seen' 데이터는 훈련 중에 본 적이 있는 종의 샘플로 구성되며, 'unseen' 데이터는 훈련 중에 본 적이 없는 종의 샘플입니다. 'heldout' 데이터는 종의 레이블이 없는 샘플로, 새로운 종을 탐지하는 데 사용됩니다. 'unknown' 데이터는 레이블이 없는 샘플로, 이들 또한 새로운 종으로 분류될 수 있습니다.

3. **메트릭**: 성능 평가는 주로 정확도(accuracy)와 1-NN 프로빙(1-Nearest Neighbor probing) 메트릭을 사용하여 이루어졌습니다. 정확도는 모델이 올바르게 분류한 샘플의 비율을 나타내며, 1-NN 프로빙은 훈련 중에 본 적이 없는 종에 대해 모델이 얼마나 잘 일반화되는지를 평가합니다.

4. **비교 결과**: 
   - **닫힌 세계(closed-world) 설정**에서, 모델은 'seen' 데이터에 대해 높은 정확도를 기록했습니다. 예를 들어, 우리의 모델은 99.28%의 정확도를 달성하여 DNABERT-2(99.23%)와 경쟁했습니다.
   - **열린 세계(open-world) 설정**에서는, 모델이 'unseen' 종을 'seen' 카테고리로 분류하는 능력을 평가했습니다. 이 경우, 우리의 모델은 47.03%의 정확도를 기록하여 다른 모델들보다 우수한 성능을 보였습니다.
   - DNA 바코드를 사용한 분류 작업에서, HyenaDNA 모델이 가장 높은 성능을 보였으며, DNA 바코드의 정보가 종의 정체성을 파악하는 데 매우 유용하다는 것을 보여주었습니다.

5. **결론**: BIOSCAN-5M 데이터셋은 다중 모달리티를 활용하여 생물 분류의 정확성을 높이는 데 기여하며, 다양한 실험을 통해 그 효과를 입증했습니다. 이 연구는 생물 다양성 모니터링 및 새로운 종 탐지에 있어 중요한 기초 자료로 활용될 수 있습니다.

---




The experimental results using the BIOSCAN-5M dataset were evaluated through comparisons with various competitive models, utilizing different test data and metrics. This study proposed methods to enhance classification accuracy in biological taxonomy by leveraging multiple modalities, including DNA barcodes, image data, and classification information.

1. **Competitive Models**: Several competitive models were employed in the experiments, including BarcodeBERT, DNABERT-2, DNABERT-S, HyenaDNA, and CNN-based models. Each model's performance was compared in classification tasks based on DNA barcodes and image data.

2. **Test Data**: The data was partitioned into four categories: 'seen', 'unseen', 'heldout', and 'unknown'. The 'seen' data consists of samples from species that were encountered during training, while 'unseen' data includes samples from species not seen during training. The 'heldout' data consists of samples without species labels, used for detecting novel species. The 'unknown' data refers to samples without any labels, which may also belong to new species.

3. **Metrics**: Performance evaluation primarily utilized accuracy and 1-NN probing metrics. Accuracy indicates the proportion of samples correctly classified by the model, while 1-NN probing assesses how well the model generalizes to unseen species.

4. **Comparison Results**: 
   - In the **closed-world setting**, the model achieved high accuracy on 'seen' data. For instance, our model reached an accuracy of 99.28%, competing closely with DNABERT-2 (99.23%).
   - In the **open-world setting**, the model's ability to classify 'unseen' species into 'seen' categories was evaluated. In this case, our model recorded an accuracy of 47.03%, outperforming other models.
   - In the classification tasks using DNA barcodes, the HyenaDNA model exhibited the highest performance, demonstrating that the information from DNA barcodes is highly informative for species identification.

5. **Conclusion**: The BIOSCAN-5M dataset contributes to enhancing classification accuracy in biological taxonomy through the utilization of multiple modalities, and the effectiveness of this approach was validated through various experiments. This research can serve as a crucial resource for biodiversity monitoring and the detection of new species.


<br/>
# 예제



BIOSCAN-5M 데이터셋은 5백만 개 이상의 곤충 표본에 대한 다중 모드 정보를 포함하고 있습니다. 이 데이터셋은 DNA 바코드, 이미지, 지리적 정보 및 분류학적 레이블을 포함하여 생물 다양성을 연구하는 데 필요한 다양한 정보를 제공합니다. 이 데이터셋을 사용하여 수행된 주요 실험은 다음과 같습니다.

1. **트레이닝 데이터와 테스트 데이터의 구성**:
   - **트레이닝 데이터**: 5,150,850개의 표본 중 289,203개의 표본이 'seen' 데이터로 사용됩니다. 이 표본들은 이미 알려진 종의 레이블을 가지고 있습니다. 
   - **테스트 데이터**: 39,373개의 표본이 'test' 데이터로 사용되며, 이 표본들은 모델의 성능을 평가하는 데 사용됩니다. 이 표본들은 'seen' 데이터에서 학습한 종의 레이블을 기반으로 합니다.

2. **입력과 출력**:
   - **입력**: 각 표본의 DNA 바코드 시퀀스, 이미지, 그리고 해당 표본의 지리적 정보가 입력으로 사용됩니다. 예를 들어, DNA 바코드는 특정 종을 식별하는 데 필요한 고유한 유전자 정보를 제공합니다.
   - **출력**: 모델의 출력은 각 표본에 대한 분류 결과로, 종, 속, 과 등의 분류학적 레이블이 포함됩니다. 예를 들어, 모델이 'Tapinoma sessile'이라는 종을 정확히 분류할 수 있다면, 이는 모델이 해당 표본을 올바르게 인식했음을 의미합니다.

3. **구체적인 태스크**:
   - **종 분류**: 모델은 주어진 DNA 바코드와 이미지를 기반으로 표본을 특정 종으로 분류하는 작업을 수행합니다. 이 작업은 'closed-world' 설정에서 수행되며, 이미 알려진 종의 레이블을 사용하여 모델을 학습합니다.
   - **제로샷 클러스터링**: 모델은 'unseen' 데이터에 대해 클러스터링을 수행하여 새로운 종을 식별하는 작업을 수행합니다. 이 작업은 모델이 학습하지 않은 종의 표본을 그룹화하는 데 중점을 둡니다.

이러한 실험을 통해 BIOSCAN-5M 데이터셋은 생물 다양성 연구와 기계 학습 모델의 성능 향상에 기여할 수 있는 중요한 자원으로 자리 잡고 있습니다.

---




The BIOSCAN-5M dataset contains multimodal information for over 5 million insect specimens. This dataset provides various information necessary for studying biodiversity, including DNA barcodes, images, geographic information, and taxonomic labels. The main experiments conducted using this dataset are as follows:

1. **Composition of Training and Test Data**:
   - **Training Data**: Out of the 5,150,850 specimens, 289,203 specimens are used as 'seen' data. These specimens have labels for species that are already known.
   - **Test Data**: 39,373 specimens are used as 'test' data, which is employed to evaluate the model's performance. These specimens are based on the labels of species learned from the 'seen' data.

2. **Inputs and Outputs**:
   - **Input**: The input consists of the DNA barcode sequence, images, and geographic information for each specimen. For example, the DNA barcode provides unique genetic information necessary for identifying a specific species.
   - **Output**: The model's output is the classification result for each specimen, which includes taxonomic labels such as species, genus, and family. For instance, if the model accurately classifies a specimen as 'Tapinoma sessile', it indicates that the model has correctly recognized that specimen.

3. **Specific Tasks**:
   - **Species Classification**: The model performs the task of classifying specimens into specific species based on the given DNA barcodes and images. This task is conducted in a 'closed-world' setting, using labels of already known species to train the model.
   - **Zero-Shot Clustering**: The model performs clustering on 'unseen' data to identify new species. This task focuses on grouping specimens of species that the model has not been trained on.

Through these experiments, the BIOSCAN-5M dataset establishes itself as an important resource that can contribute to biodiversity research and enhance the performance of machine learning models.

<br/>
# 요약


BIOSCAN-5M 데이터셋은 5백만 개 이상의 곤충 표본에 대한 다중 모달 정보를 포함하며, DNA 바코드, 이미지 및 분류 정보를 활용하여 세 가지 벤치마크 실험을 수행하였다. 첫 번째 실험에서는 DNA 바코드를 기반으로 한 마스크 언어 모델을 사전 훈련하여 분류 성능을 향상시켰고, 두 번째 실험에서는 제로샷 클러스터링을 통해 의미 있는 클러스터를 도출하였다. 마지막으로, 다중 모달 학습을 통해 이미지와 DNA 바코드를 결합하여 세부 분류 정확도를 높였다.

---

The BIOSCAN-5M dataset contains multimodal information on over 5 million insect specimens and conducts three benchmark experiments utilizing DNA barcodes, images, and classification information. The first experiment pretrains a masked language model based on DNA barcodes to enhance classification performance, while the second experiment investigates zero-shot clustering to derive meaningful clusters. Finally, multimodal learning is employed to combine images and DNA barcodes, improving fine-grained classification accuracy.

<br/>
# 기타


1. **다이어그램 및 피규어**
   - **Figure 1**: BIOSCAN-5M 데이터셋의 구성 요소를 보여줍니다. 각 샘플에 대한 세부 정보(세포 이미지, DNA 바코드, 지리적 정보 등)를 포함하여 데이터셋의 다중 모달 특성을 강조합니다. 이는 생물 다양성 연구에서의 중요성을 나타냅니다.
   - **Figure 2**: 다양한 생물체의 원본 이미지 샘플을 보여줍니다. 이는 데이터셋의 다양성을 강조하며, 고해상도 이미지가 생물 분류에 어떻게 기여하는지를 보여줍니다.
   - **Figure 3**: 샘플 수집 위치의 지리적 분포를 나타냅니다. 이는 데이터셋의 지리적 다양성을 강조하며, 생물 다양성 모니터링의 중요성을 보여줍니다.
   - **Figure 4**: 제로샷 클러스터링 성능을 보여주는 그래프입니다. DNA 인코더가 이미지 인코더보다 더 높은 성능을 보이는 것을 확인할 수 있습니다. 이는 DNA 바코드가 종 정체성에 대한 정보를 잘 제공한다는 것을 시사합니다.

2. **테이블**
   - **Table 1**: 다양한 생물학적 데이터셋의 비교를 보여줍니다. BIOSCAN-5M 데이터셋이 다른 데이터셋에 비해 더 많은 샘플과 다양한 메타데이터를 포함하고 있음을 강조합니다.
   - **Table 2**: 데이터셋의 통계적 요약을 제공합니다. 각 분류 수준에서의 레이블 수와 비율을 보여주며, 데이터셋의 클래스 불균형 비율(IR)을 나타냅니다. 이는 세부 분류 작업의 어려움을 강조합니다.
   - **Table 4**: DNA 기반 모델의 성능을 비교합니다. 각 모델의 정확도를 보여주며, BIOSCAN-5M 데이터셋을 사용한 모델이 다른 모델보다 우수한 성능을 보임을 나타냅니다.
   - **Table 5**: 다중 모달 학습의 결과를 보여줍니다. 서로 다른 쿼리 및 키 조합에 대한 정확도를 비교하여, 이미지와 DNA 간의 상호작용이 분류 성능에 미치는 영향을 강조합니다.

3. **어펜딕스**
   - 어펜딕스에서는 실험 설정, 데이터 전처리 방법, 모델 아키텍처 및 하이퍼파라미터에 대한 추가 정보를 제공합니다. 이는 연구 결과의 재현성을 높이는 데 기여합니다. 또한, 각 실험의 컴퓨팅 자원과 시간 소요에 대한 세부 정보를 제공하여, 연구의 신뢰성을 높입니다.




1. **Diagrams and Figures**
   - **Figure 1**: Illustrates the components of the BIOSCAN-5M dataset. It highlights the multi-modal nature of the dataset by including details for each sample (cell images, DNA barcodes, geographical information, etc.), emphasizing its importance in biodiversity research.
   - **Figure 2**: Displays original image samples of various organisms. This emphasizes the diversity of the dataset and shows how high-resolution images contribute to biological classification.
   - **Figure 3**: Represents the geographical distribution of sample collection locations. This highlights the geographical diversity of the dataset and underscores the importance of biodiversity monitoring.
   - **Figure 4**: A graph showing zero-shot clustering performance. It confirms that DNA encoders outperform image encoders, suggesting that DNA barcodes provide valuable information about species identity.

2. **Tables**
   - **Table 1**: Compares various biological datasets. It emphasizes that the BIOSCAN-5M dataset contains more samples and diverse metadata compared to other datasets.
   - **Table 2**: Provides a statistical summary of the dataset. It shows the number and percentage of labels at each classification level and indicates the class imbalance ratio (IR), highlighting the challenges of fine-grained classification tasks.
   - **Table 4**: Compares the performance of DNA-based models. It shows the accuracy of each model, indicating that models using the BIOSCAN-5M dataset outperform others.
   - **Table 5**: Displays results of multi-modal learning. It compares accuracy for different query and key combinations, emphasizing the impact of interactions between images and DNA on classification performance.

3. **Appendices**
   - The appendices provide additional information on experimental settings, data preprocessing methods, model architectures, and hyperparameters. This contributes to the reproducibility of the research findings. Additionally, detailed information about computing resources and time required for each experiment enhances the credibility of the research.

<br/>
# refer format:


### BibTeX 형식

```bibtex
@inproceedings{gharaee2024bioscan5m,
  title={BIOSCAN-5M: A Multimodal Dataset for Insect Biodiversity},
  author={Gharaee, Zahra and Lowe, Scott C. and Gong, ZeMing and Millan Arias, Pablo and Pellegrino, Nicholas and Wang, Austin T. and Haurum, Joakim Bruslund and Zarubiieva, Iuliia and Kari, Lila and Steinke, Dirk and Taylor, Graham W. and Fieguth, Paul and Chang, Angel X.},
  booktitle={Proceedings of the 38th Conference on Neural Information Processing Systems (NeurIPS 2024) Track on Datasets and Benchmarks},
  year={2024},
  url={https://biodiversitygenomics.net/5M-insects/}
}
```

### 시카고 스타일

Gharaee, Zahra, Scott C. Lowe, ZeMing Gong, Pablo Millan Arias, Nicholas Pellegrino, Austin T. Wang, Joakim Bruslund Haurum, Iuliia Zarubiieva, Lila Kari, Dirk Steinke, Graham W. Taylor, Paul Fieguth, and Angel X. Chang. 2024. "BIOSCAN-5M: A Multimodal Dataset for Insect Biodiversity." In *Proceedings of the 38th Conference on Neural Information Processing Systems (NeurIPS 2024) Track on Datasets and Benchmarks*. Accessed [날짜]. https://biodiversitygenomics.net/5M-insects/.
