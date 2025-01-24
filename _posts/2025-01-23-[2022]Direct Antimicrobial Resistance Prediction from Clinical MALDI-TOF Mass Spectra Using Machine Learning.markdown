---
layout: post
title:  "[2022]Direct Antimicrobial Resistance Prediction from Clinical MALDI-TOF Mass Spectra Using Machine Learning"  
date:   2025-01-23 21:25:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

MALDI-TOF 질량 스펙트럼 데이터라는 신박한 걸 썼나봄.. 리그레션이랑 라이트부스트랑 멀티레이어펄셉트론과 같은 전통적인 머신러닝 모델들을 이 데이터로 학습 및 테스트  



짧은 요약(Abstract) :    



이 논문은 MALDI-TOF 질량 분석법 데이터를 이용해 임상 균주의 항균제 내성을 머신러닝으로 직접 예측하는 새로운 접근법을 제안합니다. 저자들은 30만 개 이상의 질량 스펙트럼과 75만 개 이상의 항균제 내성 표본이 포함된 공개 데이터베이스를 구축하여 이를 활용했습니다. Staphylococcus aureus, Escherichia coli, Klebsiella pneumoniae 같은 주요 병원균을 대상으로 한 테스트에서 AUROC 점수 0.74~0.80의 높은 예측 성능을 보였습니다. 63명의 환자를 대상으로 한 후향적 임상 사례 연구에서는 9건의 치료 변경 제안을 통해 89%의 사례에서 긍정적인 영향을 확인했습니다. 이 방법은 항생제 관리와 치료 최적화에 있어 중요한 도구가 될 가능성이 있습니다.

---



Early use of effective antimicrobial treatments is critical for the outcome of infections and the prevention of treatment resistance. Antimicrobial resistance testing enables the selection of optimal antibiotic treatments, but current culture-based techniques can take up to 72 hours to generate results. We have developed a novel machine learning approach to predict antimicrobial resistance directly from matrix-assisted laser desorption/ionization–time of flight (MALDI-TOF) mass spectra profiles of clinical isolates. We trained calibrated classifiers on a newly created publicly available database of mass spectra profiles from the clinically most relevant isolates with linked antimicrobial susceptibility phenotypes. This dataset combines more than 300,000 mass spectra with more than 750,000 antimicrobial resistance phenotypes from four medical institutions. Validation on a panel of clinically important pathogens, including *Staphylococcus aureus*, *Escherichia coli*, and *Klebsiella pneumoniae*, resulting in areas under the receiver operating characteristic curve of 0.80, 0.74, and 0.74, respectively, demonstrated the potential of using machine learning to substantially accelerate antimicrobial resistance determination and change of clinical management. Furthermore, a retrospective clinical case study of 63 patients found that implementing this approach would have changed the clinical treatment in nine cases, which would have been beneficial in eight cases (89%). MALDI-TOF mass spectra-based machine learning may thus be an important new tool for treatment optimization and antibiotic stewardship.



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





이 연구에서는 MALDI-TOF 질량 스펙트럼 데이터를 활용하여 항균제 내성을 예측하기 위해 새로운 머신러닝 기반 방법론을 제안했습니다. 연구팀은 DRIAMS(Database of Resistance Information on Antimicrobials and MALDI-TOF Mass Spectra)라는 대규모 공개 데이터베이스를 구축했습니다. 이 데이터베이스에는 2016년부터 2018년까지 스위스의 4개 병원에서 수집된 30만 개 이상의 임상 균주 스펙트럼과 75만 개 이상의 항균제 내성 표본이 포함되어 있습니다.

스펙트럼 데이터는 2,000 Da에서 20,000 Da 범위의 질량 대 전하비(m/z)로 나뉘며, 3 Da 크기의 고정된 구간으로 나누어 총 6,000차원의 벡터로 변환되었습니다. 항균제 내성 라벨은 실험실에서 기록된 "민감성(Susceptible)", "중간(Intermediate)", "내성(Resistant)"의 세 범주를 기반으로 "민감성"은 음성 클래스, "중간"과 "내성"은 양성 클래스로 이진화되었습니다.

모델은 로지스틱 회귀(Logistic Regression), 경사 부스팅 결정 트리(LightGBM), 다층 퍼셉트론(MLP)으로 구성된 세 가지 분류기를 사용하여 학습되었습니다. 각각의 분류기는 항균제-병원체 조합별로 훈련되었으며, 성능 평가를 위해 수신자 조작 특성 곡선(AUROC)과 정밀도-재현율 곡선(AUPRC)이 사용되었습니다. 특히, LightGBM과 MLP는 AUROC에서 가장 높은 성능을 보여 주요 병원균(예: Staphylococcus aureus, Escherichia coli, Klebsiella pneumoniae)에 대해 효과적인 내성 예측 결과를 나타냈습니다.

---


In this study, a novel machine learning-based methodology was proposed to predict antimicrobial resistance using MALDI-TOF mass spectrometry data. The research team constructed a large-scale public database called DRIAMS (Database of Resistance Information on Antimicrobials and MALDI-TOF Mass Spectra). This database comprises over 300,000 clinical isolate spectra and more than 750,000 antimicrobial resistance phenotypes collected from four hospitals in Switzerland between 2016 and 2018.

The spectral data were converted into fixed bins of 3 Da within a mass-to-charge ratio (m/z) range of 2,000 to 20,000 Da, resulting in a 6,000-dimensional vector representation. Antimicrobial resistance labels recorded as "Susceptible," "Intermediate," and "Resistant" in laboratory reports were binarized, where "Susceptible" was assigned to the negative class and "Intermediate" and "Resistant" were grouped into the positive class.

Three classifiers were employed: logistic regression, gradient-boosted decision trees (LightGBM), and a multilayer perceptron (MLP). Each classifier was trained on specific antimicrobial-pathogen combinations, and performance was evaluated using the area under the receiver operating characteristic curve (AUROC) and the area under the precision-recall curve (AUPRC). Notably, LightGBM and MLP demonstrated superior performance in terms of AUROC, achieving effective resistance prediction for key pathogens such as *Staphylococcus aureus*, *Escherichia coli*, and *Klebsiella pneumoniae*.



   
 
<br/>
# Results  





이 연구는 주요 병원균 (*Staphylococcus aureus*, *Escherichia coli*, *Klebsiella pneumoniae*)의 항균제 내성을 예측하기 위해 제안된 방법을 평가했습니다. 테스트 데이터는 스위스 4개 병원에서 수집된 DRIAMS-A 데이터셋에서 2016년부터 2018년까지 기록된 MALDI-TOF 질량 스펙트럼과 관련 항균제 내성 데이터로 구성되었습니다. 주요 항균제로는 *S. aureus*에 대한 옥사실린(oxacillin), *E. coli* 및 *K. pneumoniae*에 대한 세프트리악손(ceftriaxone)이 포함되었습니다.

성능 평가를 위해 수신자 조작 특성 곡선(AUROC)과 정밀도-재현율 곡선(AUPRC)이 사용되었습니다. LightGBM, MLP(다층 퍼셉트론), 로지스틱 회귀를 비교 모델로 사용했으며, LightGBM과 MLP가 AUROC 및 AUPRC 기준으로 가장 높은 성능을 나타냈습니다.

구체적으로, *S. aureus*의 옥사실린 내성 예측은 AUROC 0.80, AUPRC 0.49를 기록했으며, *E. coli*와 *K. pneumoniae*의 세프트리악손 내성 예측은 AUROC 0.74 및 AUPRC 0.30~0.33을 기록했습니다. 이는 현장 실험실에서 사용되는 기존의 표준적 페노타입 내성 테스트보다 빠르고 정확한 결과를 제공합니다. 또한, 후향적 임상 사례 분석에서는 63건 중 9건에서 치료를 변경할 수 있었으며, 이 중 89%는 환자에게 유익한 영향을 미쳤습니다.


---



This study evaluated the proposed approach for predicting antimicrobial resistance in key pathogens (*Staphylococcus aureus*, *Escherichia coli*, *Klebsiella pneumoniae*). The test dataset was derived from DRIAMS-A, comprising MALDI-TOF mass spectra and associated antimicrobial resistance data collected from four Swiss hospitals between 2016 and 2018. Key antimicrobials included oxacillin for *S. aureus* and ceftriaxone for *E. coli* and *K. pneumoniae*.

Performance was assessed using the area under the receiver operating characteristic curve (AUROC) and the area under the precision-recall curve (AUPRC). The models compared included logistic regression, LightGBM, and a multilayer perceptron (MLP). Among these, LightGBM and MLP demonstrated the highest performance.

Specifically, oxacillin resistance prediction for *S. aureus* achieved an AUROC of 0.80 and an AUPRC of 0.49. Ceftriaxone resistance prediction for *E. coli* and *K. pneumoniae* achieved AUROCs of 0.74 and AUPRCs ranging from 0.30 to 0.33. These results indicate faster and more accurate predictions compared to standard phenotypic resistance testing used in clinical laboratories. Furthermore, a retrospective clinical case study revealed that in 9 out of 63 cases, treatment changes could have been suggested, with 89% of these changes proving beneficial to patient outcomes.





<br/>
# 예제  


**테스트 데이터 예시:**

- *S. aureus* (황색포도상구균): 옥사실린(oxacillin) 내성을 예측
  - 테스트 데이터: DRIAMS-A 데이터셋에서 황색포도상구균에 대한 옥사실린 내성 라벨이 포함된 질량 스펙트럼 샘플 (약 10%의 양성 클래스 비율).
  - 제안 모델 (LightGBM, MLP): AUROC 0.80, AUPRC 0.49.
  - 비교 모델 (로지스틱 회귀): AUROC 0.72, AUPRC 0.40.

**예제 상황:**
- 제안된 모델(LightGBM)은 옥사실린 내성을 가진 황색포도상구균 샘플을 높은 정확도로 구분할 수 있었음.
- 비교 모델(로지스틱 회귀)은 양성 클래스가 적은 환경에서 높은 비율의 양성 샘플을 놓치는 경향이 있었음.

---

**비교의 세부 예시:**

1. 테스트 케이스: 
   - 테스트 데이터: 한 환자의 혈액 샘플에서 MALDI-TOF 질량 스펙트럼 분석을 통해 수집된 데이터.
   - 실제 라벨: 옥사실린 내성 양성.
   
2. 모델 출력:
   - **LightGBM**: 내성으로 예측 (정확).
   - **MLP**: 내성으로 예측 (정확).
   - **로지스틱 회귀**: 민감성으로 잘못 예측 (오류).

3. 결론:
   - 제안된 LightGBM 모델은 민감한 라벨을 놓치지 않고 정확히 내성을 예측.
   - 로지스틱 회귀는 양성 클래스 비율이 낮은 경우 성능이 저하됨.

---



**Test Data Example:**

- *S. aureus* (Staphylococcus aureus): Predicting oxacillin resistance
  - Test Data: Mass spectra samples with oxacillin resistance labels from the DRIAMS-A dataset (approximately 10% positive class ratio).
  - Proposed Models (LightGBM, MLP): AUROC 0.80, AUPRC 0.49.
  - Baseline Model (Logistic Regression): AUROC 0.72, AUPRC 0.40.

**Example Scenario:**
- The proposed LightGBM model successfully differentiated oxacillin-resistant *S. aureus* samples with high accuracy.
- The baseline logistic regression model showed a tendency to miss a significant proportion of positive samples under low positive class ratios.

---

**Detailed Comparison Example:**

1. Test Case:
   - Test Data: A patient’s blood sample analyzed through MALDI-TOF mass spectrometry.
   - True Label: Positive for oxacillin resistance.
   
2. Model Outputs:
   - **LightGBM**: Predicted resistance (correct).
   - **MLP**: Predicted resistance (correct).
   - **Logistic Regression**: Predicted susceptibility (incorrect).

3. Conclusion:
   - The proposed LightGBM model accurately predicted resistance without missing positive samples.
   - Logistic regression underperformed in scenarios with low positive class ratios.



<br/>  
# 요약   



이 연구는 MALDI-TOF 질량 스펙트럼 데이터를 활용하여 항균제 내성을 예측하기 위한 새로운 머신러닝 모델을 제안했습니다. 연구팀은 스위스 4개 병원에서 수집한 DRIAMS 데이터셋을 기반으로 30만 개 이상의 질량 스펙트럼과 75만 개 이상의 항균제 내성 라벨을 사용하여 LightGBM, MLP, 로지스틱 회귀 모델을 훈련했습니다. 주요 테스트 데이터는 황색포도상구균(*S. aureus*)의 옥사실린 내성 및 대장균(*E. coli*)과 폐렴간균(*K. pneumoniae*)의 세프트리악손 내성 라벨이 포함된 샘플로 구성되었습니다. 결과적으로, 제안된 LightGBM 모델은 옥사실린 내성 예측에서 AUROC 0.80과 AUPRC 0.49를 기록하며 기존 로지스틱 회귀 모델(AUROC 0.72, AUPRC 0.40)을 능가했습니다. 이러한 결과는 제안된 모델이 저비율 양성 클래스에서도 더 높은 정확도로 내성을 예측할 수 있음을 보여줍니다.

---



This study proposed a novel machine learning model to predict antimicrobial resistance using MALDI-TOF mass spectrometry data. The researchers trained LightGBM, MLP, and logistic regression models using the DRIAMS dataset, which included over 300,000 mass spectra and 750,000 antimicrobial resistance labels collected from four Swiss hospitals. The key test data consisted of samples labeled with oxacillin resistance in *S. aureus* and ceftriaxone resistance in *E. coli* and *K. pneumoniae*. As a result, the proposed LightGBM model outperformed the baseline logistic regression model by achieving an AUROC of 0.80 and an AUPRC of 0.49 for predicting oxacillin resistance, compared to AUROC 0.72 and AUPRC 0.40 for the baseline model. These findings demonstrate that the proposed model can achieve higher accuracy in predicting resistance, even in scenarios with low positive class ratios.


<br/>  
# 기타  





1. **Fig. 1: MALDI-TOF 기반 항균제 내성 예측 워크플로우**
   - 이 그림은 데이터 수집, 스펙트럼 전처리, 머신러닝 분류 및 결과 평가까지의 전체 과정을 보여줍니다. 
   - 데이터는 DRIAMS 데이터베이스에서 수집되었으며, 항균제 내성은 이진화된 라벨로 변환되었습니다(민감성은 음성, 내성과 중간은 양성). 
   - 6,000개의 고정된 특성 벡터로 표현된 질량 스펙트럼 데이터를 바탕으로 로지스틱 회귀, LightGBM, MLP 모델을 훈련시켰습니다. 
   - AUROC 및 AUPRC 같은 성능 메트릭이 사용되었습니다.

2. **Fig. 3: DRIAMS 데이터셋에서 AUROC 성능 검증**
   - DRIAMS-A부터 D까지의 각 데이터셋을 사용하여 AUROC 점수를 비교한 결과를 나타냅니다.
   - 동일한 데이터셋에서 훈련 및 테스트한 경우 더 높은 성능을 보였으며, 특히 *E. coli*와 *K. pneumoniae*에서 두드러졌습니다. 
   - 데이터셋 간의 전이 학습도 테스트되었으며, 큰 데이터셋(DRIAMS-A)을 활용한 경우 더 나은 성능을 보였습니다.

3. **Fig. 5: SHAP 값을 사용한 LightGBM 및 MLP 모델의 피처 중요도 분석**
   - 이 그림은 LightGBM 및 MLP 모델에서 예측에 기여하는 주요 피처를 보여줍니다.
   - SHAP 값을 사용해 각 피처가 내성 예측 결과에 미치는 영향을 분석했으며, 특정 질량 대 전하비(m/z)가 높은 기여도를 나타냈습니다.
   - 이를 통해 모델이 특정 단백질 피크를 기반으로 항균제 내성을 예측한다는 사실을 확인했습니다.

4. **Fig. 6: 후향적 임상 사례 연구**
   - 63건의 침습성 세균 감염 사례를 분석하여 제안된 분류기의 임상적 이점을 평가했습니다.
   - 분류기 사용 시 항생제 요법의 변경이 가능했던 사례를 강조했으며, 8건의 경우 치료 변경이 환자에게 유익했습니다.
   - 표준 치료와 비교하여 예측 기반 치료가 보다 정확한 치료 제안을 제공했습니다.

---



1. **Fig. 1: MALDI-TOF-Based Antimicrobial Resistance Prediction Workflow**
   - This figure illustrates the full pipeline, from data collection, preprocessing of spectra, machine learning classification, to evaluation of results.
   - Data were collected from the DRIAMS database, and antimicrobial resistance was binarized into susceptible (negative) and intermediate/resistant (positive) labels.
   - Mass spectra data were represented as 6,000 fixed feature vectors, and logistic regression, LightGBM, and MLP models were trained. Metrics such as AUROC and AUPRC were used for evaluation.

2. **Fig. 3: Validation of AUROC Performance on DRIAMS Datasets**
   - This figure shows the AUROC scores for each dataset from DRIAMS-A to D.
   - Higher performance was observed when training and testing were conducted on the same dataset, particularly for *E. coli* and *K. pneumoniae*.
   - Transfer learning across datasets was also tested, with larger datasets (DRIAMS-A) yielding improved performance.

3. **Fig. 5: Feature Importance Analysis with SHAP Values for LightGBM and MLP Models**
   - This figure highlights the features that significantly contribute to predictions in LightGBM and MLP models.
   - SHAP values were used to analyze the impact of each feature on resistance predictions, revealing specific mass-to-charge ratio (m/z) peaks with high contributions.
   - These findings confirm that the models predict antimicrobial resistance based on specific protein peaks.

4. **Fig. 6: Retrospective Clinical Case Study**
   - This figure evaluates the clinical benefits of the proposed classifiers using 63 cases of invasive bacterial infections.
   - Highlighted cases where classifier predictions enabled changes in antibiotic regimens, with 8 cases demonstrating beneficial outcomes.
   - Prediction-based treatment provided more accurate therapeutic suggestions compared to standard care.


<br/>
# refer format:     


@article{weis2022antimicrobial,
  title = {Direct Antimicrobial Resistance Prediction from Clinical MALDI-TOF Mass Spectra Using Machine Learning},
  author = {Caroline Weis and Aline Cuénod and Bastian Rieck and Olivier Dubuis and Susanne Graf and Claudia Lang and Michael Oberle and Maximilian Brackmann and Kirstine K. Søgaard and Michael Osthoff and Karsten Borgwardt and Adrian Egli},
  journal = {Nature Medicine},
  volume = {28},
  number = {1},
  pages = {164--174},
  year = {2022},
  doi = {10.1038/s41591-021-01619-9},
  publisher = {Nature Publishing Group}
}    







Weis, Caroline, Aline Cuénod, Bastian Rieck, Olivier Dubuis, Susanne Graf, Claudia Lang, Michael Oberle, Maximilian Brackmann, Kirstine K. Søgaard, Michael Osthoff, Karsten Borgwardt, and Adrian Egli. "Direct Antimicrobial Resistance Prediction from Clinical MALDI-TOF Mass Spectra Using Machine Learning." Nature Medicine 28, no. 1 (2022): 164–174. https://doi.org/10.1038/s41591-021-01619-9.   


  


