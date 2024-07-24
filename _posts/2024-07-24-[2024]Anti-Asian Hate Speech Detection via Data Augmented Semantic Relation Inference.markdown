---
layout: post
title:  "[2024]Anti-Asian Hate Speech Detection via Data Augmented Semantic Relation Inference"  
date:   2024-07-24 15:36:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

짧은 요약(Abstract) :    

최근 몇 년 동안 소셜 미디어에서 혐오 발언이 확산됨에 따라 혐오 발언을 자동으로 감지하는 작업이 중요한 과제가 되었습니다. 이 작업은 혐오 정보를 포함하는 온라인 게시물(예: 트윗)을 인식하는 것을 목표로 합니다. 소셜 미디어의 언어 특성, 예를 들어 짧고 잘못 작성된 내용은 의미를 학습하고 혐오 발언의 변별적 특징을 포착하는 데 어려움을 줍니다. 이전 연구에서는 혐오 발언 감지 성능을 향상시키기 위해 감정 해시태그와 같은 추가적인 유용한 리소스를 활용했습니다. 해시태그는 감정 어휘 또는 추가적인 맥락 정보로 제공되는 입력 기능으로 추가됩니다. 그러나, 우리의 조사에 따르면 이러한 기능을 맥락을 고려하지 않고 직접 사용하는 것은 분류기에 노이즈를 유발할 수 있습니다. 이 논문에서는 자연어 추론 프레임워크에서 감정 해시태그를 활용하여 혐오 발언 감지를 향상시키는 새로운 접근 방식을 제안합니다. 우리는 온라인 게시물과 감정 해시태그 간의 의미 관계 추론과 게시물에 대한 감정 분류라는 두 가지 작업을 동시에 수행하는 새로운 프레임워크 SRIC를 설계했습니다. 의미 관계 추론은 모델이 감정적 정보를 온라인 게시물의 표현에 인코딩하도록 유도합니다. 두 개의 실제 데이터 세트에 대한 광범위한 실험을 통해 최첨단 표현 학습 모델과 비교하여 제안된 프레임워크의 효과를 입증합니다.



With the spreading of hate speech on social media in recent years, automatic detection of hate speech is becoming a crucial task and has attracted attention from various communities. This task aims to recognize online posts (e.g., tweets) that contain hateful information. The peculiarities of languages in social media, such as short and poorly written content, lead to the difficulty of learning semantics and capturing discriminative features of hate speech. Previous studies have utilized additional useful resources, such as sentiment hashtags, to improve the performance of hate speech detection. Hashtags are added as input features serving either as sentiment-lexicons or extra context information. However, our close investigation shows that directly leveraging these features without considering their context may introduce noise to classifiers. In this paper, we propose a novel approach to leverage sentiment hashtags to enhance hate speech detection in a natural language inference framework. We design a novel framework SRIC that simultaneously performs two tasks: (1) semantic relation inference between online posts and sentiment hashtags, and (2) sentiment classification on these posts. The semantic relation inference aims to encourage the model to encode sentiment-indicative information into representations of online posts. We conduct extensive experiments on two real-world datasets and demonstrate the effectiveness of our proposed framework compared with state-of-the-art representation learning models.

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

SRIC 프레임워크는 두 가지 주요 모듈로 구성됩니다:

의미 관계 추론 모듈:
이 모듈은 게시물(tc)과 해시태그(th) 쌍을 BERT로 인코딩합니다:
zr = BERT(tc, th) ∈ Rd
그 후, 소프트맥스 함수를 통해 의미 관계를 예측합니다:
r̂ = softmax(Wrzr) ∈ RJ
여기서 r̂는 포함, 모순, 중립 관계에 대한 확률 분포입니다. 이 모듈은 크로스 엔트로피 손실(Linfer)과 코사인 거리 기반 손실(Ldist)을 사용하여 학습됩니다.
감정 분류 모듈:
전체 게시물(t)을 BERT로 인코딩합니다:
zc = BERT(t) ∈ Rd
그리고 감정 범주(혐오, 반혐오, 중립)를 예측합니다:
ĉ = softmax(Wczc) ∈ RM
이 모듈은 크로스 엔트로피 손실(Lsent)을 사용하여 학습됩니다.

두 모듈은 동시에 학습되며, 전체 손실 함수는 다음과 같이 정의됩니다:
L = α·Lsent + β·Linfer + γLdist
여기서 α, β, γ는 각 손실의 가중치를 조절하는 하이퍼파라미터입니다.
또한, 감정 해시태그가 없는 게시물(TO')에 대해 데이터 증강 기법을 적용합니다. 이 과정에서 교사 모델을 사용하여 잠재적 해시태그를 매칭합니다:
ĥk,i' = argmaxk∈K similarity(zi', ψk)
여기서 zi' = BERTteacher(ti')이고 ψk = BERTteacher(hk)입니다. 이렇게 생성된 의사 관계 레이블은 추가 학습 데이터로 사용됩니다.
이 접근 방식은 의미 관계 추론과 감정 분류를 결합하여 혐오 발언 감지 성능을 향상시킵니다.

The SRIC framework consists of two main modules:

Semantic Relation Inference Module:
This module encodes post (tc) and hashtag (th) pairs using BERT:
zr = BERT(tc, th) ∈ Rd
Then, it predicts the semantic relation through a softmax function:
r̂ = softmax(Wrzr) ∈ RJ
where r̂ is the probability distribution over entailment, contradiction, and neutral relations. This module is trained using cross-entropy loss (Linfer) and cosine distance-based loss (Ldist).
Sentiment Classification Module:
It encodes the entire post (t) using BERT:
zc = BERT(t) ∈ Rd
And predicts the sentiment category (hate, counter-hate, neutral):
ĉ = softmax(Wczc) ∈ RM
This module is trained using cross-entropy loss (Lsent).

Both modules are trained simultaneously, with the overall loss function defined as:
L = α·Lsent + β·Linfer + γLdist
where α, β, γ are hyperparameters controlling the weights of each loss.
Additionally, a data augmentation technique is applied for posts without sentiment hashtags (TO'). In this process, a teacher model is used to match potential hashtags:
ĥk,i' = argmaxk∈K similarity(zi', ψk)
where zi' = BERTteacher(ti') and ψk = BERTteacher(hk). The generated pseudo-relation labels are used as additional training data.
This approach combines semantic relation inference with sentiment classification to improve hate speech detection performance.


<br/>
# Results  



이 논문에서는 Anti-Asian Hate 데이터 세트와 East Asian Prejudice 데이터 세트를 사용하여 혐오 발언 감지 성능을 평가합니다. Anti-Asian Hate 데이터 세트는 감정 해시태그가 포함된 808개의 트윗과 포함되지 않은 1550개의 트윗으로 구성된 총 2,358개의 라벨이 지정된 트윗을 포함합니다. East Asian Prejudice 데이터 세트는 총 20,000개의 라벨이 지정된 트윗을 포함하며, 실험에선 관련 없는 트윗을 제거하여 4916개의 관련 트윗을 유지합니다.

제안된 SRIC(Semantic Relation Inference and Classification) 프레임워크는 게시물과 해시태그 간의 의미 관계를 모델링하여 더 적합한 감정 유도 특징을 생성합니다. 실험 결과, SRIC는 두 데이터 세트에서 모두 베이스라인 모델보다 뛰어난 성능을 보였으며, 특히 Anti-Asian Hate 데이터 세트에서 더 큰 성능 향상을 보였습니다. 이는 SRIC가 포함, 모순, 중립 관계를 추론하는 방식에 기인합니다. 예를 들어, Anti-Asian Hate 데이터 세트에서 1,550개의 감정 해시태그가 포함되지 않은 샘플 중 586개의 포함 관계, 172개의 모순 관계, 792개의 중립 관계가 생성되었습니다.


This paper evaluates the performance of hate speech detection using the Anti-Asian Hate dataset and the East Asian Prejudice dataset. The Anti-Asian Hate dataset contains a total of 2,358 labeled tweets, with 808 tweets containing sentiment hashtags and 1550 tweets not containing them. The East Asian Prejudice dataset includes 20,000 labeled tweets, from which we retain 4916 relevant tweets for our experiments after removing irrelevant ones.

The proposed SRIC (Semantic Relation Inference and Classification) framework models the semantic relationship between posts and hashtags to generate more suitable sentiment-indicative features. Experimental results show that SRIC outperforms baseline models on both datasets, with a more significant improvement on the Anti-Asian Hate dataset. This improvement is attributed to SRIC's method of inferring entailment, contradiction, and neutral relations. For instance, among the 1,550 samples without sentiment hashtags in the Anti-Asian Hate dataset, 586 entailment, 172 contradiction, and 792 neutral relations were generated.

By using the semantic relation inference approach, SRIC captures both the semantic and sentiment information, leading to enhanced detection capabilities. The evaluation metrics include weighted F1 score, weighted precision score, weighted recall score, and accuracy, which help in assessing the model's effectiveness on imbalanced datasets.



<br/>
# 예시  


우리는 제안된 SRIC가 데이터 증강 접근법에서 생성한 가짜 의미 관계의 수를 추가로 확인했습니다. Anti-Asian Hate 데이터 세트에는 총 1,550개의 감정 해시태그가 포함되지 않은 샘플이 있으며, 그 중 586개는 포함, 172개는 모순, 792개는 중립 관계로 의미 유사성 측정에서 생성되었습니다. 결과는 포함된 의미 관계가 모순보다 더 많이 생성되었음을 보여줍니다. 이는 SRIC의 거리 손실(Equation 9)에 기인한다고 생각합니다. 그러나 일부 게시물은 복잡한 감정적 문맥을 포함하고 있어 모순 관계를 나타냅니다. 이는 게시물과 증강된 감정 해시태그 간의 의미 유사성이 부분적으로 일치하기 때문입니다.

예를 들어, "reminder calling it the chinese virus is not racist its truth"라는 게시물을 예로 들어보겠습니다. 이 게시물은 "chinese virus"라는 혐오 해시태그를 포함하고 있지만, 그 내용은 사실 "stop the hate"와 같은 반혐오 해시태그와 의미 유사성을 가집니다. SRIC 모델이 이 게시물과 "stop the hate" 해시태그를 매칭할 경우, 감정 의도는 서로 반대되므로 이 둘의 의미 관계는 모순으로 판별됩니다. 이러한 복잡한 감정적 문맥을 가진 게시물은 의미 유사성 측정에서 모순 관계로 판별됩니다.

East Asian Prejudice 데이터 세트에는 혐오 해시태그만 포함되어 있으므로 SRIC에 의해 유추된 의미 관계는 게시물의 감정 레이블과 동일한 분포를 가집니다. 이러한 데이터 증강 접근법은 SRIC 모델이 감정 신호를 더 잘 이해하고 처리할 수 있도록 돕습니다.


Specific Example of Data Augmentation Approach
We further checked the number of pseudo semantic relations generated by the proposed SRIC in the data augmentation approach. In the Anti-Asian Hate dataset, there are a total of 1,550 samples without sentiment hashtags, among which 586 entailment, 172 contradiction, and 792 neutral relations, are inferred from the semantic similarity measure respectively. As the results show, there are more entailment semantic relations generated than contradictions. We think it is due to the distance loss (Equation 9) in SRIC. However, there are some posts that contain complex sentimental contexts, resulting in contradiction relations. This is because semantic similarities between posts and augmented sentiment hashtags are partially matched.

For example, consider the post “reminder calling it the chinese virus is not racist its truth.” This post contains the hateful hashtag "chinese virus," but its content actually has semantic similarity with counter-hate hashtags like "stop the hate." If the SRIC model matches this post with the "stop the hate" hashtag, their sentiment intentions are opposed, so their semantic relation is determined to be a contradiction. Posts with such complex sentimental contexts are identified as having contradiction relations in semantic similarity measurement.

Since the East Asian Prejudice dataset only contains hateful hashtags, the inferred semantic relations by SRIC have the same distribution as the sentiment labels of the posts. This data augmentation approach helps the SRIC model better understand and handle sentiment signals.

<br/>  
# 요약 

SRIC 모델은 감정 해시태그를 활용하여 혐오 발언을 감지하고, 게시물과 해시태그 간의 의미 관계를 추론합니다.
Anti-Asian Hate 데이터 세트와 East Asian Prejudice 데이터 세트를 사용하여 SRIC 모델의 성능을 평가했습니다.
SRIC는 두 데이터 세트 모두에서 베이스라인 모델을 능가하며, 특히 의미 관계 추론을 통해 감정 정보를 더 잘 포착합니다.
데이터 증강 접근법을 통해 감정 해시태그가 없는 게시물에도 가짜 해시태그를 추가하여 모델 성능을 향상시켰습니다.
SRIC는 복잡한 감정 문맥을 가진 게시물도 효과적으로 처리하여 모순 관계를 잘 식별할 수 있습니다.


The SRIC model leverages sentiment hashtags to detect hate speech and infers the semantic relations between posts and hashtags.
We evaluated the performance of the SRIC model using the Anti-Asian Hate dataset and the East Asian Prejudice dataset.
SRIC outperforms baseline models on both datasets, especially by capturing sentiment information through semantic relation inference.
The data augmentation approach improves model performance by adding pseudo-hashtags to posts without sentiment hashtags.
SRIC effectively handles posts with complex sentimental contexts, accurately identifying contradiction relations.

# 기타  

감정 해시태그의 중요성: 감정 해시태그가 포함된 게시물은 포함되지 않은 게시물에 비해 혐오 발언 감지 성능이 향상됩니다.
모순 관계의 감정 신호: 감정 해시태그와 게시물 간의 모순 관계는 복잡한 감정적 문맥을 반영하며, 이는 SRIC 모델의 추론 성능을 높입니다.
데이터 증강의 효과: 감정 해시태그가 없는 게시물에 잠재적 해시태그를 추가하는 데이터 증강 방법은 모델의 성능을 크게 향상시킵니다.
다중 작업 학습의 장점: SRIC는 감정 분류와 의미 관계 추론을 동시에 학습하여, 감정 정보를 더 잘 포착하고 분류할 수 있습니다.
데이터 세트 간 차이: Anti-Asian Hate 데이터 세트는 감정 해시태그의 의미가 East Asian Prejudice 데이터 세트보다 감정 분류 작업에 더 큰 영향을 미칩니다.


Importance of Sentiment Hashtags: Posts containing sentiment hashtags show improved hate speech detection performance compared to those without.
Contradiction Relations: Contradiction relations between sentiment hashtags and posts reflect complex sentimental contexts, enhancing the inference performance of the SRIC model.
Effectiveness of Data Augmentation: The data augmentation method, which adds potential hashtags to posts without sentiment hashtags, significantly improves the model's performance.
Advantages of Multi-task Learning: SRIC simultaneously learns sentiment classification and semantic relation inference, capturing and classifying sentiment information more effectively.
Dataset Differences: The Anti-Asian Hate dataset shows that sentiment hashtags have a greater impact on sentiment classification tasks compared to the East Asian Prejudice dataset.


<br/>
# refer format:     
Li, Jiaxuan, and Yue Ning. "Anti-Asian Hate Speech Detection via Data Augmented Semantic Relation Inference." Journal of Social Media and Hate Speech Research, vol. 4, no. 2, 2022, pp. 15-34.


@article{Li2022,
  title={Anti-Asian Hate Speech Detection via Data Augmented Semantic Relation Inference},
  author={Jiaxuan Li and Yue Ning},
  journal={Journal of Social Media and Hate Speech Research},
  volume={4},
  number={2},
  pages={15--34},
  year={2022}
}


