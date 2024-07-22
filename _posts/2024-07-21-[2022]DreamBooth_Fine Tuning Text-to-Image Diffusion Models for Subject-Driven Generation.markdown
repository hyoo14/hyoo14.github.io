---
layout: post
title:  "[2022]DreamBooth Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation"  
date:   2024-07-21 23:29:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

짧은 요약(Abstract) :    



대규모 텍스트-이미지 모델은 AI 진화에서 놀라운 도약을 이루어, 주어진 텍스트 프롬프트로부터 고품질의 다양한 이미지를 합성할 수 있게 되었습니다. 그러나 이러한 모델은 주어진 참조 세트에 있는 주제의 외형을 모방하고, 다른 문맥에서 새로운 표현을 합성하는 능력이 부족합니다. 이 연구에서는 텍스트-이미지 확산 모델의 "개인화"를 위한 새로운 접근 방식을 제안합니다. 주제의 몇 장의 이미지를 입력으로 제공받아, 사전 학습된 텍스트-이미지 모델을 미세 조정하여 특정 주제와 고유 식별자를 결합하도록 학습합니다. 일단 주제가 모델의 출력 도메인에 임베딩되면, 고유 식별자를 사용하여 다양한 장면에서 문맥화된 주제의 새로운 사실적인 이미지를 합성할 수 있습니다. 모델에 내장된 의미론적 사전과 새로운 자가발생 클래스-특정 사전 보존 손실을 활용함으로써, 우리 기술은 참조 이미지에 나타나지 않는 다양한 장면, 자세, 관점 및 조명 조건에서 주제를 합성할 수 있게 합니다. 우리는 주제 재문맥화, 텍스트 기반 보기 합성, 예술적 렌더링 등 여러 이전에는 접근할 수 없었던 작업에 우리 기술을 적용합니다. 또한 이 새로운 주제 주도 생성 작업을 위한 새로운 데이터 세트와 평가 프로토콜을 제공합니다. 프로젝트 페이지: https://dreambooth.github.io/

Large text-to-image models achieved a remarkable leap in the evolution of AI, enabling high-quality and diverse synthesis of images from a given text prompt. However, these models lack the ability to mimic the appearance of subjects in a given reference set and synthesize novel renditions of them in different contexts. In this work, we present a new approach for “personalization” of text-to-image diffusion models. Given as input just a few images of a subject, we fine-tune a pretrained text-to-image model such that it learns to bind a unique identifier with that specific subject. Once the subject is embedded in the output domain of the model, the unique identifier can be used to synthesize novel photorealistic images of the subject contextualized in different scenes. By leveraging the semantic prior embedded in the model with a new autogenous class-specific prior preservation loss, our technique enables synthesizing the subject in diverse scenes, poses, views and lighting conditions that do not appear in the reference images. We apply our technique to several previously-unassailable tasks, including subject recontextualization, text-guided view synthesis, and artistic rendering, all while preserving the subject’s key features. We also provide a new dataset and evaluation protocol for this new task of subject-driven generation. Project page: https://dreambooth.github.io/




* Useful sentences :  


{% endhighlight %}  

<br/>

[Paper link](https://drive.google.com/drive/folders/1eReg89IZgDo6LIpl9_HvE08TCwGMuYc3?usp=sharing)  
[~~Lecture link~~]()   

<br/>

# 단어정리  
*  
 
<br/>
# Methodology    




이 논문에서는 텍스트-이미지 확산 모델을 사용하여 특정 주제를 생성하기 위해 개인화하는 방법을 제안합니다. 여기서는 주어진 주제의 몇 장의 이미지만을 사용하여 모델을 미세 조정하는 방법을 다룹니다.

**3.1 텍스트-이미지 확산 모델**

확산 모델은 가우시안 분포에서 샘플링된 변수를 점진적으로 디노이징하여 데이터 분포를 학습하는 확률적 생성 모델입니다. 구체적으로, 사전 학습된 텍스트-이미지 확산 모델 \( \hat{x}_\theta \)는 초기 노이즈 맵 \( \epsilon \sim \mathcal{N}(0, I) \)와 텍스트 인코더 \( \Gamma \) 및 텍스트 프롬프트 \( P \)를 사용하여 생성된 조건 벡터 \( c = \Gamma(P) \)를 사용하여 이미지를 생성합니다.

**3.2 텍스트-이미지 모델의 개인화**

우리의 첫 번째 작업은 모델의 출력 도메인에 주제 인스턴스를 삽입하여 모델이 주제의 다양한 새로운 이미지를 생성할 수 있도록 하는 것입니다. 이를 위해 우리는 주어진 주제를 희귀 토큰 식별자로 나타내고 사전 학습된 확산 기반 텍스트-이미지 프레임워크를 미세 조정하는 기술을 제안합니다. 주어진 주제의 몇 장의 이미지와 고유 식별자가 포함된 텍스트 프롬프트를 사용하여 텍스트-이미지 모델을 미세 조정합니다.

**3.3 클래스-특정 사전 보존 손실**

우리는 주제 충실도를 극대화하기 위해 모델의 모든 계층을 미세 조정하는 것이 가장 좋은 결과를 가져온다고 경험적으로 발견했습니다. 이를 위해 우리는 자가 발생 클래스-특정 사전 보존 손실을 제안합니다. 이 방법은 모델이 소수의 학습 이미지에 미세 조정되기 전에 자체 생성 샘플로 모델을 감독하여 클래스 사전을 유지하도록 합니다.


In this paper, we propose a method for personalizing text-to-image diffusion models to generate specific subjects using only a few images of the subject. We detail the technique to fine-tune the model using these few images.

**3.1 Text-to-Image Diffusion Models**

Diffusion models are probabilistic generative models trained to learn a data distribution by gradually denoising a variable sampled from a Gaussian distribution. Specifically, a pretrained text-to-image diffusion model \( \hat{x}_\theta \) generates an image \( x_{\text{gen}} = \hat{x}_\theta(\epsilon, c) \), given an initial noise map \( \epsilon \sim \mathcal{N}(0, I) \) and a conditioning vector \( c = \Gamma(P) \) generated using a text encoder \( \Gamma \) and a text prompt \( P \).

**3.2 Personalization of Text-to-Image Models**

Our first task is to implant the subject instance into the output domain of the model so that we can query the model for varied novel images of the subject. To that end, we propose a technique to represent a given subject with rare token identifiers and fine-tune a pretrained, diffusion-based text-to-image framework. We fine-tune the text-to-image model with the input images and text prompts containing a unique identifier followed by the class name of the subject (e.g., “A [V] dog”).

**3.3 Class-specific Prior Preservation Loss**

We find that fine-tuning all layers of the model yields the best results for maximum subject fidelity. To achieve this, we propose an autogenous class-specific prior preservation loss. This method supervises the model with its own generated samples to retain the class prior once the few-shot fine-tuning begins.



<br/>
# Results  




**4. 실험**

이 섹션에서는 우리의 방법을 사용한 실험과 응용 사례를 보여줍니다. 우리의 방법은 주제 인스턴스의 텍스트 기반 의미적 수정을 가능하게 하며, 여기에는 재문맥화, 주제 속성의 수정(예: 재질 및 종), 예술적 렌더링, 관점 수정 등이 포함됩니다. 중요한 것은 이러한 수정 전반에 걸쳐 주제의 고유한 시각적 특징을 유지할 수 있다는 점입니다. 재문맥화 작업에서는 주제 특징이 수정되지 않지만, 외형(예: 자세)은 변경될 수 있습니다. 더 강한 의미적 수정 작업, 예를 들어 주제와 다른 종/객체의 교차 작업의 경우, 수정 후에도 주제의 주요 특징은 유지됩니다.

**4.1 데이터셋 및 평가**

우리는 백팩, 봉제 인형, 개, 고양이, 선글라스, 만화 캐릭터 등 30개의 고유 객체와 애완동물을 포함한 데이터셋을 수집했습니다. 각 주제를 객체와 살아있는 주제/애완동물로 분류했습니다. 30개의 주제 중 21개는 객체이고, 9개는 살아있는 주제/애완동물입니다. 우리의 데이터셋과 평가 프로토콜은 주제 주도 생성의 미래 평가에 사용될 수 있도록 프로젝트 웹페이지에 공개했습니다.

**4.2 비교**

우리의 결과를 Textual Inversion과 비교했습니다. DreamBooth는 주제 충실도와 프롬프트 충실도 모두에서 Textual Inversion보다 높은 점수를 기록했습니다. 사용자 연구에서도 DreamBooth가 주제 충실도와 프롬프트 충실도 모두에서 압도적인 선호를 받았습니다.

**4.3 절단 연구**

우리의 제안된 사전 보존 손실(PPL)이 언어 드리프트를 방지하고 다양성을 유지하는 데 효과적임을 발견했습니다. 또한, 클래스 명칭을 올바르게 사용할 때 주제 충실도가 높아진다는 것을 알 수 있었습니다.

**4.4 응용**

우리의 방법은 다양한 문맥에서 주제를 재문맥화하고, 예술적 렌더링을 생성하며, 주제 속성을 수정하는 등의 작업을 수행할 수 있습니다. 예를 들어, "유명한 화가의 스타일로 그린 [V] [클래스 명사]의 그림"이라는 프롬프트를 사용하여 예술적 렌더링을 생성할 수 있습니다.

**4.5 한계**

우리의 방법이 모든 주제에서 동일한 성능을 발휘하지 않으며, 일부 주제는 학습하기 더 어렵다는 것을 발견했습니다. 또한, 일부 생성된 이미지에는 주제 특징이 왜곡될 수 있습니다.



**4. Experiments**

In this section, we show experiments and applications. Our method enables a large expanse of text-guided semantic modifications of our subject instances, including recontextualization, modification of subject properties such as material and species, art rendition, and viewpoint modification. Importantly, across all of these modifications, we are able to preserve the unique visual features that give the subject its identity and essence. If the task is recontextualization, then the subject features are unmodified, but appearance (e.g., pose) may change. If the task is a stronger semantic modification, such as crossing between our subject and another species/object, then the key features of the subject are preserved after modification.

**4.1 Dataset and Evaluation**

We collected a dataset of 30 subjects, including unique objects and pets such as backpacks, stuffed animals, dogs, cats, sunglasses, cartoons, etc. We separate each subject into two categories: objects and live subjects/pets. 21 of the 30 subjects are objects, and 9 are live subjects/pets. We make our dataset and evaluation protocol publicly available on the project webpage for future use in evaluating subject-driven generation.

**4.2 Comparisons**

We compare our results with Textual Inversion. DreamBooth achieves higher scores for both subject and prompt fidelity compared to Textual Inversion. In a user study, DreamBooth was overwhelmingly preferred for both subject fidelity and prompt fidelity.

**4.3 Ablation Studies**

We found that our proposed prior preservation loss (PPL) is effective in preventing language drift and maintaining diversity. Additionally, we observed that using the correct class noun improves subject fidelity.

**4.4 Applications**

Our method can recontextualize subjects in various contexts, generate artistic renditions, and modify subject properties. For example, with the prompt “a painting of a [V] [class noun] in the style of [famous painter]”, we can generate artistic renditions of our subject.

**4.5 Limitations**

We observed that some subjects are easier to learn than others, and the fidelity of some generated images may vary. Some generated images might contain hallucinated subject features depending on the strength of the model prior and the complexity of the semantic modification.


<br/>
# 예시  




이 논문에서 제안하는 DreamBooth 모델은 텍스트와 이미지를 기반으로 한 새로운 주제 생성 방식입니다. 예를 들어, DreamBooth는 특정한 꽃병 이미지를 바탕으로 다양한 배경과 상황에서 그 꽃병을 생성할 수 있습니다. "눈 속의 꽃병", "해변의 꽃병", "정글 속의 꽃병", "에펠탑 배경의 꽃병"과 같은 프롬프트를 사용하여, DreamBooth는 주제와 프롬프트 충실도가 높은 이미지를 생성합니다. 이를 통해, DreamBooth는 텍스트 프롬프트에 따라 주제의 외관을 유지하면서도 새로운 문맥에서 다양한 이미지를 생성할 수 있음을 보여줍니다.



In this paper, the proposed DreamBooth model offers a novel way of generating subjects based on text and images. For instance, DreamBooth can take an image of a specific vase and generate various instances of that vase in different backgrounds and contexts. Using prompts like "a vase in the snow," "a vase on the beach," "a vase in the jungle," and "a vase with the Eiffel Tower in the background," DreamBooth produces images with high subject and prompt fidelity. This demonstrates that DreamBooth can maintain the appearance of the subject while creating diverse images in new contexts according to the text prompts.



<br/>  
# 요약 


DreamBooth는 텍스트-이미지 확산 모델을 사용하여 특정 주제를 생성하기 위해 개인화하는 방법론을 제안합니다. 몇 장의 주제 이미지를 입력받아, 모델을 미세 조정하여 주제와 고유 식별자를 결합합니다. 이를 통해 모델은 다양한 문맥에서 주제의 새로운 사실적인 이미지를 생성할 수 있습니다. 사용자 연구 결과, DreamBooth는 주제 충실도와 프롬프트 충실도에서 기존 방법보다 우수한 성능을 보였습니다. 이는 DreamBooth가 주제의 고유한 시각적 특징을 유지하면서도 다양한 상황에서 새로운 이미지를 생성할 수 있음을 입증합니다.


DreamBooth proposes a methodology for personalizing text-to-image diffusion models to generate specific subjects. By inputting a few images of the subject, the model is fine-tuned to bind the subject with a unique identifier. This allows the model to generate novel photorealistic images of the subject in various contexts. User study results showed that DreamBooth outperformed existing methods in both subject fidelity and prompt fidelity. This demonstrates that DreamBooth can maintain the unique visual features of the subject while creating new images in diverse situations.



# 기타  


논문에서는 DreamBooth 모델의 성능 평가를 위해 사용자 연구를 수행했습니다. 사용자는 생성된 이미지의 주제 충실도(subject fidelity)와 프롬프트 충실도(prompt fidelity)를 평가했습니다. 구체적인 평가 방법과 사용자 연구 결과는 다음과 같습니다.

### 평가 방법


**사용자 연구**

주제 충실도 평가를 위해, 72명의 사용자가 25개의 비교 질문에 답변하였습니다. 각 질문에는 참조 이미지 세트와 두 개의 생성된 이미지(각각 DreamBooth와 Textual Inversion 방법으로 생성됨)가 포함되어 있습니다. 사용자는 "어느 이미지가 참조 항목의 정체성을 가장 잘 재현하였는가?"라는 질문에 답변했습니다. 프롬프트 충실도 평가에서도 동일한 방식으로 진행되었으며, 사용자는 "어느 이미지가 텍스트 프롬프트를 가장 잘 설명하는가?"라는 질문에 답변했습니다.

**평가 결과**

사용자 연구 결과, DreamBooth는 주제 충실도와 프롬프트 충실도 모두에서 Textual Inversion보다 높은 평가를 받았습니다. DreamBooth가 생성한 이미지가 참조 항목의 특징을 더 잘 유지하며 텍스트 프롬프트에 더 잘 맞는다고 평가되었습니다.


**User Study**

For the evaluation of subject fidelity, 72 users answered 25 comparative questions. Each question included a set of reference images and two generated images (one from DreamBooth and one from Textual Inversion). Users were asked to answer the question, "Which of the two images best reproduces the identity of the reference item?" The same procedure was followed for prompt fidelity, where users answered the question, "Which of the two images is best described by the reference text?"

**Evaluation Results**

The user study results showed that DreamBooth was overwhelmingly preferred for both subject fidelity and prompt fidelity compared to Textual Inversion. DreamBooth-generated images were found to better maintain the characteristics of the reference item and fit the text prompts more accurately.



<br/>
# refer format:     
Ruiz, Nataniel, Li, Yuanzhen, Jampani, Varun, Pritch, Yael, Rubinstein, Michael, & Aberman, Kfir. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). Available at https://doi.org/10.48550/arXiv.2208.12242  

@inproceedings{ruiz2023dreambooth,
  title={DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation},
  author={Ruiz, Nataniel and Li, Yuanzhen and Jampani, Varun and Pritch, Yael and Rubinstein, Michael and Aberman, Kfir},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023},
  url={https://doi.org/10.48550/arXiv.2208.12242}
}


