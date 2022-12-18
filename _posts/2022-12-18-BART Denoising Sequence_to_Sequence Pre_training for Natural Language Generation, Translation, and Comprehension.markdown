---
layout: post
title:  "BART Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension"
date:   2022-12-18 20:00:19 +0900
categories: study
---





{% highlight ruby %}
짧은 요약 :  

BART -> 노이즈 제거 오토인코더  
Seq2Seq PLM 이용  
학습 (1) 노이즈 주고   
      (2) 복구(corrupted text -> original)  
트랜스포머 기반 버트, gpt 등 사용  
-노이즈 주는 메서드들 평가->랜덤셔플이 best  
                                -> infilling scheme  ->single noise token 교체  
-TEXT GENERATION에 효과적, TEXT UNDERSTANDING에도 괜찮은 성능  
(RoBERTaW정도의 성능 IN GLUE, SQuAD)  
(SOTA in 대화, QA, 요약-3.5ROGUE, 번역-1.1BLEU)  
-다른 pre-Train 메서드들 적용해봄(효과 검증 위해)  
    
{% endhighlight %}


[링크](https://drive.google.com/drive/folders/1ejDoh5Iyh49gi0zI6Z0Q0h3LFGYxbnTL?usp=sharing)


# 단어정리  
*o: ㅇ  


# 1 Introduction  
*ㅇ