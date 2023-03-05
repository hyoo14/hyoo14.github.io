---
layout: post
title:  "[2020]Dense Passage Retrieval for Open-Domain QA"
date:   2023-03-04 13:11:42 +0900
categories: study
---





{% highlight ruby %}
짧은 요약 :  

*Open domain QA는 효율적 검색 내포된 task  
**기존 tf-idf, BM25 쓰임(sparse vector)   
**dense representation만 사용으로 더 효과적 성능 가능   
**실험상 성능 sota 찍음, qa서 루신-bm25보다 9~19% 향상  
   
{% endhighlight %}


[Paper with my notes](https://drive.google.com/drive/folders/1HvR4KPgOxNDmgoDysz9t3EE1z1Wl-o6Q?usp=sharing)  


[Lecture link](https://slideslive.com/38939151/dense-passage-retrieval-for-opendomain-question-answering)  


# 단어정리  
* defacto: 사실상  
* factoid: 사실  
* inverted: 거꾸로  
* inverted index: 역 인덱스, 책의 맨 뒤에 있는 주요 키워드에 대한 내용이 몇 페이지에 있는 "찾아보기 페이지"에 비유할 수 있음  
* inverse: 역의, 반대의  
* surrogate: 대리의, 대용의  
* fraction: 부분, 일부, 분수  
* schemes: 계획, 제도, 책략, 기획, 설계, 기구  
* conjecture: 추측, 억측, 어림짐작  
* lexical: 어휘의  
* lexicon: 어휘, 어휘목록, 사전  
* block: 블록  
* middle block: 중간 블록  
* non-iid:(non-independent and identically distributed) 데이터가 비독립적이고 동일하지 않게 분산. 즉 데이터는 종속적일 수 있고 동일한 확률분포를 갖고있지 않음    


   

# 1 Introduction  
* 오픈도메인 QA는 대용량 호부군 중에서 정답을 찾는 것  
**초기 매우 복잡한 컴포넌트가 사용  
**시간이 자나며 2단계로 단순화  
***검색(추림)  
***읽기(정답 찾기)  
**머신리더 없을 시 성능 하향 등 개선이 필요해짐  
* 검색 일반적으로 tf-idf/bm25 같은 keyword 위주이지만 dense representation 사용해서 동의어/파라프이징 잘 찾아줄 수 있음  
**dense vector는 학습 가능하고 엔드테스크 튜닝도 가능  
* dense vector학습 데이터가 많이 필요하지만 tf-idf/bm25 보다 성능 안 좋다 여겨졌었음  
**하지만 ORQA가 인식 깨고 sota성능 보임  
**masked sentence 추가 P-T 이용(ORQA)  
**F-T로 QA쌍 학습    
