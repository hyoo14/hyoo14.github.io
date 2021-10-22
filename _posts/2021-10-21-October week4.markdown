---
layout: post
title:  "October week4"
date:   2021-10-21 16:30:10 +0900
categories: study
---




{% highlight ruby %}
짧은 요약 :

정리를 위해 시작해보았습니다.

{% endhighlight %}

2021년 10월 21일

버트학습 -> 재무교과서 x(pdf 파일 제공 찾기 어려움) -> KIFRS 기준서 & 해설서 OR KGAAP 기준서 & 해설서

PDF 읽기 -> pdftotext

근데 pip install 이 안 됨..

https://pythonq.com/so/python/362057 사이트 참고

sudo apt-get update
sudo apt-get install build-essential libpoppler-cpp-dev pkg-config python-dev

설치 후 install 됨.

사용법:
import pdftotext
filename = ".pdf"
with open(filename, "rb") as f:
    pdf = pdftotext.PDF(f)

for page in pdf:
    print(page)


KIFRS 기준서 다운로드.. 너무 많...
bert 모델부터 볼까...



2021년 10월 22일


dataset, tokenizer, data loader 부분 보기

근데 pretrain 내용은 
https://beomi.github.io/2020/02/26/Train-BERT-from-scratch-on-colab-TPU-Tensorflow-ver/ 이거 보고 하는 것이 좋은 듯..
