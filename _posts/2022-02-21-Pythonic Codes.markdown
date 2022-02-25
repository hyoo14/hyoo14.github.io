---
layout: post
title:  "Pythonic Codes"
date:   2022-02-21 18:55:10 +0900
categories: python study
---




{% highlight ruby %}
짧은 요약 :

Pythonic한 코딩을 위한 정리  

{% endhighlight %}


# PEP8 (Python Enhancement Proposal#8)  
*공백관련(220221)  
**라인 길이는 79문자 이하  
**들여쓰기는 4칸 스페이스로  
**긴 식을 다음 줄에 이어서 쓸 경우 다음줄에 4스페이스 들여쓰기  
**각 함수와 클래스 사이에는 빈줄 두 줄  
**클래스 안에서 메서드와 메스드 사이는 빈칸 한 줄  
**딕셔너리의 키와 콜론(:) 사이에는 공백 없고, 한줄 안에 키-벨류 넣을 경우 콜론과 벨류 사이 공백하나  
**변수대입서 '=' 전후 스페이스 하나  
**타입 표기를 덧붙이는 경우에는 변수 이름과 콜론 사이에 공백을 넣지 않고, 콜론과 타입 정보 사이에 스페이스 하나  
***num = 1  # type: int (예시) -> 타입 힌트 표기하는 것을 의미함  


*명명관련(220222)  
**함수, 변수, 애트리뷰트(클래스 내부 함수, 변수)는 lowercase_underscore처럼 소문자와 언더바로  
**보호되야할 인스턴스의 애트리뷰트는 _leading_underscore처럼 앞에 언더바 사용  
**private(한 클래스 안에서만 쓰이고 다른 곳에서는 쓰면 안 되는) 인스턴스 애트리뷰트는 __leading_underscore처럼 앞에 언더바 두개 사용  
**클래시는 CapitalizedWord처럼 PascalCase 사용  
**모델(함수, 클래스 모아놓은 파일) 수준의 상수는 ALL_CAPS처럼 모든 글자를 대문자로 하고 단어와 단어 사이 언더바로  
**클래스에 들어 있는 인스턴스 메서드는 호출 대상 객체를 가리키는 첫번째 인자의 이름으로 반드시 self를 사용  
**클래스 메서드는 클래스를 가리키는 첫 번째 인자의 이름으로 반드시 cls를 사용  
(참고로 클래스는 설계도, 객체는 설계도로 구현한 대상, 인스턴스는 실제 소프트웨어로 구현된 것)  
(참고로 클래스 메서드는 인스턴스를 만들어 실체화 하지 않아도 클래스를 통해 직접적으로 호출 할 수 있음, cls로 클래스 호출)  
(참고로 인스턴스 메서드는 클래스를 통해 호출할 수 없고, 클래스의 인스턴스를 만들어 실체화 하여 생성된 인스턴스를 통해서 호출할 수 있음, self로 인스턴스 호출)  
(참고로 정적 메서드는 인스턴스, 클래스 애트리뷰트 접근/호출 못 함)  
 

*expression and statement(식과 문)  
**긍정식을 부정하지 말고 부정을 내부에 넣어라(if not a is b --> if a is not b)  
**빈 컨테이너(리스트, 딕셔너리, 세트), 빈 시퀀스(스트링, 리스트, 튜플) 체크할 때는 길이 체크보다는 False취급 사실 활용하라(if len(something) == 0 --> if not 컨테이너)  
**마찮가지로 비어있지 않은 것을 체크할 때도 True 취급 활용(if len(someting) > 0 --> if 컨테이너)  
**한줄짜리 if문, for 루프, while 루프, except 문 사용하지 말고 여러 줄에 나눠라  
**식을 한줄 안에 다 쓸 수 없는 경우 식을 괄호로 둘러싸고 줄바꿈과 들여쓰기를 추가해서 가독성 높여라  
**여러줄에 걸쳐 식을 쓸 때는 줄이 계속된다는 표시를 하는 \문자 보다는 괄호를 사용하라  


*임포트  
**import, from x import y는 항상 파일 맨 앞에 위치  
**모듈 임포트시 절대 경로를 사용하라(현재 기준으로 상대경로 사용 x, 예를 들어 bar패키지 foo라면 현재 bar패키지 안에 있더라도 from bar import foo라고 해야)  
**반드시 상대적 경로를 임포트해야 한다면 from . import foo처럼 사용  
**임포트를 적을 때는 표준라이브러리 모듈, 서드 파티 모듈, 사용자가 직접 만든 모듈 선서로 섹션을 나눠라, 그리고 알파벳 순서로 임포트하라  


# f-string 사용하라  
*기존의 c style tuple or dictionary과 format 보다 간결하고, 명확하게 표현됨  
**비교  
{% highlight ruby %}
key, value = 'my_values', 1.234  
f_string = f'{key:<10}={value:.2f}'  
c_tuple = '%-10s = %.2f' % (key, value)  
c_dict = '%(key)-10s = %(value).2f' %{'key': key, 'value': value}  
str_args = '{:<10} = {:.2f}'.format(key, value)  
{% endhighlight %}

**사용예  
{% highlight ruby %}
key, value = 'my_values', 1.234  
places, number = 3, 1.23456  
formatted = f'{key} = {value}'  
formatted = f'내가 고른 숫자는 {number:.{places}f}' #하드코딩 대신 변수를 사용해 형식 문자열 안에 파라미터화함  
{% endhighlight %}



