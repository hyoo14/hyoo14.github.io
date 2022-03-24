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



{% highlight ruby %}

# 전반적으로 다루려고 하는 것들

-정렬된 리스트 탐색시 bisect사용  
-별표식을 사용해 언패킹? ->오류 줄이는 방법? ??  
-zip을 사용하여 여러 리스트를 동시에 이터레이션하기!  
-왈러스 연산자? ??  
-f-문자열 (+문자열, 바이트객체?).  
-typing 모듈? ??  
-대입식 ??  
-튜플의 나머지 원소를 모두 잡아내는 언패킹??   

-매타클래스보다는 클래스 데코레이터를 사용하라 ??  
-__init_subclass__() 메서드 ??  
-동시성( 블로킹 I/O에는 스레드를 사용하지만, 병렬화를 위해 스레드를 사용하지 말라) ???  
-강건성과 성능(최적화하기 전에 프로파일링하라) ???  

-제너레이터  
-이터레이터  
-데코레이터  
-왈러스 연산자  
-다형성  
-메타클래스  
-디스크립터  
-동시성/병렬성  
-스레드  
-큐  
-코루틴  
-이벤트 루프  
-비동기  
-프로파일링  
-모킹과 테스트  
-가상환경  
-타입힌트  
-경고  





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

# f-string 사용하라 (better way 04)  

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


# range 보다는 enumerate를 사용하라(yield에 대하여, betterway 07, 220322)

*yield는 generator를 만드는데 사용됨  
**결과값 반환할 때 return대신 사용하는 return과는 다소 다른 방식  
***return list("ABC") vs yield "A" \n yield "B" \n yield "C"  
***결과를 여러번 나누어서 제공  
***return은 list를 반환하고 yield는 generator를 반환  

*그렇다면 generator는 무엇인가?  
**데이터 접근할 때 순차적으로 접근 가능하게 함  
**리스트라면 전부 접근해야해서 예를들어, 만개의 데이터가 들어있는 리스트라면  
처리시간 1초에 필요 메모리 1mb라 가정했을 때, 10000초와 10gb(10000mb)가 필요함  
**하지만 제너레이터는 순차적으로 하나씩 가져올 수 있어서 가져올 때마다 1초와 1mb로 처리  
**메모리 부족하거나 한번에 보여주지 않아도 될 때에 유용(그래서lazy iterator라고 불리기도)  
**이론적으로 무한데이터도 만들 수 있음  
**yield from을 쓰면 리스트를 바로 제너레이터로 변환할 수 있음  
***yield from ["A", "B", "C"] 이런식으로  
**리스트 표현식처럼 제너레이터 표현식도 있음  
***abc = (ch for ch in "ABC") 이런식  

*위와 비슷한 맥락에서 for i in range(len(something_list)): 대신  
it = enumerate(something_list)  
next(it)  
이렇게 쓸 수 있고 이 enumerate는 lazy generator(yield 사용 후 만들어지는)로 이루어져 있음  
**깔끔하게 for i, something in enumerate(something_list): 이렇게 짤 수 있음  
**enumerate에 두번째 파라미터를 지정해줘서 시작도 지정할 수 있음  
***for i, something in enumerate(something_list, 1): 이런식으로..  


# mutable과 imutable 자료구조(220323 wednesday)
*가역적 비가역적.. 변하는 자료구조와 안 변하는 자료구조  
**가역적인 것은 대부분의 자료구조들 list, dictionary, set  
**비가역적인 것은 tuple.  
***숫자형, 문자형 지정 변수도 비가역적..  
***비가역적 자료구조들은 변하지 않음 -> 예를 들어, x=3이고 y=x라 했을 때 y += 1을 해도 x는 유지됨  
**mutable은 call-by-reference, immutable은 call-by-value로 볼 수 있음  
**참고문헌: https://ledgku.tistory.com/54  


# 복잡한 식을 쓰는 대신 도우미 함수를 작성하라(Better way 5, 220324 thursday)
*if/else 조건식으로 간결하게 표현도 가능한 경우 있음  
**사용예  
{% highlight ruby %}
red_str = my_values.get('빨강', [""])  
red = int(red_str[0]) if red_str[0] else 0  
{% endhighlight %}

*하지만 두세번만 반복되는 경우에도 함수 따로 만드는 것 권장  
**사용예  
{% highlight ruby %}
def get_first_int(values, key, efault=0):  
    found = values.get(key, [""])  
    if found[0]:  
        return int(found[0])  
    return default  
{% endhighlight %}

