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


# 인덱스 대신 언패킹해라(Better way 6, 220324 thursday)
*지양하는 사용예  
{% highlight ruby %}
listA = [("korean", "kimchi"), ("italian","pasta")]
for i in range(len(listA)):
    print(listA[i])
{% endhighlight %}

*지향하는 사용예  
{% highlight ruby %}
listA = [("korean", "kimchi"), ("italian","pasta")]
for (national, food) in listA:
    print( (national, food) )
{% endhighlight %}


# Better way 8,9,10(Friday, 220325 )
*zip을 사용하라  
**(참고)리스트 컴프리헨션 사용하면 새로운 리스트 만들기 편함  
***counts = [len(n) for n in names] #names = ['Zelenskyy', '우크라이나'] #결과=[9,5]  
**두개 리스트 접근할 때 zip 편함  
***사용예  
{% highlight ruby %}
for name, count in zip(names, counts):
    print(f'{name}: {count}')

{% endhighlight %}
****다만 길이가 다를 경우에는 작은 길이까지만 고려해줌  
****이 경우, zip_longest를 사용하면 짧은 경우에는 None을 넣어줌  


*loop 이후에 else블록 사용하지 말아라  
*월러스 연산자 ( := ) 사용해서 대입 반복 피하라  
**월러스 연산자는 대입연산자로 대입 해줌.. 이후 조건 체크 바로 해줄 수 있어서 편함  
***사용예  
{% highlight ruby %}
while fresh_fruit := pick_fruit(): #fresh_fruit에 값 넣어줌, 만약 없으면 break됨  


if (count := fresh_fruit.get('레몬', 0) ) >= 2: #카운트에 개수 세서 넣어줌. 뒤에 조건문 붙여줘서 한줄로 편하게 처리  


{% endhighlight %}



# Bettery way 27 -map과 filter 대신 comprehension을 사용하라! (Thursday, 220609)  - (오랜만에 업데이트)  
*map과 filter 대신 comprehension  


{% highlight ruby %}

-다음 task를 수행할 때..  
a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  
squares = []  
for x in a:  
    if x % 2 == 0:
        squares.append(x**2)  
print(squares)  


-map, filter 이용 대신  
alt = map(lambda x: x**2, filter(lambda x: x % 2 == 0, a))  
assert even_squares = list(alt)  


-comprehension을 사용!  
squares = [x**2 for x in a if x % 2 == 0]  


-dict와 set도 각각 dict comprehension과 set comprehension이 있어 사용 가능!  
even_squares_dict = {x: x**2 for x in a if x % 2 == 0}# {2: 4, 4: 16, 6: 36, 8: 64, 10: 100}    
threes_cubed_set = {x**3 for x in a if x % 3 == 0}# {216, 729, 27}     



{% endhighlight %}



# Better way 28, 29  
*comprehension의 하위 식은 3개 미만으로 해라. 왜냐하면 너무 복잡해지기 때문.  
*comprehension에 왈러스 연산자( := )와 함수 호출을 사용하여 가독성 높일 수 있음.  


# Bettery way 30 (Wednesday, 220615)  
*제너레이터(generator, yield를 사용하여 반환하는 함수)를 만들어서 사용하면  
 함수가 실제로 실행되지 않고 즉시 이터레이터를 반환, 메모리 크기를 제한할 수 있어  
 입력 길이가 아무리 길어도 처리 가능  
**단, 반환하는 이터레이터에 상태가 있기 때문에 호출하는 쪽에서 재사용이 불가능함  


# Better way 31 (Wednesday, 220615)  
*이터레이터 결과는 단 한번만 만들어져서 재호출 시 아무 결과도 안 옴  
**(StopIteration 예외가 되는 것임)  
*카피해서 리스트에 넣어두는 해결책이 있으나 입력이 너무 길 경우 메모리를 많이 먹으므로 비추  
*대안으로 호출 때마다 이터레이터를 받아주는 방법이 있음(근데 람다식 쓰는 것이 보기는 안좋다고함)  
**def normalize_func(get_ter):
    total = sum(get_iter()) # 새 이터레이터    
    result = []  
    for value in get_iter(): # 새 이터레이터    
        percent = 100 * value / total    
        result.append(percent)  
    return result  

  percentages = normalize_func(lambda: read_visits(path))  
*더 나은 방법으로 이터레이터 프로토콜을 구현한 새로운 컨테이너 클래스를 제공하는 것이 있음  
**class ReadVisits:
    def __init__(self, data_path):  
        self.data_path = data_path      
    def __iter__(self):  
        with open(self.data_path) as f:  
            for line in f:  
                yield int(line)  
  visits = ReadVisits(path)  #클래스를 불러와서 객체 정의    
  percentages = normalize(visits) #객체를 보내줌  
**유일한 단점은 입력 데이터를 여러번 읽는 다는 것.. 그래도 이게 제일 나은 듯?  
*예외처리의 경우 - 반복 가능한 이터레이터인지 검사  
**if iter(numbers) is numbers: #반복 안 가능?  
      raise TypeError('컨테이너를 제공해야 합니다')  
*예외처리 대안 - collection.abc 내장모듈 instance9 사용하여 검사  
**from collections.abc import Iterator  
  if instance(numbers, Iterator): #반복 가능한 이터레이션인지 검사  
      raise TypeError('컨테이너를 제공해야 합니다')  



# Better way 32, 33 (Thursday, 220616)  
# 긴 리스트 컴프리헨션보다는 제너레이터를, yield from을 사용해 여러 제너레이터 합성을
*리스트컴프리핸션 입력 사이즈 커지면 메모리 많이 사용  
**제너레이터 식으로 해결  
***next함수로 다음 값 가져올 수 있음  
*제너레이터 식은 두 제너레이터 식을 합성할 수도 있음  


*yield from 사용으로 가독성도 높여줌  
** for delta in move(4, 5.0): yield delta --> yield from move(4, 5.0)  
**yield from은 근본적으로 인터프리터가 for루프 내포해서 성능 더 좋아짐  


# Bettery way 34, 35, 36 (Saturday, 220618)  
# send로 제너레이터에 데이터를 주입하지 말라, 제너레이터 안에서 throw로 상태 변화하지 말라, 이터레이터나 제너레이터 다룰 때 itertools 사용하라  


*yield 식 사용한 제너레이터 함수는 이터레이션 출력이 가능하지만 단방향임  
**그래서 send메서드 사용해서 양방향 채널 만들어줄 수 있음  
***하지만 방금 시작한 제너레이터는 최초 send 호출 때 보낼 인자가 없고 None만 뜸  
****첫 값을 None을 주는 해결책이 있고, 코드가독성을 위해 yield from을 사용할 수도 있음  
*****하지만 이 경우 다음 yield from 시작 때마다 None뜸.  
******그래서 그냥 next를 사용하는 것을 권장  


*제너레이터 안에서 Exception던질 수 있는 throw 메서드 있음-마지막 yield 실행에서 예외 발생  
**가독성 많이 떨어짐. __iter__메서드 포함 클래스를 정의하여 사용하는 것이 더 나은 대안  


*복잡한 이터레이션 코드 itertools에 거의 있음->help(itertools)해보시오  
**연결 - chain:순차적으로 함침, repeat:반복, cycle:사이클반복, tee:병렬적으로 리스트 만들어줌, zip_logest:두 이터레이터 중 짧은 쪽의 경우 지정값 넣어서 합쳐줌  
**원소 필터링 - islice:인덱싱으로 슬라이싱, takewhile:False나올 때까지 반환, dropwhile:False시작부터 True 전까지 반환, filterfalse:False인 것만 반환  
**원소 조합 - accumulate:축적 반환, product:데카르트곱 반환, permutations:순열 반환, combinations:조합 반환, combinations_with_replacement:중복 조합 반환  



# Better way 52 (Sunday, 220619)  
# 자식 프로세스를 관리하기 위해 subprocess를 사용하라  

*subprocess 모듈을 사용해 자식 프로세스를 실행하고 입력과 출력 스트림을 관리할 수 있음   
*자식 프로세스는 파이썬 인터프리터와 병렬로 실행되므로 CPU코어 최대 활용 가능(not컨커런트)  
*간단 자식 프로세스 실행은 run함수, 파이프라인 필요한 경우 Popen 클래스 사용해야.  
*자식프로세스 멈추거나 교착 방지하려면 communicate 메서드에 대해 timeout 사용하면 됨  


# Better way 53, 54 (Monday, 220620)  
# 블로킹 I/O의 경우 스레드를 사용하고 병렬성을 피하라, 스레드에서 데이터 경합을 피하기 위해 Lock을 사용하라  
*GIL(Global Interpreter Lock)은 CPython 자체와 사용하는 C 확장 모듈이 실행되면서 인터럽트가 함부로 발생하는 걸 방지  
**(참고로 source -> bytecode --by interpreter)  
*GIL 땜에 멀티스레딩이 코드 실행단에서는 성능향상이 안 됨  
*그럼에도 멀티스레딩하는 이유는 동시성(컨커런시 concurency) 구현하기 쉽고 I/O 블로킹은 잘 되기 때문  
**(GIL은 프로그램 병렬 실해은 막지만 시스템콜에는 영향 못 끼침)  
*즉, multi threading은 코드를 가급적 손보지 않고 블리코이 I/O를 병렬로 실행하고 싶을 때 사용  



*GIL의 락은 코드 딴이지, 자료구조 접근 까지는 못 막아줌  
*파이썬 스레딩이 자료구조를 접근하고 일시 중단되고 연산 순서가 섞이는데, 락 따로 해줘야함  
*threading 내장 Lock 클래스 사용해주면 됨  
**with문을 통해 사용하면 코드 가독성에 굿  
***with self.lock: self.count += offset  


# Better way 55 (Tuesday, 220621)  
# Queue를 사용해 스레드 사이의 작업을 조율하라  

*순차적 작업을 동시에 파이썬 스레드 이용할 시(특히 I/O 위주) 파으파리인 유용(파이썬 쓰레드를 사용한)  
*근데 Busy waiting, 종료 알리기, 메미로 사용 폭발의 문제가 발생할 수 있음  
**그래서 Queue 클래스를 가져와서 사용하면 블로킹 연산, 버퍼 크기 지정, jin을 통한 완료 대기를 지원해줘서 문제를 어느정도 해결해줌  
***굳이 dequeue로 직접 구현하는 것 보다 나음  


# Better way 56, 57, 58, 59 (Wednesday, 220622)
# 언제 동시성이 필요할지 인식하는 방법을 알아두라(56)    

*단일스레드 프로그램 -> 동시 실행되는 여러 흐름의 프로그램  
**바꾸는 것은 어려운 일  
***예를 들어, 블로킹 I/O -> I/O를 병렬로 수행  
****동시 실행되는 여러 실행 흐름 만들어 내는 과정 : 팬아웃(fan-out)   
****동시 작업 단위 모두 끝날 때까지 기다리는 과정 : 팬인(fan-in)  
*****57-60 챕터에서 이것들을 하는 파이썬 내장도구들을 알아보고 장단과 대안을 알아볼 것임  


# 요구에 따라 팬아웃을 진행하려면 새로운 스레드를 생성하지 말아라(57)  

*스레드(thread, 쓰레드) 이용하여 팬아웃, 팬인 구현 가능하지만  
**스레드 시작하고 실행하는데 비용이 많이 들고, 스레드가 많이 필요하면 메모리 많이 사용된다  
**또한 스레드 사이를 조율하기 위해 Lock과 같은 method 필요하다  
***스레드를 시작하거나 종료하기를 기다리는 코드에게 스레드 실행 중 발생한 예외를 반환하는 내장 기능은 없고 그래서 스레드 디버깅은 너무 어렵  


# 동시성과 Queue를 사용하기 위해 코드를 어떻게 리팩터링해야 하는지 이해하라(58)  

*작업자 스레드 수를 고정하고 Queue를 사용하면 스레드를 사용할 때 팬인과 팬아웃의 규모확장성을 개선할 수 있음  
**queue에 넣는 과정 - 팬아웃, queue에서 빼는 과정 - 팬인  
*하지만 Queue를 사용토록 리팩터링하려면 많은 작업이 필요  
*또한 Queue는 전체 I/O 병렬성의 정도를 제한하는 단점 있음(Queue크기만큼으로 제한)  


#동시성을 위해 스레드가 필요한 경우에는 ThreadpoolExecutor를 사용하라(59)  

*ThreadpoolExecutor를 사용하면 간단한 리팩터링으로 I/O 병렬성 활성화할 수 있고 동시성 팬아웃 시작시 스레드 시작 비용 줄일 수 있음(풀 있으니..)  
*풀 사용하므로 스레드 직접 사용 시의 잠재적 메모리 낭비 없애주지만 max workers 의 개수 미리 지정하므로 I/O 병렬성을 제한하는 것은 어쩔 수 없음  



# Better way 60 (Thursday, 220630)  
# I/O를 할 때는 코루틴을 사용해 동시성을 높여라  


*I/O 동시성 처리를 위해 코루틴 사용  
**동시에 실행되는 것처럼 보이는(동시성) 함수 많이 쓸 수 있음  
***코루틴은 async와 await 키워드를 사용해 구현되며 제너레이터를 싱행하기 위한 인프라를 사용  
****코루틴 시작으로 드는 비용은 함수 호출 비용뿐이고 1KB 미만의 메모리만 사용  
****코루틴은 매 await 식에서 일시 중단되고, 일시 중단된 대기 가능성(awaitable)이 해결된 다음에 async함수로부터 실행을 재개한다는 것이 스레드와 다름(제너레이터의 yield동작과 비슷)  
****여러 분리된 async 함수가 서로 장단을 맞춰 실행되면 마치 모든 async함수가 동시에 실행되는 것처럼 보임  
*****이를 통해 파이썬 스레드의 동시성 송작을 흉내낼 수 있음  
******하지만 이런 코루틴은 스레드와 달리 메모리 부가 비용이나 시작비용, 컨텍스트 전환비용이 들지 않고 복잡한 락과 동기화 코드가 필요 없음  
*******이를 가능케 하는 메커니즘은 이벤트 루프(event loop)임  
***참고로 def 정의 앞에 async를 붙여서 코루틴 사용함  
****안에서 async함수 호출할 때 await 붙여줌  
*****이 await는 마치 yield처럼 호출 즉시 실행되지 않고 제너레이터를 반환하는 것과 같은데 이러한 실행 연기 메커니즘이 팬아웃을 수행  
*****asyncio 내장 라이브러리가 제공하는 gather함수는 팬인을 수행  
******gather에 대해 적용한 await 식은 이벤트 루프가 코루틴을 동시에 실행하면서 코루틴이 완료될 때마다 코루틴 실행을 재개하라고 요청(예제에서)  
******asyncio.run 함수를 사용해 코루틴을 이벤트 루프상에서 실행하고 각 함수가 의존하는 I/O 수행 가능  
***코루틴 사용 시 기존코드에 await와 async만 붙여서 요구사항 충족할 수 있어 편리함  
***코루틴은 외부환경에 대한 명령(I/O와 같은)과 원하는 명령을 수행하는 방법을 구현하는 것(이벤트 루프)를 분리해준다는 점에서 아주 좋음  
**참고로 async 키워드로 정의한 함수를 코루틴이라 부르고 코루틴을 호출하는 호출자는 await키워드를 사용해 자신이 의존하는 코루틴의 결과를 받을 수 있음  
**코루틴은 수만 개의 함수가 동시에 실행되는 것처럼 보이게 만드는 효과적 방법 제공  
**I/O를 병렬화하면서 스레드로 I/O를 수행할 때 발생할 수 있는 문제를 극복하기 위해 팬인과 팬아웃에 코루틴 사용 가능  






