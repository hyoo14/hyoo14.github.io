---
layout: post
title:  "Tibero install and link with Python in Ubuntu"
date:   2022-07-31 18:31:19 +0900
categories: study
---




{% highlight ruby %}
짧은 요약 : Tibero install and link with Python in Ubuntu  



{% endhighlight %}



# 설치  

-유저추가(원할 경우에만)  
adduser tibero  
tibero  
enter/enter/enter…  

-환경변수 설정  
cd /home/tibero  
vim .bashrc  

맨 마지막 부분에  

export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64  

export TB_HOME=/home/tibero/Tibero/tibero6  
export TB_SID=tibero  
export LD_LIBRARY_PATH=$TB_HOME/lib:$TB_HOME/client/lib  
export PATH=$PATH:$JAVA_HOME/bin:$TB_HOME/bin:$TB_HOME/client/bin  



-technet에서 unix 64bit용 tar 파일 다운받고 /home/tibero/Tibero 폴더에 옮겨서 압축해제  
tar xvf tibero6-bin-FS07_CS_2005-linux64-199301-opt.tar.gz   

-테크넷에서 tibero6 license 받고 이 license 파일을  /home/tibero/Tibero/tibero6/license에 위치시킴(참고로, 라이센스 신청시 ubuntu의 hostname 잘 써줘야함)  

-/home/tibero/Tibero/tibero6/config에서 gen_tip.sh 파일 실행  
./gen_tip.sh  

-Tibero 서버를 'NOMOUNT 모드'로 기동 및  'sys' 사용자로 접속 및 동작 확인  
tbboot nomount  
tbsql sys/tibero  

-아래 명령어 입력으로, 최초 데이터베이스 생성  
SQL> create database "tibero"   
  user sys identified by tibero   
  maxinstances 8   
  maxdatafiles 100   
  character set MSWIN949   
  national character set UTF16   
  logfile   
    group 1 'log001.log' size 100M,   
    group 2 'log002.log' size 100M,     
    group 3 'log003.log' size 100M   
  maxloggroups 255   
  maxlogmembers 8   
  noarchivelog   
    datafile 'system001.dtf' size 100M autoextend on next 100M maxsize unlimited   
    default temporary tablespace TEMP   
      tempfile 'temp001.dtf' size 100M autoextend on next 100M maxsize unlimited   
      extent management local autoallocate   
    undo tablespace UNDO   
      datafile 'undo001.dtf' size 100M autoextend on next 100M maxsize unlimited   
      extent management local autoallocate;  

-quit으로 종료 후 다시 tbboot로 tibero 가동  (여기까지는 포스트가 잘 올라감..)

-/home/tibero/Tibero/tibero6/scripts에서 system.sh 실행시킴  
./system.sh  

-다음 명령어로 티베로 프로세스가 잘 동작하는지 확인  
ps -ef | grep tbsvr  


# ubuntu에서 연동  

-ODBC 설치  
sudo apt install build-essential  
sudo apt install libssl-dev python3-dev  
sudo apt install unixodbc unixodbc-dev  
-/etc/odbcinst.ini 파일에 ODBC 드라이버 이름을 설정 (티베로 클라이언트 설치되어있어야함)  
[Tibero6]  
Description = Tibero6 ODBC driver  
Driver = /home/tibero/Tibero/tibero6/client/lib/libtbodbc.so  
Setup = /home/tibero/Tibero/tibero6/client/lib/libtbodbc.so  
Setup = 1  
FileUsage = 1  
-/etc/odbc.ini 파일에 MYCODE 라는 이름으로 ODBC를 설정, 이후 ODBC 접속은 MYCODE라는 DSN을 사용  
[ODBC]  
Trace = 1  
TraceFile = /tmp/odbc.trace  
  
[MYCODE]  
Trace = no  
Driver = Tibero6  
Description = Tibero6 ODBC Datasource  
# ConnectionType = Direct  
SERVER = 192.168.7.82 #실 서버 주소  
PORT = 8629  
# AuthenticationType=No Authentication  
SID = tibero  
User = sys  
Password = tibero  
Database = tibero  


-isql MYCODE 로 연결 되는지 확인(연결되면 CONNECT 뜸)  

-그 외 간단한 SELECT 테스트  
SQL> select * from TIBERO.TEST;  
-스크립트 작성하여 동작 잘 하는 지 확인    
vi test_tibero_odbc.py  

import pyodbc #알아서 잘 깔아줘야함 PYODBC  

try:  
    user = 'sys'  
    passwd = 'tibero'  

    sql = 'select * from test;'  
    conn = pyodbc.connect('DSN=MYCODE;UID='+user+';PWD='+passwd)  
    
    conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8') #인코딩 이슈 해결  
    conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8') #  
    conn.setdecoding(pyodbc.SQL_WMETADATA, encoding='utf-32le') #  
    conn.setencoding(encoding='utf-8') #인코딩 이슈 해결  

    curs = conn.cursor()  
    row = curs.execute(sql)  
    for i in row:  
        print(i[0])  

    conn.close()  

except Exception as ex:  
    print(ex)  


# 참고문헌  
-https://technet.tmaxsoft.com/upload/download/online/tibero/pver-20180723-000001/index2.html  
-https://blog.boxcorea.com/wp/archives/2881  




# 윈도우에서 연동 






