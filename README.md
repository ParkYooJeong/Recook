# RECOOK

> #### Welcome to π [Recook](http://j4a204.p.ssafy.io/) π
> size : Responsive(387 x 858)

<br>

## Project Summary π§­

μ¬λ£ κΈ°λ° λ μνΌ μΆμ² μλΉμ€

##### πΈ μ μ λ°°κ²½

- μ½λ‘λ19λ‘ μΈν μΈμ κ°μ, μ§μμ μλ¦¬νλ νμ μ¦κ°
- μ§μμ μλ¦¬λ₯Ό νλ μ¬λλ€μ΄ μ¦κ°νλ©΄μ λ μνΌμ νμμ± μ¦κ°

##### πΈ μλΉμ€ μ»¨μ

- μ νν μ¬λ£λ‘ λ§λ€ μ μλ μλ¦¬ λ μνΌλ₯Ό μΆμ²ν΄μ£Όλ μλΉμ€
- μ¬λ£μ λ§λ λ μνΌλ₯Ό μΌμΌμ΄ μ°Ύμλ΄μΌ νλ λΆνΈν¨ ν΄μ

##### πΈ νκ²

- μ§μμ λ¨μ μ¬λ£λ‘ λ§μλ μλ¦¬λ₯Ό λ§λ€μ΄ λ¨Ήκ³  μΆμ μ¬λ

##### πΈ κΈ°κ°

- Feb 22th 2021 ~ Apr 9th 2021

##### πΈ κ²°κ³Όλ¬Ό

- [Architecture & Usecase_Diagram](./document/Architecture.md)
- [Sequance_Diagram](./document/SequanceDiagram.md)
- [PPT](./document/νΉνPJT_μ΅μ’λ°ν_A204_μ΅μ’.pdf)
- [UCC](https://www.youtube.com/watch?v=tknwLMpBXCE)





<br>

## Project Info :pushpin:

### Usage π

#### π» Front-end

- Vue.js

  - Project Setup

    ```bash
    $ npm install 
    ```

  - Compiles and hot-reloads for development

    ```bash
    $ npm run serve
    ```

  - Compiles and minifies for production

    ```bash
    $ npm run build
    ```

  - Run your tests

    ```bash
    $ npm run test
    ```

  - Lints and fixes files

    ```bash
    $ npm run lint
    ```

- Customize configuration

  - [Configuration Reference](https://cli.vuejs.org/config/)

#### π» Back-end

**Install**

- Java (Open JDK 14)

- Maven

- STS 

- Mariadb

  - create tables

    run dump.sql => [dump.sql](./document/dump.sql)

- Django

  - Project Setup

    ```bash
    $ pip3 install -r requirements.txt
    # μ€λ₯λλ λͺ¨λμ μλμΌλ‘ κΉμμ£ΌμΈμ
    ```

  - DB Connection

    ```bash
    $ python manage.py inspectdb
    ```
  
  - migration
  
    ``` bash
    $ python manage.py makemigrations
    ```
  
  - Run
  
    ```bash
    $ python manage.py runserver
    ```

<br>



### Tech Stack π§©

- Backend : Spring Boot, Django, MariaDB
- Frontend : Vue.js

![κΈ°μ μ€ν](./document/img/κΈ°μ μ€ν.png)



### Database Modeling :link:

![ERD](./document/img/erd.png)



<br>



### Features :sparkles:

##### 	π λ©μΈ κΈ°λ₯

```
- μ μ μ μ·¨ν₯(μμμ’λ₯, μλ λ₯΄κΈ°)μ λΆμνκΈ° μν μ€λ¬Έμ‘°μ¬
- μ¬λ£λ₯Ό μ ννλ©΄ λ§λ€ μ μλ λ μνΌ μΆμ²
- μ¬μ©μμ μ·¨ν₯(μμμ’λ₯, μλ λ₯΄κΈ°)μ λΆμν κ°μΈ λ§μΆ€ν λ μνΌ μΆμ²
- κ° λ μνΌ λ§λ€ λΉμ·ν μ¬λ£λ₯Ό κ°μ§κ³  λ§λ€ μ μλ μ°κ΄ λ μνΌ μ κ³΅
```

##### 	π λΆκ° κΈ°λ₯

```
- ν΄λΉ λ μνΌμ ν¬ν¨λ μ μ μ μλ λ₯΄κΈ° μ λ³΄ μλ¦Ό
- λ μνΌλ₯Ό μ°ν΄ λκ³  λͺ¨μλ³Ό μ μλ κΈ°λ₯
- λ μνΌ μ λͺ© κ²μ κΈ°λ₯
- μ΅κ·Όμ λ³Έ λ μνΌ νμΈ κ°λ₯
- ν΄λΉ λ μνΌλ₯Ό λ³΄κ³  μ μ κ° λ§λ  μμ μ¬μ§ κ²μ κΈ°λ₯
- νμ μ λ€μ΄ κ²μν μμ μ¬μ§ λͺ¨μλ³΄κΈ° κΈ°λ₯
- μμ κ΄λ ¨ μμ μ κ³΅
```

##### 	π μ¬μ©ν λΉλ°μ΄ν° μΆμ² μκ³ λ¦¬μ¦
- νμ νν°λ§ Collaborative Filtering
- μ»¨νμΈ  κΈ°λ° νν°λ§ Content based Filtering

<br>



### Pages in Detail :mag:

> κ° νμ΄μ§ λ³ μκ°

- ##### Survey

  ![μ·¨ν₯μ‘°μ¬](./document/gif/μ·¨ν₯μ‘°μ¬.gif)
- ##### Main

  ![λ©μΈνλ©΄](./document/gif/λ©μΈνλ©΄.gif)

  

- ##### Recipe Detail

  ![λ μνΌ_μμΈ](./document/gif/λ μνΌμμΈ.gif)

  

- ##### Review

  ![λ¦¬λ·°_λͺ¨μλ³΄κΈ°](./document/gif/λ¦¬λ·°λͺ¨μλ³΄κΈ°.gif)

  

- ##### MyPage(My Review & Like)

  ![λ¦¬λ·°_μ°](./document/gif/λ¦¬λ·°μ°.gif)

  

- ##### Cook Video

  ![μ νλΈ](./document/gif/μ νλΈ.gif)

### Recipe Source π

- [ν΄λ¨Ήλ¨λ](https://haemukja.com/)




