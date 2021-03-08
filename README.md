## [ SSAFY 특화 PJT 서울 2반 A204팀 ] Sub 2

### 서비스 기획

### [ 레시피 추천 서비스 ]

#### 제안 배경
    - 코로나19로 인한 외식 감소, 집에서 요리하는 횟수 증가
    - 남은 재료 및 배달음식을 버리게 되어 음식물 쓰레기 증가
#### 타겟
    - 집에서 남은 재료나 배달 음식으로 그럴듯한 요리를 만들어 먹고 싶은 사람

#### 서비스 컨셉

```
- 남은 재료나 배달 음식(치킨)으로 만들 수 있는 요리의 레시피를 추천해주는 서비스
- 재료에 맞는 레시피를 일일이 찾아봐야 하는 불편함 해소
```

#### 기능

```
메인 기능

- 남은 재료, 음식으로 만들 수 있는 레시피 추천
- 자신만의 레시피를 업로드할 수 있고 해당 레시피가 추천을 많이 받는다면 다른 사람이 검색했을때 추천 레시피에 등장!

- 부가 기능
  - 알러지 정보 제공
  - 궁합이 잘 맞는 음식끼리 새로운 레시피 추천

```

#### 필요 기술 스택

```
- ? 어떤 기술을 사용하면 좋을까요 ?
- 추천 알고리즘 : 개인화 (선호하는 음식, 유니크한 개인의 정보 등) 
⇒ 설문 
⇒ 유사도 검사 : 레시피별 성격검사 (몇인분인지,,, 어떤 맛인지,,, 맵기 짠기 신맛 5대 맛) ex) 같은 돈까스지만 다 다른 돈까스 
⇒ 논문 참고 : 음식의 성향? 
⇒ 알러지 / 맵기 / 고향 / 트라우마 관련된 특정 음식 등 
⇒ 한식 / 일식 / 중식 / 양식 ⇒ 유저의 데이터가 쌓여서 좋아하는 것 추천해주기!!!!! 좋아요 목록 = 빅데이터
- 데이터 가공
- 유저의 활동 데이터를 재사용!!! : 유저의 움직임을 로깅하고 다 활용할 수 있도록! 
⇒ 활동을 할수록 정확도가 올라간다!
- 스키마 설계 컨설턴트님과 함께!!!
- 서버 과정 : 클라이언트 - 자바 - 파이썬(사설망) - DB ⇒ 중간에 자바 서버 거치기
```



