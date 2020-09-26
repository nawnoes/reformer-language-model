# reformer-playground
[reformer-pytorch](https://github.com/lucidrains/reformer-pytorch)를 이용한 다양한 Reformer Language Model 사전 학습 테스트

## index
    1. Vocab 생성
        - WordPiece
        - SentencePiece
    2. Data 전처리
    3. Encoder Pretrain
    4. Decoder Pretrain
    5. Fine-tuning
## 📘 Vocab 생성
#### WordPiece

## 💾 Data 
#### 사용 데이터
- 한국어 위키
- 한국어 나무 위키
- 모두의 말뭉치
    + 신문
    + 문어
    + 구어
    + 메신저
    + 웹
#### 데이터 전처리
- 주제 별로 한칸 개행.
- 라인 별로 개
```
꿀맛행                 #라인 별 개행
꿀맛은 다음을 의미한다.  #라인 별 개행
                     #주제별 한칸 개행
쿠로야나기 료
《쿠로야나기 료》는 《따끈따끈 베이커리》에 등장하는 등장인물으로, 투니버스판 이름은 최강기. 성우는 코야스 타케히토, 최재호

꿀맛 (1961년 영화)
《꿀맛》(A Taste Of Honey)은 영국에서 제작된 토니 리처드슨 감독의 1961년 드라마 영화이다.
동명의 희극을 각색한 영화이다.
도라 브라이언 등이 주연으로 출연하였고 토니 리차드슨 등이 제작에 참여하였다.
```

## 🖥 학습 서버
nipa 정보통신진흥원 GPU 자원
- GPU 할당량: 10TF
- GPU 카드: RTX6000

## 🏭 Language Model 

### 1. Masked Language Model(ex. BERT without NSP,SOP..) 
BERT에서 사용 MLM을 이용한 언어모델 학습. NSP와 SOP는 제외 하여 학습 진행.
![](https://paper-attachments.dropbox.com/s_972195A84441142620E4C92312EA63C9665C3A86AFFD1D713034FA568ADFC5F9_1555424126367_BERT-language-modeling-masked-lm.png)
#### 모델 설정
```
# Model Hyperparameter
max_len = 512     #전체 토큰 
batch_size = 128
dim = 512
depth = 6
heads = 8
causal = False
```
#### 학습 데이터
한국어 위키 512 토큰.
```
지미 카터제임스 얼 "지미" 카터 주니어(, 1924년 10월 1일 ~ )는 민주당 출신 미국 39번째 대통령 (1977년 ~ 1981년)이다.지미 카터는 조지아주 섬터 카운티 플레인스 마을에서 태어났다.조지아 공과대학교를 졸업하였다.그 후 해군에 들어가 전함·원자력·잠수함의 승무원으로 일하였다.1953년 미국 해군 대위로 예편하였고 이후 땅콩·면화 등을 가꿔 많은 돈을 벌었다.그의 별명이 "땅콩 농부" (Peanut Farmer)로 알려졌다.1962년 조지아 주 상원 의원 선거에서 낙선하나 그 선거가 부정선거 였음을 입증하게 되어 당선되고, 1966년 조지아 주 지사 선거에 낙선하지만 1970년 조지아 주 지사를 역임했다.대통령이 되기 전 조지아주 상원의원을 두번 연임했으며, 1971년부터 1975년까지 조지아 지사로 근무했다.조지아 주지사로 지내면서, 미국에 사는 흑인 등용법을 내세웠다.1976년 대통령 선거에 민주당 후보로 출마하여 도덕주의 정책으로 내세워, 포드를 누르고 당선되었다.카터 대통령은 에너지 개발을 촉구했으나 공화당의 반대로 무산되었다.카터는 이집트와 이스라엘을 조정하여, 캠프 데이비드에서 안와르 사다트 대통령과 메나헴 베긴 수상과 함께 중동 평화를 위한 캠프데이비드 협정을 체결했다.그러나 이것은 공화당과 미국의 유대인 단체의 반발을 일으켰다.1979년 백악관에서 양국 간의 평화조약으로 이끌어졌다.또한 소련과 제2차 전략 무기 제한 협상에 조인했다.카터는 1970년대 후반 당시 대한민국 등 인권 후진국의 국민들의 인권을 지키기 위해 노력했으며, 취임 이후 계속해서 도덕정치를 내세웠다.그러나 주 이란 미국 대사관 인질 사건에서 인질 구출 실패를 이유로 1980년 대통령 선거에서 공화당의 로널드 레이건 후보에게 져 결국 재선에 실패했다.또한 임기 말기에 터진 소련의 아프가니스탄 침공 사건으로 인해 1980년 하계 올림픽에 반공국가들의 보이콧을 내세웠다.지미 카터는 대한민국과의 관계에서도 중요한 영향을 미쳤던 대통령 중 하나다.인권 문제와 주한미군 철수 문제로 한때 한미 관계가 불편하기도 했다.
```
#### Fine-Tuning Test 결과
##### Korquad v1.0
|Task| exact_match | f1 score|
|------|---|---|
|Korquad v1.0|56.8|84.96|

### 2. Auto Regressive(ex. GPT여 계)
![](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRwoAurfiM2WIfF9tzx40wo9PcsHxpa0t_dCQ&usqp=CAU)
##### 진행중..🚛
### 3. Replaced Token Detention(ex. ELECTRA)
![](https://t2.daumcdn.net/thumb/R720x0.fpng/?fname=http://t1.daumcdn.net/brunch/service/user/Zvf/image/_twj8fBpj3opipMwC-w7Scv89yM.png)
##### 진행중..🚒

 
 ## Reformer-pytorch
 - `ReformerLM`의 **return_embeddings**은 reformer의 결과 값만 받고 싶은경우 설정