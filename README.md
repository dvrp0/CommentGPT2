# CommentGPT2
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YGfubih3VCV_V1GbJPraL4j_As2TMREj?usp=sharing)

[KoGPT2](https://github.com/SKT-AI/KoGPT2)를 [특정 뉴스 기사](https://news.v.daum.net/v/20190314112104189)의 댓글 데이터로 파인튜닝시킨 모델입니다. 간단한 실험 용도로 제작했기에 한 개의 기사의 데이터만 사용했습니다.

## 학습
```
!python train.py --train --batch-size 96 --dataset_path "PATH" --tpu_cores 8 --max-len 32 --max_epochs 100
```
학습 코드는 [haven-jeon](https://github.com/haven-jeon/KoGPT2-chatbot) 님의 코드를 조금 변형하여 사용했습니다.   
사용된 데이터는 댓글 10,007개(약 1.28MB)이며, Colab TPU에서 1시간 가량 학습했습니다.

## 생성
```
!python train.py --generate --inputs "INPUT" --model_params "PATH"
```

### 예시
```
1: 진짜 친일파들이 했던 수법 빨개이는 결국 국민을 분열시켰다 처단했어야 하는데 그게 안됐고 해방 후엔 북한으로 보냈으면
2: 진짜 똘라이짓 하며 정치하는지 몰랐겠지~ᄏ
3: 진짜 미친거 아냐? 미안하지만  일본으로 이민가쇼 한국에선 못 살겠다..
```
```
1: 도대체 어느나라 국민이야? 넌 어디다 갓나온 돌잔치 아줌마와 같은 여자냐?
2: 도대체 역사를 똑바로대로 읽어야 하냐고? 반민특위가 제대로 활동도 못하고 뭔 소린 나불거리는가. 어떻게 이런 망언을
3: 도대체 정부는 뭐하고 있습니까? 그게 불만이 아니면! 국민들이 왜 이토록 아베에 대한 생각을 가지는지 이해가 안가네요.
```
```
1: 하 친일 올가미가 아니라 니네년이 할말아니다.
2: 하 이쓰레기들이 발악을 하는 걸로 보인다 5.친일파재산환수법 반대-자한당 태극기
3: 하 친일파의 후예들이 이런 소리를 하니까 가슴이 철렁 내려앉는구나.
```

## TODO
* [KcBERT Pre-Training Corpus](https://www.kaggle.com/junbumlee/kcbert-pretraining-corpus-korean-news-comments)로 학습
