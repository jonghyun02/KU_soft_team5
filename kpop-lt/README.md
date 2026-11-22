# kpop-lt: K-pop 가사 번역 논문 구현

논문: Haven Kim, Jongmin Jung, Dasaem Jeong, Juhan Nam, *K-pop Lyric Translation: Dataset, Analysis, and Neural-Modelling*, LREC-COLING 2024 (arXiv 2309.11093).
논문에 적힌 값은 그대로 쓰고, 논문에 없는 값은 `kplt/config.py` 와 각 파일 주석에 `[해석]` 으로 표시했다.

## 구현한 것

- **데이터셋 복원**: 저자 저장소에는 가사 대신 줄 지문만 있다. 그래서 가사를 받아 지문으로 한영 줄·마디 짝을 되살린다.
- **3절 분석**: 줄·마디 의미 유사도(표 2, 그림 3), 음소 반복(표 3), 부록 표 8
- **4절 번역 모델**:
  - Marian 구조 모델을 처음부터 General, Non-singable, Singable 순서로 학습한다.
  - 설정은 줄 단위·마디 단위와 `<SYL>` 토큰 있음·없음을 곱한 4개다.
  - 설정마다 반지도(2단계 끝)와 미세조정(3단계 끝) 모델이 나와 8개다. 여기에 기준 모델 opus-mt-ko-en 을 더한다.
- **평가(표 5·6)**: SCD, 음절 오류율, Semsec, NSP 점수, Phodeg, Phovar

## 구조

```
kpop-lt/
  run.cmd            run.cmd data | analysis | chain CONFIG | nsp | infer | evaluate
  translate.cmd      구글 번역 전부 (막히면 기다렸다 이어감)
  train_all.cmd      4개 설정을 창 4개로 동시에 학습
  after_train.cmd    NSP 모델 학습, 9개 모델 번역, 표 5·6
  requirements.txt
  kplt/
    config.py        논문 값과 [해석] 값
    text.py          음절 세기, 마디 나누기
    phoneme.py       음소, pho, Phodeg, Phovar
    metrics.py       sts, Semline, Semsec, SCD, 오류율, NSP 점수
    gtranslate.py    무료 구글 번역 (sqlite 캐시)
    pipeline.py      전체 실행기
    analysis.py      3절 분석
    evaluate.py      4절 평가
    data/
      crawl.py       가사 수집 (Genius, 유튜브 설명란)
      match.py       지문으로 줄 되살리기
      dump.py        Genius 덤프에서 곡 뽑기
      build.py       학습 데이터 3단계
      translate_all.py  번역 묶음 실행
    model/
      common.py      토크나이저, 모델
      train.py, infer.py, nsp.py
  tests/test_core.py 논문 예시값 시험 20개
```

실행하면 `vendor/ raw/ data/ cache/ runs/ out/ logs/` 가 생긴다 (저장소에는 안 올림).

## 데이터 찾은 곳

| 쓰임 | 출처 | 결과 |
|---|---|---|
| 논문 데이터셋 1,000곡 | 저자 저장소 [havenpersona/lt_dataset](https://github.com/havenpersona/lt_dataset): 곡 목록과 줄 지문, 일부 비공식 영어 번역. 가사 본문은 Genius 공개 검색 주소와 유튜브 설명란에서 받음 | 963곡, 41,351줄 쌍 (논문 59,054줄의 70%) |
| General 50만 쌍 | AI Hub 한국어-영어 번역(병렬) 말뭉치 (논문이 쓴 Park et al. 2022) | 정리본 160만 쌍에서 무작위 |
| 영어 가사 1만 곡, NSP 7,103곡, 평가 K-pop 2,038곡 | 논문은 출처를 안 밝힘. HF `sebastiandizon/genius-song-lyrics` (Genius 전체 덤프, 약 9GB) | 영어 곡은 language=en, 평가 곡은 language=ko 중 pop. 논문 1,000곡과 겹침 0 |
| 영어 가사 한국어 번역, sts 용 한국어 가사 영어 번역 | 구글 번역 무료 주소 (논문도 구글 번역) | 많이 보내면 IP 가 몇 시간 막힘 |

데이터셋 묶음별 복원 (줄 쌍 / 지문 줄):

| 묶음 | 곡 | 줄 |
|---|---|---|
| K-pop 공식 번역 | 233 | 10,705 / 12,829 (83%) |
| K-pop 비공식 번역 | 617 | 28,823 / 39,527 (73%) |
| 디즈니 | 62 | 822 / 2,295 (36%), 한국어 가사가 Genius 에 거의 없음 |
| 뮤지컬 | 33 | 814 / 1,577 (52%) |
| 그 밖 (애니, 제이팝 등) | 18 | 187 / 619 |

- 한국어 가사를 하나도 못 찾은 37곡은 뺐다.
- 저자 `match.py` 의 버그 3개를 고쳐 다시 짰다: `split("last")`, `enumerate(zip(...))`, 비공식 번역 파일의 `[Verse]` 머리줄 때문에 영어 줄이 한 칸 밀리는 것.

## 필요한 것

저장소에 없는 것 (용량·저작권 때문에 안 올림):

1. Python 3.11. 이 폴더에 `.venv` 를 만든다. torch 는 CUDA 판(`--index-url https://download.pytorch.org/whl/cu128`)으로 깔고, 나머지는 `pip install -r requirements.txt` 로 깐다. 표 8(KoNLPy)은 Java 가 있어야 한다.
2. 저자 저장소: `git clone https://github.com/havenpersona/lt_dataset vendor/lt_dataset`
3. Genius 덤프: HF `sebastiandizon/genius-song-lyrics` 의 csv 를 `raw/genius_dump/song_lyrics.csv` 로 둔다.
4. AI Hub 한영 말뭉치 tsv (영어, 탭, 한국어 순서): `kplt/data/build.py` 의 `AIHUB_TSV` 경로를 자기 위치로 바꾼다.
5. 가사 받기: `.venv\Scripts\python -m kplt.data.crawl --shard 0 --n-shards 3` 를 조각 0·1·2 로 따로 띄운다 (몇 시간 걸림).
6. GPU: 12GB 카드에 맞춰 놓았다.

실행 순서:

1. `run.cmd data`: 가사 정렬, 덤프 추출, 학습 데이터 3단계. Non-singable 은 영어 가사 1만 곡을 구글 번역해야 해서 오래 걸린다.
2. `translate.cmd`: 한국어 가사 영어 번역 (분석·평가의 sts 용). 끝나면 `data/translate_done.json` 이 생긴다.
3. `train_all.cmd`: General 데이터만 있으면 시작할 수 있다. 다음 단계는 그 데이터가 생기면 이어간다.
4. `after_train.cmd`: 결과는 `out/tables.md`
5. `run.cmd analysis`: 번역이 끝난 뒤 돌린다. 결과는 `out/analysis.json`, 그림 3

시험: `.venv\Scripts\python -X utf8 -m pytest -q tests` (2번 저자 저장소가 있어야 다 통과)

남은 일 (2026-11-22 기준):

- 구글 번역이 막혀서 기다리는 중이다.
- 학습·평가는 아직 안 돌려서 결과 수치가 없다.
- 데이터셋이 70%만 복원돼서, 디즈니·뮤지컬 분석과 Singable 학습 데이터가 논문보다 작다.
