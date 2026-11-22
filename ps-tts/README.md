# ps-tts: PS-TTS 논문 대사 고르기 구현 (텍스트판)

논문: Changi Hong 외, *PS-TTS: Phonetic Synchronization in Text-to-Speech for Achieving Natural Automated Dubbing* (arXiv 2604.09111).
영어 대사를 넣으면 원문 길이와 입 모양(모음)에 맞는 한국어 더빙 대사를 고른다. 논문에서 대사를 고르는 단계만 글 입력, 글 출력으로 옮겼고 영상, 얼굴 검출, 음성 합성은 하지 않는다.

## 구현한 것

모델 1은 직접 학습한 영한 번역 모델(enko-nmt)이다. 모델 2는 논문의 GPT-4o 자리이고 지금은 Claude(`claude -p`)로 대신한다.

- **기존 판** (`run.cmd`, `pstts/`)
  1. 영어 음절 수에 1.28(임시값)을 곱해 한국어 목표 음절 수 N을 정한다. 입력에 목표 음절 수가 있으면 그 값을 쓴다.
  2. 모델 1이 초벌 번역을, 모델 2가 한국어 후보를 만든다.
  3. 음절 수(N±2쯤), 뜻(LaBSE 0.75 이상), 말투(반말, 해요체, 하십시오체)가 맞는 후보만 남긴다.
  4. PS-Comet(1.6 x 뒤집은 모음 DTW + 0.4 x COMET)이 가장 큰 후보를 고른다.
  - 논문에 없는 조건(말투, DTW 반경 26 등)을 더한 판이다. 숫자 조건도 있지만 기본은 꺼져 있다(`--number-check`로 켬).
- **논문 그대로 판** (`run_paper.cmd`, `pstts/paper/`)
  1. ISO: 모델 1 초벌이 길이(식 6)와 뜻(식 7) 조건을 넘는지 본다. 못 넘으면 모델 2가 바꿔 쓴 문장을 최대 60개까지 차례로 본다.
  2. PS: ISO 문장으로 모델 2가 후보 60개를 만들고 모음 DTW가 가장 작은 것을 고른다.
  3. PS-Comet: 같은 후보에서 1.6 x 뒤집은 DTW + 0.4 x COMET이 가장 큰 것을 최종 대사로 고른다.
  - 기존 판이 더한 조건은 모두 뺐다. TTS에서 나오는 값(음소 길이, 모음 거리표)만 지금 근사값이다(`--tts approx`).
  - Your-TTS를 학습하고 `scripts/paper_build_vowel_table.py`로 그 학습 폴더에 모음 거리표를 만든 뒤 `--tts yourtts --tts-dir <학습 폴더>`로 바꾼다. 학습 레시피는 `scripts/yourtts/`에 있고, 학습은 아직 안 했다.
- 시험 996개(기존 판 615, 논문 판 381)는 만든 PC에서 `.venv\Scripts\python.exe -X utf8 -m pytest -q`로 모두 통과했다. 5개는 저장소 밖 자료(`D:/transformer-base/enko-nmt/data/clean_v2/valid.tsv`, `models/your_tts/`의 공개 YourTTS)를 읽어서 그 자료가 없으면 실패한다.

## 구조

```
ps-tts/
  run.cmd, setup.cmd               기존 판 실행, 설치(.venv 만들기)
  run_paper.cmd, setup_paper.cmd   논문 그대로 판 실행, 추가 설치(setup.cmd 다음)
  requirements.txt, requirements-paper.txt
  pstts/                기존 판 (모듈 일부는 논문 판도 같이 씀)
    pipeline.py         전체 흐름
    ruler.py            목표 음절 수
    nmt.py              모델 1 (enko-nmt)
    llm.py              모델 2 (후보 만들기)
    select.py           후보 거르기, PS-Comet 고르기
    scorers.py          LaBSE, COMET
    register.py         말투 판정
    textnorm.py         숫자 읽기, 음절 세기
    phonetics.py, vowels.py, timing.py, dtw.py   발음, 모음 거리표, 모음 DTW
    cli.py, io.py, config.py, records.py         명령줄, 입출력, 설정, 자료형
    data/               모음 거리표
    paper/              논문 그대로 판
      pipeline.py       전체 흐름
      iso.py            ISO (3.1절)
      ps.py             PS (3.2절), PS-Comet (3.3절)
      prompts.py        모델 2 프롬프트 (바꿔 쓰기, PS 후보)
      g2p.py            영어, 한국어 음소화
      frontend.py       TTS 값 근사 (approx)
      yourtts.py, vowel_embed.py   학습한 Your-TTS에서 음소 길이, 모음 거리표 꺼내기
      cli.py, io.py, config.py
  scripts/              측정, 비교 스크립트
    paper_build_vowel_table.py   학습한 Your-TTS로 모음 거리표(식 8, 9) 만들기
    yourtts/            Your-TTS 학습 레시피 (받기, 매니페스트, 음소화, d-vector, 학습, 스모크)
  tests/                시험 (tests/paper/ 는 논문 판)
  examples/lines.csv    예시 대사 8줄
```

`.venv/`, `cache/`, `out/`, `models/` 와 코드 주석이 가리키는 긴 설명서(README_paper.md 등)는 저장소에 안 올렸다.

## 데이터 찾은 것과 필요한 것

논문 4.1절 기준이고, 학습할 모델은 Your-TTS 하나다. 링크는 2026-11-22에 열리는 것을 확인했다.

### 학습에 필요한 것

| 데이터 | 쓰는 곳 | 링크 | 비고 |
|---|---|---|---|
| LibriTTS train-clean-360 (영어, 904명, 약 190시간) | Your-TTS 영어 | [OpenSLR 60](https://www.openslr.org/60/) | 바로 받을 수 있음. 압축 27.5GB, CC BY 4.0. `scripts/yourtts/download_libritts.cmd`로 받음 |
| KMSSS (한국어, 550명) | Your-TTS 한국어 | [AI Hub 542 다화자 음성합성 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=542) (추정) | 신청·승인 필요, 내국인만. 논문 참고문헌에는 aihub.or.kr 주소만 있음. 542는 3,495명, 10,152시간이라 그중 550명을 쓴 것으로 보임 |
| 논문이 고른 KMSSS 550명 목록 | 한국어 학습 | **없음** | 비공개. 직접 골라야 함 |
| 8:1:1로 나눈 목록 | 학습·검증·시험 나누기 | **없음** | 비공개. 레시피가 시드 0으로 직접 나눔 |
| 모음표용 문장 목록 (한국어 6,000, 영어 5,000) | 식 8, 9 모음 거리표 | **없음** | 논문에는 TTS 학습 데이터에서 뽑았다고만 적혀 있음. `scripts/paper_build_vowel_table.py`로 직접 뽑음 |
| 논문이 학습한 Your-TTS 가중치 | 있으면 학습이 필요 없음 | **없음** | 비공개 |
| 논문의 모음 거리표 숫자 | 식 9 | **없음** | Fig. 5(c) 그림만 있음 |
| 화자 인코더 model_se.pth (데이터 아님, 논문에 없음) | 화자 벡터(d-vector) 뽑기 | [공개 YourTTS](https://github.com/coqui-ai/TTS/releases/download/v0.10.1_models/tts_models--multilingual--multi-dataset--your_tts.zip) | zip을 `models/your_tts/`에 풀면 `compute_dvectors.py` 기본 경로와 맞음 |

LaBSE, COMET, g2pK, phonemizer는 학습 없이 그대로 쓴다.

### 학습에는 필요 없는 것 (논문의 평가·분석용)

| 데이터 | 쓰는 곳 | 링크 | 비고 |
|---|---|---|---|
| 한국어 립리딩 | 입 모양 평가 20개 | [AI Hub 538 립리딩(입모양) 음성인식 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=538) | 신청 필요, 내국인만. 논문이 뽑은 20개 목록은 없음 |
| TCD-TIMIT (영어) | 입 모양 평가 20개 | [TCD-TIMIT](https://sigmedia.tv/datasets/tcd_timit/) | sigmedia.tcd.ie에 등록한 뒤 받음, 비상업 용도만. 20개 목록은 없음 |
| 성우 더빙 비교용 영화 클립 15개 | 성우 더빙과 비교 | **없음** | 논문이 영화 4편에서 직접 만들었고 공개하지 않음 |
| FLORES | DTW와 COMET 상관 분석 | [FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus) | Hugging Face 로그인 후 약관 동의. 논문이 인용한 첫 FLORES는 네팔어·싱할라어뿐이라 영·한은 FLORES+에서 씀 |

### 알아둘 것

- KMSSS가 AI Hub 542라는 것은 추정이다. "KMSSS"라는 이름으로 나온 데이터는 찾지 못했다.
- KMSSS 양: 542의 화자당 평균(약 2.9시간)으로 치면 550명은 약 1,600시간으로, LibriTTS-360의 8배쯤이다.
- 학습 시간 어림(RTX 4070 SUPER 12GB, batch 12, 잰 값 아님): LibriTTS만이면 12~22일, KMSSS까지 넣으면 약 4~7개월.
  논문이 화자마다 문장을 다 썼는지는 알 수 없다.
- 참고: [Latent Filling (arXiv 2310.03538)](https://arxiv.org/pdf/2310.03538)은 AI Hub 다화자 데이터의 이전 규모(3,086명, 약 7,414시간)를 적은 논문이다.

### 그 밖에 저장소에 없는 것

1. Python 3.11 가상환경: `setup.cmd`를 돌리고, 논문 판을 쓰려면 이어서 `setup_paper.cmd`를 돌린다. torch는 CUDA판(cu128)으로 깐다.
2. 모델 1: enko-nmt 폴더(코드, 가중치, `data/spm_v2/`의 SentencePiece 모델). `--nmt-dir`, `--nmt-ckpt`로 경로를 준다(기본값은 `pstts/config.py`). 모델 1이 없으면 기존 판은 `--no-draft`로 돌리고, 논문 판은 입력의 `draft_ko` 칸에 초벌을 적는다.
3. 모델 2: 로그인된 Claude Code CLI(`claude`).
4. LaBSE, COMET 가중치: 처음 실행할 때 Hugging Face에서 받는다.
