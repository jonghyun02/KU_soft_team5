"""논문(arXiv 2309.11093v4, LREC-COLING 2024) 값을 한곳에 모은 설정.

각 값 옆에 논문 위치를 적었고, 논문에 없는 값은 [해석] 으로 표시했다.
README.md 의 해석 표와 같은 내용이다.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "lt_dataset"          # 저자 저장소 (meta.csv, symbol.json, processed_lyrics)
RAW = ROOT / "raw"                                # 내려받은 원자료
DATA = ROOT / "data"                              # 만든 데이터
RUNS = ROOT / "runs"                              # 학습 결과
OUT = ROOT / "out"                                # 표·그림
CACHE = ROOT / "cache"                            # 번역·임베딩 캐시

SEED = 42                                         # [해석] 논문에 없음

# ---- 3절 분석 ----
STS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 3.1절
# 3절 비교 대상: 공식 번역만 (K-pop 234, 애니 62, 뮤지컬 34). meta.csv 기준 장르 이름
ANALYSIS_GENRES = {"k-pop": "K-pop", "disney": "Animation", "musical": "Theatre"}

# ---- 4.1절, 표 4, 부록 C ----
TOKENIZER = "Helsinki-NLP/opus-mt-ko-en"          # 4.1절 (Tiedemann and Thottingal, 2020)
BASELINE = "Helsinki-NLP/opus-mt-ko-en"           # 4절 기준 모델 (같은 Marian 구조의 사전학습 번역기)
N_LAYERS = 6                                      # 표 4
N_HEADS = 8                                       # 표 4
BATCH_SIZE = 8                                    # 표 4
MAX_POS = 1024                                    # 표 4
WARMUP_STEPS = 500                                # 표 4
WEIGHT_DECAY = 0.1                                # 표 4
VOCAB_SIZE = 65051                                # 표 4 (= 토크나이저 65001 + 추가 토큰 50)
# 표 4 "Model size 2048" [해석]: 기준 모델 opus-mt-ko-en 과 "같은 구조" (4절) 라고 했으므로
# d_model 512, FFN 2048 (opus-mt-ko-en 값) 으로 읽었다. d_model=2048 로 읽으면 기준 모델과 구조가 달라진다.
D_MODEL = 512
FFN_DIM = 2048
LEARNING_RATE = 5e-5                              # [해석] 논문에 없음. 표 4 이름이 HF Trainer 인자와 같아 Trainer 기본값
LR_SCHEDULER = "constant_with_warmup"             # [해석] 논문은 warmup 500 만. 끝을 정할 epoch 수가 없어 warmup 뒤 고정
EARLY_STOP_PATIENCE = 3                           # 부록 C: 검증 손실이 3 epoch 연속 줄지 않으면 멈춤
MAX_EPOCHS = 100                                  # [해석] 논문에 상한 없음. 조기 종료가 먼저 걸리도록 크게 (LR 이 고정이라 값에 결과가 안 달림)
VALID_RATIO = 0.1                                 # 부록 C: 9:1
MAX_LEN = 1024                                    # [해석] 토큰 길이 상한 = 표 4 최대 위치. 넘는 단위는 자르지 않고 뺀다

# ---- 4.2절 데이터 ----
N_GENERAL = 500_000                               # 4.2절 General (Park et al., 2022 = AI Hub 한영 말뭉치)
N_NONSINGABLE = 10_000                            # 4.2절 영어 가사 1만 곡
N_NSP_SONGS = 7_103                               # 4.3절 NSP 미세조정용 영어 가사
N_EVAL_SONGS = 2_038                              # 4.4절 평가용 K-pop 가사
# 추가 토큰 50개 [해석]: <SYL1>..<SYL49> + <SEP>. 49 음절이 넘는 줄은 <SYL49>
SYL_MAX = 49
SYL_TOKENS = [f"<SYL{i}>" for i in range(1, SYL_MAX + 1)]
SEP_TOKEN = "<SEP>"
EXTRA_TOKENS = SYL_TOKENS + [SEP_TOKEN]
# General 문장을 몇 조각으로 나눌지 [해석]: 1 이상, 양쪽 단어 수와 SECTION_MAX_SEGMENTS 중 작은 값까지 균등
SECTION_MAX_SEGMENTS = 8

# ---- 4.4절 추론 ----
NUM_BEAMS = 4                                     # 4.4절
GEN_MAX_NEW_TOKENS = 1024                         # [해석] = MAX_POS

# ---- 4.3절 NSP ----
NSP_BASE = "google-bert/bert-base-uncased"        # 4.3절 각주 2
NSP_EPOCHS = 3                                    # [해석] 논문에 없음
NSP_LR = 2e-5                                     # [해석] 논문에 없음
NSP_BATCH = 32                                    # [해석] 논문에 없음

CONFIGS = {
    # 이름: (form, syl)
    "line": ("line", False),
    "line_syl": ("line", True),
    "sect": ("section", False),
    "sect_syl": ("section", True),
}
STAGES = ["general", "nonsingable", "singable"]   # 그림 4 순서
