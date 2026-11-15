"""대본과 받아쓰기 맞추기 -> films/{film_id}/work/align.json  (명령 align)

입력은 docs/{doc_id}/script.json 과 films/{film_id}/work/asr.json, 출력은 연결·판정 기록(명세 4.6)이다.
대본 대사 줄마다 어느 받아쓰기 pass 의 어느 낱말 범위에서 들렸는지(same·changed), 영상에 없는지(absent)를
정하고, 어느 줄에도 속하지 않은 받아쓰기 낱말은 added 줄로 묶는다. 들린 글자는 늘 받아쓰기 낱말 그대로다.

순서(한 pass 를 대본과 맞추는 ``match_pass``)
 1. 비교용 글: 대본 대사와 받아쓰기 낱말을 공백·문장부호 없는 글자열로(textnorm.loc, 숫자는 읽는 꼴).
 2. 닻: 양쪽에서 한 번씩만 나오는 k-gram 을 씨앗으로 최대 일치 블록을 만들고, 대본 순서와 영상 순서가
    함께 늘어나는 사슬 가운데 일치 글자 합이 가장 큰 것을 뼈대로 고른다(가중 LIS).
    영화 전체를 한 번의 최소 편집 거리로 맞추지 않는다(놓친 줄이 엉뚱한 글자와 짝지어진다).
    하나뿐인 조각이 거의 없으면(대본에 같은 내용이 두 번 든 문서) 더 긴 조각을 몇 번 나와도 씨앗으로 쓴다.
 3. 자리 옮긴 씬: 뼈대에 못 든 블록에서 다시 사슬을 뽑아, 가까이 모인 블록의 일치 글자 합이
    MOVED_MIN 이상이면 옮겨진 덩어리로 받아들인다(뼈대와 겹친 가장자리 글자는 잘라 내고, 기준에 못 미친
    사슬은 버리고 남은 블록에서 다시 찾는다).
 4. 닻 사이: 틈마다 더 짧은 k-gram 으로 닻을 되풀이해 찾고(틈이 좁을수록 짧은 k), 닻이 충분한 줄은
    씨앗 줄로 삼는다(짧은 조각으로만 걸렸고 그 조각이 이웃 대본 줄에도 있는 줄은 빼고). 씨앗 줄 사이에
    남은 줄과 남은 낱말은 줄 단위 DP 로 맞춘다: 줄마다 이어진 낱말 범위 후보의 '정확히 맞은 글자 수'(LCS)를
    재서 기준을 넘는 후보만 쓰고, 순서를 지키며 합이 가장 큰 짝을 고른다. 씨앗 줄은 양옆으로 맞는 낱말만큼 넓힌다.
 5. 낱말 범위: 줄마다 이어진 [처음, 끝]. 이웃 줄과 겹치지 않는다.
 6. 다시 짝짓기(``_repair``): 맞춘 두 줄 사이 창에서 못 맞춘 줄과 남은 낱말을 순서대로 다시 짝짓는다(같은 뜻의
    짧은 대답, 자모가 닮은 줄, 줄의 일부만 말한 줄, 글자까지 같은데 이웃이 안 맞아 버려진 짧은 줄).
    창이 좁을수록, 그 말이 영화에 드물수록 기준이 낮다.
 7. 순서 밖에서 되찾기(``_reclaim_moved``): 이웃 줄과 순서만 바꿔 말한 줄.
그 뒤 ``leftover`` 가 남은 낱말을 옆 줄에 붙이거나(한도와 근거가 있을 때만) added 로 자르고, ``decide`` 가 줄마다
pass 를 고르고(두 pass 의 범위를 서로 같은 소리만큼 넓혀 견준다), rescue 가 아무도 못 맞춘 줄을 찾은 뒤, 여러 pass 의 남은
낱말로 한 번 더 다시 짝짓고(``_repair_across``), ``textnorm.verdict`` 가 same/changed 와 차이 조각을 정한다.
added 로 남은 낱말 가운데 자막 제작 문구 꼴과 그 영화의 버릇 글귀(``asr.film_habits``)는 added 로 만들지 않고 method 의
noise_dropped 에 남긴다(``drop_noise``). 마지막으로 LLM 짝 판정(``sinkprep.align_llm``: 못 맞춘 줄과 그 시간대의 added 낱말,
맞춘 줄의 떨어진 머리·꼬리를 claude 가 판정하고 구조 제약 안에서만 받아들인다, decided_by llm)을 캐시에 있는 만큼(``--llm``
이면 캐시에 없는 것도 불러서) 적용한다.

pass 여럿(``build_ops``). pass 의 구실은 asr.json pass 의 context 로 가른다(``pass_roles``).
 - context 가 null 인 pass 는 대본 없이 받아쓴 후보다. id 가 plain 인 것(없으면 맨 앞)이 기준 pass, 나머지가 대안 pass
   (다른 모델의 받아쓰기). 둘을 따로 맞춘 뒤 줄마다 고른다: 기준이 대본과 같으면 기준, 대안이 같으면 대안, 둘 다 다르면
   대본과 뚜렷이 더 닮은 쪽(비슷하면 기준). 가려 적은 글('새X', 'ㅅㄲ')이 든 후보는 다른 후보가 있으면 안 고른다. 둘 다 대본을
   보지 않았으니 소리 점수 없이 규칙으로 고른다(decided_by rule).
 - 대본을 문맥으로 주고 받아쓴 pass(ctx)는 대본 쪽으로 끌릴 수 있다. 기준으로 고른 줄과 아무도 못 맞춘 줄에서만, 낱말
   확률로 견줘서 쓴다(decided_by acoustic).
 - rescue pass(말 구간 검출이 놓친 곳을 다시 받아쓴 낱말)는 아무도 못 맞춘 줄을 찾는 데 쓰고, 줄에 안 쓰인 낱말은 기준
   pass 가 받아쓰지 않은 곳에 있을 때 added 로 만든다.
 - 한 번 난 소리는 한 줄에만 담는다. 같은 pass 안에서는 낱말 범위가 겹치지 않고, 다른 pass 의 줄끼리는 같은 소리인 낱말
   (``_same_word``: 시간이 짧은 쪽 길이의 절반 넘게 겹침)을 함께 갖지 않는다(``_Claims``). 다른 pass 에서 고른 줄의
   소리인 기준 낱말은 added 로 세지 않는다. added 줄은 기준 pass 와 rescue pass 의 낱말로만 만든다.

기준값은 tools/eval_align_synth.py(가짜 영화 채점)와 실제 영화 한 편(추격자)의 줄 목록을 눈으로 읽어 골랐다.
가짜 받아쓰기는 실제보다 쉽고 실제 영화에는 정답 표본이 없으므로, 정답 표본이 생기면 다시 확인해야 한다.
"""
from __future__ import annotations

import bisect
import heapq
import json
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field

from rapidfuzz.distance import LCSseq

from . import paths, textnorm, timeline, util

STAGE = "align"
VERSION = "0.4.0"           # 0.4.0: LLM 짝 판정(align_llm, decided_by llm), 자막 제작 문구·버릇 글귀는 added 로 만들지 않음
#                             0.3.0: 여러 pass 다시 짝짓기, 같은 소리 범위로 넓혀 pass 고르기, rescue 겹쳐 적기 거름, 짧은 대답 짝
#                             0.2.0: pass 구실을 context 로(기준·대안), rescue 낱말의 added, conf 계산식, 붙이기 한도·다시 짝짓기

# ------------------------------------------------------------------ 상수
BASE_PASS = "plain"        # 대본 없이 받아쓴 pass 가운데 이 id 가 기준 pass 다(없으면 그런 pass 가운데 맨 앞)
# pass 의 구실은 asr.json pass 의 context 로 가른다(``pass_roles``). context 가 null 이면 대본 없이 받아쓴 후보다.
CTX_CONTEXTS = ("script_hotwords", "script_window")      # 대본을 문맥으로 주고 받아쓴 pass(대본 쪽으로 끌릴 수 있다)
RESCUE_CONTEXTS = ("full_window_stable",)                # 말 구간 검출이 놓친 곳을 다시 받아써 안정된 조각만 남긴 pass

# 닻
K_LEVELS = (5, 4, 3)       # 닻 씨앗 k-gram 길이(긴 것부터)
AREA_K4 = 1_000_000        # 틈의 (대본 글자 수 × 받아쓰기 글자 수)가 이 이하일 때만 4-gram 닻을 쓴다
AREA_K3 = 40_000           # 이 이하일 때만 3-gram 닻을 쓴다(넓은 틈에서는 우연히 맞는 짧은 조각이 많다)
SPARSE_COVER = 0.05        # 뼈대 닻이 덮은 글자가 (짧은 쪽 글자 수 × 이 값)보다 적으면 되풀이 조각도 씨앗으로 써 본다
REPEAT_K = 8               # 그때의 k-gram 길이
REPEAT_CAP = 3             # 그때 양쪽에서 이 횟수까지 나오는 조각을 씨앗으로 쓴다
MOVED_MIN = 20             # 옮겨진 덩어리로 받아들이는 일치 글자 합
MOVED_GAP_S = 200          # 한 덩어리 안에서 이웃 블록 사이 대본 글자 틈
MOVED_GAP_A = 400          # 같은 것, 받아쓰기 글자 틈
MOVED_ROUNDS = 40          # 옮겨진 덩어리 찾기 되풀이 횟수(바퀴마다 남은 블록의 가장 무거운 사슬을 본다)
ANCHOR_GAP = 4.0           # 한 줄의 닻이 시간상 이만큼(+ 사이 글자당 ANCHOR_GAP_PER_CHAR) 넘게 벌어지면 묶음을 끊는다
ANCHOR_GAP_PER_CHAR = 0.5
ANCHOR_SKIP = 6            # 한 줄의 이웃한 두 닻 사이에 받아쓰기 글자가 이만큼(+ 사이 대본 글자 × ANCHOR_SKIP_RATIO) 넘게 끼면 끊는다
ANCHOR_SKIP_RATIO = 2.0
# 줄 받아들이기
SHORT_MAX = 3              # 이 글자 수 이하인 줄은 낱말 범위가 대본과 글자 그대로 같아야 한다
MID_MAX = 7                # 4 ~ MID_MAX 글자 줄은 R_MID, 더 긴 줄은 R_LONG
R_MID = 0.6                # 맞은 글자 ÷ 줄 글자
R_LONG = 0.4
SIM_MIN = 0.4              # 2 × 맞은 글자 ÷ (줄 글자 + 범위 글자)
BLOCK_OK = 8               # 이만큼 이어진 닻 블록이 있으면 비율이 R_BLOCK 만 넘어도 받아들인다(줄의 일부만 말한 경우)
R_BLOCK = 0.2
SEED_MIN_HIT = 4           # 닻 글자가 이만큼이거나 비율이 SEED_MIN_RATIO 이상이면 씨앗 줄
SEED_MIN_RATIO = 0.6
SEED_MIN_BLOCK = 5         # 닻 블록이 이보다 짧고 그 글이 가까운 다른 대본 줄에도 있으면 씨앗 줄로 굳히지 않는다(줄 단위 DP 에서 다툰다)
SEED_NEAR = 2              # 그때 살피는 앞뒤 대본 줄 수
# 닻 사이 줄 단위 DP
BETA = 0.5                 # 범위 안 안 맞은 글자 하나의 벌점(맞은 글자 하나 = 1점)
SPAN_MAX_RATIO = 1.8       # 후보 범위 글자 수 상한 = 줄 글자 × 이 값 + SPAN_MAX_EXTRA
SPAN_MAX_EXTRA = 3
GAP_CAP = 600              # 씨앗 줄 한쪽에서 살피는 낱말 수 상한
# 다시 짝짓기(맞춘 줄 사이 창에서 못 맞춘 줄과 남은 낱말을 순서대로)
SLOT_MAX_S = 15.0          # 줄 하나·남은 낱말 전체 짝(SLOT_JAMO_MIN)은 앞뒤 맞춘 줄 사이가 이 초 이하일 때만(0 이면 다시 짝짓기를 끔)
SLOT_JAMO_MIN = 0.6        # 창에 못 맞춘 줄이 하나뿐이고 남은 낱말 전체가 그 줄일 때의 자모 닮음(2 × 맞은 자모 ÷ 자모 수 합) 하한
SLOT_JAMO_MANY = 0.65      # 좁은 창에서 줄(4글자 이상)과 남은 낱말 일부의 자모 닮음 하한
SLOT_JAMO_SHORT = 0.8      # 좁은 창에서 짧은 줄(SHORT_MAX 글자 이하)과 남은 낱말의 자모 닮음 하한
SLOT_JAMO_WIDE = 0.8       # 넓은 창에서 줄(4글자 이상)과 남은 낱말 일부의 자모 닮음 하한
SLOT_AREA = 160            # 좁은 창: 못 맞춘 줄 수 × 남은 낱말 수가 이 이하(우연히 닮을 짝의 수가 적다)
SLOT_CHANCE = 0.1          # 짧은 말은 '그 말이 영화에 나온 횟수 × 창의 낱말 수 ÷ 영화 낱말 수'가 이 이하일 때만(우연히 거기 있을 기대 횟수)
SLOT_CHANCE_NEAR = 0.2     # 그 값이 이 이하이고 앞뒤 맞춘 줄 가운데 하나에서 SHORT_NEAR 초 안이면 받는다('예' / '네' 처럼 흔한 대답)
PART_MIN = 4               # 줄의 일부만 말한 짝: 남은 낱말이 이 글자 수 이상(좁은 창)
PART_MIN_WIDE = 6          # 같은 것(넓은 창)
PART_RATIO = 0.85          # 줄의 일부만 말한 짝: 남은 낱말의 자모가 줄 안의 이어진 구간과 닮은 정도
REPAIR_LEAD = 3            # 넓은 창에서 후보 범위는 줄과 같은 두 글자가 있는 낱말의 이 낱말 수 앞에서부터 시작할 수 있다
SHORT_NEAR = 20.0          # 짧은 줄은 맞춘 이웃 대본 줄이 이 초 안에 있어야 남긴다(아니면 다시 짝짓기의 기준으로 다시 본다)
ACROSS_REPAIR = 1          # 1: pass 를 다 고른 뒤 고른 줄들 사이 창에서 기준·rescue 의 남은 낱말로 한 번 더 다시 짝짓는다(0 이면 끔)
SWAP_MIN = 6               # 순서 밖에서 되찾기: 이 글자 수 이상인 못 맞춘 줄만(0 이면 끔)
SWAP_R = 0.8               # 순서 밖에서 되찾기: 맞은 글자 ÷ 줄 글자, 글자 닮음 하한
SWAP_PAD = 30.0            # 순서 밖에서 되찾기: 앞뒤 맞춘 줄의 시각 밖으로 이 초까지 본다
# 남은 낱말
ATTACH_GAP = 0.3           # 옆 줄에 붙이는 틈(초). 붙는 낱말끼리도 이 틈 안으로 이어져 있어야 한다
ATTACH_EXTRA = 2           # 근거 없이 붙일 수 있는 글자 수 = 그 줄의 그 쪽(머리·꼬리)에 아직 안 맞은 대본 글자 수 + 이 값(한두 음절 군말·덧말)
ATTACH_RATIO = 0.75        # 그 글자 수는 줄 글자 수 × 이 값을 넘지 못한다(적어도 ATTACH_EXTRA)
ATTACH_MAX = 8             # 그 글자 수의 절대 상한
ATTACH_DROP = 0.25         # 근거 없이 붙였을 때 그 줄의 글자 닮음이 이만큼 넘게 떨어지면 붙이지 않는다
ATTACH_GAIN = 0.5          # 근거 있는 붙이기: 줄 쪽 낱말부터, 붙는 글자의 이 비율 이상이 대본 글자와 새로 맞는 데까지
ATTACH_GAIN_JAMO = 0.7     # 같은 것을 자모로 볼 때의 비율(받아쓰기가 글자를 틀린 경우)
ATTACH_JAMO_MIN = 3        # 자모로 보는 것은 붙는 글자가 이만큼 이상일 때만(짧으면 우연히 닮는다)
ADDED_GAP = 0.8            # added 줄을 자르는 쉼(초)
ADDED_MAX_WORDS = 12       # added 줄 하나의 낱말 수 상한(넘으면 가장 긴 쉼에서 나눈다)
ADDED_MAX_S = 6.0          # added 줄 하나의 길이 상한(초)
HABIT_SIM = 0.8            # added 묶음 안의 이어진 낱말 범위가 그 영화의 버릇 글귀(asr.film_habits)와 이만큼(rapidfuzz ratio, 글자·숫자만)
#                            닮으면 그 범위는 added 로 만들지 않는다. rescue 조각을 거르는 기준(asr.RESCUE_MIN_SIM 0.6)보다 엄하게 둔다:
#                            plain 낱말은 말 구간 안의 받아쓰기라 진짜 대사일 몫이 크다. 추격자에서 0.5~0.8 사이에 든 묶음은 없었다
CREDIT_TAIL = ("제공", "제작", "번역", "협찬", "감사합니다", "고맙습니다", "부탁드립니다", "부탁드려요", "부탁해요")
#                            자막 제작·구독 문구(textnorm.NOISE_CREDIT) 바로 뒤에 붙는 짝말(글자·숫자만 견줌). 문구와 함께 뺀다
# pass 고르기
SAME_NEAR_S = 0.5          # 한 줄의 두 pass 범위가 같은 소리 구간이라고 보는 기준: 두 범위가 겹치거나 사이 틈이 이 초 이하
#                            (pass 마다 낱말 시각이 다르다. 추격자의 turbo·large-v3 는 같은 낱말의 경계가 0.5초까지 어긋나고,
#                            쉼 뒤 첫 낱말의 시작은 앞으로 늘어나 있곤 해서 같은 한마디가 전혀 안 겹치기도 한다)
ALT_MARGIN = 0.02          # 기준과 대안이 둘 다 대본과 다를 때, 대안의 글자 닮음이 기준보다 이만큼 넘게 높아야 대안을 고른다
SPAN_UNION = 1             # 1: 한 줄의 기준·대안 범위를 서로의 범위와 같은 소리인 제 낱말로 넓힌 뒤 견준다(0 이면 끔)
SPAN_TRUST = 0.25          # 그때 다른 pass 의 범위는 그 pass 가 그 줄을 이 pass 보다 이만큼 넘게 못하게 맞추지 않았을 때만 근거로 쓴다
#                            (붙이기 전 범위의 대본과의 글자 닮음으로 견준다. 다시 짝짓기로 약하게 맞춘 엉뚱한 범위를 근거로 쓰지 않게)
ALT_CHARGE = 1.0           # 대안을 고르면 대안 줄이 덮지 못하는 기준 범위의 글자(added 로 떨어질 말)를 대안 쪽의 안 맞은 글자로
#                            이 배수만큼 센다(0 이면 끔)
LEFT_SAME_JAMO = 0.5       # 그 말을 대안이 같은 소리로 비슷하게(자모 닮음 이 값 이상) 적어 어느 줄에도 넣지 않았으면 세지 않는다
TAU_DEFAULT = 0.3          # 평균 로그 확률 차이 허용폭 기본값(자료가 TAU_MIN_N 줄 미만일 때)
TAU_MIN_N = 20
TAU_K = 2.0                # tau = TAU_K × (ctx − plain 차이의 90% 분위 편차), [TAU_LO, TAU_HI] 로 자름
TAU_LO, TAU_HI = 0.05, 1.0
CTX_BETTER = 0.3           # 둘 다 대본과 다를 때, ctx 의 다른 낱말이 plain 의 다른 낱말보다 이만큼 더 확실하면 ctx
CTX_ONLY_SIM = 0.8         # plain 이 못 맞춘 줄을 ctx 만 맞췄을 때 필요한 글자 닮음
CTX_FLOOR_Q = 0.05         # ctx 낱말 확률 문턱 = plain same 줄들의 평균 로그 확률의 이 분위
CTX_FLOOR_DEFAULT = math.log(0.5)
P_FLOOR = 1e-3             # 로그 확률을 낼 때 낱말 확률의 바닥
# conf
CONF_NOISE = 0.3           # 받아쓰기 헛소리 꼴인 added 줄의 conf 에 곱하는 값(자막 제작 문구는 늘, 인사말·되풀이는 확률이 낮을 때)
NOISE_P_LO = 0.5           # 인사말·되풀이 꼴: 낱말 확률 평균이 이 이하이면 CONF_NOISE 를 다 곱하고,
NOISE_P_HI = 0.9           #   이 이상이면 깎지 않는다(또렷이 들린 인사말·되풀이 대사). 그 사이는 곧은 줄로 잇는다
CONF_RESCUE = 0.8          # rescue pass 낱말로 만든 added 줄의 conf 에 곱하는 값(말 구간 검출이 말이 아니라고 본 곳의 받아쓰기)
CONF_LLM = 0.8             # LLM 짝 판정으로 정한 줄(decided_by llm)의 conf 에 곱하는 값. 그 짝은 규칙의 근거(글자·자모가 맞는 정도)가
#                            모자라 규칙이 버린 짝을 소리를 듣지 않고 글만 읽고 정한 것이다. 같은 글자 닮음·낱말 확률이면 규칙 짝보다 낮게
#                            둔다(식의 절반이 글자 닮음이라 그런 짝은 이미 낮게 나오고, 이 값은 그 위에 한 번 더 깎는 몫). 값은 근거 자료
#                            없이 rescue(CONF_RESCUE)와 같게 정했다
RESCUE_ECHO_S = 0.3        # rescue 낱말이 끝난 뒤 이 초 안에 시작하는 다른 pass 의 낱말이 같은 소리로 시작하면(자모 닮음
RESCUE_ECHO_SIM = 0.8      #   RESCUE_ECHO_SIM 이상) 그 소리를 시작만 늘려 한 번 더 적은 것으로 보고 rescue 낱말을 쓰지 않는다
CONF_AGREE = 0.5           # 대본 없이 받아쓴 다른 pass 가 그 줄에 같은 글을 적었으면 낱말 확률의 모자란 몫(1 − 평균)을 이 비율로 줄인다
CONF_DISAGREE = 0.85       # 대본 없이 받아쓴 다른 pass 가 그 줄에 다른 글을 적었으면 conf 에 곱하는 값
CONF_LONELY = 0.7          # 대본 앞뒤 대사가 모두 absent 인데 약하게(글자 닮음 LONELY_SIM 미만) 맞은 줄의 conf 에 곱하는 값
LONELY_SIM = 0.8
CONF_ABSENT_TOP = 0.95     # 주변에 설명 안 된 말이 전혀 없는 absent 의 conf
CONF_ABSENT_DROP = 0.6     # 설명 안 된 말이 줄에 필요한 시간만큼 있으면 이만큼 낮춘다
SEC_PER_CHAR = 0.13        # 한 글자를 말하는 시간(초)
FREE_MIN = 0.5             # 낱말이 덮지 못한 말 구간은 이 초 이상일 때만 센다(낱말 사이 짧은 틈은 말이 아니다)
ABSENT_PAD = 5.0           # absent 주변의 설명 안 된 말을 찾을 때 앞뒤 맞춘 줄 바깥으로 넓히는 시간(초)
ECHO_MIN = 0.6             # absent 줄의 글이 가까운 들린 글에 이 비율 이상 닮은 채 들어 있으면 그만큼 conf 를 낮춘다
ECHO_MIN_CHARS = 4         # 메아리는 이 글자 수 이상인 줄만 본다(짧은 말은 어디에나 있다)
CONF_TEXT = ("same·changed: 0.5 × 글자 닮음(2 × 맞은 글자 ÷ (대본 글자 + 들린 글자), 공백·문장부호 뺀 글자열) "
             "+ 0.5 × 낱말 확률 평균(확률이 없으면 글자 닮음만). "
             f"대본 없이 받아쓴 다른 pass 가 그 줄에 같은 글을 적었으면 낱말 확률 평균 p 를 1 − (1 − p) × {CONF_AGREE} 로 올리고, "
             f"다른 글을 적었으면 전체에 × {CONF_DISAGREE}. LLM 짝 판정으로 정한 줄(decided_by llm)은 × {CONF_LLM}. "
             f"대본에서 앞뒤 말 있는 대사가 모두 absent 인데 글자 닮음이 {LONELY_SIM} 미만이면 × {CONF_LONELY}. "
             f"added: 낱말 확률 평균(없으면 0.5). rescue pass 낱말이면 × {CONF_RESCUE}. 자막 제작 문구 꼴이면 × {CONF_NOISE}. "
             f"인사말만이거나 같은 낱말 되풀이 꼴이면 낱말 확률 평균이 {NOISE_P_LO} 이하일 때 × {CONF_NOISE}, "
             f"{NOISE_P_HI} 이상이면 그대로, 그 사이는 곧은 줄로 이은 값을 곱한다. LLM 짝 판정이 헛소리로 본 낱말(method.llm_pair.noise)이 "
             f"글자의 절반 이상이면 × {CONF_NOISE}(헛소리 꼴과 겹치면 둘 가운데 작은 값 하나만). "
             f"absent: {CONF_ABSENT_TOP} − {CONF_ABSENT_DROP} × max(min(1, 설명 안 된 말 시간 ÷ 필요한 시간), 메아리). "
             f"설명 안 된 말 시간은 대본 순서로 앞뒤 맞춘 줄 사이의 시간 창(양쪽으로 {ABSENT_PAD}초 넓힘)에 든 added 줄의 길이, "
             f"낱말이 덮지 못한 말 구간(vad, {FREE_MIN}초 이상)의 길이, 그 창에 걸친 맞춘 줄의 범위에서 대본과 안 맞은 "
             f"글자 몫의 시간(줄 길이 × 안 맞은 글자 ÷ 들린 글자)의 합. 필요한 시간은 그 사이 absent 줄 글자 × {SEC_PER_CHAR}초. "
             f"메아리는 그 줄({ECHO_MIN_CHARS}글자 이상)의 글이 그 창의 설명 안 된 글(added 줄의 글, 맞춘 줄의 들린 글에서 "
             f"대본과 안 맞은 글자)에 가장 닮게 들어 있는 정도(가장 잘 맞는 구간의 글자 닮음, {ECHO_MIN} 미만이면 0). "
             "말 없는 대사(글자 없는 요소)의 absent 는 null.")
CHOOSE_TEXT = ("기준 pass(context 가 null 인 pass 가운데 id plain, 없으면 맨 앞)와 대안 pass(context 가 null 인 나머지)를 "
               "따로 대본과 맞춘 뒤 줄마다 고른다(decided_by rule). 견주기 전에 두 범위를 서로의 범위 소리 안에 있는 제 남은 낱말로 넓힌다"
               f"(다른 pass 의 그쪽 글이 받쳐 줄 때만, 다른 pass 가 그 줄을 {SPAN_TRUST} 넘게 못하게 맞추지 않았을 때만, 바깥 끝의 대답·군말과 "
               "물음표 너머 차례는 빼고). 기준이 대본과 같으면 기준. 아니고 대안이 대본과 같으면 대안. "
               f"둘 다 다르면 대안의 글자 닮음이 {ALT_MARGIN} 넘게 높을 때만 대안(대안 줄이 덮지 못해 added 로 떨어질 기준 낱말의 글자는 "
               "대안 쪽의 안 맞은 글자로 센다. 대안이 그 소리를 어느 줄에도 안 넣은 낱말로 적었으면 세지 않는다). 글자 닮음이 똑같으면 "
               f"자모 닮음이 {ALT_MARGIN} 넘게 높은 대안. 다른 줄과 소리가 겹쳐 못 고른 대안은 모든 줄을 고른 뒤 다시 본다. "
               "가려 적은 글(한글 옆의 X, 자음만 적은 꼴)이 든 후보는 다른 "
               f"후보가 있으면 고르지 않는다. 대안은 기준 범위와 겹치거나 틈이 {SAME_NEAR_S}초 이하이고(같은 소리) 다른 줄이 가진 "
               "낱말과 겹치지 않을 때만 고른다. 기준이 못 맞춘 줄을 대안만 맞췄으면 다른 줄과 겹치지 않을 때 받아들인다. "
               "대본을 보고 받아쓴 pass(ctx)는 기준으로 고른 줄과 아무도 못 맞춘 줄에서만 낱말 확률로 견준다(decided_by acoustic). "
               "rescue pass 는 아무도 못 맞춘 줄만 찾고, 줄에 안 쓰인 rescue 낱말은 기준 pass 의 낱말·묶음과 어느 줄의 "
               f"시간 범위와도 겹치지 않고 바로 뒤({RESCUE_ECHO_S}초 안)에 같은 소리로 시작하는 다른 pass 의 낱말이 없을 때 added 로 "
               "만든다. 다른 pass 에서 고른 줄이 덮은 기준 낱말은 added 로 세지 않는다. 그 뒤 고른 줄(어느 pass 든) 사이 창에서 못 맞춘 "
               "줄과 기준·rescue 의 남은 낱말을 한 번 더 다시 짝짓는다(한 줄의 범위는 한 pass 안에서 이어진 낱말). "
               "added 로 남을 묶음 가운데 자막 제작·구독 문구(textnorm.NOISE_CREDIT 이 걸린 낱말과 바로 뒤의 짝말 CREDIT_TAIL, "
               "'by' 와 그 뒤 한 낱말)와, 그 영화의 버릇 글귀(asr.film_habits)와 "
               f"글자 닮음 {HABIT_SIM} 이상인 이어진 낱말 범위는 added 로 만들지 않고 noise_dropped 에 남긴다. "
               "그 뒤 LLM 짝 판정(llm_pair, decided_by llm)을 캐시에 있는 만큼 적용한다.")
TIME_SUSPECT_TEXT = (f"낱말 시각을 바로잡았지만 믿기 어려움(영상 길이 밖의 낱말이 있었거나 시작 시각이 "
                     f"{timeline.FAR_BACK_S}초 넘게 뒤집혀 있었음)")       # final 단계가 같은 줄에 적는 글과 같다(review 가 합친다)
# 문제 기록
MATCH_LOW = 0.3            # 일치율이 이 아래면 version_mismatch
ABSENT_MANY = 0.5          # 말 있는 대사 중 absent 비율이 이 위면 absent_many
ITEM_LOW_CONF = 0.2        # conf 가 이 아래인 same·changed·added 항목만 asr_low_conf
ITEM_PROBLEMS_MAX = 50     # 영화당 항목 수준 문제 상한
SHOW_ERRORS = 10           # 검사 실패 때 화면에 보여 주는 오류 수

CONSTANTS = {k: v for k, v in list(globals().items())
             if k.isupper() and isinstance(v, (int, float, tuple)) and not isinstance(v, bool)}

# 같은 뜻의 짧은 대답. 다시 짝짓기에서만 같은 말로 본다(판정은 그대로다: '예' 를 '네' 로 말했으면 changed).
ANSWER_CLASSES = (("예", "네"), ("응", "어", "엉"), ("아니요", "아뇨", "아니오", "아냐"), ("아녜요", "아니에요", "아니예요"))
_ANSWER = {w: g[0] for g in ANSWER_CLASSES for w in g}
# 혼자서 한 차례의 말이 되는 짧은 대답·되묻기. 그 쪽 대본 글자가 다 맞은 줄에는 근거 없이 붙이지 않는다(상대의 말일 수 있다).
# 물음표·마침표가 찍혀 있으면 문장 끝이라 어차피 따로 떨어진다. 이 목록은 받아쓰기가 부호를 빼먹었을 때를 위한 것이다.
REPLY_WORDS = frozenset(_ANSWER) | {"아니", "왜", "왜요", "뭐", "뭐요"}
_TOKEN_SPLIT = re.compile(r"[\W_]+")
_CONTINUES = (",", "，", "、", "…", "..", "-", "~")       # 낱말 끝에 붙으면 말이 이어지는 꼴


# ------------------------------------------------------------------ 자료 꼴
@dataclass(frozen=True)
class Line:
    """말 있는 대본 대사 한 줄."""
    idx: int               # 말 있는 대사 목록 안 번호(0부터)
    el: str                # 대본 요소 id
    scene: str | None
    text: str              # 요소 text
    key: str               # textnorm.key(text): same 판정 열쇠
    loc: str               # textnorm.loc(text): 자리 찾기용 글자열


@dataclass
class LineMatch:
    """한 pass 에서 대본 줄 하나가 맞은 자리."""
    line: int              # Line.idx
    el: str                # 대본 요소 id
    w0: int                # 낱말 범위 처음(pass 안 번호 i)
    w1: int                # 낱말 범위 끝(포함)
    hit: int               # 자리 찾기용 글자열에서 정확히 맞은 글자 수(LCS)
    n: int                 # 줄 글자 수
    m: int                 # 범위 글자 수
    sim: float             # 2 × hit ÷ (n + m)
    how: str               # anchors(닻 사슬) | lines(닻 사이 줄 맞추기) | moved(옮겨진 덩어리·순서 밖) | slot(다시 짝짓기)


class PassWords:
    """한 pass 의 낱말과 그 글자열."""

    def __init__(self, pass_id: str, words: list[dict], other: str | None = None, context: str | None = None):
        """other: 대본 대사의 열쇠 글자열(숫자·영문 약어를 읽는 꼴 가운데 대본에 있는 것으로 고르는 데 쓴다).
        context: asr.json pass 의 context(구실을 가르는 값. ``pass_roles``)."""
        words = sorted(words, key=lambda w: w["i"])
        self.id = pass_id
        self.context = context
        self.n = len(words)
        self.w = [str(x.get("w") or "").strip() for x in words]
        self.s = [float(x["s"]) for x in words]
        self.e = [max(float(x["e"]), float(x["s"])) for x in words]
        self.p = [x.get("p") for x in words]
        self.loc = [textnorm.loc(w, other) for w in self.w]
        self.backsteps = sum(b < a for a, b in zip(self.s, self.s[1:]))     # 시작 시각이 앞 낱말보다 이른 낱말 수
        self.off = [0]
        for c in self.loc:
            self.off.append(self.off[-1] + len(c))
        self.chars = "".join(self.loc)
        self.owner = [k for k, c in enumerate(self.loc) for _ in c]

    def span_loc(self, a: int, b: int) -> str:
        return self.chars[self.off[a]:self.off[b + 1]]

    def text(self, a: int, b: int) -> str:
        return " ".join(w for w in self.w[a:b + 1] if w)

    def start(self, a: int, b: int) -> float:
        return min(self.s[a:b + 1])

    def end(self, a: int, b: int) -> float:
        return max(self.e[a:b + 1])

    def logp(self, a: int, b: int) -> float | None:
        """낱말 확률의 평균 로그(확률 없는 낱말은 뺀다). 하나도 없으면 None."""
        ps = [p for p in self.p[a:b + 1] if p is not None]
        return sum(math.log(max(float(p), P_FLOOR)) for p in ps) / len(ps) if ps else None

    def mean_p(self, a: int, b: int) -> float | None:
        ps = [float(p) for p in self.p[a:b + 1] if p is not None]
        return sum(ps) / len(ps) if ps else None

    def mid(self, w: int) -> float:
        return (self.s[w] + self.e[w]) / 2

    def char_time(self, j: int) -> tuple[float, float]:
        """글자 j 의 시각(낱말 구간을 글자 수로 고르게 나눈 것)."""
        w = self.owner[j]
        n = len(self.loc[w])
        s, e = self.s[w], max(self.e[w], self.s[w] + 0.02)
        step = (e - s) / n
        k = j - self.off[w]
        return s + k * step, s + (k + 1) * step


def script_lines(script: dict, other: str | None = None) -> tuple[list[Line], list[str]]:
    """(말 있는 대사 줄 목록, 말 없는 대사 요소 id 목록). 대본 순서 그대로.
    other: 받아쓰기(기준 pass)의 열쇠 글자열. 대본의 숫자·영문 약어를 읽는 꼴 가운데 받아쓰기에 있는 것으로 고른다."""
    lines: list[Line] = []
    silent: list[str] = []
    for e in script.get("elements") or []:
        if e.get("type") != "dialogue":
            continue
        text = str(e.get("text") or "")
        k = textnorm.key(text)
        if not k:
            silent.append(e["id"])
            continue
        lines.append(Line(len(lines), e["id"], e.get("scene"), text, k, textnorm.loc(text, other)))
    return lines, silent


@dataclass(frozen=True)
class Roles:
    """받아쓰기 pass 들의 구실(pass id)."""
    base: str                    # 기준 pass: 대본 없이 받아쓴 pass 가운데 id 가 BASE_PASS 인 것, 없으면 맨 앞
    alts: tuple[str, ...]        # 대안 pass: 대본 없이 받아쓴 나머지(다른 모델로 받아쓴 둘째 후보)
    ctx: str | None              # 대본을 문맥으로 주고 받아쓴 pass(첫째 것만 쓴다)
    rescue: str | None           # 말 구간 검출이 놓친 곳을 다시 받아쓴 pass(첫째 것만 쓴다)
    unused: tuple[str, ...]      # 쓰지 않는 pass(모르는 context, 둘째 ctx·rescue)

    def of(self, pid: str) -> str:
        return ("base" if pid == self.base else "alt" if pid in self.alts else "ctx" if pid == self.ctx
                else "rescue" if pid == self.rescue else "unused")


def pass_roles(passes) -> Roles:
    """pass 의 구실을 context 로 가른다. passes: asr.json dict, 또는 (pass id, context) 짝의 목록.

    context 가 null 인 pass 는 대본 없이 받아쓴 후보다: id 가 BASE_PASS 인 것(없으면 맨 앞)이 기준, 나머지는 대안.
    context 가 CTX_CONTEXTS 에 있으면 대본 힌트 pass, RESCUE_CONTEXTS 에 있으면 rescue pass. 그 밖의 context 는 쓰지 않는다.
    context 가 null 인 pass 가 하나도 없으면 맨 앞 pass 를 기준으로 삼는다. pass 가 없으면 기준은 BASE_PASS(빈 pass)."""
    if isinstance(passes, dict):
        passes = [(p["id"], p.get("context")) for p in passes.get("passes") or []]
    pairs = list(passes)
    free = [pid for pid, c in pairs if c is None]
    base = BASE_PASS if BASE_PASS in free or not pairs else free[0] if free else pairs[0][0]
    rest = [(pid, c) for pid, c in pairs if pid != base]
    ctx = next((pid for pid, c in rest if c in CTX_CONTEXTS), None)
    rescue = next((pid for pid, c in rest if c in RESCUE_CONTEXTS), None)
    alts = tuple(pid for pid, c in rest if c is None)
    used = {base, ctx, rescue, *alts}
    return Roles(base, alts, ctx, rescue, tuple(pid for pid, _ in pairs if pid not in used))


def _base_pass(asr: dict) -> dict:
    """기준 pass(``pass_roles``). pass 가 없으면 빈 BASE_PASS."""
    base = pass_roles(asr).base
    return next((p for p in asr.get("passes") or [] if p.get("id") == base), {"id": base, "words": []})


def _words_key(words: list[dict]) -> str:
    return "".join(textnorm.key(str(w.get("w") or "")) for w in sorted(words, key=lambda w: w["i"]))


def _load(script: dict, asr: dict) -> tuple[list[Line], list[str], dict[str, "PassWords"]]:
    """(말 있는 대사 줄, 말 없는 대사 요소 id, {pass id: PassWords}). 숫자·영문 약어는 서로 맞은편 글을 보고 읽는다."""
    lines, silent = script_lines(script, _words_key(_base_pass(asr).get("words") or []))
    other = "".join(ln.key for ln in lines)
    passes = {p["id"]: PassWords(p["id"], p.get("words") or [], other, p.get("context")) for p in asr.get("passes") or []}
    return lines, silent, passes


_MASKED = re.compile(r"(?<=[가-힣])[Xx]+|[Xx]+(?=[가-힣])|(?<![A-Za-z])X{2,}(?![A-Za-z])|[ㄱ-ㅎ]{2,}")


def masked(text: str) -> bool:
    """받아쓰기가 가려 적은 글이 들었는가: 한글 바로 옆에 붙은 X('X됐다', '새X', '씨X새끼'), X 만 이어진 낱말('XX'),
    자음만 이어 적은 꼴('ㅅㄲ야', 'ㅅㅂ'). 추격자의 large-v3 받아쓰기가 욕설을 이렇게 적는다."""
    return bool(_MASKED.search(text))


verdict = textnorm.verdict     # (status, diffs) = verdict(대본 요소 text, 들린 글): same·changed 판정과 차이 조각


# ------------------------------------------------------------------ 닻
def _unique_blocks(s: str, a: str, k: int, cap: int = 1) -> list[tuple[int, int, int]]:
    """양쪽에서 한 번씩만 나오는 k-gram 을 씨앗으로 한 최대 일치 블록 (s 자리, a 자리, 길이) 목록.
    cap 을 올리면 양쪽에서 cap 번까지 나오는 k-gram 도 씨앗으로 쓴다(짝을 모두 만든다)."""
    if len(s) < k or len(a) < k:
        return []
    cs: dict[str, list[int]] = {}
    for p in range(len(s) - k + 1):
        cs.setdefault(s[p:p + k], []).append(p)
    ca: dict[str, list[int]] = {}
    for q in range(len(a) - k + 1):
        g = a[q:q + k]
        if g in cs and len(cs[g]) <= cap:
            ca.setdefault(g, []).append(q)
    seeds = sorted((p - q, p) for g, qs in ca.items() if len(qs) <= cap for q in qs for p in cs[g])
    runs: list[list[int]] = []                       # [대각선, 처음, 끝)
    for d, p in seeds:
        if runs and runs[-1][0] == d and p <= runs[-1][2]:
            runs[-1][2] = max(runs[-1][2], p + k)
        else:
            runs.append([d, p, p + k])
    out: list[list[int]] = []
    for d, p0, p1 in runs:
        while p0 > 0 and p0 - d > 0 and s[p0 - 1] == a[p0 - d - 1]:
            p0 -= 1
        while p1 < len(s) and p1 - d < len(a) and s[p1] == a[p1 - d]:
            p1 += 1
        if out and out[-1][0] == d and p0 <= out[-1][2]:
            out[-1][2] = max(out[-1][2], p1)
        else:
            out.append([d, p0, p1])
    return [(p0, p0 - d, p1 - p0) for d, p0, p1 in out]


def _chain(blocks: list[tuple[int, int, int]]) -> list[int]:
    """블록 (p, q, n) 가운데 p 와 q 가 함께 늘어나고 서로 겹치지 않는 사슬 중 n 의 합이 가장 큰 것(가중 LIS).
    고른 블록 번호를 차례대로 돌려준다."""
    if not blocks:
        return []
    order = sorted(range(len(blocks)), key=lambda t: blocks[t])
    q_ends = sorted({b[1] + b[2] for b in blocks})
    size = len(q_ends)
    tree_v = [0] * (size + 1)
    tree_a = [-1] * (size + 1)
    best = [0] * len(blocks)
    prev = [-1] * len(blocks)
    pending: list[tuple[int, int]] = []              # (p 끝, 블록 번호)
    for t in order:
        p, q, n = blocks[t]
        while pending and pending[0][0] <= p:
            _, u = heapq.heappop(pending)
            i = bisect.bisect_left(q_ends, blocks[u][1] + blocks[u][2]) + 1
            while i <= size:
                if best[u] > tree_v[i]:
                    tree_v[i], tree_a[i] = best[u], u
                i += i & -i
        i = bisect.bisect_right(q_ends, q)
        v, arg = 0, -1
        while i > 0:
            if tree_v[i] > v:
                v, arg = tree_v[i], tree_a[i]
            i -= i & -i
        best[t] = n + v
        prev[t] = arg
        heapq.heappush(pending, (p + n, t))
    top = max(order, key=lambda t: best[t])
    out = []
    while top >= 0:
        out.append(top)
        top = prev[top]
    return out[::-1]


class _Spans:
    """겹침을 묻는 구간 모음(반열린 구간)."""

    def __init__(self):
        self.items: list[tuple[int, int]] = []

    def add(self, lo: int, hi: int) -> None:
        bisect.insort(self.items, (lo, hi))

    def hits(self, lo: int, hi: int) -> bool:
        k = bisect.bisect_left(self.items, (lo, lo))
        for a, b in self.items[max(0, k - 1):]:
            if a >= hi:
                break
            if b > lo:
                return True
        return False

    def clip(self, lo: int, hi: int) -> tuple[int, int] | None:
        """[lo, hi) 의 앞뒤에서 이 구간들과 겹친 부분을 잘라낸 구간. 남는 것이 없거나 가운데가 겹치면 None."""
        for _ in range(2):
            k = bisect.bisect_right(self.items, (lo, float("inf"))) - 1
            if k >= 0 and self.items[k][1] > lo:
                lo = self.items[k][1]
            k = bisect.bisect_left(self.items, (hi, -1)) - 1
            if k >= 0 and self.items[k][0] < hi <= self.items[k][1]:
                hi = self.items[k][0]
        if lo >= hi or self.hits(lo, hi):
            return None
        return lo, hi

    def holes(self, lo: int, hi: int):
        """[lo, hi) 에서 이 구간들에 덮이지 않은 자리(range 하나 또는 목록)."""
        cut = [(max(a, lo), min(b, hi)) for a, b in self.items if a < hi and b > lo]
        if not cut:
            return range(lo, hi)
        out: list[int] = []
        at = lo
        for a, b in cut:
            out.extend(range(at, a))
            at = max(at, b)
        out.extend(range(at, hi))
        return out


def _moved(rest: list[tuple[int, int, int]], used_s: _Spans, used_a: _Spans) -> list[list[tuple[int, int, int]]]:
    """뼈대에 못 든 블록에서 옮겨진 덩어리를 찾는다. 덩어리마다 블록 목록(차례대로)."""
    clusters: list[list[tuple[int, int, int]]] = []

    def trim(b: tuple[int, int, int]) -> tuple[int, int, int] | None:
        """이미 쓴 구간과 겹친 앞뒤 글자만 잘라낸 블록. 블록은 양옆으로 글자가 같은 만큼 늘린 것이라, 옮겨진 씬의
        가장자리 글자가 이웃과 우연히 같으면 뼈대 블록과 한두 글자 겹친다. 통째로 버리면 그 씬을 놓친다."""
        p, q, n = b
        cs, ca = used_s.clip(p, p + n), used_a.clip(q, q + n)
        if cs is None or ca is None:
            return None
        head = max(cs[0] - p, ca[0] - q)
        tail = max(p + n - cs[1], q + n - ca[1])
        n2 = n - head - tail
        if n2 < K_LEVELS[-1] or used_s.hits(p + head, p + head + n2) or used_a.hits(q + head, q + head + n2):
            return None
        return p + head, q + head, n2

    for _ in range(MOVED_ROUNDS):
        rest = [t for t in map(trim, rest) if t is not None]
        if not rest:
            break
        ch = [rest[t] for t in _chain(rest)]
        groups = [[ch[0]]]
        for b0, b1 in zip(ch, ch[1:]):
            if b1[0] - (b0[0] + b0[2]) > MOVED_GAP_S or b1[1] - (b0[1] + b0[2]) > MOVED_GAP_A:
                groups.append([])
            groups[-1].append(b1)
        for g in groups:
            s0, s1 = g[0][0], g[-1][0] + g[-1][2]
            a0, a1 = g[0][1], g[-1][1] + g[-1][2]
            if sum(b[2] for b in g) >= MOVED_MIN and not used_s.hits(s0, s1) and not used_a.hits(a0, a1):
                clusters.append(g)
                used_s.add(s0, s1)
                used_a.add(a0, a1)
        # 사슬에 든 블록은 다시 보지 않는다(받아들인 것은 이미 쓴 구간이고, 못 받아들인 묶음은 다시 뽑아도 같다).
        # 가장 무거운 사슬이 기준에 못 미치는 작은 묶음들뿐일 때 거기서 멈추면, 사슬에 못 든 진짜 덩어리를 놓친다.
        tried = set(ch)
        rest = [b for b in rest if b not in tried]
    return clusters


def _sub(text: str, idx) -> str:
    if isinstance(idx, range):
        return text[idx.start:idx.stop]
    return "".join(text[i] for i in idx)


def _anchor_pairs(S: str, A: str) -> tuple[list[int], list[int], list[int]]:
    """대본 글자마다 (짝지은 받아쓰기 글자 번호 또는 -1, 덩어리 번호, 그 글자가 든 닻 블록 길이).
    덩어리 0 은 뼈대, 1 이상은 옮겨진 덩어리."""
    match = [-1] * len(S)
    seg = [0] * len(S)
    blen = [0] * len(S)
    if not S or not A:
        return match, seg, blen
    blocks: list[tuple[int, int, int]] = []
    for k in K_LEVELS:
        if k == K_LEVELS[0] or len(S) * len(A) <= (AREA_K4 if k == 4 else AREA_K3):
            blocks = _unique_blocks(S, A, k)
            if blocks:
                break
    chain = _chain(blocks)
    if sum(blocks[t][2] for t in chain) < SPARSE_COVER * min(len(S), len(A)):
        # 하나뿐인 조각이 거의 없다(대본에 같은 내용이 두 번 들어 있는 문서 등): 더 긴 조각을 몇 번 나와도 씨앗으로 쓴다
        alt = _unique_blocks(S, A, REPEAT_K, REPEAT_CAP)
        alt_chain = _chain(alt)
        if sum(alt[t][2] for t in alt_chain) > sum(blocks[t][2] for t in chain):
            blocks, chain = alt, alt_chain
    in_chain = set(chain)
    backbone = [blocks[t] for t in chain]
    used_s, used_a = _Spans(), _Spans()
    for p, q, n in backbone:
        used_s.add(p, p + n)
        used_a.add(q, q + n)
    hull_s, hull_a = _Spans(), _Spans()
    clusters = _moved([b for t, b in enumerate(blocks) if t not in in_chain], used_s, used_a)
    for g in clusters:
        hull_s.add(g[0][0], g[-1][0] + g[-1][2])
        hull_a.add(g[0][1], g[-1][1] + g[-1][2])

    stack: list[tuple[object, object, int]] = []     # (대본 자리들, 받아쓰기 자리들, 덩어리 번호)

    def put(si, ai, p: int, q: int, n: int, sid: int) -> None:
        for d in range(n):
            match[si[p + d]] = ai[q + d]
            seg[si[p + d]] = sid
            blen[si[p + d]] = n

    ident_s, ident_a = range(len(S)), range(len(A))
    ps = qs = 0
    for p, q, n in backbone:
        put(ident_s, ident_a, p, q, n, 0)
        stack.append((hull_s.holes(ps, p), hull_a.holes(qs, q), 0))
        ps, qs = p + n, q + n
    stack.append((hull_s.holes(ps, len(S)), hull_a.holes(qs, len(A)), 0))
    for sid, g in enumerate(clusters, 1):
        for k, (p, q, n) in enumerate(g):
            put(ident_s, ident_a, p, q, n, sid)
            if k:
                p0, q0, n0 = g[k - 1]
                stack.append((range(p0 + n0, p), range(q0 + n0, q), sid))

    while stack:
        si, ai, sid = stack.pop()
        ns, na = len(si), len(ai)
        if ns < K_LEVELS[-1] or na < K_LEVELS[-1]:
            continue
        area = ns * na
        s, a = _sub(S, si), _sub(A, ai)
        for k in K_LEVELS:
            if (k == 4 and area > AREA_K4) or (k == 3 and area > AREA_K3):
                break
            found = _unique_blocks(s, a, k)
            if not found:
                continue
            ps = qs = 0
            for t in _chain(found):
                p, q, n = found[t]
                put(si, ai, p, q, n, sid)
                stack.append((si[ps:p], ai[qs:q], sid))
                ps, qs = p + n, q + n
            stack.append((si[ps:], ai[qs:], sid))
            break
    return match, seg, blen


# ------------------------------------------------------------------ 줄 받아들이기
def _accept(n: int, hit: int, m: int, block: int = 0) -> bool:
    """줄(글자 n)이 낱말 범위(글자 m)에서 hit 글자 맞았을 때 맞춘 줄로 받아들일지."""
    if n <= SHORT_MAX:
        return hit == n == m
    r = hit / n
    sim = 2.0 * hit / (n + m)
    if r >= (R_MID if n <= MID_MAX else R_LONG) and sim >= SIM_MIN:
        return True
    return block >= BLOCK_OK and r >= R_BLOCK


def _solid(line_loc: str, span_loc: str) -> bool:
    """4~MID_MAX 글자 줄의 맞은 글자가 흩어져 있기만 한 것은 아닌가: 두 글자 이상 이어진 공통 조각이 있거나 자모가 닮았다
    (SLOT_JAMO_MIN 이상). 짧은 줄은 차례만 맞는 글자 세 개로도 R_MID 를 넘어서, 영상에 없는 줄이 대본에 없는 말과
    짝지어진다('야, 야, 이형사.' 와 '야, 준공이야 가사면'). 더 긴 줄과 SHORT_MAX 이하 줄은 늘 참."""
    n = len(line_loc)
    if n <= SHORT_MAX or n > MID_MAX:
        return True
    if any(line_loc[i:i + 2] in span_loc for i in range(n - 1)):
        return True
    return _jamo_sim(line_loc, span_loc) >= SLOT_JAMO_MIN


def _score(line_loc: str, span_loc: str) -> tuple[float, int]:
    h = textnorm.lcs(line_loc, span_loc)
    return h - BETA * (len(span_loc) - h), h


def _line_cands(line: Line, pw: PassWords, F: list[int], piece: list[int]) -> list[tuple[int, int, float]]:
    """줄 하나가 F(낱말 번호 목록) 안에서 맞을 수 있는 이어진 범위 후보 (처음 자리, 끝 자리 다음, 점수)."""
    n = len(line.loc)
    L = len(F)
    out: list[tuple[int, int, float]] = []
    if n <= SHORT_MAX:
        for b in range(L):
            if not pw.loc[F[b]]:
                continue
            acc = ""
            c = b
            while c < L and piece[c] == piece[b] and len(acc) < n:
                acc += pw.loc[F[c]]
                c += 1
            if acc == line.loc and pw.loc[F[c - 1]]:
                out.append((b, c, float(n)))
        return out
    chars = set(line.loc)
    cap = int(n * SPAN_MAX_RATIO) + SPAN_MAX_EXTRA
    need = math.ceil((R_MID if n <= MID_MAX else R_LONG) * n)
    for b in range(L):
        wl = pw.loc[F[b]]
        if not wl or chars.isdisjoint(wl):
            continue
        acc = ""
        c = b
        while c < L and piece[c] == piece[b]:
            wl = pw.loc[F[c]]
            acc += wl
            c += 1
            if len(acc) > cap:
                break
            if len(acc) < need or not wl or chars.isdisjoint(wl):
                continue
            sc, h = _score(line.loc, acc)
            if h < need or 2.0 * h / (n + len(acc)) < SIM_MIN or sc <= 0 or not _solid(line.loc, acc):
                continue
            out.append((b, c, sc))
    return out


def _dp(L: int, layers: list[list[tuple[int, int, float]]]) -> list[int]:
    """층(줄)마다 후보 (b, c, 점수) 가운데 하나를 고르거나 건너뛰어, 순서를 지키고 겹치지 않으면서 점수 합이
    가장 큰 짝을 고른다. 층마다 고른 후보 번호(-1 은 건너뜀)."""
    prev = [0.0] * (L + 1)
    hist: list[tuple[list[int], list[int]]] = []
    for cands in layers:
        cur = prev[:]
        arg = [-1] * (L + 1)
        for idx, (b, c, sc) in enumerate(cands):
            v = prev[b] + sc
            if v > cur[c] + 1e-9:
                cur[c] = v
                arg[c] = idx
        src = list(range(L + 1))
        for p in range(1, L + 1):
            if cur[p - 1] > cur[p] + 1e-9:
                cur[p] = cur[p - 1]
                src[p] = src[p - 1]
        hist.append((arg, src))
        prev = cur
    chosen = [-1] * len(layers)
    p = L
    for k in range(len(layers) - 1, -1, -1):
        arg, src = hist[k]
        q = src[p]
        chosen[k] = arg[q]
        p = layers[k][arg[q]][0] if arg[q] >= 0 else q
    return chosen


def _pieces(F: list[int], cuts: set[int] | frozenset = frozenset()) -> list[int]:
    """낱말 번호 목록 F 의 낱말마다 조각 번호: 번호가 이어지지 않는 곳과 cuts 에 든 낱말 앞에서 조각이 바뀐다."""
    piece = []
    pid = 0
    for k, w in enumerate(F):
        if k and (w != F[k - 1] + 1 or w in cuts):
            pid += 1
        piece.append(pid)
    return piece


# ------------------------------------------------------------------ 한 pass 맞추기
def _as_lines(script) -> list[Line]:
    return script if isinstance(script, list) else script_lines(script)[0]


def _as_words(words) -> PassWords:
    return words if isinstance(words, PassWords) else PassWords(BASE_PASS, words)


def match_pass(script, words) -> list[LineMatch]:
    """한 pass 의 낱말을 대본과 맞춘다.

    script: script.json dict 또는 ``script_lines`` 가 준 Line 목록. words: asr.json 한 pass 의 words 목록
    (또는 PassWords). 돌려주는 것: 맞춘 줄마다 LineMatch(줄 번호 순). 못 맞춘 줄은 목록에 없다.
    낱말 범위는 줄끼리 겹치지 않는다."""
    if isinstance(script, dict) and not isinstance(words, PassWords):
        lines = script_lines(script, _words_key(words))[0]
        pw = PassWords(BASE_PASS, words, "".join(ln.key for ln in lines))
    else:
        lines = _as_lines(script)
        pw = _as_words(words)
    if not lines or not pw.chars:
        return []
    S = "".join(ln.loc for ln in lines)
    s_off = [0]
    for ln in lines:
        s_off.append(s_off[-1] + len(ln.loc))
    match, seg, blen = _anchor_pairs(S, pw.chars)

    # 줄마다 닻의 주된 묶음 -> 씨앗 줄(낱말 범위의 알맹이)
    core: dict[int, list[int]] = {}                  # 줄 -> [처음 낱말, 끝 낱말, 덩어리, 가장 긴 블록]
    word_hits: dict[int, Counter] = {}
    for ln in lines:
        ks = [i for i in range(s_off[ln.idx], s_off[ln.idx + 1]) if match[i] >= 0]
        if not ks:
            continue
        runs = [[ks[0]]]
        for i0, i1 in zip(ks, ks[1:]):
            gap = pw.char_time(match[i1])[0] - pw.char_time(match[i0])[1]
            skipped = i1 - i0 - 1
            if (seg[i1] != seg[i0] or match[i1] < match[i0]
                    or gap > ANCHOR_GAP + ANCHOR_GAP_PER_CHAR * skipped
                    or match[i1] - match[i0] - 1 > ANCHOR_SKIP + ANCHOR_SKIP_RATIO * skipped):
                runs.append([])
            runs[-1].append(i1)
        run = max(runs, key=len)
        n, hit = len(ln.loc), len(run)
        hits = Counter(pw.owner[match[i]] for i in run)
        w0, w1 = pw.owner[match[run[0]]], pw.owner[match[run[-1]]]
        if n <= SHORT_MAX:
            ok = hit == n and pw.span_loc(w0, w1) == ln.loc
        else:
            ok = hit >= SEED_MIN_HIT or hit / n >= SEED_MIN_RATIO
            # 짧은 조각(3·4글자 닻)으로만 걸렸는데 그 조각의 글이 가까운 다른 대본 줄에도 있으면 씨앗으로 굳히지 않는다.
            # 비슷한 두 줄('너희 집 어디야. 내가 갈께.' 와 '너희 집 어디냐고.')에서 짧은 줄이 먼저 씨앗이 되면 긴 줄의 말
            # 한가운데를 가져가고 긴 줄은 absent 가 된다. 줄 단위 DP 에 넣어 이웃 줄과 점수로 다투게 한다.
            # (짧은 조각 씨앗을 모두 풀면 씨앗 사이 창이 넓어져, 영상에 없는 줄이 대본에 없는 말과 짝지어지는 일이 는다.)
            top = max(blen[i] for i in run)
            if ok and hit < n and top < SEED_MIN_BLOCK:
                i0 = next(i for i in run if blen[i] == top)
                piece = S[i0:i0 + top]
                near = [lines[k].loc for k in range(max(0, ln.idx - SEED_NEAR), min(len(lines), ln.idx + SEED_NEAR + 1))
                        if k != ln.idx]
                if any(piece[d:d + K_LEVELS[-1]] in other for other in near for d in range(len(piece) - K_LEVELS[-1] + 1)):
                    ok = False
            # 알맹이는 글자가 다 맞은 낱말까지만. 일부만 맞은 가장자리 낱말은 놓아 주고 줄 단위 DP 에서 다시 다툰다
            # (닻 블록이 줄 경계를 한두 글자 넘어 이웃 줄의 낱말을 끌어오는 것을 막는다).
            solid = [w for w in range(w0, w1 + 1) if hits[w] and hits[w] >= len(pw.loc[w])]
            if solid:
                w0, w1 = solid[0], solid[-1]
            else:
                w0 = w1 = max(range(w0, w1 + 1), key=lambda w: (hits[w], -w))
        if ok:
            core[ln.idx] = [w0, w1, seg[run[0]], max(blen[i] for i in run)]
            word_hits[ln.idx] = Counter({w: c for w, c in hits.items() if w0 <= w <= w1})

    # 알맹이끼리 낱말이 겹치면 글자가 더 많이 맞은 줄이 갖는다
    changed = True
    while changed:
        changed = False
        order = sorted(core, key=lambda li: (core[li][0], core[li][1], li))
        for x, y in zip(order, order[1:]):
            cx, cy = core[x], core[y]
            if cy[0] > cx[1]:
                continue
            changed = True
            w = cx[1]
            if cy[0] == w and word_hits[x][w] >= word_hits[y][w] and cy[1] > w:
                cy[0] += 1                                   # 낱말 하나에 걸침: 그 낱말에서 글자가 더 맞은 줄이 갖는다
                del word_hits[y][w]
            elif cy[0] == w and word_hits[x][w] < word_hits[y][w] and cx[0] < w:
                cx[1] -= 1
                del word_hits[x][w]
            elif cy[0] == w and (word_hits[x][w] >= word_hits[y][w]):
                del core[y]                                  # 그 낱말뿐인 줄이 졌으면 씨앗에서 뺀다
            elif cy[0] == w:
                del core[x]
            elif sum(word_hits[x].values()) >= sum(word_hits[y].values()):
                del core[y]
            else:
                del core[x]
            break

    word_line = [-1] * pw.n
    for li, c in core.items():
        for w in range(c[0], c[1] + 1):
            word_line[w] = li
    spans: dict[int, list] = {li: [c[0], c[1], "moved" if c[2] else "anchors"] for li, c in core.items()}

    def free_right(w: int) -> list[int]:
        out = []
        while w < pw.n and word_line[w] == -1 and len(out) < GAP_CAP:
            out.append(w)
            w += 1
        return out

    def free_left(w: int) -> list[int]:
        out = []
        while w >= 0 and word_line[w] == -1 and len(out) < GAP_CAP:
            out.append(w)
            w -= 1
        return out[::-1]

    def ext_cands(li: int, F: list[int], piece: list[int], right: bool) -> list[tuple[int, int, float]]:
        ln = lines[li]
        n = len(ln.loc)
        if n <= SHORT_MAX:
            return []
        a, b = spans[li][0], spans[li][1]
        base_loc = pw.span_loc(a, b)
        base, h0 = _score(ln.loc, base_loc)
        if h0 >= n:
            return []
        chars = set(ln.loc)
        cap = int(n * SPAN_MAX_RATIO) + SPAN_MAX_EXTRA
        out = []
        L = len(F)
        acc = ""
        ks = range(L) if right else range(L - 1, -1, -1)
        first = F[0] if right else F[-1]
        for t, k in enumerate(ks, 1):
            if piece[k] != piece[0 if right else L - 1] or F[k] != first + (t - 1 if right else 1 - t):
                break
            wl = pw.loc[F[k]]
            acc = acc + wl if right else wl + acc
            if len(base_loc) + len(acc) > cap:
                break
            if not wl or chars.isdisjoint(wl):
                continue
            sc, h = _score(ln.loc, base_loc + acc if right else acc + base_loc)
            if h > h0 and sc > base + 1e-9:
                out.append((0, t, sc - base) if right else (L - t, L, sc - base))
        return out

    # 씨앗 줄 사이: 남은 줄과 남은 낱말을 줄 단위 DP 로
    seeded = sorted(core)
    bounds = [None] + seeded + [None]
    for P, N in zip(bounds, bounds[1:]):
        lo = P + 1 if P is not None else 0
        hi = N if N is not None else len(lines)
        todo = list(range(lo, hi))
        f1 = free_right(spans[P][1] + 1) if P is not None else []
        f2 = free_left(spans[N][0] - 1) if N is not None else []
        if P is None and N is None:                      # 씨앗 줄이 하나도 없다(아주 짧은 글): 낱말 전체가 창
            f1, f2 = free_right(0), free_left(pw.n - 1)
        if f1 and f2 and f1[0] <= f2[0] <= f1[-1] + 1:   # 같은 틈(앞쪽 창과 뒤쪽 창이 닿거나 겹침)
            F = f1 + [w for w in f2 if w > f1[-1]]
        else:                                            # 다른 틈(옮겨진 덩어리의 경계)이거나 틈이 아주 길다
            F = f1 + f2
        if not F:
            continue
        piece = _pieces(F)
        layers: list[list[tuple[int, int, float]]] = []
        tags: list[tuple[str, int]] = []
        if P is not None and f1 and F[0] == spans[P][1] + 1:
            layers.append(ext_cands(P, F, piece, True))
            tags.append(("P", P))
        for li in todo:
            layers.append(_line_cands(lines[li], pw, F, piece))
            tags.append(("U", li))
        if N is not None and f2 and F[-1] == spans[N][0] - 1:
            layers.append(ext_cands(N, F, piece, False))
            tags.append(("N", N))
        if not any(layers):
            continue
        for (tag, li), cands, pick in zip(tags, layers, _dp(len(F), layers)):
            if pick < 0:
                continue
            b, c, _ = cands[pick]
            if tag == "P":
                spans[li][1] = F[c - 1]
            elif tag == "N":
                spans[li][0] = F[b]
            else:
                spans[li] = [F[b], F[c - 1], "lines"]
            for w in F[b:c]:
                word_line[w] = li

    out: dict[int, LineMatch] = {}
    for li in sorted(spans):
        w0, w1, how = spans[li]
        ln = lines[li]
        span = pw.span_loc(w0, w1)
        hit = textnorm.lcs(ln.loc, span)
        n, m = len(ln.loc), len(span)
        if _accept(n, hit, m, core[li][3] if li in core else 0) and _solid(ln.loc, span):
            out[li] = LineMatch(li, ln.el, w0, w1, hit, n, m, round(2.0 * hit / (n + m), 4), how)
    _drop_lonely_short(lines, out, lambda mt: (pw.start(mt.w0, mt.w1), pw.end(mt.w0, mt.w1)))
    _repair(lines, pw, out)
    _reclaim_moved(lines, pw, out)
    return [out[li] for li in sorted(out)]


def _jamo_sim(a: str, b: str) -> float:
    return textnorm.similarity(textnorm.jamo_string(a), textnorm.jamo_string(b))


def _polite_pair(a: str, b: str) -> bool:
    """두 짧은 되묻기·대답이 '요' 만 다른가('뭐요' / '뭐', '왜' / '왜요'). 둘 다 REPLY_WORDS 의 말이어야 한다."""
    return a != b and a in REPLY_WORDS and b in REPLY_WORDS and (a + "요" == b or b + "요" == a)


def _canon_tokens(ln: Line) -> list[str]:
    """줄의 낱말(열쇠) 가운데 짧은 대답(ANSWER_CLASSES)을 그 묶음의 첫 낱말로 바꾼 목록. 숫자·영문을 읽는 꼴로 바꾼 줄은
    낱말 경계가 달라지므로 줄 전체 한 덩어리."""
    toks = [k for k in (textnorm.key(t) for t in _TOKEN_SPLIT.split(ln.text)) if k]
    if "".join(toks) != ln.loc:
        return [ln.loc] if ln.loc else []
    return [_ANSWER.get(t, t) for t in toks]


def _canon_line(ln: Line) -> str:
    """다시 짝짓기용 글자열: 줄의 낱말 가운데 짧은 대답(ANSWER_CLASSES)을 그 묶음의 첫 낱말로 바꾼 것."""
    return "".join(_canon_tokens(ln))


def _repair(lines: list[Line], pw: PassWords, got: dict[int, LineMatch], cuts: set[int] | frozenset = frozenset(),
            edges: bool = False) -> None:
    """다시 짝짓기: 대본 순서로 이웃한 두 맞춘 줄 사이 창에서, 못 맞춘 줄과 그 사이의 남은 낱말을 순서를 지키며 짝짓는다.

    글자 기준(``_accept``)은 받아쓰기가 한두 글자 틀린 짧은 줄('괜찮니' / '괜창니'), 같은 뜻의 다른 대답('예' / '네'),
    줄의 일부만 말한 줄을 못 넘고, 이웃이 안 맞은 짧은 줄은 '외로운 짧은 줄'로 버려진다. 그 줄의 말은 added 로 남아
    같은 대사가 absent 와 added 로 갈린다. 창이 좁으면 우연히 닮을 짝이 적으니 기준을 낮춰 다시 본다.

    짝의 종류와 기준(창의 크기 = 못 맞춘 줄 수 × 남은 낱말 수. SLOT_AREA 이하면 좁은 창)
      같은 글   짧은 대답을 같은 말로 본 글자열(``_canon_line``)이 같다(되묻기는 '요' 만 다른 것도: '뭐요' / '뭐',
                ``_polite_pair``). 짧은 말(SHORT_MAX 글자 이하)은 우연히 그 창에
                있을 기대 횟수(그 말이 이 pass 에 나온 횟수 × 창의 낱말 수 ÷ pass 낱말 수)가 SLOT_CHANCE 이하일 때만.
                창의 낱말이 못 맞춘 줄들의 글자 수보다 적으면(잘린 대목) 기대 횟수에 그 비율을 곱한다.
      끝 낱말   edges 가 참일 때(pass 를 다 고른 뒤의 ``_repair_across`` 만), 좁은 창에서 낱말이 둘 이상인 줄의 첫 낱말이
                대답·되묻기(REPLY_WORDS)이면 그 낱말, 끝 낱말이 대답·되묻기이거나 2~SHORT_MAX 글자이면 그 낱말과 같은 글(군말
                꼴은 빼고, 기대 횟수 조건 그대로): '예? 하하... 나 이거. 참.' 을 '네?' 로, '노량진... 왜?' 를 '왜' 로만 말한 줄.
                줄 머리의 '이거', '안', '너' 같은 낱말은 잘린 줄 자리의 대본에 없는 말 머리와 우연히 같다(가짜 영화에서 잰 것).
                pass 하나만 보고 하면 다른 pass 가 그 줄을 더 많이 들은 것을 모르고 끝 낱말 하나로 줄을 차지한다.
      닮은 글   자모 닮음이 기준 이상이고 길이가 줄의 절반~두 배. 기준: 4글자 이상 줄은 창에 줄이 하나뿐이고 남은 낱말
                전체가 그 줄이면 SLOT_JAMO_MIN(앞뒤 맞춘 줄 사이가 SLOT_MAX_S 초 이하일 때), 좁은 창이면 SLOT_JAMO_MANY,
                넓은 창이면 SLOT_JAMO_WIDE. 짧은 줄은 좁은 창이거나 줄 하나·낱말 전체일 때만 SLOT_JAMO_SHORT.
                남은 낱말 쪽이 짧은 말이면(2 × SHORT_MAX + 2 글자 이하) 위의 기대 횟수 조건도 넘어야 한다.
      줄의 일부 남은 낱말(좁은 창 PART_MIN 글자, 넓은 창 PART_MIN_WIDE 글자 이상)이 줄 안의 이어진 구간과 자모로
                PART_RATIO 이상 닮았다(줄의 한 문장만 말한 경우. 나머지 말은 ``leftover`` 가 붙일지 정한다).
    앞뒤 맞춘 줄의 범위 가장자리에 있으면서 그 줄의 맞은 글자 수에 보탬이 안 되는 낱말은 이 창의 낱말로도 본다
    (그 낱말이 못 맞춘 줄과 짝이 되면 맞춘 줄의 범위를 줄인다). cuts: 이 낱말 번호 앞에서는 줄의 범위가 이어지지 못한다
    (여러 pass 의 낱말을 섞은 가상 pass 에서 pass 가 바뀌는 곳). got 을 제자리에서 고친다."""
    from rapidfuzz import fuzz

    if SLOT_MAX_S <= 0 or not pw.n:
        return
    owned = [False] * pw.n
    for m in got.values():
        for w in range(m.w0, m.w1 + 1):
            owned[w] = True
    can = [_ANSWER.get(c, c) for c in pw.loc]
    jam = [textnorm.jamo_string(c) for c in can]
    letters = sum(1 for c in can if c)
    seen: Counter = Counter()                            # 짧은 말(이어진 낱말의 글자열)이 이 pass 에 나온 횟수
    cap_seen = 2 * SHORT_MAX + 2
    for b in range(pw.n):
        if not can[b]:
            continue
        acc = ""
        for c in range(b, pw.n):
            acc += can[c]
            if len(acc) > cap_seen:
                break
            if can[c]:
                seen[acc] += 1

    def loose(li: int, tail: bool) -> int:
        """맞춘 줄 li 의 범위 끝(tail)·처음에서, 빼도 맞은 글자 수가 줄지 않는 낱말 수."""
        m, loc = got[li], lines[li].loc
        t = 0
        while m.w1 - m.w0 - t > 0:
            a, b = (m.w0, m.w1 - t - 1) if tail else (m.w0 + t + 1, m.w1)
            if textnorm.lcs(loc, pw.span_loc(a, b)) < m.hit:
                break
            t += 1
        return t

    def shrink(li: int, a: int, b: int) -> None:
        m, ln = got[li], lines[li]
        span = pw.span_loc(a, b)
        hit = textnorm.lcs(ln.loc, span)
        got[li] = LineMatch(li, ln.el, a, b, hit, m.n, len(span), round(2.0 * hit / (m.n + len(span)), 4), m.how)

    matched = sorted(got)
    bounds: list[int | None] = [None] + matched + [None]
    for A, B in zip(bounds, bounds[1:]):
        lo = A + 1 if A is not None else 0
        hi = B if B is not None else len(lines)
        todo = list(range(lo, hi))
        if not todo:
            continue
        w0 = got[A].w1 + 1 if A is not None else 0
        w1 = got[B].w0 - 1 if B is not None else pw.n - 1
        if w0 > w1 + 1 or any(owned[w0:w1 + 1]) or w1 - w0 + 1 > GAP_CAP:
            continue                                     # 앞뒤 줄의 시각이 뒤집혔거나(옮겨진 덩어리의 경계) 사이에 다른 줄의 낱말이 있다
        ta = loose(A, True) if A is not None else 0
        tb = loose(B, False) if B is not None else 0
        F = list(range(w0 - ta, w1 + 1 + tb))
        free = [w for w in range(w0, w1 + 1) if can[w]]
        if not free and not ta and not tb:
            continue
        n_free = max(1, len(free))
        narrow = len(todo) * n_free <= SLOT_AREA
        # 짧은 말이 우연히 이 창에 있을 기대 횟수의 계수. 창의 낱말이 못 맞춘 줄들을 다 말하기에 모자라면(잘린 대목)
        # 줄마다 말했을 가능성이 그만큼 낮으니 같은 비율로 키운다.
        need = sum(len(lines[li].loc) for li in todo)
        chance_k = n_free / max(1, letters) * max(1.0, need / max(1, sum(len(can[w]) for w in free)))
        near = (bool(free) and A is not None and B is not None and
                pw.start(got[B].w0, got[B].w1) - pw.end(got[A].w0, got[A].w1) <= SLOT_MAX_S)
        a_end = pw.end(got[A].w0, got[A].w1) if A is not None else None
        b_start = pw.start(got[B].w0, got[B].w1) if B is not None else None

        def close(x: int, y: int) -> bool:
            """짧은 말이 앞뒤 맞춘 줄 가운데 하나에서 SHORT_NEAR 초 안에 있는가(``_drop_lonely_short`` 와 같은 기준)."""
            gaps = [g for g in ((pw.s[F[x]] - a_end) if a_end is not None else None,
                                (b_start - pw.e[F[y]]) if b_start is not None else None) if g is not None]
            return not gaps or min(gaps) <= SHORT_NEAR
        if not narrow:                                   # 넓은 창은 줄과 같은 글자 조각이 있는 자리만 본다(아래)
            wtext = "".join(can[w] for w in F)
            woff = [0]
            for w in F:
                woff.append(woff[-1] + len(can[w]))
        layers: list[list[tuple[int, int, float]]] = []
        for li in todo:
            ln = lines[li]
            toks = _canon_tokens(ln)
            text = "".join(toks)
            ends = set()
            if edges and narrow and len(toks) >= 2:
                if toks[0] in REPLY_WORDS and not textnorm.is_filler(toks[0]):
                    ends.add(toks[0])
                if (toks[-1] in REPLY_WORDS or 2 <= len(toks[-1]) <= SHORT_MAX) and not textnorm.is_filler(toks[-1]):
                    ends.add(toks[-1])
            n = len(text)
            jl = textnorm.jamo_string(text)
            short = n <= SHORT_MAX
            if narrow:
                xs = range(len(F))
            else:
                # 넓은 창에서는 기준이 높아(같은 글, 자모 닮음 SLOT_JAMO_WIDE, 줄의 일부 PART_RATIO) 짝이 되려면 줄과 같은
                # 글자 조각이 있어야 한다. 짧은 줄은 그 글 전체, 긴 줄은 이어진 두 글자(6글자 이상 줄은 서로 다른 두 조각이
                # 가까이)가 있는 낱말 둘레에서만 후보를 만든다(줄 수 × 낱말 수만큼 닮음을 재지 않으려고: 대본과 영상이
                # 전혀 다른 구간이 길 때의 시간).
                hot: dict[int, set[str]] = {}
                for g in ({text} if short else {text[i:i + 2] for i in range(n - 1)}):
                    at = wtext.find(g)
                    while at >= 0:
                        hot.setdefault(bisect.bisect_right(woff, at) - 1, set()).add(g)
                        at = wtext.find(g, at + 1)
                lead = 0 if short else REPAIR_LEAD
                if n >= 2 * SHORT_MAX:
                    hot = {h: g for h, g in hot.items()
                           if len(set().union(*(hot.get(k, ()) for k in range(h - lead, h + lead + 1)))) >= 2}
                xs = sorted({x for h in hot for x in range(max(0, h - lead), h + 1)})
            cands: list[tuple[int, int, float]] = []
            for x in xs:
                if not can[F[x]]:
                    continue
                acc = ja = ""
                for y in range(x, len(F)):
                    if y > x and F[y] in cuts:
                        break
                    acc += can[F[y]]
                    ja += jam[F[y]]
                    m = len(acc)
                    if m > 2 * n + 2:
                        break
                    if not can[F[y]]:
                        continue
                    # 짧은 말(줄이든 남은 낱말이든)은 어디에나 있다: 우연히 이 창에 있을 기대 횟수가 낮을 때만 근거로 친다.
                    # 조금 흔한 말(SLOT_CHANCE_NEAR 까지)은 앞뒤 맞춘 줄 가운데 하나에서 SHORT_NEAR 초 안에 있어야 한다
                    chance = seen[acc] * chance_k
                    rare = (not short and m > cap_seen) or chance <= SLOT_CHANCE or \
                        (chance <= SLOT_CHANCE_NEAR and close(x, y))
                    if acc == text or (short and _polite_pair(text, acc)):
                        if rare or not short:
                            cands.append((x, y + 1, n + 1.0))
                        continue
                    if acc in ends and (chance <= SLOT_CHANCE or (chance <= SLOT_CHANCE_NEAR and close(x, y))):
                        cands.append((x, y + 1, PART_RATIO * m - 0.5))
                        continue
                    if math.ceil(n / 2) <= m and rare:
                        js = textnorm.similarity(jl, ja)
                        alone = len(todo) == 1 and near and F[x] <= free[0] and F[y] >= free[-1]
                        if short:
                            ok = (narrow or alone) and js >= SLOT_JAMO_SHORT
                        else:
                            ok = js >= (SLOT_JAMO_MIN if alone else SLOT_JAMO_MANY if narrow else SLOT_JAMO_WIDE)
                        if ok:
                            cands.append((x, y + 1, js * n))
                            continue
                    # 이어진 구간과의 닮음(partial_ratio)은 긴 줄에서 느리다. 먼저 줄 전체와의 맞은 자모 수로 거른다:
                    # 구간 닮음이 r 이상이려면 남은 낱말 자모의 r ÷ (2 − r) 이상이 줄에 차례대로 있어야 한다.
                    if m >= (PART_MIN if narrow else PART_MIN_WIDE) and n - m >= 2 and \
                            textnorm.lcs(ja, jl) >= PART_RATIO / (2.0 - PART_RATIO) * len(ja) and \
                            fuzz.partial_ratio(ja, jl) >= 100.0 * PART_RATIO:
                        cands.append((x, y + 1, PART_RATIO * m - 0.5))
            layers.append(cands)
        if not any(layers):
            continue
        first, last = w1 + 1, w0 - 1
        for li, cands, pick in zip(todo, layers, _dp(len(F), layers)):
            if pick < 0:
                continue
            x, y, _ = cands[pick]
            ln = lines[li]
            a, b = F[x], F[y - 1]
            span = pw.span_loc(a, b)
            hit = textnorm.lcs(ln.loc, span)
            n, m = len(ln.loc), len(span)
            got[li] = LineMatch(li, ln.el, a, b, hit, n, m, round(2.0 * hit / (n + m), 4), "slot")
            first, last = min(first, a), max(last, b)
        if A is not None and first <= got[A].w1:         # 앞 줄의 가장자리 낱말을 가져왔다
            shrink(A, got[A].w0, first - 1)
        if B is not None and last >= got[B].w0:
            shrink(B, last + 1, got[B].w1)


def _reclaim_moved(lines: list[Line], pw: PassWords, got: dict[int, LineMatch]) -> None:
    """순서 밖에서 되찾기: 못 맞춘 줄(SWAP_MIN 글자 이상)의 글이 앞뒤 맞춘 줄 가까이(SWAP_PAD 초)의 남은 낱말에
    거의 그대로(맞은 글자 비율과 글자 닮음이 SWAP_R 이상) 있으면 그 낱말을 그 줄로 삼는다(how "moved").
    씬 안에서 줄 두 개의 순서만 바뀌면 순서를 지키는 맞추기는 한 줄을 놓치고 같은 글이 added 로 남기 때문이다.
    got 을 제자리에서 고친다."""
    from rapidfuzz import fuzz

    if SWAP_MIN <= 0 or not got:
        return
    owned = [False] * pw.n
    for m in got.values():
        for w in range(m.w0, m.w1 + 1):
            owned[w] = True
    runs: list[tuple[int, int]] = []                     # 남은 낱말이 이어진 구간 [a, b]
    w = 0
    while w < pw.n:
        if owned[w]:
            w += 1
            continue
        a = w
        while w < pw.n and not owned[w]:
            w += 1
        if pw.off[w] - pw.off[a] >= SWAP_MIN * SWAP_R:
            runs.append((a, w - 1))
    if not runs:
        return
    matched = sorted(got)
    span = {li: (pw.start(got[li].w0, got[li].w1), pw.end(got[li].w0, got[li].w1)) for li in matched}
    found: list[tuple[float, int, int, int, int]] = []    # (점수, 줄, 처음 낱말, 끝 낱말, 맞은 글자)
    for ln in lines:
        n = len(ln.loc)
        if ln.idx in got or n < SWAP_MIN:
            continue
        k = bisect.bisect_left(matched, ln.idx)
        near = [span[matched[j]] for j in (k - 1, k) if 0 <= j < len(matched)]
        t0 = min(s for s, _ in near) - SWAP_PAD if k > 0 else float("-inf")
        t1 = max(e for _, e in near) + SWAP_PAD if k < len(matched) else float("inf")
        for a, b in runs:
            if pw.e[b] < t0 or pw.s[a] > t1:
                continue
            text = pw.chars[pw.off[a]:pw.off[b + 1]]
            al = fuzz.partial_ratio_alignment(ln.loc, text, score_cutoff=100.0 * SWAP_R - 15.0)
            if al is None:
                continue
            c0, c1 = pw.off[a] + al.dest_start, pw.off[a] + max(al.dest_end, al.dest_start + 1)
            x, y = pw.owner[c0], pw.owner[c1 - 1]
            best = None
            for x2 in (x, x + 1):                        # 가장자리 낱말은 넣은 것과 뺀 것을 다 본다
                for y2 in (y, y - 1):
                    if not (a <= x2 <= y2 <= b) or not (t0 <= (pw.s[x2] + pw.e[y2]) / 2 <= t1):
                        continue
                    sp = pw.span_loc(x2, y2)
                    hit = textnorm.lcs(ln.loc, sp)
                    if hit >= SWAP_R * n and 2.0 * hit >= SWAP_R * (n + len(sp)):
                        sc = hit - BETA * (len(sp) - hit)
                        if best is None or sc > best[0]:
                            best = (sc, ln.idx, x2, y2, hit)
            if best is not None:
                found.append(best)
    for sc, li, x, y, hit in sorted(found, key=lambda t: (-t[0], t[1], t[2])):
        if li in got or any(owned[x:y + 1]):
            continue
        ln = lines[li]
        m = pw.off[y + 1] - pw.off[x]
        got[li] = LineMatch(li, ln.el, x, y, hit, len(ln.loc), m, round(2.0 * hit / (len(ln.loc) + m), 4), "moved")
        for w in range(x, y + 1):
            owned[w] = True


def _drop_lonely_short(lines: list[Line], got: dict, span_of) -> None:
    """짧은 줄(SHORT_MAX 글자 이하)은 대본에서 바로 앞이나 뒤 줄이 맞춰져 있고 SHORT_NEAR 초 안에 있을 때만 남긴다.
    '예' 같은 말은 어디에나 있어서, 잘린 줄이 멀리 있는 같은 말에 붙는 것을 막는다. got 을 제자리에서 고친다."""
    while True:
        drop = []
        for li in sorted(got):
            if len(lines[li].loc) > SHORT_MAX:
                continue
            s, e = span_of(got[li])
            near = False
            for nb in (li - 1, li + 1):
                if nb in got:
                    s2, e2 = span_of(got[nb])
                    if max(s - e2, s2 - e, 0.0) <= SHORT_NEAR:
                        near = True
            if not near:
                drop.append(li)
        if not drop:
            return
        for li in drop:
            del got[li]


def match_window(lines: list[Line], todo: list[int], pw: PassWords, F: list[int],
                 cuts: set[int] | frozenset = frozenset()) -> dict[int, LineMatch]:
    """정해진 낱말 창 F(낱말 번호 목록, 오름차순) 안에서 줄 todo(대본 순서)를 줄 단위 DP 로 맞춘다.
    받아쓰지 않은 틈을 다시 받아쓴 pass(rescue)에 쓴다. cuts: 이 낱말 번호 앞에서는 줄의 범위가 이어지지 못한다
    (rescue 낱말 둘 사이에 다른 pass 가 받아쓴 말이 있는 곳: 한 줄이 그 말을 건너뛰어 이어지면 안 된다)."""
    if not F or not todo:
        return {}
    piece = _pieces(F, cuts)
    layers = [_line_cands(lines[li], pw, F, piece) for li in todo]
    if not any(layers):
        return {}
    out: dict[int, LineMatch] = {}
    for li, cands, pick in zip(todo, layers, _dp(len(F), layers)):
        if pick < 0:
            continue
        b, c, _ = cands[pick]
        ln = lines[li]
        span = pw.span_loc(F[b], F[c - 1])
        hit = textnorm.lcs(ln.loc, span)
        n, m = len(ln.loc), len(span)
        if _accept(n, hit, m):
            out[li] = LineMatch(li, ln.el, F[b], F[c - 1], hit, n, m, round(2.0 * hit / (n + m), 4), "lines")
    return out


# ------------------------------------------------------------------ 1차 맞춤(받아쓰기 단계가 부른다)
def _windows(lines: list[Line], spans: dict[int, tuple[float, float]], duration: float) -> dict[int, list[list[float]]]:
    """못 맞춘 줄마다 있을 수 있는 시간 창 목록. spans: 맞춘 줄 -> (시작, 끝).
    앞뒤 맞춘 줄이 시간 순서면 그 사이 한 창, 뒤집혀 있으면(옮겨진 덩어리의 경계) 앞 줄 뒤쪽 틈과 뒤 줄 앞쪽 틈 두 창."""
    order = sorted(spans.values())
    starts = [s for s, _ in order]
    ends = sorted(e for _, e in order)
    out: dict[int, list[list[float]]] = {}
    prev_end: float | None = None
    nxt: list[float | None] = [None] * len(lines)
    upcoming: float | None = None
    for li in range(len(lines) - 1, -1, -1):
        nxt[li] = upcoming
        if li in spans:
            upcoming = spans[li][0]
    for li in range(len(lines)):
        if li in spans:
            prev_end = spans[li][1]
            continue
        t0 = prev_end if prev_end is not None else 0.0
        t1 = nxt[li] if nxt[li] is not None else max(duration, t0)
        if t0 <= t1:
            out[li] = [[round(t0, 3), round(t1, 3)]]
            continue
        k = bisect.bisect_left(starts, t0)            # 앞 줄 끝 다음에 시작하는 첫 맞춘 줄
        a1 = starts[k] if k < len(starts) else max(duration, t0)
        k = bisect.bisect_right(ends, t1)             # 뒤 줄 시작 전에 끝나는 마지막 맞춘 줄
        b0 = ends[k - 1] if k else 0.0
        out[li] = [[round(t0, 3), round(max(a1, t0), 3)], [round(min(b0, t1), 3), round(t1, 3)]]
    return out


def prealign(script: dict, asr: dict, video_duration: float | None = None) -> dict:
    """기준 pass(plain)만으로 한 1차 맞춤. 받아쓰기 단계가 ctx 힌트 창과 rescue 틈 창을 정할 때 부른다.

    돌려주는 것::

        {"pass": "plain",                 # 맞춘 pass id
         "match": 0.812,                  # 맞춘 줄의 맞은 글자 합 ÷ 말 있는 대사 글자 합(자리 찾기용 글자열 기준)
         "lines": [                       # 말 있는 대본 대사마다 하나, 대본 순서
           {"el": "E0020", "text": "...", "chars": 12,
            "matched": True,
            "words": [1520, 1524],        # 맞춘 낱말 범위(pass 안 번호, 끝 포함). 못 맞췄으면 None
            "start": 723.4, "end": 725.1, # 맞춘 범위의 시각. 못 맞췄으면 None
            "sim": 0.91,                  # 글자 닮음. 못 맞췄으면 None
            "windows": [[723.4, 725.1]]}  # 이 줄이 있을 시간 창. 맞춘 줄은 제 구간 하나.
           ...]}                          # 못 맞춘 줄은 '앞뒤 맞춘 줄 사이' 창 하나, 앞뒤 줄의 시각이 뒤집힌 곳
                                          # (옮겨진 덩어리의 경계)에서는 앞 줄 뒤쪽 틈과 뒤 줄 앞쪽 틈 두 개.
                                          # 맨 앞은 0, 맨 뒤는 video_duration(없으면 마지막 낱말 끝)이 끝점.
    """
    lines, _, passes = _load(script, asr)
    base_id = _base_pass(asr)["id"]
    pw = passes.get(base_id) or PassWords(base_id, [])
    got = {m.line: m for m in match_pass(lines, pw)}
    duration = float(video_duration) if video_duration else (max(pw.e) if pw.n else 0.0)
    spans = {li: (pw.start(m.w0, m.w1), pw.end(m.w0, m.w1)) for li, m in got.items()}
    wins = _windows(lines, spans, duration)
    rows = []
    for ln in lines:
        m = got.get(ln.idx)
        if m is not None:
            s, e = round(spans[ln.idx][0], 3), round(spans[ln.idx][1], 3)
            rows.append({"el": ln.el, "text": ln.text, "chars": len(ln.loc), "matched": True, "words": [m.w0, m.w1],
                         "start": s, "end": e, "sim": m.sim, "windows": [[s, e]]})
        else:
            rows.append({"el": ln.el, "text": ln.text, "chars": len(ln.loc), "matched": False, "words": None,
                         "start": None, "end": None, "sim": None, "windows": wins[ln.idx]})
    total = sum(len(ln.loc) for ln in lines)
    return {"pass": pw.id, "match": round(sum(m.hit for m in got.values()) / total, 3) if total else 0.0, "lines": rows}


# ------------------------------------------------------------------ pass 고르기
@dataclass
class Choice:
    """줄 하나에 고른 pass 와 낱말 범위."""
    pass_id: str
    w0: int
    w1: int
    how: str
    decided_by: str                                   # rule | acoustic | llm
    cands: dict = field(default_factory=dict)         # pass id -> (w0, w1): 후보로 견준 범위


def _quantile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[int(q * (len(xs) - 1))]


def _same_sound(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """한 줄을 두 pass 가 맞춘 시간 구간 a, b 가 같은 소리 구간인가: 겹치거나, 사이 틈이 SAME_NEAR_S 초 이하다.
    겹친 비율로 보지 않는다: pass 마다 낱말 시각이 달라 같은 한마디('예')가 전혀 안 겹치기도 하고, 한 pass 가 줄의 일부만
    맞췄으면 길이가 크게 다르다. 한 범위가 이웃 줄의 소리까지 삼키는 것은 ``_Claims.clashes`` 가 따로 막는다."""
    return max(a[0], b[0]) - min(a[1], b[1]) <= SAME_NEAR_S


def _same_word(s1: float, e1: float, s2: float, e2: float) -> bool:
    """서로 다른 pass 의 두 낱말이 같은 소리인가: 겹친 시간이 짧은 쪽 낱말 길이의 절반 이상이다.

    pass 마다 낱말 시각이 다르다. 쉼 뒤 첫 낱말은 시작이 앞으로, 쉼 앞 끝 낱말은 끝이 뒤로 늘어나 있곤 해서(2초짜리 '근데')
    낱말의 가운데 시각이나 줄의 시간 구간으로 보면 같은 소리를 놓치거나 옆 소리를 잡는다. 짧은 쪽을 기준으로 하면 한 pass 가
    붙여 적은 낱말과 다른 pass 가 나눠 적은 낱말, 늘어난 낱말과 제 길이 낱말이 모두 같은 소리로 잡힌다."""
    short = min(e1 - s1, e2 - s2)
    ov = min(e1, e2) - max(s1, s2)
    if short <= 0.0:                                     # 길이가 0 인 낱말은 다른 낱말 안에 있으면 같은 소리
        return ov >= 0.0
    return ov > 0.0 and ov >= 0.5 * short


class _Claims:
    """고른 줄들이 가진 낱말 범위. 서로 다른 pass 에서 온 두 줄이 같은 소리를 함께 갖는지 묻는 데 쓴다.

    한 번 난 소리는 한 줄에만 들어가야 한다. 같은 pass 안에서는 낱말 범위가 겹치지 않으면 되지만, 다른 pass 의 줄끼리는
    낱말 번호로 견줄 수 없어 시각으로 본다. 두 줄의 낱말 가운데 같은 소리인 짝(``_same_word``)이 있거나, 한 줄의 가운데
    시각이 다른 줄의 시간 범위 안이면(한 줄이 다른 줄의 쉼 안에 들어앉은 꼴) 겹친 것이다."""

    def __init__(self, passes: dict[str, PassWords]):
        self.passes = passes
        self.rows: dict[int, tuple[str, int, int, float, float]] = {}      # 줄 -> (pass id, w0, w1, 시작, 끝)
        self._sorted: list[tuple[float, float, int]] | None = None
        self._starts: list[float] = []
        self._longest = 0.0

    def put(self, li: int, pid: str, w0: int, w1: int) -> None:
        pw = self.passes[pid]
        self.rows[li] = (pid, w0, w1, pw.start(w0, w1), pw.end(w0, w1))
        self._sorted = None

    def drop(self, li: int) -> None:
        if self.rows.pop(li, None) is not None:
            self._sorted = None

    def _near(self, s: float, e: float):
        """시간 구간이 [s, e] 와 닿는 줄 번호들."""
        if self._sorted is None:
            self._sorted = sorted((r[3], r[4], li) for li, r in self.rows.items())
            self._starts = [r[0] for r in self._sorted]
            self._longest = max((b - a for a, b, _ in self._sorted), default=0.0)
        j = bisect.bisect_right(self._starts, e) - 1
        while j >= 0 and self._sorted[j][0] >= s - self._longest:
            if self._sorted[j][1] >= s:
                yield self._sorted[j][2]
            j -= 1

    def _hits(self, pw: PassWords, w0: int, w1: int, other: PassWords, a: int, b: int) -> bool:
        return any(_same_word(pw.s[y], pw.e[y], other.s[x], other.e[x])
                   for y in range(w0, w1 + 1) for x in range(a, b + 1))

    def clashes(self, li: int, pid: str, w0: int, w1: int) -> bool:
        """줄 li 를 pass pid 의 낱말 [w0, w1] 로 삼으면, 다른 pass 에서 고른 다른 줄과 같은 소리를 함께 갖게 되는가."""
        pw = self.passes[pid]
        s, e = pw.start(w0, w1), pw.end(w0, w1)
        for k in self._near(s, e):
            q, a, b, s2, e2 = self.rows[k]
            if k == li or q == pid:
                continue
            if s <= (s2 + e2) / 2 <= e or s2 <= (s + e) / 2 <= e2 or self._hits(pw, w0, w1, self.passes[q], a, b):
                return True
        return False

    def covered(self, pid: str, w: int) -> bool:
        """pass pid 의 낱말 w 가 다른 pass 에서 고른 어느 줄의 시간대에 있는가: 그 줄의 낱말과 같은 소리이거나,
        가운데 시각이 그 줄의 시간 범위 안이다(그 줄의 쉼 안에 든 낱말도 그 줄의 시간대다)."""
        pw = self.passes[pid]
        m = pw.mid(w)
        for k in self._near(pw.s[w], pw.e[w]):
            q, a, b, s2, e2 = self.rows[k]
            if q != pid and (s2 <= m <= e2 or self._hits(pw, w, w, self.passes[q], a, b)):
                return True
        return False

    def under(self, q: str, a: int, b: int, pid: str) -> list[int]:
        """pass q 의 낱말 [a, b] 와 같은 소리인 pass pid 의 낱말 번호(오름차순)."""
        pw, other = self.passes[pid], self.passes[q]
        s, e = other.start(a, b), other.end(a, b)
        return [w for w in range(pw.n) if pw.s[w] <= e and pw.e[w] >= s and self._hits(pw, w, w, other, a, b)]


def _differing_logp(pa: PassWords, ra: tuple[int, int], pb: PassWords, rb: tuple[int, int]
                    ) -> tuple[float | None, float | None]:
    """두 pass 가 같은 줄에 적은 낱말 가운데 서로 다른 낱말만의 평균 로그 확률 (a 쪽, b 쪽).
    줄 전체 평균은 긴 줄에서 한 낱말의 낮은 확률을 묻어 버린다. 한쪽에 다른 낱말이 없으면 그쪽은 줄 전체 평균."""
    ka = [textnorm.key(pa.w[w]) for w in range(ra[0], ra[1] + 1)]
    kb = [textnorm.key(pb.w[w]) for w in range(rb[0], rb[1] + 1)]
    left = Counter(kb)
    only_a = []
    for w, k in zip(range(ra[0], ra[1] + 1), ka):
        if left[k] > 0:
            left[k] -= 1
        else:
            only_a.append(w)
    left = Counter(ka)
    only_b = []
    for w, k in zip(range(rb[0], rb[1] + 1), kb):
        if left[k] > 0:
            left[k] -= 1
        else:
            only_b.append(w)

    def mean(pw: PassWords, ws: list[int], whole: tuple[int, int]) -> float | None:
        ps = [pw.p[w] for w in ws if pw.p[w] is not None]
        if not ps:
            return pw.logp(*whole)
        return sum(math.log(max(float(p), P_FLOOR)) for p in ps) / len(ps)

    return mean(pa, only_a, ra), mean(pb, only_b, rb)


def _sounds_in(pw: PassWords, w: int, other: PassWords, x0: int, x1: int) -> bool:
    """pass pw 의 낱말 w 가 다른 pass 의 낱말 [x0, x1] 의 소리 안에 있는가: 그 가운데 한 낱말과 같은 소리이거나(``_same_word``),
    가운데 시각이 그 범위의 시간 안이다."""
    if x0 > x1:
        return False
    if other.start(x0, x1) <= pw.mid(w) <= other.end(x0, x1):
        return True
    return any(_same_word(pw.s[w], pw.e[w], other.s[x], other.e[x]) for x in range(x0, x1 + 1))


_ASIDE_MORE = frozenset({"그래", "그래요"})       # 맞장구도 혼자서 한 차례가 된다


def _aside(k: str) -> bool:
    """혼자서 한 차례가 될 수 있는 짧은 말(대답·되묻기 REPLY_WORDS, 맞장구, 한두 음절 군말): 누구의 말인지 글로는 못 가른다."""
    return k in REPLY_WORDS or k in _ASIDE_MORE or (len(k) <= textnorm.FILLER_SHORT and textnorm.is_filler(k))


def _asks(word: str) -> bool:
    """낱말이 물음표로 끝나는가(뒤따르는 닫는 따옴표·괄호는 넘긴다)."""
    return word.strip().rstrip("\"'”’」』)]").endswith(("?", "？"))


def _extend_over(pw: PassWords, a: int, b: int, other: PassWords, x0: int, x1: int, free: list[bool],
                 line_text: str = "") -> tuple[int, int]:
    """pass pw 의 범위 [a, b] 를 양 끝에서, 다른 pass 가 같은 줄로 맞춘 범위 [x0, x1] 의 소리 안에 있는 pw 의 낱말로 넓힌다.

    두 pass 는 따로 대본과 맞췄으니, 한 pass 가 그 줄로 본 소리를 다른 pass 가 남은 낱말로 두었다면 그 낱말은 그 줄의 말이다
    (줄의 머리·꼬리를 바꿔 말해 한 pass 에서만 붙은 경우). 넓히지 않으면 그 말이 그 줄 옆의 added 로 떨어진다.
    넓히는 낱말은
      어느 다른 줄의 범위에도 들지 않았고(free) 범위에 이어져 있으며, 다른 pass 범위의 소리 안에 있다(``_sounds_in``).
      다른 pass 가 이 범위 앞(시작이 이 범위 시작보다 이른 낱말)·뒤(끝이 이 범위 끝보다 늦은 낱말)에 적은 글이 받쳐 준다:
        넓힌 낱말들의 자모 절반 이상이 그 글에 차례대로 있다(낱말 시각이 늘어나 옆 소리까지 덮은 다른 pass 의 낱말 때문에
        엉뚱한 말이 붙지 않게).
      넓힌 범위의 바깥 끝 낱말이 대답·되묻기·한두 음절 군말(``_aside``)이면 대본 줄의 그 쪽 끝이 같은 말일 때만 둔다
        (상대의 말일 수 있다. 넓힌 범위 안쪽의 '네?' 같은 말은 그 줄의 말이다).
      물음표를 넘어 차례가 바뀌는 곳은 넘지 않는다: 대본 줄이 물음으로 끝나면 물음표로 끝난 낱말 뒤로(그 뒤는 상대의 대답일 수
        있다), 대본 줄에 물음이 없으면 물음표로 끝난 낱말 앞으로(앞 사람의 물음일 수 있다) 넓히지 않는다."""
    line_key = textnorm.key(line_text)
    ends_asking = line_text.strip().rstrip("\"'”’」』)]").rstrip(".…!~ ").endswith(("?", "？"))
    asks_inside = "?" in line_text or "？" in line_text
    def jam(ws) -> str:
        return textnorm.jamo_string("".join(textnorm.key(pw.w[w]) for w in ws))

    t0, t1 = pw.start(a, b), pw.end(a, b)
    # 다른 pass 가 이 범위 소리 밖, 앞·뒤에 적은 글
    rest = [x for x in range(x0, x1 + 1) if not _sounds_in(other, x, pw, a, b)]
    before = textnorm.jamo_string("".join(textnorm.key(other.w[x]) for x in rest if other.mid(x) < t0))
    after = textnorm.jamo_string("".join(textnorm.key(other.w[x]) for x in rest if other.mid(x) > t1))

    def backed(ext: str, head: bool) -> bool:
        """넓힌 낱말(자모 ext)의 절반 이상이 다른 pass 가 그쪽에 적은 글에 차례대로 있다."""
        return bool(ext) and textnorm.lcs(ext, before if head else after) >= 0.5 * len(ext)

    def ok(w: int) -> bool:
        return bool(textnorm.key(pw.w[w])) and free[w] and _sounds_in(pw, w, other, x0, x1)

    def aside_edge(w: int, head: bool) -> bool:
        """바깥 끝 낱말 w 가 대답·군말인데 대본 줄의 그 끝 말이 아니거나, 그 말을 이 범위가 이미 그 끝에 갖고 있다."""
        k = textnorm.key(pw.w[w])
        if not _aside(k):
            return False
        edge = line_key[:len(k)] if head else line_key[len(line_key) - len(k):]
        mine = textnorm.key(pw.w[a if head else b])
        return _ANSWER.get(edge, edge) != _ANSWER.get(k, k) or _ANSWER.get(mine, mine) == _ANSWER.get(k, k)

    a2 = a
    while a2 > 0 and ok(a2 - 1) and not (_asks(pw.w[a2 - 1]) and not asks_inside) \
            and backed(jam(range(a2 - 1, a)), True):
        a2 -= 1
    while a2 < a and aside_edge(a2, True):
        a2 += 1
    b2 = b
    while b2 + 1 < pw.n and ok(b2 + 1) and not (_asks(pw.w[b2]) and ends_asking) \
            and backed(jam(range(b + 1, b2 + 2)), False):
        b2 += 1
    while b2 > b and aside_edge(b2, False):
        b2 -= 1
    return a2, b2


def _left_out(base: PassWords, a: int, b: int, other: PassWords, x0: int, x1: int, other_free: list[bool]) -> int:
    """기준 범위 [a, b] 의 낱말 가운데 다른 pass 의 범위 [x0, x1] 가 덮지 못하는 것(그 소리 안에 없고 같은 글로 적히지도
    않은 것)의 열쇠 글자 수. 다른 pass 의 줄을 고르면 이 말은 그 줄 옆의 added 로 떨어진다. 세지 않는 것:
      대답·되묻기·한두 음절 군말(``_aside``): 상대의 말일 수 있다.
      다른 pass 가 같은 소리를 어느 줄에도 넣지 않은 낱말로, 비슷한 글(자모 닮음 LEFT_SAME_JAMO 이상)로 적은 것: 다른
        pass 는 그 말을 듣고도 이 줄 밖으로 두었다(두 pass 가 엇갈리니 글로는 못 가른다)."""
    echo = set(_echoed(base, a, b, other, x0, x1))
    out = 0
    for w in range(a, b + 1):
        k = textnorm.key(base.w[w])
        if w in echo or _aside(k) or _sounds_in(base, w, other, x0, x1):
            continue
        lo = bisect.bisect_left(other.s, base.s[w] - 30.0)
        if any(other_free[x] and _same_word(base.s[w], base.e[w], other.s[x], other.e[x])
               and _jamo_sim(k, textnorm.key(other.w[x])) >= LEFT_SAME_JAMO
               for x in range(lo, other.n) if other.s[x] <= base.e[w]):
            continue
        out += len(k)
    return out


def _rivals(claims: _Claims, base_id: str, owned: list[bool], pid: str, w0: int, w1: int) -> tuple[int, int] | None:
    """pass pid 의 낱말 [w0, w1] 자리에 기준 pass 가 적은 글: 그 낱말들과 같은 소리이고 어느 줄에도 안 쓰인 기준 낱말의
    범위(없으면 None). 다른 pass 만 맞춘 줄의 후보로 남긴다."""
    base = claims.passes[base_id]
    ws = [w for w in claims.under(pid, w0, w1, base_id) if not owned[w] and base.loc[w]]
    return (ws[0], ws[-1]) if ws else None


def decide(lines: list[Line], passes: dict[str, PassWords], matches: dict[str, dict[int, LineMatch]],
           roles: Roles | None = None, cores: dict[str, dict[int, LineMatch]] | None = None
           ) -> tuple[dict[int, Choice], dict]:
    """줄마다 pass 를 고른다(rescue 는 ``build_ops`` 가 따로 넣는다).

    passes: {pass id: PassWords}, matches: {pass id: {줄 번호: LineMatch}}, roles: pass 의 구실(None 이면 PassWords 의
    context 로 ``pass_roles`` 가 가른다). 돌려주는 것: ({줄 번호: Choice}, 통계 dict).

    규칙 1. 기준 pass 와 대안 pass(둘 다 대본을 보지 않은 받아쓰기라 대본 쪽으로 끌리는 치우침이 없다: decided_by rule)
      기준 범위가 대본과 같으면 기준.
      기준이 다르고 대안 범위가 대본과 같으면 대안.
      둘 다 대본과 다르면 기준. 대안의 글자 닮음(대본과)이 기준보다 ALT_MARGIN 넘게 높으면 대안.
      가려 적은 글(``masked``: 'X됐다', '새X')이 든 후보는 다른 후보가 있으면 고르지 않는다(기준이 가려 적었으면 대안).
      대안을 고르는 조건: 두 범위가 같은 소리 구간이고(``_same_sound``: 겹치거나 틈이 SAME_NEAR_S 초 이하), 대안 범위가 다른 줄이 가진 낱말과
        겹치지 않는다(``_Claims.clashes``: 한 번 난 소리는 한 줄에만).
      기준이 못 맞춘 줄을 대안만 맞췄으면 그 범위가 다른 줄이 가진 낱말과 겹치지 않을 때 받아들인다. 그 자리에 기준이
        적은 글(어느 줄에도 안 쓰인 낱말)은 후보로 남긴다.
    규칙 2. 대본 힌트 pass(ctx): 기준으로 고른 줄과 아무도 못 맞춘 줄에서만 견준다(decided_by acoustic)
      기준이 대본과 다르고 ctx 범위가 대본과 같으면: 두 범위가 같은 소리 구간이고 ctx 낱말의 평균 로그 확률이 기준보다
        tau 넘게 낮지 않고 다른 줄이 가진 낱말과 겹치지 않을 때만 ctx. 평균은 두 pass 가 서로 다르게 적은 낱말끼리만 낸다.
      둘 다 대본과 다르면 기준. ctx 가 대본에 더 가깝거나 확률이 CTX_BETTER 넘게 높고 위 조건이 맞으면 ctx.
      아무도 못 맞춘 줄을 ctx 만 맞췄으면(힌트 되풀이 위험): 글자 닮음 CTX_ONLY_SIM 이상, ctx 평균 로그 확률이
        ctx_floor 이상, 다른 줄이 가진 낱말과 겹치지 않고, 그 자리에 기준이 적은 낱말(어느 줄에도 안 쓰인 것)이 있으면
        그 낱말의 확률보다 tau 넘게 낮지 않을 때만 ctx.
      tau 는 기준·ctx 가 둘 다 대본과 같은 줄들의 (ctx − 기준 평균 로그 확률) 퍼짐에서, ctx_floor 는 기준이 대본과 같은
        줄들의 평균 로그 확률 분포에서 정한다(줄이 TAU_MIN_N 미만이면 기본값). 같은 줄들에서 잰 ctx − 기준 의 가운데값
        (ctx_bias: 두 pass 의 확률 눈금 차이)은 견주기 전에 ctx 쪽에서 뺀다.
    """
    if roles is None:
        roles = pass_roles([(pid, pw.context) for pid, pw in passes.items()])
    base_id = roles.base
    base = passes[base_id]
    m_base = matches.get(base_id, {})
    alt_ids = [pid for pid in roles.alts if pid in passes]
    ctx = passes.get(roles.ctx) if roles.ctx else None
    m_ctx = matches.get(roles.ctx, {}) if ctx is not None else {}

    def heard(pid: str, m: LineMatch) -> str:
        return passes[pid].text(m.w0, m.w1)

    def span(pid: str, m) -> tuple[float, float]:
        return passes[pid].start(m.w0, m.w1), passes[pid].end(m.w0, m.w1)

    kb = {li: textnorm.key(heard(base_id, m)) for li, m in m_base.items()}
    stats = {"base_same": 0, "base_only": 0, "base_kept": 0, "alt_same": 0, "alt_closer": 0, "alt_jamo": 0, "alt_unmasked": 0,
             "alt_elsewhere": 0, "alt_clash": 0, "alt_only": 0, "alt_only_rejected": 0}

    # 1. 기준과 대안
    claims = _Claims(passes)
    for li, m in m_base.items():
        claims.put(li, base_id, m.w0, m.w1)
    # pass 마다 어느 줄의 범위에도 들지 않은 낱말(다른 pass 의 범위로 넓힐 때 쓸 수 있는 낱말)
    free: dict[str, list[bool]] = {}
    for pid in (base_id, *alt_ids):
        f = [True] * passes[pid].n
        for m in matches.get(pid, {}).values():
            f[m.w0:m.w1 + 1] = [False] * (m.w1 - m.w0 + 1)
        free[pid] = f

    def widen(pid: str, a: int, b: int, q: str, x0: int, x1: int, line_text: str) -> tuple[int, int]:
        if not SPAN_UNION:
            return a, b
        a2, b2 = _extend_over(passes[pid], a, b, passes[q], x0, x1, free[pid], line_text)
        free[pid][a2:a] = [False] * (a - a2)
        free[pid][b + 1:b2 + 1] = [False] * (b2 - b)
        return a2, b2

    def core(pid: str, li: int) -> LineMatch:
        """붙이기 전에 그 pass 가 그 줄로 맞춘 범위(cores 가 없으면 붙인 범위)."""
        return (cores or {}).get(pid, {}).get(li) or matches[pid][li]

    def trusted(pid: str, q: str, li: int, key: str) -> bool:
        """pass q 의 범위를 pass pid 의 범위를 넓히는 근거로 쓸 만한가: q 도 그 줄을 pid 만큼(SPAN_TRUST 안으로) 맞췄다."""
        sim = {p: textnorm.similarity(key, textnorm.key(passes[p].text(core(p, li).w0, core(p, li).w1))) for p in (pid, q)}
        return sim[q] >= sim[pid] - SPAN_TRUST

    out: dict[int, Choice] = {}
    retry: list[tuple[int, tuple]] = []              # 다른 줄과 소리가 겹쳐 못 고른 대안: 다른 줄을 다 고른 뒤 다시 본다
    for ln in lines:
        li = ln.idx
        mb = m_base.get(li)
        if mb is None:
            continue
        alts = [(pid, matches[pid][li]) for pid in alt_ids if li in matches.get(pid, {})]
        near = [(pid, ma) for pid, ma in alts if _same_sound(span(base_id, mb), span(pid, ma))]
        stats["alt_elsewhere"] += len(alts) - len(near)      # 두 pass 가 그 줄을 서로 다른 자리에 맞췄다: 기준을 지킨다
        # 같은 소리 구간의 두 범위를 서로의 범위만큼 넓혀 같은 소리를 두고 견준다
        rb = (mb.w0, mb.w1)
        for pid, ma in near:
            if trusted(base_id, pid, li, ln.key):
                rb = widen(base_id, *rb, pid, ma.w0, ma.w1, ln.text)
        if rb != (mb.w0, mb.w1):
            claims.put(li, base_id, *rb)
            kb[li] = textnorm.key(base.text(*rb))
        ra = {pid: widen(pid, ma.w0, ma.w1, base_id, mb.w0, mb.w1, ln.text) if trusted(pid, base_id, li, ln.key)
              else (ma.w0, ma.w1) for pid, ma in near}
        cands = {base_id: rb}
        for pid, ma in alts:
            cands[pid] = ra.get(pid, (ma.w0, ma.w1))
        out[li] = Choice(base_id, *rb, mb.how, "rule", cands)
        if kb[li] == ln.key:
            stats["base_same"] += 1
            continue
        if not alts:
            stats["base_only"] += 1
            continue
        sb = textnorm.similarity(ln.key, kb[li])
        mask_b = masked(base.text(*rb))
        best = None                                  # (등급, 닮음, pass, (w0, w1), 통계 이름, how)
        for pid, ma in near:
            x0, x1 = ra[pid]
            text = passes[pid].text(x0, x1)
            ka = textnorm.key(text)
            mask_a = masked(text)
            if mask_a and not mask_b:
                continue
            # 대안을 고르면 기준이 그 줄로 맞춘 낱말(붙이기 전 범위) 가운데 대안 줄이 덮지 못하는 말은 added 로 떨어진다
            cb = core(base_id, li)
            left = ALT_CHARGE * _left_out(base, max(cb.w0, rb[0]), min(cb.w1, rb[1]), passes[pid], x0, x1, free[pid])
            sa = 2.0 * textnorm.lcs(ln.key, ka) / (len(ln.key) + len(ka) + left) if ln.key or ka else 1.0
            if ka == ln.key:
                rank = (3, sa, pid, (x0, x1), "alt_same", ma.how)
            elif mask_b and not mask_a:
                rank = (2, sa, pid, (x0, x1), "alt_unmasked", ma.how)
            elif sa > sb + ALT_MARGIN:
                rank = (1, sa, pid, (x0, x1), "alt_closer", ma.how)
            elif not left and abs(sa - sb) < 1e-9 and \
                    _jamo_sim(ln.key, ka) > _jamo_sim(ln.key, kb[li]) + ALT_MARGIN:
                # 글자 닮음이 똑같으면 소리(자모)가 뚜렷이 더 닮은 쪽: '성모병원' 에 '정보병원' 과 '전부병원'
                rank = (1, sa, pid, (x0, x1), "alt_jamo", ma.how)
            else:
                continue
            if best is None or rank[:2] > best[:2]:
                best = rank
        if best is None:
            stats["base_kept"] += 1
        elif claims.clashes(li, best[2], *best[3]):
            retry.append((li, best))
        else:
            _, _, pid, (x0, x1), name, how = best
            out[li] = Choice(pid, x0, x1, how, "rule", cands)
            claims.put(li, pid, x0, x1)
            stats[name] += 1
    # 소리가 겹친 다른 줄이 그 사이 다른 pass 의 범위로 바뀌었을 수 있다(줄 차례에 따라 결과가 갈리지 않게)
    for li, (_, _, pid, (x0, x1), name, how) in retry:
        if claims.clashes(li, pid, x0, x1):
            stats["alt_clash"] += 1
            continue
        out[li] = Choice(pid, x0, x1, how, "rule", out[li].cands)
        claims.put(li, pid, x0, x1)
        stats[name] += 1
    alt_only: list[int] = []
    for ln in lines:                                 # 기준이 못 맞춘 줄을 대안만 맞춘 경우
        li = ln.idx
        if li in out:
            continue
        for pid in alt_ids:
            ma = matches.get(pid, {}).get(li)
            if ma is None:
                continue
            if claims.clashes(li, pid, ma.w0, ma.w1):
                stats["alt_only_rejected"] += 1
                continue
            out[li] = Choice(pid, ma.w0, ma.w1, ma.how, "rule", {pid: (ma.w0, ma.w1)})
            claims.put(li, pid, ma.w0, ma.w1)
            alt_only.append(li)
            break
    # 대안만 맞춘 짧은 줄도 고른 이웃 줄이 가까이 있어야 남긴다(대안 pass 안에서 이웃이던 줄이 겹침으로 빠졌을 수 있다)
    kept = dict(out)
    _drop_lonely_short(lines, kept, lambda c: span(c.pass_id, c))
    for li in alt_only:
        if li not in kept:
            del out[li]
            claims.drop(li)
            stats["alt_only_rejected"] += 1
    alt_only = [li for li in alt_only if li in out]
    stats["alt_only"] = len(alt_only)

    # 2. 대본 힌트 pass
    stats.update({"tau": None, "tau_lines": 0, "ctx_bias": 0.0, "ctx_floor": None, "ctx_floor_lines": 0,
                  "ctx_same": 0, "ctx_closer": 0, "ctx_kept_base": 0, "ctx_clash": 0, "ctx_only": 0,
                  "ctx_only_rejected": 0, "ctx_only_outvoted": 0})
    if ctx is not None:
        ctx_id = roles.ctx
        kc = {li: textnorm.key(heard(ctx_id, m)) for li, m in m_ctx.items()}
        diffs, same_lp = [], []
        for li, m in m_base.items():
            if kb[li] != lines[li].key:
                continue
            lp = base.logp(m.w0, m.w1)
            if lp is None:
                continue
            same_lp.append(lp)
            if li in m_ctx and kc[li] == lines[li].key:
                lc = ctx.logp(m_ctx[li].w0, m_ctx[li].w1)
                if lc is not None:
                    diffs.append(lc - lp)
        bias = 0.0                                       # 같은 글에 대한 ctx − 기준 로그 확률의 치우침(가운데값)
        if len(diffs) >= TAU_MIN_N:
            bias = _quantile(diffs, 0.5)
            tau = min(TAU_HI, max(TAU_LO, TAU_K * _quantile([abs(d - bias) for d in diffs], 0.9)))
        else:
            tau = TAU_DEFAULT
        floor = _quantile(same_lp, CTX_FLOOR_Q) if len(same_lp) >= TAU_MIN_N else CTX_FLOOR_DEFAULT
        stats.update({"tau": round(tau, 4), "tau_lines": len(diffs), "ctx_bias": round(bias, 4),
                      "ctx_floor": round(floor, 4), "ctx_floor_lines": len(same_lp)})
        for ln in lines:
            li = ln.idx
            c, mc = out.get(li), m_ctx.get(li)
            if c is None or mc is None:
                continue
            c.cands[ctx_id] = (mc.w0, mc.w1)
            if c.pass_id != base_id or kb[li] == ln.key:
                continue                                 # 기준이 대본과 같거나 대안으로 고른 줄은 그대로 둔다
            mp = m_base[li]
            lp, lc = _differing_logp(base, (mp.w0, mp.w1), ctx, (mc.w0, mc.w1))
            if lc is not None:
                lc -= bias
            same_sound = _same_sound(span(base_id, mp), span(ctx_id, mc))
            backed = same_sound and lp is not None and lc is not None and lc >= lp - tau
            closer = kc[li] == ln.key or textnorm.similarity(ln.key, kc[li]) > textnorm.similarity(ln.key, kb[li])
            clearer = lp is not None and lc is not None and lc >= lp + CTX_BETTER
            c.decided_by = "acoustic"
            if not (backed and (closer or clearer)):
                stats["ctx_kept_base"] += 1
            elif claims.clashes(li, ctx_id, mc.w0, mc.w1):
                # ctx 범위가 이웃 줄의 소리까지 덮는다: 고르면 한 번 난 소리가 두 줄에 들어간다
                stats["ctx_clash"] += 1
            else:
                out[li] = Choice(ctx_id, mc.w0, mc.w1, mc.how, "acoustic", c.cands)
                claims.put(li, ctx_id, mc.w0, mc.w1)
                stats["ctx_same" if kc[li] == ln.key else "ctx_closer"] += 1

    owned = [False] * base.n
    for c in out.values():
        if c.pass_id == base_id:
            for w in range(c.w0, c.w1 + 1):
                owned[w] = True
    for li in alt_only:                                  # 그 자리에 기준 pass 가 적은 글을 후보로 남긴다
        c = out[li]
        r = _rivals(claims, base_id, owned, c.pass_id, c.w0, c.w1)
        if r is not None:
            c.cands = {base_id: r, **c.cands}
    if ctx is not None:                                  # 아무도 못 맞춘 줄을 ctx 만 맞춘 경우
        for ln in lines:
            li = ln.idx
            mc = m_ctx.get(li)
            if li in out or mc is None:
                continue
            lc = ctx.logp(mc.w0, mc.w1)
            ok = (mc.sim >= CTX_ONLY_SIM and lc is not None and lc - bias >= floor
                  and not claims.clashes(li, ctx_id, mc.w0, mc.w1))
            cands = {ctx_id: (mc.w0, mc.w1)}
            if ok:
                # 그 시간대에 기준 pass 가 어느 줄에도 안 맞는 낱말을 받아썼으면 그 낱말과 확률을 견준다(둘 다 맞춘 줄과 같은 기준)
                r = _rivals(claims, base_id, owned, ctx_id, mc.w0, mc.w1)
                if r is not None:
                    cands = {base_id: r, **cands}
                    lp = base.logp(*r)
                    if lp is not None and lc - bias < lp - tau:
                        ok = False
                        stats["ctx_only_outvoted"] += 1
            if ok:
                out[li] = Choice(ctx_id, mc.w0, mc.w1, mc.how, "acoustic", cands)
                claims.put(li, ctx_id, mc.w0, mc.w1)
                stats["ctx_only"] += 1
            else:
                stats["ctx_only_rejected"] += 1
    return out, stats


# ------------------------------------------------------------------ 남은 낱말
def _split_added(pw: PassWords, a: int, b: int) -> list[tuple[int, int]]:
    """added 묶음 [a, b] 가 너무 길면(ADDED_MAX_WORDS 낱말, ADDED_MAX_S 초) 가장 긴 쉼에서 나눈다(같으면 가운데에 가까운 곳).
    받아쓰기가 문장부호를 거의 안 찍고 낱말 사이 틈이 0 이면 여러 사람의 말이 한 묶음이 되기 때문이다."""
    if a >= b or (b - a + 1 <= ADDED_MAX_WORDS and pw.end(a, b) - pw.start(a, b) <= ADDED_MAX_S):
        return [(a, b)]
    mid = (a + b + 1) / 2
    k = max(range(a + 1, b + 1), key=lambda k: (round(pw.s[k] - pw.e[k - 1], 3), -abs(k - mid), -k))
    return _split_added(pw, a, k - 1) + _split_added(pw, k, b)


def leftover(pw: PassWords, lines: list[Line], ranges: dict[int, tuple[int, int]],
             blocked: list[tuple[float, float]] = (), skip: list[bool] | None = None
             ) -> tuple[dict[int, tuple[int, int]], list[tuple[int, int]]]:
    """어느 줄에도 속하지 않은 낱말을 옆 줄에 붙이거나 added 묶음으로 자른다.

    pw: 낱말을 가를 pass(기준 pass, 또는 rescue pass). lines: 말 있는 대사 줄 목록.
    ranges: {줄 번호: (처음 낱말, 끝 낱말)} (이 pass 에서 고른 줄만).
    blocked: 다른 pass 에서 고른 줄이 소리를 낸 시간 구간들(가운데 시각이 그 안인 낱말은 남은 낱말로 치지 않는다).
    skip: 낱말 번호마다 참이면 그 낱말도 남은 낱말로 치지 않는다(rescue pass: 기준 pass 가 이미 받아쓴 소리).
    돌려주는 것: (넓힌 ranges, added 묶음 [(처음 낱말, 끝 낱말)] 시간 순).

    규칙
      남은 낱말은 쉼(ADDED_GAP 초과)이나 문장 끝 부호(. ? !)에서 묶음으로 자른다.
      옆 맞춘 줄에 붙일 수 있는 것은 그 줄에서 틈 ATTACH_GAP 이하로(근거가 있으면 ADDED_GAP 이하로) 이어지고 사이에
      문장 끝 부호가 없는 낱말들이다.
      받아쓰기가 부호를 거의 안 찍고 낱말 사이 틈이 0 인 영화에서는 이 조건만으로 다른 사람의 말과 이웃 대본 줄의 말이
      한 줄에 다 붙으므로, 그 가운데 아래 둘 중 하나에 드는 만큼만 붙인다.
        근거 있는 붙이기: 줄 쪽 낱말부터, 붙는 글자의 ATTACH_GAIN 이상이 대본 글자와 새로 맞는 데까지(바꿔 말했거나
          맞추기가 놓친 그 줄의 말). 받아쓰기가 글자를 틀렸을 수 있으니 ATTACH_JAMO_MIN 글자 이상이고 아래 한도의
          앞부분(그 쪽에 안 맞은 대본 글자 수 + ATTACH_EXTRA) 안이면 자모로도 본다(붙는 자모의 ATTACH_GAIN_JAMO 이상이
          대본 자모와 새로 맞음). 짧은 대답은 같은 말로 본다('예' / '네'). 줄 쪽 이웃 낱말을 그대로 되풀이한 낱말
          ('야, 야, 야', '아이씨 아이씨')도 그 줄의 말로 본다.
        근거 없는 붙이기: 줄에 이어진 낱말 전부(틈이나 문장 끝 너머는 다른 말이다)의 글자 수가 한도 이하일 때 통째로.
          이어진 말이 한도보다 길면 그 앞머리만 떼어 붙이지 않는다. 한도는 '그 줄의 그 쪽(뒤에 붙으면
          줄 꼬리, 앞에 붙으면 줄 머리)에 아직 안 맞은 대본 글자 수 + ATTACH_EXTRA' 이고 줄 글자 수 × ATTACH_RATIO(적어도
          ATTACH_EXTRA)와 ATTACH_MAX 를 넘지 못한다(한두 음절 군말·덧말, 또는 그 줄이 바꿔 말한 부분). 붙인 뒤 그 줄의
          글자 닮음이 ATTACH_DROP 넘게 떨어지면 붙이지 않는다(글자가 다 맞은 한두 글자 줄에 두 글자가 붙는 경우).
          그 쪽 대본 글자가 다 맞았으면 짧은 대답·되묻기 한 낱말(REPLY_WORDS)은 상대의 말일 수 있어 붙이지 않는다.
          쉼표·말줄임표가 붙어 뒤 줄로 이어지는 꼴('네, 어디세요')만 뒤 줄의 말머리로 붙인다.
      양쪽 줄에 다 붙을 수 있으면 대본 글자가 더 많이 새로 맞는 쪽, 같으면 군말은 뒤 줄의 말머리, 그 밖은 더 붙어 있는 쪽.
      다른 pass 에서 고른 줄 옆의 낱말은 그 줄에 붙일 수 없으니(범위는 한 pass 안이다) added 로 둔다. 그런 줄 바로 앞의
      한두 음절 군말은 그 줄의 말머리일 수 있으니 근거 없이 앞 줄에 붙이지도 않는다.
      added 묶음은 ADDED_MAX_WORDS 낱말·ADDED_MAX_S 초를 넘으면 가장 긴 쉼에서 나눈다. 글자가 없는 묶음은 버린다.
    """
    out = dict(ranges)
    owner = [-1] * pw.n
    for li, (a, b) in ranges.items():
        for w in range(a, b + 1):
            owner[w] = li
    spans = sorted(blocked)
    b_starts = [s for s, _ in spans]
    can = [_ANSWER.get(c, c) for c in pw.loc]            # 짧은 대답은 같은 말로 보고 견준다('예' / '네')
    canon: dict[int, str] = {}

    def is_blocked(w: int) -> bool:
        if skip is not None and skip[w]:
            return True
        mid = (pw.s[w] + pw.e[w]) / 2
        k = bisect.bisect_right(b_starts, mid)
        return any(s <= mid <= e for s, e in spans[max(0, k - 4):k])

    def chained(ws: list[int], limit: float) -> int:
        """ws(줄에 가까운 낱말부터) 가운데 줄 쪽에서 틈 limit 이하로 이어지고 문장이 끝나지 않는 데까지의 낱말 수."""
        for k in range(1, len(ws)):
            early, late = sorted((ws[k - 1], ws[k]))
            if pw.s[late] - pw.e[early] > limit or textnorm.ends_sentence(pw.w[early]):
                return k
        return len(ws)

    def is_reply(ws: list[int], after: bool) -> bool:
        keys = [k for k in (textnorm.key(pw.w[w]) for w in ws) if k]
        if len(keys) != 1 or keys[0] not in REPLY_WORDS:
            return False
        return after or not pw.w[max(ws)].rstrip().endswith(_CONTINUES)

    def take(li: int, ws: list[int], after: bool, gap: float) -> tuple[int, int]:
        """줄 li 에 붙일 낱말 수와 그때 새로 맞는 대본 글자 수. ws 는 줄에 가까운 낱말부터. after: 줄 뒤에 붙는가.
        gap: 줄과 ws[0] 사이 틈(초)."""
        if li not in canon:
            canon[li] = _canon_line(lines[li])
        loc = canon[li]
        base = "".join(can[w] for w in range(out[li][0], out[li][1] + 1))
        n = len(loc)
        same = [op for op in LCSseq.opcodes(loc, base) if op.tag == "equal"]
        h0 = sum(op.src_end - op.src_start for op in same)
        # 그 쪽(줄 꼬리 또는 머리)에 아직 안 맞은 대본 글자 수: 그만큼은 바꿔 말한 부분일 수 있다
        rest = (n - same[-1].src_end if after else same[0].src_start) if same else n
        budget = min(rest + ATTACH_EXTRA, max(ATTACH_EXTRA, int(ATTACH_RATIO * n)), ATTACH_MAX)
        jl = textnorm.jamo_string(loc)
        hj0 = textnorm.lcs(jl, textnorm.jamo_string(base))
        reach = chained(ws, ATTACH_GAP) if gap <= ATTACH_GAP else 0      # 근거 없이 붙일 수 있는 데까지
        best = (0, 0)
        acc = near = ""
        gain = gj = near_gain = 0
        echo = True                                      # 지금까지의 낱말이 모두 줄 쪽 이웃 낱말의 되풀이인가
        for k in range(1, chained(ws, ADDED_GAP) + 1):   # 근거가 있으면 짧은 쉼(ADDED_GAP 이하) 너머의 낱말도 붙인다
            w = ws[k - 1]
            toward = ws[k - 2] if k > 1 else out[li][1 if after else 0]
            echo = echo and bool(can[w]) and can[w] == can[toward]
            acc = acc + can[w] if after else can[w] + acc
            whole = base + acc if after else acc + base
            was, gain = gain, textnorm.lcs(loc, whole) - h0
            # 붙는 마지막 낱말이 제 몫을 해야 한다(앞 낱말이 번 글자에 얹혀 대본에 없는 말이 따라 붙지 않게)
            if echo or (gain > was and gain >= ATTACH_GAIN * len(acc)):
                best = (k, gain)
            elif ATTACH_JAMO_MIN <= len(acc) <= rest + ATTACH_EXTRA:      # 받아쓰기가 글자를 틀려도 소리가 닮았으면 근거로 친다
                was_j, gj = gj, textnorm.lcs(jl, textnorm.jamo_string(whole)) - hj0
                if gj > was_j and gj >= ATTACH_GAIN_JAMO * len(textnorm.jamo_string(acc)):
                    best = (k, max(gain, 1))
            if k == reach:
                near, near_gain = acc, gain
        if best[0] < reach and near and len(near) <= budget and not (rest == 0 and is_reply(ws[:reach], after)):
            m0 = len(base)
            if 2.0 * (h0 + near_gain) / (n + m0 + len(near)) >= 2.0 * h0 / (n + m0) - ATTACH_DROP:
                best = (reach, near_gain)
        return best

    def is_short_filler(ws: list[int]) -> bool:
        keys = [k for k in (textnorm.key(pw.w[w]) for w in ws) if k]
        return bool(keys) and sum(map(len, keys)) <= textnorm.FILLER_SHORT and all(textnorm.is_filler(k) for k in keys)

    added: list[tuple[int, int]] = []
    w = 0
    while w < pw.n:
        if owner[w] != -1 or is_blocked(w):
            w += 1
            continue
        a = w
        while w < pw.n and owner[w] == -1 and not is_blocked(w):
            w += 1
        b = w - 1                                        # 남은 낱말 줄 [a, b]
        groups: list[list[int]] = [[a, a]]
        for k in range(a + 1, b + 1):
            if pw.s[k] - pw.e[k - 1] > ADDED_GAP or textnorm.ends_sentence(pw.w[k - 1]):
                groups.append([k, k])
            else:
                groups[-1][1] = k
        left = owner[a - 1] if a > 0 else -1
        right = owner[b + 1] if b + 1 < pw.n else -1
        g0, g1 = groups[0], groups[-1]
        gap_l = pw.s[a] - pw.e[a - 1] if left != -1 else None
        gap_r = pw.s[b + 1] - pw.e[b] if right != -1 else None
        ws_l = list(range(g0[0], g0[1] + 1))
        ws_r = list(range(g1[1], g1[0] - 1, -1))
        nl = gl = nr = gr = 0
        if gap_l is not None and gap_l <= ADDED_GAP and not textnorm.ends_sentence(pw.w[a - 1]):
            nl, gl = take(left, ws_l, True, gap_l)
        if gap_r is not None and gap_r <= ADDED_GAP and not textnorm.ends_sentence(pw.w[b]):
            nr, gr = take(right, ws_r, False, gap_r)
        if len(groups) == 1 and nl + nr > len(ws_l):     # 한 묶음을 양쪽 줄이 다 원한다
            if gl != gr:
                left_first = gl > gr
            elif is_short_filler(ws_l):
                left_first = False                       # 군말은 뒤 줄의 말머리
            else:
                left_first = gap_l < gap_r
            if left_first:
                nr, gr = take(right, ws_r[:len(ws_r) - nl], False, gap_r) if nl < len(ws_r) else (0, 0)
            else:
                nl, gl = take(left, ws_l[:len(ws_l) - nr], True, gap_l) if nr < len(ws_l) else (0, 0)
        if (nl and not gl and len(groups) == 1 and nl == len(ws_l) and b + 1 < pw.n and owner[b + 1] == -1
                and pw.s[b + 1] - pw.e[b] <= ADDED_GAP and is_short_filler(ws_l)):
            nl = 0                                       # 군말은 뒤 줄의 말머리인데 그 줄은 다른 pass 에서 골랐다: added 로 둔다
        if nl:
            out[left] = (out[left][0], g0[0] + nl - 1)
            g0[0] += nl
        if nr:
            out[right] = (g1[1] - nr + 1, out[right][1])
            g1[1] -= nr
        for ga, gb in groups:
            if ga > gb:
                continue
            for xa, xb in _split_added(pw, ga, gb):
                if any(pw.loc[k] for k in range(xa, xb + 1)):
                    added.append((xa, xb))
    return out, added


def _attached(lines: list[Line], pw: PassWords, got: dict[int, LineMatch]) -> dict[int, LineMatch]:
    """맞춘 줄마다 ``leftover`` 규칙으로 옆의 남은 낱말을 붙인 범위로 바꾼 사본."""
    ranges, _ = leftover(pw, lines, {li: (m.w0, m.w1) for li, m in got.items()})
    out = {}
    for li, m in got.items():
        a, b = ranges[li]
        if (a, b) == (m.w0, m.w1):
            out[li] = m
            continue
        span = pw.span_loc(a, b)
        hit = textnorm.lcs(lines[li].loc, span)
        out[li] = LineMatch(li, m.el, a, b, hit, m.n, len(span), round(2.0 * hit / (m.n + len(span)), 4), m.how)
    return out


# ------------------------------------------------------------------ conf
def _agreement(op: dict, free: set[str]) -> int:
    """대본 없이 받아쓴 다른 pass(free: 기준·대안 pass id)가 이 줄에 적은 글이 고른 글과 같으면 1, 다르면 -1,
    견줄 후보가 없으면 0. 글은 op 의 candidates 에서 읽는다(공백·문장부호 뺀 글자열로 견준다)."""
    cands = op.get("candidates") or []
    ch = op.get("chosen")
    if not isinstance(ch, int) or not 0 <= ch < len(cands):
        return 0
    mine = textnorm.key(cands[ch].get("text"))
    others = [textnorm.key(c.get("text")) for i, c in enumerate(cands)
              if i != ch and str(c.get("src") or "").startswith("asr:") and str(c["src"])[4:] in free]
    if not others:
        return 0
    return 1 if mine in others else -1


def _noise_factor(shape: str | None, mean_p: float | None) -> float:
    """받아쓰기 헛소리 꼴인 added 줄의 conf 에 곱할 값. 자막 제작 문구는 늘 CONF_NOISE. 인사말·되풀이 꼴은 진짜 대사일 수
    있으니 낱말 확률이 낮을 때만 깎는다(NOISE_P_LO 이하 CONF_NOISE, NOISE_P_HI 이상 1, 그 사이는 곧은 줄)."""
    if shape is None:
        return 1.0
    if shape == "credit" or mean_p is None:
        return CONF_NOISE
    t = min(1.0, max(0.0, (mean_p - NOISE_P_LO) / (NOISE_P_HI - NOISE_P_LO)))
    return CONF_NOISE + (1.0 - CONF_NOISE) * t


def _llm_noise(method: dict | None) -> set[tuple[str, int]]:
    """align.json method 의 llm_pair.noise(LLM 짝 판정이 헛소리로 본 낱말 범위) -> {(pass id, 낱말 번호)}."""
    from . import align_llm

    pair = (method or {}).get("llm_pair") if isinstance(method, dict) else None
    return align_llm.parse_refs(pair.get("noise")) if isinstance(pair, dict) else set()


def op_conf(script: dict, asr: dict, ops: list[dict], method: dict | None = None) -> list[float | None]:
    """ops(align.json 의 ops 꼴) 마다 conf(0~1, 소수 셋째 자리 또는 None). 계산식은 CONF_TEXT.

    final 단계가 align.json 을 읽고 같은 값을 다시 얻을 수 있게 ops·script·asr 와 align.json 의 method(LLM 짝 판정의
    헛소리 낱말 method.llm_pair.noise 만 읽는다. None 이면 없음)만으로 계산한다.
    """
    from rapidfuzz import fuzz

    noise = _llm_noise(method)
    index = timeline.asr_index(asr)
    roles = pass_roles(asr)
    free_ids = {roles.base, *roles.alts}
    els = {e["id"]: e for e in script.get("elements") or []}
    order = {e["id"]: k for k, e in enumerate(script.get("elements") or [])}
    conf: list[float | None] = [None] * len(ops)
    spans: dict[int, tuple[float, float]] = {}
    sims: dict[int, float] = {}
    loose: list[tuple[float, float, float, str]] = []    # 설명 안 된 말: (시작, 끝, 그 가운데 설명 안 된 시간, 설명 안 된 글)
    covered: list[tuple[float, float]] = []              # 낱말이 덮은 시간
    for k, op in enumerate(ops):
        if op.get("op") == "absent":
            continue
        ws = timeline.op_words(index, op)
        s, e = min(w["s"] for w in ws), max(w["e"] for w in ws)
        spans[k] = (s, e)
        pid = timeline.op_range(op)[0]
        if pid != roles.base:
            covered.extend((w["s"], w["e"]) for w in ws)
        ps = [float(w["p"]) for w in ws if w["p"] is not None]
        mean_p = sum(ps) / len(ps) if ps else None
        hk = textnorm.key(" ".join(w["w"] for w in ws))
        if op.get("op") == "added":
            c = mean_p if mean_p is not None else 0.5
            if pid == roles.rescue:
                c *= CONF_RESCUE
            f = _noise_factor(textnorm.noise_shape([w["w"] for w in ws]), mean_p)
            if noise:                                    # LLM 짝 판정이 헛소리로 본 낱말이 글자의 절반 이상
                _, i0, i1 = timeline.op_range(op)
                sizes = [(int(w["i"]), len(textnorm.key(str(w.get("w") or "")))) for w in index[pid][i0:i1 + 1]]
                total = sum(n for _, n in sizes)
                if total and 2 * sum(n for i, n in sizes if (pid, i) in noise) >= total:
                    f = min(f, CONF_NOISE)
            c *= f
            loose.append((s, e, e - s, hk))
        else:
            sk = textnorm.key(els[op["script_el"]].get("text"))
            sim = sims[k] = textnorm.similarity(sk, hk)
            agree = _agreement(op, free_ids)
            if mean_p is not None and agree > 0:
                mean_p = 1.0 - (1.0 - mean_p) * CONF_AGREE
            c = 0.5 * sim + 0.5 * mean_p if mean_p is not None else sim
            if agree < 0:
                c *= CONF_DISAGREE
            if op.get("decided_by") == "llm":
                c *= CONF_LLM
            hit = textnorm.lcs(sk, hk)
            if hk and hit < len(hk):                     # 그 줄의 범위에 든, 대본과 안 맞은 글자
                rest = "\n".join(hk[o.dest_start:o.dest_end] for o in LCSseq.opcodes(sk, hk) if o.tag != "equal")
                loose.append((s, e, (e - s) * (len(hk) - hit) / len(hk), rest))
        conf[k] = c

    script_ops = sorted((k for k, op in enumerate(ops) if op.get("script_el") is not None),
                        key=lambda k: order[ops[k]["script_el"]])
    said_ops = [k for k in script_ops if textnorm.sayable(els[ops[k]["script_el"]].get("text"))]
    # 대본 앞뒤의 말 있는 대사가 모두 absent 인데 약하게 맞은 줄(영상에 없는 대목 한가운데의 우연한 짝일 수 있다)
    for at, k in enumerate(said_ops):
        if k in spans and sims[k] < LONELY_SIM and \
                all(not 0 <= j < len(said_ops) or said_ops[j] not in spans for j in (at - 1, at + 1)):
            conf[k] *= CONF_LONELY
    for k in spans:
        conf[k] = round(min(1.0, max(0.0, conf[k])), 3)

    # absent: 그 줄이 있을 만한 시간 창에 설명 안 된 말(added 줄, 낱말이 덮지 못한 말 구간, 맞춘 줄 범위의 대본과 안 맞은
    # 글자 몫)이 얼마나 있는가, 그 줄의 글이 거기에 들어 있는가
    base_words = index.get(roles.base) or []
    covered = sorted(covered + [(float(w["s"]), float(w["e"])) for w in base_words])
    ci = 0
    for vs, ve in sorted((float(a), float(b)) for a, b in asr.get("vad") or []):
        at = vs
        while ci < len(covered) and covered[ci][1] <= vs:
            ci += 1
        cj = ci
        while cj < len(covered) and covered[cj][0] < ve:
            if covered[cj][0] - at >= FREE_MIN:
                loose.append((at, covered[cj][0], covered[cj][0] - at, ""))
            at = max(at, covered[cj][1])
            cj += 1
        if ve - at >= FREE_MIN:
            loose.append((at, ve, ve - at, ""))
    loose.sort()
    l_starts = [x[0] for x in loose]
    longest = max((x[1] - x[0] for x in loose), default=0.0)

    def within(t0: float, t1: float) -> list[tuple[float, str]]:
        """시간 창 [t0, t1] 에 걸친 설명 안 된 말마다 (그 창 안의 설명 안 된 시간, 설명 안 된 글)."""
        out = []
        j = bisect.bisect_left(l_starts, t1) - 1
        while j >= 0 and loose[j][0] >= t0 - longest:
            s, e, sec, text = loose[j]
            ov = min(e, t1) - max(s, t0)
            if ov > 0:
                out.append((sec * ov / (e - s), text))
            j -= 1
        return out

    timed = sorted(spans[k] for k in script_ops if k in spans)
    starts = [s for s, _ in timed]
    ends = sorted(e for _, e in timed)
    horizon = max([e for _, e in timed] + [x[1] for x in loose] + [0.0])
    run: list[int] = []
    prev_end: float | None = None

    def settle(run: list[int], t0: float | None, t1: float | None) -> None:
        said = [k for k in run if textnorm.sayable(els[ops[k]["script_el"]].get("text"))]
        if not said:
            return
        lo = t0 if t0 is not None else 0.0
        hi = t1 if t1 is not None else max(horizon, lo)
        if lo <= hi:
            wins = [(lo - ABSENT_PAD, hi + ABSENT_PAD)]
        else:                                            # 옮겨진 덩어리의 경계: 앞 줄 뒤쪽 틈과 뒤 줄 앞쪽 틈
            k = bisect.bisect_left(starts, lo)
            a1 = starts[k] if k < len(starts) else max(horizon, lo)
            k = bisect.bisect_right(ends, hi)
            b0 = ends[k - 1] if k else 0.0
            wins = sorted([(lo - ABSENT_PAD, a1 + ABSENT_PAD), (b0 - ABSENT_PAD, hi + ABSENT_PAD)])
            if wins[1][0] <= wins[0][1]:                 # 두 창이 닿으면 하나로(같은 말을 두 번 세지 않는다)
                wins = [(wins[0][0], max(wins[0][1], wins[1][1]))]
        rows = [r for a, b in wins for r in within(a, b)]
        got = sum(sec for sec, _ in rows)
        pool = "\n".join(text for _, text in rows if text)
        keys = {k: textnorm.key(els[ops[k]["script_el"]].get("text")) for k in said}
        need = SEC_PER_CHAR * sum(len(t) for t in keys.values())
        filled = min(1.0, got / need)
        for k in said:
            echo = 0.0
            if pool and len(keys[k]) >= ECHO_MIN_CHARS:
                echo = fuzz.partial_ratio(keys[k], pool) / 100.0
                if echo < ECHO_MIN:
                    echo = 0.0
            conf[k] = round(CONF_ABSENT_TOP - CONF_ABSENT_DROP * max(filled, echo), 3)

    for k in script_ops:
        if k in spans:
            settle(run, prev_end, spans[k][0])
            run = []
            prev_end = spans[k][1]
        else:
            run.append(k)
    settle(run, prev_end, None)
    return conf


# ------------------------------------------------------------------ 전체
@dataclass
class AlignResult:
    ops: list[dict]            # align.json 의 ops (item 포함, item 차례)
    scenes: list[dict]         # align.json 의 scenes
    method: dict               # align.json 의 method (anchor, acoustic, forced, translate + 추가 칸)
    items: list[dict]          # timeline.build_timeline 이 준 항목 뼈대(ops 의 item 과 같은 id)
    conf: dict                 # {item id: conf} (대사 항목만)
    match: float               # 일치율 0~1
    stats: dict                # 줄 수 집계: dialogue, spoken, silent_turns, same, changed, absent, added, mean_word_p
    llm_run: dict | None = None    # LLM 짝 판정의 실행 내력(캐시·호출·실패 수, 시간). 실행마다 달라 align.json 에 넣지 않는다


def _ref(pid: str, w0: int, w1: int) -> list[str]:
    return [f"{pid}:{w0}", f"{pid}:{w1}"]


def _echoed(base: PassWords, a: int, b: int, other: PassWords, x0: int, x1: int) -> list[int]:
    """기준 pass 의 낱말 [a, b] 가운데 다른 pass 의 낱말 [x0, x1] 이 같은 글로 적은 낱말 번호.

    한 줄을 두 pass 가 따로 맞췄고 다른 pass 의 범위를 골랐을 때 쓴다. 두 범위는 같은 줄의 소리인데 낱말 시각이 pass 마다
    어긋나서, 시간으로는 안 겹치는 낱말도 같은 소리일 수 있다('예' 와 '네', 줄 끝의 '해?' 와 '어떡해'). 두 글자열을 순서대로
    맞춰(LCS) 글자의 절반 이상이 맞은 기준 낱말을 그 줄의 소리로 본다. 짧은 대답은 같은 말로 본다(ANSWER_CLASSES).
    글이 안 맞는 낱말(기준 범위에만 든 덧말·이웃 말)은 여기서 빼지 않는다: 시간으로도 안 겹치면 남은 낱말이 된다."""
    words = [w for w in range(a, b + 1) if base.loc[w]]
    mine = [_ANSWER.get(base.loc[w], base.loc[w]) for w in words]
    theirs = "".join(_ANSWER.get(c, c) for c in other.loc[x0:x1 + 1])
    owner = [k for k, c in enumerate(mine) for _ in c]
    hit = Counter()
    for op in LCSseq.opcodes("".join(mine), theirs):
        if op.tag == "equal":
            hit.update(owner[op.src_start:op.src_end])
    return [w for k, w in enumerate(words) if 2 * hit[k] >= len(mine[k])]


def _rescue_lines(lines: list[Line], choice: dict[int, Choice], passes: dict[str, PassWords], roles: Roles,
                  claims: _Claims, duration: float) -> dict[int, LineMatch]:
    """아무 pass 도 못 맞춘 줄을 rescue 낱말에서 찾는다: 앞뒤 맞춘 줄 사이 시간 창 안의 rescue 낱말로 줄 단위 맞추기.

    rescue 낱말은 다른 pass 의 말 사이사이에 흩어져 있다. 그래서
      한 줄의 범위는 다른 pass 가 받아쓴 말(기준 pass 의 낱말, 이미 고른 줄)을 건너뛰어 이어지지 못한다.
      찾은 범위가 다른 줄이 가진 낱말과 겹치면 버린다(``_Claims.clashes``).
      같은 rescue 낱말을 두 줄이 쓰지 않는다: 못 맞춘 줄 묶음마다 창이 따로라서(옮겨진 덩어리의 경계에서는 창이 겹친다)
      같은 낱말을 두 줄이 원할 수 있다. 맞은 글자가 많은 줄이 갖고, 진 줄은 남은 낱말로 다시 찾는다.
    돌려주는 것: {줄 번호: LineMatch} (rescue pass 의 낱말 범위)."""
    rescue = passes[roles.rescue]
    base = passes[roles.base]

    def span_of(c: Choice) -> tuple[float, float]:
        return passes[c.pass_id].start(c.w0, c.w1), passes[c.pass_id].end(c.w0, c.w1)

    spans = {li: span_of(c) for li, c in choice.items()}
    wins = _windows(lines, spans, duration)
    # 이웃한 두 rescue 낱말 사이에 다른 pass 의 말이 있으면 그 사이에서 줄이 끊긴다
    marks = sorted([base.mid(w) for w in range(base.n) if base.loc[w]] + [(s + e) / 2 for s, e in spans.values()])
    cuts = {w for w in range(1, rescue.n)
            if bisect.bisect_left(marks, rescue.s[w]) > bisect.bisect_right(marks, rescue.e[w - 1])}
    mids = [rescue.mid(w) for w in range(rescue.n)]
    groups: list[list[int]] = []
    li = 0
    while li < len(lines):
        if li in choice:
            li += 1
            continue
        groups.append([])
        while li < len(lines) and li not in choice:
            groups[-1].append(li)
            li += 1
    taken = [False] * rescue.n
    found: dict[int, LineMatch] = {}
    todo = groups
    while todo:
        new: dict[int, LineMatch] = {}
        for g in todo:
            F = sorted({w for t0, t1 in wins[g[0]] for w in range(rescue.n) if t0 <= mids[w] <= t1 and not taken[w]})
            for k, m in match_window(lines, g, rescue, F, cuts).items():
                if not claims.clashes(k, roles.rescue, m.w0, m.w1):
                    new[k] = m
        lost: set[int] = set()
        for k in sorted(new, key=lambda k: (-new[k].hit, k)):
            m = new[k]
            if any(taken[m.w0:m.w1 + 1]):
                lost.add(k)
            else:
                taken[m.w0:m.w1 + 1] = [True] * (m.w1 - m.w0 + 1)
                found[k] = m
        # 낱말을 빼앗긴 줄이 든 묶음만, 남은 낱말로 다시(이미 찾은 줄은 그대로 둔다). 한 묶음의 줄은 창이 같다
        todo = [rest for rest in ([k for k in g if k not in found] for g in todo if lost.intersection(g)) if rest]
    got: dict[int, object] = dict(choice)
    got.update(found)
    _drop_lonely_short(lines, got, lambda x: span_of(x) if isinstance(x, Choice)
                       else (rescue.start(x.w0, x.w1), rescue.end(x.w0, x.w1)))
    return {k: m for k, m in found.items() if k in got}


def _repair_across(lines: list[Line], passes: dict[str, PassWords], roles: Roles, choice: dict[int, Choice],
                   claims: _Claims, skip_base: list[bool] | None, skip_rescue: list[bool] | None) -> int:
    """pass 를 다 고른 뒤의 다시 짝짓기(``_repair``). 고른 줄(어느 pass 든)의 낱말과 기준·rescue pass 의 남은 낱말을 시각
    차례로 섞은 가상 pass 를 만들어, 고른 줄 사이 창에서 못 맞춘 줄과 남은 낱말을 짝짓는다.

    pass 마다의 다시 짝짓기는 그 pass 의 낱말만 보고 그 pass 에서 맞춘 줄로 창을 나눈다. 그래서 rescue 낱말('네' 하나)은
    다시 짝짓기를 거치지 않고, 이웃 줄이 다른 pass 에서 골라진 곳은 창이 실제와 다르며, 한 창의 말이 두 pass 에 나뉘어
    있으면 그 창의 '남은 말'이 모자라 보여(잘린 대목으로 보여) 짧은 대답을 받지 않는다. 기준은 ``_repair`` 그대로다.
    한 줄의 범위는 한 pass 안에서 이어진 낱말이다(pass 가 바뀌는 곳에서 끊는다). 다른 pass 에서 고른 줄과 같은 소리를
    갖게 되는 짝은 버린다. choice·claims 를 제자리에서 고치고, 새로 맞춘 줄 수를 돌려준다."""
    keep: dict[str, set[int]] = {pid: set() for pid in passes}
    for c in choice.values():
        keep[c.pass_id].update(range(c.w0, c.w1 + 1))
    base = passes[roles.base]
    keep[roles.base].update(w for w in range(base.n) if skip_base is None or not skip_base[w])
    if roles.rescue is not None and skip_rescue is not None:
        keep[roles.rescue].update(w for w in range(passes[roles.rescue].n) if not skip_rescue[w])
    rank = {pid: k for k, pid in enumerate(passes)}
    rows = []
    for pid, ws in keep.items():
        pw, t = passes[pid], -math.inf
        for w in sorted(ws):
            t = max(t, pw.mid(w))                         # pass 안의 차례는 지킨다
            rows.append((t, rank[pid], w, pid))
    rows.sort()
    seq = [(pid, w) for _, _, w, pid in rows]
    if not seq:
        return 0
    pos = {x: k for k, x in enumerate(seq)}
    other = "".join(ln.key for ln in lines)
    vp = PassWords("across", [{"i": k, "w": passes[pid].w[w], "s": passes[pid].s[w], "e": passes[pid].e[w],
                               "p": passes[pid].p[w]} for k, (pid, w) in enumerate(seq)], other)
    cuts = frozenset(k for k in range(1, len(seq)) if seq[k][0] != seq[k - 1][0] or seq[k][1] != seq[k - 1][1] + 1)
    got: dict[int, LineMatch] = {}
    for li, c in choice.items():
        ks = [pos[(c.pass_id, w)] for w in range(c.w0, c.w1 + 1)]
        a, b = min(ks), max(ks)
        span = vp.span_loc(a, b)
        hit = textnorm.lcs(lines[li].loc, span)
        n, m = len(lines[li].loc), len(span)
        got[li] = LineMatch(li, lines[li].el, a, b, hit, n, m, round(2.0 * hit / (n + m), 4) if n + m else 1.0, c.how)
    before = {li: (m.w0, m.w1) for li, m in got.items()}
    _repair(lines, vp, got, cuts, edges=True)
    taken: set[int] = set()
    n_new = 0
    for li in sorted(set(got) - set(before)):
        m = got[li]
        pid = seq[m.w0][0]
        ws = [seq[k][1] for k in range(m.w0, m.w1 + 1)]
        if any(seq[k][0] != pid for k in range(m.w0, m.w1 + 1)) or ws != list(range(ws[0], ws[-1] + 1)):
            continue                                     # (cuts 가 막으므로 생기지 않는다)
        if claims.clashes(li, pid, ws[0], ws[-1]):
            continue
        choice[li] = Choice(pid, ws[0], ws[-1], "slot", "rule", {pid: (ws[0], ws[-1])})
        claims.put(li, pid, ws[0], ws[-1])
        taken.update(range(m.w0, m.w1 + 1))
        n_new += 1
    for li, (a, b) in before.items():                    # 이웃 줄이 가장자리 낱말을 내준 경우: 받은 줄이 있을 때만 줄인다
        m = got[li]
        if (m.w0, m.w1) == (a, b) or not (set(range(a, b + 1)) - set(range(m.w0, m.w1 + 1))) & taken:
            continue
        c = choice[li]
        ws = [seq[k][1] for k in range(m.w0, m.w1 + 1) if seq[k][0] == c.pass_id]
        if not ws:
            continue
        c.w0, c.w1 = ws[0], ws[-1]
        c.cands[c.pass_id] = (c.w0, c.w1)
        claims.put(li, c.pass_id, c.w0, c.w1)
    return n_new


def _habit_key(text: str) -> str:
    """버릇 글귀와 견줄 글: NFC, 대소문자 무시, 글자·숫자만(asr.film_habits 의 글과 같은 꼴)."""
    return "".join(c for c in util.nfc(util.nfc(text or "").casefold()) if c.isalnum())


def _credit_runs(pw: PassWords, x: int, y: int) -> list[tuple[int, int]]:
    """말 덩이 [x, y] 안의 자막 제작·구독 문구 범위(이어진 낱말): 정규식 ``textnorm.NOISE_CREDIT`` 이 걸린 글자를 덮는 낱말에,
    바로 뒤에 이어진 짝말(CREDIT_TAIL 의 낱말, 'by' 와 그 뒤 한 낱말: '한글자막 by 한효정', '자막 제공', '시청해주셔서 감사합니다')
    까지. 그 밖의 낱말은 넣지 않는다(문구 뒤에 이어 받아써진 대사를 잃지 않게)."""
    text, spans = "", []
    for k in range(x, y + 1):
        if text:
            text += " "
        spans.append((len(text), len(text) + len(pw.w[k]), k))
        text += pw.w[k]
    runs: list[tuple[int, int]] = []
    for m in textnorm.NOISE_CREDIT.finditer(text):
        ks = [k for s, e, k in spans if s < m.end() and e > m.start()]
        if not ks:
            continue
        i, j = ks[0], ks[-1]
        name = _habit_key(pw.w[j]).endswith("by")       # 'by' 뒤 한 낱말은 만든 이 이름
        while j < y:
            nk = _habit_key(pw.w[j + 1])
            if name and nk:
                j, name = j + 1, False
            elif nk == "by":
                j, name = j + 1, True
            elif nk in CREDIT_TAIL:
                j += 1
            else:
                break
        if runs and i <= runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], max(runs[-1][1], j))
        else:
            runs.append((i, j))
    return runs


def drop_noise(passes: dict[str, PassWords], added_ops: list[tuple[str, int, int]], habits: dict | None
               ) -> tuple[list[tuple[str, int, int]], list[dict]]:
    """added 로 남을 묶음 가운데 받아쓰기 헛소리가 확실한 것을 뺀다: (남길 묶음, 뺀 기록 [{"asr": [처음, 끝], "why"}]).

    묶음을 먼저 말 덩이로 나눈다: 앞 낱말이 문장 끝 부호로 끝나거나 낱말이 말줄임표·줄표로 시작하는 곳('한글 자막 제공 ..욕봤다'
    에서 '..욕봤다' 는 새 말이다).
    credit: 말 덩이 안의 자막 제작·구독 문구(``_credit_runs``: ``textnorm.NOISE_CREDIT`` 이 걸린 낱말과 바로 뒤의 짝말)만.
      문구 뒤에 부호 없이 이어 받아써진 대사('한글 자막 제공 눋머니 …')는 남긴다.
    habit: 말 덩이 안의 이어진 낱말 범위가 그 영화의 버릇 글귀(``asr.film_habits``: 여러 창에서 똑같이 나온 헛소리 글귀)와
      글자 닮음 HABIT_SIM 이상이면 그 범위만(가장 닮은 범위, 같으면 긴 범위).
    남은 낱말은 원래 묶음 안에서 이어진 데까지 다시 묶어(글자가 있는 것만) added 로 둔다. habits 가 없으면(받아쓰기 캐시가
    없는 영화) credit 만 본다. 남길 묶음의 차례는 그대로다."""
    from rapidfuzz import fuzz

    hs = sorted(h for h in (habits or {}) if h)
    kept: list[tuple[str, int, int]] = []
    dropped: list[dict] = []
    for pid, a, b in added_ops:
        pw = passes[pid]
        gone: set[int] = set()
        cuts = [k for k in range(a + 1, b + 1)
                if textnorm.ends_sentence(pw.w[k - 1]) or pw.w[k].lstrip().startswith(("..", "…", "-", "—"))]
        for x, y in zip([a] + cuts, [k - 1 for k in cuts] + [b]):
            for i, j in _credit_runs(pw, x, y):
                dropped.append({"asr": _ref(pid, i, j), "why": "credit"})
                gone.update(range(i, j + 1))
            todo: list[tuple[int, int]] = []
            for k in range(x, y + 1):                         # 문구를 뺀 나머지 이어진 범위에서 버릇 글귀를 찾는다
                if k in gone:
                    continue
                if todo and todo[-1][1] == k - 1:
                    todo[-1] = (todo[-1][0], k)
                else:
                    todo.append((k, k))
            while todo:
                x0, y0 = todo.pop(0)
                best = None
                for h in hs:
                    for i in range(x0, y0 + 1):
                        acc = ""
                        for j in range(i, y0 + 1):
                            acc += _habit_key(pw.w[j])
                            if not acc:
                                continue
                            r = fuzz.ratio(h, acc) / 100.0
                            if r >= HABIT_SIM and (best is None or (r, j - i) > (best[0], best[2] - best[1])):
                                best = (r, i, j)
                if best is None:
                    continue
                _, i, j = best
                dropped.append({"asr": _ref(pid, i, j), "why": "habit"})
                gone.update(range(i, j + 1))
                todo = [r for r in ((x0, i - 1), (j + 1, y0)) if r[0] <= r[1]] + todo
        run: list[int] = []
        for k in list(range(a, b + 1)) + [None]:
            if k is not None and k not in gone:
                run.append(k)
                continue
            if run and any(pw.loc[i] for i in run):
                kept.append((pid, run[0], run[-1]))
            run = []
    dropped.sort(key=lambda d: (d["asr"][0].rsplit(":", 1)[0], int(d["asr"][0].rsplit(":", 1)[1])))
    return kept, dropped


def build_ops(script: dict, asr: dict, video_duration: float | None = None, *, habits: dict | None = None,
              llm_pair: bool = False, llm_calls: bool = False, workers: int | None = None) -> AlignResult:
    """대본(script.json)과 받아쓰기(asr.json)를 맞춰 연결·판정 기록을 만든다.

    video_duration: 영상 길이(초). 대사 아닌 항목의 추정 시각 끝점에 쓴다. None 이면 받아쓰기의 마지막 시각.
    asr.passes 는 기준 pass 하나만 있어도 된다. pass 의 구실은 context 로 가른다(``pass_roles``): 대본 없이 받아쓴 pass 가
    둘 이상이면 줄마다 기준과 대안 가운데 고르고(``decide``), 대본 힌트 pass(ctx)와 rescue pass 가 있으면 그것도 쓴다.
    added 줄은 기준 pass 의 남은 낱말과, 기준 pass 가 받아쓰지 않은 곳의 rescue 낱말로 만든다(자막 제작 문구 꼴과
    habits 의 버릇 글귀는 빼고 method.noise_dropped 에 남긴다: ``drop_noise``).
    llm_pair: LLM 짝 판정(``sinkprep.align_llm``)을 적용한다. llm_calls 가 거짓이면 캐시(cache/llm)에 있는 판정만 쓰고(부르지
    않음), 참이면 캐시에 없는 요청을 claude 로 묻는다(workers: 동시 호출 수). 거짓이면 method.llm_pair 는 None.
    돌려주는 것: AlignResult. ``ops`` 는 item id 까지 붙은 align.json 의 ops 이고, 같은 입력(과 같은 LLM 캐시)이면 늘 같다.
    호출 수·시간은 ``AlignResult.llm_run`` 에만 둔다(align.json 에 들어가지 않는다).
    """
    lines, silent, passes = _load(script, asr)       # 숫자·영문 약어는 서로 맞은편 글에 있는 읽는 꼴로(자리 찾기용 글자열)
    if not passes:
        raise ValueError("asr.json 에 pass 가 없음")
    roles = pass_roles(asr)
    base_id = roles.base
    base = passes[base_id]
    # pass 마다 따로 맞추고, 남은 낱말을 같은 규칙으로 옆 줄에 붙인 뒤에 견준다(범위가 같은 기준이어야 공정하다)
    found = {pid: {m.line: m for m in match_pass(lines, passes[pid])}
             for pid in (base_id, *roles.alts, roles.ctx) if pid is not None}
    matches = {pid: _attached(lines, passes[pid], got) for pid, got in found.items()}
    choice, dstats = decide(lines, passes, matches, roles, found)

    def span_of(c: Choice) -> tuple[float, float]:
        pw = passes[c.pass_id]
        return pw.start(c.w0, c.w1), pw.end(c.w0, c.w1)

    if video_duration is None:
        video_duration = max([max(pw.e) for pw in passes.values() if pw.n] +
                             [float(c["end"]) for c in asr.get("chunks") or []] + [0.0])
    claims = _Claims(passes)
    for li, c in choice.items():
        claims.put(li, c.pass_id, c.w0, c.w1)

    # rescue: 아무 pass 도 못 맞춘 줄만
    rescue = passes.get(roles.rescue) if roles.rescue else None
    n_rescue = 0
    if rescue is not None and rescue.n:
        for li, m in _rescue_lines(lines, choice, passes, roles, claims, float(video_duration)).items():
            choice[li] = Choice(roles.rescue, m.w0, m.w1, m.how, "rule", {roles.rescue: (m.w0, m.w1)})
            claims.put(li, roles.rescue, m.w0, m.w1)
            n_rescue += 1

    # 남은 낱말: 기준 pass 에서 고른 줄에 붙이거나 added 로. 다른 pass 에서 고른 줄의 소리인 기준 낱말은 뺀다(같은 소리를
    # 두 번 담지 않는다): 그 줄의 시간대에 있는 낱말(``_Claims.covered``)과, 기준 pass 가 그 줄로 맞춘 범위에서 고른 줄의
    # 낱말과 글이 같은 낱말(``_echoed``: 두 pass 의 낱말 시각이 어긋나 시간으로는 안 겹쳐도 같은 소리다)
    def base_skip() -> list[bool] | None:
        owned = [False] * base.n
        for c in choice.values():
            if c.pass_id == base_id:
                owned[c.w0:c.w1 + 1] = [True] * (c.w1 - c.w0 + 1)
        if all(c.pass_id == base_id for c in choice.values()):
            return None
        skip = [not owned[w] and claims.covered(base_id, w) for w in range(base.n)]
        for c in choice.values():
            r = c.cands.get(base_id)
            if c.pass_id != base_id and r is not None:
                for w in _echoed(base, r[0], r[1], passes[c.pass_id], c.w0, c.w1):
                    skip[w] = not owned[w]               # 이웃 줄이 가져간 낱말은 그 줄의 것
        return skip

    # 줄에 쓰이지 않은 rescue 낱말은 기준 pass 가 받아쓰지 않은 곳(기준 pass 의 낱말·묶음과 겹치지 않음)에 있고 어느 줄의
    # 시간 범위와도 겹치지 않을 때만 쓴다. 바로 뒤(RESCUE_ECHO_S 안)에 같은 소리로 시작하는 낱말이 있으면 그 소리를 시작만
    # 늘려 한 번 더 적은 것이다('첫' 뒤의 '첫번째', '너' 뒤의 '너 미진이 딸이냐?'): 쓰지 않는다.
    if rescue is not None and rescue.n:
        chunks = sorted((float(c["start"]), float(c["end"])) for c in asr.get("chunks") or [])
        c_starts = [s for s, _ in chunks]
        b_order = sorted(range(base.n), key=lambda x: (base.s[x], x))
        b_starts = [base.s[x] for x in b_order]
        longest = max((base.e[x] - base.s[x] for x in range(base.n)), default=0.0)

    def rescue_skip() -> list[bool]:
        heads = sorted((pw.s[w], textnorm.jamo_string(textnorm.key(pw.w[w])))
                       for pid, pw in passes.items() if pid != roles.rescue
                       for w in (range(pw.n) if pid == base_id else
                                 [x for c in choice.values() if c.pass_id == pid for x in range(c.w0, c.w1 + 1)]))
        h_starts = [s for s, _ in heads]
        skip = []
        for w in range(rescue.n):
            s, e, m = rescue.s[w], rescue.e[w], rescue.mid(w)
            k = bisect.bisect_right(c_starts, m) - 1
            in_chunk = k >= 0 and m <= chunks[k][1]
            # 기준 pass 가 그 소리를 이미 받아썼는가(rescue 낱말은 시작 시각이 앞의 침묵 쪽으로 늘어나 있곤 하다)
            near = b_order[bisect.bisect_left(b_starts, s - longest):bisect.bisect_right(b_starts, e)]
            heard = any(_same_word(s, e, base.s[x], base.e[x]) for x in near)
            j = textnorm.jamo_string(textnorm.key(rescue.w[w]))
            echo = bool(j) and any(
                textnorm.similarity(j, hj[:len(j)]) >= RESCUE_ECHO_SIM
                for _, hj in heads[bisect.bisect_left(h_starts, s):bisect.bisect_right(h_starts, e + RESCUE_ECHO_S)])
            skip.append(in_chunk or heard or echo or claims.covered(roles.rescue, w))
        return skip

    # 다시 짝짓기(여러 pass): 고른 줄들 사이 창에서 못 맞춘 줄과 기준·rescue 의 남은 낱말을 짝짓는다
    n_across = 0
    if ACROSS_REPAIR:
        n_across = _repair_across(lines, passes, roles, choice, claims, base_skip(),
                                  rescue_skip() if rescue is not None and rescue.n else None)

    ranges = {li: (c.w0, c.w1) for li, c in choice.items() if c.pass_id == base_id}
    ranges, added = leftover(base, lines, ranges, skip=base_skip())
    for li, (a, b) in ranges.items():
        c = choice[li]
        c.w0, c.w1 = a, b
        c.cands[base_id] = (a, b)
        claims.put(li, base_id, a, b)
    added_ops = [(base_id, a, b) for a, b in added]

    # 줄에 쓰이지 않은 rescue 낱말을 같은 규칙으로 rescue 줄에 붙이거나 added 로 자른다
    if rescue is not None and rescue.n:
        r_ranges = {li: (c.w0, c.w1) for li, c in choice.items() if c.pass_id == roles.rescue}
        r_ranges, r_added = leftover(rescue, lines, r_ranges, skip=rescue_skip())
        for li, (a, b) in r_ranges.items():
            c = choice[li]
            c.w0, c.w1 = a, b
            c.cands[roles.rescue] = (a, b)
            claims.put(li, roles.rescue, a, b)
        added_ops += [(roles.rescue, a, b) for a, b in r_added]

    # 받아쓰기 헛소리가 확실한 묶음(자막 제작 문구 꼴, 그 영화의 버릇 글귀)은 added 로 만들지 않는다
    added_ops, noise_dropped = drop_noise(passes, added_ops, habits)

    # LLM 짝 판정: 못 맞춘 줄과 그 시간대의 added 낱말, 맞춘 줄의 떨어진 머리·꼬리(구조 제약 안에서만 받아들인다)
    pair = None
    llm_run = None
    if llm_pair:
        from . import align_llm

        added_ops, pair = align_llm.run_stage(
            script, lines, passes, choice, claims, added_ops,
            lambda pid, w0, w1: Choice(pid, w0, w1, "llm", "llm", {pid: (w0, w1)}),
            calls=llm_calls, workers=workers)
        llm_run = pair.pop("run")

    # ops
    by_el = {ln.el: ln for ln in lines}
    ops: list[dict] = []
    how_of: dict[str, str] = {}
    hit_chars = 0
    for e in script.get("elements") or []:
        if e.get("type") != "dialogue":
            continue
        ln = by_el.get(e["id"])
        c = choice.get(ln.idx) if ln is not None else None
        if c is None:
            ops.append({"item": None, "op": "absent", "script_el": e["id"], "asr": None, "match": "text",
                        "decided_by": "rule"})
            continue
        pw = passes[c.pass_id]
        heard = pw.text(c.w0, c.w1)
        hk = textnorm.key(heard)
        status = "same" if hk == ln.key else "changed"
        hit_chars += textnorm.lcs(ln.key, hk)
        cands, chosen = [], None
        for pid, (a, b) in c.cands.items():
            lp = passes[pid].logp(a, b)
            if pid == c.pass_id:
                chosen = len(cands)
            cands.append({"text": passes[pid].text(a, b), "src": f"asr:{pid}",
                          "score": None if lp is None else round(lp, 4)})
        cands.append({"text": ln.text, "src": "script", "score": None})
        ops.append({"item": None, "op": status, "script_el": e["id"], "asr": _ref(c.pass_id, c.w0, c.w1),
                    "match": "text", "text_score": round(textnorm.similarity(ln.key, hk), 3),
                    "candidates": cands, "chosen": chosen, "decided_by": c.decided_by})
        how_of[e["id"]] = c.how
    for pid, a, b in added_ops:
        ops.append({"item": None, "op": "added", "script_el": None, "asr": _ref(pid, a, b), "match": None,
                    "decided_by": "rule"})
    shared = timeline.shared_words(ops)
    if shared:                                           # 한 번 난 소리는 한 줄에만: 같은 낱말을 두 줄이 가지면 구현 오류다
        raise AssertionError(f"같은 받아쓰기 낱말을 두 줄이 가짐 {len(shared)}곳: {shared[0]}")

    # 항목 차례와 id
    items = timeline.build_timeline(script, ops, asr, video_duration)
    timeline.attach_items(ops, items)
    confs = op_conf(script, asr, ops, {"llm_pair": pair})
    conf = {op["item"]: c for op, c in zip(ops, confs)}
    ops.sort(key=lambda op: int(op["item"][1:]))         # item 번호 차례(글자 차례로 하면 I10000 이 I9999 앞에 온다)
    at_op = {op["item"]: k for k, op in enumerate(ops)}
    for it in items:                                     # 항목의 op 번호를 item 차례로 다시(align.json 의 ops 차례와 같게)
        if it["op"] is not None:
            it["op"] = at_op[it["id"]]

    # 씬 구간: 그 씬의 맞춘 대사 범위. 옮겨진 줄이 절반을 넘으면 moved, 닻으로 맞춘 줄이 있으면 anchors, 아니면 lines.
    by_scene: dict[str, list[dict]] = {}
    for it in items:
        if it["type"] == "dialogue" and it["src_el"] is not None and it["start"] is not None and it["scene"]:
            by_scene.setdefault(it["scene"], []).append(it)
    scenes = []
    for sc in script.get("scenes") or []:
        its = by_scene.get(sc["id"])
        if not its:
            scenes.append({"scene": sc["id"], "start": None, "end": None, "how": "none"})
            continue
        moved = [it for it in its if how_of.get(it["src_el"]) == "moved" or it["moved"]]
        if 2 * len(moved) > len(its):
            use, how = moved, "moved"
        else:
            use = [it for it in its if it not in moved] or its
            how = "anchors" if any(how_of.get(it["src_el"]) == "anchors" for it in use) else "lines"
        scenes.append({"scene": sc["id"], "start": min(it["start"] for it in use),
                       "end": max(it["end"] for it in use), "how": how})

    total = sum(len(ln.key) for ln in lines)
    match = round(hit_chars / total, 3) if total else 0.0
    cnt = Counter(op["op"] for op in ops)
    ps = [float(w["p"]) for it in items if it["type"] == "dialogue" for w in it["words"] if w["p"] is not None]
    stats = {"dialogue": len(lines) + len(silent) + cnt["added"], "spoken": len(lines), "silent_turns": len(silent),
             "same": cnt["same"], "changed": cnt["changed"], "absent": cnt["absent"], "added": cnt["added"],
             "absent_spoken": cnt["absent"] - len(silent), "match": match,
             "mean_word_p": round(sum(ps) / len(ps), 3) if ps else None}
    used = Counter(c.pass_id for c in choice.values())
    n_added = Counter(pid for pid, _, _ in added_ops)
    pass_stats = {pid: {"role": roles.of(pid), "words": pw.n,
                        "matched_lines": len(matches[pid]) if pid in matches else n_rescue if pid == roles.rescue else 0,
                        "chosen_lines": used[pid], "added_lines": n_added[pid], "backsteps": pw.backsteps}
                  for pid, pw in passes.items()}
    fixes = [it for it in items if it["type"] == "dialogue" and (it.get("time_fix") or {}).get("fixed")]
    method = {
        "anchor": "unique k-gram blocks (k=5,4,3) + weighted LIS + moved chains + line DP between anchors (LCS)",
        "acoustic": "mean word log-probability, base pass vs script-context pass (scored on the words that differ); "
                    "tau from this film" if roles.ctx else None,
        "forced": "none: word times are Whisper cross-attention DTW times of the transcribed words; "
                  "no separate forced alignment is run",
        "translate": None,
        "aligner": f"sinkprep.align {VERSION}",
        "base_pass": base_id,
        "roles": {"base": roles.base, "alts": list(roles.alts), "ctx": roles.ctx, "rescue": roles.rescue,
                  "unused": list(roles.unused)},
        "choose": CHOOSE_TEXT,
        "constants": {k: (list(v) if isinstance(v, tuple) else v) for k, v in sorted(CONSTANTS.items())},
        "decide": dict(dstats, across_slot=n_across),
        "conf": CONF_TEXT,
        "passes": pass_stats,
        "time_fix": {"lines": len(fixes), "words": sum(it["time_fix"]["fixed"] for it in fixes),
                     "suspect": [it["id"] for it in fixes if timeline.suspect(it["time_fix"])]},
        "noise_dropped": noise_dropped,
        "llm_pair": pair,
        "stats": stats,
    }
    return AlignResult(ops, scenes, method, items, conf, match, stats, llm_run)


# ------------------------------------------------------------------ 실행
def film_problems(doc_id: str, film_id: str, *, spoken: int, absent_spoken: int, match: float,
                  conf_rows: list[tuple[float | None, str, str]], path: str) -> list[dict]:
    """영화·항목 수준 문제(align 과 final 이 같은 기준·같은 글로 적는다). detail 에는 수치만 적는다(대사 글을 넣지 않는다).

    spoken: 말 있는 대본 대사 수, absent_spoken: 그 가운데 absent 수, match: 일치율,
    conf_rows: same·changed·added 줄마다 (conf, 항목 id, 상태), path: 문제를 볼 파일(ROOT 기준 상대 경로).
    """
    out = []
    base = {"doc_id": doc_id, "film_id": film_id, "path": path}
    if spoken and match < MATCH_LOW:
        out.append(dict(base, level="film", item_id="", problem="version_mismatch", score=match,
                        detail=f"일치율 {match:.3f} < {MATCH_LOW} (말 있는 대사 {spoken}줄)"))
    if spoken and absent_spoken / spoken > ABSENT_MANY:
        r = absent_spoken / spoken
        out.append(dict(base, level="film", item_id="", problem="absent_many", score=round(r, 3),
                        detail=f"말 있는 대사 {spoken}줄 중 absent {absent_spoken}줄({r:.3f} > {ABSENT_MANY})"))
    low = sorted((c, item, kind) for c, item, kind in conf_rows if c is not None and c < ITEM_LOW_CONF)
    for c, item, kind in low[:ITEM_PROBLEMS_MAX]:
        out.append(dict(base, level="item", item_id=item, problem="asr_low_conf", score=c,
                        detail=f"{kind} 줄 conf {c:.3f} < {ITEM_LOW_CONF}"))
    return out


def _problems(doc_id: str, film_id: str, res: AlignResult) -> list[dict]:
    """영화·항목 수준 문제. 낱말 시각이 거꾸로 가거나 영상 길이 밖이던 줄(timeline 이 시각을 바로잡는다) 가운데 바로잡아도
    믿기 어려운 줄은 항목 수준 other 로 남긴다(final 단계가 같은 줄에 같은 글로 적는다: review 가 한 줄로 합친다)."""
    st = res.stats
    path = f"films/{film_id}/work/align.json"
    out = film_problems(doc_id, film_id, spoken=st["spoken"], absent_spoken=st["absent_spoken"], match=res.match,
                        conf_rows=[(res.conf.get(op["item"]), op["item"], op["op"])
                                   for op in res.ops if op["op"] != "absent"],
                        path=path)
    for iid in res.method["time_fix"]["suspect"][:ITEM_PROBLEMS_MAX]:
        out.append({"level": "item", "doc_id": doc_id, "film_id": film_id, "item_id": iid, "problem": "other",
                    "score": None, "detail": TIME_SUSPECT_TEXT, "path": path})
    return out


def select_docs(args) -> list[dict]:
    """대상: catalog 에서 영상이 있고(video_path) 대표 대본이며(is_primary) asr.json 이 있는 영화의 문서."""
    from . import catalog

    docs = [d for d in catalog.load()
            if d.get("status") != "excluded" and d.get("video_path") and d.get("is_primary") == "true"
            and d.get("film_id") and (paths.FILMS / d["film_id"] / "work" / "asr.json").exists()]
    if getattr(args, "docs", None):
        want = set(args.docs.split(","))
        docs = [d for d in docs if d["doc_id"] in want]
    if getattr(args, "limit", None):
        docs = docs[: args.limit]
    return docs


_CODE = re.compile(r"^\[([^\]]+)\]")


def _codes(errs: list[str]) -> str:
    """검사 오류를 코드별 수로 줄인 글('[C1] 3, [schema] 1'). 오류 글에는 대사가 섞일 수 있어 문제 기록에는 이것만 적는다."""
    c = Counter(m.group(0) if (m := _CODE.match(e)) else "[?]" for e in errs)
    return ", ".join(f"{k} {v}" for k, v in sorted(c.items()))


def _fail(doc_id: str, film_id: str, detail: str, score=None) -> None:
    """영화 수준 실패 한 줄을 문제 기록으로(앞선 실행의 기록을 갈아 치운다). detail 에 대사 글을 넣지 않는다."""
    util.set_problems(STAGE, film_id, [{"level": "film", "doc_id": doc_id, "film_id": film_id, "item_id": "",
                                        "problem": "other", "score": score, "detail": detail,
                                        "path": f"films/{film_id}/work/align.json"}])


def film_habits(film_id: str) -> dict[str, int]:
    """그 영화의 버릇 글귀(``asr.film_habits``). 받아쓰기 캐시가 없어 읽을 수 없으면 빈 dict(조용히 넘어간다)."""
    from . import asr as asr_mod

    try:
        return dict(asr_mod.film_habits(film_id))
    except (FileNotFoundError, OSError, ValueError, KeyError, TypeError, IndexError):
        return {}


def process(doc: dict, *, force: bool = False, llm_calls: bool = False, workers: int | None = None
            ) -> tuple[str, str, float | None]:
    """문서 하나(= 영화 하나)를 맞춰 align.json 을 쓴다. (상태 낱말 ok|skip|fail, 설명, 일치율).

    임시 파일(align.json.check)에 쓰고 검사를 통과한 뒤에 align.json 으로 바꿔 놓는다. 검사에 실패하면 앞선 align.json 은
    그대로 두고(단계 지문도 그대로라 다음 실행이 다시 맞춘다) 문제 기록에는 오류 코드별 수만 적는다.
    LLM 짝 판정은 늘 적용한다: llm_calls 가 거짓이면 캐시에 있는 판정만(부르지 않음), 참이면 캐시에 없는 요청도 claude 로.
    적용한 판정의 해시가 단계 지문에 들어가서 캐시가 생기거나 바뀌면 다시 만든다(그래서 건너뛰기 전에 맞추기를 해 본다)."""
    from . import validate

    doc_id, film_id = doc["doc_id"], doc["film_id"]
    sp = paths.DOCS / doc_id / "script.json"
    ap = paths.FILMS / film_id / "work" / "asr.json"
    out = paths.FILMS / film_id / "work" / "align.json"
    if not sp.exists():
        return "fail", "script.json 없음", None
    try:
        duration = float(doc.get("video_duration") or 0) or None
    except ValueError:
        duration = None
    habits = film_habits(film_id)
    fp = {"version": VERSION, "timeline_version": timeline.VERSION, "script_sha1": util.sha1_file(sp),
          "asr_sha1": util.sha1_file(ap), "video_duration": "" if duration is None else f"{duration:.3f}",
          "habits_sha1": util.sha1_bytes(json.dumps(habits, ensure_ascii=False, sort_keys=True).encode("utf-8"))}
    old = util.stage_fp(STAGE, film_id) or {}
    script = util.read_json(sp)
    asr = util.read_json(ap)
    res = build_ops(script, asr, duration, habits=habits, llm_pair=True, llm_calls=llm_calls, workers=workers)
    fp["llm_pair"] = res.method["llm_pair"]["decisions_sha1"]
    run_note = _llm_note(res)
    if not force and {k: old.get(k) for k in fp} == fp and out.exists() and util.sha1_file(out) == old.get("out_sha1"):
        return "skip", run_note, old.get("match")
    tmp = out.with_name(out.name + ".check")
    util.write_json(tmp, {"schema": "sink.align/1", "script_doc": doc_id, "asr_sha1": fp["asr_sha1"],
                          "method": res.method, "ops": res.ops, "scenes": res.scenes})
    try:
        errs = validate.validate_align(tmp, asr_path=ap, script_path=sp)
        if not errs:
            os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()
    if errs:
        for e in errs[:SHOW_ERRORS]:
            print(f"  ! {film_id}: {e}", file=sys.stderr, flush=True)
        msg = f"align.json 검사 실패 {len(errs)}건({_codes(errs)}): align.json 을 쓰지 않음"
        _fail(doc_id, film_id, msg, len(errs))
        return "fail", msg, None
    util.set_problems(STAGE, film_id, _problems(doc_id, film_id, res))
    util.set_stage_fp(STAGE, film_id, dict(fp, out_sha1=util.sha1_file(out), match=res.match))
    st = res.stats
    by = ", ".join(f"{pid} {p['chosen_lines']}줄" + (f"+added {p['added_lines']}" if p["added_lines"] else "")
                   for pid, p in res.method["passes"].items() if p["chosen_lines"] or p["added_lines"])
    return "ok", (f"일치율 {res.match:.3f}, same {st['same']} changed {st['changed']} absent {st['absent']} "
                  f"added {st['added']} (말 없는 대사 {st['silent_turns']}; {by}); {run_note}"), res.match


def _llm_note(res: AlignResult) -> str:
    """LLM 짝 판정과 헛소리 빼기 요약(화면용 한 줄)."""
    pair = res.method.get("llm_pair") or {}
    run_ = res.llm_run or {}
    acc = pair.get("accepted") or {}
    why = Counter(d["why"] for d in res.method.get("noise_dropped") or [])
    note = (f"헛소리 묶음 뺌 {sum(why.values())}(" + ", ".join(f"{k} {v}" for k, v in sorted(why.items())) + "); "
            f"LLM 짝 판정 창 {pair.get('windows', 0)}·요청 {pair.get('requests', 0)}(캐시 {run_.get('cached', 0)}, "
            f"새로 부름 {run_.get('called', 0)}, 캐시 없음 {run_.get('missing', 0)}, 실패 {run_.get('failed', 0)}), "
            f"받아들임 줄 {acc.get('lines', 0)}·머리 {acc.get('head', 0)}·꼬리 {acc.get('tail', 0)}, "
            f"버림 {sum((pair.get('rejected') or {}).values())}")
    if run_.get("stopped"):
        note += f"; 멈춤: {run_['stopped']}"
    return note


def run(args) -> int:
    from . import catalog, llm

    docs = select_docs(args)
    force = bool(getattr(args, "force", False))
    # LLM 짝 판정: 'sinkprep align --llm' 일 때만 캐시에 없는 요청을 claude 로 묻는다. 그 밖('sinkprep all' 포함)은 캐시만.
    # 캐시만 쓰는 실행은 llm.load_cached 로 읽기만 하고 llm.call 을 부르지 않으므로(align_llm.ask) llm 의 실행 상태(캐시 전용
    # 표시·호출 수)를 건드리지 않는다. 부르는 실행만 그 상태를 새로 연다(llm 단계의 reset_run 과 같은 약속)
    calls = getattr(args, "cmd", None) == "align" and bool(getattr(args, "llm", False))
    workers = getattr(args, "workers", None) or llm.DEFAULT_WORKERS
    if calls:
        llm.set_workers(workers)
        llm.reset_run(offline=False)
    counts: Counter = Counter()
    updates: dict[str, dict] = {}
    t0 = time.time()
    for k, d in enumerate(docs, 1):
        try:
            st, msg, match = process(d, force=force, llm_calls=calls, workers=workers)
        except Exception as e:  # noqa: BLE001 - 한 영화가 실패해도 나머지는 계속
            # 예외 글에는 대사가 섞일 수 있다: 화면에는 다 보이고 문제 기록에는 예외 이름만 남긴다
            st, msg, match = "error", f"{type(e).__name__}: {e}", None
            _fail(d["doc_id"], d["film_id"], f"맞추기가 예외로 끝남({type(e).__name__}): align.json 을 쓰지 않음")
        counts[st] += 1
        if st == "ok" or (st == "skip" and match is not None
                          and (d.get("status") not in ("aligned", "done") or d.get("match") != f"{match:.3f}")):
            upd = {"match": f"{match:.3f}"}
            if st == "ok" or d.get("status") not in ("aligned", "done"):
                upd["status"] = "aligned"
            updates[d["doc_id"]] = upd
        if st != "skip":
            print(f"[{k}/{len(docs)}] {d['doc_id']} {d['film_id']} {st}: {msg}", flush=True)
            if st in ("fail", "error"):
                print(f"  ! {d['doc_id']}: {msg}", file=sys.stderr, flush=True)
    if updates:
        catalog.finalize(updates=updates)
    why = llm.stopped()
    print(f"{STAGE}: {dict(counts)} ({time.time() - t0:.0f}초)"
          + (f", claude 호출 {llm.calls_made()}번" if calls else ", LLM 짝 판정은 캐시만(부르려면 align --llm)")
          + (f"; 멈춤: {why}" if calls and why else ""), flush=True)
    return 1 if (counts["fail"] or counts["error"]) else 0


if __name__ == "__main__":
    sys.exit(run(None))
