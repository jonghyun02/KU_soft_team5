"""PS-TTS 논문 그대로 판.

논문 3장의 절차(ISO 60번 되풀이, PS 후보 60개, 모음 DTW, PS-Comet)를 그대로 따른다.
TTS 에서 나오는 값(음소 길이, 모음 거리표)만 Frontend 뒤에 두고, 지금은 근사(approx)를 쓴다.
기존 판(pstts)이 논문에 더한 조건은 넣지 않는다. 자세한 것은 README_paper.md.
"""
