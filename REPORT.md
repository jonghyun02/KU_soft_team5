# sink 대본 형식 통일 전처리 결과 보고

만든 시각: 2026-11-13 13:27 · 생성기 `tools/make_report.py`(모든 숫자는 디스크 결과물에서 다시 셈)

## 1. 원본과 카탈로그
- 원본 파일 701개(manifest 제외) + 압축에서 나온 파일 13개 = 714개
- catalog.csv 712행: 문서 684개, 제외 28개, 영화 638편
- 파일별 실제 형식: hwp5 289, hwp3 143, pdf_text 112, pdf_scan 74, doc 57, txt 20, archive 12, rtf 2, docx 2, other 1, xls 1, pdf_broken 1
- 문서별 형식: hwp5 280, hwp3 141, pdf_text 110, pdf_scan 72, doc 56, txt 19, rtf 2, docx 2, xls 1, pdf_broken 1
- 제외 사유: duplicate_file_of 16, archive_container 11, not_script 1
- 대본 언어(review 판정): ko 404, en 274, mixed 4, es 1, fr 1
- 번역 대본: 13개
- 확장자와 실제 형식이 다른 파일 4개(앞 바이트로 판별): 드디어 반칙왕 시나라오를 [54331].doc -> other; _가을로_ 시나리오 [55354].doc -> rtf; _M_ 시나리오 [55542].doc -> rtf; JFK [56704].doc -> txt

## 2. 한컴 변환(HWP3/HWP5/doc/docx/rtf -> PDF)
- 변환 대상 파일 481개(최근 기록 기준)
  - doc: 성공 fallback_antiword 4
  - doc: 성공 hancom 52
  - docx: 성공 hancom 2
  - hwp3: 성공 fallback_hwp3_text 1
  - hwp3: 성공 hancom 140
  - hwp5: 성공 fallback_hwp5_bodytext 3
  - hwp5: 성공 hancom 277
  - rtf: 성공 hancom 2
- 파일당 시간 중앙값 9.1초, 최대 79.5초
- 교차 검사(대체 경로로 읽은 원본 글자 중 PDF 에 없는 비율 loss): 473개 파일 검사, 중앙값 0.0000, 0.002 미만 470개; PDF 대신 원본 글 사용(loss>0.03) 2개 ['fm55169', 'fm56063'], PDF 쓰고 확인 필요(0.01~0.03) 1개 ['fm54163']
- 형식별 내용 대조(`tools/check_convert.py`, 한컴 변환 성공 파일 전부, 공백 뺀 글자 다중집합 겹침 / 20자 구간 일치):
  - doc: 51개, 겹침 중앙값 1.000, 구간 일치 중앙값 100%
  - docx: 2개, 겹침 중앙값 0.992, 구간 일치 중앙값 97%
  - hwp3: 140개, 겹침 중앙값 0.996, 구간 일치 중앙값 100%
  - hwp5: 278개, 겹침 중앙값 0.998, 구간 일치 중앙값 100%
  - rtf: 2개, 겹침 중앙값 0.997, 구간 일치 중앙값 95%

## 3. OCR (RapidOCR, PP-OCRv5 korean / PP-OCRv6 en)
- OCR 파일 81개, 7723쪽, 그림 실패 쪽 11개; 사유: pdf_scan 73, textlayer_broken 5, converted_textlayer_broken 2, pdf_broken 1
- 글자층 복원(lines)으로 충분해 쓰지 않는 예전 OCR 결과 5개(cache/ocr_superseded): fm36187123/0, fm3659631/0, fm56423/0, fm56494/0, fm56712/0
- 파일별 평균 신뢰도: 중앙값 0.956, 최저 0.889, 0.85 미만 0개

## 4. 형식 통일 줄(lines.jsonl)과 구조 파싱(script.json)
- lines.jsonl 있는 문서 684/684, 줄 0개 문서 0개, 총 줄 2,163,497
- 줄 출처: hwp 1,202,057, pdf_text 480,594, ocr 250,713, doc 179,153, txt 39,231, docx 6,194, rtf 4,655, xls 900
- script.json 있는 문서 684/684
- 요소 종류별 줄 비율: dialogue 58.1%, action 36.4%, scene 4.2%, transition 0.6%, front 0.5%, onscreen 0.1%, unknown 0.0%
- 문서 품질 점수 분포: 10% 0.97 · 중앙값 1.00 · 90% 1.00
  - typed_ratio: 10% 1.00 · 중앙값 1.00 · 90% 1.00
  - dialogue_ratio: 10% 0.36 · 중앙값 0.55 · 90% 0.74
  - unknown_ratio: 10% 0.00 · 중앙값 0.00 · 90% 0.00

## 5. 파서 정확도(독립 정답 표본)
규칙 판 = 규칙 파서(sink-parse)만, 지금 판 = docs/*/script.json(llm 단계가 받아들인 문서는 LLM 판). 값은 요소 종류% / 화자%. hard-test 는 처음 열었을 때(LLM 단계 고치기 전) LLM 창 채점 100/100(120줄 조각 98.9/100)이었고, 그 뒤 hard-test 문서 사례를 보고 조립을 고쳤으므로 지금 값은 개발용으로 본다.

- 개발용(dev): 줄 610개(대사 줄 330개), 규칙 판 98.4 / 97.6, 지금 판 98.4 / 97.6
- 확인용(test, 파서 개발 중 안 봄): 줄 180개(대사 줄 118개), 규칙 판 98.3 / 98.3, 지금 판 98.3 / 98.3
- dev+test+hard-dev: 줄 970개(대사 줄 537개), 규칙 판 94.1 / 92.0, 지금 판 98.7 / 98.1
- 어려운 꼴 개발용(hard-dev): 줄 180개(대사 줄 89개), 규칙 판 75.6 / 62.9, 지금 판 100.0 / 100.0
- 어려운 꼴 확인용(hard-test): 줄 180개(대사 줄 118개), 규칙 판 81.7 / 75.4, 지금 판 100.0 / 100.0
- 정답 표본: 구간 26개, 문서 22개, 두 사람 독립 표시 후 갈린 줄만 판정

## 5b. LLM 파싱 단계(llm)
- 모델 claude-opus-5-5(effort medium), 프롬프트 lp1, 조립 0.2.2. 다룬 문서 37개: LLM 판 사용 29, 규칙 판 유지 8
- docs 의 파서: sink-parse 655, sink-parse-llm 29
- claude 응답 캐시 855개(cache/llm). 같은 입력은 다시 부르지 않아 다시 돌려도 바이트 단위로 같다. `sinkprep all` 은 캐시만 쓴다.

## 6. 확인 필요 목록(qa/needs_review.csv)
- 21건(문서 21개): other 9, parse_low_dialogue 7, speaker_missing 3, extract_broken 2

## 7. 수락 기준 확인
- A1 모든 원본·압축 해제 파일이 정확히 한 행에: 오류 0개
- A2 영화마다 is_primary 최대 1개: 오류 0개
- A3 앞 바이트 판별: 위장 파일 4개 판별(위 1절), tests/test_detect.py
- B1·B2·B3·E1·E2 `sinkprep validate`(catalog, needs_review, docs 전체, examples): 검사 1375개, 실패 0개, 오류 0개, 4.3초
- B4 needs_review 기준값: typed_ratio<0.85, dialogue_ratio<0.15, 화자 없는 대사>5%, 말 없는 대사(이름 줄만)>=10%, 추정 화자(speaker_inferred)>10%, OCR conf<0.85, 추출 실패·쪽 그림 실패, 한컴 PDF 글자 손실>1%
- B5 까다로운 경우 11가지: tests/fixtures/parse/, tests/test_parse.py

(보고 생성 16초)
