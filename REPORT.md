# sink 결과 보고: 대본 형식 통일 전처리와 영상 맞추기

만든 시각: 2026-11-15 21:17 · 생성기 `tools/make_report.py`(모든 숫자는 디스크 결과물에서 다시 셈)

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
- lines 판(머리 행 made.version): 0.4.1 684
- 공백 복원(머리 행 `made.spacing` 에 from=source, 한컴 PDF 줄의 공백·탭을 원본 문서 글로 맞춤): 10문서: fm54367(손실 0.2402), fm54441(손실 0.5193), fm54519(손실 0.2272), fm55036(손실 0.3959), fm55099(손실 0.4014), fm55455(손실 0.3503), fm55655(손실 0.4047), fm55664(손실 0.3089), fm55844(손실 0.3786), fm416766(손실 0.3692)
- 공백 손실을 못 견준 파일(from=null, skipped): 2개: fm55630/0 source_unreadable, fm56026/0 source_unreadable
- script.json 있는 문서 684/684
- 파서 판(parser.name·version): sink-parse 0.3.2 655, sink-parse-llm 0.2.2+lp1 29
- parse 0.3.x 새 규칙이 걸린 문서(docs 의 script.json quality.notes 꼬리표, 값 이름은 TYPES.md 5.2절): `scene_colon_heads` 7(fm54275, fm54773, fm54787, fm55664, fm55680, fm715668, fm6535167), `paren_names` 18(fm54275, fm54411, fm54455, fm54773, fm55087, fm55150, fm55180, fm55358, fm55664, fm55813, fm55927, fm765450 …), `wordless_turns` 50(fm54127, fm54275, fm54306, fm54404, fm54411, fm54596, fm54621, fm54714, fm54742, fm54773, fm54921, fm54980 …), `tight_cont` 44(fm54151, fm54180, fm54183, fm54230, fm54241, fm54256, fm54278, fm54362, fm54398, fm54455, fm54476, fm54502 …)
- 요소 종류별 줄 비율: dialogue 58.1%, action 36.4%, scene 4.2%, transition 0.6%, front 0.5%, onscreen 0.1%, unknown 0.0%
- 문서 품질 점수 분포: 10% 0.98 · 중앙값 1.00 · 90% 1.00
  - typed_ratio: 10% 1.00 · 중앙값 1.00 · 90% 1.00
  - dialogue_ratio: 10% 0.36 · 중앙값 0.56 · 90% 0.74
  - unknown_ratio: 10% 0.00 · 중앙값 0.00 · 90% 0.00

## 5. 파서 정확도(독립 정답 표본)
규칙 판 = 규칙 파서(sink-parse)만, 지금 판 = docs/*/script.json(llm 단계가 받아들인 문서는 LLM 판). 값은 요소 종류% / 화자%. hard-test 는 처음 열었을 때(LLM 단계 고치기 전) LLM 창 채점 100/100(120줄 조각 98.9/100)이었고, 그 뒤 hard-test 문서 사례를 보고 조립을 고쳤으므로 지금 값은 개발용으로 본다.

- 개발용(dev): 줄 610개(대사 줄 330개), 규칙 판 98.4 / 97.6, 지금 판 98.4 / 97.6
- 확인용(test, 파서 개발 중 안 봄): 줄 180개(대사 줄 118개), 규칙 판 98.3 / 98.3, 지금 판 98.3 / 98.3
- dev+test+hard-dev: 줄 970개(대사 줄 537개), 규칙 판 94.1 / 92.0, 지금 판 98.7 / 98.1
- 어려운 꼴 개발용(hard-dev): 줄 180개(대사 줄 89개), 규칙 판 75.6 / 62.9, 지금 판 100.0 / 100.0
- 어려운 꼴 확인용(hard-test): 줄 180개(대사 줄 118개), 규칙 판 81.7 / 75.4, 지금 판 100.0 / 100.0
- 정답 표본: 구간 26개, 문서 22개. Claude 모델(서브에이전트) 둘이 따로 표시하고 갈린 줄만 판정한 정답이다(사람이 붙인 정답이 아님). 위 정확도는 이 정답에 대한 값

## 5b. LLM 파싱 단계(llm)
- 모델 claude-opus-5-5(effort medium), 프롬프트 lp1, 조립 0.2.2. 다룬 문서 37개: LLM 판 사용 29, 규칙 판 유지 8
- docs 의 파서: sink-parse 655, sink-parse-llm 29
- claude 응답 캐시 869개(cache/llm). 같은 입력은 다시 부르지 않아 다시 돌려도 바이트 단위로 같다. `sinkprep all` 은 캐시만 쓴다.

## 6. 확인 필요 목록(qa/needs_review.csv)
- 49건(문서 23개): asr_low_conf 27, other 9, parse_low_dialogue 7, speaker_missing 4, extract_broken 2
- 수준별: item 27, doc 22

## 6b. 영상 쪽(영화별: 영상 등록·받아쓰기·맞추기·최종본)

catalog 에서 영상이 있는 영화 1편. 결과물은 `films/{film_id}/` (work/asr.json, work/align.json, final.json). 파일이 없으면 '없음'.

| 영화 | 대표 문서 | 영상 길이(초) | 받아쓰기 낱말(pass) | 일치율 | 최종본 항목 | final.json 검사 |
|---|---|---|---|---|---|---|
| local_0270 | fm55664 | 7548.093 | plain 4,012, lv3 4,104, rescue 270 | 0.613 | 2,096 | 검사 통과 |

### local_0270 (대표 문서 fm55664, catalog status done, match 0.613)

- 영상: `videos/local_0270.mp4`, 길이 7548.093초, 언어 ko, sha1 faa323e90398…. 시각 대응표(`cache/audio/faa323e90398ee539b1b8c79a1a3a63e7fc302d7.timemap.json`): mode map, 틈 750곳 합 42.088초, 소리 7506.005초 → 영상 7548.093초, sinkprep.media 0.2.0
- 받아쓰기(`films/local_0270/work/asr.json`, 언어 ko): 말 구간 1,156개 합 1,894.9초, 묶음 514개, pass 3개, 검사 통과
  - `plain`: 낱말 4,012개, 평균 확률 0.765, context -, 모델 mobiuslabsgmbh/faster-whisper-large-v3-turbo@0a363e91 · ct2 4.8.3 cuda float16 · sinkprep.asr 0.3.0 decode1
  - `lv3`: 낱말 4,104개, 평균 확률 0.786, context -, 모델 Systran/faster-whisper-large-v3@edaa852e · ct2 4.8.3 cuda float16 · sinkprep.asr 0.3.0 decode1
  - `rescue`: 낱말 270개, 평균 확률 0.688, context full_window_stable, 모델 mobiuslabsgmbh/faster-whisper-large-v3-turbo@0a363e91 · ct2 4.8.3 cuda float16 · sinkprep.asr 0.3.0 decode1 · rescue: full windows x2 win25 quiet5 staggered novad | outside plain chunks, not plain words dup0.5 same+gap0.1 | frag seg+gap0.6 | sim0.6 margin1 coloc0.5 otherstart rapidfuzz-ratio nfc-alnum | loop>4 habit7x3
  - rescue 통계: asr.json 에는 rescue pass 의 낱말만 있고 조각 수·버린 수 같은 통계 칸은 없음(asr 단계 출력에만)
- 맞추기(`films/local_0270/work/align.json`, sinkprep.align 0.4.0, 기준 pass plain): 대사 1,423 = 말 있는 대본 대사 975 + 말 없는 대사 59 + added 389; same 228, changed 490, absent 316(말 있는 대사 가운데 257), 일치율 0.613, 낱말 평균 확률 0.77, 검사 통과
  - 말 있는 대본 대사 가운데 받아쓰기와 짝지은 줄(same+changed) 718/975(73.6%)
  - pass 별(method.passes): `plain` role base, matched_lines 607, chosen_lines 512, added_lines 323, backsteps 0; `lv3` role alt, matched_lines 634, chosen_lines 160, added_lines 0, backsteps 0; `rescue` role rescue, matched_lines 30, chosen_lines 46, added_lines 66, backsteps 0
  - 고른 줄의 받아쓰기 pass(ops 의 asr 범위): plain 835, lv3 160, rescue 112; decided_by: rule 1,388, llm 35
  - pass 고르기(method.decide, 0 아닌 값): base_same 180, base_only 7, base_kept 291, alt_same 31, alt_closer 92, alt_jamo 2, alt_elsewhere 4, alt_clash 4, alt_only 35, alt_only_rejected 5, across_slot 23
  - 다시 짝지은 줄: decide.across_slot 23
  - 시각 보정(method.time_fix): 줄 1, 낱말 1, 믿기 어려운 줄 0개
  - 씬 구간(align scenes how): none 102, anchors 76, lines 14, moved 3
- 최종본(`films/local_0270/final.json`, sinkprep.final 0.3.0): 항목 2,096개(dialogue 1,423, action 468, scene 195, onscreen 8, transition 2), 검사 통과
  - quality: dialogue 1,423, same 228, changed 490, added 389, absent 316, speaker_unknown 392, mean_word_p 0.77, spoken 975, silent_turns 59, absent_spoken 257, line_coverage 0.736, match 0.613, speaker_inferred 0, mean_conf 0.695, low_conf 27, moved 7, words 4,273, by_pass {lv3 160, plain 835, rescue 112}, diff_kinds {filler 53, number 18, repeat 12, spelling 56, word 980}, time_fixed 1, time_fixed_words 1, time_suspect 0개, pass_overlaps 0개, lang_from_heard 0
  - 대사 status: changed 490, added 389, absent 316, same 228; 항목 timing: aligned 1,107, estimated 673, none 316
  - 씬 195개, 구간(start)이 있는 씬 195개
  - 화자 있는 대사(speakers 가 빈 목록이 아님): 대본 대사 1,031/1,034(99.7%), added 포함 1,031/1,423(72.5%); speaker_from script 1,031, unknown 392
  - 시각 보정: time_fixed 1줄(낱말 1), time_suspect 0개, pass_overlaps 0개
  - film.script.match 0.613, catalog match 0.613
- 확인 필요(`qa/needs_review.csv`): film 수준 0건(없음), item 수준 27건(asr_low_conf 27)
  - 단계 문제 기록(`cache/problems/{asr,align,final}/local_0270.json`, review 가 위 목록으로 모음): asr 없음, align item 27, final item 27

## 7. 수락 기준 확인(A1~A3, B1~B5, C1~C3, D1~D9, E1~E2)

검사는 `sinkprep validate`(인자 없음)와 같은 파일·같은 함수(catalog, needs_review, docs 전체, films, examples, 덮기 B1·D1)를 이 도구 안에서 돌려 오류를 모두 셌다: 검사 1,378개, 실패 0개, 오류 0개, 5초. 종류별 align 2개 중 실패 0, asr 2개 중 실패 0, catalog 1개 중 실패 0, final 2개 중 실패 0, lines 685개 중 실패 0, needs_review 1개 중 실패 0, script 685개 중 실패 0.

- A1 모든 원본·압축 해제 파일이 정확히 한 행에: 오류 0개
- A2 영화마다 is_primary 최대 1개: 오류 0개
- A3 앞 바이트 판별: 위장 파일 4개 판별(위 1절), tests/test_detect.py
- B1·B2·B3·E1·E2 검사 결과 오류 코드별(전체, 아래 C·D 포함): 없음
  - docs: 없음
  - coverage: 없음
  - catalog: 없음
  - needs_review: 없음
  - films: 없음
  - examples: 없음
- B4 needs_review 기준값: typed_ratio<0.85, dialogue_ratio<0.15, 화자 없는 대사>5%, 말 없는 대사(이름 줄만)>=10%, 추정 화자(speaker_inferred)>10%, OCR conf<0.85, 추출 실패·쪽 그림 실패, 한컴 PDF 글자 손실>1%
- B5 까다로운 경우 11가지: tests/fixtures/parse/, tests/test_parse.py
- 영상 쪽 검사 대상: films 의 align 1, asr 1, final 1, examples 의 asr·align·final 도 같은 검사
- C1 오류 films 0 · examples 0. 검사: validate_asr: pass 마다 낱말 번호 i 가 0..n-1 로 빠짐·겹침 없음, s<=e. validate_align: op 의 asr 참조(pass:i)가 asr.json 안에 있고 script_el 이 script.json 에 있음, asr_sha1 = asr.json 파일 sha1
- C2 오류 films 0 · examples 0. 검사: validate_align: op 의 item 과 final.json 대사 항목이 1:1, 같은 pass 의 같은 낱말을 두 op 이 갖지 않음, added op 에 asr 범위가 있음, 대사 항목 words = 그 op 의 asr 범위 낱말(timeline.op_words). final 의 film.id = 폴더 이름
- C3 asr_sha1 어긋남 0개, asr.json 스키마 오류 0개(films·examples). 검사: asr.json 스키마(sink.asr/1)에 정렬 칸이 없고 align.json 의 asr_sha1 이 지금 asr.json 과 같음([C1] 로 냄). 같은 asr.json 에 받아쓰기 캐시(cache/asr 의 rescue 기록: 버릇 글귀)와 LLM 캐시(cache/llm 짝 판정)도 같을 때 다시 맞추면 바이트가 같은지는 tests/test_align.py·test_final.py(align 은 asr.json 말고 두 캐시도 입력으로 읽는다)
- D1 영상이 있는 영화 1편 가운데 final.json 이 있는 영화 1편; [D1] 오류 0개(없는 final.json·catalog 와 어긋남), films 의 final.json 스키마 오류 0개. 검사: 인자 없는 validate 가 catalog 에서 영상이 있는 대표 문서마다 films/{film_id}/final.json 이 있는지, 그 film 칸(대표 문서·영상 sha1·길이·match)이 catalog 와 같은지 봄([D1]), 스키마는 [schema]
- D2 오류 films 0 · examples 0. 검사: validate_final: same·changed·added 는 start<end, words 있음, 낱말이 줄 안, heard = words, 시각이 영상 길이 +0.5초 안, 줄 안 낱말 시작 시각이 줄지 않음
- D3 오류 films 0 · examples 0. 검사: validate_final: absent 는 start=end=null·script_ref 있음, added 는 script_ref=null
- D4 오류 films 0 · examples 0. 검사: validate_final: changed 는 diffs 있음·same 은 없음, match=text 줄은 textnorm.key(heard)와 key(script_ref.text)가 same 이면 같고 changed 면 다름
- D5 오류 films 0 · examples 0. 검사: validate_final: 시각 있는 대사에 ko 가 있고, lang=ko 면 ko_src=same·ko = heard, 시각 있는 대사의 들린 글이 영상 언어의 글자면 lang 이 영상 언어(textnorm.spoken_lang)
- D6 오류 films 0 · examples 0. 검사: validate_final: match=translation 이면 diffs 없음
- D7 오류 films 0 · examples 0. 검사: validate_final: items 가 시각·id 차례, 시각 없는 항목은 대본 순서로 앞뒤 항목 사이
- D8 오류 films 0 · examples 0. 검사: validate_final: scene·speakers·script_ref.el·src_el 참조가 있음, script.json 을 찾으면 그 scene·action·dialogue·onscreen·transition 요소가 한 번씩 나옴
- D9 오류 films 0 · examples 0, 대본(script.json)과 견준 final.json 1/1개. **영상이 있는 영화 1편 중 final.json 이 검사를 통과한 영화 1편**(local_0270). 검사: validate_final: script.json 을 찾았을 때 대본에서 복사한 칸(대사의 speakers·how·lang·paren·sung·overlap·scene·script_ref.text, 대사 아닌 항목의 text·scene·kind, 씬·인물 칸)이 대본과 같고(시각 있는 대사의 lang 은 textnorm.spoken_lang 으로 정한 값) 씬 구간이 그 씬 항목 시각과 맞음. 곧 CBoE 입력 칸이 final.json 만으로 채워짐
- films 의 그 밖의 오류 코드(스키마·NFC·시각 자리수 등): 없음

(보고 생성 21초: 검사 5초, 파서 정확도 3초)
