# sinkprep 설계 (모듈 사이 약속)

형식 명세: `.omc/specs/deep-interview-sink-format.md` (4장). 이 문서는 구현 모듈끼리의 약속만 적는다.

## 실행
- 가상환경: `C:\projects\sink\.venv` (editable 설치). 실행은 항상 `.venv\Scripts\python -I -m sinkprep <명령>`.
- `-I`는 필수다. 원본 대본은 인터넷에서 받은 신뢰할 수 없는 데이터라 실행하지 않고, 원본이 있는 폴더를 현재 폴더로 두고 파이썬을 돌리지 않는다.
- 원본 `filmmakers_scripts/`(= 명세 `raw/`)에는 쓰지 않는다(`paths.assert_not_raw`, `util.write_*`가 막음).
- 모든 파일 쓰기는 `util.write_json / write_jsonl / write_text`(원자적, UTF-8)로 한다. 글자는 `util.nfc`.

## 흐름과 파일
```
catalog   filmmakers_scripts/**  (+ 압축 해제 extracted/{post_id}/**)  -> catalog.csv
convert   hwp3/hwp5/doc/docx/rtf 파일                                 -> cache/pdf/{doc_id}/{k}.pdf, cache/convert_log.jsonl
ocr       pdf_scan 파일 + 글자층 깨진 PDF(변환 PDF 포함)              -> cache/ocr/{doc_id}/{k}.json
lines     catalog + cache                                              -> docs/{doc_id}/lines.jsonl
parse     lines.jsonl                                                  -> docs/{doc_id}/script.json
review    cache/problems/** + 문서 품질                                 -> qa/needs_review.csv
validate  schema/*.schema.json + 의미 규칙                              -> 오류 목록(종료코드 1)
```
`k`는 catalog `files` 칸 안 파일 순서(0부터).

## catalog.csv
명세 4.3 그대로. 추가 약속:
- `files`, `sha1`은 `|`로 나눈 목록. 경로는 ROOT 기준 상대 경로(슬래시), 예: `filmmakers_scripts/한국영화 시나리오/간신 [5691744].docx`, `extracted/54292/강원도의 힘.hwp`.
- `real_format`이 `mixed`면 파일마다 형식이 다르다. 파일별 형식은 `file_formats` 칸(`|` 목록)에 둔다(명세 4.3에 칸 하나 추가).
- `status=excluded`면 `note`에 사유(예: `not_script:exe`, `archive_container`, `duplicate_file_of:fm55536`).
- 압축 파일 자체는 문서가 아니다: catalog에 `status=excluded, note=archive_container:<풀린 폴더>` 행으로 남기고, 풀린 파일들이 문서가 된다.
- 읽기: `sinkprep.catalog.load() -> list[dict]` (CSV 행 dict, `files`는 list로 풀어서 `files_list`, `file_formats_list` 추가).

- 원본 `filmmakers_scripts/manifest.csv`는 대본이 아니라 다운로드 기록이다(글 번호·제목·첨부 원래 이름). catalog가 제목과 글 번호를 얻는 데 쓰고, A1 대상 파일에서는 뺀다.
- catalog 단계는 원본만 보고 정할 수 있는 칸을 채운다. 글자를 읽어야 정해지는 칸(`script_lang`, `is_translation` 보정, `extract_quality`, `is_primary`, `status`)은 parse 뒤 `review` 단계가 `catalog.finalize()`로 갱신한다. catalog 단계에서는 `script_lang=unknown`(제목으로 알 수 있으면 그 값), `is_primary`는 임시값.

## lines.jsonl 머리 행
첫 행은 머리 행이다: `{"schema":"sink.lines/1","doc_id":"fm55574","files":["…"],"made":{"stage":"lines","version":"0.1.0"}}`.
둘째 행부터 명세 4.4.1의 줄 행(`n`은 0부터). 그래서 모든 JSON/JSONL 파일이 schema 버전을 가진다(E1).

## 형식 값
`real_format`: hwp3, hwp5, hwpx, doc, docx, rtf, pdf_text, pdf_scan, pdf_broken(열리지 않음), txt, xls, archive, other.
lines의 `src`: hwp, doc, docx, rtf, pdf_text, ocr, txt, xls(엑셀 표 대본: 한 행 = 한 줄, 칸 사이 탭, `cell`={table:시트, row, col:첫 칸}, xlrd 로 읽음).

## convert 결과
- `cache/pdf/{doc_id}/{k}.pdf`
- `cache/convert_log.jsonl` 한 줄: `{"doc_id","k","src","real_format","ok","method","pages","error","seconds"}`.
  `method`: hancom | fallback_hwp3_text | fallback_hwp5_bodytext | fallback_hwp5_prvtext | fallback_hwpx | fallback_antiword | fallback_rtf | fallback_docx.
- 대체 경로로 글자만 얻은 경우: `cache/pdf/{doc_id}/{k}.txt` (UTF-8 평문, 레이아웃 없음).

## ocr 결과 `cache/ocr/{doc_id}/{k}.json`
```json
{"engine":"…","engine_version":"…","source":"cache/pdf/fm1/0.pdf|filmmakers_scripts/…pdf","dpi":300,
 "pages":[{"page":1,"width":2480,"height":3508,
   "lines":[{"text":"S#1. 빵집. 낮","box":[x0,y0,x1,y1],"conf":0.97}]}]}
```
`box`는 픽셀 좌표. lines 단계가 쪽 크기로 나눠 0~1로 바꾼다.

## 문제 기록
단계마다 `util.set_problems(stage, key, [...])`. problem dict 칸: `level, doc_id, film_id, item_id, problem, score, detail, path`.
problem 값: extract_broken, ocr_low, parse_low_dialogue, speaker_missing, version_mismatch, translation_unmatched, asr_low_conf, absent_many, other (명세 4.8).

## 다시 돌리기
단계마다 `util.stage_fp(stage, doc_id)`에 입력 지문(원본 sha1, 앞 단계 출력 sha1, 단계 버전)을 남기고, 같으면 건너뛴다(`--force`면 다시).

## 구현 중 정한 것 (2026-11-13, 모듈 작성 보고 반영)
- catalog: `file_formats`는 `real_format` 바로 뒤 칸. 제외 행 ID는 문서 다음 번호(`fm{글번호}_2`…). 원본 디스크 파일 701개(manifest 702 중 exe 1개는 내려받지 않음 → fm56878 note `missing_attachment`).
  note 값: `content:subtitle_transcript`, `content:synopsis`, `content:transcript`, `content:recording_script_xls`, `tv_episode:…`, `tv_series`, `missing_attachment:…`, `archive_container:…`, `duplicate_file_of:…`, `not_script:…`.
  `catalog.finalize(updates={doc_id:{칸:값}})` 로만 갱신(A2 어기면 ValueError).
  catalog 를 다시 만들 때 doc_id·files·sha1·film_id 가 그대로인 문서는 뒤 단계가 채운 칸(`catalog.LATER_COLUMNS`: review 칸과 video_*·match)을 이어받는다(carry_over). 이어받은 뒤 A2 를 어기는 영화는 새 임시값으로 되돌린다. 그래서 `sinkprep all` 을 다시 돌려도 catalog.csv 가 그대로다.
- review: script_lang(글자 비율·불용어로 ko/en/fr/es/mixed), is_translation(외국영화인데 ko 대본이면 true), extract_quality(script quality.score, OCR 이면 평균 conf 와 작은 값), status(script.json 있으면 parsed).
  is_primary 는 영화마다 (script 있음, 번역 아님, 품질, 판본 shooting>final>bound>revised>unknown>draft) 순으로 하나.
  영화 ID 가 local_ 인 것(KMDb/TMDB 키 없음)은 needs_review 에 넣지 않는다(못 읽은 대본 목록이 아니므로; catalog film_id 로 보임).
- convert: `cache/convert_log.jsonl` 칸에 `note`(rtf_sectd_removed(no_paperw), hancom_open_yes) 추가. method: hancom | fallback_hwp3_text | fallback_hwp5_bodytext | fallback_hwp5_prvtext | fallback_hwpx | fallback_antiword | fallback_rtf | fallback_docx.
  한컴 실패 뒤 대체 글만 얻은 문서는 problem `other`(score 0.5, 레이아웃 없음). HWP3 대체(`sinkprep.hwp3text`)는 문단 리스트를 직접 읽고(표·글상자 칸 포함, 머리말·꼬리말·각주·숨은 설명 제외), 구조가 망가진 곳에서 멈추면 그때까지 글을 쓰고 `error`에 `hwp3_partial:<까닭>@<위치>`를 남긴다. doc/rtf 는 한워드로 넘어가지 않게 빈 문서에 Insert. 매크로가 기본 틀이 아닌 HWP5 는 한컴으로 열지 않음.
  일부만 읽은 HWP3 은 problem `extract_broken`(score = 본문 중 읽은 비율, `error` 에 `;read=0.40`).
  교차 검사(convert.crosscheck): 한컴 변환이 성공해도 대체 경로로 읽은 원본 글과 견준다. 원본 글자(문자·숫자) 중 PDF 에 없는 비율 `loss` 를 로그 칸에 남기고,
  0.01 초과면 problem `other`(PDF 는 그대로, 레이아웃이 있어야 파싱됨), 0.03 초과면 PDF 를 `{k}.incomplete.pdf` 로 옮기고 대체 글 `{k}.txt` 를 쓴다(note `hancom_pdf_incomplete`).
  전체 481문서에서 정상 변환은 loss 0.002 미만, 한컴 2024 가 줄을 빠뜨린 문서는 fm54163 0.015, fm56063 0.045, fm55169 0.158(쪽 밖으로 밀리거나 사라진 대사). 대조 도구 `tools/check_convert.py`.
- ocr: RapidOCR(PP-OCRv5 검출 + korean 인식 / 영어는 PP-OCRv6 small), 300dpi, CPU. 결과 문서 칸 `schema:"sink.ocr/1"`, source_sha1, lang, lang_how, made, seconds, failed_pages, conf_mean, why, textlayer; 쪽 칸 render, warn, rotate, part/parts/crop(두 쪽 펼침); 줄 칸 segs(떨어진 덩어리).
  쪽 그림 실패도 extract_broken. 글자층 검사(textlayer)가 깨졌다고 해도 사유가 띄어쓰기 빠짐·이상한 문자·한 글자 줄·같은 줄·겹친 글자뿐이고,
  lines 가 글자 좌표로 다시 만든 줄이 같은 기준을 통과하면 OCR 하지 않는다(ocr._restorable_by_lines, 원본 글자층 PDF·한컴 변환 PDF 모두). 글자층 글자가 OCR 보다 정확하기 때문.
  이 기준을 넣기 전에 OCR 한 결과 중 이제 복원으로 충분한 것은 cache/ocr_superseded/ 로 옮겨 lines 가 글자층을 쓰게 한다.
- lines: OCR 줄의 segs 를 덩어리마다 넣어 칸 사이를 탭으로. PDF 는 rawdict 글자 좌표(앞 공백 빼고 첫 글자 x), 공백 4칸 이상이면 조각 분리, 글자 크기 0.25배 넘게 벌어진 곳은 공백 복원, U+F000 은 공백. 칸 사이 탭 기준은 글자 폭 1.5배.
  같은 글자가 글자 크기 0.15배 안쪽에 또 찍혀 있으면(겹쳐 찍은 굵은 글씨) 하나만 남긴다(lines._drop_overprint). 모양 같은 글자 U+2028·U+2029→공백, U+A789→:, U+2011→-, U+00AD 삭제(lines.GLYPH_MAP).
  두 쪽 펼침(lines._gutter): 가로 쪽(가로>1.15×세로)의 0.3~0.7 너비에 글자가 없는 세로 띠(쪽 너비 1.5% 이상)가 있고 양쪽에 글이 15% 이상씩이면 그 띠에서 나눈다. 가로 쪽 절반 넘게 띠가 있으면 문서 전체를 펼침으로 보고 띠 없는 쪽은 띠 위치 중앙값에서 나눈다. 이때 page 는 논리 쪽 번호(왼쪽 먼저), x 는 그 쪽 안 위치. 말뭉치 31문서(밀정·날 보러와요 등)가 이 모양이었다(전에는 왼쪽·오른쪽 같은 높이 글이 탭으로 이어졌음).
- script.json dropped.why: page_header | page_footer | page_number | cont_marker | continued | revision_mark | scene_number_margin | blank | timecode | subtitle_cue 등 snake_case. quality.notes 는 문자열 목록.
- script.json 대사 요소의 선택 칸 `speaker_inferred`(파서 0.2.0, 명세 4.4.2 보완): 화자를 앞 대사에서 이어받은 요소에만 `true` 로 쓴다(false 는 쓰지 않음). schema/script.schema.json 의 el_dialogue 에 선택 속성으로, validate 빠른 검사(_EL_DIALOGUE_KEYS_INF)에도 넣었다. 칸이 없는 옛 파일도 유효.
- E1 은 다섯 형식(lines/script/asr/align/final)에 적용. cache/ 중간 JSON 은 대상 아님(ocr 결과는 schema 칸을 갖지만 검사 대상 아님).
- 검토(architect) 반영 2026-11-13:
  - review: 번역 판정은 양방향(외국영화의 ko 대본, 한국영화의 외국어 대본·제목 '영문'). 대표 대본 고르기에서 두 언어가 섞인 판(mixed)은 한 언어 판 뒤로.
    catalog 에 없는 doc_id 의 낡은 문제 기록은 목록에서 뺌. pdf_broken 이라 catalog 가 '글자를 꺼낼 수 없음'으로 적었어도 OCR 결과가 있으면 그 판정은 뺌(OCR 단계 문제가 실제 상태를 알림).
  - parse quality: `empty_dialogue`(말 없이 이름 줄만 잡힌 대사), `inferred_speaker` 를 센다. 대사 20개 이상인 문서에서 말 없는 대사 10% 이상이면 parse_low_dialogue, 추정 화자 10% 초과면 speaker_missing. score 에도 반영.
  - ocr/lines: OCR 결과는 source 경로·source_sha1 이 지금 입력과 같을 때만 건너뛰거나 쓴다(ocr.result_is_current). 원본·변환 PDF 가 바뀌면 다시 OCR.
  - convert: antiword 는 작업 폴더(cache/convert_work)를 현재 폴더로 실행. 한컴 시간 초과·일꾼 오류는 일시적일 수 있어 예전 PDF 를 지우지 않고(note kept_previous_pdf) 지문도 남기지 않는다.
  - catalog: 압축은 풀기 전에 반디집 목록(bz l)으로 항목 이름을 보고 '..'·절대 경로·드라이브 경로가 있으면 풀지 않는다(중첩 압축도).
  - validate: D7 뒤 절반(시각 없는 항목은 대본 순서로 앞뒤 항목 사이) 검사 추가.
