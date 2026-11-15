# 원본 파일 유형 분류 (코드가 나누는 기준과 결과)

만든 시각: 2026-11-15 20:19 · 생성기 `tools/make_types.py`(모든 숫자는 디스크 결과물에서 다시 셈) · 문서별 표는 `types.csv`

규칙 파서 판 기준(`--rules`): LLM 판으로 바뀐 문서도 보관된 규칙 판(cache/parse_rules)으로 센다.

원본 `filmmakers_scripts/` 파일 701개를 코드는 다섯 번 나눈다. 앞 단계의 유형이 뒤 단계의 처리 경로를 정한다.

| 차례 | 나누는 기준 | 코드 | 결과가 남는 곳 |
|---|---|---|---|
| 1 | 파일 형식(앞 바이트·내부 구조) | `sinkprep/detect.py` | catalog.csv `real_format`, `file_formats` |
| 2 | 문서로 묶기와 제외 | `sinkprep/catalog.py` | catalog.csv 한 행 = 한 문서, `status`, `note` |
| 3 | 글자를 얻는 경로 | `sinkprep/convert.py`, `ocr.py`, `lines.py` | cache/convert_log.jsonl, cache/ocr, lines.jsonl `src` |
| 4 | 내용 종류·언어·판본 | `sinkprep/catalog.py`, `review.py` | catalog.csv `note`, `script_lang`, `version_label` |
| 5 | 대본 꼴(대사 표기·씬 머리) | `sinkprep/parse.py`, `parse_layout.py`, `parse_text.py` | script.json 요소, `quality.notes` |

## 1. 파일 형식

`detect.real_format()` 이 확장자가 아니라 앞 바이트와 내부 구조로 정한다. 압축은 풀어서 안의 파일을 다시 판별한다.

| 형식 | 판별 규칙 | 파일 | 문서 | 한국영화 | 외국영화 |
|---|---|---|---|---|---|
| `hwp5` | OLE 복합 문서이고 FileHeader 스트림이 `HWP Document File` 로 시작 | 289 | 280 | 181 | 99 |
| `hwp3` | 파일 머리가 `HWP Document File V3` | 143 | 141 | 122 | 19 |
| `hwpx` | ZIP 이고 mimetype 이 `application/hwp+zip` 이거나 Contents/section*.xml 이 있음 | 0 | 0 | 0 | 0 |
| `doc` | OLE 복합 문서이고 WordDocument 스트림이 있음 | 57 | 56 | 18 | 38 |
| `docx` | ZIP 이고 word/document.xml 이 있음 | 2 | 2 | 2 | 0 |
| `rtf` | `{\rtf` 로 시작 | 2 | 2 | 2 | 0 |
| `pdf_text` | `%PDF-` 이고 글자 50자 미만인 쪽이 절반보다 적음 | 112 | 110 | 14 | 96 |
| `pdf_scan` | `%PDF-` 이고 글자 50자 미만인 쪽이 절반 이상(그림만 있는 스캔) | 74 | 72 | 47 | 25 |
| `pdf_broken` | `%PDF-` 인데 열리지 않거나 암호가 걸렸거나 0쪽 | 1 | 1 | 0 | 1 |
| `txt` | UTF-8·UTF-16·cp949 로 풀리고 제어 문자가 거의 없음 | 20 | 19 | 7 | 12 |
| `xls` | OLE 의 Workbook/Book 스트림, 또는 ZIP 의 xl/workbook.xml | 1 | 1 | 1 | 0 |
| `archive` | zip(`PK`)·rar(`Rar!`)·alz(`ALZ`) 압축: 풀어서 안의 파일을 다시 판별 | 12 | 0 | 0 | 0 |
| `other` | 실행 파일(`MZ`)·그림·알 수 없는 바이너리 | 1 | 0 | 0 | 0 |

- 파일 714개(원본 701개 + 압축에서 나온 파일), 문서 684개(한국영화 394, 외국영화 290)
- 확장자와 실제 형식이 다른 파일 4개: .doc 인데 실제는 rtf 2, .doc 인데 실제는 other 1, .doc 인데 실제는 txt 1

## 2. 문서로 묶기와 제외

`catalog.group_post()` 가 게시글 하나의 파일들을 문서로 묶는다. 같은 바탕 이름에 번호가 이어지는 파일(상·중·하, part1…)은 한 문서로 합치고(`merge_parts`), 대본이 아닌 파일·같은 글 안 중복 파일·압축 파일 자체는 제외 행으로 남긴다.

- 게시글 686개, catalog.csv 712행 = 문서 684개 + 제외 28개
- 파일이 둘 이상인 문서 1개, 압축에서 나온 문서 13개, 한 게시글에서 문서가 둘 이상 나온 글 13개

| 제외 사유(`note`) | 뜻 | 행 |
|---|---|---|
| `duplicate_file_of` | 같은 글 안에서 내용이 같은 파일(sha1 일치) | 16 |
| `archive_container` | 압축 파일 자체(풀린 파일이 문서가 됨) | 11 |
| `not_script` | 대본으로 읽을 수 없는 형식(실행 파일·그림·자막 파일 등) | 1 |

## 3. 글자를 얻는 경로

형식에 따라 글자를 얻는 길이 갈린다. 문서별 값은 `types.csv` 의 `route` 칸.

| 경로 | 이름 | 방법 | 문서 | 원본 형식 | 코드 |
|---|---|---|---|---|---|
| `hancom_pdf` | 한컴 변환 PDF 의 글자층 | 한글 2024 로 PDF 를 만들고 글자 좌표로 줄을 만든다(레이아웃 있음) | 471 | hwp5 276, hwp3 139, doc 52, rtf 2, docx 2 | convert.convert_file, lines.lines_from_pdf |
| `fallback_text` | 대체 경로 평문 | 한컴이 못 열거나 PDF 에서 글자가 3% 넘게 빠진 파일. 파일을 직접 풀어 글만 얻는다(레이아웃 없음) | 8 | doc 4, hwp5 3, hwp3 1 | convert.fallback_text, lines.lines_from_text |
| `pdf_textlayer` | 원본 PDF 의 글자층 | 글자 좌표로 줄을 만든다(레이아웃 있음) | 105 | pdf_text 105 | lines.lines_from_pdf |
| `ocr` | OCR | 스캔 PDF 와 글자층이 깨진 PDF 를 300dpi 그림으로 읽는다(RapidOCR) | 80 | pdf_scan 72, pdf_text 5, hwp3 1, pdf_broken 1, hwp5 1 | ocr.ocr_pdf, lines.lines_from_ocr |
| `plain_text` | 평문 | 인코딩만 풀어 줄로 나눈다(레이아웃 없음) | 19 | txt 19 | lines.lines_from_text |
| `xls_table` | 엑셀 표 | 한 행을 한 줄로, 칸 사이는 탭 | 1 | xls 1 | lines.lines_from_xls |

- 변환 방법(파일 기준, cache/convert_log.jsonl): doc fallback_antiword 4, doc hancom 52, docx hancom 2, hwp3 fallback_hwp3_text 1, hwp3 hancom 140, hwp5 fallback_hwp5_bodytext 3, hwp5 hancom 277, rtf hancom 2
- OCR 사유(문서 기준): 스캔 PDF 72, 원본 PDF 글자층이 깨짐 5, 한컴 변환 PDF 글자층이 깨짐 2, 글자층을 읽을 수 없는 PDF 1
- 두 쪽 펼침(한 장에 대본 두 쪽을 나란히 인쇄) 문서 31개: 가운데 빈 띠에서 왼쪽·오른쪽 쪽으로 나눈다(`lines._gutter`, OCR 은 `ocr.split_two_up`). 경로별 hancom_pdf 30, ocr 1
- 공백 복원(`types.csv` 의 `spacing`, lines.jsonl 머리 행 `made.spacing`): 한컴 PDF 글자층에 띄어쓰기·이름 뒤 탭이 빠진 파일(공백 손실 0.15 초과)은 줄 글의 공백·탭을 원본 문서 글에 맞춘다(`lines.restore_spacing`). `source` 10문서 (fm54367 0.240, fm54441 0.519, fm54519 0.227, fm55036 0.396, fm55099 0.401, fm55455 0.350, fm55655 0.405, fm55664 0.309, fm55844 0.379, fm416766 0.369), `skipped`(원본 글을 못 읽어 견주지 못함) 2문서 (fm55630, fm56026). 경로별 hancom_pdf 10
- 줄을 만든 lines 판(머리 행 `made.version`, 공백 손실은 0.4.0 부터 잰다): 0.4.1 684

## 4. 내용 종류·언어·판본

- 내용 종류(`note`, 원본을 직접 보고 정한 `catalog.POST_OVERRIDES`·`FILE_OVERRIDES`): 영화 대본이 아닌 문서 8개: 자막 받아쓰기(시각 목록) 2, 드라마 한 회 2, 시놉시스 1, 녹음 대본(엑셀) 1, 영화 받아쓰기 1, 드라마 여러 회 1
- 대본 언어(`script_lang`, `review.detect_lang` 이 글자 비율과 불용어로 판정): ko 404, en 274, mixed 4, es 1, fr 1
- 번역 대본(`is_translation`, 외국영화의 한국어 대본 또는 한국영화의 외국어 대본): 13개
- 판본(`version_label`, 제목·첨부 이름의 단서 `catalog.VERSION_CUES`): unknown 644, final 14, bound 10, revised 5, translation 5, draft 4, shooting 2
- 영화 638편, 영화마다 대표 대본(`is_primary`) 하나: 대표 638개, 같은 영화의 다른 판 46개
- 분류 폴더별 언어: 한국영화 ko 392, 한국영화 mixed 1, 한국영화 en 1, 외국영화 en 273, 외국영화 ko 12, 외국영화 mixed 3, 외국영화 es 1, 외국영화 fr 1

## 5. 대본 꼴

### 5.1 대사 표기(이름을 적는 법)

문서의 대표 표기는 script.json 대사 요소 가운데 이름이 적힌 것(앞 대사에서 이어받은 화자 제외)의 첫머리를 보고 가장 많은 표기로 정했다(이름 적힌 대사가 5개 미만이면 `none`). 문서별 값은 `types.csv` 의 `dialogue_form`, 그 표기의 비율은 `form_share`.

| 값 | 표기 | 모양 | 보기(지어낸 글) | 문서 | 한국영화 | 외국영화 | 표기 비율 중앙값 | 확인 필요 문서 | 파서 분기 |
|---|---|---|---|---|---|---|---|---|---|
| `own_line` | 이름 줄 | 이름만 한 줄에 적고 대사는 다음 줄부터 | `철수` 다음 줄 `어디 가?` | 371 | 107 | 264 | 0.99 | 10 | DocParser._vocab (block_mode·flat_mode), _collapsed_cues |
| `tab` | 이름⇥대사 | 이름과 대사 사이가 탭(칸 띄움) | `철수⇥어디 가?` | 185 | 179 | 6 | 0.98 | 6 | DocParser._inline_split_raw (tab) |
| `colon` | 이름 : 대사 | 이름 뒤 쌍점(쌍반점·빗금도 같은 표기로 봄) | `철수 : 어디 가?`, `철수/ 어디 가?` | 87 | 73 | 14 | 0.96 | 2 | DocParser._inline_split_raw (colon) |
| `space1` | 이름 대사 | 이름 뒤 공백 하나뿐 | `철수 어디 가?` | 26 | 25 | 1 | 0.92 | 1 | DocParser._vocab (space_vocab: 줄 첫 낱말 빈도) |
| `space2` | 이름  대사 | 이름 뒤 공백 둘 이상 | `철수   어디 가?` | 5 | 4 | 1 | 0.92 | 0 | DocParser._inline_split_raw (space2) |
| `quote` | 이름"대사" | 이름 뒤(공백 없이 또는 하나 두고) 따옴표 | `철수"어디 가?"` | 3 | 2 | 1 | 0.83 | 0 | DocParser._inline_split_raw (quote) |
| `bracket` | [이름] 대사 | 이름을 대괄호로 쌈 | `[철수] 어디 가?` | 2 | 2 | 0 | 0.99 | 0 | DocParser._bracketed |
| `angle` | <이름> 대사 | 이름을 꺾쇠로 쌈(대사가 다음 줄에 와도) | `<철수> 어디 가?` | 1 | 1 | 0 | 1.00 | 0 | DocParser._inline_split_raw (angle) |
| `none` | 대사 없음 | 이름이 적힌 대사가 5개 미만(시놉시스, 자막 받아쓰기 등) |  | 4 | 1 | 3 | - | 4 |  |

- `own_line` 371개 가운데 들여쓰기 없는 이름 줄(`flat_names`) 27개, 대사가 지문 여백에 놓인 대문자 이름 줄(`collapsed_cues`) 6개, 나머지는 들여쓰거나 가운데 놓은 이름 줄
- 이름 꾸밈(`name_deco`, 이름 적힌 대사의 30% 이상): 글자 사이를 띄운 이름(`철 수`) 37, 글머리표가 붙은 이름(`*철수`) 3
- 한 문서에 표기가 섞인 문서(대표 표기 비율 0.8 미만) 74개: own_line 35, tab 22, colon 12, space1 3, space2 2

### 5.2 파서가 켠 특수 분기

규칙 파서가 문서마다 `quality.notes` 에 적은 분기다(한 문서에 여럿 가능). 문서별 값은 `types.csv` 의 `parser_branches`.

| 값 | 뜻 | 문서 | 한국영화 | 외국영화 | 코드 |
|---|---|---|---|---|---|
| `flat_names` | 들여쓰기 없는 이름 줄(빈 줄 뒤 이름, 빈 줄 앞까지 대사) | 27 | 14 | 13 | DocParser._vocab flat_mode |
| `space1_vocab` | 이름 뒤 공백 하나: 줄 첫 낱말 빈도로 이름을 찾음 | 34 | 23 | 11 | DocParser._vocab space_vocab |
| `letterspaced_names` | 글자 사이를 띄운 이름 | 41 | 41 | 0 | DocParser._letterspaced |
| `collapsed_cues` | 대사가 지문 여백에 놓인 대문자 이름 줄(영어 대본) | 6 | 0 | 6 | DocParser._collapsed_cues |
| `bullet_names` | 글머리표가 붙은 이름 | 3 | 3 | 0 | DocParser._candidates |
| `bracket_names` | 대괄호 이름 | 2 | 2 | 0 | DocParser._bracketed |
| `voice_cue` | '소리 : 대사'(화자 없는 화면 밖 소리) | 10 | 10 | 0 | DocParser._candidates |
| `group_cue` | '우리 : 대사'(무리 화자) | 1 | 1 | 0 | DocParser._candidates |
| `shifted_pages` | 쪽마다 여백이 달라 쪽 밀림을 맞춤 | 135 | 79 | 56 | parse_layout.analyze_x |
| `scene_indented_series` | 들여쓰거나 가운데 놓인 번호 씬 머리 | 7 | 7 | 0 | DocParser._accept_headings |
| `scene_upper_slug` | INT./EXT.·번호 없이 대문자 짧은 줄이 씬 머리 | 5 | 0 | 5 | DocParser._accept_headings |
| `scene_restart` | 씬 번호가 1로 다시 시작(앞 번호는 인물표·설정) | 8 | 7 | 1 | DocParser._accept_headings |
| `front_dialogue` | 첫 씬 머리 앞에 대사가 있음 | 66 | 36 | 30 | DocParser._front_end |
| `front_none` | 앞머리 없이 첫 줄부터 본문 | 21 | 7 | 14 | DocParser._front_end |
| `speaker_inherit` | 이름 줄 없이 이어진 대사는 앞 화자를 이어받음(speaker_inferred) | 104 | 38 | 66 | DocParser._build |
| `vo_resolved` | (V.O.) 를 화면 밖 소리와 내레이션으로 가름 | 249 | 52 | 197 | DocParser._resolve_vo |
| `n_ambiguous` | (N) 을 밤과 내레이션으로 가름 | 31 | 30 | 1 | DocParser._dialogue_fields |
| `subtitle_file` | 자막 파일 꼴(시각 줄·자막 번호를 뺌) | 1 | 0 | 1 | DocParser.parse |
| `scene_colon_heads` | 쌍점이 든 번호 씬 머리(`12. 장소 : 설명 / 밤`)를 받아들인 씬 머리와 같은 칸에서 빈 번호를 차례대로 메울 때 받음 | 8 | 8 | 0 | DocParser._colon_heads |
| `paren_names` | 괄호에 든 이름 `(이름)⇥대사`: 이름표에 있는 인물의 화면 밖 목소리(off) | 20 | 20 | 0 | DocParser._paren_names |
| `wordless_turns` | 말 없이 점·괄호만 있는 한 줄 차례(`이름⇥.......`)를 이름 줄이 아닌 한 줄 대사로 봄 | 55 | 54 | 1 | DocParser._vocab, _wordless |
| `tight_cont` | 지문 앞에 늘 빈 줄을 두는 문서: 대사 바로 아래 붙여 쓴 들여쓴 줄은 앞 대사의 이어짐 | 46 | 44 | 2 | DocParser._cont (tight_cont) |

### 5.3 씬 머리

- 씬 머리를 하나도 못 찾은 문서 31개(대표 표기별 colon 11, tab 7, own_line 6, none 4, space2 2, space1 1), 문서당 씬 수 중앙값 119

### 5.4 유형별 확인 필요(qa/needs_review.csv)

| 대표 표기 | 문서 | `asr_low_conf` | `extract_broken` | `other` | `parse_low_dialogue` | `speaker_missing` |
|---|---|---|---|---|---|---|
| `own_line` | 371 | 0 | 1 | 5 | 1 | 3 |
| `tab` | 185 | 1 | 1 | 2 | 1 | 1 |
| `colon` | 87 | 0 | 0 | 2 | 0 | 0 |
| `space1` | 26 | 0 | 0 | 0 | 1 | 0 |
| `space2` | 5 | 0 | 0 | 0 | 0 | 0 |
| `quote` | 3 | 0 | 0 | 0 | 0 | 0 |
| `bracket` | 2 | 0 | 0 | 0 | 0 | 0 |
| `angle` | 1 | 0 | 0 | 0 | 0 | 0 |
| `none` | 4 | 0 | 0 | 0 | 4 | 0 |

| 글자 경로 | 문서 | `asr_low_conf` | `extract_broken` | `other` | `parse_low_dialogue` | `speaker_missing` |
|---|---|---|---|---|---|---|
| `hancom_pdf` | 471 | 1 | 0 | 2 | 5 | 4 |
| `pdf_textlayer` | 105 | 0 | 0 | 0 | 0 | 0 |
| `ocr` | 80 | 0 | 1 | 0 | 1 | 0 |
| `plain_text` | 19 | 0 | 0 | 0 | 1 | 0 |
| `fallback_text` | 8 | 0 | 1 | 7 | 0 | 0 |
| `xls_table` | 1 | 0 | 0 | 0 | 0 | 0 |

- 유형을 센 판(`parser`): sink-parse 684. docs/ 에서 지금 LLM 판인 문서 29개는 보관된 규칙 판으로 셌다

## 6. types.csv 칸

| 칸 | 뜻 |
|---|---|
| `doc_id` | 문서 ID(catalog.csv 와 같음) |
| `category` | 분류 폴더(ko_film, foreign_film) |
| `title` | 정리한 제목(title_norm) |
| `real_format` | 파일 형식(1절) |
| `files` | 파일 수 |
| `route` | 글자를 얻은 경로(3절, 파일마다 다르면 + 로 이음) |
| `convert_method` | 변환 방법(convert_log 의 method) |
| `ocr_why` | OCR 한 까닭 |
| `two_up` | 두 쪽 펼침 |
| `script_lang` | 대본 언어 |
| `is_translation` | 번역 대본 |
| `version_label` | 판본 |
| `content` | 영화 대본이 아닌 내용 종류 |
| `is_primary` | 영화의 대표 대본 |
| `parser` | 유형을 센 판을 만든 파서(`--rules` 면 늘 규칙 파서) |
| `lines` | 줄 수 |
| `scenes` | 씬 수 |
| `dialogues` | 대사 요소 수 |
| `dialogue_form` | 대표 대사 표기(5.1절) |
| `form_share` | 이름 적힌 대사 중 대표 표기의 비율 |
| `name_deco` | 이름 꾸밈(letterspaced, bullet) |
| `parser_branches` | 파서가 켠 특수 분기(5.2절, 세로선으로 이음) |
| `score` | 유형을 센 판의 품질 점수(quality.score) |
| `needs_review` | 확인 필요 문제(qa/needs_review.csv 그대로, 세로선으로 이음) |
| `spacing` | 공백 복원(3절): `source` 한컴 PDF 줄의 공백·탭을 원본 문서 글로 맞춤, `skipped` 원본 글을 읽지 못해 공백 손실을 견주지 못함, 빈 값은 기록 없음(견줬고 문턱 아래, 한컴 PDF 경로가 아님, 또는 공백 손실을 재기 전의 lines 판) |
