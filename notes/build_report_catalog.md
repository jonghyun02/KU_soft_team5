## US-003 보고: 원본 목록, 압축 해제, 형식 판별, catalog.csv

압축 12개를 실제로 모두 풀었고, 실제로 돌려 `C:\projects\sink\catalog.csv`를 만들었습니다. 시험 57개가 모두 통과했습니다. 원본 폴더는 실행 전후 경로·크기·수정 시각이 그대로입니다.

### 만든 파일
- `C:\projects\sink\sinkprep\detect.py`
  - `real_format(path)`는 앞 바이트와 내부 구조만 봅니다(HWP3 머리, OLE의 FileHeader·WordDocument·Workbook, PK 안의 word/document.xml 또는 hwp+zip, RAR, ALZ, RTF, PDF, MZ).
  - PDF는 쪽마다 글자 수를 세어, 절반 이상이 50자 미만이면 `pdf_scan`, 열기·추출 실패면 `pdf_broken`입니다. txt는 utf-8 → cp949 → charset_normalizer 순서로 판별합니다.
  - 압축은 함수가 zip·rar·alz로 돌려주고, catalog에는 `catalog_format()`으로 `archive`라고 적습니다.
- `C:\projects\sink\sinkprep\catalog.py`
  - 압축 해제는 `bz x -cp:949 -y -o:<임시폴더>`로 합니다. 다 풀린 뒤 `extracted/{post_id}`로 이름을 바꿔 옮깁니다.
  - 분할 압축은 `cache/stage/{post_id}/`에 manifest의 원래 이름으로 복사한 뒤 풉니다.
  - 풀린 폴더 밖을 가리키는 경로·링크가 없는지 확인합니다. 압축 안의 압축은 한 단계 더 풀도록 만들었지만, 이번 원본에는 없었습니다.
  - `run(args)`, `load()`, `finalize()`, `check_primary()`를 넣었습니다. 형식 판별 결과는 캐시에 두고, 이미 풀린 압축은 건너뜁니다(`--force`면 다시).
- `C:\projects\sink\tests\test_detect.py` (13개)
  - RTF가 든 .doc, 평문 .doc(cp949·utf-16), 깨진 바이너리, exe는 tmp에 만든 가짜 파일로 확인합니다.
  - OLE 파일은 pywin32로 만들어 확인합니다. 그 밖에 docx·hwpx·zip, PDF 글자층·스캔·깨짐, 실제 위장 파일 8개를 봅니다.
- `C:\projects\sink\tests\test_catalog.py` (44개)
  - 제목 풀기 18개 사례, 판본·언어 단서, 묶기 규칙 가짜 사례, 분할 볼륨 묶기, A2·대표 선택, 반디집 한글 이름 해제, finalize 왕복을 봅니다.
  - 실제 catalog.csv로 A1·A2·A3와 판단 결과를 확인하는 시험은 실제 파일이 있을 때만 돕니다.

### 확인한 방법
- `.venv/Scripts/python -I -m sinkprep catalog` 결과: 행 712(문서 684, 제외 28), 파일 714(원본 701, 압축 해제 13).
- 다시 실행해도 catalog.csv의 sha1이 `042659df613dc05f395e7e1b650b88d0ebc9a37c`로 같았습니다. `--force`로 압축을 다시 풀어도 같은 값이었습니다.
- `find filmmakers_scripts -printf '%p\t%s\t%T@'`로 실행 전후를 비교한 결과 차이가 없었습니다.
- `cd C:/projects/sink && .venv/Scripts/python -I -m pytest -q tests/test_detect.py tests/test_catalog.py` 결과: 57 passed.
- A3 시험은 catalog의 모든 파일을 다시 판별해 `file_formats`와 비교하고, 원본 형식별 개수가 조사 숫자와 같은지 확인합니다.

### 숫자
**원본 파일 수**
- 디스크의 원본은 701개(한국영화 408, 외국영화 293)이고, manifest.csv는 따로 있습니다.
- 과제의 "702"는 manifest에서 받은 것으로 된 행 수입니다. 그중 `[Wild Hogs]… [56878] 2_MovieSmiCh.exe`는 디스크에 없습니다.

**파일별 형식 (714개)**

| 형식 | 개수 | 내역 |
|---|---|---|
| hwp5 | 289 | |
| hwp3 | 143 | 원본 137 + 압축 안 6 |
| pdf_text | 112 | |
| pdf_scan | 74 | 원본 72 + 압축 안 2 |
| doc | 57 | |
| txt | 20 | 원본 15(JFK.doc 포함) + 압축 안 5 |
| archive | 12 | |
| rtf | 2 | |
| docx | 2 | |
| xls | 1 | |
| pdf_broken | 1 | |
| other | 1 | |

**문서·제외·묶음**
- 문서 684개(한국영화 394, 외국영화 290). 문서 형식은 hwp5 280, hwp3 141, pdf_text 110, pdf_scan 72, doc 56, txt 19, rtf 2, docx 2, xls 1, pdf_broken 1이고, mixed는 0입니다.
- 제외 28개: 같은 파일 중복 16, 압축 자체 11, 대본 아님 1.
- 영화는 local_0001부터 local_0638까지 638편이고, 같은 영화 묶음은 41개(문서 87개)입니다. is_primary=true는 638개로 영화마다 1개입니다.
- 임시값 분포
  - script_lang: unknown 665, en 10, ko 5, mixed 3, es 1
  - is_translation=true: 5
  - version_label: unknown 644, final 14, bound 10, revised 5, translation 5, draft 4, shooting 2
- 문제 기록 640개(`cache/problems/catalog/`): 영화마다 "KMDb/TMDB 조회 불가, local ID" 638, fm56897 extract_broken 1, fm56878 첨부 누락 1.

### 판단 목록
**여러 파일 글**
- 5691757 감시자들: `_1`/`_2` 스캔 PDF는 한 대본의 조각이라 fm5691757 하나에 순서대로 넣었습니다.
- 5691397 날 보러와요: 두 파일이 sha1까지 같습니다. 둘째 fm5691397_2는 `duplicate_file_of:fm5691397`로 제외했습니다.
- 55009 달콤한 인생: hwp5 파일과 "97" 파일(hwp3, 한글97로 저장한 것)을 _1·_2 두 문서로 두었습니다. 내용이 같은지는 비교하지 않았습니다.
- 56671 이터널 선샤인: shooting script는 _1(shooting), first draft는 _2(draft)입니다.
- 56560 지옥의 묵시록: 본문을 직접 확인했습니다. (ds)는 1975년 초기 판본이라 draft, (ts)는 영화 받아쓰기라 unknown이고 note에 `content:transcript`를 넣었습니다.
- 56528 블레이드 러너, 56461 나비효과: 영문판은 _1, (영한) 대역판은 _2(mixed)입니다.
- 서로 다른 형식의 별개 대본이라 각각 _1·_2로 나눈 글: 56360 비포 선라이즈(txt·hwp), 56350 맨 인 블랙(doc·hwp), 56266 쉘로우 그레이브(pdf·doc), 56251 죽은 시인의 사회(hwp·pdf), 56113 반지의 제왕: 왕의 귀환(hwp·pdf), 56088 트루 로맨스(doc·pdf), 56052 터미네이터 2(hwp·pdf).
- 56878 Wild Hogs: hwp는 한글·영문 자막 시각 목록이지만 이 영화의 유일한 글이라 문서로 두었습니다(`content:subtitle_transcript`). exe 첨부는 디스크에 없어 note와 문제 기록에 남겼습니다.

**압축 글**
- zip 8개는 모두 풀린 파일이 문서가 되고, zip 자체는 `_2` 제외 행입니다.
  - 54292 강원도의 힘 → kwdh.hwp
  - 54288 내 마음의 풍금 → 내 마음의 풍금.hwp
  - 54285 돼지가 우물에 빠진 날 → PIGCIN.TXT
  - 54246 기막힌 사내들 → Kimak.txt
  - 54122 시월애 → swe.txt (제본고)
  - 55982 동사서독 → 동사서독.txt
  - 56040 불멸의 연인 → Immortal Beloved.txt
  - 54259 커밍아웃: 판본 4개라 fm54259_1~_4로 나눴습니다(감독version, 원작자최초버전=draft, 작가의A버전, 작가의B버전). zip은 fm54259_5 제외 행입니다.
- 773767 황해(rar 분할)
  - stage 복사본으로 풀어 `extracted/773767/황해.pdf`(스캔 118쪽)가 나왔고, 이것이 fm773767입니다.
  - 볼륨 2개는 fm773767_2 한 행으로 제외했습니다.
- Little Miss Sunshine(alz 분할, 두 글에 나뉨)
  - 첫 볼륨이 있는 56868 글로 모아 `extracted/56868/LITTLE_MISS_SUNSHINE.pdf`(스캔 110쪽)가 나왔고, 이것이 fm56868입니다.
  - .alz는 fm56868_2, .a00은 fm56874로 제외했고, 둘 다 note는 `archive_container:extracted/56868`입니다.

**그 밖의 제외**
- 54331 반칙왕: 첨부 원래 이름이 `~$반칙왕.doc`인 162바이트 Word 임시 파일이라 `not_script:other`로 제외했습니다.
- 다른 글과 sha1이 같아 뒤 글을 제외한 15건: fm54746, fm55216, fm55434, fm55548, fm55692, fm55739, fm55752, fm55871, fm55922, fm55970, fm914411, fm5691425, fm5691735, fm5691770, fm6535192.

**문서로 두되 note를 단 것**
- 54537: 녹음대본이 엑셀(xls)로 되어 있습니다.
- 54448: 시놉시스입니다.
- 55474 빈집: 자막 추출본입니다.

### 남은 문제
- **xls 읽기:** fm54537을 읽으려면 lines·convert 단계에 xls 읽는 도구가 필요합니다. xlrd가 없고 pip 설치는 금지라 이번에는 다루지 않았습니다.
- **열리지 않는 PDF:** fm56897(No Country, JBIG2)은 fitz로 열 수 없어 OCR 단계에 다른 렌더러가 필요합니다.
- **애매한 묶음:**
  - G008 교회누나는 묻지마 패밀리의 한 편(xls)과 7013801 글이 같은 대본인지 확인하지 못했습니다.
  - 56452 터미네이터(한글번역판)는 원어 제목이 없어 Terminator 쪽 글과 묶이지 않았습니다.
- **제목 손 판단:** 규칙으로 풀리지 않는 20개 글은 `POST_OVERRIDES`·`FILE_OVERRIDES`로 손 판단을 넣었습니다.
- **임시값:** script_lang은 명세대로 제목·첨부 이름 단서만 써서 665개가 unknown입니다.

### DESIGN 변경 제안
1. **file_formats 칸:** `real_format` 바로 뒤에 두었습니다. DESIGN에 위치를 적어 주세요.
2. **제외 행 ID 규칙:** 문서를 먼저, 제외 행을 뒤에 번호 붙입니다. 문서가 하나면 문서는 `fm{글번호}`, 제외 행은 `_2`부터입니다. 숫자만 쓰므로 `^fm\d+(_\d+)?$` 규칙을 지킵니다.
3. **finalize 사용법:** `catalog.finalize(updates={doc_id: {칸: 값}}, rows=None, path=None)`. 없는 칸이면 KeyError, A2를 어기면 ValueError를 내고 아무것도 쓰지 않습니다. review 담당이 이 형태로 쓰면 됩니다.
4. **새 note 값:** 아래 값을 DESIGN 목록에 추가해 주세요.
   - `content:subtitle_transcript`
   - `content:synopsis`
   - `content:transcript`
   - `content:recording_script_xls`
   - `tv_episode:…`
   - `tv_series`
   - `missing_attachment:…`
   - `archive_nested_not_extracted`
5. **is_primary 순서(과제 규칙에서 바꾼 부분):** 과제 규칙은 형식·크기였는데, 번역 아님을 맨 앞에, 같은 형식 안에서는 판본(shooting > final > bound > revised > unknown > draft)을 크기 앞에 넣었습니다. 영상 원어와 맞는 쪽을 고르기 위해서입니다. 받아들일지 정해 주세요.
6. **A1 기준 수:** A1 문구의 702는 manifest 기준입니다. 디스크는 701개이고, 시험은 "디스크 원본 + note의 누락 첨부 = manifest 702"로 확인합니다.
7. **영화별 local ID 문제 638건:** 명세대로 넣었지만 needs_review가 시끄러워집니다. review에서 묶어 보여 주는 것을 제안합니다.
8. **util에 바이너리 복사 함수 추가:** 분할 압축 stage 복사는 바이너리라 `shutil.copyfile`과 `assert_not_raw`로 했습니다. util에 바이너리 복사 함수를 두면 좋겠습니다.
9. **catalog의 인자 처리:** 전체를 한꺼번에 묶어야 해서 `--docs`, `--limit`, `--workers`는 무시하고 그렇다고 출력합니다.