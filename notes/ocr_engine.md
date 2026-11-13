# OCR 엔진 선택과 글자층 검사 (US-005, 2026-11-13)

## 결론
- 엔진: **RapidOCR 3.10.0 (onnxruntime 1.31.1, CPU)** 로 PaddleOCR 계열 모델을 돌린다.
  - 한국어: PP-OCRv5 mobile 검출 + `korean_PP-OCRv5_rec_mobile` 인식
  - 영어(로마자 쓰는 유럽어 포함): PP-OCRv6 small 검출·인식
- 쪽 그림: 300dpi 회색. PyMuPDF 로 그리고, 예외가 나거나 JBIG2 경고와 함께 망가진 그림(한 색, 절반 넘게 검정)이 나오면 pdftoppm 으로 다시 그린다. 둘 다 안 되면 그 쪽은 `render: failed` 로 남긴다.
  - pdftoppm 은 `C:\Program Files\Git\mingw64\bin` 에 없다(그곳엔 pdftotext 만). WinGet Poppler 25.07.0(`%LOCALAPPDATA%\Microsoft\WinGet\Packages\oschwartz10612.Poppler_…\poppler-25.07.0\Library\bin`)을 쓴다.
  - No Country [56897]: 123쪽 중 PyMuPDF 109쪽, pdftoppm 으로 살린 쪽 3, 둘 다 실패 11쪽(2, 12, 16, 52, 57, 61, 70, 73, 95, 96, 122). 실패 쪽은 JBIG2 기호 사전 자체가 깨져 pdftoppm 도 검정·흰 쪽만 낸다.
- 쪽 방향: 쪽 평균 conf 가 0.80 미만이거나 잉크는 있는데 글자가 거의 안 읽히면 0·90·270·180도로 모두 읽고, 글줄다운 상자(높이 0.5인치 이하)의 conf×글자 수 합이 가장 큰 방향을 고른다. 고른 방향은 문서 안에서 기억해 다음 쪽부터 바로 쓴다.
- 두 쪽 펼침: 돌린 그림이 가로로 길고(너비 > 1.15×높이) 가운데 35~65% 안에 빈 세로 띠가 있으면 두 쪽으로 나눠 따로 읽는다(`part` 1·2).
- 언어: catalog `script_lang` 이 정해져 있으면 그 값, 아니면 앞쪽을 한국어 모델로 읽어 한글 비율(15% 이상이면 ko)로 고른다. 표지만 보고 정하지 않도록 글자 300개 또는 글 있는 쪽 4개를 모아서 판단한다.

## 엔진 비교 (같은 5쪽, 300dpi)
표본: 한국어 스캔 3쪽(변호인 [3710018] 5·30쪽, 검은사제들 [5488034] 20쪽), 영어 스캔 2쪽(The Dark Knight [56924] 10쪽, Inside Man [56795] 15쪽).
쪽 그림을 직접 보고 손으로 옮겨 적은 기준 줄: 한국어 54줄(1,227자), 영어 69줄(1,888자). 확실하게 읽히는 줄만 옮겼다.

- 글자 오류율(CER) = 기준 줄마다 가장 잘 맞는 OCR 줄(또는 덩어리)과의 편집 거리 합 / 기준 글자 수. "공백 뺌"은 띄어쓰기를 무시, "공백 포함"은 띄어쓰기 오류도 셈. 따옴표·말줄임표 모양은 통일해서 셈.
- 들여쓰기: 기준 줄이 시작하는 x(그림에서 잰 화자 칸·대사 칸·지문 칸 위치)와 OCR 상자 시작 x 가 ±30px(약 2.5mm) 안이면 맞음.
- 쪽당 시간: 엔진 데운 뒤 한 쪽 인식 시간(그림 그리기 제외), CPU 12 논리 코어, 다른 일 없음.

### 한국어 3쪽
| 엔진 | 모델 | 쪽당 초 | CER 공백 뺌 | CER 공백 포함 | 들여쓰기 | conf |
|---|---|---|---|---|---|---|
| **RapidOCR (채택)** | v5 mobile 검출 + korean v5 mobile 인식 | **1.40** | **3.4%** | 5.6% | 52/54 (*) | 있음 |
| RapidOCR | 위와 같음, 검출 입력 최대 4000px | 2.43 | 3.4% | 5.6% | 52/54 | 있음 |
| RapidOCR | v5 server 검출 + korean v5 mobile | 4.36 | 3.4% | 5.4% | 52/54 | 있음 |
| RapidOCR | v6 small 검출 + korean v5 mobile | 1.53 | 3.6% | 5.6% | 53/54 | 있음 |
| PaddleOCR 3.7.0 (paddlepaddle 3.3.1 CPU) | v5 mobile 검출 + korean v5 mobile | 18.74 | 3.7% | 5.5% | 53/54 | 있음 |
| Windows.Media.Ocr (winrt 4.1) | 설치된 인식기: ko 만 | 0.63 | 18.3% | 15.5% | 52/54 | 없음 |

(*) 빗나간 2줄도 줄의 `segs` 를 직접 보니 대사 칸 덩어리가 x=731·750(기대 736·753)에서 시작한다. 채점 스크립트가 덩어리 대신 줄 전체를 고른 탓이다.

### 영어 2쪽
| 엔진 | 모델 | 쪽당 초 | CER 공백 뺌 | CER 공백 포함 | 들여쓰기 |
|---|---|---|---|---|---|
| **RapidOCR (채택)** | v6 small 검출·인식(en) | **1.29** | **0.6%** | **0.6%** | 69/69 |
| RapidOCR | v5 mobile 검출 + en v5 mobile | 1.32 | 0.7% | 1.1% | 69/69 |
| RapidOCR | v5 server 검출 + en v5 mobile | 4.56 | 0.6% | 1.0% | 69/69 |
| RapidOCR | v5 mobile 검출 + latin v5 mobile | 1.34 | 1.7% | 1.6% | 69/69 |
| RapidOCR | v5 mobile 검출 + korean v5 mobile(한 모델로 둘 다) | 1.45 | 2.6% | 2.6% | 69/69 |
| PaddleOCR 3.7.0 | v5 mobile 검출 + en v5 mobile | 21.63 | 1.5% | 1.8% | 69/69 |
| Windows.Media.Ocr | ko 인식기(영어 인식기 없음) | 0.58 | 16.3% | 17.6% | 64/69 |

### 고른 이유
1. 정확도: 한국어는 PP-OCRv5 korean 인식 모델이 RapidOCR·PaddleOCR 어느 쪽으로 돌려도 CER 3.4~3.7% 로 같다(같은 모델). 검출 모델을 server 로 키워도 나아지지 않았다. 영어는 PP-OCRv6 small 이 가장 낫다(0.6%).
2. 속도: 같은 모델인데 RapidOCR(onnxruntime) 가 1.3~1.4초/쪽, PaddleOCR 3.7 + paddlepaddle 3.3.1 CPU 는 18~22초/쪽. paddlepaddle 3.3.1 CPU 는 oneDNN 을 켜면 `NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<pir::DoubleAttribute>]` 로 죽어서 `enable_mkldnn=False` 로만 돌았다.
3. 줄 상자: RapidOCR 검출 상자는 화자 칸과 대사 칸을 따로 잡는다(예: `김광삼` x=490, `부동산 등기를?…` x=750). 이것을 `group_lines` 가 같은 줄로 묶고 덩어리별 상자를 `segs` 에 남겨 들여쓰기가 살아 있다.
4. Windows OCR 은 이 PC 에 한국어 인식기만 있고(영어 인식기 설치는 관리자 권한 필요), 기울임 한글 대사에서 크게 틀렸고(변호인 5쪽 CER 37%), 신뢰도(conf)를 주지 않아 `ocr_low` 판단을 할 수 없다.
5. 설치: `.venv` 에 `rapidocr==3.10.0` 하나만 더했다(새 의존성 opencv-python-headless 5.0.0.93, shapely, pyclipper, omegaconf 등). 기존 numpy 2.4.6·onnxruntime 1.31.1 은 그대로다. paddlepaddle 은 `.venv` 에 넣지 않았다(비교는 따로 만든 가상환경에서).

### GPU
- 시험 때 RTX 4070 SUPER 12GB 의 VRAM 이 11.3GB 쓰이고 있었다(`D:\AI\llama.cpp\llama-server.exe`). 남은 1GB 로는 GPU 추론이 불안하고, 같은 venv 의 onnxruntime(CPU)을 onnxruntime-gpu 로 바꾸면 다른 모듈(pymupdf_layout 이 onnxruntime 사용)에 영향이 있을 수 있어 CPU 로 정했다.
- `paddlepaddle-gpu==3.3.1`(cu126) Windows 휠은 있음을 확인했으나 설치하지 않았다.
- 병렬: 한 프로세스(스레드 전부) 1.49초/쪽, 2×6 1.70, 3×4 1.46, 4×3 1.66, 6×2 1.58 (12쪽 벽시계 기준). 프로세스를 늘려도 처리량이 늘지 않아 `--workers` 기본 1.

## 스캔 PDF 전체 표본 조사 (74개 파일 × 2쪽: 3쪽과 가운데 쪽)
- 원본 폴더 기준 글자층 없는 PDF 72개 + 열리지 않는 PDF 1개(No Country) = 73개, PyMuPDF 로 센 쪽수 6,814쪽(스캔 6,691 + No Country 123).
- catalog 기준 OCR 대상 스캔 파일은 74개(압축에서 나온 `extracted/56868/LITTLE_MISS_SUNSHINE.pdf`, `extracted/773767/황해.pdf` 추가, 중복 `fm55871` 제외), 6,956쪽.
- 언어 판정: ko 48, en 26 (게시판 구분과 모두 일치).
- 표본 평균 conf: 한국어 0.923, 영어 0.990. 표본에서 0.85 아래인 문서 없음.
- **거꾸로(180도) 스캔된 한국어 문서 7개**: 기다리다 미쳐 [55571], 피터팬의 공식 [55599], 러브토크 [55602], B형 남자 친구 [55623], 내생에 최악의 남자 [55627], 오래된 정원 [55758], 아내가 결혼햇다 [55813]. 방향 찾기로 모두 180도로 돌려 읽었다(표본 conf 0.89~0.97).
- **옆으로 눕힌 두 쪽 펼침 1개**: 좋은 놈 나쁜 놈 이상한 놈 [55777]. 90도로 돌리고 두 쪽으로 나눠 읽는다(표본 conf 0.875~0.91). 방향 처리를 넣기 전에는 이 문서 전체 conf 0.612 였다.

## 걸린 시간과 전체 예상
| 문서 | 쪽 | 초 | 초/쪽 |
|---|---|---|---|
| Good Night, And Good Luck [56787] (영어 스캔) | 88 | 132.6 | 1.51 |
| 감시자들 [5691757] 1번 (한국어 스캔) | 40 | 61.4 | 1.54 |
| 감시자들 [5691757] 2번 | 41 | 64.1 | 1.56 |
| No Country [56897] (JBIG2, 10쪽 실패) | 123 | 174.6 | 1.42 |
| 애자 [416338] (글자층 깨짐, 시험과 겹쳐 돌아감) | 80 | 139.5 | 1.74 |
(`python -I -m sinkprep ocr --docs fm55777,fm5691757,fm56787,fm416338,fm56897`, 그림 그리기·쓰기 포함, 6개 파일 662초)

- OCR 대상 전체(catalog + 글자층 검사, `ocr.plan`): 86개 파일 8,271쪽 = 스캔 6,833 + 열리지 않는 PDF 123 + 원본 글자층 깨짐 10개 1,223 + 한컴 변환 PDF 글자층 깨짐 2개 92.
- 쪽당 약 1.55초로 잡으면 **전체 약 3.6시간**(스캔·열리지 않는 PDF 6,956쪽만이면 약 3.0시간). 거꾸로·눕힌 문서는 첫 쪽만 네 방향을 읽어 문서당 수 초 더 든다.

## 결과 파일 `cache/ocr/{doc_id}/{k}.json`
DESIGN 형식에 칸을 더했다.
```json
{"schema":"sink.ocr/1","engine":"rapidocr","engine_version":"rapidocr 3.10.0; onnxruntime 1.31.1; det …; rec …",
 "source":"filmmakers_scripts/…pdf","source_sha1":"…","dpi":300,"lang":"ko","lang_how":"hint|guess",
 "made":{"stage":"ocr","version":"0.1.0"},"seconds":123.4,"failed_pages":[2,12],"conf_mean":0.91,
 "why":"pdf_scan|pdf_broken|textlayer_broken|converted_textlayer_broken","textlayer":{…검사 결과…},
 "pages":[{"page":1,"width":2480,"height":3508,"render":"pymupdf|pdftoppm|failed","warn":"…",
   "lines":[{"text":"김광삼     부동산 등기를?","box":[490,439,1381,497],"conf":0.937,
             "segs":[{"text":"김광삼","box":[490,439,609,497],"conf":0.999},
                     {"text":"부동산 등기를?","box":[750,442,1381,493],"conf":0.929}]}]}]}
```
- `text` 는 덩어리 사이를 간격만큼(글자 폭 어림, 최소 2칸) 공백으로 이은 것. `segs` 는 덩어리가 둘 이상일 때만 있다.
- 그림을 못 만든 쪽도 `width/height`(PDF 쪽 크기 × dpi/72)를 넣고 `lines: []`.
- 돌려 읽은 쪽: `"rotate": 90|180|270`(시계 방향). `box`·`width`·`height` 는 돌린 그림 기준.
- 두 쪽 펼침: 같은 `page` 번호로 항목 둘, `"part": 1|2, "parts": 2, "crop": [x0,y0,x1,y1]`(돌린 그림 안 범위). `box`·`width` 는 그 부분 기준이라 들여쓰기 비율이 쪽 하나와 같다.
- 문제 기록: 문서 평균 conf(글자 수 가중) < 0.85 면 `ocr_low`, 그림 못 만든 쪽이 있으면 `extract_broken`(score = 성공 쪽 비율).

## 글자층 검사 (textlayer.check)
원본 PDF 중 글자층 있는 112개를 모두 재서 기준을 정했다(실행 14초).

| 지표 | 깨짐 기준 | 정상 PDF 최댓값 | 깨진 예 |
|---|---|---|---|
| 한글 낱말 평균 음절 `ko_token_len` | > 5.0 | 2.88 (가비) | 베를린 10.53, 구타유발자들 9.07, 애자 8.8 |
| 구두점만 있는 줄 `punct_line` (`*` 줄 뺌) | > 0.10 | 0.024 (티끌모아 로맨스) | 더킹 0.489, 구타유발자들 0.410, 애자 0.245 |
| 한 글자 줄 `single_line` | > 0.15 | 0.018 (gone girl) | 더킹 0.475, 구타유발자들 0.338 |
| 앞 줄과 같은 줄 `dup_line` | > 0.15 | 0.099 (Bucket List) | Speed 0.500, Rear Window 0.475, Vertigo 0.463, 애자 0.253 |
| 같은 자리 겹친 글자 `overlap_char` | > 0.15 | 0.026 (가여운 것을 대본) | Speed·Rear Window·Vertigo 0.50, 애자 0.29 |
| 이상한 문자 `odd_char` (U+2028/2029, U+A789, U+FFFD, 사용자 영역, 제어문자) | > 0.2% | 0.005% (대관람차) | Past Lives 22.5% (U+2029, U+A789) |
| 라틴-1 보충 글자 `mojibake` | > 10% | 2.7% (추락의 해부, 프랑스어) | (없음) |
| 두 칸 동시 대사가 한 줄로 섞인 곳 `dual_merged` | >= 3 | 1 (Collateral) | Eternal Sunshine 28, 킬빌 5 |
| 쪽당 글자 `chars_per_page` | < 80 | 361.9 (변산) | (없음) |

- 처음에는 영어 대본 7개가 구두점 줄로 걸렸는데, 모두 여백 수정 표시 `*`(·`**`) 줄이었다. 이 줄을 빼고 나서 정상 PDF 는 하나도 걸리지 않는다.
- 깨짐 판정 12개: 알려진 10개(더킹, 애자, 구타유발자들 두 개, 베를린, Past Lives, Speed, Rear Window, Vertigo, Eternal Sunshine [914411]) + Eternal Sunshine [56671] 1번 파일(914411 과 sha1 같음) + 킬빌 [56013](분할 화면 두 칸 지문·대사가 한 줄로 섞임 5곳).
- 한컴 변환 PDF 17개(시험 시점 `cache/pdf`) 중 2개(fm54352, fm55985)가 `ko_spacing_lost` 로 걸렸다. 쪽 그림에는 띄어쓰기가 있는데 글자층에 공백 글자가 없다(배분 정렬로 간격만 벌어짐).
