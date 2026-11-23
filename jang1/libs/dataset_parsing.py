"""arXiv:2309.11093 한-영 병렬 노래 가사(Section/Line) 데이터셋 탐색, 병렬 파서 및 합성 K-pop 코퍼스 모듈."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Sequence, Set, Tuple

from libs.phonetics import normalize_lyric_text
from libs.preprocessing_common import (
    LyricSectionUnit,
    batched_pool_map,
    build_lyric_section_unit,
    fast_scandir_files,
    slice_long_section_unit,
)

logger = logging.getLogger(__name__)


@dataclass
class LyricSplitLayout:
    """단일 데이터 분할(train 또는 val)의 가사 JSON/JSONL/CSV 및 보컬 WAV 디렉터리 묶음입니다."""

    split_name: str
    data_files: List[Path]
    wav_dirs: List[Path]


_SYNTHETIC_KPOP_SONGS: Tuple[Tuple[str, Tuple[Tuple[str, Tuple[Tuple[str, str], ...]], ...]], ...] = (
    (
        "kpop_starlight_01",
        (
            (
                "verse",
                (
                    ("차가운 밤공기 속에 홀로 서 있어", "Standing all alone inside the chilly midnight air"),
                    ("멀리 반짝이는 별을 올려다보네", "Looking up at shining stars so high above me now"),
                    ("어둠이 깊어질수록 선명해지는", "Growing even clearer as the darkness falls around"),
                    ("너의 목소리가 귓가에 들려와", "I can hear your gentle voice inside my lonely heart"),
                ),
            ),
            (
                "prechorus",
                (
                    ("Oh baby 두려워하지 마", "Oh baby never be afraid now"),
                    ("우리의 시간이 시작돼 tonight", "Our shining moment is beginning tonight"),
                ),
            ),
            (
                "chorus",
                (
                    ("빛나는 별빛 아래 너와 나", "Under the shining starlight you and me"),
                    ("영원히 멈추지 않을 이 노래", "Singing this melody forever free"),
                    ("빛나는 별빛 아래 너와 나", "Under the shining starlight you and me"),
                    ("Yeah 날아올라 저 하늘 끝까지", "Yeah flying higher to the open sky"),
                ),
            ),
            (
                "bridge",
                (
                    ("가끔은 흔들리고 길을 잃어도", "Even when we stumble and we lose our only way"),
                    ("잡은 두 손을 절대 놓지 않을게", "I will hold your hands and never let you fade away"),
                ),
            ),
        ),
    ),
    (
        "kpop_neon_dream_02",
        (
            (
                "verse",
                (
                    ("도시의 불빛이 하나둘씩 켜지고", "City lights are turning on across the busy street"),
                    ("심장 소리는 점점 더 빨라져 가네", "My heart is beating faster to the rhythm of the beat"),
                    ("유리창 너머로 비친 나의 모습은", "Looking at my shadow in the window through the night"),
                    ("어제보다 조금 더 강해진 것 같아", "Feeling so much stronger in the glowing neon light"),
                ),
            ),
            (
                "chorus",
                (
                    ("La la la 춤을 춰요 오늘 밤", "La la la dancing in the glowing night"),
                    ("모든 걱정은 다 잊어버리고", "Leaving every single memory behind"),
                    ("La la la 춤을 춰요 오늘 밤", "La la la dancing in the glowing night"),
                    ("Oh yeah 자유롭게 꿈을 펼쳐봐", "Oh yeah open up the wings inside your mind"),
                ),
            ),
            (
                "outro",
                (
                    ("새벽이 올 때까지 멈추지 마", "Never stop the music till the morning comes"),
                    ("새벽이 올 때까지 멈추지 마", "Never stop the music till the morning comes"),
                ),
            ),
        ),
    ),
    (
        "kpop_spring_breeze_03",
        (
            (
                "verse",
                (
                    ("따스한 봄바람이 내 볼을 스치면", "When the gentle breeze of spring is touching on my face"),
                    ("함께 걷던 그 길에 꽃이 피어나", "Flowers start to blossom in our favorite little place"),
                    ("기억 속의 계절은 아직 그대로인데", "Seasons in my memory are staying just the same"),
                    ("살며시 불어오는 바람에 네 이름을 불러", "Whispering softly in the wind I call your sweet name"),
                ),
            ),
            (
                "chorus",
                (
                    ("다시 만날 수 있을까 우리 둘이서", "Will we ever meet again together you and I"),
                    ("하얀 꽃잎이 흩날리는 이 거리에서", "Walking down the avenue beneath thepetal sky"),
                    ("다시 만날 수 있을까 우리 둘이서", "Will we ever meet again together you and I"),
                    ("Oh 내 마음은 언제나 너를 향해 있어", "Oh my heart is always waiting for you every day"),
                ),
            ),
        ),
    ),
    (
        "kpop_fire_rhythm_04",
        (
            (
                "intro",
                (
                    ("Yeah 준비됐어 지금 시작해", "Yeah we are ready let us start right now"),
                    ("Boom boom 타오르는 불꽃처럼", "Boom boom burning like a blazing fire"),
                ),
            ),
            (
                "chorus",
                (
                    ("거침없이 달려가 세상을 흔들어", "Running without hesitation shaking up the world"),
                    ("누구도 막을 수 없어 우리의 열정", "Nobody can ever stop the fire in our soul"),
                    ("거침없이 달려가 세상을 흔들어", "Running without hesitation shaking up the world"),
                    ("Woah 끝없이 높이 올라가 tonight", "Woah rising higher to the top tonight"),
                ),
            ),
            (
                "hook",
                (
                    ("Na na na 멈추지 마", "Na na na never slow down"),
                    ("Na na na 멈추지 마", "Na na na never slow down"),
                ),
            ),
        ),
    ),
)


def build_synthetic_kpop_sections(
    num_songs: int = 12,
    direction: str = "ko2en",
    max_lines_per_section: int = 12,
) -> Tuple[List[LyricSectionUnit], List[LyricSectionUnit]]:
    """외부 데이터셋 없이도 전처리·학습·추론 검증이 가능하도록 한-영 K-pop 섹션 데이터를 생성합니다."""
    directions = ("ko2en", "en2ko") if direction == "both" else (direction,)
    all_songs: List[List[LyricSectionUnit]] = []

    total_songs = max(2, num_songs)
    for song_i in range(total_songs):
        base_id, base_sections = _SYNTHETIC_KPOP_SONGS[song_i % len(_SYNTHETIC_KPOP_SONGS)]
        song_id = f"{base_id}_{song_i:03d}"
        song_units: List[LyricSectionUnit] = []
        for dir_tag in directions:
            for sec_idx, (sec_type, line_pairs) in enumerate(base_sections):
                ko_lines = [p[0] for p in line_pairs]
                en_lines = [p[1] for p in line_pairs]
                src_l = ko_lines if dir_tag == "ko2en" else en_lines
                tgt_l = en_lines if dir_tag == "ko2en" else ko_lines
                unit = build_lyric_section_unit(
                    song_id=f"{song_id}_{dir_tag}",
                    section_idx=sec_idx,
                    section_type=sec_type,
                    src_lines=src_l,
                    tgt_lines=tgt_l,
                    direction=dir_tag,
                )
                if unit is not None:
                    song_units.extend(
                        slice_long_section_unit(unit, max_lines=max_lines_per_section)
                    )
        if song_units:
            all_songs.append(song_units)

    val_song_count = max(1, len(all_songs) // 5)
    train_songs = all_songs[:-val_song_count] if len(all_songs) > 1 else all_songs
    val_songs = all_songs[-val_song_count:]

    train_units = [u for song in train_songs for u in song]
    val_units = [u for song in val_songs for u in song]
    return train_units, val_units


def discover_lyric_splits(
    root_dir: Optional[Path],
    requested_splits: Set[str],
) -> List[LyricSplitLayout]:
    """디렉터리에서 kpop-lt(lyrics.json, stages/*) 또는 일반 JSON/JSONL/CSV/TSV 가사 파일 분할을 탐색합니다."""
    if root_dir is None or not root_dir.exists():
        return []

    if root_dir.is_file():
        return [
            LyricSplitLayout(
                split_name=sp,
                data_files=[root_dir],
                wav_dirs=[root_dir.parent],
            )
            for sp in sorted(requested_splits)
        ]

    layouts: List[LyricSplitLayout] = []
    singable_sec_dir = root_dir / "stages" / "singable" / "section"
    if singable_sec_dir.exists():
        for sp_tag, fname in (("train", "train.jsonl"), ("val", "valid.jsonl")):
            if sp_tag in requested_splits and (singable_sec_dir / fname).exists():
                layouts.append(
                    LyricSplitLayout(
                        split_name=sp_tag,
                        data_files=[singable_sec_dir / fname],
                        wav_dirs=[root_dir],
                    )
                )
        if layouts:
            return layouts

    split_folders = (("train", ("train", "Training")), ("val", ("val", "valid", "Validation")))
    found_sub_splits = False
    for sp_tag, folder_names in split_folders:
        for fn in folder_names:
            cand = root_dir / fn
            if cand.exists() and cand.is_dir():
                found_sub_splits = True
                if sp_tag in requested_splits:
                    files = [
                        p
                        for p in sorted(cand.rglob("*"))
                        if p.is_file() and p.suffix.lower() in (".json", ".jsonl", ".tsv", ".csv")
                    ]
                    if files:
                        layouts.append(
                            LyricSplitLayout(
                                split_name=sp_tag,
                                data_files=files,
                                wav_dirs=[cand, root_dir],
                            )
                        )
                break

    if not found_sub_splits:
        all_files = [
            p
            for p in sorted(root_dir.rglob("*"))
            if p.is_file()
            and p.suffix.lower() in (".json", ".jsonl", ".tsv", ".csv")
            and p.name != "preprocessing_manifest.json"
        ]
        if all_files:
            for sp_tag in ("train", "val"):
                if sp_tag in requested_splits:
                    layouts.append(
                        LyricSplitLayout(
                            split_name=sp_tag,
                            data_files=all_files,
                            wav_dirs=[root_dir],
                        )
                    )

    return layouts


def _parse_single_lyric_file(
    file_path: Path,
    direction: str = "ko2en",
    max_lines_per_section: int = 12,
) -> List[LyricSectionUnit]:
    """단일 JSON/JSONL/TSV/CSV 파일에서 노래별 Section -> Line 계층 병렬 데이터를 파싱합니다."""
    directions = ("ko2en", "en2ko") if direction == "both" else (direction,)
    units: List[LyricSectionUnit] = []
    sfx = file_path.suffix.lower()

    def _add_section(
        song_id: str,
        sec_idx: int,
        sec_type: str,
        ko_lines: Sequence[str],
        en_lines: Sequence[str],
        ko_wav: str = "",
        en_wav: str = "",
    ) -> None:
        for dir_tag in directions:
            src_l = ko_lines if dir_tag == "ko2en" else en_lines
            tgt_l = en_lines if dir_tag == "ko2en" else ko_lines
            s_wav = ko_wav if dir_tag == "ko2en" else en_wav
            t_wav = en_wav if dir_tag == "ko2en" else ko_wav
            u = build_lyric_section_unit(
                song_id=f"{song_id}_{dir_tag}",
                section_idx=sec_idx,
                section_type=sec_type,
                src_lines=src_l,
                tgt_lines=tgt_l,
                direction=dir_tag,
                src_wav_name=s_wav,
                tgt_wav_name=t_wav,
            )
            if u is not None:
                units.extend(slice_long_section_unit(u, max_lines=max_lines_per_section))

    try:
        if sfx == ".jsonl":
            lines_raw = file_path.read_text(encoding="utf-8-sig").splitlines()
            for idx, ln in enumerate(lines_raw):
                if not ln.strip():
                    continue
                item = json.loads(ln)
                if isinstance(item, dict) and "src" in item and "tgt" in item:
                    src_list = [str(x) for x in item["src"]]
                    tgt_list = [str(x) for x in item["tgt"]]
                    sec_type = str(item.get("section_type", "chorus" if len(src_list) > 2 else "verse"))
                    _add_section(f"{file_path.stem}_{idx:05d}", 0, sec_type, src_list, tgt_list)
            return units

        if sfx == ".json":
            payload = json.loads(file_path.read_text(encoding="utf-8-sig"))
            if isinstance(payload, dict):
                if "sections" in payload and isinstance(payload["sections"], list):
                    song_id = str(payload.get("song_id") or payload.get("id") or file_path.stem)
                    for s_i, sec_obj in enumerate(payload["sections"]):
                        if isinstance(sec_obj, dict):
                            s_type = str(sec_obj.get("type") or sec_obj.get("section_type") or "verse")
                            ko_l = sec_obj.get("ko") or sec_obj.get("src") or []
                            en_l = sec_obj.get("en") or sec_obj.get("tgt") or []
                            if not ko_l and "lines" in sec_obj and isinstance(sec_obj["lines"], list):
                                ko_l = [str(x.get("ko") or x.get("src") or "") for x in sec_obj["lines"] if isinstance(x, dict)]
                                en_l = [str(x.get("en") or x.get("tgt") or "") for x in sec_obj["lines"] if isinstance(x, dict)]
                            _add_section(
                                song_id,
                                s_i,
                                s_type,
                                [str(x) for x in ko_l],
                                [str(x) for x in en_l],
                                str(sec_obj.get("ko_wav") or ""),
                                str(sec_obj.get("en_wav") or ""),
                            )
                    return units

                for song_key, song_val in payload.items():
                    if isinstance(song_val, list):
                        for s_i, sec_val in enumerate(song_val):
                            if isinstance(sec_val, list) and sec_val:
                                if isinstance(sec_val[0], (list, tuple)) and len(sec_val[0]) >= 2:
                                    ko_l = [str(pair[0]) for pair in sec_val if len(pair) >= 2]
                                    en_l = [str(pair[1]) for pair in sec_val if len(pair) >= 2]
                                    sec_type = "chorus" if (s_i % 2 == 1) else "verse"
                                    _add_section(str(song_key), s_i, sec_type, ko_l, en_l)
            elif isinstance(payload, list):
                for idx, elem in enumerate(payload):
                    if isinstance(elem, dict):
                        s_id = str(elem.get("song_id") or elem.get("id") or f"{file_path.stem}_{idx:04d}")
                        s_type = str(elem.get("section_type") or "verse")
                        ko_l = elem.get("ko") or elem.get("src") or []
                        en_l = elem.get("en") or elem.get("tgt") or []
                        if isinstance(ko_l, str):
                            ko_l = [ko_l]
                        if isinstance(en_l, str):
                            en_l = [en_l]
                        _add_section(s_id, idx, s_type, [str(x) for x in ko_l], [str(x) for x in en_l])
            return units

        if sfx in (".tsv", ".csv"):
            delim = "\t" if sfx == ".tsv" else ","
            with file_path.open("r", encoding="utf-8-sig") as f:
                reader = csv.reader(f, delimiter=delim)
                cur_ko: List[str] = []
                cur_en: List[str] = []
                sec_idx = 0
                for row in reader:
                    if len(row) < 2:
                        continue
                    c0 = normalize_lyric_text(row[0])
                    c1 = normalize_lyric_text(row[1])
                    if c0.lower() in ("ko", "src", "korean", "source") and c1.lower() in ("en", "tgt", "english", "target"):
                        continue
                    if c0 and c1:
                        cur_ko.append(c0)
                        cur_en.append(c1)
                        if len(cur_ko) >= 4:
                            _add_section(f"{file_path.stem}_{sec_idx:04d}", sec_idx, "verse", cur_ko, cur_en)
                            cur_ko, cur_en = [], []
                            sec_idx += 1
                if cur_ko:
                    _add_section(f"{file_path.stem}_{sec_idx:04d}", sec_idx, "verse", cur_ko, cur_en)
    except (OSError, json.JSONDecodeError, csv.Error) as exc:
        logger.warning("가사 파일 파싱 건너뜀 (%s): %s", file_path, exc)

    return units


def parse_lyric_split(
    layout: LyricSplitLayout,
    direction: str = "ko2en",
    max_lines_per_section: int = 12,
    max_samples: Optional[int] = None,
    num_io_threads: int = 8,
    shared_single_source: bool = False,
) -> List[LyricSectionUnit]:
    """LyricSplitLayout 내 파일들을 병렬 파싱하여 섹션 단위 학습 샘플 리스트를 반환합니다."""
    def _worker(fp: Path) -> List[LyricSectionUnit]:
        return _parse_single_lyric_file(
            fp, direction=direction, max_lines_per_section=max_lines_per_section
        )

    all_units: List[LyricSectionUnit] = []
    for file_units in batched_pool_map(
        _worker, layout.data_files, num_workers=num_io_threads, batch_size=64
    ):
        all_units.extend(file_units)
        if not shared_single_source and max_samples is not None and len(all_units) >= max_samples * 2:
            break

    if shared_single_source and len(all_units) > 1:
        val_cut = max(1, len(all_units) // 10)
        if layout.split_name == "val":
            all_units = all_units[-val_cut:]
        else:
            all_units = all_units[:-val_cut]

    if max_samples is not None and max_samples > 0:
        all_units = all_units[:max_samples]

    return all_units


__all__ = [
    "LyricSplitLayout",
    "build_synthetic_kpop_sections",
    "discover_lyric_splits",
    "parse_lyric_split",
    "fast_scandir_files",
]
