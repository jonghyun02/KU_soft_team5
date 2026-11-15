"""Broadcast Cinema NLE Dark Qt6 Theme (QSS) and programmatic Studio Icon builder."""

from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SPEAKER_HEX_PALETTE: list[str] = [
    "#FFFFFF",  # 화자_1: Pure White
    "#FFE885",  # 화자_2: Warm Soft Yellow
    "#A7F3D0",  # 화자_3: Mint Green
    "#93C5FD",  # 화자_4: Soft Sky Blue
    "#F5D0FE",  # 화자_5: Soft Lavender
    "#FED7AA",  # 화자_6: Soft Peach
]

EMOTION_HEX_MAP: dict[str, str] = {
    "분노": "#E11D48",
    "냉소": "#9333EA",
    "슬픔": "#2563EB",
    "기쁨": "#059669",
    "공포": "#D97706",
    "중립": "#475569",
}

STRATEGY_KO_LABELS: dict[str, str] = {
    "positional_tag_standard": "표준 위치 태그 (CAPE·정서·강조 통합)",
    "tokuyama_intensifier_insertion": "정도 부사 보강 (Tokuyama 강조 삽입)",
    "concise_low_cps": "저-CPS 고가독성 간결형",
    "llm_augmented_zeroshot": "로컬 4-bit LLM 증강 제로샷 { } 재작성",
}

STUDIO_DARK_QSS = """
QMainWindow, QDialog {
    background-color: #090D14;
    color: #EDF2FA;
    font-family: "Malgun Gothic", "Segoe UI", sans-serif;
    font-size: 12px;
}
QMenuBar {
    background-color: #0E1522;
    color: #E2E8F0;
    border-bottom: 1px solid #222E44;
    padding: 2px 6px;
}
QMenuBar::item:selected, QMenu::item:selected {
    background-color: #1E3252;
    color: #FFFFFF;
}
QMenu {
    background-color: #101726;
    color: #EDF2FA;
    border: 1px solid #334566;
}
QToolBar {
    background-color: #101726;
    border-bottom: 1px solid #222E44;
    spacing: 6px;
    padding: 5px 8px;
}
QStatusBar {
    background-color: #0C121E;
    color: #94A3B8;
    border-top: 1px solid #222E44;
}
QFrame#StudioPanel {
    background-color: #101622;
    border: 1px solid #222E44;
    border-radius: 7px;
}
QFrame#SubCard {
    background-color: #0B101B;
    border: 1px solid #222E44;
    border-radius: 6px;
}
QFrame#CandCard {
    background-color: #151E2F;
    border: 1px solid #283650;
    border-radius: 6px;
}
QFrame#CandCardSelected {
    background-color: #192B47;
    border: 1.5px solid #38BDF8;
    border-radius: 6px;
}
QLabel {
    color: #EDF2FA;
    background: transparent;
}
QLabel#MutedLabel {
    color: #94A3B8;
    font-size: 11px;
}
QLabel#PanelTitle {
    color: #F8FAFC;
    font-size: 13px;
    font-weight: bold;
}
QLabel#GpuBadge {
    background-color: rgba(52, 211, 153, 0.15);
    color: #34D399;
    border: 1px solid rgba(52, 211, 153, 0.45);
    border-radius: 10px;
    padding: 3px 10px;
    font-weight: bold;
    font-size: 11px;
}
QPushButton {
    background-color: #162033;
    color: #EDF2FA;
    border: 1px solid #334566;
    border-radius: 5px;
    padding: 5px 11px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #1F2D47;
    border-color: #4B6596;
}
QPushButton:pressed {
    background-color: #131C2E;
}
QPushButton:checked {
    background-color: #1E365C;
    border-color: #38BDF8;
    color: #FFFFFF;
}
QPushButton#PrimaryBtn {
    background-color: #0284C7;
    border-color: #38BDF8;
    color: #FFFFFF;
}
QPushButton#PrimaryBtn:hover {
    background-color: #0369A1;
}
QPushButton#AccentBtn {
    background-color: #162B44;
    border-color: #38BDF8;
    color: #38BDF8;
}
QPushButton#AssBtn {
    background-color: #2A2111;
    border-color: #F59E0B;
    color: #FCD34D;
}
QComboBox, QLineEdit, QPlainTextEdit {
    background-color: #151E2E;
    color: #F1F5F9;
    border: 1px solid #334566;
    border-radius: 5px;
    padding: 4px 8px;
    selection-background-color: #0284C7;
}
QTabWidget::pane {
    border: 1px solid #222E44;
    background-color: #101622;
    border-radius: 6px;
}
QTabBar::tab {
    background-color: #0C121E;
    color: #94A3B8;
    border: 1px solid #222E44;
    padding: 6px 12px;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    margin-right: 3px;
    font-weight: 600;
}
QTabBar::tab:selected {
    background-color: #1A2B47;
    color: #FFFFFF;
    border-bottom-color: #38BDF8;
}
QTableWidget {
    background-color: #0D131F;
    alternate-background-color: #111827;
    color: #EDF2FA;
    gridline-color: #1E293B;
    border: 1px solid #222E44;
    border-radius: 5px;
    selection-background-color: #1E365C;
    selection-color: #FFFFFF;
}
QHeaderView::section {
    background-color: #141E30;
    color: #94A3B8;
    padding: 6px;
    border: none;
    border-bottom: 1px solid #334566;
    border-right: 1px solid #1E293B;
    font-weight: bold;
    font-size: 11px;
}
QScrollArea {
    border: none;
    background: transparent;
}
QScrollBar:vertical {
    background: #0B101B;
    width: 10px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #2B3B59;
    min-height: 24px;
    border-radius: 5px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QSplitter::handle {
    background-color: #1A2436;
}
QSplitter::handle:hover {
    background-color: #38BDF8;
}
QProgressBar {
    background-color: #0B101B;
    border: 1px solid #24324B;
    border-radius: 4px;
    text-align: center;
    color: #E2E8F0;
    font-size: 10px;
}
QProgressBar::chunk {
    background-color: #0284C7;
    border-radius: 3px;
}
QCheckBox {
    color: #CBD5E1;
    spacing: 6px;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #1B263B;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #38BDF8;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #F8FAFC;
    border: 1px solid #0284C7;
    width: 13px;
    margin: -4px 0;
    border-radius: 6px;
}
"""


def ensure_studio_ico(ico_path: Path) -> Path:
    """Generate a crisp multi-resolution Windows `.ico` icon for MoS-SDH Studio if not yet present."""
    if ico_path.exists() and ico_path.stat().st_size > 1000:
        return ico_path

    ico_path.parent.mkdir(parents=True, exist_ok=True)
    size = 256
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Rounded dark-blue broadcast studio badge
    draw.rounded_rectangle([(12, 12), (244, 244)], radius=52, fill=(14, 26, 48, 255), outline=(56, 189, 248, 255), width=8)
    # Soundwave + subtitle bars motif
    bar_specs = [
        (56, 98, 74, 150, (56, 189, 248, 255)),
        (88, 66, 106, 168, (99, 102, 241, 255)),
        (120, 48, 138, 174, (251, 191, 36, 255)),
        (152, 74, 170, 164, (52, 211, 153, 255)),
        (184, 96, 202, 148, (56, 189, 248, 255)),
    ]
    for x1, y1, x2, y2, col in bar_specs:
        draw.rounded_rectangle([(x1, y1), (x2, y2)], radius=9, fill=col)

    # Bottom SDH caption bar highlight
    draw.rounded_rectangle([(46, 192), (210, 216)], radius=8, fill=(248, 250, 252, 240))
    font_p = Path("C:/Windows/Fonts/malgunbd.ttf")
    if font_p.exists():
        fnt = ImageFont.truetype(str(font_p), 22)
        draw.text((82, 190), "MoS", font=fnt, fill=(15, 23, 42, 255))

    img.save(ico_path, format="ICO", sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
    return ico_path
