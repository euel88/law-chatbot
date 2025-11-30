"""
PDF 번역 모듈 - PDFMathTranslate 스타일 구현
텍스트 블록 추출, OCR, 번역, 레이아웃 보존
"""

import fitz  # PyMuPDF
import io
import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Tesseract OCR (선택적)
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TextBlock:
    """텍스트 블록 데이터 클래스"""
    text: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    page_num: int
    font_name: str = ""
    font_size: float = 11.0
    is_formula: bool = False
    color: Tuple[float, float, float] = (0, 0, 0)  # RGB


@dataclass
class ImageBlock:
    """이미지 블록 데이터 클래스 (OCR 대상)"""
    image: Image.Image
    bbox: Tuple[float, float, float, float]
    page_num: int
    ocr_text: str = ""
    translated_text: str = ""


class PDFTextExtractor:
    """PDF에서 텍스트 블록 추출"""

    # 수식 폰트 패턴 (PDFMathTranslate 방식)
    FORMULA_FONT_PATTERNS = [
        r'math', r'symbol', r'cmex', r'cmsy', r'cmmi', r'cmr',
        r'msam', r'msbm', r'eufm', r'eurb', r'eusb',
        r'CMEX', r'CMSY', r'CMMI', r'CMR'
    ]

    # 수식 유니코드 범위
    FORMULA_UNICODE_RANGES = [
        (0x0370, 0x03FF),  # 그리스 문자
        (0x2200, 0x22FF),  # 수학 연산자
        (0x2100, 0x214F),  # Letterlike Symbols
        (0x2190, 0x21FF),  # 화살표
    ]

    def __init__(self, preserve_formulas: bool = True):
        self.preserve_formulas = preserve_formulas

    def is_formula_font(self, font_name: str) -> bool:
        """수식 폰트인지 확인"""
        if not font_name:
            return False
        font_lower = font_name.lower()
        for pattern in self.FORMULA_FONT_PATTERNS:
            if pattern.lower() in font_lower:
                return True
        return False

    def is_formula_char(self, char: str) -> bool:
        """수식 문자인지 확인"""
        if not char:
            return False
        code = ord(char[0])
        for start, end in self.FORMULA_UNICODE_RANGES:
            if start <= code <= end:
                return True
        return False

    def extract_text_blocks(self, doc: fitz.Document) -> List[TextBlock]:
        """PDF에서 모든 텍스트 블록 추출"""
        all_blocks = []

        for page_num, page in enumerate(doc):
            # 텍스트 블록 추출 (딕셔너리 형태로)
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

            for block in blocks.get("blocks", []):
                if block.get("type") == 0:  # 텍스트 블록
                    self._process_text_block(block, page_num, all_blocks)

        return all_blocks

    def _process_text_block(self, block: Dict, page_num: int,
                           result: List[TextBlock]) -> None:
        """텍스트 블록 처리"""
        bbox = block.get("bbox", (0, 0, 0, 0))

        # 라인별 처리
        for line in block.get("lines", []):
            line_text = ""
            font_name = ""
            font_size = 11.0
            is_formula = False
            color = (0, 0, 0)

            for span in line.get("spans", []):
                span_text = span.get("text", "")
                span_font = span.get("font", "")
                span_size = span.get("size", 11.0)
                span_color = span.get("color", 0)

                # RGB 색상 변환
                if isinstance(span_color, int):
                    r = (span_color >> 16) & 0xFF
                    g = (span_color >> 8) & 0xFF
                    b = span_color & 0xFF
                    color = (r/255, g/255, b/255)

                line_text += span_text
                font_name = span_font
                font_size = span_size

                # 수식 감지
                if self.preserve_formulas:
                    if self.is_formula_font(span_font):
                        is_formula = True
                    elif any(self.is_formula_char(c) for c in span_text):
                        is_formula = True

            if line_text.strip():
                line_bbox = line.get("bbox", bbox)
                result.append(TextBlock(
                    text=line_text.strip(),
                    bbox=tuple(line_bbox),
                    page_num=page_num,
                    font_name=font_name,
                    font_size=font_size,
                    is_formula=is_formula,
                    color=color
                ))

    def extract_images(self, doc: fitz.Document,
                      min_size: int = 50) -> List[ImageBlock]:
        """PDF에서 이미지 추출 (OCR 대상)"""
        images = []

        for page_num, page in enumerate(doc):
            # 이미지 리스트 가져오기
            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]

                try:
                    # 이미지 추출
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    # PIL 이미지로 변환
                    pil_image = Image.open(io.BytesIO(image_bytes))

                    # 크기 필터링
                    if pil_image.width < min_size or pil_image.height < min_size:
                        continue

                    # 이미지 위치 찾기 (근사값)
                    image_rects = page.get_image_rects(xref)
                    if image_rects:
                        rect = image_rects[0]
                        bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
                    else:
                        bbox = (0, 0, pil_image.width, pil_image.height)

                    images.append(ImageBlock(
                        image=pil_image,
                        bbox=bbox,
                        page_num=page_num
                    ))
                except Exception as e:
                    logger.warning(f"이미지 추출 실패 (page {page_num}, xref {xref}): {e}")

        return images


class OCRProcessor:
    """이미지 OCR 처리기"""

    def __init__(self, lang: str = "kor+eng"):
        self.lang = lang
        self.available = TESSERACT_AVAILABLE

    def process_image(self, image: Image.Image) -> str:
        """이미지에서 텍스트 추출"""
        if not self.available:
            return ""

        try:
            # 이미지 전처리
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # OCR 수행
            text = pytesseract.image_to_string(image, lang=self.lang)
            return text.strip()
        except Exception as e:
            logger.error(f"OCR 실패: {e}")
            return ""

    def process_images(self, images: List[ImageBlock]) -> List[ImageBlock]:
        """여러 이미지 일괄 OCR 처리"""
        if not self.available:
            return images

        for img_block in images:
            img_block.ocr_text = self.process_image(img_block.image)

        return images


class Translator:
    """텍스트 번역기 (OpenAI 기반)"""

    def __init__(self, openai_client, source_lang: str = "en",
                 target_lang: str = "ko"):
        self.client = openai_client
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.cache = {}  # 번역 캐시

    def translate_text(self, text: str) -> str:
        """단일 텍스트 번역"""
        if not text or not text.strip():
            return text

        # 캐시 확인
        cache_key = f"{self.source_lang}_{self.target_lang}_{text}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        if not self.client:
            return text

        try:
            lang_names = {
                'ko': '한국어', 'en': '영어', 'ja': '일본어',
                'zh': '중국어', 'de': '독일어', 'fr': '프랑스어',
                'es': '스페인어', 'ru': '러시아어'
            }
            target_name = lang_names.get(self.target_lang, self.target_lang)

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": f"""당신은 전문 번역가입니다.
텍스트를 {target_name}로 정확하게 번역하세요.
- 수식, 숫자, 변수명은 그대로 유지하세요.
- 학술 용어는 적절히 번역하되 괄호 안에 원문을 표기할 수 있습니다.
- 번역문만 출력하세요."""
                    },
                    {"role": "user", "content": text}
                ],
                temperature=0.3,
                max_tokens=2000
            )

            translated = response.choices[0].message.content.strip()
            self.cache[cache_key] = translated
            return translated

        except Exception as e:
            logger.error(f"번역 실패: {e}")
            return text

    def translate_blocks(self, blocks: List[TextBlock],
                        progress_callback=None) -> List[Tuple[TextBlock, str]]:
        """텍스트 블록 일괄 번역"""
        results = []
        total = len(blocks)

        for idx, block in enumerate(blocks):
            # 수식은 번역하지 않음
            if block.is_formula:
                results.append((block, block.text))
            else:
                translated = self.translate_text(block.text)
                results.append((block, translated))

            if progress_callback:
                progress_callback((idx + 1) / total)

        return results

    def translate_images(self, images: List[ImageBlock],
                        progress_callback=None) -> List[ImageBlock]:
        """이미지 OCR 텍스트 번역"""
        total = len(images)

        for idx, img in enumerate(images):
            if img.ocr_text:
                img.translated_text = self.translate_text(img.ocr_text)

            if progress_callback:
                progress_callback((idx + 1) / total)

        return images


class PDFRenderer:
    """번역된 텍스트를 PDF에 렌더링"""

    # 한글 지원 폰트
    KOREAN_FONTS = [
        "malgun",  # 맑은 고딕
        "gulim",   # 굴림
        "batang",  # 바탕
        "dotum",   # 돋움
    ]

    def __init__(self):
        self.default_font = "helv"  # 기본 폰트

    def create_translated_pdf(self,
                             source_doc: fitz.Document,
                             translated_blocks: List[Tuple[TextBlock, str]],
                             translated_images: List[ImageBlock] = None,
                             overlay_mode: bool = True) -> fitz.Document:
        """번역된 PDF 생성"""

        # 새 문서 생성 (원본 복사)
        output_doc = fitz.open()

        for page_num in range(len(source_doc)):
            # 원본 페이지 복사
            source_page = source_doc[page_num]
            new_page = output_doc.new_page(
                width=source_page.rect.width,
                height=source_page.rect.height
            )

            if overlay_mode:
                # 오버레이 모드: 원본 위에 번역 텍스트 덮어쓰기
                new_page.show_pdf_page(new_page.rect, source_doc, page_num)
                self._overlay_translations(new_page, translated_blocks, page_num)
            else:
                # 교체 모드: 텍스트 영역 흰색으로 덮고 번역 텍스트 삽입
                new_page.show_pdf_page(new_page.rect, source_doc, page_num)
                self._replace_with_translations(new_page, translated_blocks, page_num)

            # 이미지 OCR 번역 추가
            if translated_images:
                self._add_image_translations(new_page, translated_images, page_num)

        return output_doc

    def _overlay_translations(self, page: fitz.Page,
                             translated_blocks: List[Tuple[TextBlock, str]],
                             page_num: int) -> None:
        """오버레이 방식으로 번역 텍스트 추가"""

        for block, translated_text in translated_blocks:
            if block.page_num != page_num:
                continue

            if block.is_formula:
                continue  # 수식은 건드리지 않음

            if block.text == translated_text:
                continue  # 변경 없음

            x0, y0, x1, y1 = block.bbox

            # 원본 텍스트 영역을 흰색으로 덮기
            rect = fitz.Rect(x0, y0, x1, y1)
            page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))

            # 번역된 텍스트 삽입
            self._insert_text(page, rect, translated_text, block.font_size)

    def _replace_with_translations(self, page: fitz.Page,
                                   translated_blocks: List[Tuple[TextBlock, str]],
                                   page_num: int) -> None:
        """교체 방식으로 번역 텍스트 추가"""

        for block, translated_text in translated_blocks:
            if block.page_num != page_num:
                continue

            if block.is_formula:
                continue

            x0, y0, x1, y1 = block.bbox
            rect = fitz.Rect(x0, y0, x1, y1)

            # 흰색 배경으로 덮기
            page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))

            # 번역 텍스트 삽입
            self._insert_text(page, rect, translated_text, block.font_size)

    def _insert_text(self, page: fitz.Page, rect: fitz.Rect,
                    text: str, font_size: float) -> None:
        """텍스트 삽입 (폰트 크기 자동 조절)"""

        # 폰트 크기 조절 (영역에 맞게)
        adjusted_size = min(font_size, rect.height * 0.8)
        adjusted_size = max(adjusted_size, 6)  # 최소 크기

        try:
            # 텍스트 삽입 시도
            text_writer = fitz.TextWriter(page.rect)

            # 한글 폰트 사용 시도
            font = fitz.Font("korea1")  # CJK 폰트

            # 텍스트 위치 계산
            text_point = fitz.Point(rect.x0, rect.y0 + adjusted_size)

            try:
                text_writer.append(text_point, text, font=font, fontsize=adjusted_size)
                text_writer.write_text(page)
            except Exception:
                # 기본 방식으로 폴백
                page.insert_text(
                    text_point,
                    text,
                    fontsize=adjusted_size,
                    fontname="helv",
                    color=(0, 0, 0)
                )
        except Exception as e:
            logger.warning(f"텍스트 삽입 실패: {e}")
            # 최종 폴백: 기본 삽입
            try:
                page.insert_textbox(
                    rect,
                    text,
                    fontsize=adjusted_size,
                    fontname="helv",
                    color=(0, 0, 0),
                    align=fitz.TEXT_ALIGN_LEFT
                )
            except Exception:
                pass

    def _add_image_translations(self, page: fitz.Page,
                               images: List[ImageBlock],
                               page_num: int) -> None:
        """이미지 번역 텍스트 추가 (이미지 아래에 캡션 형태)"""

        for img in images:
            if img.page_num != page_num:
                continue

            if not img.translated_text:
                continue

            x0, y0, x1, y1 = img.bbox

            # 이미지 아래에 번역 텍스트 박스 추가
            caption_rect = fitz.Rect(x0, y1 + 2, x1, y1 + 30)

            # 반투명 배경
            page.draw_rect(caption_rect, color=(0.9, 0.9, 0.9),
                          fill=(0.95, 0.95, 0.95))

            # 번역 텍스트 삽입
            try:
                page.insert_textbox(
                    caption_rect,
                    f"[OCR 번역] {img.translated_text[:100]}...",
                    fontsize=8,
                    fontname="helv",
                    color=(0.2, 0.2, 0.2),
                    align=fitz.TEXT_ALIGN_LEFT
                )
            except Exception:
                pass


class PDFTranslator:
    """PDF 번역 통합 클래스"""

    def __init__(self, openai_client=None,
                 source_lang: str = "en",
                 target_lang: str = "ko"):
        self.openai_client = openai_client
        self.source_lang = source_lang
        self.target_lang = target_lang

        # 컴포넌트 초기화
        self.extractor = PDFTextExtractor(preserve_formulas=True)
        self.ocr_processor = OCRProcessor(lang="kor+eng")
        self.translator = Translator(openai_client, source_lang, target_lang)
        self.renderer = PDFRenderer()

    def translate_pdf(self,
                     pdf_bytes: bytes,
                     translate_text: bool = True,
                     translate_images: bool = False,
                     progress_callback=None) -> bytes:
        """PDF 파일 번역"""

        # PDF 열기
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        translated_blocks = []
        translated_images = []

        total_steps = 4 if translate_images else 3
        step = 0

        def update_progress(sub_progress: float):
            if progress_callback:
                overall = (step + sub_progress) / total_steps
                progress_callback(overall)

        # Step 1: 텍스트 블록 추출
        if progress_callback:
            progress_callback(0.1, "텍스트 블록 추출 중...")

        text_blocks = self.extractor.extract_text_blocks(doc)
        step = 1

        # Step 2: 텍스트 번역
        if translate_text and text_blocks:
            if progress_callback:
                progress_callback(0.3, "텍스트 번역 중...")

            translated_blocks = self.translator.translate_blocks(
                text_blocks,
                progress_callback=lambda p: update_progress(p)
            )
        else:
            translated_blocks = [(b, b.text) for b in text_blocks]

        step = 2

        # Step 3: 이미지 OCR 및 번역 (선택)
        if translate_images:
            if progress_callback:
                progress_callback(0.5, "이미지 OCR 처리 중...")

            images = self.extractor.extract_images(doc)
            images = self.ocr_processor.process_images(images)

            if progress_callback:
                progress_callback(0.6, "이미지 텍스트 번역 중...")

            translated_images = self.translator.translate_images(
                images,
                progress_callback=lambda p: update_progress(p)
            )
            step = 3

        # Step 4: 번역된 PDF 생성
        if progress_callback:
            progress_callback(0.8, "번역된 PDF 생성 중...")

        output_doc = self.renderer.create_translated_pdf(
            doc, translated_blocks, translated_images
        )

        # 바이트로 변환
        output_bytes = output_doc.tobytes()

        # 정리
        doc.close()
        output_doc.close()

        if progress_callback:
            progress_callback(1.0, "완료!")

        return output_bytes

    def get_pdf_info(self, pdf_bytes: bytes) -> Dict:
        """PDF 정보 추출"""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        info = {
            'page_count': len(doc),
            'metadata': doc.metadata,
            'text_blocks_count': 0,
            'images_count': 0
        }

        # 텍스트 블록 수 계산
        text_blocks = self.extractor.extract_text_blocks(doc)
        info['text_blocks_count'] = len(text_blocks)

        # 이미지 수 계산
        images = self.extractor.extract_images(doc)
        info['images_count'] = len(images)

        doc.close()

        return info


def translate_pdf_file(pdf_bytes: bytes,
                      openai_client=None,
                      source_lang: str = "en",
                      target_lang: str = "ko",
                      translate_text: bool = True,
                      translate_images: bool = False,
                      progress_callback=None) -> bytes:
    """PDF 파일 번역 헬퍼 함수"""

    translator = PDFTranslator(
        openai_client=openai_client,
        source_lang=source_lang,
        target_lang=target_lang
    )

    return translator.translate_pdf(
        pdf_bytes,
        translate_text=translate_text,
        translate_images=translate_images,
        progress_callback=progress_callback
    )
