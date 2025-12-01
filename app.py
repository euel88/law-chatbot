"""
AI 법률 연구 도우미 - 판례, 유권해석, 법령 종합 검색 서비스 + PDF 번역
법제처 API + ChatGPT를 활용한 법률 자료 검색 및 분석
PDF 문서 번역 기능 (PDFMathTranslate 스타일)

실행 방법:
streamlit run app.py
"""

import streamlit as st
import requests
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import asyncio
import nest_asyncio
import aiohttp
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import logging
from enum import Enum
import re
import io
import base64

# PDF 생성 모듈 (선택적 import)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# PDF 번역 모듈 (선택적 import)
try:
    from pdf_translator import PDFTranslator, translate_pdf_file
    PDF_TRANSLATOR_AVAILABLE = True
except ImportError:
    PDF_TRANSLATOR_AVAILABLE = False

# Streamlit 환경에서 asyncio 이벤트 루프 충돌 방지
nest_asyncio.apply()

# 환경변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== 페이지 설정 =====
st.set_page_config(
    page_title="AI 법률 도우미 & PDF 번역",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== 커스텀 CSS =====
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .user-message {
        background-color: #e8f4f8;
        margin-left: 20%;
    }

    .assistant-message {
        background-color: #f0f2f6;
        margin-right: 20%;
    }

    .legal-opinion {
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }

    .search-result {
        background-color: #f8f9fa;
        border-left: 4px solid #1976d2;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }

    .api-status-ok { color: #388e3c; font-weight: bold; }
    .api-status-error { color: #d32f2f; font-weight: bold; }

    .category-header {
        background-color: #1976d2;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 1rem 0 0.5rem 0;
    }

    .pdf-upload-area {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #fafafa;
        margin: 1rem 0;
    }

    .pdf-preview {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #fff;
    }

    .translation-progress {
        padding: 1rem;
        background-color: #e3f2fd;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ===== 서비스 유형 Enum =====
class ServiceType(Enum):
    INFO = "법률 정보 제공"
    CONTRACT = "계약서 검토"
    OPINION = "법률자문의견서"
    RESEARCH = "법률 연구"

# ===== 리스크 레벨 =====
class RiskLevel(Enum):
    HIGH = ("🔴 High", "즉시 중단/전면 재검토 필요")
    MEDIUM = ("🟠 Medium", "수정 협상 필수")
    LOW = ("🟡 Low", "문구 명확화 권장")

# ===== 세션 상태 초기화 =====
def init_session_state():
    """세션 상태 초기화"""
    defaults = {
        'chat_history': [],
        'current_service': None,
        'fact_sheet': {},
        'case_documents': [],
        'law_api_key': '',
        'openai_api_key': '',
        'api_keys_set': False,
        'search_results': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ===== API 키 관리 함수 =====
def get_law_api_key() -> str:
    """법제처 API 키 가져오기"""
    # 1. 세션에서 확인
    if st.session_state.law_api_key:
        return st.session_state.law_api_key
    # 2. Streamlit secrets 확인
    try:
        if hasattr(st, 'secrets') and 'LAW_API_KEY' in st.secrets:
            return st.secrets['LAW_API_KEY']
    except Exception:
        pass
    # 3. 환경변수 확인
    return os.getenv('LAW_API_KEY', '')

def get_openai_api_key() -> str:
    """OpenAI API 키 가져오기"""
    # 1. 세션에서 확인
    if st.session_state.openai_api_key:
        return st.session_state.openai_api_key
    # 2. Streamlit secrets 확인
    try:
        if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
            return st.secrets['OPENAI_API_KEY']
    except Exception:
        pass
    # 3. 환경변수 확인
    return os.getenv('OPENAI_API_KEY', '')

def get_openai_client():
    """OpenAI 클라이언트 가져오기"""
    api_key = get_openai_api_key()
    if api_key:
        # API 키 형식 검증 (sk-로 시작해야 함)
        if not api_key.startswith('sk-'):
            logger.warning(f"잘못된 OpenAI API 키 형식: {api_key[:20]}... (sk-로 시작해야 합니다)")
            return None
        return OpenAI(api_key=api_key)
    return None

# ===== AI 변호사 프롬프트 템플릿 =====
AI_LAWYER_SYSTEM_PROMPT = """
당신은 한국의 전문 법률자문의견서 작성 전문가이자 가상의 변호사입니다.
실제 변호사의 사고 방식(사실관계 파악 → Issue-Spotting → 법리 검토 → 위험측정 → 전략 수립)을 완벽히 구현합니다.

핵심 원칙:
1. 증거 우선주의: 구두 진술만으로 판단하지 않고 물적 증빙 확보 최우선
2. 근거 기반 분석: 모든 법적 주장은 출처(법령·판례·행정해석) 명시
3. 사용자 중심 접근: 모든 쟁점을 의뢰인 관점에서 유리/불리로 평가
4. IRAC 방법론: Issue → Rule → Application → Conclusion 구조
5. 리스크 계층화: High/Medium/Low 등급화
6. 실행가능한 해결책: 최소 2가지 이상의 대안 제시

필수 고지: ⚖️ 본 내용은 AI가 작성한 참고자료이며, 법률자문이 아닙니다.
구체적인 사안에 대해서는 반드시 변호사 등 전문가의 검토가 필요합니다.
"""

# ===== 법률 AI 엔진 클래스 =====
class LegalAIEngine:
    """AI 법률 연구 엔진 - 법제처 API 전체 연동"""

    def __init__(self):
        self.law_api_key = get_law_api_key()
        self.api_endpoints = {
            'search': 'https://www.law.go.kr/DRF/lawSearch.do',
            'service': 'https://www.law.go.kr/DRF/lawService.do'
        }

        # 기본 법률 데이터 target 코드
        self.basic_targets = {
            'law': {'name': '현행법령(공포일)', 'key': 'law'},
            'eflaw': {'name': '현행법령(시행일)', 'key': 'eflaw'},
            'prec': {'name': '판례', 'key': 'prec'},
            'admrul': {'name': '행정규칙', 'key': 'admrul'},
            'ordin': {'name': '자치법규', 'key': 'ordin'},
            'detc': {'name': '헌재결정례', 'key': 'detc'},
            'expc': {'name': '법령해석례', 'key': 'expc'},
            'decc': {'name': '행정심판례', 'key': 'decc'},
            'trty': {'name': '조약', 'key': 'trty'},
        }

        # 위원회 결정문 target 코드
        self.committee_targets = {
            'ppc': {'name': '개인정보보호위원회', 'key': 'ppc'},
            'eiac': {'name': '고용보험심사위원회', 'key': 'eiac'},
            'ftc': {'name': '공정거래위원회', 'key': 'ftc'},
            'acr': {'name': '국민권익위원회', 'key': 'acr'},
            'fsc': {'name': '금융위원회', 'key': 'fsc'},
            'nlrc': {'name': '노동위원회', 'key': 'nlrc'},
            'kcc': {'name': '방송미디어통신위원회', 'key': 'kcc'},
            'iaciac': {'name': '산업재해보상보험재심사위원회', 'key': 'iaciac'},
            'oclt': {'name': '중앙토지수용위원회', 'key': 'oclt'},
            'ecc': {'name': '중앙환경분쟁조정위원회', 'key': 'ecc'},
            'sfc': {'name': '증권선물위원회', 'key': 'sfc'},
            'nhrck': {'name': '국가인권위원회', 'key': 'nhrck'},
        }

        # 부처별 법령해석 target 코드
        self.ministry_targets = {
            'moelCgmExpc': {'name': '고용노동부 법령해석', 'key': 'moelCgmExpc'},
            'molitCgmExpc': {'name': '국토교통부 법령해석', 'key': 'molitCgmExpc'},
            'moefCgmExpc': {'name': '기획재정부 법령해석', 'key': 'moefCgmExpc'},
            'mofCgmExpc': {'name': '해양수산부 법령해석', 'key': 'mofCgmExpc'},
            'moisCgmExpc': {'name': '행정안전부 법령해석', 'key': 'moisCgmExpc'},
            'meCgmExpc': {'name': '기후에너지환경부 법령해석', 'key': 'meCgmExpc'},
            'kcsCgmExpc': {'name': '관세청 법령해석', 'key': 'kcsCgmExpc'},
            'ntsCgmExpc': {'name': '국세청 법령해석', 'key': 'ntsCgmExpc'},
            'moeCgmExpc': {'name': '교육부 법령해석', 'key': 'moeCgmExpc'},
            'msitCgmExpc': {'name': '과학기술정보통신부 법령해석', 'key': 'msitCgmExpc'},
            'mpvaCgmExpc': {'name': '국가보훈부 법령해석', 'key': 'mpvaCgmExpc'},
            'mndCgmExpc': {'name': '국방부 법령해석', 'key': 'mndCgmExpc'},
            'mafraCgmExpc': {'name': '농림축산식품부 법령해석', 'key': 'mafraCgmExpc'},
            'mcstCgmExpc': {'name': '문화체육관광부 법령해석', 'key': 'mcstCgmExpc'},
            'mojCgmExpc': {'name': '법무부 법령해석', 'key': 'mojCgmExpc'},
            'mohwCgmExpc': {'name': '보건복지부 법령해석', 'key': 'mohwCgmExpc'},
            'motieCgmExpc': {'name': '산업통상자원부 법령해석', 'key': 'motieCgmExpc'},
            'mogefCgmExpc': {'name': '성평등가족부 법령해석', 'key': 'mogefCgmExpc'},
            'mofaCgmExpc': {'name': '외교부 법령해석', 'key': 'mofaCgmExpc'},
            'mssCgmExpc': {'name': '중소벤처기업부 법령해석', 'key': 'mssCgmExpc'},
            'mouCgmExpc': {'name': '통일부 법령해석', 'key': 'mouCgmExpc'},
            'molegCgmExpc': {'name': '법제처 법령해석', 'key': 'molegCgmExpc'},
            'mfdsCgmExpc': {'name': '식품의약품안전처 법령해석', 'key': 'mfdsCgmExpc'},
            'mpmCgmExpc': {'name': '인사혁신처 법령해석', 'key': 'mpmCgmExpc'},
            'kmaCgmExpc': {'name': '기상청 법령해석', 'key': 'kmaCgmExpc'},
            'khsCgmExpc': {'name': '국가유산청 법령해석', 'key': 'khsCgmExpc'},
            'rdaCgmExpc': {'name': '농촌진흥청 법령해석', 'key': 'rdaCgmExpc'},
            'npaCgmExpc': {'name': '경찰청 법령해석', 'key': 'npaCgmExpc'},
            'dapaCgmExpc': {'name': '방위사업청 법령해석', 'key': 'dapaCgmExpc'},
            'mmaCgmExpc': {'name': '병무청 법령해석', 'key': 'mmaCgmExpc'},
            'kfsCgmExpc': {'name': '산림청 법령해석', 'key': 'kfsCgmExpc'},
            'nfaCgmExpc': {'name': '소방청 법령해석', 'key': 'nfaCgmExpc'},
            'okaCgmExpc': {'name': '재외동포청 법령해석', 'key': 'okaCgmExpc'},
            'ppsCgmExpc': {'name': '조달청 법령해석', 'key': 'ppsCgmExpc'},
            'kdcaCgmExpc': {'name': '질병관리청 법령해석', 'key': 'kdcaCgmExpc'},
            'kostatCgmExpc': {'name': '국가데이터처 법령해석', 'key': 'kostatCgmExpc'},
            'kipoCgmExpc': {'name': '지식재산처 법령해석', 'key': 'kipoCgmExpc'},
            'kcgCgmExpc': {'name': '해양경찰청 법령해석', 'key': 'kcgCgmExpc'},
            'naaccCgmExpc': {'name': '행정중심복합도시건설청 법령해석', 'key': 'naaccCgmExpc'},
        }

        # 특별행정심판례 target 코드
        self.special_tribunal_targets = {
            'ttSpecialDecc': {'name': '조세심판원 특별행정심판례', 'key': 'ttSpecialDecc'},
            'kmstSpecialDecc': {'name': '해양안전심판원 특별행정심판례', 'key': 'kmstSpecialDecc'},
            'acrSpecialDecc': {'name': '국민권익위원회 특별행정심판례', 'key': 'acrSpecialDecc'},
            'adapSpecialDecc': {'name': '인사혁신처 소청심사위원회 재결례', 'key': 'adapSpecialDecc'},
        }

        # 사건번호/안건번호 패턴 정규식
        # 판례 사건번호 패턴: 2020다12345, 2021구합12345, 2019노1234 등
        self.prec_case_pattern = re.compile(
            r'(\d{2,4})[\s]*'  # 연도 (2자리 또는 4자리)
            r'([가-힣]{1,4})'  # 사건유형 (다, 구합, 노, 가합 등)
            r'[\s]*(\d+)'  # 사건번호
            r'(?:,?\s*(\d+))?'  # 병합사건 (선택)
        )

        # 법령해석례 안건번호 패턴: 18-0701, 22-0123, 2018-0701 등
        self.expc_case_pattern = re.compile(
            r'(\d{2,4})'  # 연도 (2자리 또는 4자리)
            r'[-\s]?'  # 구분자
            r'(\d{3,5})'  # 안건번호
        )

    def detect_case_number(self, query: str) -> Dict[str, Any]:
        """사용자 입력에서 사건번호/안건번호 패턴 감지

        Returns:
            Dict with keys:
                - 'type': 'prec' (판례), 'expc' (법령해석례), 'decc' (행정심판례), or None
                - 'case_numbers': 감지된 사건번호 리스트
                - 'formatted': API용 포맷된 사건번호
        """
        result = {
            'type': None,
            'case_numbers': [],
            'formatted': None,
            'original': query
        }

        query_clean = query.strip()

        # 1. 판례 사건번호 패턴 검사 (예: 2020다12345, 2021구합12345)
        prec_matches = self.prec_case_pattern.findall(query_clean)
        if prec_matches:
            case_numbers = []
            for match in prec_matches:
                year, case_type, number, merged = match
                # 2자리 연도를 4자리로 변환
                if len(year) == 2:
                    year = '20' + year if int(year) < 50 else '19' + year
                case_no = f"{year}{case_type}{number}"
                if merged:
                    case_no += f",{merged}"
                case_numbers.append(case_no)

            if case_numbers:
                result['type'] = 'prec'
                result['case_numbers'] = case_numbers
                # API용 nb 파라미터 형식
                result['formatted'] = ','.join(case_numbers)
                return result

        # 2. 법령해석례 안건번호 패턴 검사 (예: 18-0701, 22-0123)
        # 패턴: XX-XXXX 형식 (하이픈 포함)
        expc_pattern_strict = re.compile(r'(\d{2,4})-(\d{3,5})')
        expc_matches = expc_pattern_strict.findall(query_clean)
        if expc_matches:
            case_numbers = []
            formatted_numbers = []
            for year, number in expc_matches:
                # 원본 형식 유지
                original = f"{year}-{number}"
                case_numbers.append(original)
                # API용 itmno 파라미터 형식 (하이픈 제거)
                formatted_numbers.append(f"{year}{number}")

            if case_numbers:
                result['type'] = 'expc'
                result['case_numbers'] = case_numbers
                # 법령해석례는 하이픈 제거한 형식
                result['formatted'] = formatted_numbers[0]  # 첫 번째만
                return result

        # 3. 행정심판례 패턴 - 일반적으로 "중행심 XXXX-XXXXX" 형식
        # 또는 "XXXX-XXXXX" 형식의 숫자만 있는 경우
        decc_pattern = re.compile(r'(\d{4})-(\d{4,6})')
        decc_matches = decc_pattern.findall(query_clean)
        if decc_matches and not expc_matches:  # expc와 구분
            case_numbers = [f"{year}-{number}" for year, number in decc_matches]
            result['type'] = 'decc'
            result['case_numbers'] = case_numbers
            result['formatted'] = case_numbers[0]
            return result

        return result

    async def search_by_case_number(self, case_info: Dict[str, Any]) -> Dict:
        """사건번호/안건번호로 직접 검색

        Args:
            case_info: detect_case_number()의 반환값

        Returns:
            검색 결과 Dict
        """
        if not case_info.get('type') or not case_info.get('formatted'):
            return {}

        case_type = case_info['type']
        formatted = case_info['formatted']
        api_key = self.law_api_key or get_law_api_key()

        if not api_key:
            logger.warning("법제처 API 키가 없습니다.")
            return {}

        results = {case_type: []}

        try:
            async with aiohttp.ClientSession() as session:
                if case_type == 'prec':
                    # 판례: nb 파라미터로 사건번호 검색
                    params = {
                        'OC': api_key,
                        'target': 'prec',
                        'type': 'JSON',
                        'nb': formatted,  # 사건번호
                        'display': 20
                    }
                    logger.info(f"판례 사건번호 검색: nb={formatted}")

                elif case_type == 'expc':
                    # 법령해석례: itmno 파라미터로 안건번호 검색
                    params = {
                        'OC': api_key,
                        'target': 'expc',
                        'type': 'JSON',
                        'itmno': formatted,  # 안건번호 (하이픈 제거)
                        'display': 20
                    }
                    logger.info(f"법령해석례 안건번호 검색: itmno={formatted}")

                elif case_type == 'decc':
                    # 행정심판례: query로 사건번호 검색 (직접 파라미터 없음)
                    params = {
                        'OC': api_key,
                        'target': 'decc',
                        'type': 'JSON',
                        'query': case_info['case_numbers'][0],  # 사건번호로 검색
                        'display': 20
                    }
                    logger.info(f"행정심판례 사건번호 검색: query={case_info['case_numbers'][0]}")
                else:
                    return {}

                async with session.get(
                    self.api_endpoints['search'],
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        text = await response.text()
                        try:
                            data = json.loads(text)
                            logger.info(f"[{case_type}] 사건번호 검색 응답: {list(data.keys())}")

                            # 결과 추출
                            items = self._extract_search_results(data, case_type)
                            results[case_type] = items
                            logger.info(f"[{case_type}] 사건번호 검색 결과: {len(items)}건")

                        except json.JSONDecodeError as e:
                            logger.error(f"사건번호 검색 JSON 파싱 오류: {e}")
                    else:
                        logger.error(f"사건번호 검색 API 오류: {response.status}")

        except Exception as e:
            logger.error(f"사건번호 검색 오류: {e}")

        return results

    def _extract_search_results(self, data: Dict, target: str) -> List[Dict]:
        """API 응답에서 검색 결과 배열 추출"""
        results = []

        # API 응답 구조에서 실제 데이터 찾기
        wrapper_keys = [
            f'{target.capitalize()}Search',
            target.capitalize(),
            f'{target.upper()}Search',
            target,
            target.lower(),
            target.upper(),
        ]

        inner_data = data
        for wkey in wrapper_keys:
            if wkey in data and isinstance(data[wkey], dict):
                inner_data = data[wkey]
                break
            elif wkey in data and isinstance(data[wkey], list):
                return data[wkey]

        if isinstance(inner_data, dict):
            data_keys = [target.lower(), target, target.capitalize()]
            for dkey in data_keys:
                if dkey in inner_data:
                    value = inner_data[dkey]
                    if isinstance(value, list) and len(value) > 0:
                        return value

            # 첫 번째 리스트 찾기
            skip_keys = {'totalCnt', 'page', 'target', 'section', '키워드',
                        'resultMsg', 'resultCode', 'numOfRows'}
            for key, value in inner_data.items():
                if key not in skip_keys and isinstance(value, list) and len(value) > 0:
                    return value

        return results

    def extract_keywords(self, user_input: str) -> List[str]:
        """사용자 입력에서 법률 관련 핵심 키워드 추출 - 법률명/조문 우선"""
        # 불용어 정의 (일반적인 단어, 조사, 구어체 표현 등)
        stopwords = [
            # 조사
            '은', '는', '이', '가', '을', '를', '의', '에', '에서', '으로', '로',
            '와', '과', '도', '만', '뿐', '까지', '부터', '에게', '한테', '께',
            # 어미/서술어
            '입니다', '합니다', '있습니다', '없습니다', '됩니다', '습니다',
            '하는', '되는', '있는', '없는', '한', '된', '할', '될',
            '했습니다', '했다', '한다', '하다', '이다',
            # 접속어
            '것', '수', '때', '등', '및', '또는', '그리고', '하지만', '그러나',
            # 의문사
            '어떻게', '무엇', '어디', '언제', '누구', '왜', '어떤',
            # 부사
            '좀', '잘', '더', '매우', '정말', '아주', '너무', '많이', '현재', '지금',
            # 대명사/지시어
            '저', '제', '나', '내', '우리', '저희', '그', '그녀', '그들',
            '이것', '저것', '그것', '여기', '거기', '저기',
            # 일반 명사 (법률과 무관)
            '현행', '해당', '요건', '확인', '검색', '관련', '판례',
            '사람', '경우', '상황', '문제', '부분', '방법', '정도',
            # 구어체/문장 표현
            '있을까요', '될까요', '인가요', '일까요', '할까요', '싶습니다',
            '엄마로써', '아빠로써', '찾아가자', '보러', '몇번', '자기에게',
            '있다고', '한다고', '라고', '다고',
        ]

        keywords = []

        # 1. 법률명 패턴 추출 (XX법, XX령, XX규칙 등) - 최우선
        law_patterns = re.findall(r'[가-힣]+(?:법|령|규칙|조례|규정|지침|고시)', user_input)
        for law in law_patterns:
            if len(law) >= 3 and law not in keywords:
                keywords.append(law)

        # 2. 조문 관련 추출 (제X조, 제X항 등)
        article_patterns = re.findall(r'제\d+조(?:의\d+)?(?:제\d+항)?', user_input)
        keywords.extend(article_patterns)

        # 3. 구체적인 법률 용어 (복합 키워드 우선)
        specific_legal_terms = [
            # 가족법/이혼 관련 - 중요!
            '면접교섭권', '면접교섭', '양육권', '친권', '양육비',
            '접근금지', '접근금지가처분', '가처분', '임시처분',
            '이혼소송', '협의이혼', '재판이혼', '재산분할청구',
            '위자료청구', '양육자지정', '친권자지정',
            # 금융/대부업 관련
            '대부업', '대부중개업', '매입채권추심업', '매입채권', '추심업',
            '자기자본', '자본금', '등록요건', '영업요건',
            '금융업', '여신업', '신용정보', '채권추심', '채권매입',
            '등록기준', '허가기준', '인가기준', '자본요건',
            # 노동법 관련
            '부당해고', '해고무효', '부당노동행위', '근로계약',
            '해고예고', '정리해고', '징계해고', '퇴직금청구',
            # 부동산/임대차 관련
            '임대차보호', '상가임대차', '주택임대차', '전세보증금',
            '명도소송', '임대차계약', '보증금반환', '계약갱신',
            # 민사 일반
            '손해배상', '불법행위', '채무불이행', '계약위반',
            '손해배상청구', '위약금', '지체상금',
            # 개인정보/공정거래
            '개인정보보호', '정보주체', '개인정보처리',
            '공정거래', '독점규제', '불공정거래', '시장지배적지위',
        ]

        for term in specific_legal_terms:
            if term in user_input and term not in keywords:
                keywords.append(term)

        # 4. 일반 법률 키워드
        general_legal_keywords = [
            # 가족법
            '이혼', '양육', '친권', '면접', '교섭', '위자료', '재산분할',
            '가처분', '임시', '처분', '접근', '금지', '배우자', '자녀',
            '상속', '유언', '증여', '유류분',
            # 노동법
            '해고', '임금', '퇴직금', '근로', '노동', '계약', '위반',
            # 민사
            '손해배상', '불법행위', '계약해지', '계약해제',
            '임대차', '전세', '월세', '보증금', '명도', '인도',
            # 소송
            '형사', '민사', '행정', '소송', '재판', '항소', '상고', '판결',
            # 형사
            '사기', '횡령', '배임', '폭행', '상해', '명예훼손',
            # 지재권
            '저작권', '특허', '상표', '영업비밀', '지식재산',
            '개인정보', '정보보호', '프라이버시',
            # 조세
            '세금', '조세', '부가세', '소득세', '법인세', '상속세', '증여세',
            # 행정
            '건축', '인허가', '허가', '신고', '등록', '면허',
            # 기타
            '교통사고', '산재', '산업재해', '보험', '보상',
            '파산', '회생', '도산', '채무', '채권', '담보', '저당', '압류',
            '해제', '취소', '무효', '철회', '해지',
            '위임', '대리', '보증', '연대보증'
        ]

        for kw in general_legal_keywords:
            if kw in user_input and kw not in keywords:
                keywords.append(kw)

        # 5. 남은 명사 추출 (2글자 이상, 4글자 이상 우선)
        words = re.findall(r'[가-힣]{2,}', user_input)
        long_words = [w for w in words if len(w) >= 4]
        short_words = [w for w in words if len(w) >= 2 and len(w) < 4]

        for word in long_words + short_words:
            is_stopword = any(word.endswith(sw) or word == sw for sw in stopwords)
            if not is_stopword and word not in keywords:
                keywords.append(word)

        # 6. 중복 제거 및 상위 키워드 반환
        unique_keywords = list(dict.fromkeys(keywords))
        return unique_keywords[:10]  # 최대 10개 키워드

    def analyze_query_with_ai(self, user_input: str) -> Dict:
        """AI를 사용하여 사용자 질의 의도 파악 및 검색 키워드 생성

        사용자의 법률 검토 질의를 분석하여:
        1. 질의 의도와 법적 쟁점 파악
        2. 관련 법령, 판례, 유권해석 검색을 위한 최적 키워드 생성
        3. 검색 우선순위 및 추천 검색 유형 제안
        """
        # 기본 키워드 먼저 추출 (fallback용)
        basic_keywords = self.extract_keywords(user_input)
        default_result = {
            'intent': '법률 정보 검색',
            'legal_issues': [],
            'law_names': [],
            'keywords': basic_keywords,
            'search_queries': basic_keywords[:3] if basic_keywords else [user_input],
            'search_priority': {
                'laws': True,
                'precedents': True,
                'interpretations': True,
                'committee_decisions': False,
                'ministry_opinions': False
            },
            'recommended_sources': ['law', 'prec']
        }

        client = get_openai_client()
        if not client:
            logger.info("OpenAI 클라이언트 없음 - 기본 키워드 사용")
            return default_result

        try:
            prompt = f"""당신은 한국 법률 검색 및 법률 검토 전문가입니다.
사용자의 법률 검토 질의를 깊이 분석하여 법제처 Open API 검색에 최적화된 정보를 추출해주세요.

## 사용자 질의
{user_input}

## 분석 지침
1. 질의에서 **핵심 법적 쟁점**을 정확히 파악하세요.
2. 배경 설명이 아닌 **실제 질문**에 집중하세요.
3. 관련 법률 조문과 법적 개념을 정확히 식별하세요.
4. 검색어는 핵심 쟁점에 직접 관련된 것만 생성하세요.

## 분석 요청
위 질의를 분석하여 아래 JSON 형식으로만 응답하세요 (다른 설명 없이):

{{
    "intent": "질문의 핵심 의도를 한 문장으로",
    "core_question": "사용자가 실제로 알고 싶은 법적 질문",
    "legal_issues": ["핵심 법적 쟁점 리스트 - 질문과 직접 관련된 것만"],
    "law_names": ["관련 법률명 (민법, 가사소송법 등)"],
    "legal_concepts": ["관련 법적 개념 (면접교섭권, 가처분 등)"],
    "keywords": ["핵심 검색 키워드 5개"],
    "search_queries": ["법제처 API 검색어 5개 - 핵심 쟁점 중심"],
    "search_priority": {{
        "laws": true/false,
        "precedents": true/false,
        "interpretations": true/false,
        "committee_decisions": true/false,
        "ministry_opinions": true/false
    }},
    "recommended_sources": ["law", "prec", "expc", "decc"]
}}

## 검색어 생성 규칙 (매우 중요!)
⚠️ 법제처 API는 단순 키워드 매칭입니다. 긴 문장은 검색되지 않습니다!

1. **검색어는 반드시 2-3단어 이하로 짧게**:
   - ❌ "민법 면접교섭권 이혼소송 중 임시처분" (너무 김)
   - ✅ "면접교섭권", "면접교섭 가처분" (짧고 핵심적)

2. **단어 조합 패턴**:
   - 법률명 단독: "민법", "가사소송법"
   - 핵심개념 단독: "면접교섭권", "양육권"
   - 2단어 조합: "면접교섭 가처분", "양육권 임시처분"

3. **피해야 할 것**:
   - 문장형 쿼리 금지
   - 조사(의, 에, 를) 사용 금지
   - 4단어 이상 조합 금지

## 예시 1 - 가족법
질의: "이혼 소송 중 접근금지 가처분 상태에서 면접교섭권을 받을 수 있나요?"
응답:
{{
    "intent": "접근금지 가처분과 면접교섭권의 관계 확인",
    "core_question": "접근금지 가처분이 있어도 자녀 면접교섭권을 행사할 수 있는지",
    "legal_issues": ["면접교섭권의 법적 성격", "접근금지 가처분의 효력 범위", "이혼소송 중 임시 면접교섭"],
    "law_names": ["민법", "가사소송법", "가사소송규칙"],
    "legal_concepts": ["면접교섭권", "접근금지가처분", "양육권", "임시처분"],
    "keywords": ["면접교섭권", "접근금지", "가처분", "양육권", "이혼"],
    "search_queries": ["면접교섭권", "면접교섭 가처분", "접근금지 가처분", "양육권", "임시처분"],
    "search_priority": {{
        "laws": true,
        "precedents": true,
        "interpretations": true,
        "committee_decisions": false,
        "ministry_opinions": false
    }},
    "recommended_sources": ["law", "prec", "expc"]
}}

## 예시 2 - 행정법
질의: "대부업법상 자기자본 요건이 얼마인가요?"
응답:
{{
    "intent": "대부업 등록요건 중 자기자본 기준 확인",
    "core_question": "대부업 등록에 필요한 자기자본 금액",
    "legal_issues": ["대부업 등록요건", "자기자본 산정기준"],
    "law_names": ["대부업법", "대부업법 시행령"],
    "legal_concepts": ["자기자본", "등록요건", "대부업"],
    "keywords": ["대부업", "자기자본", "등록요건"],
    "search_queries": ["대부업법 자기자본", "대부업 등록요건", "대부업법 시행령 자본"],
    "search_priority": {{
        "laws": true,
        "precedents": false,
        "interpretations": true,
        "committee_decisions": false,
        "ministry_opinions": true
    }},
    "recommended_sources": ["law", "expc", "admrul"]
}}
"""

            response = client.chat.completions.create(
                model="gpt-5.1",
                messages=[
                    {"role": "system", "content": "당신은 한국 법률 검색 전문가입니다. JSON 형식으로만 응답합니다."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=800
            )

            result_text = response.choices[0].message.content.strip()
            logger.info(f"AI 원본 응답: {result_text[:200]}")

            # JSON 파싱 시도
            try:
                # JSON 블록 추출
                if '```json' in result_text:
                    json_str = result_text.split('```json')[1].split('```')[0].strip()
                elif '```' in result_text:
                    json_str = result_text.split('```')[1].split('```')[0].strip()
                elif result_text.startswith('{'):
                    json_str = result_text
                else:
                    # JSON 시작 위치 찾기
                    start_idx = result_text.find('{')
                    end_idx = result_text.rfind('}') + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = result_text[start_idx:end_idx]
                    else:
                        raise ValueError("JSON 형식을 찾을 수 없음")

                result = json.loads(json_str)

                # 필수 필드 확인 및 보정
                if 'search_queries' not in result or not result['search_queries']:
                    result['search_queries'] = result.get('keywords', basic_keywords)[:5]
                if 'keywords' not in result or not result['keywords']:
                    result['keywords'] = basic_keywords
                if 'legal_issues' not in result:
                    result['legal_issues'] = []
                if 'legal_concepts' not in result:
                    result['legal_concepts'] = []
                if 'core_question' not in result:
                    result['core_question'] = result.get('intent', '')
                if 'search_priority' not in result:
                    result['search_priority'] = default_result['search_priority']
                if 'recommended_sources' not in result:
                    result['recommended_sources'] = default_result['recommended_sources']

                logger.info(f"AI 의도 분석 결과: intent={result.get('intent')}, core_question={result.get('core_question')}")
                logger.info(f"검색 쿼리: {result.get('search_queries')}")
                return result

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"AI 응답 JSON 파싱 실패: {e}, 응답: {result_text[:200]}")
                return default_result

        except Exception as e:
            logger.error(f"AI 의도 분석 오류: {e}")
            return default_result

    def verify_search_results(self, user_query: str, ai_analysis: Dict, results: List[Dict], category: str) -> List[Dict]:
        """AI를 사용하여 검색 결과의 관련성을 검증하고 필터링

        Args:
            user_query: 원본 사용자 질의
            ai_analysis: AI 의도 분석 결과
            results: 검색된 결과 리스트
            category: 결과 카테고리 (prec, expc, decc 등)

        Returns:
            관련성이 높은 결과만 필터링된 리스트
        """
        if not results:
            return results

        client = get_openai_client()
        if not client:
            return results  # AI 없으면 원본 반환

        # 결과가 너무 많으면 상위 20개만 검증
        results_to_verify = results[:20]

        # 결과 요약 생성
        result_summaries = []
        for idx, item in enumerate(results_to_verify):
            title = item.get('사건명') or item.get('제목') or item.get('판례명') or item.get('안건명') or ''
            case_no = item.get('사건번호') or item.get('안건번호') or ''
            summary = f"{idx+1}. {title} ({case_no})"
            result_summaries.append(summary)

        results_text = "\n".join(result_summaries)

        try:
            prompt = f"""당신은 법률 검색 결과 분석 전문가입니다.

## 사용자 질의
{user_query}

## 핵심 질문
{ai_analysis.get('core_question', ai_analysis.get('intent', ''))}

## 핵심 법적 쟁점
{', '.join(ai_analysis.get('legal_issues', []))}

## 핵심 법적 개념
{', '.join(ai_analysis.get('legal_concepts', []))}

## 검색된 {category} 결과
{results_text}

## 요청
위 검색 결과 중에서 사용자 질의의 핵심 쟁점과 **직접적으로 관련**있는 결과의 번호만 선택하세요.

응답 형식 (JSON만):
{{"relevant_indices": [1, 3, 5], "reason": "선택 이유 한 줄"}}

판단 기준:
1. 제목/사건명이 핵심 법적 쟁점과 직접 관련되어야 함
2. 배경 설명(이혼, 소송 등)만 일치하는 것은 관련 없음으로 처리
3. 예: 질문이 "면접교섭권"이면 재산분할/위자료 판례는 관련 없음
"""

            response = client.chat.completions.create(
                model="gpt-5.1",
                messages=[
                    {"role": "system", "content": "법률 검색 결과 관련성 평가 전문가입니다. JSON 형식으로만 응답합니다."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=500
            )

            result_text = response.choices[0].message.content.strip()
            logger.info(f"AI 검증 결과: {result_text[:200]}")

            # JSON 파싱
            try:
                if '{' in result_text:
                    start_idx = result_text.find('{')
                    end_idx = result_text.rfind('}') + 1
                    json_str = result_text[start_idx:end_idx]
                    verification = json.loads(json_str)

                    relevant_indices = verification.get('relevant_indices', [])
                    if relevant_indices:
                        # 1-indexed를 0-indexed로 변환
                        filtered_results = [
                            results_to_verify[i-1] for i in relevant_indices
                            if 1 <= i <= len(results_to_verify)
                        ]
                        logger.info(f"AI 검증: {len(results_to_verify)}개 중 {len(filtered_results)}개 관련 결과 선택")
                        logger.info(f"선택 이유: {verification.get('reason', '')}")
                        return filtered_results if filtered_results else results[:5]

            except (json.JSONDecodeError, ValueError, IndexError) as e:
                logger.warning(f"AI 검증 결과 파싱 실패: {e}")

        except Exception as e:
            logger.error(f"AI 검색 결과 검증 오류: {e}")

        return results  # 오류 시 원본 반환

    async def _search_by_target(self, session, query: str, target: str,
                                display: int = 10) -> List[Dict]:
        """특정 target으로 검색"""
        # API 키 재확인
        api_key = self.law_api_key or get_law_api_key()
        if not api_key:
            logger.warning(f"법제처 API 키가 없습니다. ({target} 검색 불가)")
            return []

        params = {
            'OC': api_key,
            'target': target,
            'query': query,
            'type': 'JSON',
            'display': display
        }

        try:
            async with session.get(
                self.api_endpoints['search'],
                params=params,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    text = await response.text()
                    try:
                        data = json.loads(text)
                        logger.info(f"[{target}] API 응답 키: {list(data.keys())}")

                        # 결과 추출 - 다양한 응답 형식 처리
                        results = []

                        # API 응답 구조: {'PrecSearch': {'prec': [...], '키워드': '...'}}
                        # 또는 {'Expc': {'expc': [...], ...}}

                        # 1. 최상위 래퍼 키 확인 (PrecSearch, LawSearch, Expc, Decc 등)
                        wrapper_keys = [
                            f'{target.capitalize()}Search',  # PrecSearch, LawSearch
                            target.capitalize(),  # Prec, Expc, Decc
                            f'{target.upper()}Search',
                            target,
                            target.lower(),
                            target.upper(),
                        ]

                        inner_data = data
                        for wkey in wrapper_keys:
                            if wkey in data and isinstance(data[wkey], dict):
                                inner_data = data[wkey]
                                break
                            elif wkey in data and isinstance(data[wkey], list):
                                results = data[wkey]
                                break

                        # 2. inner_data에서 실제 데이터 배열 추출
                        if not results and isinstance(inner_data, dict):
                            # target 이름과 일치하는 키에서 배열 찾기
                            data_keys = [
                                target.lower(),  # prec, expc, decc
                                target,
                                target.capitalize(),
                            ]

                            for dkey in data_keys:
                                if dkey in inner_data:
                                    value = inner_data[dkey]
                                    if isinstance(value, list) and len(value) > 0:
                                        results = value
                                        break

                            # 3. 그래도 없으면 inner_data에서 첫 번째 리스트 찾기
                            if not results:
                                skip_keys = {'totalCnt', 'page', 'target', 'section', '키워드',
                                           'resultMsg', 'resultCode', 'numOfRows'}
                                for key, value in inner_data.items():
                                    if key not in skip_keys:
                                        if isinstance(value, list) and len(value) > 0:
                                            results = value
                                            break

                        logger.info(f"[{target}] 검색 결과: {len(results)}건 (쿼리: {query})")
                        # 디버깅: 첫 번째 결과의 구조 출력
                        if results and len(results) > 0:
                            first_item = results[0]
                            logger.info(f"[{target}] 첫 번째 결과 키: {list(first_item.keys()) if isinstance(first_item, dict) else type(first_item)}")
                            logger.info(f"[{target}] 첫 번째 결과 내용: {str(first_item)[:500]}")
                        return results

                    except json.JSONDecodeError as e:
                        logger.error(f"JSON 파싱 오류 ({target}): {e}")
                        logger.error(f"응답 내용: {text[:500]}")
                        return []
                else:
                    logger.error(f"API 응답 오류 ({target}): 상태코드 {response.status}")
        except asyncio.TimeoutError:
            logger.error(f"API 타임아웃 ({target})")
        except Exception as e:
            logger.error(f"검색 오류 ({target}): {e}")
        return []

    async def search_basic_legal_data(self, query: str, search_queries: List[str] = None) -> Dict:
        """기본 법률 데이터 검색 (법령, 판례, 행정규칙 등) - AI 검색어 기반"""
        # 검색 결과 수 설정 (판례, 유권해석 중심으로 대폭 증가)
        display_counts = {
            'law': 30,        # 현행법령(공포일)
            'eflaw': 30,      # 현행법령(시행일)
            'prec': 50,       # 판례 - 최대한 많이
            'admrul': 20,     # 행정규칙
            'ordin': 20,      # 자치법규
            'detc': 30,       # 헌재결정례
            'expc': 50,       # 법령해석례 - 최대한 많이
            'decc': 50,       # 행정심판례 - 최대한 많이
            'trty': 10,       # 조약
        }

        all_results = {target: [] for target in self.basic_targets.keys()}

        # 메인 쿼리로 검색
        async with aiohttp.ClientSession() as session:
            tasks = []
            for target_code in self.basic_targets.keys():
                display = display_counts.get(target_code, 20)
                tasks.append(self._search_by_target(session, query, target_code, display))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for idx, target_code in enumerate(self.basic_targets.keys()):
                if not isinstance(results[idx], Exception) and results[idx]:
                    all_results[target_code].extend(results[idx])

        # AI가 생성한 추가 검색어로 확장 검색 (판례, 법령해석례, 행정심판례, 법령 대상)
        if search_queries:
            important_targets = ['prec', 'expc', 'decc', 'detc', 'law', 'eflaw']
            for search_query in search_queries[:3]:  # 상위 3개 검색어만
                if search_query != query:  # 메인 쿼리와 다른 경우만
                    async with aiohttp.ClientSession() as session:
                        tasks = []
                        for target_code in important_targets:
                            tasks.append(self._search_by_target(session, search_query, target_code, 20))

                        kw_results = await asyncio.gather(*tasks, return_exceptions=True)

                        for idx, target_code in enumerate(important_targets):
                            if not isinstance(kw_results[idx], Exception) and kw_results[idx]:
                                # 중복 제거하며 추가
                                existing_ids = {item.get('판례일련번호', item.get('안건번호', item.get('사건번호', '')))
                                              for item in all_results[target_code]}
                                for item in kw_results[idx]:
                                    item_id = item.get('판례일련번호', item.get('안건번호', item.get('사건번호', '')))
                                    if item_id and item_id not in existing_ids:
                                        all_results[target_code].append(item)
                                        existing_ids.add(item_id)

        return all_results

    async def search_committee_decisions(self, query: str,
                                        selected_committees: List[str] = None) -> Dict:
        """위원회 결정문 검색"""
        if selected_committees is None:
            selected_committees = list(self.committee_targets.keys())

        async with aiohttp.ClientSession() as session:
            tasks = []
            for committee in selected_committees:
                if committee in self.committee_targets:
                    tasks.append(self._search_by_target(session, query, committee, 10))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            valid_committees = [c for c in selected_committees if c in self.committee_targets]
            return {
                valid_committees[idx]: results[idx] if not isinstance(results[idx], Exception) else []
                for idx in range(len(valid_committees))
            }

    async def search_ministry_interpretations(self, query: str,
                                             selected_ministries: List[str] = None) -> Dict:
        """부처별 법령해석 검색"""
        if selected_ministries is None:
            # 주요 부처만 기본 검색
            selected_ministries = [
                'moelCgmExpc', 'molitCgmExpc', 'moisCgmExpc',
                'mohwCgmExpc', 'molegCgmExpc'
            ]

        async with aiohttp.ClientSession() as session:
            tasks = []
            for ministry in selected_ministries:
                if ministry in self.ministry_targets:
                    tasks.append(self._search_by_target(session, query, ministry, 10))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            valid_ministries = [m for m in selected_ministries if m in self.ministry_targets]
            return {
                valid_ministries[idx]: results[idx] if not isinstance(results[idx], Exception) else []
                for idx in range(len(valid_ministries))
            }

    async def search_special_tribunals(self, query: str) -> Dict:
        """특별행정심판례 검색"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for target_code in self.special_tribunal_targets.keys():
                tasks.append(self._search_by_target(session, query, target_code, 10))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            return {
                target_code: results[idx] if not isinstance(results[idx], Exception) else []
                for idx, target_code in enumerate(self.special_tribunal_targets.keys())
            }

    async def comprehensive_search(self, query: str,
                                  search_options: Dict = None) -> Dict:
        """종합 법률 검색 - AI 의도 분석 기반 검색 + 사건번호 직접 검색

        AI가 사용자 질의를 분석하여:
        1. 법적 쟁점 파악
        2. 관련 법령, 판례, 유권해석 검색어 생성
        3. 최적의 검색 소스 추천

        사건번호/안건번호가 감지되면 직접 검색도 수행
        """
        if search_options is None:
            search_options = {
                'basic': True,
                'committees': [],
                'ministries': [],
                'special_tribunals': True
            }

        # 1. 사건번호/안건번호 패턴 감지
        case_info = self.detect_case_number(query)
        case_number_results = {}
        is_case_number_only = False  # 사건번호만 입력된 경우 플래그

        if case_info.get('type'):
            logger.info(f"=== 사건번호/안건번호 감지 ===")
            logger.info(f"유형: {case_info['type']}, 번호: {case_info['case_numbers']}")
            # 사건번호로 직접 검색
            case_number_results = await self.search_by_case_number(case_info)
            case_count = sum(len(v) for v in case_number_results.values())
            logger.info(f"사건번호 검색 결과: {case_count}건")

            # 사건번호만 입력된 경우 판단 (다른 키워드가 거의 없는 경우)
            # 사건번호 패턴을 제거한 후 남은 텍스트가 법원명 등만 있는지 확인
            remaining_text = query
            for case_no in case_info['case_numbers']:
                remaining_text = remaining_text.replace(case_no, '')
            # 법원명, 공백, 기타 짧은 키워드만 남은 경우
            court_names = ['대법원', '고등법원', '지방법원', '행정법원', '가정법원',
                          '서울', '부산', '대구', '인천', '광주', '대전', '울산', '수원', '청주']
            for court in court_names:
                remaining_text = remaining_text.replace(court, '')
            remaining_text = remaining_text.strip()
            # 남은 텍스트가 거의 없으면 사건번호 전용 검색
            if len(remaining_text) <= 5:
                is_case_number_only = True
                logger.info("사건번호 전용 검색 모드 활성화")

        # 2. AI를 사용하여 질의 의도 분석 및 검색 키워드 생성
        # 사건번호만 입력된 경우 AI 분석 생략
        if is_case_number_only:
            ai_analysis = {
                'intent': '특정 사건번호 검색',
                'legal_issues': [],
                'law_names': [],
                'keywords': case_info.get('case_numbers', []),
                'search_queries': case_info.get('case_numbers', []),
                'search_priority': {'laws': False, 'precedents': True, 'interpretations': True,
                                   'committee_decisions': False, 'ministry_opinions': False},
                'recommended_sources': [case_info.get('type', 'prec')]
            }
            keywords = case_info.get('case_numbers', [])
            search_queries = []  # 일반 검색 쿼리는 빈 리스트
        else:
            ai_analysis = self.analyze_query_with_ai(query)
            keywords = ai_analysis.get('keywords', [])
            search_queries = ai_analysis.get('search_queries', [query])
        law_names = ai_analysis.get('law_names', [])
        legal_issues = ai_analysis.get('legal_issues', [])
        search_priority = ai_analysis.get('search_priority', {})
        recommended_sources = ai_analysis.get('recommended_sources', [])

        logger.info(f"=== AI 법률 검토 분석 결과 ===")
        logger.info(f"의도: {ai_analysis.get('intent', '알 수 없음')}")
        logger.info(f"법적 쟁점: {legal_issues}")
        logger.info(f"관련 법령: {law_names}")
        logger.info(f"검색 키워드: {keywords}")
        logger.info(f"생성된 검색 쿼리: {search_queries}")
        logger.info(f"추천 검색 소스: {recommended_sources}")

        results = {
            'query': query,
            'keywords': keywords,
            'search_queries': search_queries,
            'ai_analysis': ai_analysis,
            'legal_issues': legal_issues,
            'law_names': law_names,
            'search_time': datetime.now().isoformat(),
            'case_info': case_info,  # 사건번호 정보 추가
            'is_case_number_only': is_case_number_only,  # 사건번호 전용 검색 여부
            'basic': {},
            'committees': {},
            'ministries': {},
            'special_tribunals': {}
        }

        # 사건번호 전용 검색 모드일 때는 일반 키워드 검색 건너뜀
        if not is_case_number_only:
            # AI가 생성한 검색 쿼리 사용 (없으면 원본 쿼리 사용)
            primary_query = search_queries[0] if search_queries else query
            # 추가 검색어도 활용 (최대 3개)
            additional_queries = search_queries[1:4] if len(search_queries) > 1 else []

            tasks = []

            # 기본 법률 데이터 검색 (AI 생성 검색어 기반)
            if search_options.get('basic', True):
                tasks.append(('basic', self.search_basic_legal_data(primary_query, search_queries)))

            # 위원회 결정문 검색 (AI가 추천하거나 사용자가 선택한 경우)
            committees = search_options.get('committees', [])
            if committees:
                tasks.append(('committees', self.search_committee_decisions(primary_query, committees)))

            # 부처별 법령해석 검색 (AI가 추천하거나 사용자가 선택한 경우)
            ministries = search_options.get('ministries', [])
            if ministries:
                tasks.append(('ministries', self.search_ministry_interpretations(primary_query, ministries)))

            # 특별행정심판례 검색
            if search_options.get('special_tribunals', False):
                tasks.append(('special_tribunals', self.search_special_tribunals(primary_query)))

            # 병렬 실행
            for key, task in tasks:
                try:
                    results[key] = await task
                except Exception as e:
                    logger.error(f"검색 오류 ({key}): {e}")
                    results[key] = {}
        else:
            logger.info("사건번호 전용 검색: 일반 키워드 검색 건너뜀")

        # 사건번호 검색 결과를 basic에 병합 (중복 제거)
        if case_number_results:
            for case_type, items in case_number_results.items():
                if items:
                    if case_type not in results['basic']:
                        results['basic'][case_type] = []
                    # 기존 결과와 병합 (사건번호 검색 결과 우선)
                    existing_ids = set()
                    for item in results['basic'][case_type]:
                        item_id = item.get('판례일련번호', item.get('법령해석례일련번호',
                                  item.get('행정심판재결례일련번호', '')))
                        if item_id:
                            existing_ids.add(str(item_id))

                    for item in items:
                        item_id = item.get('판례일련번호', item.get('법령해석례일련번호',
                                  item.get('행정심판재결례일련번호', '')))
                        if not item_id or str(item_id) not in existing_ids:
                            results['basic'][case_type].insert(0, item)  # 앞에 추가
                            if item_id:
                                existing_ids.add(str(item_id))

                    logger.info(f"[{case_type}] 사건번호 검색 결과 {len(items)}건 병합 완료")

        # AI를 사용하여 검색 결과 검증 및 필터링
        logger.info("=== AI 검색 결과 검증 시작 ===")

        # 기본 검색 결과 검증 (판례, 법령해석례, 행정심판례)
        if results.get('basic'):
            for category in ['prec', 'expc', 'decc', 'detc']:
                if results['basic'].get(category):
                    original_count = len(results['basic'][category])
                    results['basic'][category] = self.verify_search_results(
                        query, ai_analysis, results['basic'][category], category
                    )
                    filtered_count = len(results['basic'][category])
                    if original_count != filtered_count:
                        logger.info(f"[{category}] 검증 완료: {original_count}건 → {filtered_count}건")

        # 위원회 결정문 검증
        if results.get('committees'):
            for comm_key, comm_results in results['committees'].items():
                if comm_results:
                    original_count = len(comm_results)
                    results['committees'][comm_key] = self.verify_search_results(
                        query, ai_analysis, comm_results, f"위원회결정문({comm_key})"
                    )
                    filtered_count = len(results['committees'][comm_key])
                    if original_count != filtered_count:
                        logger.info(f"[{comm_key}] 검증 완료: {original_count}건 → {filtered_count}건")

        # 부처별 법령해석 검증
        if results.get('ministries'):
            for ministry_key, ministry_results in results['ministries'].items():
                if ministry_results:
                    original_count = len(ministry_results)
                    results['ministries'][ministry_key] = self.verify_search_results(
                        query, ai_analysis, ministry_results, f"부처법령해석({ministry_key})"
                    )
                    filtered_count = len(results['ministries'][ministry_key])
                    if original_count != filtered_count:
                        logger.info(f"[{ministry_key}] 검증 완료: {original_count}건 → {filtered_count}건")

        logger.info("=== AI 검색 결과 검증 완료 ===")

        return results

    async def get_detail(self, target: str, item_id: str) -> Dict:
        """상세 정보 조회"""
        params = {
            'OC': self.law_api_key,
            'target': target,
            'ID': item_id,
            'type': 'JSON'
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.api_endpoints['service'],
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status == 200:
                        return await response.json()
        except Exception as e:
            logger.error(f"상세 조회 오류: {e}")
        return {}

    def format_document_as_markdown(self, detail_data: Dict, target: str) -> str:
        """상세 정보를 마크다운 형식으로 변환"""
        if not detail_data:
            return ""

        # 응답 데이터 파싱 (중첩 구조 처리)
        data = detail_data
        if isinstance(detail_data, dict):
            # JSON 응답 구조에 따라 데이터 추출
            for key in ['prec', 'expc', 'decc', 'detc', 'PrecService', 'ExpcService', 'DeccService', 'DetcService']:
                if key in detail_data:
                    data = detail_data[key]
                    break

        md_content = []

        if target == 'prec':  # 판례
            title = self._get_value(data, '사건명', 'caseName', '제목')
            case_no = self._get_value(data, '사건번호', 'caseNo')
            court = self._get_value(data, '법원명', 'courtName')
            date = self._get_value(data, '선고일자', 'judgmentDate')
            case_type = self._get_value(data, '사건종류명', 'caseTypeName')
            verdict_type = self._get_value(data, '판결유형', 'verdictType')

            md_content.append(f"# {title or '판례'}")
            md_content.append("")
            md_content.append("## 기본 정보")
            md_content.append(f"- **사건번호**: {case_no or '-'}")
            md_content.append(f"- **법원**: {court or '-'}")
            md_content.append(f"- **선고일자**: {date or '-'}")
            md_content.append(f"- **사건종류**: {case_type or '-'}")
            md_content.append(f"- **판결유형**: {verdict_type or '-'}")
            md_content.append("")

            # 판시사항
            summary = self._get_value(data, '판시사항', 'judgmentSummary')
            if summary:
                md_content.append("## 판시사항")
                md_content.append(self._clean_html(summary))
                md_content.append("")

            # 판결요지
            gist = self._get_value(data, '판결요지', 'judgmentGist')
            if gist:
                md_content.append("## 판결요지")
                md_content.append(self._clean_html(gist))
                md_content.append("")

            # 참조조문
            ref_law = self._get_value(data, '참조조문', 'referenceLaw')
            if ref_law:
                md_content.append("## 참조조문")
                md_content.append(self._clean_html(ref_law))
                md_content.append("")

            # 참조판례
            ref_prec = self._get_value(data, '참조판례', 'referencePrec')
            if ref_prec:
                md_content.append("## 참조판례")
                md_content.append(self._clean_html(ref_prec))
                md_content.append("")

            # 판례내용 (본문)
            content = self._get_value(data, '판례내용', 'judgmentContent', 'content')
            if content:
                md_content.append("## 판례 본문")
                md_content.append(self._clean_html(content))
                md_content.append("")

        elif target == 'expc':  # 법령해석례
            title = self._get_value(data, '안건명', 'title', '제목')
            case_no = self._get_value(data, '안건번호', 'caseNo')
            org = self._get_value(data, '해석기관명', '회신기관명', 'replyOrg')
            date = self._get_value(data, '해석일자', '회신일자', 'replyDate')
            query_org = self._get_value(data, '질의기관명', 'queryOrg')

            md_content.append(f"# {title or '법령해석례'}")
            md_content.append("")
            md_content.append("## 기본 정보")
            md_content.append(f"- **안건번호**: {case_no or '-'}")
            md_content.append(f"- **해석기관**: {org or '-'}")
            md_content.append(f"- **질의기관**: {query_org or '-'}")
            md_content.append(f"- **해석일자**: {date or '-'}")
            md_content.append("")

            # 질의요지
            query = self._get_value(data, '질의요지', 'queryGist')
            if query:
                md_content.append("## 질의요지")
                md_content.append(self._clean_html(query))
                md_content.append("")

            # 회답
            answer = self._get_value(data, '회답', 'answer')
            if answer:
                md_content.append("## 회답")
                md_content.append(self._clean_html(answer))
                md_content.append("")

            # 이유
            reason = self._get_value(data, '이유', 'reason')
            if reason:
                md_content.append("## 이유")
                md_content.append(self._clean_html(reason))
                md_content.append("")

        elif target == 'decc':  # 행정심판례
            title = self._get_value(data, '사건명', 'caseName', '제목')
            case_no = self._get_value(data, '사건번호', 'caseNo')
            ruling_type = self._get_value(data, '재결례유형명', 'rulingType')
            ruling_date = self._get_value(data, '의결일자', 'decisionDate')
            disp_date = self._get_value(data, '처분일자', 'dispositionDate')
            disp_org = self._get_value(data, '처분청', 'dispositionOrg')
            ruling_org = self._get_value(data, '재결청', 'rulingOrg')

            md_content.append(f"# {title or '행정심판례'}")
            md_content.append("")
            md_content.append("## 기본 정보")
            md_content.append(f"- **사건번호**: {case_no or '-'}")
            md_content.append(f"- **재결유형**: {ruling_type or '-'}")
            md_content.append(f"- **의결일자**: {ruling_date or '-'}")
            md_content.append(f"- **처분일자**: {disp_date or '-'}")
            md_content.append(f"- **처분청**: {disp_org or '-'}")
            md_content.append(f"- **재결청**: {ruling_org or '-'}")
            md_content.append("")

            # 주문
            order = self._get_value(data, '주문', 'mainText')
            if order:
                md_content.append("## 주문")
                md_content.append(self._clean_html(order))
                md_content.append("")

            # 청구취지
            purpose = self._get_value(data, '청구취지', 'claimPurpose')
            if purpose:
                md_content.append("## 청구취지")
                md_content.append(self._clean_html(purpose))
                md_content.append("")

            # 이유
            reason = self._get_value(data, '이유', 'reason')
            if reason:
                md_content.append("## 이유")
                md_content.append(self._clean_html(reason))
                md_content.append("")

            # 재결요지
            gist = self._get_value(data, '재결요지', 'rulingGist')
            if gist:
                md_content.append("## 재결요지")
                md_content.append(self._clean_html(gist))
                md_content.append("")

        elif target == 'detc':  # 헌재결정례
            title = self._get_value(data, '사건명', 'caseName', '결정명', '제목')
            case_no = self._get_value(data, '사건번호', 'caseNo')
            date = self._get_value(data, '종국일자', '선고일자', 'decisionDate')
            result = self._get_value(data, '결정유형', '종국결과', 'decisionType')

            md_content.append(f"# {title or '헌재결정례'}")
            md_content.append("")
            md_content.append("## 기본 정보")
            md_content.append(f"- **사건번호**: {case_no or '-'}")
            md_content.append(f"- **종국일자**: {date or '-'}")
            md_content.append(f"- **결정유형**: {result or '-'}")
            md_content.append("")

            # 판시사항
            summary = self._get_value(data, '판시사항', 'judgmentSummary')
            if summary:
                md_content.append("## 판시사항")
                md_content.append(self._clean_html(summary))
                md_content.append("")

            # 결정요지
            gist = self._get_value(data, '결정요지', 'decisionGist')
            if gist:
                md_content.append("## 결정요지")
                md_content.append(self._clean_html(gist))
                md_content.append("")

            # 본문
            content = self._get_value(data, '결정내용', '본문', 'content')
            if content:
                md_content.append("## 결정 본문")
                md_content.append(self._clean_html(content))
                md_content.append("")
        else:
            # 기타 문서 유형 (위원회 결정문 등)
            title = self._get_value(data, '사건명', '안건명', 'caseName', 'title', '제목')
            md_content.append(f"# {title or '법률 문서'}")
            md_content.append("")

            # 모든 필드를 순회하며 출력
            for key, value in data.items():
                if value and isinstance(value, str) and len(value) > 0:
                    md_content.append(f"## {key}")
                    md_content.append(self._clean_html(str(value)))
                    md_content.append("")

        return "\n".join(md_content)

    def _clean_html(self, text: str) -> str:
        """HTML 태그 제거 및 텍스트 정리"""
        if not text:
            return ""
        # HTML 태그 제거
        import re
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<p\s*/?>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</p>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<[^>]+>', '', text)
        # HTML 엔티티 변환
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&amp;', '&')
        text = text.replace('&quot;', '"')
        # 연속된 공백/줄바꿈 정리
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()

    def generate_pdf_content(self, markdown_content: str, title: str = "법률 문서") -> bytes:
        """마크다운 내용을 PDF로 변환"""
        if not REPORTLAB_AVAILABLE:
            logger.warning("reportlab 모듈이 설치되지 않았습니다. PDF 생성을 건너뜁니다.")
            return b""

        buffer = io.BytesIO()

        # PDF 문서 생성
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=20*mm,
            leftMargin=20*mm,
            topMargin=20*mm,
            bottomMargin=20*mm
        )

        # 한글 폰트 설정 시도
        font_name = 'Helvetica'  # 기본 폰트
        try:
            # 시스템 한글 폰트 경로 시도
            font_paths = [
                '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
                '/usr/share/fonts/nanum/NanumGothic.ttf',
                '/System/Library/Fonts/AppleSDGothicNeo.ttc',
                'C:/Windows/Fonts/malgun.ttf',
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
            ]
            for font_path in font_paths:
                if os.path.exists(font_path):
                    pdfmetrics.registerFont(TTFont('KoreanFont', font_path))
                    font_name = 'KoreanFont'
                    break
        except Exception as e:
            logger.warning(f"한글 폰트 로드 실패: {e}, 기본 폰트 사용")

        # 스타일 설정
        styles = getSampleStyleSheet()

        # 제목 스타일
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontName=font_name,
            fontSize=16,
            spaceAfter=12,
            leading=20
        )

        # 섹션 헤더 스타일
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontName=font_name,
            fontSize=12,
            spaceBefore=12,
            spaceAfter=6,
            leading=16
        )

        # 본문 스타일
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontName=font_name,
            fontSize=10,
            leading=14,
            spaceAfter=8
        )

        # 컨텐츠 빌드
        story = []

        lines = markdown_content.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                story.append(Spacer(1, 6))
            elif line.startswith('# '):
                story.append(Paragraph(line[2:], title_style))
            elif line.startswith('## '):
                story.append(Paragraph(line[3:], heading_style))
            elif line.startswith('- **'):
                # 리스트 항목
                story.append(Paragraph('• ' + line[2:], body_style))
            else:
                # 일반 텍스트
                # XML 특수문자 이스케이프
                escaped = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                try:
                    story.append(Paragraph(escaped, body_style))
                except Exception:
                    # 파싱 오류 시 원본 텍스트 사용
                    story.append(Paragraph(line, body_style))

        try:
            doc.build(story)
        except Exception as e:
            logger.error(f"PDF 생성 오류: {e}")
            return b""

        return buffer.getvalue()

    async def get_documents_for_download(self, items: List[Dict], target: str) -> List[Dict]:
        """여러 문서의 상세 정보를 가져와서 다운로드 형식으로 변환"""
        results = []

        for item in items:
            # ID 추출
            item_id = None
            id_fields = ['판례일련번호', '법령해석례일련번호', '행정심판례일련번호',
                        '헌재결정례일련번호', 'ID', 'id', '일련번호']
            for field in id_fields:
                if field in item:
                    item_id = str(item[field])
                    break

            if not item_id:
                continue

            # 상세 정보 조회
            detail = await self.get_detail(target, item_id)
            if detail:
                md_content = self.format_document_as_markdown(detail, target)

                # 제목 추출
                title = self._get_item_display(item, '사건명', '안건명', '제목', 'caseName', 'title')
                case_no = self._get_value(item, '사건번호', '안건번호', 'caseNo')

                results.append({
                    'id': item_id,
                    'title': title or f"{target}_{item_id}",
                    'case_no': case_no or '',
                    'target': target,
                    'markdown': md_content,
                    'detail': detail
                })

        return results

    def merge_documents_as_markdown(self, documents: List[Dict]) -> str:
        """여러 문서를 하나의 마크다운으로 병합"""
        merged = []
        merged.append("# 법률 문서 모음")
        merged.append(f"\n생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        merged.append(f"\n문서 수: {len(documents)}건")
        merged.append("\n---\n")

        for idx, doc in enumerate(documents, 1):
            merged.append(f"\n## 문서 {idx}: {doc.get('title', '제목 없음')}")
            merged.append(f"사건번호: {doc.get('case_no', '-')}")
            merged.append("\n---\n")
            merged.append(doc.get('markdown', ''))
            merged.append("\n\n---\n")

        return "\n".join(merged)

    def create_fact_sheet(self, user_input: str, legal_data: Dict) -> Dict:
        """사실관계 정리"""
        fact_sheet = {
            'query': user_input,
            'timestamp': datetime.now().isoformat(),
            'statistics': {},
            'key_facts': self._extract_key_facts(user_input),
            'timeline': self._extract_timeline(user_input)
        }

        # 기본 데이터 통계
        if legal_data.get('basic'):
            for key, items in legal_data['basic'].items():
                if items:
                    fact_sheet['statistics'][key] = len(items)

        # 위원회 데이터 통계
        if legal_data.get('committees'):
            for key, items in legal_data['committees'].items():
                if items:
                    fact_sheet['statistics'][f'committee_{key}'] = len(items)

        # 부처 데이터 통계
        if legal_data.get('ministries'):
            for key, items in legal_data['ministries'].items():
                if items:
                    fact_sheet['statistics'][f'ministry_{key}'] = len(items)

        # 특별행정심판 통계
        if legal_data.get('special_tribunals'):
            for key, items in legal_data['special_tribunals'].items():
                if items:
                    fact_sheet['statistics'][f'tribunal_{key}'] = len(items)

        return fact_sheet

    def _extract_key_facts(self, text: str) -> List[str]:
        """핵심 사실 추출"""
        facts = []

        # 날짜 패턴
        date_pattern = r'\d{4}[년\.\-]\d{1,2}[월\.\-]\d{1,2}[일]?'
        dates = re.findall(date_pattern, text)
        for date in dates:
            facts.append(f"관련 일자: {date}")

        # 금액 패턴
        money_pattern = r'\d+[만천백]?\s?원'
        amounts = re.findall(money_pattern, text)
        for amount in amounts:
            facts.append(f"관련 금액: {amount}")

        return facts

    def _extract_timeline(self, text: str) -> List[Dict]:
        """타임라인 추출"""
        timeline = []
        date_pattern = r'(\d{4}[년\.\-]\d{1,2}[월\.\-]\d{1,2}[일]?)'

        sentences = text.split('.')
        for sentence in sentences:
            dates = re.findall(date_pattern, sentence)
            if dates:
                for date in dates:
                    timeline.append({
                        'date': date,
                        'event': sentence.strip()
                    })

        return sorted(timeline, key=lambda x: x['date'])

    # API 필드명 매핑 (camelCase -> 한글)
    FIELD_MAPPING = {
        'evtNm': '사건명',
        'itmNm': '안건명',
        'caseNm': '사건명',
        'caseName': '사건명',
        'caseNo': '사건번호',
        'caseNumber': '사건번호',
        'courtNm': '법원명',
        'courtName': '법원명',
        'judgeDate': '선고일자',
        'judgmentDate': '선고일자',
        'decisionDate': '의결일자',
        'replyDate': '회신일자',
        'replyOrg': '회신기관',
        'lawNm': '법령명',
        'lawName': '법령명',
    }

    # 제외할 값들 (메타데이터, 상태값, 필드명 등)
    SKIP_VALUES = {
        # 상태값
        'success', 'true', 'false', 'null', 'none', 'error', 'ok',
        # 숫자
        '00', '0', '1', '2', '3', '4', '5',
        # camelCase 필드명들
        'evtnm', 'itmnm', 'casenm', 'caseno', 'courtnm', 'lawNm', 'lawnm',
        'casename', 'casenumber', 'courtname', 'judgmentdate', 'decisiondate',
        'replydate', 'replyorg', 'lawname', 'enforcementdate', 'promulgationdate',
        # API 메타데이터 키
        'target', 'type', 'page', 'totalcnt', 'section', 'display', 'sort',
        'query', 'search', 'keyword', 'q',
        # 기타
        'prec', 'expc', 'decc', 'detc', 'law', 'eflaw', 'admrul', 'ordin', 'trty'
    }

    def _is_valid_value(self, value, query: str = '', exclude_urls: bool = True) -> bool:
        """유효한 데이터 값인지 확인

        Args:
            value: 검사할 값
            query: 현재 검색어 (중복 제외용)
            exclude_urls: URL 형식 값 제외 여부 (기본: True)
        """
        if not value:
            return False

        val_str = str(value).strip()

        # 빈 값 체크
        if not val_str:
            return False

        val_lower = val_str.lower()

        # SKIP_VALUES 체크
        if val_lower in self.SKIP_VALUES:
            return False

        # URL 형식 값 제외 (상세링크 등)
        if exclude_urls:
            # 법제처 API 상세링크 패턴: /DRF/lawService.do?... 형식
            if val_str.startswith('/DRF/') or val_str.startswith('http'):
                return False
            # lawService.do, lawSearch.do 등 API 엔드포인트 패턴
            if 'lawService.do' in val_str or 'lawSearch.do' in val_str:
                return False
            # URL 파라미터 패턴: OC=, target=, ID= 등이 포함된 경우
            if re.search(r'[?&](OC|target|ID|type)=', val_str):
                return False

        # 검색어와 동일한 값은 제외 (에코된 검색어)
        if query:
            query_lower = query.strip().lower()
            if val_lower == query_lower:
                return False
            # 검색어가 값에 포함된 경우도 제외 (부분 일치)
            if len(query_lower) > 5 and query_lower in val_lower and len(val_str) < len(query) + 10:
                return False

        # camelCase 패턴 감지 (소문자+대문자 연속)
        if re.match(r'^[a-z]+[A-Z][a-z]+$', val_str):
            return False

        # 너무 짧은 값 제외 (1-2자 숫자)
        if len(val_str) <= 2 and val_str.isdigit():
            return False

        # 영문 소문자로만 된 짧은 값 제외 (필드명일 가능성)
        if len(val_str) <= 10 and val_str.isalpha() and val_str.islower():
            return False

        return True

    def _get_value(self, item: Dict, *keys, default='', query: str = '') -> str:
        """여러 가능한 키에서 값을 찾는 헬퍼 함수"""
        if not isinstance(item, dict):
            if item and self._is_valid_value(item, query):
                return str(item)
            return default

        # 1. 지정된 키에서 찾기 (매핑된 키 포함)
        all_keys = list(keys)
        for key in keys:
            if key in self.FIELD_MAPPING:
                all_keys.append(self.FIELD_MAPPING[key])
            # 역매핑도 확인
            for eng, kor in self.FIELD_MAPPING.items():
                if key == kor:
                    all_keys.append(eng)

        for key in all_keys:
            if key in item:
                val = item[key]
                if self._is_valid_value(val, query):
                    return str(val)

        # 2. 키 이름에 포함된 단어로 찾기 (부분 일치)
        search_terms = ['명', '번호', '일자', 'Nm', 'No', 'Date', 'Name', 'Title']
        for key, value in item.items():
            if self._is_valid_value(value, query):
                for term in search_terms:
                    if term in key:
                        return str(value)

        return default

    def _get_item_display(self, item: Dict, *preferred_keys, query: str = '') -> str:
        """아이템 표시용 문자열 반환 - 안건명/사건명 우선, URL 제외"""
        if not isinstance(item, dict):
            if item and self._is_valid_value(item, query):
                return str(item)
            return '(정보 없음)'

        # 1. 우선 키에서 찾기 (안건명, 사건명, 제목 등)
        all_keys = list(preferred_keys)
        for key in preferred_keys:
            if key in self.FIELD_MAPPING:
                all_keys.append(self.FIELD_MAPPING[key])
            for eng, kor in self.FIELD_MAPPING.items():
                if key == kor:
                    all_keys.append(eng)

        for key in all_keys:
            if key in item:
                val = item[key]
                if self._is_valid_value(val, query):
                    return str(val)

        # 2. 명칭/이름 관련 키 추가 탐색
        name_keys = ['안건명', '사건명', '제목', '판례명', '결정명', '재결례명',
                     '법령명', '법령명한글', '행정규칙명', '자치법규명', '조약명',
                     'caseName', 'title', 'lawName', 'evtNm', 'itmNm', 'caseNm']
        for key in name_keys:
            if key in item and key not in all_keys:
                val = item[key]
                if self._is_valid_value(val, query):
                    return str(val)

        # 3. 유효한 값들 수집 (URL/상세링크 관련 키 제외)
        skip_keys = {
            'target', 'type', 'id', 'page', 'totalcnt', 'section', 'success',
            # 상세링크 관련 키 제외
            '판례상세링크', '법령해석례상세링크', '행정심판례상세링크', '헌재결정례상세링크',
            '상세링크', 'detailLink', 'link', 'url',
            # API 메타데이터
            'oc', 'display', 'sort', 'query', 'keyword', '키워드'
        }
        valid_parts = []
        for key, value in item.items():
            if key.lower() not in skip_keys and self._is_valid_value(value, query):
                # 상세링크 키인지 추가 확인
                if '상세링크' in key or '링크' in key or 'link' in key.lower():
                    continue
                valid_parts.append(str(value))

        return " | ".join(valid_parts[:3]) if valid_parts else '(정보 없음)'

    def _build_context(self, legal_data: Dict) -> str:
        """검색 결과를 컨텍스트로 구성 - 판례/유권해석 중심 확장"""
        context_parts = []

        # 기본 법률 데이터
        if legal_data.get('basic'):
            basic = legal_data['basic']

            # 법령 (상위 15개)
            if basic.get('law') or basic.get('eflaw'):
                laws = (basic.get('law', []) or []) + (basic.get('eflaw', []) or [])
                if laws:
                    context_parts.append(f"\n[관련 법령] (총 {len(laws)}건)")
                    for idx, law in enumerate(laws[:15], 1):
                        name = self._get_value(law, '법령명한글', '법령명', 'lawNameKorean', 'lawName', '법령명약칭')
                        dept = self._get_value(law, '소관부처명', '소관부처', 'competentDept')
                        date = self._get_value(law, '시행일자', '공포일자', 'enforcementDate', 'promulgationDate')
                        if name:
                            context_parts.append(f"{idx}. {name}")
                            if dept:
                                context_parts.append(f"   - 소관부처: {dept}")
                            if date:
                                context_parts.append(f"   - 시행/공포일: {date}")

            # 판례 (상위 30개 - 핵심 자료)
            if basic.get('prec'):
                precs = basic['prec']
                context_parts.append(f"\n[관련 판례] (총 {len(precs)}건) ★ 핵심 자료")
                for idx, prec in enumerate(precs[:30], 1):
                    name = self._get_value(prec, '사건명', '판례명', 'caseName', 'caseNm', '제목')
                    date = self._get_value(prec, '선고일자', '판결일자', 'judgmentDate', 'decisionDate')
                    court = self._get_value(prec, '법원명', '법원', 'courtName', 'court')
                    case_no = self._get_value(prec, '사건번호', 'caseNo', 'caseNumber')
                    if name or case_no:
                        context_parts.append(f"{idx}. {name or '(사건명 없음)'}")
                        if case_no:
                            context_parts.append(f"   - 사건번호: {case_no}")
                        if court:
                            context_parts.append(f"   - 법원: {court}")
                        if date:
                            context_parts.append(f"   - 선고일: {date}")

            # 헌재결정례 (상위 15개)
            if basic.get('detc'):
                detcs = basic['detc']
                context_parts.append(f"\n[헌재결정례] (총 {len(detcs)}건)")
                for idx, case in enumerate(detcs[:15], 1):
                    name = self._get_value(case, '사건명', '결정명', 'caseName', '제목')
                    date = self._get_value(case, '종국일자', '선고일자', '결정일자', 'decisionDate')
                    case_no = self._get_value(case, '사건번호', 'caseNo', 'caseNumber')
                    if name or case_no:
                        context_parts.append(f"{idx}. {name or '(사건명 없음)'}")
                        if case_no:
                            context_parts.append(f"   - 사건번호: {case_no}")
                        if date:
                            context_parts.append(f"   - 종국일: {date}")

            # 법령해석례 (상위 25개 - 핵심 자료)
            if basic.get('expc'):
                expcs = basic['expc']
                context_parts.append(f"\n[법령해석례/유권해석] (총 {len(expcs)}건) ★ 핵심 자료")
                for idx, interp in enumerate(expcs[:25], 1):
                    name = self._get_value(interp, '안건명', '제목', 'title', 'caseName')
                    no = self._get_value(interp, '안건번호', 'caseNo', 'number')
                    org = self._get_value(interp, '회신기관명', '회신기관', 'replyOrg')
                    date = self._get_value(interp, '회신일자', 'replyDate')
                    if name or no:
                        context_parts.append(f"{idx}. {name or '(안건명 없음)'}")
                        if no:
                            context_parts.append(f"   - 안건번호: {no}")
                        if org:
                            context_parts.append(f"   - 회신기관: {org}")
                        if date:
                            context_parts.append(f"   - 회신일자: {date}")

            # 행정심판례 (상위 25개 - 핵심 자료)
            if basic.get('decc'):
                deccs = basic['decc']
                context_parts.append(f"\n[행정심판례] (총 {len(deccs)}건) ★ 핵심 자료")
                for idx, ruling in enumerate(deccs[:25], 1):
                    name = self._get_value(ruling, '사건명', '제목', 'caseName', 'title')
                    date = self._get_value(ruling, '의결일자', '재결일자', 'decisionDate')
                    case_no = self._get_value(ruling, '사건번호', 'caseNo', 'caseNumber')
                    result = self._get_value(ruling, '재결결과', '재결구분명', 'result')
                    if name or case_no:
                        context_parts.append(f"{idx}. {name or '(사건명 없음)'}")
                        if case_no:
                            context_parts.append(f"   - 사건번호: {case_no}")
                        if result:
                            context_parts.append(f"   - 재결결과: {result}")
                        if date:
                            context_parts.append(f"   - 의결일: {date}")

            # 행정규칙 (상위 10개)
            if basic.get('admrul'):
                admruls = basic['admrul']
                context_parts.append(f"\n[행정규칙] (총 {len(admruls)}건)")
                for idx, rule in enumerate(admruls[:10], 1):
                    name = self._get_value(rule, '행정규칙명', '제목', 'ruleName', 'title')
                    dept = self._get_value(rule, '소관부처명', '소관부처', 'competentDept')
                    if name:
                        context_parts.append(f"{idx}. {name}")
                        if dept:
                            context_parts.append(f"   - 소관부처: {dept}")

            # 자치법규 (상위 10개)
            if basic.get('ordin'):
                ordins = basic['ordin']
                context_parts.append(f"\n[자치법규] (총 {len(ordins)}건)")
                for idx, ordin in enumerate(ordins[:10], 1):
                    name = self._get_value(ordin, '자치법규명', '제목', 'ordinName', 'title')
                    local = self._get_value(ordin, '지자체기관명', '자치단체명', 'localGovt')
                    context_parts.append(f"{idx}. {name}")
                    if local:
                        context_parts.append(f"   - 지자체: {local}")

            # 조약 (상위 5개)
            if basic.get('trty'):
                trtys = basic['trty']
                if trtys:
                    context_parts.append(f"\n[조약] (총 {len(trtys)}건)")
                    for idx, treaty in enumerate(trtys[:5], 1):
                        name = treaty.get('조약명', treaty.get('조약명한글', ''))
                        date = treaty.get('체결일자', '')
                        context_parts.append(f"{idx}. {name}")
                        if date:
                            context_parts.append(f"   - 체결일자: {date}")

        # 위원회 결정문
        if legal_data.get('committees'):
            for comm_key, items in legal_data['committees'].items():
                if items:
                    comm_name = self.committee_targets.get(comm_key, {}).get('name', comm_key)
                    context_parts.append(f"\n[{comm_name} 결정문]")
                    for idx, item in enumerate(items[:5], 1):
                        name = item.get('사건명', item.get('안건명', ''))
                        date = item.get('의결일자', item.get('결정일자', ''))
                        context_parts.append(f"{idx}. {name} ({date})")

        # 부처별 법령해석
        if legal_data.get('ministries'):
            for min_key, items in legal_data['ministries'].items():
                if items:
                    min_name = self.ministry_targets.get(min_key, {}).get('name', min_key)
                    context_parts.append(f"\n[{min_name}]")
                    for idx, item in enumerate(items[:5], 1):
                        name = item.get('안건명', item.get('제목', ''))
                        date = item.get('회신일자', item.get('등록일자', ''))
                        context_parts.append(f"{idx}. {name} ({date})")

        # 특별행정심판례
        if legal_data.get('special_tribunals'):
            for trib_key, items in legal_data['special_tribunals'].items():
                if items:
                    trib_name = self.special_tribunal_targets.get(trib_key, {}).get('name', trib_key)
                    context_parts.append(f"\n[{trib_name}]")
                    for idx, item in enumerate(items[:5], 1):
                        name = item.get('사건명', item.get('안건명', ''))
                        date = item.get('재결일자', item.get('의결일자', ''))
                        context_parts.append(f"{idx}. {name} ({date})")

        return "\n".join(context_parts)

    def _generate_fallback_response(self, query: str, legal_data: Dict) -> str:
        """API 키 없을 때 검색 결과 기반 기본 응답"""
        context = self._build_context(legal_data)

        # 통계 계산
        stats = []
        if legal_data.get('basic'):
            for key, items in legal_data['basic'].items():
                if items:
                    name = self.basic_targets.get(key, {}).get('name', key)
                    stats.append(f"{name} {len(items)}건")

        if legal_data.get('committees'):
            for key, items in legal_data['committees'].items():
                if items:
                    name = self.committee_targets.get(key, {}).get('name', key)
                    stats.append(f"{name} {len(items)}건")

        if legal_data.get('ministries'):
            for key, items in legal_data['ministries'].items():
                if items:
                    name = self.ministry_targets.get(key, {}).get('name', key)
                    stats.append(f"{name} {len(items)}건")

        if legal_data.get('special_tribunals'):
            for key, items in legal_data['special_tribunals'].items():
                if items:
                    name = self.special_tribunal_targets.get(key, {}).get('name', key)
                    stats.append(f"{name} {len(items)}건")

        stats_text = ", ".join(stats) if stats else "검색 결과 없음"

        return f"""## 법률 데이터 검색 결과

**질의:** {query}

**검색 통계:** {stats_text}

{context if context else "관련 법률 데이터를 찾지 못했습니다."}

---
⚠️ **안내:** OpenAI API 키가 설정되지 않아 AI 분석 기능을 사용할 수 없습니다.
위 검색 결과는 법제처 Open API에서 가져온 원본 데이터입니다.

AI 분석을 이용하시려면 사이드바에서 OpenAI API 키를 입력해주세요.

⚖️ 본 내용은 참고자료이며, 구체적인 사안에 대해서는 반드시 변호사 등 전문가의 검토가 필요합니다.
"""

        try:
            client = get_openai_client()
            if not client:
                return "AI 응답을 생성할 수 없습니다. OpenAI API 키를 확인해주세요."
            response = client.chat.completions.create(
                model="gpt-5.1",
                messages=[
                    {"role": "system", "content": AI_LAWYER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"AI 응답 생성 오류: {e}")
            return "AI 응답을 생성할 수 없습니다. API 키를 확인해주세요."

    async def generate_legal_advice(self, query: str, legal_data: Dict, fact_sheet: Dict) -> str:
        """AI 법률 조언 생성 - 실제 검색 결과 기반"""
        # API 키 확인
        if not get_openai_api_key():
            return self._generate_fallback_response(query, legal_data)

        context = self._build_context(legal_data)
        timeline = "\n".join([f"- {item['date']}: {item['event']}"
                             for item in fact_sheet.get('timeline', [])])

        # 검색 통계 요약
        stats_summary = self._get_search_stats_summary(legal_data)

        # 추출된 키워드
        keywords = legal_data.get('keywords', [])
        keywords_str = ', '.join(keywords) if keywords else '없음'

        # 검색 결과가 있는지 확인
        has_results = bool(context and context.strip())

        if has_results:
            prompt = f"""당신은 한국 법률 전문가입니다. 아래에 법제처 Open API에서 검색된 **실제 법률 자료**가 제공됩니다.
반드시 이 검색 결과를 기반으로 답변해야 합니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 의뢰인 질문/상황:
{query}

## 추출된 검색 키워드:
{keywords_str}

## 검색 통계:
{stats_summary}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 🔍 법제처에서 검색된 실제 법률 자료:
{context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## ⚠️ 필수 지침 (반드시 준수):
1. **위에 제공된 검색 결과만 사용하세요.** 일반적인 법률 지식으로 답변하지 마세요.
2. **판례를 인용할 때는 반드시 위 목록에서 사건번호를 정확히 복사하세요.**
   예: "대법원 2020다12345 판결에서..."
3. **법령해석례를 인용할 때는 안건번호를 명시하세요.**
   예: "법제처 안건번호 22-0123에 따르면..."
4. **행정심판례를 인용할 때는 사건번호를 명시하세요.**
   예: "중앙행정심판위원회 2023-12345 재결에서..."
5. 위 검색 결과에 없는 내용은 "검색 결과에 포함되지 않음"이라고 명시하세요.

## 답변 형식:

### 📋 핵심 요약
[의뢰인 상황에 대한 2-3문장 핵심 결론]

### 📚 관련 판례 (위 검색 결과에서 인용)
[검색된 판례 목록에서 관련 판례를 선택하여 사건번호와 함께 상세 설명]
- **사건번호**: [위에서 복사]
- **법원/선고일**: [위에서 복사]
- **판시사항**: [내용 설명]
- **의뢰인 사안 적용**: [분석]

### 📋 관련 법령해석례/행정심판례 (위 검색 결과에서 인용)
[검색된 해석례/심판례에서 관련 건을 선택하여 안건번호와 함께 설명]

### 📖 관련 법령
[검색된 법령 중 관련 법령 인용]

### 💡 종합 의견 및 조언
[위 자료들을 종합한 분석]

---
⚖️ 본 내용은 AI가 작성한 참고자료이며, 법률자문이 아닙니다.
구체적인 사안에 대해서는 반드시 변호사 등 전문가의 검토가 필요합니다.
"""
        else:
            prompt = f"""당신은 한국 법률 전문가입니다.

## 의뢰인 질문/상황:
{query}

## 추출된 검색 키워드:
{keywords_str}

## ⚠️ 검색 결과:
법제처 Open API 검색 결과가 없습니다.

## 지침:
1. 검색 결과가 없음을 먼저 안내하세요.
2. 일반적인 법률 정보를 제공하되, "법제처 검색 결과 없음"을 명시하세요.
3. 다른 검색어 제안을 포함하세요.

---
⚖️ 본 내용은 AI가 작성한 참고자료이며, 법률자문이 아닙니다.
구체적인 사안에 대해서는 반드시 변호사 등 전문가의 검토가 필요합니다.
"""

        try:
            client = get_openai_client()
            if not client:
                return self._generate_fallback_response(query, legal_data)
            response = client.chat.completions.create(
                model="gpt-5.1",
                messages=[
                    {"role": "system", "content": AI_LAWYER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=2500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"AI 응답 생성 오류: {e}")
            return self._generate_fallback_response(query, legal_data)

    async def _generate_contract_review(self, query: str, legal_data: Dict, fact_sheet: Dict) -> str:
        """계약서 검토 응답 생성"""
        # API 키 확인
        if not get_openai_api_key():
            return self._generate_fallback_response(query, legal_data)

        context = self._build_context(legal_data)
        timeline = "\n".join([f"- {item['date']}: {item['event']}"
                             for item in fact_sheet.get('timeline', [])])

        # 검색 통계 요약
        stats_summary = self._get_search_stats_summary(legal_data)

        # 추출된 키워드
        keywords = legal_data.get('keywords', [])
        keywords_str = ', '.join(keywords) if keywords else '없음'

        # 검색 결과가 있는지 확인
        has_results = bool(context and context.strip())

        if has_results:
            prompt = f"""당신은 한국 법률 전문가입니다. 아래에 법제처 Open API에서 검색된 **실제 법률 자료**가 제공됩니다.
반드시 이 검색 결과를 기반으로 답변해야 합니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 의뢰인 질문/상황:
{query}

## 추출된 검색 키워드:
{keywords_str}

## 검색 통계:
{stats_summary}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 🔍 법제처에서 검색된 실제 법률 자료:
{context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## ⚠️ 필수 지침 (반드시 준수):
1. **위에 제공된 검색 결과만 사용하세요.** 일반적인 법률 지식으로 답변하지 마세요.
2. **판례를 인용할 때는 반드시 위 목록에서 사건번호를 정확히 복사하세요.**
   예: "대법원 2020다12345 판결에서..."
3. **법령해석례를 인용할 때는 안건번호를 명시하세요.**
   예: "법제처 안건번호 22-0123에 따르면..."
4. **행정심판례를 인용할 때는 사건번호를 명시하세요.**
   예: "중앙행정심판위원회 2023-12345 재결에서..."
5. 위 검색 결과에 없는 내용은 "검색 결과에 포함되지 않음"이라고 명시하세요.

## 답변 형식:

### 📋 핵심 요약
[의뢰인 상황에 대한 2-3문장 핵심 결론]

### 📚 관련 판례 (위 검색 결과에서 인용)
[검색된 판례 목록에서 관련 판례를 선택하여 사건번호와 함께 상세 설명]
- **사건번호**: [위에서 복사]
- **법원/선고일**: [위에서 복사]
- **판시사항**: [내용 설명]
- **의뢰인 사안 적용**: [분석]

### 📋 관련 법령해석례/행정심판례 (위 검색 결과에서 인용)
[검색된 해석례/심판례에서 관련 건을 선택하여 안건번호와 함께 설명]

### 📖 관련 법령
[검색된 법령 중 관련 법령 인용]

### 💡 종합 의견 및 조언
[위 자료들을 종합한 분석]

---
⚖️ 본 내용은 AI가 작성한 참고자료이며, 법률자문이 아닙니다.
구체적인 사안에 대해서는 반드시 변호사 등 전문가의 검토가 필요합니다.
"""
        else:
            prompt = f"""당신은 한국 법률 전문가입니다.

## 의뢰인 질문/상황:
{query}

## 추출된 검색 키워드:
{keywords_str}

## ⚠️ 검색 결과:
법제처 Open API 검색 결과가 없습니다.

## 지침:
1. 검색 결과가 없음을 먼저 안내하세요.
2. 일반적인 법률 정보를 제공하되, "법제처 검색 결과 없음"을 명시하세요.
3. 다른 검색어 제안을 포함하세요.

---
⚖️ 본 내용은 AI가 작성한 참고자료이며, 법률자문이 아닙니다.
구체적인 사안에 대해서는 반드시 변호사 등 전문가의 검토가 필요합니다.
"""

        try:
            client = get_openai_client()
            if not client:
                return self._generate_fallback_response(query, legal_data)
            response = client.chat.completions.create(
                model="gpt-5.1",
                messages=[
                    {"role": "system", "content": AI_LAWYER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=2500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"AI 응답 생성 오류: {e}")
            return self._generate_fallback_response(query, legal_data)

    def _get_search_stats_summary(self, legal_data: Dict) -> str:
        """검색 통계 요약 생성"""
        stats = []

        if legal_data.get('basic'):
            basic = legal_data['basic']
            if basic.get('law') or basic.get('eflaw'):
                laws = (basic.get('law', []) or []) + (basic.get('eflaw', []) or [])
                if laws:
                    stats.append(f"- 법령: {len(laws)}건")
            if basic.get('prec'):
                stats.append(f"- 판례: {len(basic['prec'])}건 ★")
            if basic.get('detc'):
                stats.append(f"- 헌재결정례: {len(basic['detc'])}건")
            if basic.get('expc'):
                stats.append(f"- 법령해석례: {len(basic['expc'])}건 ★")
            if basic.get('decc'):
                stats.append(f"- 행정심판례: {len(basic['decc'])}건 ★")
            if basic.get('admrul'):
                stats.append(f"- 행정규칙: {len(basic['admrul'])}건")
            if basic.get('ordin'):
                stats.append(f"- 자치법규: {len(basic['ordin'])}건")

        if legal_data.get('committees'):
            for key, items in legal_data['committees'].items():
                if items:
                    name = self.committee_targets.get(key, {}).get('name', key)
                    stats.append(f"- {name}: {len(items)}건")

        if legal_data.get('ministries'):
            for key, items in legal_data['ministries'].items():
                if items:
                    name = self.ministry_targets.get(key, {}).get('name', key)
                    stats.append(f"- {name}: {len(items)}건")

        if legal_data.get('special_tribunals'):
            for key, items in legal_data['special_tribunals'].items():
                if items:
                    name = self.special_tribunal_targets.get(key, {}).get('name', key)
                    stats.append(f"- {name}: {len(items)}건")

        return "\n".join(stats) if stats else "검색 결과 없음"

# ===== UI 함수들 =====
def display_chat_message(role: str, content: str):
    """채팅 메시지 표시"""
    if role == "user":
        st.markdown(f'''
        <div class="chat-message user-message">
            <strong>👤 의뢰인:</strong><br>
            {content}
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(content)

def display_search_results_detail(legal_data: Dict, engine: LegalAIEngine, query: str = ''):
    """검색된 판례/유권해석 상세 표시"""
    if not legal_data:
        return

    basic = legal_data.get('basic', {})

    # 판례 상세
    if basic.get('prec'):
        with st.expander(f"📚 검색된 판례 ({len(basic['prec'])}건)", expanded=True):
            for idx, prec in enumerate(basic['prec'][:20], 1):
                display_name = engine._get_item_display(prec, '사건명', '판례명', 'caseName', '제목', query=query)
                case_no = engine._get_value(prec, '사건번호', 'caseNo', 'caseNumber', query=query)
                court = engine._get_value(prec, '법원명', '법원', 'courtName', 'court', query=query)
                date = engine._get_value(prec, '선고일자', '판결일자', 'judgmentDate', 'decisionDate', query=query)
                detail_link = engine._get_value(prec, '판례상세링크', 'detailLink', query=query)
                if display_name and display_name != '(정보 없음)':
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.markdown(f"**{idx}. {display_name}**")
                        if case_no or court or date:
                            st.caption(f"사건번호: {case_no or '-'} | 법원: {court or '-'} | 선고일: {date or '-'}")
                    with col2:
                        if detail_link:
                            full_link = f"https://www.law.go.kr{detail_link}" if detail_link.startswith('/') else detail_link
                            st.markdown(f"[상세보기]({full_link})")

    # 법령해석례 상세
    if basic.get('expc'):
        with st.expander(f"📋 검색된 법령해석례 ({len(basic['expc'])}건)", expanded=True):
            for idx, expc in enumerate(basic['expc'][:20], 1):
                display_name = engine._get_item_display(expc, '안건명', '제목', 'title', 'caseName', query=query)
                no = engine._get_value(expc, '안건번호', 'caseNo', 'number', query=query)
                org = engine._get_value(expc, '회신기관명', '회신기관', 'replyOrg', query=query)
                date = engine._get_value(expc, '회신일자', 'replyDate', query=query)
                detail_link = engine._get_value(expc, '법령해석례상세링크', 'detailLink', query=query)
                if display_name and display_name != '(정보 없음)':
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.markdown(f"**{idx}. {display_name}**")
                        if no or org or date:
                            st.caption(f"안건번호: {no or '-'} | 회신기관: {org or '-'} | 회신일: {date or '-'}")
                    with col2:
                        if detail_link:
                            full_link = f"https://www.law.go.kr{detail_link}" if detail_link.startswith('/') else detail_link
                            st.markdown(f"[상세보기]({full_link})")

    # 행정심판례 상세
    if basic.get('decc'):
        with st.expander(f"⚖️ 검색된 행정심판례 ({len(basic['decc'])}건)", expanded=True):
            for idx, decc in enumerate(basic['decc'][:20], 1):
                display_name = engine._get_item_display(decc, '사건명', '제목', 'caseName', 'title', query=query)
                case_no = engine._get_value(decc, '사건번호', 'caseNo', 'caseNumber', query=query)
                result = engine._get_value(decc, '재결결과', '재결구분명', 'result', query=query)
                date = engine._get_value(decc, '의결일자', '재결일자', 'decisionDate', query=query)
                detail_link = engine._get_value(decc, '행정심판례상세링크', 'detailLink', query=query)
                if display_name and display_name != '(정보 없음)':
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.markdown(f"**{idx}. {display_name}**")
                        if case_no or result or date:
                            st.caption(f"사건번호: {case_no or '-'} | 재결결과: {result or '-'} | 의결일: {date or '-'}")
                    with col2:
                        if detail_link:
                            full_link = f"https://www.law.go.kr{detail_link}" if detail_link.startswith('/') else detail_link
                            st.markdown(f"[상세보기]({full_link})")

    # 헌재결정례 상세
    if basic.get('detc'):
        with st.expander(f"🏛️ 검색된 헌재결정례 ({len(basic['detc'])}건)", expanded=False):
            for idx, detc in enumerate(basic['detc'][:10], 1):
                display_name = engine._get_item_display(detc, '사건명', '결정명', 'caseName', '제목', query=query)
                case_no = engine._get_value(detc, '사건번호', 'caseNo', 'caseNumber', query=query)
                date = engine._get_value(detc, '종국일자', '선고일자', '결정일자', 'decisionDate', query=query)
                detail_link = engine._get_value(detc, '헌재결정례상세링크', 'detailLink', query=query)
                if display_name and display_name != '(정보 없음)':
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.markdown(f"**{idx}. {display_name}**")
                        st.caption(f"사건번호: {case_no or '-'} | 종국일: {date or '-'}")
                    with col2:
                        if detail_link:
                            full_link = f"https://www.law.go.kr{detail_link}" if detail_link.startswith('/') else detail_link
                            st.markdown(f"[상세보기]({full_link})")

    # 위원회 결정문 표시
    committees = legal_data.get('committees', {})
    if committees:
        total_committee = sum(len(items) for items in committees.values() if items)
        if total_committee > 0:
            with st.expander(f"🏢 위원회 결정문 ({total_committee}건)", expanded=False):
                for comm_key, items in committees.items():
                    if items:
                        comm_name = engine.committee_targets.get(comm_key, {}).get('name', comm_key)
                        st.markdown(f"**{comm_name}** ({len(items)}건)")
                        for idx, item in enumerate(items[:10], 1):
                            display_name = engine._get_item_display(item, '사건명', '제목', 'caseName', 'title', query=query)
                            case_no = engine._get_value(item, '사건번호', 'caseNo', query=query)
                            date = engine._get_value(item, '의결일자', '결정일자', 'decisionDate', query=query)
                            detail_link = engine._get_value(item, '상세링크', 'detailLink', query=query)
                            if display_name and display_name != '(정보 없음)':
                                col1, col2 = st.columns([5, 1])
                                with col1:
                                    st.markdown(f"{idx}. {display_name}")
                                    if case_no or date:
                                        st.caption(f"사건번호: {case_no or '-'} | 일자: {date or '-'}")
                                with col2:
                                    if detail_link:
                                        full_link = f"https://www.law.go.kr{detail_link}" if detail_link.startswith('/') else detail_link
                                        st.markdown(f"[상세]({full_link})")
                        st.markdown("---")

    # 부처별 법령해석 표시
    ministries = legal_data.get('ministries', {})
    if ministries:
        total_ministry = sum(len(items) for items in ministries.values() if items)
        if total_ministry > 0:
            with st.expander(f"🏛️ 부처별 법령해석 ({total_ministry}건)", expanded=False):
                for min_key, items in ministries.items():
                    if items:
                        min_name = engine.ministry_targets.get(min_key, {}).get('name', min_key)
                        st.markdown(f"**{min_name}** ({len(items)}건)")
                        for idx, item in enumerate(items[:10], 1):
                            display_name = engine._get_item_display(item, '안건명', '제목', 'title', query=query)
                            no = engine._get_value(item, '안건번호', 'caseNo', query=query)
                            date = engine._get_value(item, '회신일자', 'replyDate', query=query)
                            detail_link = engine._get_value(item, '법령해석례상세링크', 'detailLink', query=query)
                            if display_name and display_name != '(정보 없음)':
                                col1, col2 = st.columns([5, 1])
                                with col1:
                                    st.markdown(f"{idx}. {display_name}")
                                    if no or date:
                                        st.caption(f"안건번호: {no or '-'} | 회신일: {date or '-'}")
                                with col2:
                                    if detail_link:
                                        full_link = f"https://www.law.go.kr{detail_link}" if detail_link.startswith('/') else detail_link
                                        st.markdown(f"[상세]({full_link})")
                        st.markdown("---")

    # 특별행정심판례 표시
    special_tribunals = legal_data.get('special_tribunals', {})
    if special_tribunals:
        total_tribunal = sum(len(items) for items in special_tribunals.values() if items)
        if total_tribunal > 0:
            with st.expander(f"⚖️ 특별행정심판례 ({total_tribunal}건)", expanded=False):
                for trib_key, items in special_tribunals.items():
                    if items:
                        trib_name = engine.special_tribunal_targets.get(trib_key, {}).get('name', trib_key)
                        st.markdown(f"**{trib_name}** ({len(items)}건)")
                        for idx, item in enumerate(items[:10], 1):
                            display_name = engine._get_item_display(item, '사건명', '제목', 'caseName', query=query)
                            case_no = engine._get_value(item, '사건번호', 'caseNo', query=query)
                            date = engine._get_value(item, '재결일자', '의결일자', 'decisionDate', query=query)
                            detail_link = engine._get_value(item, '행정심판례상세링크', 'detailLink', query=query)
                            if display_name and display_name != '(정보 없음)':
                                col1, col2 = st.columns([5, 1])
                                with col1:
                                    st.markdown(f"{idx}. {display_name}")
                                    if case_no or date:
                                        st.caption(f"사건번호: {case_no or '-'} | 재결일: {date or '-'}")
                                with col2:
                                    if detail_link:
                                        full_link = f"https://www.law.go.kr{detail_link}" if detail_link.startswith('/') else detail_link
                                        st.markdown(f"[상세]({full_link})")
                        st.markdown("---")

def display_download_section(legal_data: Dict, engine: LegalAIEngine):
    """문서 다운로드 섹션 표시"""
    if not legal_data:
        return

    basic = legal_data.get('basic', {})

    # 다운로드 가능한 문서 수집
    download_targets = []
    if basic.get('prec'):
        download_targets.append(('prec', '판례', basic['prec']))
    if basic.get('expc'):
        download_targets.append(('expc', '법령해석례', basic['expc']))
    if basic.get('decc'):
        download_targets.append(('decc', '행정심판례', basic['decc']))
    if basic.get('detc'):
        download_targets.append(('detc', '헌재결정례', basic['detc']))

    if not download_targets:
        return

    st.markdown("---")
    st.markdown("### 📥 문서 다운로드")

    # 다운로드 형식 선택
    col1, col2 = st.columns(2)
    with col1:
        download_format = st.selectbox(
            "다운로드 형식",
            options=['markdown', 'pdf'],
            format_func=lambda x: 'Markdown (.md)' if x == 'markdown' else 'PDF (.pdf)',
            key='download_format'
        )
    with col2:
        download_type = st.selectbox(
            "다운로드 유형",
            options=['individual', 'merge'],
            format_func=lambda x: '개별 문서' if x == 'individual' else '병합 문서',
            key='download_type'
        )

    # 문서 선택
    st.markdown("#### 📋 다운로드할 문서 선택")

    # 세션 상태에 선택된 문서 저장
    if 'selected_docs' not in st.session_state:
        st.session_state.selected_docs = {}

    for target_code, target_name, items in download_targets:
        with st.expander(f"{target_name} ({len(items)}건)", expanded=False):
            # 전체 선택 체크박스
            select_all_key = f"select_all_{target_code}"
            select_all = st.checkbox(f"전체 선택", key=select_all_key)

            for idx, item in enumerate(items[:20]):
                # ID 추출
                item_id = None
                id_fields = ['판례일련번호', '법령해석례일련번호', '행정심판례일련번호',
                            '헌재결정례일련번호', 'ID', 'id', '일련번호']
                for field in id_fields:
                    if field in item:
                        item_id = str(item[field])
                        break

                if not item_id:
                    continue

                # 제목 추출
                title = engine._get_item_display(item, '사건명', '안건명', '제목', 'caseName', 'title')
                case_no = engine._get_value(item, '사건번호', '안건번호', 'caseNo')

                doc_key = f"{target_code}_{item_id}"
                default_value = select_all or st.session_state.selected_docs.get(doc_key, False)

                if st.checkbox(
                    f"{idx + 1}. {title or '제목 없음'} ({case_no or '-'})",
                    value=default_value,
                    key=f"doc_{doc_key}"
                ):
                    st.session_state.selected_docs[doc_key] = {
                        'target': target_code,
                        'id': item_id,
                        'title': title,
                        'case_no': case_no,
                        'item': item
                    }
                else:
                    if doc_key in st.session_state.selected_docs:
                        del st.session_state.selected_docs[doc_key]

    # 선택된 문서 수 표시
    selected_count = len([k for k, v in st.session_state.selected_docs.items() if v])
    st.info(f"선택된 문서: {selected_count}건")

    # 다운로드 버튼
    if st.button("📥 선택한 문서 다운로드", type="primary", disabled=selected_count == 0):
        with st.spinner("문서를 가져오는 중..."):
            # 선택된 문서들 가져오기
            selected_items = [v for v in st.session_state.selected_docs.values() if v]

            if not selected_items:
                st.warning("다운로드할 문서를 선택해주세요.")
                return

            # 상세 정보 조회 및 변환
            downloaded_docs = []
            progress_bar = st.progress(0)

            for idx, doc_info in enumerate(selected_items):
                try:
                    detail = asyncio.run(engine.get_detail(doc_info['target'], doc_info['id']))
                    if detail:
                        md_content = engine.format_document_as_markdown(detail, doc_info['target'])
                        downloaded_docs.append({
                            'id': doc_info['id'],
                            'title': doc_info['title'] or f"문서_{doc_info['id']}",
                            'case_no': doc_info['case_no'] or '',
                            'target': doc_info['target'],
                            'markdown': md_content
                        })
                except Exception as e:
                    logger.error(f"문서 조회 실패: {e}")

                progress_bar.progress((idx + 1) / len(selected_items))

            if not downloaded_docs:
                st.error("문서를 가져오는데 실패했습니다.")
                return

            # 다운로드 처리
            if download_type == 'merge':
                # 병합 다운로드
                merged_content = engine.merge_documents_as_markdown(downloaded_docs)

                if download_format == 'markdown':
                    st.download_button(
                        label="💾 병합 문서 다운로드 (MD)",
                        data=merged_content,
                        file_name=f"법률문서_병합_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                else:
                    pdf_content = engine.generate_pdf_content(merged_content, "법률 문서 모음")
                    if pdf_content:
                        st.download_button(
                            label="💾 병합 문서 다운로드 (PDF)",
                            data=pdf_content,
                            file_name=f"법률문서_병합_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                    else:
                        st.warning("PDF 생성에 실패했습니다. Markdown으로 다운로드합니다.")
                        st.download_button(
                            label="💾 병합 문서 다운로드 (MD)",
                            data=merged_content,
                            file_name=f"법률문서_병합_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
            else:
                # 개별 다운로드
                st.markdown("#### 📁 개별 다운로드")
                for doc in downloaded_docs:
                    safe_title = re.sub(r'[<>:"/\\|?*]', '_', doc['title'][:30])

                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{doc['title']}** ({doc['case_no']})")
                    with col2:
                        if download_format == 'markdown':
                            st.download_button(
                                label="📄 MD",
                                data=doc['markdown'],
                                file_name=f"{safe_title}.md",
                                mime="text/markdown",
                                key=f"dl_md_{doc['id']}"
                            )
                        else:
                            pdf_content = engine.generate_pdf_content(doc['markdown'], doc['title'])
                            if pdf_content:
                                st.download_button(
                                    label="📄 PDF",
                                    data=pdf_content,
                                    file_name=f"{safe_title}.pdf",
                                    mime="application/pdf",
                                    key=f"dl_pdf_{doc['id']}"
                                )
                            else:
                                st.download_button(
                                    label="📄 MD",
                                    data=doc['markdown'],
                                    file_name=f"{safe_title}.md",
                                    mime="text/markdown",
                                    key=f"dl_md_fallback_{doc['id']}"
                                )

            st.success(f"✅ {len(downloaded_docs)}건의 문서 준비 완료!")

def display_search_statistics(fact_sheet: Dict, engine: LegalAIEngine):
    """검색 결과 통계 표시"""
    stats = fact_sheet.get('statistics', {})
    if not stats:
        return

    st.markdown("### 📊 검색 결과 통계")

    # 기본 데이터
    basic_stats = {k: v for k, v in stats.items()
                  if not k.startswith(('committee_', 'ministry_', 'tribunal_'))}
    if basic_stats:
        cols = st.columns(4)
        for idx, (key, count) in enumerate(basic_stats.items()):
            name = engine.basic_targets.get(key, {}).get('name', key)
            with cols[idx % 4]:
                st.metric(name, count)

    # 위원회 결정문
    committee_stats = {k.replace('committee_', ''): v for k, v in stats.items()
                      if k.startswith('committee_')}
    if committee_stats:
        st.markdown("#### 위원회 결정문")
        cols = st.columns(4)
        for idx, (key, count) in enumerate(committee_stats.items()):
            name = engine.committee_targets.get(key, {}).get('name', key)
            with cols[idx % 4]:
                st.metric(name, count)

    # 부처별 법령해석
    ministry_stats = {k.replace('ministry_', ''): v for k, v in stats.items()
                     if k.startswith('ministry_')}
    if ministry_stats:
        st.markdown("#### 부처별 법령해석")
        cols = st.columns(4)
        for idx, (key, count) in enumerate(ministry_stats.items()):
            name = engine.ministry_targets.get(key, {}).get('name', key)
            with cols[idx % 4]:
                st.metric(name, count)

async def process_search(query: str, search_options: Dict):
    """검색 처리"""
    engine = LegalAIEngine()

    # 검색 상태 표시 영역
    status_container = st.container()

    with status_container:
        st.info("🔍 법률 데이터 검색을 시작합니다...")
        progress = st.progress(0)

        # API 키 확인
        api_key = get_law_api_key()
        if not api_key:
            st.error("❌ 법제처 API 키가 설정되지 않았습니다. 사이드바에서 API 키를 입력해주세요.")
            return {}, {}, "법제처 API 키가 필요합니다.", engine

        # 1. 종합 검색
        progress.progress(20, "법제처 데이터베이스 검색 중...")
        legal_data = await engine.comprehensive_search(query, search_options)

        # 검색 결과 요약 표시
        basic = legal_data.get('basic', {})
        search_summary = []
        if basic.get('prec'):
            search_summary.append(f"판례 {len(basic['prec'])}건")
        if basic.get('expc'):
            search_summary.append(f"법령해석례 {len(basic['expc'])}건")
        if basic.get('decc'):
            search_summary.append(f"행정심판례 {len(basic['decc'])}건")
        if basic.get('law') or basic.get('eflaw'):
            laws = (basic.get('law', []) or []) + (basic.get('eflaw', []) or [])
            if laws:
                search_summary.append(f"법령 {len(laws)}건")

        if search_summary:
            progress.progress(50, f"검색 완료: {', '.join(search_summary)}")
        else:
            st.warning("⚠️ 검색 결과가 없습니다. 다른 검색어로 시도해보세요.")
            progress.progress(50, "검색 결과 없음")

        # 사건번호 검색 결과 표시
        case_info = legal_data.get('case_info', {})
        is_case_number_only = legal_data.get('is_case_number_only', False)
        if case_info and case_info.get('type'):
            case_type_names = {
                'prec': '판례',
                'expc': '법령해석례',
                'decc': '행정심판례'
            }
            case_type_name = case_type_names.get(case_info['type'], case_info['type'])

            # 사건번호 검색 결과 확인
            case_type = case_info['type']
            case_results = basic.get(case_type, []) if basic else []

            if case_results:
                st.success(f"🔢 **사건번호 검색 성공:** {', '.join(case_info['case_numbers'])} ({case_type_name}) - {len(case_results)}건")
            elif is_case_number_only:
                st.warning(f"⚠️ **사건번호 검색 결과 없음:** {', '.join(case_info['case_numbers'])} ({case_type_name})\n\n"
                          f"해당 사건번호의 {case_type_name}을(를) 법제처 데이터베이스에서 찾을 수 없습니다.\n"
                          f"- 사건번호를 확인해주세요\n"
                          f"- 최근 사건의 경우 아직 등록되지 않았을 수 있습니다")
            else:
                st.info(f"🔢 **사건번호 감지:** {', '.join(case_info['case_numbers'])} ({case_type_name}) - 직접 검색 결과 없음, 키워드 검색 결과 표시")

        # AI 분석 결과 표시 (사건번호 전용 검색이 아닐 때만)
        ai_analysis = legal_data.get('ai_analysis', {})
        if ai_analysis and ai_analysis.get('intent') and not is_case_number_only:
            with st.expander("🤖 AI 질의 분석 결과", expanded=True):
                # 핵심 질문
                core_question = ai_analysis.get('core_question', '')
                if core_question:
                    st.markdown(f"**❓ 핵심 질문:** {core_question}")

                # 의도 분석
                st.markdown(f"**📌 분석된 의도:** {ai_analysis.get('intent', '알 수 없음')}")

                # 법적 쟁점
                legal_issues = ai_analysis.get('legal_issues', [])
                if legal_issues:
                    st.markdown("**⚖️ 파악된 법적 쟁점:**")
                    for issue in legal_issues:
                        st.markdown(f"  - {issue}")

                # 핵심 법적 개념
                legal_concepts = ai_analysis.get('legal_concepts', [])
                if legal_concepts:
                    st.markdown(f"**🔖 핵심 법적 개념:** {', '.join(legal_concepts)}")

                # 관련 법령
                law_names = ai_analysis.get('law_names', [])
                if law_names:
                    st.markdown(f"**📚 관련 법령:** {', '.join(law_names)}")

                # 검색 키워드
                search_queries = ai_analysis.get('search_queries', [])
                if search_queries:
                    st.markdown(f"**🔍 생성된 검색어:** {', '.join(search_queries)}")

                st.info("💡 검색 결과는 AI가 관련성을 검증하여 핵심 쟁점과 직접 관련된 자료만 표시합니다.")

        # 2. 사실관계 정리
        progress.progress(60, "검색 결과 분석 중...")
        fact_sheet = engine.create_fact_sheet(query, legal_data)

        # 3. AI 분석
        progress.progress(80, "AI 분석 중...")
        advice = await engine.generate_legal_advice(query, legal_data, fact_sheet)

        progress.progress(100, "완료!")
        time.sleep(0.5)
        progress.empty()

        # 최종 검색 결과 요약
        if search_summary:
            st.success(f"✅ 검색 완료: {', '.join(search_summary)}")
        else:
            st.warning("검색 결과가 없습니다.")

    return legal_data, fact_sheet, advice, engine

# ===== PDF 번역 UI 함수 =====
def render_pdf_translation_tab():
    """PDF 번역 탭 렌더링"""
    st.header("📄 PDF 문서 번역")
    st.markdown("PDF 문서를 업로드하면 텍스트를 추출하고 번역합니다. (수식은 보존됩니다)")

    if not PDF_TRANSLATOR_AVAILABLE:
        st.error("PDF 번역 모듈을 사용할 수 없습니다. 필요한 패키지를 설치해주세요.")
        st.code("pip install pymupdf Pillow pytesseract reportlab", language="bash")
        return

    # OpenAI API 키 확인
    openai_key = get_openai_api_key()
    if not openai_key:
        st.warning("OpenAI API 키가 설정되지 않았습니다. 사이드바에서 API 키를 입력해주세요.")

    # 번역 설정
    col1, col2 = st.columns(2)
    with col1:
        source_lang = st.selectbox(
            "원본 언어",
            options=["en", "ko", "ja", "zh", "de", "fr", "es", "ru"],
            format_func=lambda x: {
                "en": "영어", "ko": "한국어", "ja": "일본어",
                "zh": "중국어", "de": "독일어", "fr": "프랑스어",
                "es": "스페인어", "ru": "러시아어"
            }.get(x, x),
            index=0
        )
    with col2:
        target_lang = st.selectbox(
            "번역 언어",
            options=["ko", "en", "ja", "zh", "de", "fr", "es", "ru"],
            format_func=lambda x: {
                "en": "영어", "ko": "한국어", "ja": "일본어",
                "zh": "중국어", "de": "독일어", "fr": "프랑스어",
                "es": "스페인어", "ru": "러시아어"
            }.get(x, x),
            index=0
        )

    # 번역 옵션
    col1, col2 = st.columns(2)
    with col1:
        translate_text = st.checkbox("텍스트 블록 번역", value=True,
                                    help="PDF의 텍스트 블록을 추출하여 번역합니다")
    with col2:
        translate_images = st.checkbox("이미지 OCR 번역", value=False,
                                      help="이미지에서 텍스트를 OCR로 추출하여 번역합니다 (Tesseract 필요)")

    st.divider()

    # PDF 파일 업로드
    uploaded_file = st.file_uploader(
        "PDF 파일을 업로드하세요",
        type=["pdf"],
        help="최대 200MB까지 업로드 가능합니다"
    )

    if uploaded_file is not None:
        # 파일 정보 표시
        st.markdown(f"**파일명:** {uploaded_file.name}")
        st.markdown(f"**파일 크기:** {uploaded_file.size / 1024 / 1024:.2f} MB")

        # PDF 정보 미리보기
        pdf_bytes = uploaded_file.read()
        uploaded_file.seek(0)  # 파일 포인터 리셋

        try:
            translator = PDFTranslator(get_openai_client()) if PDF_TRANSLATOR_AVAILABLE else None
            if translator:
                pdf_info = translator.get_pdf_info(pdf_bytes)
            else:
                st.error("PDF 번역기를 초기화할 수 없습니다.")
                return

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("페이지 수", pdf_info['page_count'])
            with col2:
                st.metric("텍스트 블록", pdf_info['text_blocks_count'])
            with col3:
                st.metric("이미지 수", pdf_info['images_count'])

        except Exception as e:
            st.error(f"PDF 분석 오류: {e}")
            return

        st.divider()

        # 번역 실행 버튼
        if st.button("🔄 PDF 번역 시작", type="primary", use_container_width=True):
            if not openai_key:
                st.error("OpenAI API 키를 먼저 설정해주세요.")
                return

            # 진행 상태 표시
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(progress, message="처리 중..."):
                progress_bar.progress(progress)
                status_text.text(message)

            try:
                with st.spinner("PDF 번역 중..."):
                    # 번역 실행
                    translated_bytes = translate_pdf_file(
                        pdf_bytes,
                        openai_client=get_openai_client(),
                        source_lang=source_lang,
                        target_lang=target_lang,
                        translate_text=translate_text,
                        translate_images=translate_images,
                        progress_callback=update_progress
                    )

                progress_bar.progress(100)
                status_text.text("번역 완료!")

                # 다운로드 버튼
                output_filename = f"translated_{uploaded_file.name}"
                st.success("PDF 번역이 완료되었습니다!")

                st.download_button(
                    label="📥 번역된 PDF 다운로드",
                    data=translated_bytes,
                    file_name=output_filename,
                    mime="application/pdf",
                    use_container_width=True
                )

                # 세션에 결과 저장
                st.session_state['translated_pdf'] = translated_bytes
                st.session_state['translated_pdf_name'] = output_filename

            except Exception as e:
                st.error(f"번역 오류: {e}")
                logger.error(f"PDF 번역 실패: {e}")

    # 이전 번역 결과가 있으면 다운로드 버튼 표시
    if 'translated_pdf' in st.session_state and st.session_state.get('translated_pdf'):
        st.divider()
        st.markdown("### 이전 번역 결과")
        st.download_button(
            label="📥 마지막 번역된 PDF 다운로드",
            data=st.session_state['translated_pdf'],
            file_name=st.session_state.get('translated_pdf_name', 'translated.pdf'),
            mime="application/pdf"
        )


# ===== 메인 앱 =====
def main():
    # 헤더
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("⚖️ AI 법률 도우미 & PDF 번역")
        st.markdown("법률 검색 + PDF 문서 번역 서비스")
    with col2:
        st.markdown("""
        <div style="text-align: right; padding: 1rem;">
            <small>v5.0 | GPT-5 + 법제처 API 전체 연동</small>
        </div>
        """, unsafe_allow_html=True)

    # ===== 사이드바 =====
    with st.sidebar:
        st.header("🔑 API 설정")

        # API 키 입력 섹션
        with st.expander("API 키 입력", expanded=not st.session_state.api_keys_set):
            st.markdown("#### 법제처 Open API")
            st.caption("https://open.law.go.kr 에서 발급")
            law_api_input = st.text_input(
                "법제처 API 키",
                value=st.session_state.law_api_key,
                type="password",
                key="law_api_input",
                placeholder="법제처 API 키를 입력하세요"
            )

            st.markdown("#### OpenAI API")
            st.caption("https://platform.openai.com 에서 발급")
            openai_api_input = st.text_input(
                "OpenAI API 키",
                value=st.session_state.openai_api_key,
                type="password",
                key="openai_api_input",
                placeholder="OpenAI API 키를 입력하세요 (선택)"
            )

            if st.button("API 키 저장", use_container_width=True):
                # OpenAI API 키 형식 검증
                if openai_api_input and not openai_api_input.startswith('sk-'):
                    st.error("⚠️ OpenAI API 키는 'sk-'로 시작해야 합니다. https://platform.openai.com 에서 올바른 API 키를 복사해주세요.")
                else:
                    st.session_state.law_api_key = law_api_input
                    st.session_state.openai_api_key = openai_api_input
                    st.session_state.api_keys_set = True
                    st.success("API 키가 저장되었습니다!")
                    st.rerun()

        st.divider()

        # API 상태 표시
        st.header("🔌 API 상태")
        law_key = get_law_api_key()
        openai_key = get_openai_api_key()

        if law_key:
            st.success("✅ 법제처 API 연결됨")
        else:
            st.error("❌ 법제처 API 키 필요")
            st.caption("검색 기능을 사용하려면 법제처 API 키가 필요합니다.")

        if openai_key:
            if openai_key.startswith('sk-'):
                st.success("✅ GPT-5.1 AI 엔진 활성화")
            else:
                st.error("⚠️ OpenAI API 키 형식 오류 (sk-로 시작해야 함)")
        else:
            st.warning("⚠️ OpenAI API 미설정")
            st.caption("AI 분석 없이 검색 결과만 표시됩니다.")

        st.divider()

        # 검색 옵션
        st.header("🔍 검색 옵션")

        # 엔진 초기화 (옵션 표시용)
        engine = LegalAIEngine()

        # 기본 데이터 검색
        search_basic = st.checkbox("📚 기본 법률 데이터", value=True,
                                   help="법령, 판례, 행정규칙, 자치법규, 헌재결정례, 법령해석례, 행정심판례, 조약")

        # 위원회 결정문 - 전체 선택 바깥에 배치
        select_all_comm = st.checkbox("🏢 위원회 결정문 (전체)", key="select_all_comm",
                                      help="공정거래위, 노동위, 금융위 등 12개 위원회")
        if not select_all_comm:
            with st.expander("위원회 개별 선택", expanded=False):
                col1, col2 = st.columns(2)
                committees_list = list(engine.committee_targets.items())
                half = len(committees_list) // 2
                with col1:
                    for key, info in committees_list[:half]:
                        st.checkbox(info['name'], key=f"comm_{key}")
                with col2:
                    for key, info in committees_list[half:]:
                        st.checkbox(info['name'], key=f"comm_{key}")

        # 부처별 법령해석 - 전체 선택 바깥에 배치
        select_all_ministry = st.checkbox("🏛️ 부처별 법령해석 (전체)", key="select_all_ministry",
                                          help="고용노동부, 국토부, 법제처 등 30개 부처")

        if not select_all_ministry:
            # 부처별 법령해석 (주요)
            major_ministries = [
                ('moelCgmExpc', '고용노동부'),
                ('molitCgmExpc', '국토교통부'),
                ('moisCgmExpc', '행정안전부'),
                ('mohwCgmExpc', '보건복지부'),
                ('molegCgmExpc', '법제처'),
                ('mojCgmExpc', '법무부'),
            ]

            select_all_major_min = st.checkbox("  └ 주요 부처 (6개)", key="select_all_major_min")
            if not select_all_major_min:
                with st.expander("주요 부처 개별 선택", expanded=False):
                    for key, name in major_ministries:
                        st.checkbox(name, key=f"min_{key}")

            # 부처별 법령해석 (기타)
            other_ministries = [(k, v['name']) for k, v in engine.ministry_targets.items()
                               if k not in [m[0] for m in major_ministries]]

            select_all_other_min = st.checkbox("  └ 기타 부처", key="select_all_other_min")
            if not select_all_other_min:
                with st.expander("기타 부처 개별 선택", expanded=False):
                    col1, col2 = st.columns(2)
                    for idx, (key, name) in enumerate(other_ministries):
                        with col1 if idx % 2 == 0 else col2:
                            st.checkbox(name, key=f"min_{key}")
        else:
            # 전체 선택 시 major_ministries 변수 정의 필요
            major_ministries = [
                ('moelCgmExpc', '고용노동부'),
                ('molitCgmExpc', '국토교통부'),
                ('moisCgmExpc', '행정안전부'),
                ('mohwCgmExpc', '보건복지부'),
                ('molegCgmExpc', '법제처'),
                ('mojCgmExpc', '법무부'),
            ]
            other_ministries = [(k, v['name']) for k, v in engine.ministry_targets.items()
                               if k not in [m[0] for m in major_ministries]]

        # 특별행정심판례
        search_special_tribunals = st.checkbox(
            "⚖️ 특별행정심판례",
            value=False,
            help="조세심판원, 해양안전심판원, 국민권익위원회, 소청심사위원회"
        )

        st.divider()

        # 새 대화 시작 버튼
        if st.button("🔄 새 검색 시작", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.search_results = None
            st.session_state.fact_sheet = {}
            st.rerun()

    # ===== 메인 컨텐츠 (탭 기반) =====
    tab1, tab2 = st.tabs(["⚖️ 법률 연구", "📄 PDF 번역"])

    # ===== 탭 1: 법률 연구 =====
    with tab1:
        # 웰컴 메시지
        if not st.session_state.chat_history:
            st.markdown("""
            <div class="chat-message assistant-message">
                <strong>⚖️ AI 변호사 (GPT-5):</strong><br>
                안녕하세요, AI 변호사입니다.<br><br>

                <b>🔍 검색 가능한 법률 데이터:</b><br>
                • <b>기본:</b> 법령, 판례, 행정규칙, 자치법규, 헌재결정례, 법령해석례, 행정심판례, 조약<br>
                • <b>위원회 결정문:</b> 공정거래위원회, 노동위원회, 금융위원회 등 12개 위원회<br>
                • <b>부처별 법령해석:</b> 고용노동부, 국토교통부 등 30개 이상 부처<br>
                • <b>특별행정심판:</b> 조세심판원, 해양안전심판원 등<br><br>

                <b>🔢 사건번호/안건번호 직접 검색:</b><br>
                • <b>판례:</b> 2020다12345, 2021구합12345 형식<br>
                • <b>법령해석례:</b> 18-0701, 22-0123 형식<br>
                • <b>행정심판례:</b> 2023-12345 형식<br><br>

                <b>💡 사용 방법:</b><br>
                1. 사이드바에서 API 키를 입력하세요<br>
                2. 검색할 데이터 소스를 선택하세요<br>
                3. 아래 입력창에 검색어 또는 사건번호를 입력하세요<br><br>

                어떤 법률 자료를 찾아드릴까요?
            </div>
            """, unsafe_allow_html=True)
        else:
            # 대화 히스토리 표시
            for msg in st.session_state.chat_history:
                display_chat_message(msg["role"], msg["content"])

        st.divider()

        # 예시 검색어
        st.markdown("### 💡 예시 검색어")
        col1, col2, col3, col4 = st.columns(4)

        examples = {
            "부당해고 구제": "부당해고 구제 절차와 관련 판례",
            "임대차 보증금": "주택임대차보호법 보증금 반환",
            "개인정보 침해": "개인정보 침해 손해배상",
            "18-0701": "18-0701"  # 법령해석례 안건번호 예시
        }

        clicked_example = None
        for idx, (btn_text, query) in enumerate(examples.items()):
            with [col1, col2, col3, col4][idx]:
                if st.button(btn_text, use_container_width=True, key=f"example_{idx}"):
                    clicked_example = query

        # 사용자 입력
        user_input = st.text_area(
            "검색어 입력",
            value=clicked_example if clicked_example else "",
            placeholder="예: 부당해고 구제 절차, 임대차 보증금 반환 판례 등",
            height=100,
            key="search_input"
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            search_button = st.button("🔍 법률 자료 검색", type="primary", use_container_width=True)
        with col2:
            if st.session_state.chat_history:
                if st.button("📄 결과 다운로드"):
                    last_response = st.session_state.chat_history[-1]
                    if last_response["role"] == "assistant":
                        st.download_button(
                            label="💾 다운로드",
                            data=last_response["content"],
                            file_name=f"법률연구_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )

        # 검색 실행
        if search_button or clicked_example:
            query = user_input if user_input else clicked_example

            if not query:
                st.warning("검색어를 입력해주세요.")
            elif not get_law_api_key():
                st.error("법제처 API 키를 입력해주세요.")
            else:
                # 세션 상태에서 선택된 위원회 수집
                engine_for_options = LegalAIEngine()

                # 위원회 전체 선택 체크 시 모든 위원회 선택
                if st.session_state.get("select_all_comm", False):
                    selected_committees = list(engine_for_options.committee_targets.keys())
                else:
                    selected_committees = [
                        key for key in engine_for_options.committee_targets.keys()
                        if st.session_state.get(f"comm_{key}", False)
                    ]

                # 세션 상태에서 선택된 부처 수집
                major_ministry_keys = ['moelCgmExpc', 'molitCgmExpc', 'moisCgmExpc',
                                       'mohwCgmExpc', 'molegCgmExpc', 'mojCgmExpc']
                other_ministry_keys = [k for k in engine_for_options.ministry_targets.keys()
                                       if k not in major_ministry_keys]

                selected_ministries = []

                # 부처 전체 선택 체크 시 모든 부처 선택
                if st.session_state.get("select_all_ministry", False):
                    selected_ministries = list(engine_for_options.ministry_targets.keys())
                else:
                    # 주요 부처 전체 선택 체크 시
                    if st.session_state.get("select_all_major_min", False):
                        selected_ministries.extend(major_ministry_keys)
                    else:
                        selected_ministries.extend([
                            key for key in major_ministry_keys
                            if st.session_state.get(f"min_{key}", False)
                        ])

                    # 기타 부처 전체 선택 체크 시
                    if st.session_state.get("select_all_other_min", False):
                        selected_ministries.extend(other_ministry_keys)
                    else:
                        selected_ministries.extend([
                            key for key in other_ministry_keys
                            if st.session_state.get(f"min_{key}", False)
                        ])

                # 검색 옵션 구성
                search_options = {
                    'basic': search_basic,
                    'committees': selected_committees,
                    'ministries': selected_ministries,
                    'special_tribunals': search_special_tribunals
                }

                # 검색 실행
                legal_data, fact_sheet, advice, engine = asyncio.run(
                    process_search(query, search_options)
                )

                # 결과 저장
                st.session_state.search_results = legal_data
                st.session_state.fact_sheet = fact_sheet

                # 채팅 히스토리에 추가
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": query,
                    "timestamp": datetime.now().isoformat()
                })

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": advice,
                    "legal_data": legal_data,
                    "fact_sheet": fact_sheet,
                    "timestamp": datetime.now().isoformat()
                })

                st.rerun()

        # 검색 통계 표시
        if st.session_state.fact_sheet:
            engine = LegalAIEngine()
            display_search_statistics(st.session_state.fact_sheet, engine)

    # 검색 결과 상세 표시 (판례, 유권해석 등)
    if st.session_state.search_results:
        engine = LegalAIEngine()
        st.markdown("---")
        st.markdown("## 📑 검색된 법률 자료")
        # fact_sheet에서 쿼리 가져오기
        current_query = st.session_state.fact_sheet.get('query', '') if st.session_state.fact_sheet else ''
        display_search_results_detail(st.session_state.search_results, engine, query=current_query)

        # 다운로드 섹션 표시
        display_download_section(st.session_state.search_results, engine)

    # 검색 통계 표시
    if st.session_state.fact_sheet:
        engine = LegalAIEngine()
        display_search_statistics(st.session_state.fact_sheet, engine)

# ===== 앱 실행 =====
if __name__ == "__main__":
    main()
