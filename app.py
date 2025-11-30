"""
AI 법률 연구 도우미 - 판례, 유권해석, 법령 종합 검색 서비스
법제처 API + ChatGPT를 활용한 법률 자료 검색 및 분석

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

# Streamlit 환경에서 asyncio 이벤트 루프 충돌 방지
nest_asyncio.apply()

# 환경변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== 페이지 설정 =====
st.set_page_config(
    page_title="AI 법률 연구 도우미",
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

    def extract_keywords(self, user_input: str) -> List[str]:
        """사용자 입력에서 법률 관련 핵심 키워드 추출"""
        # 불용어 정의
        stopwords = ['은', '는', '이', '가', '을', '를', '의', '에', '에서', '으로', '로',
                    '와', '과', '도', '만', '뿐', '까지', '부터', '에게', '한테', '께',
                    '입니다', '합니다', '있습니다', '없습니다', '됩니다', '습니다',
                    '하는', '되는', '있는', '없는', '한', '된', '할', '될',
                    '것', '수', '때', '등', '및', '또는', '그리고', '하지만', '그러나',
                    '어떻게', '무엇', '어디', '언제', '누구', '왜', '어떤',
                    '좀', '잘', '더', '매우', '정말', '아주', '너무', '많이',
                    '저', '제', '나', '내', '우리', '저희', '그', '그녀', '그들']

        # 법률 관련 중요 키워드 (우선 추출)
        legal_keywords = [
            '해고', '부당해고', '임금', '퇴직금', '근로', '노동', '계약', '위반',
            '손해배상', '불법행위', '채무불이행', '계약해지', '계약해제',
            '임대차', '전세', '월세', '보증금', '명도', '인도',
            '상속', '유언', '증여', '재산분할', '이혼', '위자료', '양육비',
            '형사', '민사', '행정', '소송', '재판', '항소', '상고',
            '사기', '횡령', '배임', '폭행', '상해', '명예훼손',
            '저작권', '특허', '상표', '영업비밀', '지식재산',
            '개인정보', '정보보호', '프라이버시',
            '세금', '조세', '부가세', '소득세', '법인세', '상속세', '증여세',
            '건축', '인허가', '허가', '신고', '등록', '면허',
            '교통사고', '산재', '산업재해', '보험', '보상',
            '파산', '회생', '도산', '채무', '채권', '담보', '저당', '압류',
            '해제', '취소', '무효', '철회', '해지',
            '위임', '대리', '보증', '연대보증'
        ]

        keywords = []

        # 1. 법률 관련 키워드 먼저 추출
        input_lower = user_input.lower()
        for kw in legal_keywords:
            if kw in user_input:
                keywords.append(kw)

        # 2. 명사 추출 (간단한 패턴 매칭)
        # 한글 단어 추출 (2글자 이상)
        words = re.findall(r'[가-힣]{2,}', user_input)
        for word in words:
            # 불용어 제거
            is_stopword = False
            for sw in stopwords:
                if word.endswith(sw) or word == sw:
                    is_stopword = True
                    break
            if not is_stopword and word not in keywords:
                keywords.append(word)

        # 3. 중복 제거 및 상위 키워드 반환
        unique_keywords = list(dict.fromkeys(keywords))
        return unique_keywords[:10]  # 최대 10개 키워드

    async def _search_by_target(self, session, query: str, target: str,
                                display: int = 10) -> List[Dict]:
        """특정 target으로 검색"""
        params = {
            'OC': self.law_api_key,
            'target': target,
            'query': query,
            'type': 'JSON',
            'display': display
        }

        try:
            async with session.get(
                self.api_endpoints['search'],
                params=params,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                if response.status == 200:
                    text = await response.text()
                    try:
                        data = json.loads(text)
                        # 각 target별 응답 키 확인
                        possible_keys = [target, target.lower(),
                                        target.replace('CgmExpc', ''),
                                        target.replace('SpecialDecc', '')]
                        for key in possible_keys:
                            if key in data:
                                return data[key] if isinstance(data[key], list) else [data[key]]
                        # 응답의 첫 번째 키 반환
                        for key, value in data.items():
                            if isinstance(value, list):
                                return value
                        return []
                    except json.JSONDecodeError:
                        return []
        except Exception as e:
            logger.error(f"검색 오류 ({target}): {e}")
        return []

    async def search_basic_legal_data(self, query: str, keywords: List[str] = None) -> Dict:
        """기본 법률 데이터 검색 (법령, 판례, 행정규칙 등) - 확장 검색"""
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

        # 추가 키워드로 확장 검색 (판례, 법령해석례, 행정심판례 대상)
        if keywords:
            important_targets = ['prec', 'expc', 'decc', 'detc']
            for keyword in keywords[:5]:  # 상위 5개 키워드만
                if keyword != query:  # 메인 쿼리와 다른 경우만
                    async with aiohttp.ClientSession() as session:
                        tasks = []
                        for target_code in important_targets:
                            tasks.append(self._search_by_target(session, keyword, target_code, 20))

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
        """종합 법률 검색 - 키워드 추출 및 확장 검색"""
        if search_options is None:
            search_options = {
                'basic': True,
                'committees': [],
                'ministries': [],
                'special_tribunals': True
            }

        # 사용자 입력에서 키워드 추출
        keywords = self.extract_keywords(query)
        logger.info(f"추출된 키워드: {keywords}")

        results = {
            'query': query,
            'keywords': keywords,
            'search_time': datetime.now().isoformat(),
            'basic': {},
            'committees': {},
            'ministries': {},
            'special_tribunals': {}
        }

        tasks = []

        # 기본 법률 데이터 검색 (키워드 기반 확장 검색)
        if search_options.get('basic', True):
            tasks.append(('basic', self.search_basic_legal_data(query, keywords)))

        # 위원회 결정문 검색
        committees = search_options.get('committees', [])
        if committees:
            tasks.append(('committees', self.search_committee_decisions(query, committees)))

        # 부처별 법령해석 검색
        ministries = search_options.get('ministries', [])
        if ministries:
            tasks.append(('ministries', self.search_ministry_interpretations(query, ministries)))

        # 특별행정심판례 검색
        if search_options.get('special_tribunals', False):
            tasks.append(('special_tribunals', self.search_special_tribunals(query)))

        # 병렬 실행
        for key, task in tasks:
            try:
                results[key] = await task
            except Exception as e:
                logger.error(f"검색 오류 ({key}): {e}")
                results[key] = {}

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
                        name = law.get('법령명한글', law.get('법령명', ''))
                        dept = law.get('소관부처명', '')
                        date = law.get('시행일자', law.get('공포일자', ''))
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
                    name = prec.get('사건명', '')
                    date = prec.get('선고일자', '')
                    court = prec.get('법원명', '')
                    case_no = prec.get('사건번호', '')
                    context_parts.append(f"{idx}. {name}")
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
                    name = case.get('사건명', '')
                    date = case.get('종국일자', case.get('선고일자', ''))
                    case_no = case.get('사건번호', '')
                    context_parts.append(f"{idx}. {name}")
                    if case_no:
                        context_parts.append(f"   - 사건번호: {case_no}")
                    if date:
                        context_parts.append(f"   - 종국일: {date}")

            # 법령해석례 (상위 25개 - 핵심 자료)
            if basic.get('expc'):
                expcs = basic['expc']
                context_parts.append(f"\n[법령해석례/유권해석] (총 {len(expcs)}건) ★ 핵심 자료")
                for idx, interp in enumerate(expcs[:25], 1):
                    name = interp.get('안건명', '')
                    no = interp.get('안건번호', '')
                    org = interp.get('회신기관명', '')
                    date = interp.get('회신일자', '')
                    context_parts.append(f"{idx}. {name}")
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
                    name = ruling.get('사건명', '')
                    date = ruling.get('의결일자', ruling.get('재결일자', ''))
                    case_no = ruling.get('사건번호', '')
                    result = ruling.get('재결결과', ruling.get('재결구분명', ''))
                    context_parts.append(f"{idx}. {name}")
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
                    name = rule.get('행정규칙명', '')
                    dept = rule.get('소관부처명', rule.get('소관부처', ''))
                    context_parts.append(f"{idx}. {name}")
                    if dept:
                        context_parts.append(f"   - 소관부처: {dept}")

            # 자치법규 (상위 10개)
            if basic.get('ordin'):
                ordins = basic['ordin']
                context_parts.append(f"\n[자치법규] (총 {len(ordins)}건)")
                for idx, ordin in enumerate(ordins[:10], 1):
                    name = ordin.get('자치법규명', '')
                    local = ordin.get('지자체기관명', ordin.get('자치단체명', ''))
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

    async def generate_legal_advice(self, query: str, legal_data: Dict,
                                   fact_sheet: Dict, service_type: ServiceType = None) -> str:
        """AI 법률 조언 생성 - 실제 검색 결과 기반"""
        openai_client = get_openai_client()

        if not openai_client:
            return self._generate_fallback_response(query, legal_data)

        context = self._build_context(legal_data)
        timeline = "\n".join([f"- {item['date']}: {item['event']}"
                             for item in fact_sheet.get('timeline', [])])

        # 검색 통계 요약
        stats_summary = self._get_search_stats_summary(legal_data)

        # 추출된 키워드
        keywords = legal_data.get('keywords', [])
        keywords_str = ', '.join(keywords) if keywords else '없음'

        prompt = f"""
{AI_LAWYER_SYSTEM_PROMPT}

[서비스 유형: 법률 연구 및 자료 검색]

## 의뢰인 상황/질문:
{query}

## 추출된 핵심 키워드:
{keywords_str}

## 사실관계 Timeline:
{timeline if timeline else "특별한 일자 정보 없음"}

## 검색 결과 통계:
{stats_summary}

## 검색된 법률 정보 (실제 법제처 데이터):
{context}

---
## 중요 지침:
1. 위에 제공된 "검색된 법률 정보"는 법제처 Open API에서 실제로 검색된 자료입니다.
2. 답변 시 반드시 위 검색 결과에서 구체적인 판례번호, 법령해석례 안건번호, 행정심판례 사건번호를 인용하세요.
3. 일반적인 법률 지식이 아닌, 검색된 실제 자료를 기반으로 답변하세요.
4. 검색 결과가 부족한 경우 그 사실을 명시하세요.

## 답변 구조:

### 1. 핵심 답변 (2-3문장 요약)
[의뢰인 상황에 대한 핵심 결론]

### 2. 관련 법령 분석
[검색된 법령 중 관련 법령을 구체적으로 인용]
- 법령명, 조항 번호 명시
- 해당 조항의 핵심 내용 설명

### 3. 관련 판례 분석 (★ 중요)
[검색된 판례를 사건번호와 함께 구체적으로 인용]
- 각 판례의 사건번호, 선고일자, 법원 명시
- 해당 판례의 판시사항/시사점 설명
- 의뢰인 상황과의 관련성 분석

### 4. 유권해석/행정심판례 분석 (★ 중요)
[검색된 법령해석례, 행정심판례를 안건번호와 함께 구체적으로 인용]
- 각 해석례/심판례의 안건번호, 회신기관/재결청 명시
- 해석/재결의 핵심 내용
- 의뢰인 상황에 대한 시사점

### 5. 종합 의견 및 실무적 조언
- 위 자료들을 종합한 법적 판단
- 구체적인 대응 방안 제시
- 주의사항

### 6. 추가 확인사항
- 더 정확한 조언을 위해 필요한 정보
- 추가 검색이 필요한 분야

---
⚖️ 본 내용은 AI가 작성한 참고자료이며, 법률자문이 아닙니다.
구체적인 사안에 대해서는 반드시 변호사 등 전문가의 검토가 필요합니다.
"""

        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": AI_LAWYER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4000  # 더 긴 응답 허용
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

    with st.spinner("🔍 법률 데이터 검색 중..."):
        progress = st.progress(0)

        # 1. 종합 검색
        progress.progress(30, "법제처 데이터베이스 검색 중...")
        legal_data = await engine.comprehensive_search(query, search_options)

        # 2. 사실관계 정리
        progress.progress(60, "검색 결과 분석 중...")
        fact_sheet = engine.create_fact_sheet(query, legal_data)

        # 3. AI 분석
        progress.progress(80, "AI 분석 중...")
        advice = await engine.generate_legal_advice(query, legal_data, fact_sheet)

        progress.progress(100, "완료!")
        time.sleep(0.3)
        progress.empty()

    return legal_data, fact_sheet, advice, engine

# ===== 메인 앱 =====
def main():
    # 헤더
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("⚖️ AI 법률 연구 도우미")
        st.markdown("판례, 유권해석, 법령 종합 검색 서비스")
    with col2:
        st.markdown("""
        <div style="text-align: right; padding: 1rem;">
            <small>v6.0 | 법제처 API 전체 연동</small>
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
            st.success("✅ OpenAI API 연결됨")
        else:
            st.warning("⚠️ OpenAI API 미설정")
            st.caption("AI 분석 없이 검색 결과만 표시됩니다.")

        st.divider()

        # 검색 옵션
        st.header("🔍 검색 옵션")

        # 엔진 초기화 (옵션 표시용)
        engine = LegalAIEngine()

        # 기본 데이터 검색
        search_basic = st.checkbox("기본 법률 데이터", value=True,
                                   help="법령, 판례, 행정규칙, 자치법규, 헌재결정례, 법령해석례, 행정심판례, 조약")

        # 위원회 결정문
        with st.expander("위원회 결정문"):
            col1, col2 = st.columns(2)
            committees_list = list(engine.committee_targets.items())
            half = len(committees_list) // 2

            with col1:
                for key, info in committees_list[:half]:
                    st.checkbox(info['name'], key=f"comm_{key}")
            with col2:
                for key, info in committees_list[half:]:
                    st.checkbox(info['name'], key=f"comm_{key}")

        # 부처별 법령해석 (주요)
        major_ministries = [
            ('moelCgmExpc', '고용노동부'),
            ('molitCgmExpc', '국토교통부'),
            ('moisCgmExpc', '행정안전부'),
            ('mohwCgmExpc', '보건복지부'),
            ('molegCgmExpc', '법제처'),
            ('mojCgmExpc', '법무부'),
        ]

        with st.expander("부처별 법령해석 (주요)"):
            for key, name in major_ministries:
                st.checkbox(name, key=f"min_{key}")

        # 부처별 법령해석 (기타)
        other_ministries = [(k, v['name']) for k, v in engine.ministry_targets.items()
                           if k not in [m[0] for m in major_ministries]]

        with st.expander("부처별 법령해석 (기타)"):
            col1, col2 = st.columns(2)
            for idx, (key, name) in enumerate(other_ministries):
                with col1 if idx % 2 == 0 else col2:
                    st.checkbox(name, key=f"min_{key}")

        # 특별행정심판례
        search_special_tribunals = st.checkbox(
            "특별행정심판례",
            value=False,
            help="조세심판원, 해양안전심판원, 국민권익위원회, 인사혁신처 소청심사위원회"
        )

        st.divider()

        # 새 대화 시작 버튼
        if st.button("🔄 새 검색 시작", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.search_results = None
            st.session_state.fact_sheet = {}
            st.rerun()

    # ===== 메인 컨텐츠 =====
    # 웰컴 메시지
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="chat-message assistant-message">
            <strong>⚖️ AI 법률 연구 도우미:</strong><br><br>

            안녕하세요! AI 법률 연구 도우미입니다.<br><br>

            <b>🔍 검색 가능한 법률 데이터:</b><br>
            • <b>기본:</b> 법령, 판례, 행정규칙, 자치법규, 헌재결정례, 법령해석례, 행정심판례, 조약<br>
            • <b>위원회 결정문:</b> 공정거래위원회, 노동위원회, 금융위원회 등 12개 위원회<br>
            • <b>부처별 법령해석:</b> 고용노동부, 국토교통부 등 30개 이상 부처<br>
            • <b>특별행정심판:</b> 조세심판원, 해양안전심판원 등<br><br>

            <b>💡 사용 방법:</b><br>
            1. 사이드바에서 API 키를 입력하세요<br>
            2. 검색할 데이터 소스를 선택하세요<br>
            3. 아래 입력창에 검색어를 입력하세요<br><br>

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
    col1, col2, col3 = st.columns(3)

    examples = {
        "부당해고 구제": "부당해고 구제 절차와 관련 판례",
        "임대차 보증금": "주택임대차보호법 보증금 반환",
        "개인정보 침해": "개인정보 침해 손해배상"
    }

    clicked_example = None
    for idx, (btn_text, query) in enumerate(examples.items()):
        with [col1, col2, col3][idx]:
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
            selected_committees = [
                key for key in engine_for_options.committee_targets.keys()
                if st.session_state.get(f"comm_{key}", False)
            ]

            # 세션 상태에서 선택된 부처 수집
            selected_ministries = [
                key for key in engine_for_options.ministry_targets.keys()
                if st.session_state.get(f"min_{key}", False)
            ]

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

# ===== 앱 실행 =====
if __name__ == "__main__":
    main()
