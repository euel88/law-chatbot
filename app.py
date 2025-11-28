"""
AI 변호사 챗봇 - 전문 법률 조언 서비스
법제처 API + ChatGPT를 활용한 변호사 사고 프로세스 구현

필요한 환경변수 (.env 파일):
- LAW_API_KEY: 법제처 Open API 키
- OPENAI_API_KEY: OpenAI API 키

실행 방법:
streamlit run app.py
"""

import streamlit as st
import requests
import json
import time
import hashlib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import aiohttp
import pandas as pd
from dotenv import load_dotenv
import openai
import logging
from enum import Enum
import re

# 환경변수 로드
load_dotenv()


# ===== API 키 로드 함수 =====
def get_api_key(key_name: str, default: str = '') -> str:
    """
    API 키를 여러 소스에서 로드 (우선순위: st.secrets > 환경변수)
    - Streamlit Cloud: st.secrets에서 로드
    - 로컬/GitHub Actions: 환경변수에서 로드
    """
    # 1. Streamlit secrets 확인 (Streamlit Cloud)
    try:
        if hasattr(st, 'secrets') and key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass

    # 2. 환경변수 확인 (로컬 .env 또는 GitHub Actions)
    return os.getenv(key_name, default)


# ===== 설정 =====
LAW_API_KEY = get_api_key('LAW_API_KEY')
OPENAI_API_KEY = get_api_key('OPENAI_API_KEY')

# OpenAI 설정
openai.api_key = OPENAI_API_KEY

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== 페이지 설정 =====
st.set_page_config(
    page_title="AI 변호사 - 법률 조언 서비스",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== 커스텀 CSS =====
st.markdown("""
<style>
    /* 채팅 인터페이스 스타일 */
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
    
    /* 법률 의견서 스타일 */
    .legal-opinion {
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* 리스크 표시 */
    .risk-high { color: #d32f2f; font-weight: bold; }
    .risk-medium { color: #f57c00; font-weight: bold; }
    .risk-low { color: #388e3c; font-weight: bold; }
    
    /* IRAC 구조 */
    .irac-section {
        background-color: #f5f5f5;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1976d2;
        border-radius: 5px;
    }
    
    /* 액션 플랜 */
    .action-plan {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ===== 서비스 유형 Enum =====
class ServiceType(Enum):
    INFO = "법률 정보 제공"
    CONTRACT = "계약서 검토"
    OPINION = "법률자문의견서"

# ===== 리스크 레벨 =====
class RiskLevel(Enum):
    HIGH = ("🔴 High", "즉시 중단/전면 재검토 필요")
    MEDIUM = ("🟠 Medium", "수정 협상 필수")
    LOW = ("🟡 Low", "문구 명확화 권장")

# ===== 세션 상태 초기화 =====
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_service' not in st.session_state:
    st.session_state.current_service = None
if 'fact_sheet' not in st.session_state:
    st.session_state.fact_sheet = {}
if 'case_documents' not in st.session_state:
    st.session_state.case_documents = []

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

변호사적 사고 프로세스:
1. 사실관계 파악·정리 → Fact Sheet 작성
2. 법규·판례 조사 → 근거 자료 리스트
3. 쟁점 도출·위험도 평가 → 쟁점표 + 위험도표
4. 전략·대안 설계 → Action Plan
5. 의견서 작성·검증 → 법률자문의견서

필수 고지: ⚖️ 본 내용은 AI가 작성한 참고자료이며, 법률자문이 아닙니다.
구체적인 사안에 대해서는 반드시 변호사 등 전문가의 검토가 필요합니다.
"""

# ===== 법률 AI 엔진 클래스 =====
class LegalAIEngine:
    """AI 변호사 사고 프로세스를 구현한 법률 AI 엔진"""
    
    def __init__(self):
        self.law_api_key = LAW_API_KEY
        self.api_endpoints = {
            'search': 'https://www.law.go.kr/DRF/lawSearch.do',
            'service': 'https://www.law.go.kr/DRF/lawService.do'
        }
        # API target 코드 정의
        self.api_targets = {
            'law': 'law',           # 현행법령
            'eflaw': 'eflaw',       # 현행법령(시행일 기준)
            'prec': 'prec',         # 판례
            'admrul': 'admrul',     # 행정규칙
            'ordin': 'ordin',       # 자치법규
            'detc': 'detc',         # 헌재결정례
            'expc': 'expc',         # 법령해석례
            'decc': 'decc',         # 행정심판례
            'trty': 'trty'          # 조약
        }
        
    async def analyze_query(self, user_query: str) -> ServiceType:
        """사용자 질의 분석 및 서비스 유형 판단"""
        query_lower = user_query.lower()
        
        # 계약서 검토 키워드
        contract_keywords = ['계약서', '검토', '독소조항', '불공정', '계약 검토']
        if any(keyword in query_lower for keyword in contract_keywords):
            return ServiceType.CONTRACT
            
        # 법률자문의견서 키워드
        opinion_keywords = ['사안 검토', '법적 의견', '대응 방안', '자문', '법률자문']
        if any(keyword in query_lower for keyword in opinion_keywords):
            return ServiceType.OPINION
            
        # 기본: 법률 정보 제공
        return ServiceType.INFO
    
    async def search_legal_data(self, query: str) -> Dict:
        """법제처 API를 통한 종합 법률 데이터 검색"""
        async with aiohttp.ClientSession() as session:
            # 병렬로 모든 법률 데이터 소스 검색
            tasks = [
                self._search_laws(session, query),
                self._search_precedents(session, query),
                self._search_admin_rules(session, query),
                self._search_ordinances(session, query),
                self._search_constitutional_cases(session, query),
                self._search_legal_interpretations(session, query),
                self._search_admin_rulings(session, query),
                self._search_treaties(session, query)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            return {
                'query': query,
                'laws': results[0] if not isinstance(results[0], Exception) else [],
                'precedents': results[1] if not isinstance(results[1], Exception) else [],
                'admin_rules': results[2] if not isinstance(results[2], Exception) else [],
                'ordinances': results[3] if not isinstance(results[3], Exception) else [],
                'constitutional_cases': results[4] if not isinstance(results[4], Exception) else [],
                'legal_interpretations': results[5] if not isinstance(results[5], Exception) else [],
                'admin_rulings': results[6] if not isinstance(results[6], Exception) else [],
                'treaties': results[7] if not isinstance(results[7], Exception) else [],
                'search_time': datetime.now().isoformat()
            }
    
    async def _search_laws(self, session, query: str) -> List[Dict]:
        """법령 검색"""
        params = {
            'OC': self.law_api_key,
            'target': 'law',
            'query': query,
            'type': 'json',
            'display': 20  # 더 많은 결과 가져오기
        }
        
        try:
            async with session.get(
                self.api_endpoints['search'],
                params=params,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('law', [])
        except Exception as e:
            logger.error(f"법령 검색 오류: {e}")
        return []
    
    async def _search_precedents(self, session, query: str) -> List[Dict]:
        """판례 검색"""
        params = {
            'OC': self.law_api_key,
            'target': 'prec',
            'query': query,
            'type': 'json',
            'display': 15  # 더 많은 판례
        }
        
        try:
            async with session.get(
                self.api_endpoints['search'],
                params=params,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('prec', [])
        except Exception as e:
            logger.error(f"판례 검색 오류: {e}")
        return []
    
    async def _search_admin_rules(self, session, query: str) -> List[Dict]:
        """행정규칙 검색"""
        params = {
            'OC': self.law_api_key,
            'target': 'admrul',
            'query': query,
            'type': 'json',
            'display': 10
        }
        
        try:
            async with session.get(
                self.api_endpoints['search'],
                params=params,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('admrul', [])
        except Exception as e:
            logger.error(f"행정규칙 검색 오류: {e}")
        return []

    async def _search_ordinances(self, session, query: str) -> List[Dict]:
        """자치법규 검색"""
        params = {
            'OC': self.law_api_key,
            'target': 'ordin',
            'query': query,
            'type': 'json',
            'display': 10
        }

        try:
            async with session.get(
                self.api_endpoints['search'],
                params=params,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('ordin', [])
        except Exception as e:
            logger.error(f"자치법규 검색 오류: {e}")
        return []

    async def _search_constitutional_cases(self, session, query: str) -> List[Dict]:
        """헌재결정례 검색"""
        params = {
            'OC': self.law_api_key,
            'target': 'detc',
            'query': query,
            'type': 'json',
            'display': 10
        }

        try:
            async with session.get(
                self.api_endpoints['search'],
                params=params,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('detc', [])
        except Exception as e:
            logger.error(f"헌재결정례 검색 오류: {e}")
        return []

    async def _search_legal_interpretations(self, session, query: str) -> List[Dict]:
        """법령해석례 검색"""
        params = {
            'OC': self.law_api_key,
            'target': 'expc',
            'query': query,
            'type': 'json',
            'display': 10
        }

        try:
            async with session.get(
                self.api_endpoints['search'],
                params=params,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('expc', [])
        except Exception as e:
            logger.error(f"법령해석례 검색 오류: {e}")
        return []

    async def _search_admin_rulings(self, session, query: str) -> List[Dict]:
        """행정심판례 검색"""
        params = {
            'OC': self.law_api_key,
            'target': 'decc',
            'query': query,
            'type': 'json',
            'display': 10
        }

        try:
            async with session.get(
                self.api_endpoints['search'],
                params=params,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('decc', [])
        except Exception as e:
            logger.error(f"행정심판례 검색 오류: {e}")
        return []

    async def _search_treaties(self, session, query: str) -> List[Dict]:
        """조약 검색"""
        params = {
            'OC': self.law_api_key,
            'target': 'trty',
            'query': query,
            'type': 'json',
            'display': 10
        }

        try:
            async with session.get(
                self.api_endpoints['search'],
                params=params,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('trty', [])
        except Exception as e:
            logger.error(f"조약 검색 오류: {e}")
        return []

    async def get_law_detail(self, law_id: str) -> Dict:
        """법령 본문 상세 조회"""
        params = {
            'OC': self.law_api_key,
            'target': 'law',
            'ID': law_id,
            'type': 'json'
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
            logger.error(f"법령 상세 조회 오류: {e}")
        return {}

    async def get_precedent_detail(self, prec_id: str) -> Dict:
        """판례 본문 상세 조회"""
        params = {
            'OC': self.law_api_key,
            'target': 'prec',
            'ID': prec_id,
            'type': 'json'
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
            logger.error(f"판례 상세 조회 오류: {e}")
        return {}

    async def get_constitutional_case_detail(self, case_id: str) -> Dict:
        """헌재결정례 본문 상세 조회"""
        params = {
            'OC': self.law_api_key,
            'target': 'detc',
            'ID': case_id,
            'type': 'json'
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
            logger.error(f"헌재결정례 상세 조회 오류: {e}")
        return {}

    async def get_legal_interpretation_detail(self, interp_id: str) -> Dict:
        """법령해석례 본문 상세 조회"""
        params = {
            'OC': self.law_api_key,
            'target': 'expc',
            'ID': interp_id,
            'type': 'json'
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
            logger.error(f"법령해석례 상세 조회 오류: {e}")
        return {}

    async def get_admin_ruling_detail(self, ruling_id: str) -> Dict:
        """행정심판례 본문 상세 조회"""
        params = {
            'OC': self.law_api_key,
            'target': 'decc',
            'ID': ruling_id,
            'type': 'json'
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
            logger.error(f"행정심판례 상세 조회 오류: {e}")
        return {}

    def create_fact_sheet(self, user_input: str, legal_data: Dict) -> Dict:
        """사실관계 정리 (Fact Sheet 작성)"""
        fact_sheet = {
            'query': user_input,
            'timestamp': datetime.now(),
            'related_laws_count': len(legal_data.get('laws', [])),
            'related_precedents_count': len(legal_data.get('precedents', [])),
            'related_admin_rules_count': len(legal_data.get('admin_rules', [])),
            'related_ordinances_count': len(legal_data.get('ordinances', [])),
            'related_constitutional_cases_count': len(legal_data.get('constitutional_cases', [])),
            'related_legal_interpretations_count': len(legal_data.get('legal_interpretations', [])),
            'related_admin_rulings_count': len(legal_data.get('admin_rulings', [])),
            'related_treaties_count': len(legal_data.get('treaties', [])),
            'key_facts': self._extract_key_facts(user_input),
            'timeline': self._extract_timeline(user_input)
        }
        return fact_sheet
    
    def _extract_key_facts(self, text: str) -> List[str]:
        """핵심 사실 추출"""
        # 간단한 패턴 매칭으로 핵심 사실 추출
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
        
        # 날짜와 관련 내용 추출
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
    
    async def generate_legal_advice(self, service_type: ServiceType, 
                                  user_query: str, legal_data: Dict, 
                                  fact_sheet: Dict) -> str:
        """AI 변호사 프로세스를 통한 법률 조언 생성"""
        
        # 서비스 유형별 프롬프트 구성
        if service_type == ServiceType.INFO:
            return await self._generate_info_response(user_query, legal_data)
        elif service_type == ServiceType.CONTRACT:
            return await self._generate_contract_review(user_query, legal_data)
        elif service_type == ServiceType.OPINION:
            return await self._generate_legal_opinion(user_query, legal_data, fact_sheet)
    
    async def _generate_info_response(self, query: str, legal_data: Dict) -> str:
        """법률 정보 제공 응답 생성"""
        # API 키 확인
        if not OPENAI_API_KEY:
            return self._generate_fallback_response(query, legal_data)

        context = self._build_context(legal_data)

        prompt = f"""
{AI_LAWYER_SYSTEM_PROMPT}

[서비스 유형: 법률 정보 제공]

의뢰인 질문: {query}

검색된 법률 정보:
{context}

위 정보를 바탕으로 다음 구조로 답변하세요:

1. **핵심 답변** (2-3문장 요약)

2. **관련 법령 설명**
   - 주요 법령과 조항
   - 핵심 내용 설명

3. **관련 판례**
   - 유사 사례 소개
   - 판결의 시사점

4. **실무적 조언**
   - 주의사항
   - 권장 행동

5. **추가 확인사항**
   - 더 정확한 조언을 위해 필요한 정보

⚖️ 필수 고지사항을 반드시 포함하세요.
"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": AI_LAWYER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"AI 응답 생성 오류: {e}")
            return "AI 응답을 생성할 수 없습니다. API 키를 확인해주세요."
    
    async def _generate_contract_review(self, query: str, legal_data: Dict) -> str:
        """계약서 검토 응답 생성"""
        # API 키 확인
        if not OPENAI_API_KEY:
            return self._generate_fallback_response(query, legal_data)

        context = self._build_context(legal_data)

        prompt = f"""
{AI_LAWYER_SYSTEM_PROMPT}

[서비스 유형: 계약서 검토]

의뢰인 요청: {query}

관련 법률 정보:
{context}

다음 체크리스트에 따라 계약서를 검토하세요:

## 계약서 검토 보고서

### 1. 계약 기본사항 점검
- [ ] 계약 당사자 확인
- [ ] 계약 목적 명확성
- [ ] 계약 기간 및 갱신
- [ ] 대가 및 지급조건

### 2. Red Flag 분석 (독소조항)
[발견된 문제점을 리스크 등급과 함께 제시]

### 3. 조항별 상세 분석
| 조항 | 내용 | 리스크 | 수정 제안 |
|------|------|--------|-----------|
| | | 🔴/🟠/🟡 | |

### 4. 협상 전략
- 우선순위 1: 
- 우선순위 2:
- 우선순위 3:

### 5. 개선안
[구체적인 수정 문구 제시]

⚖️ 필수 고지사항을 포함하세요.
"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": AI_LAWYER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"계약서 검토 오류: {e}")
            return "계약서 검토를 생성할 수 없습니다."
    
    async def _generate_legal_opinion(self, query: str, legal_data: Dict,
                                    fact_sheet: Dict) -> str:
        """법률자문의견서 생성"""
        # API 키 확인
        if not OPENAI_API_KEY:
            return self._generate_fallback_response(query, legal_data)

        context = self._build_context(legal_data)
        timeline = "\n".join([f"- {item['date']}: {item['event']}" 
                             for item in fact_sheet['timeline']])
        
        prompt = f"""
{AI_LAWYER_SYSTEM_PROMPT}

[서비스 유형: 법률자문의견서 작성]

의뢰인: [개인/법인]
질의사항: {query}

사실관계 Timeline:
{timeline}

검색된 법률 정보:
{context}

다음 구조로 전문 법률자문의견서를 작성하세요:

# 법률자문의견서

## 1. 의뢰인 정보
- 성명/상호: 
- 질의일자: {datetime.now().strftime('%Y년 %m월 %d일')}

## 2. 질의사항
{query}

## 3. 관련 법령 및 판례
| 구분 | 조항/판례번호 | 주요 내용 | 비고 |
|------|--------------|-----------|------|
| 법령 | | | |
| 판례 | | | |

## 4. 사실관계 정리
{timeline}

## 5. 쟁점 및 법리 검토 (IRAC)

### 쟁점 1: [쟁점명]
- **Issue**: 
- **Rule**: 
- **Application**: 
- **Conclusion**: 
- **리스크 등급**: 🔴 High / 🟠 Medium / 🟡 Low

### 쟁점 2: [쟁점명]
[동일 구조 반복]

## 6. 리스크 평가
| 쟁점 | 발생가능성 | 예상 손실 | 등급 | 대응 우선순위 |
|------|-----------|----------|------|--------------|
| | | | 🔴/🟠/🟡 | |

## 7. 대응 방안 (Action Plan)

### 전략 1 (권장안)
- **개요**: 
- **절차**: ① → ② → ③
- **예상 기간**: 
- **예상 비용**: 
- **성공 가능성**: %

### 전략 2 (대안)
[동일 구조]

## 8. 결론
[3줄 요약]
1. 
2. 
3. 

## 9. 필수 고지사항
⚖️ 본 의견서는 AI가 작성한 참고자료이며, 법률자문이 아닙니다.
구체적인 사안에 대해서는 반드시 변호사 등 전문가의 검토가 필요합니다.

작성일: {datetime.now().strftime('%Y년 %m월 %d일')}
AI 변호사 GPT (전자서명)
"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": AI_LAWYER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # 더 정확한 응답을 위해 낮은 temperature
                max_tokens=3500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"법률자문의견서 생성 오류: {e}")
            return "법률자문의견서를 생성할 수 없습니다."
    
    def _build_context(self, legal_data: Dict) -> str:
        """검색 결과를 컨텍스트로 구성"""
        context_parts = []

        # 법령 정보
        if legal_data.get('laws'):
            laws_text = "\n[관련 법령]\n"
            for idx, law in enumerate(legal_data['laws'][:10], 1):
                laws_text += f"{idx}. {law.get('법령명한글', law.get('법령명', ''))}\n"
                laws_text += f"   - 소관부처: {law.get('소관부처명', '')}\n"
                laws_text += f"   - 시행일자: {law.get('시행일자', '')}\n\n"
            context_parts.append(laws_text)

        # 판례 정보
        if legal_data.get('precedents'):
            prec_text = "\n[관련 판례]\n"
            for idx, prec in enumerate(legal_data['precedents'][:7], 1):
                prec_text += f"{idx}. {prec.get('사건명', '')} ({prec.get('선고일자', '')})\n"
                prec_text += f"   - 법원: {prec.get('법원명', '')}\n"
                prec_text += f"   - 사건번호: {prec.get('사건번호', '')}\n\n"
            context_parts.append(prec_text)

        # 행정규칙 정보
        if legal_data.get('admin_rules'):
            admin_text = "\n[관련 행정규칙]\n"
            for idx, rule in enumerate(legal_data['admin_rules'][:5], 1):
                admin_text += f"{idx}. {rule.get('행정규칙명', '')}\n"
                admin_text += f"   - 소관부처: {rule.get('소관부처명', rule.get('소관부처', ''))}\n\n"
            context_parts.append(admin_text)

        # 자치법규 정보
        if legal_data.get('ordinances'):
            ordin_text = "\n[관련 자치법규]\n"
            for idx, ordin in enumerate(legal_data['ordinances'][:5], 1):
                ordin_text += f"{idx}. {ordin.get('자치법규명', '')}\n"
                ordin_text += f"   - 지자체: {ordin.get('지자체기관명', ordin.get('자치단체명', ''))}\n"
                ordin_text += f"   - 종류: {ordin.get('자치법규종류', '')}\n\n"
            context_parts.append(ordin_text)

        # 헌재결정례 정보
        if legal_data.get('constitutional_cases'):
            const_text = "\n[관련 헌재결정례]\n"
            for idx, case in enumerate(legal_data['constitutional_cases'][:5], 1):
                const_text += f"{idx}. {case.get('사건명', '')} ({case.get('종국일자', '')})\n"
                const_text += f"   - 사건번호: {case.get('사건번호', '')}\n\n"
            context_parts.append(const_text)

        # 법령해석례 정보
        if legal_data.get('legal_interpretations'):
            interp_text = "\n[관련 법령해석례]\n"
            for idx, interp in enumerate(legal_data['legal_interpretations'][:5], 1):
                interp_text += f"{idx}. {interp.get('안건명', '')}\n"
                interp_text += f"   - 안건번호: {interp.get('안건번호', '')}\n"
                interp_text += f"   - 회신기관: {interp.get('회신기관명', '')}\n"
                interp_text += f"   - 회신일자: {interp.get('회신일자', '')}\n\n"
            context_parts.append(interp_text)

        # 행정심판례 정보
        if legal_data.get('admin_rulings'):
            ruling_text = "\n[관련 행정심판례]\n"
            for idx, ruling in enumerate(legal_data['admin_rulings'][:5], 1):
                ruling_text += f"{idx}. {ruling.get('사건명', '')} ({ruling.get('의결일자', '')})\n"
                ruling_text += f"   - 사건번호: {ruling.get('사건번호', '')}\n"
                ruling_text += f"   - 재결청: {ruling.get('재결청', '')}\n"
                ruling_text += f"   - 재결구분: {ruling.get('재결구분명', '')}\n\n"
            context_parts.append(ruling_text)

        # 조약 정보
        if legal_data.get('treaties'):
            treaty_text = "\n[관련 조약]\n"
            for idx, treaty in enumerate(legal_data['treaties'][:5], 1):
                treaty_text += f"{idx}. {treaty.get('조약명', treaty.get('조약명한글', ''))}\n"
                treaty_text += f"   - 체결일자: {treaty.get('체결일자', '')}\n"
                treaty_text += f"   - 발효일자: {treaty.get('발효일자', '')}\n\n"
            context_parts.append(treaty_text)

        return "\n".join(context_parts)

    def _generate_fallback_response(self, query: str, legal_data: Dict) -> str:
        """API 키 없을 때 검색 결과 기반 기본 응답 생성"""
        context = self._build_context(legal_data)

        # 검색 결과 통계
        stats = []
        if legal_data.get('laws'):
            stats.append(f"법령 {len(legal_data['laws'])}건")
        if legal_data.get('precedents'):
            stats.append(f"판례 {len(legal_data['precedents'])}건")
        if legal_data.get('admin_rules'):
            stats.append(f"행정규칙 {len(legal_data['admin_rules'])}건")
        if legal_data.get('ordinances'):
            stats.append(f"자치법규 {len(legal_data['ordinances'])}건")
        if legal_data.get('constitutional_cases'):
            stats.append(f"헌재결정례 {len(legal_data['constitutional_cases'])}건")
        if legal_data.get('legal_interpretations'):
            stats.append(f"법령해석례 {len(legal_data['legal_interpretations'])}건")
        if legal_data.get('admin_rulings'):
            stats.append(f"행정심판례 {len(legal_data['admin_rulings'])}건")
        if legal_data.get('treaties'):
            stats.append(f"조약 {len(legal_data['treaties'])}건")

        stats_text = ", ".join(stats) if stats else "검색 결과 없음"

        response = f"""## 법률 데이터 검색 결과

**질의:** {query}

**검색 통계:** {stats_text}

{context if context else "관련 법률 데이터를 찾지 못했습니다."}

---
⚠️ **안내:** OpenAI API 키가 설정되지 않아 AI 분석 기능을 사용할 수 없습니다.
위 검색 결과는 법제처 Open API에서 가져온 원본 데이터입니다.

AI 분석을 이용하시려면 `.env` 파일에 `OPENAI_API_KEY`를 설정해주세요.

⚖️ 본 내용은 참고자료이며, 구체적인 사안에 대해서는 반드시 변호사 등 전문가의 검토가 필요합니다.
"""
        return response

# ===== Streamlit UI 함수들 =====
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
        st.markdown(f'''
        <div class="chat-message assistant-message">
            <strong>⚖️ AI 변호사:</strong><br>
            {content}
        </div>
        ''', unsafe_allow_html=True)

def display_legal_opinion(content: str):
    """법률자문의견서 형식으로 표시"""
    formatted_content = content.replace("\n", "<br>")
    st.markdown(f'''
    <div class="legal-opinion">
        {formatted_content}
    </div>
    ''', unsafe_allow_html=True)

async def process_user_query(query: str):
    """사용자 질의 처리 메인 함수"""
    # AI 엔진 초기화
    engine = LegalAIEngine()
    
    # 1. 서비스 유형 판단
    service_type = await engine.analyze_query(query)
    st.session_state.current_service = service_type
    
    # 진행 상황 표시
    with st.spinner(f"🔍 {service_type.value} 서비스로 처리 중..."):
        progress = st.progress(0)
        
        # 2. 법률 데이터 검색
        progress.progress(25, "법제처 데이터베이스 검색 중...")
        legal_data = await engine.search_legal_data(query)
        
        # 3. 사실관계 정리
        progress.progress(50, "사실관계 분석 중...")
        fact_sheet = engine.create_fact_sheet(query, legal_data)
        st.session_state.fact_sheet = fact_sheet
        
        # 4. AI 변호사 분석
        progress.progress(75, "AI 변호사가 법리 검토 중...")
        legal_advice = await engine.generate_legal_advice(
            service_type, query, legal_data, fact_sheet
        )
        
        progress.progress(100, "완료!")
        time.sleep(0.5)
        progress.empty()
    
    # 채팅 히스토리에 추가
    st.session_state.chat_history.append({
        "role": "user",
        "content": query,
        "timestamp": datetime.now()
    })
    
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": legal_advice,
        "service_type": service_type,
        "legal_data": legal_data,
        "fact_sheet": fact_sheet,
        "timestamp": datetime.now()
    })

# ===== 메인 앱 함수 =====
async def main():
    # 헤더
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("⚖️ AI 변호사 - 전문 법률 조언 서비스")
        st.markdown("법제처 데이터베이스와 AI를 활용한 맞춤형 법률 서비스")
    with col2:
        st.markdown("""
        <div style="text-align: right; padding: 1rem;">
            <small>v5.0 | GPT-5 + 법제처 API 전체 연동</small>
        </div>
        """, unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.header("🎯 서비스 안내")

        st.markdown("""
        ### 제공 서비스
        1. **법률 정보 제공**
           - 일반적인 법률 지식
           - 절차 및 요건 설명

        2. **계약서 검토**
           - 독소조항 분석
           - 리스크 평가
           - 수정안 제시

        3. **법률자문의견서**
           - IRAC 분석
           - 리스크 매트릭스
           - Action Plan 제공
        """)

        st.divider()

        # 검색 데이터 소스 안내
        st.header("📚 검색 데이터 소스")
        with st.expander("법제처 API 연동 목록", expanded=False):
            st.markdown("""
            - 📜 **법령** (현행법령, 시행일 기준)
            - ⚖️ **판례** (대법원, 하급심)
            - 📋 **행정규칙**
            - 🏛️ **자치법규** (조례, 규칙)
            - 🏛️ **헌재결정례**
            - 📖 **법령해석례**
            - 📑 **행정심판례**
            - 🌐 **조약**
            """)

        st.divider()

        # 현재 서비스 타입 표시
        if st.session_state.current_service:
            st.info(f"현재 모드: {st.session_state.current_service.value}")

        # 검색 통계 표시
        if st.session_state.fact_sheet:
            st.header("📊 검색 결과 통계")
            fact = st.session_state.fact_sheet
            cols = st.columns(2)
            with cols[0]:
                st.metric("법령", fact.get('related_laws_count', 0))
                st.metric("판례", fact.get('related_precedents_count', 0))
                st.metric("행정규칙", fact.get('related_admin_rules_count', 0))
                st.metric("자치법규", fact.get('related_ordinances_count', 0))
            with cols[1]:
                st.metric("헌재결정례", fact.get('related_constitutional_cases_count', 0))
                st.metric("법령해석례", fact.get('related_legal_interpretations_count', 0))
                st.metric("행정심판례", fact.get('related_admin_rulings_count', 0))
                st.metric("조약", fact.get('related_treaties_count', 0))

        st.divider()

        # API 상태
        st.header("🔌 시스템 상태")
        if LAW_API_KEY:
            st.success("✅ 법제처 API 연결")
        else:
            st.error("❌ 법제처 API 키 필요")

        if OPENAI_API_KEY:
            st.success("✅ GPT-5 AI 엔진 활성화")
        else:
            st.error("❌ OpenAI API 키 필요")

        # 새 대화 시작 버튼
        if st.button("🔄 새 상담 시작", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.current_service = None
            st.session_state.fact_sheet = {}
            st.rerun()
    
    # 메인 채팅 인터페이스
    chat_container = st.container()
    
    # 기존 대화 내역 표시
    with chat_container:
        if not st.session_state.chat_history:
            # 웰컴 메시지
            st.markdown("""
            <div class="chat-message assistant-message">
                <strong>⚖️ AI 변호사 (GPT-5):</strong><br>
                안녕하세요, AI 변호사입니다.<br><br>

                <b>🔍 검색 가능한 법률 데이터:</b><br>
                • 법령 (현행법령, 시행일 기준)<br>
                • 판례 (대법원, 하급심)<br>
                • 행정규칙, 자치법규 (조례/규칙)<br>
                • 헌재결정례, 법령해석례<br>
                • 행정심판례, 조약<br><br>

                <b>📋 제공 서비스:</b><br>
                • 법률 정보 제공 - "~은 무엇인가요?"<br>
                • 계약서 검토 - "계약서 검토해주세요"<br>
                • 법률자문의견서 - "~사안에 대한 법적 검토"<br><br>

                어떤 법률 문제를 도와드릴까요?
            </div>
            """, unsafe_allow_html=True)
        else:
            # 대화 히스토리 표시
            for msg in st.session_state.chat_history:
                display_chat_message(msg["role"], msg["content"])
    
    # 입력 영역
    st.divider()
    
    # 예시 질문 버튼들
    st.markdown("### 💡 자주 묻는 질문")
    col1, col2, col3 = st.columns(3)
    
    example_queries = {
        "임대차 계약 시 주의사항": "임대차 계약을 체결할 때 주의해야 할 사항은 무엇인가요?",
        "부당해고 구제 방법": "회사에서 부당해고를 당했습니다. 어떻게 대응해야 하나요?",
        "계약서 검토 요청": "프리랜서 용역계약서를 검토해주세요. 특히 손해배상 조항이 걱정됩니다."
    }
    
    for idx, (btn_text, query) in enumerate(example_queries.items()):
        with [col1, col2, col3][idx]:
            if st.button(btn_text, use_container_width=True):
                asyncio.run(process_user_query(query))
                st.rerun()
    
    # 사용자 입력
    user_input = st.text_area(
        "법률 질문을 입력하세요",
        placeholder="예: 전세 계약 만료가 다가오는데 보증금을 돌려받지 못할까 걱정됩니다. 어떻게 대비해야 하나요?",
        height=100
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.button("🚀 법률 상담 시작", type="primary", use_container_width=True):
            if user_input:
                await process_user_query(user_input)
                st.rerun()
            else:
                st.warning("질문을 입력해주세요.")
    
    with col2:
        if st.button("📄 의견서 다운로드"):
            if st.session_state.chat_history:
                # 마지막 응답을 텍스트 파일로 다운로드
                last_response = st.session_state.chat_history[-1]
                if last_response["role"] == "assistant":
                    st.download_button(
                        label="💾 다운로드",
                        data=last_response["content"],
                        file_name=f"법률의견서_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
    
    with col3:
        if st.button("📊 상세 분석"):
            if st.session_state.fact_sheet:
                with st.expander("사실관계 분석 (Fact Sheet)"):
                    st.json(st.session_state.fact_sheet)

# ===== 앱 실행 =====
if __name__ == "__main__":
    # API 키 상태를 세션에 저장
    if 'api_warning_shown' not in st.session_state:
        st.session_state.api_warning_shown = False

    # API 키 경고 표시 (앱은 계속 실행)
    missing_keys = []
    if not LAW_API_KEY:
        missing_keys.append("법제처 API")
    if not OPENAI_API_KEY:
        missing_keys.append("OpenAI API")

    if missing_keys:
        st.warning(f"⚠️ {', '.join(missing_keys)} 키가 설정되지 않았습니다.")

        with st.expander("🔧 API 키 설정 방법", expanded=True):
            tab1, tab2, tab3 = st.tabs(["💻 로컬 환경", "☁️ Streamlit Cloud", "🔐 GitHub Actions"])

            with tab1:
                st.markdown("#### Step 1: API 키 발급")
                col1, col2 = st.columns(2)
                with col1:
                    st.info("**법제처 Open API**")
                    st.markdown("""
                    1. [open.law.go.kr](https://open.law.go.kr) 접속
                    2. 회원가입 후 로그인
                    3. API 키 발급 신청
                    """)
                with col2:
                    st.info("**OpenAI API**")
                    st.markdown("""
                    1. [platform.openai.com](https://platform.openai.com) 접속
                    2. 계정 생성 후 로그인
                    3. API Keys 메뉴에서 키 생성
                    """)

                st.markdown("#### Step 2: `.env` 파일 생성")
                st.code("""LAW_API_KEY=여기에_법제처_API_키_입력
OPENAI_API_KEY=여기에_OpenAI_API_키_입력""", language="bash")
                st.caption("프로젝트 루트 폴더에 `.env` 파일을 생성하세요.")

            with tab2:
                st.markdown("#### Streamlit Cloud Secrets 설정")
                st.markdown("""
                1. Streamlit Cloud 앱 대시보드 접속
                2. **Settings** → **Secrets** 메뉴 클릭
                3. 아래 내용 입력 후 저장:
                """)
                st.code("""LAW_API_KEY = "여기에_법제처_API_키_입력"
OPENAI_API_KEY = "여기에_OpenAI_API_키_입력" """, language="toml")

            with tab3:
                st.markdown("#### GitHub Actions Secrets 설정")
                st.markdown("""
                1. GitHub 저장소 페이지 접속
                2. **Settings** → **Secrets and variables** → **Actions** 클릭
                3. **New repository secret** 버튼 클릭
                4. 아래 두 개의 Secret 추가:
                """)

                col1, col2 = st.columns(2)
                with col1:
                    st.code("Name: LAW_API_KEY\nSecret: 법제처_API_키", language="yaml")
                with col2:
                    st.code("Name: OPENAI_API_KEY\nSecret: OpenAI_API_키", language="yaml")

                st.markdown("#### Workflow 파일에서 사용")
                st.code("""# .github/workflows/deploy.yml
env:
  LAW_API_KEY: ${{ secrets.LAW_API_KEY }}
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}""", language="yaml")
                st.caption("GitHub Actions workflow에서 secrets를 환경변수로 전달합니다.")

        st.divider()

    # 비동기 실행 (API 키 유무와 관계없이 앱 실행)
    asyncio.run(main())
