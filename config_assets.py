# config_asset.py

STOCKS = {
    'Technology': [
        'AAPL',  # Apple - 소비자 중심의 기술 플랫폼 구축 (iOS 생태계)
        'MSFT',  # Microsoft - 클라우드 및 생산성 플랫폼 (Azure, Microsoft 365)
        'NVDA',  # Nvidia - AI와 그래픽 처리 유닛(GPU) 선도 기업
        'GOOG',  # Alphabet - 검색 및 광고 플랫폼, AI 연구 (DeepMind)
        'AMZN',  # Amazon - 전자상거래 및 클라우드 플랫폼 (AWS)
        'META',  # Meta - 소셜 미디어와 메타버스 플랫폼 (Facebook, Instagram)
        'CRM',   # Salesforce - 고객 관리(CRM) 소프트웨어의 리더
        'ADBE',  # Adobe - 디지털 미디어 및 마케팅 솔루션 플랫폼
        'AMD',   # AMD - CPU 및 GPU 혁신, Nvidia와의 경쟁으로 AI 관련 주목
        'AMAT',  # Applied Materials - 반도체 및 디스플레이 제조 장비 선도 기업
        'SNOW',  # Snowflake - 데이터 분석 및 AI 클라우드 플랫폼
        'PLTR',  # Palantir - 빅데이터와 AI 분석 솔루션 제공
        'PANW',  # Palo Alto Networks - 사이버 보안, AI로 위협 감지 및 대응
        'ASML',  # ASML - 첨단 반도체 제조장비, 반도체 공정의 필수 공급자
        'TSLA',  # Tesla - 자율주행 AI 기술의 선두주자
        'AVGO',  # Broadcom - AI 및 네트워크 솔루션 제공
        'IONQ',  # IONQ - 양자 컴퓨팅 기술 제공
        'WWD'    # 월마 디벨로프먼트
    ],
    'Financials': [
        'V',     # Visa - 글로벌 결제 네트워크 플랫폼
        'MA',    # Mastercard - 디지털 결제와 신용카드 네트워크 선도
        'SQ',    # Block (Square) - 소상공인 및 비즈니스 결제 플랫폼
        'PYPL',  # PayPal - 글로벌 디지털 결제 플랫폼
        'COIN',  # Coinbase - 암호화폐 거래소 플랫폼
        'BLK',   # BlackRock - 글로벌 자산 관리와 AI 기반 투자
        'MSTR',  # MicroStrategy Incorporated
        'GS',    # Goldman Sachs - 투자은행 및 자산 관리에서 AI 적용
        'HOOD'   # Robinhood Markets - 개인 투자자 중심의 수수료 없는 거래 플랫폼        
    ],
    'Consumer Cyclical': [
        'TSLA',  # Tesla - 자율주행 및 전기차 AI 혁신
        'UBER',  # Uber - AI 기반 라이드셰어링 및 물류 플랫폼
        'BKNG',  # Booking Holdings - 여행 및 숙박 예약 플랫폼
        'RIVN',  # Rivian - 전기차 시장 진입, Tesla와의 경쟁
        'NKE',   # Nike - 브랜드와 디지털 스포츠 웨어 플랫폼
        'SBUX',  # Starbucks - 소비자 경험과 리워드 플랫폼
        'RKLB'   # Rocket Lab - 우주 통신 플랫폼
    ],
    'Healthcare': [
        'LLY',   # Eli Lilly - 제약 혁신과 AI 기반 신약 개발
        'UNH',   # UnitedHealth - 건강 관리 서비스 플랫폼
        'ISRG',  # Intuitive Surgical - AI 기반 로봇 수술 플랫폼
        'TDOC',  # Teladoc - 원격 의료 플랫폼
        'JNJ',   # Johnson & Johnson - 헬스케어와 소비자 제품 플랫폼
        'MRK',   # Merck - AI와 데이터 기반 신약 연구 강화
        'TEM',   # Tempus AI - 데이터 기반 의료 플랫폼
        'NTRA',   # Natera - 유전체 분석 및 유전체 데이터 플랫폼
        'TEVA',   # Teva Pharmaceutical Industries - 제약 혁신과 유전체 분석소수몽키 AI 항공 보안, 에너지 관련주식
        'RXRX',   # 리커전, 신약발견
        'SDGR'    #슈뢰딩거, 신약설계
    ],
    'Communication Services': [
        'META',  # Meta - 소셜 미디어 플랫폼과 메타버스 혁신
        'SPOT',  # Spotify - AI 기반 음악 추천 플랫폼
        'NFLX',  # Netflix - 스트리밍 콘텐츠 플랫폼
        'PM'     # Philip Morris - 담배 및 화장품 플랫폼
    ],
    'Industrials': [
        'GE',    # General Electric - 산업 및 에너지 플랫폼
        'UPS',   # UPS - 물류와 공급망 최적화 플랫폼
        'BA',    # Boeing - 항공 및 방산, 항공기 및 우주항공 분야 선도 기업
        'CAT',   # Caterpillar - 건설 및 자원 관리 플랫폼
        'HON',   # Honeywell - AI 기반 산업 자동화 플랫폼
        'RTX',   # Raytheon Technologies - 방산 및 항공우주 기술의 글로벌 선도 기업
        'LMT',   # Lockheed Martin - 방산 및 우주항공 분야의 글로벌 리더
        'WM',    # Waste Management - 환경 및 자원 재활용 플랫폼
        'ETN',   # Eaton - 에너지 관리 솔루션 플랫폼
        'ADSK'  # Autodesk - 산업 설계 및 제작 플랫폼
    ],
    'Consumer Defensive': [
        'WMT',   # Walmart - 글로벌 소매 플랫폼, 전자상거래 강화
        'KO',    # Coca-Cola - 글로벌 음료 플랫폼
        'PEP',   # PepsiCo - 식음료 플랫폼, 글로벌 시장 선점
        'PG',    # Procter & Gamble - 소비재 플랫폼, 브랜드 다양성
        'COST',  # Costco - 대형 회원제 소매 플랫폼
        'CL',    # Colgate-Palmolive - 글로벌 소비재 플랫폼
        'HSY'    # Hershey - 초콜릿 및 식음료 플랫폼
        'MELI'   # MercadoLibre - 라틴아메리카 온라인 쇼핑 플랫폼
    ],
    'Energy': [
        'XOM',   # ExxonMobil - 전통 에너지 플랫폼
        'CVX',   # Chevron - 에너지 및 화학 플랫폼
        'NEE',   # NextEra Energy - 신재생 에너지 및 태양광
        'SEDG',  # SolarEdge - 태양광 발전 장비 제조
        'ENPH',  # Enphase Energy - 태양광 및 에너지 저장 솔루션
        'FSLR',   # First Solar - 태양광 패널 제조
        'ENS'    # 에너지솔루션, 천연가스
    ],
    'Basic Materials': [
        'LIN',   # Linde - 산업 가스 및 재료 플랫폼
        'ALB',   # Albemarle - 리튬 생산, 전기차 및 배터리 혁신
        'NEM',   # Newmont - 글로벌 금광 및 자원 개발
        'APD',   # Air Products - 산업 가스 및 기술 혁신
        'PPG'    # PPG Industries - 코팅 및 재료 솔루션
    ],
    'Real Estate': [
        'AMT',   # American Tower - 통신 및 데이터 인프라 부동산 플랫폼
        'PLD',   # Prologis - 물류 및 창고 관리 플랫폼
        'EQIX',  # Equinix - 데이터 센터 및 클라우드 인프라
        'PSA',   # Public Storage - 자산 관리 및 개인 보관 플랫폼
        'SPG'    # Simon Property - 쇼핑몰 및 상업용 부동산 플랫폼
    ],
    'Utilities': [
        'NEE',   # NextEra Energy - 신재생 에너지 및 전력 인프라
        'DUK',   # Duke Energy - 전력 및 가스 인프라 제공
        'SO',    # Southern Company - 전력 및 가스 서비스 플랫폼
        'AEP',   # American Electric Power - 전력 공급 및 재생 에너지 발전
        'VST',   # Vistra Energy - 전력 및 가스 플랫폼
        'SRE',   # Sempra Energy - 전력 및 가스 기반의 에너지 플랫폼
        'LUNR'   # LUNR - 전력 및 가스 플랫폼
    ],
    'Index': [
        'VOO',      # Vanguard S&P 500 ETF - S&P 500 지수 추종 ETF
        'QQQ',      # Invesco QQQ ETF - 나스닥 100 지수 추종 ETF
    ],
    'Coin': [
        'BTC-USD',  # Bitcoin - 디지털 자산, 암호화폐 시장의 선두
        'ETH-USD',  # Ethereum - 스마트 계약 및 블록체인 플랫폼
        'XRP-USD',  # Ripple - 결제 네트워크 및 암호화폐
        'SOL-USD',  # Solana - 빠른 거래를 지원하는 블록체인 플랫폼
        'DOGE-USD'  # Dogecoin - 밈 기반 암호화폐, 최근 활용 증가
    ]
}

