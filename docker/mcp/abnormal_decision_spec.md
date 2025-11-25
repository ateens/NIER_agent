# `abnormal_decision` 도구 명세

## 개요
`abnormal_decision` 도구는 기존의 `timeseries_analysis`, `insight_retrieval`, `response_orchestration` 세 가지 도구를 통합하여, 사용자의 이상 판정 요청을 단일 단계로 처리합니다.

## 변경 사항 요약
- **통합**: 3개의 도구 -> 1개의 도구 (`abnormal_decision`)
- **데이터 흐름**: `cache_key`를 통한 데이터 전달 방식 제거. 메모리 내에서 직접 데이터 전달.
- **목적**: 시계열 분석, 유사 사례 검색, 최종 응답 생성을 일괄 수행.

## 입력 인자 (Input Arguments)

| 인자명 | 타입 | 필수 여부 | 기본값 | 설명 | 출처 (기존 도구) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `station_id` | int | **Yes** | - | 측정소 ID | `timeseries_analysis` |
| `element` | str | **Yes** | - | 측정 항목 (예: SO2, NO2 등) | `timeseries_analysis` |
| `start_time` | str | **Yes** | - | 분석 시작 시간 (YYYY-MM-DD HH:MM:SS) | `timeseries_analysis` |
| `end_time` | str | **Yes** | - | 분석 종료 시간 (YYYY-MM-DD HH:MM:SS) | `timeseries_analysis` |

### 제거된 인자 (내부 처리 또는 불필요)
- `history`, `messages`, `window_size`, `comparison_type`, `missing_value_policy`, `collection`, `device`, `filters`: 내부 기본값 또는 전역 상수로 처리됨.
- `cache_key`, `values`: 내부 데이터 흐름으로 대체됨.
- `perform_embedding`, `perform_search`: 항상 `True`로 동작 (이상 판정 목적).
- `response_type`: "analysis"로 고정 (필요 시 "general" 분기 가능하나 주 목적은 분석).
- `query`, `raw_data`, `neighbors`, `insight`, `station_context`: 내부 로직에서 생성 및 전달.
- `log_stage`, `log_payload`: 내부 로깅 로직으로 통합 가능.

## 출력 명세 (Output Specification)

도구의 반환값은 `response_orchestration`의 최종 출력과 동일한 구조를 가집니다.

```json
{
  "answer": "string",      // LLM이 사용자에게 전달할 최종 분석 텍스트 (마크다운 형식)
  "graph_image": {         // (Optional) 그래프 이미지 정보
    "type": "image",
    "url": "string",
    "mimeType": "image/png"
  }
}
```

## 기존 도구와의 비교

### 1. `timeseries_analysis`
- **기존**: 시계열 데이터를 DB에서 조회하고 분석 후, 결과를 캐싱하고 `cache_key` 반환.
- **변경**: 데이터를 조회 및 분석하고, 그 결과(raw values)를 다음 단계(`insight_retrieval` 로직)로 직접 전달.

### 2. `insight_retrieval`
- **기존**: `cache_key`를 입력받아 캐시된 데이터를 로드하고, 임베딩 생성 및 벡터 검색 수행.
- **변경**: 앞 단계에서 전달받은 raw values를 사용하여 즉시 임베딩 생성 및 검색 수행.

### 3. `response_orchestration`
- **기존**: 모든 분석 결과(`raw_data`, `neighbors`, `insight` 등)를 인자로 받아 최종 텍스트 생성.
- **변경**: 내부 변수로 저장된 분석 결과들을 사용하여 최종 텍스트 생성.
