# Container Ports Overview

| Service | Container Name (compose) | Default Port | Host Access | Notes |
|---------|--------------------------|--------------|-------------|-------|
| LangFlow | langflow | 7860 | http://127.0.0.1:7860 | Uses host network (`network_mode: host`), so 추가 포트 매핑 불필요. |
| MCP Server | mcp | 9999 | http://127.0.0.1:9999/sse | FastMCP SSE 엔드포인트, LangFlow에서 MCP 연결 시 사용. |
| vLLM | vllm | 8005 | http://127.0.0.1:8005/v1 | OpenAI 호환 API; LangFlow/MCP에서 `OPENAI_API_BASE` 로 참조. |
| ChromaDB | chroma | 8000 | http://127.0.0.1:8000 | 로컬 벡터 DB. `/app/chroma_db` 볼륨을 사용. |

> 참고: 모든 컨테이너는 `host.docker.internal` 이름을 통해 서로 통신하도록 구성되어 있습니다. |
