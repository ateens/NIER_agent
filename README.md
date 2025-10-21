# NIER LangFlow Development Sandbox

이 리포지터리는 NIER 파이프라인을 LangFlow + MCP + vLLM + ChromaDB 구성으로 옮기기 위한 Docker 기반 작업 공간입니다.

## 구조

- `artifacts/langflow/` : LangFlow UI에서 Export 한 플로우 파일을 저장합니다.
- `data/` : ChromaDB 스냅샷 및 기타 파이프라인 입력 데이터.
- `docker/` : 서비스별 Dockerfile 혹은 설정(`docker-compose.yml`).
- `docker/env_snapshots/` : 기존 Conda 환경을 `pip freeze` 로 추출한 참고용 스냅샷.
- `docker/mcp/app/` : MCP 서버 코드 및 예제 (`examples/vllm_example.py`).
- `models/` : vLLM 또는 임베딩 모델 배치.

## 사용 방법

1. **LangFlow 플로우 작성** – `./run_langflow.sh` 실행 → 플로우 작성 후 `artifacts/langflow/`에 Export. 종료는 `./stop_langflow.sh`.
2. **MCP 서버 구현** – `docker/mcp/app/` 아래에 FastMCP 코드를 작성 (`examples/vllm_example.py` 참고).
3. **Docker Compose 실행** – 현재는 `models/`, `data/` 를 볼륨으로 마운트하여 개발 중에도 바로 수정 사항이 반영되도록 구성되어 있습니다.
   ```bash
   cd docker
   docker compose build
   docker compose up -d
   ```
   vLLM은 `vllm/vllm-openai` 공식 이미지를 사용합니다.
4. **서비스 엔드포인트**
   - LangFlow: http://127.0.0.1:7860 (host 네트워크)
   - MCP: http://127.0.0.1:9999/sse
   - vLLM: http://127.0.0.1:8005/v1
   - ChromaDB: http://127.0.0.1:8000

필요에 따라 `.env.langflow` 와 Dockerfile 을 실제 배포 환경에 맞게 조정하세요.
