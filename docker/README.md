# Docker Services Overview

- `docker-compose.yml`: orchestrates LangFlow, MCP server, vLLM, and ChromaDB.
- `env_snapshots/`: pip requirement snapshots exported from existing Conda envs.
- `langflow/`: build LangFlow image with exported flows (`flows/`) and custom nodes.
- `mcp/`: FastMCP server skeleton (`app/` placeholder). Replace with real implementation.
- `vllm/`: GPU-enabled vLLM server. Requires pretrained model directory under `models/`.
- `chromadb/`: standalone ChromaDB service storing data in `docker/chromadb/chroma_db`.

빌드/실행은 `docker` 디렉터리에서 다음 명령으로 수행합니다.

```bash
docker compose build
docker compose up -d
```

서비스를 중지하려면 `docker compose down` 을 사용하세요.
