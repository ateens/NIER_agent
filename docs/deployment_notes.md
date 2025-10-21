# Deployment Checklist (NIER LangFlow)

현재 개발 환경에서는 모델(`models/`)과 데이터(`data/`) 디렉터리를 Docker 컨테이너에 **마운트**하여 빠르게 반복 작업을 할 수 있도록 구성되어 있습니다. 최종 배포 시에는 아래 항목을 수행하여 모든 리소스를 이미지에 포함시키고, 외부 의존성을 제거하세요.

1. **Dockerfile 수정**
   - `docker/mcp/Dockerfile`
     - `COPY models/ /app/models/`
     - `COPY data/ /app/data/`
   - `docker/langflow/Dockerfile`
     - `COPY models/ /app/models/`
     - `COPY data/ /app/data/`
   > 개발 단계에서는 빌드 시간을 줄이기 위해 위 항목을 주석 처리해 두었습니다. 배포 시에는 주석을 해제하거나 동일한 내용을 추가해 주세요.

2. **docker-compose.yml 조정**
   - `mcp`와 `langflow` 서비스에서 사용 중인 볼륨 마운트(`../models:/app/models`, `../data:/app/data`)를 제거합니다.
   - 필요하다면 `extra_hosts`(host.docker.internal) 설정도 배포 대상 환경에 맞춰 조정합니다.

3. **이미지 빌드 및 테스트**
   ```bash
   docker-compose build
   docker-compose up -d
   # 모델/데이터가 이미지에 포함됐는지 확인
   docker exec -it docker_mcp_1 ls /app/models
   ```

4. **테스트 후 태깅 및 배포**
   - `docker save` 또는 레지스트리 푸시를 사용하여 4개 서비스(vllm, mcp, chroma, langflow) 이미지를 아티팩트로 전달합니다.
   - 운영 환경에서 `docker-compose up -d` 로 동일한 스택을 재현할 수 있도록 `docker` 디렉터리와 `.env.langflow` 등을 함께 패키징합니다.

개발 중에는 현재 구조(볼륨 마운트)를 유지해도 되며, 위 체크리스트는 **최종 배포/배포 패키징 시에만** 수행하면 됩니다.
