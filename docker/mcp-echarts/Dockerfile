FROM node:18-alpine

# Install mcp-echarts globally
RUN npm install -g mcp-echarts

# Create a working directory
WORKDIR /app

# Copy patched files
# Context is the project root, so include docker/ prefix
COPY docker/mcp-echarts/patches/index.js /usr/local/lib/node_modules/mcp-echarts/build/tools/index.js
COPY docker/mcp-echarts/patches/graph.js /usr/local/lib/node_modules/mcp-echarts/build/tools/graph.js
COPY docker/mcp-echarts/patches/line.js /usr/local/lib/node_modules/mcp-echarts/build/tools/line.js
COPY docker/mcp-echarts/patches/minio.js /usr/local/lib/node_modules/mcp-echarts/build/utils/minio.js
COPY docker/mcp-echarts/patches/server_index.js /usr/local/lib/node_modules/mcp-echarts/build/index.js

# Expose the port
EXPOSE 3033

# Run the server
ENTRYPOINT ["mcp-echarts"]
CMD ["--port", "3033", "--transport", "sse"]
