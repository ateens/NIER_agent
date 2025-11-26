#!/usr/bin/env node
"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const node_crypto_1 = require("node:crypto");
const node_process_1 = __importDefault(require("node:process"));
const node_util_1 = require("node:util");
const mcp_js_1 = require("@modelcontextprotocol/sdk/server/mcp.js");
const sse_js_1 = require("@modelcontextprotocol/sdk/server/sse.js");
const stdio_js_1 = require("@modelcontextprotocol/sdk/server/stdio.js");
const streamableHttp_js_1 = require("@modelcontextprotocol/sdk/server/streamableHttp.js");
const types_js_1 = require("@modelcontextprotocol/sdk/types.js");
const dotenv_1 = require("dotenv");
const express_1 = __importDefault(require("express"));
const tools_1 = require("./tools/index.js");
// Load environment variables from .env file (completely silent to avoid stdout contamination)
node_process_1.default.env.DOTENV_CONFIG_QUIET = "true";
(0, dotenv_1.config)({ override: false, debug: false });
/**
 * MCP Server for ECharts.
 * This server provides tools for generating ECharts visualizations and validate ECharts configurations.
 */
function createEChartsServer() {
    const server = new mcp_js_1.McpServer({
        name: "mcp-echarts",
        version: "0.1.0",
    });
    for (const tool of tools_1.tools) {
        const { name, description, inputSchema, run } = tool;
        // biome-ignore lint/suspicious/noExplicitAny: <explanation>
        server.tool(name, description, inputSchema.shape, run);
    }
    return server;
}
// Parse command line arguments
const { values } = (0, node_util_1.parseArgs)({
    options: {
        transport: {
            type: "string",
            short: "t",
            default: "stdio",
        },
        port: {
            type: "string",
            short: "p",
            default: "3033",
        },
        endpoint: {
            type: "string",
            short: "e",
            default: "", // We'll handle defaults per transport type
        },
        help: {
            type: "boolean",
            short: "h",
        },
    },
});
// Display help information if requested
if (values.help) {
    console.log(`
MCP ECharts CLI

Options:
  --transport, -t  Specify the transport protocol: "stdio", "sse", or "streamable" (default: "stdio")
  --port, -p       Specify the port for SSE or streamable transport (default: 3033)
  --endpoint, -e   Specify the endpoint for the transport:
                   - For SSE: default is "/sse"
                   - For streamable: default is "/mcp"
  --help, -h       Show this help message
  `);
    node_process_1.default.exit(0);
}
// Main function to start the server
function main() {
    return __awaiter(this, void 0, void 0, function* () {
        var _a;
        const transport = ((_a = values.transport) === null || _a === void 0 ? void 0 : _a.toLowerCase()) || "stdio";
        const port = Number.parseInt(values.port, 10);
        if (transport === "sse") {
            const endpoint = values.endpoint || "/sse";
            yield runSSEServer(port, endpoint);
        }
        else if (transport === "streamable") {
            const endpoint = values.endpoint || "/mcp";
            yield runStreamableHTTPServer(port, endpoint);
        }
        else {
            yield runStdioServer();
        }
    });
}
function runStdioServer() {
    return __awaiter(this, void 0, void 0, function* () {
        const server = createEChartsServer();
        const transport = new stdio_js_1.StdioServerTransport();
        yield server.connect(transport);
    });
}
function runSSEServer(port, endpoint) {
    return __awaiter(this, void 0, void 0, function* () {
        const app = (0, express_1.default)();
        app.use(express_1.default.json({ limit: "50mb" }));
        // Store transports by session ID
        const transports = {};
        // SSE endpoint
        app.get(endpoint, (req, res) => __awaiter(this, void 0, void 0, function* () {
            const server = createEChartsServer();
            const transport = new sse_js_1.SSEServerTransport("/messages", res);
            transports[transport.sessionId] = transport;
            res.on("close", () => {
                delete transports[transport.sessionId];
            });
            yield server.connect(transport);
        }));
        // Message endpoint for SSE
        app.post("/messages", (req, res) => __awaiter(this, void 0, void 0, function* () {
            const sessionId = req.query.sessionId;
            const transport = transports[sessionId];
            if (transport) {
                yield transport.handlePostMessage(req, res, req.body);
            }
            else {
                res.status(400).send("No transport found for sessionId");
            }
        }));
        app.listen(port, () => {
            console.log(`MCP ECharts SSE server running on http://localhost:${port}${endpoint}`);
        });
    });
}
function runStreamableHTTPServer(port, endpoint) {
    return __awaiter(this, void 0, void 0, function* () {
        const app = (0, express_1.default)();
        app.use(express_1.default.json({ limit: "50mb" }));
        // Store transports by session ID
        const transports = {};
        // Handle POST requests for client-to-server communication
        app.post(endpoint, (req, res) => __awaiter(this, void 0, void 0, function* () {
            const sessionId = req.headers["mcp-session-id"];
            let transport;
            if (sessionId && transports[sessionId]) {
                // Reuse existing transport
                transport = transports[sessionId];
            }
            else if (!sessionId && (0, types_js_1.isInitializeRequest)(req.body)) {
                // New initialization request
                transport = new streamableHttp_js_1.StreamableHTTPServerTransport({
                    sessionIdGenerator: () => (0, node_crypto_1.randomUUID)(),
                    onsessioninitialized: (sessionId) => {
                        transports[sessionId] = transport;
                    },
                });
                // Clean up transport when closed
                transport.onclose = () => {
                    if (transport.sessionId) {
                        delete transports[transport.sessionId];
                    }
                };
                const server = createEChartsServer();
                yield server.connect(transport);
            }
            else {
                // Invalid request
                res.status(400).json({
                    jsonrpc: "2.0",
                    error: {
                        code: -32000,
                        message: "Bad Request: No valid session ID provided",
                    },
                    id: null,
                });
                return;
            }
            // Handle the request
            yield transport.handleRequest(req, res, req.body);
        }));
        // Handle GET requests for server-to-client notifications via SSE
        app.get(endpoint, (req, res) => __awaiter(this, void 0, void 0, function* () {
            const sessionId = req.headers["mcp-session-id"];
            if (!sessionId || !transports[sessionId]) {
                res.status(400).send("Invalid or missing session ID");
                return;
            }
            const transport = transports[sessionId];
            yield transport.handleRequest(req, res);
        }));
        // Handle DELETE requests for session termination
        app.delete(endpoint, (req, res) => __awaiter(this, void 0, void 0, function* () {
            const sessionId = req.headers["mcp-session-id"];
            if (!sessionId || !transports[sessionId]) {
                res.status(400).send("Invalid or missing session ID");
                return;
            }
            const transport = transports[sessionId];
            yield transport.handleRequest(req, res);
        }));
        app.listen(port, () => {
            console.log(`MCP ECharts Streamable HTTP server running on http://localhost:${port}${endpoint}`);
        });
    });
}
// Error handling for uncaught exceptions and unhandled rejections
node_process_1.default.on("uncaughtException", (error) => {
    console.error("Uncaught exception:", error);
    node_process_1.default.exit(1);
});
node_process_1.default.on("unhandledRejection", (reason, promise) => {
    console.error("Unhandled rejection at:", promise, "reason:", reason);
    node_process_1.default.exit(1);
});
// Start application
main().catch((error) => {
    node_process_1.default.exit(1);
});
