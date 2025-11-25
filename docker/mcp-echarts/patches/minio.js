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
exports.isMinIOConfigured = isMinIOConfigured;
exports.storeBufferToMinIO = storeBufferToMinIO;
const node_fs_1 = __importDefault(require("node:fs"));
const node_os_1 = __importDefault(require("node:os"));
const node_path_1 = __importDefault(require("node:path"));
const minio_1 = require("minio");
const BUCKET_NAME = process.env.MINIO_BUCKET_NAME || "mcp-echarts";
/**
 * Check if MinIO is properly configured
 */
function isMinIOConfigured() {
    return !!(process.env.MINIO_ACCESS_KEY &&
        process.env.MINIO_SECRET_KEY &&
        process.env.MINIO_ENDPOINT);
}
/**
 * Get MinIO client (only create when properly configured)
 */
function getMinIOClient() {
    if (!isMinIOConfigured()) {
        return null;
    }
    const endpoint = process.env.MINIO_ENDPOINT;
    const accessKey = process.env.MINIO_ACCESS_KEY;
    const secretKey = process.env.MINIO_SECRET_KEY;
    if (!endpoint || !accessKey || !secretKey) {
        return null;
    }
    return new minio_1.Client({
        endPoint: endpoint,
        port: Number.parseInt(process.env.MINIO_PORT || "9000"),
        useSSL: process.env.MINIO_USE_SSL === "true",
        accessKey: accessKey,
        secretKey: secretKey,
    });
}
/**
 * Store Buffer to MinIO and return public URL
 */
function storeBufferToMinIO(buffer, extension, mimeType) {
    return __awaiter(this, void 0, void 0, function* () {
        const minioClient = getMinIOClient();
        if (!minioClient) {
            throw new Error("MinIO client not configured");
        }
        // Generate unique filename
        const timestamp = Date.now();
        const objectName = `charts/${timestamp}.${extension}`;
        // Create temporary file
        const tempFilePath = node_path_1.default.join(node_os_1.default.tmpdir(), `temp_${timestamp}.${extension}`);
        node_fs_1.default.writeFileSync(tempFilePath, buffer);
        try {
            // Ensure bucket exists
            const bucketExists = yield minioClient.bucketExists(BUCKET_NAME);
            if (!bucketExists) {
                yield minioClient.makeBucket(BUCKET_NAME, "us-east-1");
            }
            // Upload file to MinIO
            yield minioClient.fPutObject(BUCKET_NAME, objectName, tempFilePath, {
                "Content-Type": mimeType,
            });
            // Clean up temporary file
            node_fs_1.default.unlinkSync(tempFilePath);
            // Generate public URL using environment variables
            const useSSL = process.env.MINIO_USE_SSL === "true";
            const protocol = useSSL ? "https" : "http";
            const endPoint = process.env.MINIO_PUBLIC_ENDPOINT || process.env.MINIO_ENDPOINT || "localhost";
            const port = process.env.MINIO_PORT || "9000";
            const url = `${protocol}://${endPoint}:${port}/${BUCKET_NAME}/${objectName}`;
            return url;
        }
        catch (error) {
            // Clean up temporary file on error
            try {
                node_fs_1.default.unlinkSync(tempFilePath);
            }
            catch (_a) {
                // Ignore cleanup errors
            }
            throw error;
        }
    });
}
