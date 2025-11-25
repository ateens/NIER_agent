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
Object.defineProperty(exports, "__esModule", { value: true });
exports.generateGraphChartTool = void 0;
const zod_1 = require("zod");
const utils_1 = require("../utils/index.js");
const schema_1 = require("../utils/schema.js");
// Node schema
const NodeSchema = zod_1.z.object({
    id: zod_1.z.string().describe("Unique identifier for the node."),
    name: zod_1.z.string().describe("Display name of the node."),
    value: zod_1.z
        .number()
        .optional()
        .describe("Value associated with the node (affects size)."),
    category: zod_1.z
        .string()
        .optional()
        .describe("Category of the node (affects color)."),
});
// Edge schema
const EdgeSchema = zod_1.z.object({
    source: zod_1.z.string().describe("Source node id."),
    target: zod_1.z.string().describe("Target node id."),
    value: zod_1.z.number().optional().describe("Weight or value of the edge."),
});
exports.generateGraphChartTool = {
    name: "generate_graph_chart",
    description: "Generate a network graph chart to show relationships (edges) between entities (nodes), such as, relationships between people in social networks.",
    inputSchema: zod_1.z.object({
        data: zod_1.z
            .object({
            nodes: zod_1.z
                .array(NodeSchema)
                .describe("Array of nodes in the network.")
                .nonempty({ message: "At least one node is required." }),
            edges: zod_1.z
                .array(EdgeSchema)
                .describe("Array of edges connecting nodes.")
                .optional()
                
        })
            .describe("Data for network graph chart, such as, { nodes: [{ id: 'node1', name: 'Node 1' }], edges: [{ source: 'node1', target: 'node2' }] }"),
        height: schema_1.HeightSchema,
        layout: zod_1.z
            .enum(["force", "circular", "none"])
            .optional()
            .default("force")
            .describe("Layout algorithm for the graph. Default is 'force'."),
        theme: schema_1.ThemeSchema,
        title: schema_1.TitleSchema,
        width: schema_1.WidthSchema,
        outputType: schema_1.OutputTypeSchema,
    }),
    run: (params) => __awaiter(void 0, void 0, void 0, function* () {
        const { data, height, layout = "force", theme, title, width, outputType, } = params;
        // Validate that all edge nodes exist in nodes array
        const nodeIds = new Set(data.nodes.map((node) => node.id));
        const validEdges = (data.edges || []).filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target));
        // Extract unique categories for legend
        const categories = Array.from(new Set(data.nodes
            .map((node) => node.category)
            .filter((cat) => Boolean(cat))));
        // Transform nodes for ECharts
        const nodes = data.nodes.map((node) => ({
            id: node.id,
            name: node.name,
            symbolSize: node.value ? Math.sqrt(node.value) * 10 : 20,
            category: node.category,
            value: node.value,
        }));
        // Transform edges for ECharts
        const links = validEdges.map((edge) => ({
            source: edge.source,
            target: edge.target,
            value: edge.value,
        }));
        const series = [
            {
                type: "graph",
                data: nodes,
                links: links,
                categories: categories.map((cat) => ({ name: cat })),
                roam: true,
                layout: layout,
                force: layout === "force"
                    ? {
                        repulsion: 100,
                        gravity: 0.02,
                        edgeLength: 150,
                        layoutAnimation: true,
                    }
                    : undefined,
                label: {
                    show: true,
                    position: "right",
                    formatter: "{b}",
                },
                lineStyle: {
                    color: "source",
                    curveness: 0.3,
                },
                emphasis: {
                    focus: "adjacency",
                    label: {
                        fontSize: 16,
                    },
                },
            },
        ];
        const echartsOption = {
            series,
            title: {
                left: "center",
                text: title,
            },
            tooltip: {
                trigger: "item",
            },
            legend: categories.length > 0
                ? {
                    left: "center",
                    orient: "horizontal",
                    bottom: 10,
                    data: categories,
                }
                : undefined,
        };
        return yield (0, utils_1.generateChartImage)(echartsOption, width, height, theme, outputType, "generate_graph_chart");
    }),
};
