#!/usr/bin/env zujs
/**
 * 02_js_zujs_multitool.js
 * -----------------------
 * QNZ zujs (JavaScript Engine) Multitool Example.
 * Demonstrates high-throughput JavaScript pipeline processing,
 * DOM streaming, data extraction, and fast in-memory object indexing.
 *
 * Execution:
 *   /home/coder/workspace/qnz/bin/zujs 02_js_zujs_multitool.js
 */

const fs = require('fs');

console.log("==================================================");
console.log("   QNZ Multitool - JavaScript / zujs Ecosystem   ");
console.log("==================================================\n");

// 1. In-Memory Data Pipeline Engine
class QNZDataPipeline {
    constructor() {
        this.records = [];
    }

    ingest(items) {
        items.forEach(item => {
            this.records.push({
                id: item.id,
                name: item.name,
                metrics: item.metrics || [],
                processed_at: new Date().toISOString()
            });
        });
    }

    queryByMetricMin(minValue) {
        return this.records.filter(r => {
            const avg = r.metrics.reduce((a, b) => a + b, 0) / (r.metrics.length || 1);
            return avg >= minValue;
        });
    }

    summarize() {
        return {
            total_records: this.records.length,
            sample: this.records.slice(0, 2)
        };
    }
}

// 2. High-speed HTML / Scraping Document Parser Simulating zujs DOM
function parseHTMLDocument(htmlContent) {
    const titleMatch = htmlContent.match(/<title>(.*?)<\/title>/i);
    const title = titleMatch ? titleMatch[1] : "Untitled";

    const linkMatches = [...htmlContent.matchAll(/href=["'](.*?)["']/gi)];
    const links = linkMatches.map(m => m[1]);

    return { title, linkCount: links.length, links };
}

// 3. Execution Pipeline
console.log("[1] Initializing zujs Data Pipeline...");
const pipeline = new QNZDataPipeline();

pipeline.ingest([
    { id: 101, name: "qnz_core_node_1", metrics: [88.5, 92.0, 95.4] },
    { id: 102, name: "zujs_worker_node_2", metrics: [45.0, 52.1, 49.8] },
    { id: 103, name: "zero_cc_compiler_node", metrics: [99.1, 98.4, 100.0] },
]);

console.log("   Pipeline Ingestion Complete.");
console.log("   Summary:", JSON.stringify(pipeline.summarize(), null, 2));

const highPerforming = pipeline.queryByMetricMin(90.0);
console.log(`\n[2] High-Performing Nodes (Avg Metric >= 90): ${highPerforming.length} nodes found`);
highPerforming.forEach(node => console.log(`   - [ID ${node.id}] ${node.name}`));

// 4. HTML Parser Test
console.log("\n[3] Running zujs HTML Streaming Parser...");
const sampleHTML = `
<!DOCTYPE html>
<html>
<head><title>QNZ zujs Multitool Dashboard</title></head>
<body>
    <a href="/qnz/c99">C99 Docs</a>
    <a href="/qnz/zujs">zujs Guide</a>
    <a href="/qnz/zero_cc">zero_cc Toolchain</a>
</body>
</html>
`;

const parsed = parseHTMLDocument(sampleHTML);
console.log(`   Document Title: "${parsed.title}"`);
console.log(`   Extracted Links (${parsed.linkCount}):`, parsed.links);

console.log("\n[OK] QNZ zujs Multitool execution completed successfully.");
