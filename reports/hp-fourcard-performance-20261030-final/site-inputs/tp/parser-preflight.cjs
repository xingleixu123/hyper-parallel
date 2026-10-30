"use strict";
// CPU-only check of the helper's local parser and proposed website input plan.
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const helperPath = "D:/Obsidian Vault D/hyperllal/训练日志绘图助手/plot.cjs";
const expectedSha = "13d43733e714e4449234a79867286237a72f894d30cc5db110aa1bd394e3b4f1";
const sha = crypto.createHash("sha256").update(fs.readFileSync(helperPath)).digest("hex");
if (sha !== expectedSha) throw new Error("helper SHA differs; independently review before plotting");
const helper = require(helperPath);
const data = path.join(__dirname, "data");
const groups = {
  loss: {names: ["01-fsdp-loss.jsonl", "02-tp-loss.jsonl", "03-zero-guides.jsonl", "04-tp-delta.jsonl"],
         metrics: ["loss", "cumulative_signed_mean", "signed_delta"], n: 1000, first: 1501, last: 2500},
  performance: {names: ["05-tp-sync-perf.jsonl", "06-tp-deferred-perf.jsonl"],
         metrics: ["nominal_tokens_per_second", "step_seconds_max_rank"], n: 30, first: 1511, last: 1540},
  allocator: {names: ["07-tp-sync-allocator.jsonl", "08-tp-deferred-allocator.jsonl"],
         metrics: ["peak_allocated_rank0_gib", "peak_allocated_rank1_gib", "peak_allocated_rank2_gib", "peak_allocated_rank3_gib",
                   "peak_reserved_rank0_gib", "peak_reserved_rank1_gib", "peak_reserved_rank2_gib", "peak_reserved_rank3_gib"],
         n: 30, first: 1511, last: 1540}
};
(async () => {
  const options = { index: false, aliases: new Map(), stepRegex: null };
  const result = {schema: "tp-website-local-helper-parser-preflight-v1", helper_sha256: sha,
                  scope: "local parser and input planning only; no browser, website worker, upload, or image rendering",
                  groups: {}};
  for (const [name, spec] of Object.entries(groups)) {
    const files = [];
    for (const [i, filename] of spec.names.entries()) files.push(await helper.parseFile(path.join(data, filename), i + 1, options));
    const catalog = new Map();
    for (const file of files) for (const [key, item] of file.series) {
      if (!catalog.has(key)) catalog.set(key, item.label);
      if (item.points.length !== spec.n || item.points[0].step !== spec.first ||
          item.points[item.points.length - 1].step !== spec.last || file.nonfinite.length || file.ignoredUnstepped) {
        throw new Error(name + " series coverage or finite check failed");
      }
    }
    const keys = [...catalog.keys()].sort((a, b) => helper.category(a) - helper.category(b) || a.localeCompare(b));
    if (JSON.stringify(keys) !== JSON.stringify(spec.metrics)) throw new Error(name + " metrics differ");
    const metrics = keys.map(key => ({key, label: catalog.get(key)}));
    const warnings = helper.verifySharedSteps(files, keys);
    const plan = await helper.planWebsiteInput(files, metrics, options);
    result.groups[name] = {files: spec.names, metrics: keys, points_per_series: spec.n,
                           first_step: spec.first, last_step: spec.last, input_mode: plan.mode,
                           input_reason: plan.reason || null, warning_count: warnings.length, warnings};
  }
  console.log(JSON.stringify(result, null, 2));
})().catch(error => { console.error(error.message); process.exitCode = 1; });
