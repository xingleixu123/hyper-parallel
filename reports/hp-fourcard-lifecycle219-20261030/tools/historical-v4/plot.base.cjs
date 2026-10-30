#!/usr/bin/env node
'use strict';

const fs = require('node:fs');
const fsp = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const readline = require('node:readline');
const crypto = require('node:crypto');

const WEBSITE = 'https://log.javazero.top/';
const NUMBER = '[+-]?(?:(?:\\d+(?:\\.\\d*)?)|(?:\\.\\d+))(?:[eE][+-]?\\d+)?';
const FIELD = new RegExp(
  '(?<![A-Za-z0-9_])' +
  '([A-Za-z][A-Za-z0-9_./%\\-]*(?:[ \\t]+[A-Za-z][A-Za-z0-9_./%\\-]*)*' +
  '(?:[ \\t]*\\([A-Za-z0-9_./% \\-]+\\))?)' +
  '["\\x27]?[ \\t]*(?::|=|[ \\t]+is)[ \\t]*' +
  '(?:tensor\\([ \\t]*|\\[[ \\t]*)?' +
  '(' + NUMBER + '|[+-]?(?:nan|inf(?:inity)?))(?![A-Za-z0-9_.])',
  'gi'
);
const STEP = /(?<![A-Za-z0-9_])(?:global[ _-]+step|step|iteration|iter)(?![A-Za-z0-9_])["']?[ \t]*(?::[ \t]*|=[ \t]*|[ \t]+)(\d+)(?![A-Za-z0-9_.])/gi;
const SKIP_KEYS = /^(?:global[ _-]+step|step|steps|iteration|iter|epoch|rank|local[ _-]+rank|world[ _-]+size)$/i;

function normKey(key) {
  return key.trim().replace(/\s+/g, ' ').toLowerCase();
}

function category(key) {
  const text = normKey(key);
  if (/(?:^|[ _/.-])loss(?:$|[ _/.-])/.test(text)) return 0;
  if (/(?:throughput|tokens?[ _/.-]*(?:per[ _-]*)?(?:sec|second|s)(?:$|[ _/.-])|(?:^|[ _/.-])(?:tps|tflops?)(?:$|[ _/.-]))/.test(text)) return 1;
  return 2;
}

function fieldsInLine(line) {
  // Keep nested JSON paths distinct, e.g. loss/mtp and loss/aux.
  const jsonStart = line.indexOf('{');
  if (jsonStart >= 0 && line.trimEnd().endsWith('}')) {
    const marker = Symbol('numeric-json-token');
    try {
      const json = JSON.parse(line.slice(jsonStart), (key, value, context) => {
        if (typeof value !== 'number') return value;
        return { [marker]: true, value, raw: context && context.source !== undefined ? context.source : String(value) };
      });
      const results = [];
      function visit(value, segments) {
        if (!value || typeof value !== 'object') return;
        if (value[marker]) {
          const key = segments.join('/');
          if (key && !SKIP_KEYS.test(segments[segments.length - 1])) {
            results.push({ key, raw: value.raw, value: value.value });
          }
          return;
        }
        for (const [key, child] of Object.entries(value)) visit(child, segments.concat(key));
      }
      visit(json, []);
      return results;
    } catch (_) {
      // A non-JSON log line may contain braces; use ordinary field parsing.
    }
  }
  const results = [];
  FIELD.lastIndex = 0;
  for (const match of line.matchAll(FIELD)) {
    const key = match[1].trim();
    const remaining = line.slice(match.index + match[0].length);
    if (/\[[ \t]*[+-]?(?:\d|\.)/.test(match[0]) && !/^[ \t]*\]/.test(remaining)) {
      throw new Error('指标「' + key + '」是多值数组；请使用 JSON 格式或先明确每个分量的含义。');
    }
    if (!SKIP_KEYS.test(key)) results.push({ key, raw: match[2], value: Number(match[2]) });
  }
  return results;
}

function stepsInLine(line, customStep) {
  if (customStep) {
    const match = customStep.exec(line);
    if (!match) return [];
    if (match[1] === undefined || !/^\d+$/.test(match[1])) {
      throw new Error('自定义步数正则的第一个捕获组必须是非负整数。');
    }
    return [Number(match[1])];
  }
  STEP.lastIndex = 0;
  return [...line.matchAll(STEP)].map(match => Number(match[1]));
}

function displayName(original, ordinal) {
  const stem = path.basename(original).replace(/\.(log|txt)$/i, '')
    .replace(/[^\p{L}\p{N}_-]+/gu, '_').replace(/^_+|_+$/g, '').slice(0, 48) || 'log';
  return String(ordinal).padStart(2, '0') + '_' + stem + '.log';
}

async function parseFile(file, ordinal, options) {
  const source = path.resolve(file);
  const stream = fs.createReadStream(source);
  const hash = crypto.createHash('sha256');
  stream.on('data', data => hash.update(data));
  const lines = readline.createInterface({ input: stream, crlfDelay: Infinity });
  const series = new Map();
  const nonfinite = [];
  const missingStep = [];
  const ambiguousStep = [];
  let ignoredUnstepped = 0;
  let lineNo = 0;

  for await (const originalLine of lines) {
    lineNo += 1;
    const line = originalLine.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, '');
    let fields;
    try {
      fields = fieldsInLine(line);
    } catch (error) {
      throw new Error(source + ':' + lineNo + ' ' + error.message);
    }
    if (!fields.length) continue;
    const stepValues = options.index ? [] : stepsInLine(line, options.stepRegex);
    if (stepValues.some(value => !Number.isSafeInteger(value))) {
      throw new Error(source + ':' + lineNo + ' 的步数超出安全整数范围。');
    }
    const distinctSteps = [...new Set(stepValues)];
    if (!options.index && distinctSteps.length > 1) {
      ambiguousStep.push(lineNo);
      continue;
    }
    for (const field of fields) {
      const rawKey = normKey(field.key);
      const alias = options.aliases.get(rawKey);
      const key = alias ? normKey(alias) : rawKey;
      const label = alias || field.key;
      if (!options.index && distinctSteps.length === 0) {
        if (category(key) < 2) missingStep.push({ line: lineNo, key: field.key });
        else ignoredUnstepped += 1;
        continue;
      }
      if (!Number.isFinite(field.value)) {
        nonfinite.push({ line: lineNo, key: field.key, raw: field.raw });
        continue;
      }
      let item = series.get(key);
      if (!item) {
        item = { key, label, points: [], lastStep: null };
        series.set(key, item);
      }
      const step = options.index ? item.points.length + 1 : distinctSteps[0];
      if (item.lastStep !== null && step <= item.lastStep) {
        throw new Error(source + ':' + lineNo + ' 的指标「' + field.key + '」出现重复或回退步数 ' +
          step + '（上一点 ' + item.lastStep + '）。请先区分 rank、重启片段或指定正确的步数。');
      }
      item.points.push({ step, value: field.value, raw: field.raw, line: lineNo, key: field.key });
      item.lastStep = step;
    }
  }

  if (ambiguousStep.length) {
    throw new Error(source + ' 包含相互冲突的步数，行号：' + ambiguousStep.slice(0, 8).join(', ') +
      '。请用 --step-regex 指明所需步数。');
  }
  if (missingStep.length) {
    throw new Error(source + ' 的 Loss 或吞吐指标缺少同一行的步数，例如 ' +
      missingStep.slice(0, 5).map(x => x.key + '（第 ' + x.line + ' 行）').join('、') +
      '。请核对日志格式，或明确选择 --index 按观测序号绘图。');
  }
  if (!series.size) throw new Error(source + ' 没有识别出带步数的数值指标。');
  return { source, name: displayName(source, ordinal), sha256: hash.digest('hex'), series, nonfinite, ignoredUnstepped };
}

function verifySharedSteps(files, metricKeys) {
  const warnings = [];
  for (const key of metricKeys) {
    const present = files.filter(file => file.series.has(key));
    if (present.length < 2) continue;
    const baseline = present[0];
    const baselinePoints = baseline.series.get(key).points;
    const baseMap = new Map(baselinePoints.map(point => [point.step, point.value]));
    for (const target of present.slice(1)) {
      const shared = target.series.get(key).points.filter(point => baseMap.has(point.step));
      if (!shared.length) {
        throw new Error('指标「' + key + '」在 ' + baseline.name + ' 与 ' + target.name +
          ' 中没有共同步数；不能把网站显示的零差值当成有效对比。');
      }
      const zeros = shared.filter(point => baseMap.get(point.step) === 0).length;
      if (zeros) warnings.push('指标「' + key + '」有 ' + zeros +
        ' 个共同步数的基准值为零，网站的相对误差对此会显示为 0，不能据此判断。');
    }
  }
  return warnings;
}

function websiteKeywordMatches(content, keyword) {
  const matches = [];
  const needle = keyword.toLowerCase();
  const number = /[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/;
  content.split('\n').forEach((line, index) => {
    const position = line.toLowerCase().indexOf(needle);
    if (position < 0) return;
    const match = line.substring(position + keyword.length).match(number);
    if (match) matches.push({ line: index + 1, value: Number(match[0]) });
  });
  return matches;
}

function websiteStep(line, keyword) {
  const position = line.toLowerCase().indexOf(keyword.toLowerCase());
  if (position < 0) return null;
  const match = line.substring(position + keyword.length).match(/[+-]?\d+/);
  return match ? Number(match[0]) : null;
}

async function planWebsiteInput(files, metrics, options) {
  const contents = await Promise.all(files.map(file => fsp.readFile(file.source, 'utf8')));
  const configMetrics = [];
  let stepKeyword = null;
  let reason = options.index ? '按观测序号绘图需要统一横轴' : '';
  if (!reason) {
    for (const metric of metrics) {
      const names = [...new Set(files.flatMap(file =>
        (file.series.get(metric.key)?.points || []).map(point => point.key)))];
      const candidates = [...new Set(names.flatMap(name => [name + ':', name + '=', name + '":', name + "':", name]))];
      const keyword = candidates.find(candidate => files.every((file, index) => {
        const expected = file.series.get(metric.key)?.points || [];
        const found = websiteKeywordMatches(contents[index], candidate);
        return found.length === expected.length &&
          found.every((item, pointIndex) => item.line === expected[pointIndex].line && item.value === expected[pointIndex].value);
      }));
      if (!keyword) {
        reason = '网站关键词无法准确匹配指标「' + metric.label + '」';
        break;
      }
      configMetrics.push({ name: metric.label, mode: 'keyword', keyword, regex: '' });
    }
  }
  if (!reason) {
    const candidates = ['global_step', 'global step', 'global-step', 'iteration', 'step', 'iter'];
    stepKeyword = candidates.find(candidate => files.every((file, index) => {
      const lines = contents[index].split('\n');
      return [...file.series.values()].every(item =>
        item.points.every(point => websiteStep(lines[point.line - 1] || '', candidate) === point.step));
    }));
    if (!stepKeyword) reason = '网站无法用同一个步数关键词准确识别所有曲线';
  }
  if (!reason) {
    return {
      mode: 'original',
      config: { metrics: configMetrics, useStepKeyword: true, stepKeyword },
      payloads: files.map((file, index) => ({
        name: file.name, mimeType: 'text/plain', buffer: Buffer.from(contents[index], 'utf8'),
      })),
    };
  }
  const config = {
    metrics: metrics.map((metric, index) => ({
      name: metric.label + (options.index ? ' (观测序号)' : ''),
      mode: 'keyword', keyword: '__codex_metric_' + index + '__:', regex: '',
    })),
    useStepKeyword: true,
    stepKeyword: '__codex_step__:',
  };
  const payloads = files.map(file => {
    const lines = [];
    for (let i = 0; i < metrics.length; i += 1) {
      const series = file.series.get(metrics[i].key);
      if (!series) continue;
      for (const point of series.points) {
        lines.push('__codex_step__: ' + point.step + ' __codex_metric_' + i + '__: ' + point.value);
      }
    }
    return { name: file.name, mimeType: 'text/plain', buffer: Buffer.from(lines.join('\n') + '\n', 'utf8') };
  });
  return { mode: 'normalized', reason, config, payloads };
}

async function verifyWebsiteWorker(page, workerUrl, plan, files, metrics) {
  for (let i = 0; i < files.length; i += 1) {
    const observed = await page.evaluate(({ workerUrl, content, config, name }) => new Promise((resolve, reject) => {
      const parser = new Worker(workerUrl, { type: 'module' });
      const timeout = setTimeout(() => { parser.terminate(); reject(new Error('网站解析校验超时')); }, 90000);
      parser.onmessage = event => {
        if (event.data.type === 'PARSE_COMPLETE') {
          clearTimeout(timeout); parser.terminate(); resolve(event.data.payload.metricsData);
        } else if (event.data.type === 'PARSE_ERROR') {
          clearTimeout(timeout); parser.terminate(); reject(new Error(event.data.payload.error));
        }
      };
      parser.onerror = event => {
        clearTimeout(timeout); parser.terminate(); reject(new Error(event.message || '网站解析器失败'));
      };
      parser.postMessage({ type: 'PARSE_FILE', payload: { fileId: name, content, config } });
    }), { workerUrl, content: plan.payloads[i].buffer.toString('utf8'), config: plan.config, name: files[i].name });
    for (let j = 0; j < metrics.length; j += 1) {
      const expected = files[i].series.get(metrics[j].key)?.points || [];
      const actual = observed[plan.config.metrics[j].name] || [];
      if (actual.length !== expected.length || actual.some((point, k) =>
        point.x !== expected[k].step || point.y !== expected[k].value)) {
        throw new Error('网站解析结果与原日志不一致：' + files[i].name + ' / ' + metrics[j].label);
      }
    }
  }
}

function csvEscape(value) {
  const text = String(value);
  return /[",\r\n]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
}

function getPlaywright() {
  try { return require('playwright'); } catch (_) { /* Try the Codex bundled runtime. */ }
  const bundled = path.join(os.homedir(), '.cache', 'codex-runtimes', 'codex-primary-runtime',
    'dependencies', 'node', 'node_modules', 'playwright');
  try { return require(bundled); } catch (_) {
    throw new Error('未找到 Playwright。请使用本机 Codex 自带运行时，或在此工具目录安装 playwright。');
  }
}

function getEdge(override) {
  const candidates = [
    override,
    process.env.EDGE_PATH,
    path.join(process.env.PROGRAMFILES || 'C:\\Program Files', 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
    path.join(process.env['PROGRAMFILES(X86)'] || 'C:\\Program Files (x86)', 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
    process.env.LOCALAPPDATA && path.join(process.env.LOCALAPPDATA, 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
  ].filter(Boolean);
  const found = candidates.find(candidate => path.basename(candidate).toLowerCase() === 'msedge.exe' && fs.existsSync(candidate));
  if (!found) throw new Error('未找到 Microsoft Edge。请使用 --edge 指定 msedge.exe 的完整路径。');
  return found;
}

function readArgs(argv) {
  const options = { files: [], out: null, headed: true, snapshot: false, index: false, aliases: new Map(), stepRegex: null, edge: null };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') options.help = true;
    else if (arg === '--view') options.headed = true;
    else if (arg === '--headless') options.headed = false;
    else if (arg === '--snapshot') options.snapshot = true;
    else if (arg === '--index') options.index = true;
    else if (['--out', '--edge', '--step-regex', '--alias'].includes(arg)) {
      if (i + 1 >= argv.length) throw new Error(arg + ' 缺少参数。');
      const value = argv[++i];
      if (arg === '--out') options.out = path.resolve(value);
      else if (arg === '--edge') options.edge = value;
      else if (arg === '--step-regex') options.stepRegex = new RegExp(value, 'i');
      else {
        const at = value.indexOf('=');
        if (at <= 0 || !value.slice(at + 1).trim()) throw new Error('--alias 格式应为 原指标名=目标指标名。');
        options.aliases.set(normKey(value.slice(0, at)), value.slice(at + 1).trim());
      }
    } else if (arg.startsWith('-')) throw new Error('未知参数：' + arg);
    else options.files.push(arg);
  }
  if (options.index && options.stepRegex) throw new Error('--index 和 --step-regex 不能同时使用。');
  return options;
}

async function makeOutputDir(requested) {
  if (requested) {
    await fsp.mkdir(requested, { recursive: true });
    if ((await fsp.readdir(requested)).length) throw new Error('输出目录不是空目录：' + requested);
    return requested;
  }
  const dir = path.join(__dirname, 'runs', new Date().toISOString().replace(/[:.]/g, '-') + '-' + crypto.randomUUID().slice(0, 8));
  await fsp.mkdir(dir, { recursive: true });
  return dir;
}

function checkCsv(text, key, files) {
  const rows = text.replace(/^\uFEFF/, '').trim().split(/\r?\n/).map(row => row.split(','));
  const header = rows.shift();
  if (!header || header[0] !== 'step') throw new Error('网站 CSV 的首列不是 step：' + key);
  const values = new Map(rows.map(row => [Number(row[0]), row]));
  for (const file of files) {
    const series = file.series.get(key);
    if (!series) continue;
    const label = file.name.replace(/\.(log|txt)$/i, '');
    const col = header.indexOf(label);
    if (col < 0) throw new Error('网站 CSV 缺少曲线：' + label + ' / ' + key);
    for (const point of series.points) {
      const row = values.get(point.step);
      if (!row || !row[col] || Number(row[col]) !== point.value) {
        throw new Error('网站 CSV 与来源不一致：' + label + ' / ' + key + ' / step ' + point.step);
      }
    }
  }
}

function checkDiffCsv(text, key, files) {
  const rows = text.replace(/^\uFEFF/, '').trim().split(/\r?\n/).map(row => row.split(','));
  const header = rows.shift();
  if (!header || header[0] !== 'step') throw new Error('网站差值 CSV 的首列不是 step：' + key);
  const values = new Map(rows.map(row => [Number(row[0]), row]));
  const present = files.filter(file => file.series.has(key));
  const base = present[0];
  const basePoints = new Map(base.series.get(key).points.map(point => [point.step, point.value]));
  const expectedSteps = new Set();
  for (const target of present.slice(1)) {
    const label = target.name + ' vs ' + base.name;
    const col = header.indexOf(label);
    if (col < 0) throw new Error('网站差值 CSV 缺少曲线：' + label + ' / ' + key);
    for (const point of target.series.get(key).points) {
      if (!basePoints.has(point.step)) continue;
      expectedSteps.add(point.step);
      const row = values.get(point.step);
      if (!row || !row[col] || Number(row[col]) !== point.value - basePoints.get(point.step)) {
        throw new Error('网站差值 CSV 与共同步数不一致：' + label + ' / ' + key + ' / step ' + point.step);
      }
    }
  }
  if (values.size !== expectedSteps.size || [...values.keys()].some(step => !expectedSteps.has(step))) {
    throw new Error('网站差值 CSV 含有非共同步数：' + key);
  }
}

async function render(files, metrics, output, options, onReady) {
  const { chromium } = getPlaywright();
  const plan = await planWebsiteInput(files, metrics, options);
  let snapshot;
  const resources = new Map();
  if (options.snapshot) {
    const snapshotDir = path.join(__dirname, 'site-snapshot');
    snapshot = JSON.parse(await fsp.readFile(path.join(snapshotDir, 'snapshot.json'), 'utf8'));
    for (const item of snapshot.resources) {
      const local = path.resolve(snapshotDir, item.path);
      if (!local.startsWith(snapshotDir + path.sep)) throw new Error('网站快照包含无效路径。');
      const bytes = await fsp.readFile(local);
      if (crypto.createHash('sha256').update(bytes).digest('hex') !== item.sha256) {
        throw new Error('网站快照文件校验失败：' + item.path);
      }
      resources.set('/' + item.path.replace(/\\/g, '/'), local);
    }
  }
  const browser = await chromium.launch({
    executablePath: getEdge(options.edge),
    headless: !options.headed,
    args: options.headed ? ['--start-maximized'] : [],
  });
  let sealed = false;
  let blockedAfterInput = 0;
  try {
    const context = await browser.newContext({
      serviceWorkers: 'block',
      offline: options.snapshot,
      acceptDownloads: true,
      viewport: { width: 1600, height: 1000 },
      deviceScaleFactor: 2,
    });
    if (options.snapshot) {
      await context.route('**/*', route => {
        const request = route.request();
        const url = new URL(request.url());
        const resourcePath = url.pathname === '/' ? '/index.html' : url.pathname;
        const local = resources.get(resourcePath);
        if (url.origin === new URL(WEBSITE).origin && request.method() === 'GET' && !url.search && local) {
          const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css',
            '.svg': 'image/svg+xml', '.webmanifest': 'application/manifest+json', '.woff2': 'font/woff2' };
          return route.fulfill({ path: local, contentType: types[path.extname(local)] || 'application/octet-stream' });
        }
        if (sealed) blockedAfterInput += 1;
        return route.abort();
      });
      if (typeof context.routeWebSocket === 'function') {
        await context.routeWebSocket('**/*', socket => socket.close());
      }
    }
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const config = plan.config;
    await page.addInitScript(settings => {
      if (location.origin !== 'https://log.javazero.top') return;
      localStorage.setItem('globalParsingConfig', JSON.stringify(settings));
      localStorage.setItem('language', 'zh');
      localStorage.setItem('theme', 'light');
    }, config);
    const response = await page.goto(WEBSITE, { waitUntil: 'domcontentloaded', timeout: 90000 });
    if (!response || !response.ok()) throw new Error('网站无法访问：' + (response ? response.status() : '无响应'));
    if (response.url() !== WEBSITE) throw new Error('网站跳转到了意外地址：' + response.url());
    await page.locator('#fileInput').waitFor({ state: 'attached', timeout: 45000 });
    const worker = page.workers()[0] || await page.waitForEvent('worker', { timeout: 45000 });
    const workerReady = await worker.evaluate(() => typeof self.onmessage);
    if (workerReady !== 'function') throw new Error('网站解析器尚未就绪，请重试或检查网站是否更新。');
    sealed = true;

    await page.locator('#fileInput').setInputFiles(plan.payloads);
    const expectedCharts = metrics.reduce((n, metric) =>
      n + 1 + (files.filter(file => file.series.has(metric.key)).length >= 2 ? 1 : 0), 0);
    await page.waitForFunction(expected => document.querySelectorAll('canvas').length >= expected, expectedCharts, { timeout: 90000 });
    if (errors.length) throw new Error('网站运行错误：' + errors.join('; '));
    await verifyWebsiteWorker(page, worker.url(), plan, files, metrics);

    if (!options.headed) {
    const chartDir = path.join(output, 'charts');
    const tableDir = path.join(output, 'website-csv');
    await fsp.mkdir(chartDir);
    await fsp.mkdir(tableDir);
    const buttons = page.getByTitle('导出', { exact: true });
    const count = await buttons.count();
    if (count !== expectedCharts) throw new Error('网站图表数与预期不符：' + count + ' / ' + expectedCharts);

    for (let i = 0; i < count; i += 1) {
      for (const format of ['PNG', 'CSV 表格']) {
        await buttons.nth(i).click();
        const downloadPromise = page.waitForEvent('download', { timeout: 30000 });
        await page.getByRole('menuitem', { name: format, exact: true }).click();
        const download = await downloadPromise;
        const match = /^metric-(comp-)?(\d+)\.(png|csv)$/i.exec(download.suggestedFilename());
        if (!match || !metrics[Number(match[2])]) throw new Error('网站导出文件名异常：' + download.suggestedFilename());
        const metric = metrics[Number(match[2])];
        const stem = String(Number(match[2]) + 1).padStart(2, '0') + '_' + metric.slug +
          (match[1] ? '_difference' : '');
        const destination = path.join(match[3].toLowerCase() === 'png' ? chartDir : tableDir, stem + '.' + match[3].toLowerCase());
        await download.saveAs(destination);
        if (match[3].toLowerCase() === 'csv') {
          const text = await fsp.readFile(destination, 'utf8');
          if (match[1]) checkDiffCsv(text, metric.key, files);
          else checkCsv(text, metric.key, files);
        }
      }
    }
    await page.getByTitle('导出报告', { exact: true }).click();
    const reportPromise = page.waitForEvent('download', { timeout: 30000 });
    await page.getByRole('menuitem', { name: '下载为 PNG', exact: true }).click();
    await (await reportPromise).saveAs(path.join(output, 'website-report.png'));
    if (errors.length) throw new Error('网站运行错误：' + errors.join('; '));
    }
    if (options.headed) {
      const session = await context.newCDPSession(page);
      try {
        const { windowId } = await session.send('Browser.getWindowForTarget');
        await session.send('Browser.setWindowBounds', { windowId, bounds: { windowState: 'maximized' } });
        const { bounds } = await session.send('Browser.getWindowBounds', { windowId });
        if (bounds.windowState === 'minimized') throw new Error('浏览器窗口仍处于最小化状态。');
        await page.bringToFront();
        await page.screenshot({ path: path.join(output, 'website-preview.png') });
      } finally {
        await session.detach();
      }
    }
    const result = {
      website: response.url(),
      site_mode: options.snapshot ? 'snapshot' : 'online',
      input_mode: plan.mode,
      step_keyword: plan.config.stepKeyword,
      matching_rules: plan.config.metrics.map(metric => ({ name: metric.name, mode: metric.mode, keyword: metric.keyword })),
      ...(plan.reason ? { input_reason: plan.reason } : {}),
      ...(options.snapshot ? { blockedAfterInput, site_snapshot_date: snapshot.snapshot_date } : {}),
    };
    await onReady(result);
    console.log('网站图表及解析结果逐点校验完成；输入：' +
      (plan.mode === 'original' ? '原始日志' : '逐点保留来源的转换日志（' + plan.reason + '）') + '。');
    if (options.headed) {
      console.log('浏览器已保持打开；关闭窗口后命令结束。');
      await new Promise(resolve => page.once('close', resolve));
    }
    return result;
  } finally {
    await browser.close();
  }
}

function usage() {
  console.log([
    '训练日志绘图助手：打开 log.javazero.top 在线网页，在隔离的 Microsoft Edge 会话中绘图',
    '用法：node plot.cjs [--out 空目录] [--view|--headless] 文件1.log [文件2.log ...]',
    '      [--step-regex 正则] [--alias 原名=目标名] [--index] [--edge 路径]',
    '默认保留可交互浏览器窗口；--headless 不打开窗口；--index 使用各指标的观测序号为 X。',
    '默认查找同一行中的 step / global_step / iteration，保留每个数值指标的原始名称。',
  ].join('\n'));
}

async function main() {
  const options = readArgs(process.argv.slice(2));
  if (options.help) { usage(); return; }
  if (!options.files.length) { usage(); throw new Error('请提供至少一份训练日志。'); }
  const files = [];
  for (let i = 0; i < options.files.length; i += 1) files.push(await parseFile(options.files[i], i + 1, options));
  const catalog = new Map();
  for (const file of files) for (const [key, item] of file.series) if (!catalog.has(key)) catalog.set(key, item.label);
  const keys = [...catalog.keys()].sort((a, b) => category(a) - category(b) || a.localeCompare(b));
  const metrics = keys.map(key => ({
    key, label: catalog.get(key),
    slug: key.replace(/[^\p{L}\p{N}_-]+/gu, '_').replace(/^_+|_+$/g, '').slice(0, 60) || 'metric',
  }));
  const warnings = verifySharedSteps(files, keys);
  for (const file of files) {
    if (file.nonfinite.length) warnings.push(file.name + ' 有 ' + file.nonfinite.length + ' 个 NaN/Inf，未作为曲线点。');
    if (file.ignoredUnstepped) warnings.push(file.name + ' 有 ' + file.ignoredUnstepped + ' 个无步数的其他数值字段未绘图。');
  }
  const output = await makeOutputDir(options.out);
  const evidence = ['source,sha256,line,source_metric,plot_metric,raw_value,step,x_kind'];
  for (const file of files) for (const item of file.series.values()) for (const point of item.points) {
    evidence.push([file.source, file.sha256, point.line, point.key, item.label, point.raw, point.step,
      options.index ? 'observation_index_1_based' : 'recorded_step'].map(csvEscape).join(','));
  }
  await fsp.writeFile(path.join(output, 'source-points.csv'), evidence.join('\n') + '\n', 'utf8');
  const summary = {
    status: 'incomplete',
    website: WEBSITE,
    x_kind: options.index ? 'observation_index_1_based' : 'recorded_step',
    metrics: metrics.map(metric => ({ key: metric.key, label: metric.label })),
    files: files.map(file => ({
      source: file.source, website_label: file.name, sha256: file.sha256,
      series: Object.fromEntries([...file.series].map(([key, item]) => [key, {
        count: item.points.length, first_step: item.points[0].step, last_step: item.points[item.points.length - 1].step,
      }])),
      nonfinite: file.nonfinite, ignored_unstepped_fields: file.ignoredUnstepped,
    })),
    warnings,
  };
  const manifest = path.join(output, 'analysis.json');
  await fsp.writeFile(manifest, JSON.stringify(summary, null, 2) + '\n', 'utf8');
  console.log('指标：' + metrics.map(metric => metric.label).join('、'));
  for (const warning of warnings) console.warn('注意：' + warning);
  console.log('输出目录：' + output);
  try {
    summary.render = await render(files, metrics, output, options, async result => {
      summary.render = result;
      summary.status = 'complete';
      await fsp.writeFile(manifest, JSON.stringify(summary, null, 2) + '\n', 'utf8');
    });
  } catch (error) {
    summary.error = error.message;
    throw error;
  } finally {
    await fsp.writeFile(manifest, JSON.stringify(summary, null, 2) + '\n', 'utf8');
  }
}

if (require.main === module) main().catch(error => { console.error('绘图失败：' + error.message); process.exitCode = 1; });
module.exports = { fieldsInLine, stepsInLine, parseFile, verifySharedSteps, category, readArgs, planWebsiteInput };
