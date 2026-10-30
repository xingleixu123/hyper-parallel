'use strict';
// Read-only observation of responses already loaded by this exact online page.
const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');

const KNOWN_SNAPSHOT_SHA256 = 'a18e5182a650bee0579d5ffa0d5d4435ee40f7be81cc4f99b6d89590d7b8eed3';
const MAX_RESPONSES = 16;
const MAX_SINGLE_BODY = 2 * 1024 * 1024;
const MAX_TOTAL_SAVED = 8 * 1024 * 1024;
const MAX_CONFIG_CHARS = 32768;
const OBSERVATION_TIMEOUT_MS = 2000;

function sha256(data) {
  return crypto.createHash('sha256').update(data).digest('hex');
}

async function withinTime(promise, label) {
  let timer;
  try {
    return await Promise.race([promise, new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error(label + ' observation timed out')), OBSERVATION_TIMEOUT_MS);
    })]);
  } finally {
    clearTimeout(timer);
  }
}

function attachSiteRenderEvidence(page, output, website) {
  const origin = new URL(website).origin;
  const privateDir = path.join(output, 'private-site-render');
  const observations = [];
  const configObservations = [];
  let omittedResponseCount = 0;
  let savedBytes = 0;
  let privateDirCreated = false;
  let queue = Promise.resolve();
  let finishPromise = null;

  async function retainBody(entry, response) {
    if (entry.http_status < 200 || entry.http_status >= 300) {
      entry.capture_status = 'non_2xx_body_not_read';
      return;
    }
    const lengthHeader = response.headers()['content-length'];
    if (lengthHeader !== undefined && /^\d+$/.test(lengthHeader)) {
      entry.declared_content_length = Number(lengthHeader);
      if (entry.declared_content_length > MAX_SINGLE_BODY) {
        entry.capture_status = 'declared_length_over_limit';
        return;
      }
    }
    try {
      const raw = await withinTime(response.body(), 'response body');
      if (!Buffer.isBuffer(raw)) throw new Error('response body is not a Buffer');
      entry.body_size = raw.length;
      entry.body_sha256 = sha256(raw);
      entry.matches_known_snapshot_sha256 = entry.body_sha256 === KNOWN_SNAPSHOT_SHA256;
      if (raw.length > MAX_SINGLE_BODY) {
        entry.capture_status = 'decoded_body_over_limit';
        return;
      }
      if (savedBytes + raw.length > MAX_TOTAL_SAVED) {
        entry.capture_status = 'total_saved_bytes_limit';
        return;
      }
      if (!privateDirCreated) {
        await fs.mkdir(privateDir, { mode: 0o700 });
        privateDirCreated = true;
      }
      const filename = 'body-' + String(entry.sequence).padStart(3, '0') + '-' + entry.body_sha256 + '.bin';
      await fs.writeFile(path.join(privateDir, filename), raw, { flag: 'wx', mode: 0o600 });
      savedBytes += raw.length;
      entry.body_file_private = 'private-site-render/' + filename;
      entry.capture_status = 'captured_and_saved';
    } catch (error) {
      entry.capture_status = 'body_or_sidecar_error';
      entry.error = String(error && error.message || error).slice(0, 400);
    }
  }

  function onResponse(response) {
    try {
      const url = new URL(response.url());
      const request = response.request();
      const resourceType = request.resourceType();
      if (url.origin !== origin || request.method() !== 'GET' ||
          !(resourceType === 'script' || /\.(?:m?js)$/i.test(url.pathname))) return;
      if (observations.length >= MAX_RESPONSES) {
        omittedResponseCount += 1;
        return;
      }
      const entry = {
        sequence: observations.length + 1,
        url: url.href,
        http_status: response.status(),
        resource_type: resourceType,
        content_type: response.headers()['content-type'] || null,
        body_size: null,
        body_sha256: null,
        matches_known_snapshot_sha256: null,
        body_file_private: null,
        capture_status: 'queued',
      };
      observations.push(entry);
      queue = queue.then(() => retainBody(entry, response)).catch(error => {
        entry.capture_status = 'observer_error';
        entry.error = String(error && error.message || error).slice(0, 400);
      });
    } catch (error) {
      // This does not change the site's requests or chart behavior.
      omittedResponseCount += 1;
    }
  }

  page.on('response', onResponse);

  async function captureConfig(stage) {
    const entry = { stage, raw: null, parsed: null, status: 'unread' };
    try {
      const raw = await withinTime(page.evaluate(() => localStorage.getItem('chartConfig')), 'chartConfig');
      if (raw === null) {
        entry.status = 'absent';
      } else if (typeof raw !== 'string') {
        entry.status = 'unexpected_type';
      } else if (raw.length > MAX_CONFIG_CHARS) {
        entry.status = 'over_length_limit';
        entry.character_count = raw.length;
      } else {
        entry.raw = raw;
        try {
          entry.parsed = JSON.parse(raw);
          entry.status = 'present_json';
        } catch (error) {
          entry.status = 'invalid_json';
        }
      }
    } catch (error) {
      entry.status = 'read_error';
      entry.error = String(error && error.message || error).slice(0, 400);
    }
    configObservations.push(entry);
    return entry;
  }

  function finish() {
    if (finishPromise) return finishPromise;
    page.off('response', onResponse);
    finishPromise = (async () => {
      await queue;
      const first = configObservations.find(x => x.stage === 'before_export');
      const last = configObservations.find(x => x.stage === 'after_render');
      const comparable = first && last && first.status === 'present_json' && last.status === 'present_json';
      const foundKnown = observations.some(x => x.matches_known_snapshot_sha256 === true);
      const allBodiesObserved = observations.length > 0 && omittedResponseCount === 0 &&
        observations.every(x => x.body_sha256 !== null);
      const result = {
        schema: 'site-render-observation-v1',
        status: 'observations_only_pending_root_review',
        site_mode: 'online',
        website_origin: origin,
        known_snapshot_sha256: KNOWN_SNAPSHOT_SHA256,
        known_snapshot_found_in_captured_responses: foundKnown ? true : (allBodiesObserved ? false : null),
        known_snapshot_match_scope: 'Only recorded JS responses; false does not identify the page entrypoint and null means incomplete/no bodies.',
        response_limits: { max_responses: MAX_RESPONSES, max_single_decoded_body_bytes: MAX_SINGLE_BODY,
          max_total_saved_body_bytes: MAX_TOTAL_SAVED, observation_timeout_ms: OBSERVATION_TIMEOUT_MS },
        same_origin_js_responses: observations,
        responses_omitted_after_limit_or_metadata_error: omittedResponseCount,
        private_sidecar_saved_bytes: savedBytes,
        chart_config_observations: configObservations,
        chart_config_raw_equal_when_both_present_json: comparable ? first.raw === last.raw : null,
        chart_or_canvas_render_state: 'not_observed',
        interpretation: 'Observed response bodies and localStorage only; no automatic no-smoothing/no-decimation or final graph acceptance.',
      };
      await fs.writeFile(path.join(output, 'site-render-evidence.json'), JSON.stringify(result, null, 2) + '\n',
        { flag: 'wx', mode: 0o600 });
      return result;
    })();
    return finishPromise;
  }

  return { captureConfig, finish };
}

module.exports = { attachSiteRenderEvidence, KNOWN_SNAPSHOT_SHA256 };