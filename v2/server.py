#!/usr/bin/env python3
"""Lamio v2 HTTP server. Serves the web UI and proxies generation to the lamio binary."""

import http.server
import json
import os
import subprocess
import sys
import time
import glob
import struct
import logging
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('lamio')

HOST = '0.0.0.0'
PORT = 5180
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
LAMIO_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build', 'src', 'lamio')
LD_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'build', 'bin')

current_model = None  # full path
current_model_name = None
telemetry = {'total_tokens': 0, 'last_tps': 0, 'last_elapsed': 0}


def get_rss():
    """Get RSS of this server process in MB."""
    try:
        with open(f'/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) // 1024
    except:
        pass
    return 0


_hdr_cache = {}

def parse_gguf_header(path):
    """Read GGUF metadata for arch, params, layers, context. Cached."""
    if path in _hdr_cache:
        return _hdr_cache[path]
    info = {'arch': 'unknown', 'params': '', 'n_layers': 0, 'n_ctx': 0, 'n_experts': 0}
    try:
        with open(path, 'rb') as f:
            magic = f.read(4)
            if magic != b'GGUF':
                _hdr_cache[path] = info
                return info
            f.read(4)  # version
            n_kv = struct.unpack('<Q', f.read(8))[0]
            f.read(8)  # n_tensors
            kvs = {}
            for idx in range(n_kv):
                klen = struct.unpack('<Q', f.read(8))[0]
                if klen > 10000:
                    log.warning(f'gguf parse desync at KV idx {idx}, klen={klen} -- stopping early')
                    break
                key = f.read(klen).decode('utf-8', errors='replace')
                vtype = struct.unpack('<I', f.read(4))[0]
                try:
                    val = _read_kv_value(f, vtype)
                except (struct.error, OverflowError, ValueError):
                    log.warning(f'gguf parse error at KV idx {idx} ({key}) -- stopping early')
                    break
                if val is not None:
                    kvs[key] = val
            arch = kvs.get('general.architecture', 'unknown')
            info['arch'] = arch
            info['n_layers'] = kvs.get(f'{arch}.block_count', 0)
            info['n_ctx'] = kvs.get(f'{arch}.context_length', 0)
            info['n_experts'] = kvs.get(f'{arch}.expert_count', 0)
    except Exception as e:
        log.warning(f'parse_gguf_header({path}): {e}')
    _hdr_cache[path] = info
    return info


def _read_kv_value(f, vtype):
    """Read a GGUF KV value based on type enum."""
    if vtype == 0:
        return struct.unpack('<B', f.read(1))[0]   # uint8
    elif vtype == 1:
        return struct.unpack('<b', f.read(1))[0]   # int8
    elif vtype == 2:
        return struct.unpack('<H', f.read(2))[0]    # uint16
    elif vtype == 3:
        return struct.unpack('<h', f.read(2))[0]    # int16
    elif vtype == 4:
        return struct.unpack('<I', f.read(4))[0]    # uint32
    elif vtype == 5:
        return struct.unpack('<i', f.read(4))[0]    # int32
    elif vtype == 6:
        return struct.unpack('<f', f.read(4))[0]    # float32
    elif vtype == 7:
        return struct.unpack('<?', f.read(1))[0]    # bool
    elif vtype == 8:
        slen = struct.unpack('<Q', f.read(8))[0]
        return f.read(slen).decode('utf-8', errors='replace')
    elif vtype == 9:
        # Array: elem_type (uint32) + count (uint64) + elements
        elem_type = struct.unpack('<I', f.read(4))[0]
        count = struct.unpack('<Q', f.read(8))[0]
        for _ in range(count):
            _read_kv_value(f, elem_type)
        return None
    elif vtype == 10:
        return struct.unpack('<Q', f.read(8))[0]  # uint64
    elif vtype == 11:
        return struct.unpack('<q', f.read(8))[0]  # int64
    elif vtype == 12:
        return struct.unpack('<d', f.read(8))[0]  # float64
    else:
        return None


class Handler(http.server.BaseHTTPRequestHandler):
    # No protocol_version override - use default HTTP/1.0 to avoid keepalive issues

    def _send_json(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _serve_file(self, path, ctype):
        try:
            with open(path, 'rb') as f:
                body = f.read()
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass
        except FileNotFoundError:
            self._send_json({'error': 'not found'}, 404)

    def do_GET(self):
        if self.path == '/api/health':
            self._send_json({
                'status': 'online' if current_model else 'ready',
                'model_loaded': current_model is not None,
                'model': current_model_name,
                'arch': parse_gguf_header(current_model)['arch'] if current_model else None,
                'n_ctx': parse_gguf_header(current_model)['n_ctx'] if current_model else 0,
            })
        elif self.path == '/api/models':
            models = []
            for p in sorted(glob.glob(os.path.join(MODELS_DIR, '*.gguf'))):
                if os.path.basename(p).startswith('ggml-vocab'):
                    continue
                name = os.path.basename(p)
                size = os.path.getsize(p)
                hdr = parse_gguf_header(p)
                models.append({
                    'name': name,
                    'size': size,
                    'arch': hdr['arch'],
                    'n_layers': hdr['n_layers'],
                    'n_ctx': hdr['n_ctx'],
                    'n_experts': hdr['n_experts'],
                    'loaded': (p == current_model),
                })
            self._send_json(models)
        elif self.path == '/api/telemetry':
            self._send_json({**telemetry, 'model': current_model_name, 'rss_mb': get_rss()})
        elif self.path == '/api/memory':
            self._send_json({'layers': []})
        elif self.path == '/' or self.path == '/index.html':
            self._serve_file(os.path.join(WEB_DIR, 'index.html'), 'text/html')
        elif self.path == '/style.css':
            self._serve_file(os.path.join(WEB_DIR, 'style.css'), 'text/css')
        elif self.path == '/app.js':
            self._serve_file(os.path.join(WEB_DIR, 'app.js'), 'application/javascript')
        else:
            self._send_json({'error': 'not found'}, 404)

    def do_POST(self):
        global current_model, current_model_name, telemetry
        length = int(self.headers.get('Content-Length', 0))
        try:
            body = json.loads(self.rfile.read(length)) if length > 0 else {}
        except json.JSONDecodeError:
            self._send_json({'error': 'invalid JSON'}, 400)
            return

        if self.path == '/api/models/load':
            model_name = body.get('model', '')
            path = os.path.join(MODELS_DIR, model_name)
            if not os.path.isfile(path):
                self._send_json({'error': 'model not found'}, 404)
                return
            current_model = path
            current_model_name = model_name
            telemetry = {'total_tokens': 0, 'last_tps': 0, 'last_elapsed': 0}
            log.info(f'loaded model: {model_name}')
            self._send_json({'ok': True})

        elif self.path == '/api/models/unload':
            current_model = None
            current_model_name = None
            log.info('unloaded model')
            self._send_json({'ok': True})

        elif self.path == '/api/generate':
            if not current_model:
                self._send_json({'error': 'no model loaded'}, 400)
                return

            prompt = body.get('prompt', '')
            history = body.get('history', [])
            temp = body.get('temperature', 0.7)
            top_k = body.get('top_k', 40)
            top_p = body.get('top_p', 0.9)
            repeat_pen = body.get('repeat_penalty', 1.1)
            n_predict = body.get('n_predict', 128)
            seed = body.get('seed', -1)
            n_ctx = body.get('n_ctx', 2048)

            # Truncate history to fit context window (rough char-based estimate: 4 chars ~= 1 token)
            max_hist_chars = max(0, (n_ctx - n_predict - len(prompt))) * 4
            truncated_hist = []
            char_count = 0
            for msg in reversed(history):
                if char_count + len(msg) > max_hist_chars:
                    break
                truncated_hist.insert(0, msg)
                char_count += len(msg)

            full_prompt = ' '.join(truncated_hist) + ' ' + prompt if truncated_hist else prompt

            args = [LAMIO_BIN, current_model, '--generate', '--prompt', full_prompt,
                    '--n-gen', str(n_predict), '--temp', str(temp),
                    '--top-k', str(top_k), '--top-p', str(top_p),
                    '--repeat-penalty', str(repeat_pen)]
            if seed and seed > 0:
                args.extend(['--seed', str(seed)])

            env = os.environ.copy()
            env['LD_LIBRARY_PATH'] = LD_PATH

            log.info(f'generate: prompt_len={len(full_prompt)} n_predict={n_predict} '
                     f'hist={len(truncated_hist)} ctx={n_ctx}')
            t0 = time.time()
            try:
                result = subprocess.run(args, capture_output=True, text=True,
                                        timeout=600, env=env)
                elapsed = time.time() - t0

                if result.returncode != 0:
                    log.error(f'lamio exit {result.returncode}: {result.stderr[-200:]}')
                    self._send_json({'error': f'lamio exited with code {result.returncode}',
                                     'stderr': result.stderr[-500:]}, 500)
                    return

                text = ''
                for line in result.stdout.splitlines():
                    if line.startswith('decoded: '):
                        text = line[9:]
                        break

                # Count actual generated tokens from output lines like "[0] token=..."
                gen_lines = [l for l in result.stdout.splitlines() if l.startswith('[') and 'token=' in l]
                n_tok = len(gen_lines)
                tps = n_tok / elapsed if elapsed > 0 else 0
                telemetry['total_tokens'] += n_tok
                telemetry['last_tps'] = round(tps, 2)
                telemetry['last_elapsed'] = round(elapsed, 2)

                log.info(f'generate done: {n_tok} tokens in {elapsed:.1f}s ({tps:.1f} t/s)')
                self._send_json({'text': text, 'tokens': n_tok, 'tps': round(tps, 2),
                                 'elapsed': round(elapsed, 2)})
            except subprocess.TimeoutExpired:
                log.error(f'generate timed out after 600s')
                self._send_json({'error': 'generation timed out (600s limit)'}, 504)
            except Exception as e:
                log.error(f'generate exception: {traceback.format_exc()}')
                self._send_json({'error': str(e)}, 500)
        else:
            self._send_json({'error': 'not found'}, 404)

    def log_message(self, format, *args):
        # Suppress default request logging; we log manually in handlers
        pass


def main():
    srv = http.server.HTTPServer((HOST, PORT), Handler)
    log.info(f'Lamio server on http://{HOST}:{PORT}')
    log.info(f'Models dir: {MODELS_DIR}')
    log.info(f'Binary: {LAMIO_BIN}')
    log.info(f'LD_LIBRARY_PATH: {LD_PATH}')

    # Verify binary exists
    if not os.path.isfile(LAMIO_BIN):
        log.error(f'Binary not found: {LAMIO_BIN}')
        log.error('Build first: cd v2 && cmake --build build -j 2')
        sys.exit(1)

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log.info('shutting down')
        srv.shutdown()


if __name__ == '__main__':
    main()
