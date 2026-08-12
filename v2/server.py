#!/usr/bin/env python3
"""Lamio v2 HTTP server. Serves the web UI and manages a persistent lamio --repl process."""

import http.server
import json
import os
import subprocess
import sys
import time
import glob
import struct
import logging
import threading
import select

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('lamio')

HOST = '0.0.0.0'
PORT = 5180
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(FILE_DIR, 'web')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(FILE_DIR)), 'models')
# Fix: FILE_DIR = .../v2, models = .../Lamio/models (one level up from v2)
if not os.path.isdir(MODELS_DIR):
    MODELS_DIR = os.path.join(os.path.dirname(FILE_DIR), 'models')
LAMIO_BIN = os.path.join(FILE_DIR, 'build', 'src', 'lamio')
LD_PATH = os.path.join(os.path.dirname(os.path.dirname(FILE_DIR)), 'build', 'bin')
# Fix: same pattern as MODELS_DIR
if not os.path.isdir(LD_PATH):
    LD_PATH = os.path.join(os.path.dirname(FILE_DIR), 'build', 'bin')

repl_proc = None
repl_model = None
repl_lock = threading.Lock()
current_model = None
current_model_name = None
telemetry = {'total_tokens': 0, 'last_tps': 0, 'last_elapsed': 0}
_hdr_cache = {}


def get_rss():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) // 1024
    except:
        pass
    return 0


def get_repl_rss():
    if repl_proc and repl_proc.poll() is None:
        try:
            with open(f'/proc/{repl_proc.pid}/status') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        return int(line.split()[1]) // 1024
        except:
            pass
    return 0


def start_repl(model_path, temp=0.7, top_k=40, top_p=0.9, repeat_penalty=1.1, seed=-1, n_gen=128):
    """Start a persistent lamio --repl process."""
    global repl_proc, repl_model
    stop_repl()
    args = [LAMIO_BIN, model_path, '--repl', '--n-gen', str(n_gen),
            '--temp', str(temp), '--top-k', str(top_k), '--top-p', str(top_p),
            '--repeat-penalty', str(repeat_penalty)]
    if seed and seed > 0:
        args.extend(['--seed', str(seed)])
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = LD_PATH
    log.info(f'starting repl: {model_path}')
    repl_proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, env=env, bufsize=0)
    repl_model = model_path
    # Wait for "repl: ready" on stderr
    wait_for_ready()


def stop_repl():
    global repl_proc, repl_model
    if repl_proc:
        try:
            repl_proc.stdin.close()
            repl_proc.wait(timeout=5)
        except:
            repl_proc.kill()
        repl_proc = None
    repl_model = None


def wait_for_ready(timeout=120):
    """Wait for 'repl: ready' on stderr."""
    if not repl_proc:
        return False
    start = time.time()
    while time.time() - start < timeout:
        ready, _, _ = select.select([repl_proc.stderr], [], [], 0.5)
        if ready:
            line = repl_proc.stderr.readline().decode('utf-8', errors='replace').strip()
            log.info(f'repl stderr: {line}')
            if 'repl: ready' in line:
                return True
            if 'failed' in line.lower() or 'error' in line.lower():
                return False
    log.warning('repl: ready timeout')
    return False


def send_prompt(prompt):
    """Send a prompt to the persistent REPL and read the response."""
    global telemetry
    if not repl_proc or repl_proc.poll() is not None:
        return {'error': 'model not loaded'}, 500
    with repl_lock:
        import fcntl
        # Make stdout non-blocking
        stdout_fd = repl_proc.stdout.fileno()
        stderr_fd = repl_proc.stderr.fileno()
        fl = fcntl.fcntl(stdout_fd, fcntl.F_GETFL)
        fcntl.fcntl(stdout_fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

        t0 = time.time()
        try:
            repl_proc.stdin.write((prompt + '\n').encode())
            repl_proc.stdin.flush()
        except BrokenPipeError:
            return {'error': 'model process died'}, 500

        output = b''
        while True:
            # Check stderr for ready signal
            ready_err, _, _ = select.select([stderr_fd], [], [], 0.1)
            if ready_err:
                try:
                    line = os.read(stderr_fd, 4096).decode('utf-8', errors='replace').strip()
                    if line:
                        log.info(f'repl stderr: {line}')
                    if 'repl: ready' in line:
                        break
                    if 'failed' in line.lower() or 'error' in line.lower() or 'assert' in line.lower():
                        log.error(f'repl error: {line}')
                        return {'error': f'model error: {line}'}, 500
                except BlockingIOError:
                    pass

            # Read stdout
            try:
                chunk = os.read(stdout_fd, 4096)
                if chunk:
                    output += chunk
            except BlockingIOError:
                pass

            if time.time() - t0 > 600:
                log.error('repl: generation timed out')
                return {'error': 'generation timed out'}, 504

        # Restore blocking mode
        fcntl.fcntl(stdout_fd, fcntl.F_SETFL, fl)

        elapsed = time.time() - t0
        text = output.decode('utf-8', errors='replace').strip()
        n_tok = len([w for w in text.split() if w]) if text else 0
        tps = n_tok / elapsed if elapsed > 0 else 0
        telemetry['total_tokens'] += n_tok
        telemetry['last_tps'] = round(tps, 2)
        telemetry['last_elapsed'] = round(elapsed, 2)
        log.info(f'generate done: {n_tok} tokens in {elapsed:.1f}s ({tps:.1f} t/s)')
        return {'text': text, 'tokens': n_tok, 'tps': round(tps, 2), 'elapsed': round(elapsed, 2)}, 200


def parse_gguf_header(path):
    if path in _hdr_cache:
        return _hdr_cache[path]
    info = {'arch': 'unknown', 'n_layers': 0, 'n_ctx': 0, 'n_experts': 0}
    try:
        with open(path, 'rb') as f:
            magic = f.read(4)
            if magic != b'GGUF':
                _hdr_cache[path] = info
                return info
            f.read(4)
            n_kv = struct.unpack('<Q', f.read(8))[0]
            f.read(8)
            kvs = {}
            for idx in range(n_kv):
                klen = struct.unpack('<Q', f.read(8))[0]
                if klen > 10000:
                    break
                key = f.read(klen).decode('utf-8', errors='replace')
                vtype = struct.unpack('<I', f.read(4))[0]
                try:
                    val = _read_kv_value(f, vtype)
                except (struct.error, OverflowError, ValueError):
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
    if vtype == 0: return struct.unpack('<B', f.read(1))[0]
    elif vtype == 1: return struct.unpack('<b', f.read(1))[0]
    elif vtype == 2: return struct.unpack('<H', f.read(2))[0]
    elif vtype == 3: return struct.unpack('<h', f.read(2))[0]
    elif vtype == 4: return struct.unpack('<I', f.read(4))[0]
    elif vtype == 5: return struct.unpack('<i', f.read(4))[0]
    elif vtype == 6: return struct.unpack('<f', f.read(4))[0]
    elif vtype == 7: return struct.unpack('<?', f.read(1))[0]
    elif vtype == 8:
        slen = struct.unpack('<Q', f.read(8))[0]
        return f.read(slen).decode('utf-8', errors='replace')
    elif vtype == 9:
        elem_type = struct.unpack('<I', f.read(4))[0]
        count = struct.unpack('<Q', f.read(8))[0]
        for _ in range(count):
            _read_kv_value(f, elem_type)
        return None
    elif vtype == 10: return struct.unpack('<Q', f.read(8))[0]
    elif vtype == 11: return struct.unpack('<q', f.read(8))[0]
    elif vtype == 12: return struct.unpack('<d', f.read(8))[0]
    else: return None


class Handler(http.server.BaseHTTPRequestHandler):
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
            model_name = current_model_name
            hdr = parse_gguf_header(current_model) if current_model else {}
            self._send_json({
                'status': 'online' if current_model else 'ready',
                'model_loaded': current_model is not None,
                'model': model_name,
                'arch': hdr.get('arch'),
                'n_ctx': hdr.get('n_ctx', 0),
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
                    'name': name, 'size': size,
                    'arch': hdr['arch'], 'n_layers': hdr['n_layers'],
                    'n_ctx': hdr['n_ctx'], 'n_experts': hdr['n_experts'],
                    'loaded': (p == current_model),
                })
            self._send_json(models)
        elif self.path == '/api/telemetry':
            self._send_json({**telemetry, 'model': current_model_name, 'rss_mb': get_repl_rss()})
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
            stop_repl()
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

            # Build chat-formatted prompt
            if truncated_hist:
                full_prompt = ''
                for i, msg in enumerate(truncated_hist):
                    role = 'User' if i % 2 == 0 else 'Assistant'
                    full_prompt += f'{role}: {msg}\n'
                full_prompt += f'User: {prompt}\nAssistant: '
            else:
                full_prompt = prompt

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
                log.error(f'generate exception: {e}')
                self._send_json({'error': str(e)}, 500)
        else:
            self._send_json({'error': 'not found'}, 404)

    def log_message(self, format, *args):
        pass


def main():
    log.info(f'Lamio server on http://{HOST}:{PORT}')
    log.info(f'Models: {MODELS_DIR}')
    log.info(f'Binary: {LAMIO_BIN}')
    if not os.path.isfile(LAMIO_BIN):
        log.error(f'Binary not found: {LAMIO_BIN}')
        sys.exit(1)
    srv = http.server.ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        stop_repl()
        log.info('shutting down')


if __name__ == '__main__':
    main()
