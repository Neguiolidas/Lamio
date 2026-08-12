#!/usr/bin/env python3
import http.server, json, os, subprocess, signal, sys, threading, time, glob, socket

HOST = '0.0.0.0'
PORT = 5180
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
LAMIO_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build', 'src', 'lamio')
LD_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'build', 'bin')

current_model = None
model_proc = None
telemetry = { 'total_tokens': 0, 'last_tps': 0, 'rss_mb': 0 }


def get_rss(pid):
    try:
        with open(f'/proc/{pid}/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) // 1024
    except:
        pass
    return 0


def model_info(path):
    name = os.path.basename(path)
    size = os.path.getsize(path)
    # Quick GGUF header read for arch
    arch = 'unknown'
    params = ''
    try:
        with open(path, 'rb') as f:
            magic = f.read(4)
            if magic != b'GGUF':
                return {'name': name, 'size': size, 'arch': arch, 'params': params, 'path': path}
            import struct
            version = struct.unpack('<I', f.read(4))[0]
            n_kv = struct.unpack('<Q', f.read(8))[0]
            n_t = struct.unpack('<Q', f.read(8))[0]
            for _ in range(n_kv):
                klen = struct.unpack('<Q', f.read(8))[0]
                key = f.read(klen).decode('utf-8', errors='replace')
                vtype = struct.unpack('<I', f.read(4))[0]
                if vtype == 0:
                    val = struct.unpack('<B', f.read(1))[0]
                elif vtype == 1:
                    val = struct.unpack('<b', f.read(1))[0]
                elif vtype == 2:
                    val = struct.unpack('<H', f.read(2))[0]
                elif vtype == 3:
                    val = struct.unpack('<i', f.read(4))[0]
                elif vtype == 4:
                    val = struct.unpack('<f', f.read(4))[0]
                elif vtype == 5:
                    val = struct.unpack('<b', f.read(1))[0]
                elif vtype == 8:
                    slen = struct.unpack('<Q', f.read(8))[0]
                    val = f.read(slen).decode('utf-8', errors='replace')
                else:
                    val = None
                    break
                if key == 'general.architecture':
                    arch = val
                elif key == f'{val}.block_count':
                    pass
    except:
        pass
    return {'name': name, 'size': size, 'arch': arch, 'params': params, 'path': path, 'loaded': path == current_model}


class Handler(http.server.BaseHTTPRequestHandler):
    def _json(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path, ctype):
        try:
            with open(path, 'rb') as f:
                body = f.read()
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', len(body))
            self.end_headers()
            self.wfile.write(body)
        except FileNotFoundError:
            self._json({'error': 'not found'}, 404)

    def do_GET(self):
        if self.path == '/api/health':
            self._json({'status': serverState, 'model_loaded': current_model is not None, 'model': current_model})
        elif self.path == '/api/models':
            models = []
            for p in sorted(glob.glob(os.path.join(MODELS_DIR, '*.gguf'))):
                info = model_info(p)
                info['loaded'] = (p == current_model)
                models.append(info)
            self._json(models)
        elif self.path == '/api/telemetry':
            rss = get_rss(model_proc.pid) if model_proc else 0
            self._json({**telemetry, 'model': current_model, 'rss_mb': rss})
        elif self.path == '/api/memory':
            self._json({'layers': []})
        elif self.path == '/' or self.path == '/index.html':
            self._serve_file(os.path.join(WEB_DIR, 'index.html'), 'text/html')
        elif self.path == '/style.css':
            self._serve_file(os.path.join(WEB_DIR, 'style.css'), 'text/css')
        elif self.path == '/app.js':
            self._serve_file(os.path.join(WEB_DIR, 'app.js'), 'application/javascript')
        else:
            self._json({'error': 'not found'}, 404)

    def do_POST(self):
        global current_model, model_proc, telemetry
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length)) if length > 0 else {}

        if self.path == '/api/models/load':
            model_name = body.get('model', '')
            path = os.path.join(MODELS_DIR, model_name)
            if not os.path.isfile(path):
                self._json({'error': 'model not found'}, 404)
                return
            current_model = path
            telemetry = {'total_tokens': 0, 'last_tps': 0, 'rss_mb': 0}
            self._json({'ok': True})

        elif self.path == '/api/models/unload':
            current_model = None
            self._json({'ok': True})

        elif self.path == '/api/generate':
            if not current_model:
                self._json({'error': 'no model loaded'}, 400)
                return

            prompt = body.get('prompt', '')
            history = body.get('history', [])
            temp = body.get('temperature', 0.7)
            top_k = body.get('top_k', 40)
            top_p = body.get('top_p', 0.9)
            repeat_pen = body.get('repeat_penalty', 1.1)
            n_predict = body.get('n_predict', 128)
            seed = body.get('seed', -1)

            full_prompt = ''
            for h in history:
                full_prompt += h + ' '
            full_prompt += prompt

            args = [LAMIO_BIN, current_model, '--generate', '--prompt', full_prompt,
                    '--n-gen', str(n_predict), '--temp', str(temp),
                    '--top-k', str(top_k), '--top-p', str(top_p),
                    '--repeat-penalty', str(repeat_pen)]
            if seed and seed > 0:
                args.extend(['--seed', str(seed)])

            env = os.environ.copy()
            env['LD_LIBRARY_PATH'] = LD_PATH

            t0 = time.time()
            try:
                result = subprocess.run(args, capture_output=True, text=True, timeout=300, env=env)
                elapsed = time.time() - t0

                output = result.stdout
                text = ''
                for line in output.splitlines():
                    if line.startswith('decoded: '):
                        text = line[9:]
                        break

                n_tok = len(text.split())
                tps = n_tok / elapsed if elapsed > 0 else 0
                telemetry['total_tokens'] += n_tok
                telemetry['last_tps'] = tps

                self._json({'text': text, 'tokens': n_tok, 'tps': tps, 'elapsed': elapsed})
            except subprocess.TimeoutExpired:
                self._json({'error': 'generation timed out'}, 504)
            except Exception as e:
                self._json({'error': str(e)}, 500)
        else:
            self._json({'error': 'not found'}, 404)

    def log_message(self, fmt, *args):
        pass


serverState = 'connecting'


def main():
    srv = http.server.HTTPServer((HOST, PORT), Handler)
    srv.timeout = 1
    print(f'Lamio UI on http://{HOST}:{PORT}')
    print(f'Models: {MODELS_DIR}')
    print(f'Binary: {LAMIO_BIN}')
    print(f'LD: {LD_PATH}')

    while True:
        srv.handle_request()


if __name__ == '__main__':
    main()
