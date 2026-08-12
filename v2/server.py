#!/usr/bin/env python3
"""Lamio v2 HTTP server. Self-contained UI + API. Uses lamio binary for inference."""

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
import fcntl
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('lamio')

HOST = '0.0.0.0'
PORT = 5180
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(FILE_DIR, 'web')
MODELS_DIR = os.path.join(os.path.dirname(FILE_DIR), 'models')
LAMIO_BIN = os.path.join(FILE_DIR, 'build', 'src', 'lamio')
LD_PATH = os.path.join(os.path.dirname(FILE_DIR), 'build', 'bin')

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


def build_chatml_prompt(messages, model_name='', chat_template=''):
    """Build a prompt using the model's chat_template (Jinja2) if available."""
    if chat_template:
        try:
            from jinja2 import Template
            tmpl = Template(chat_template)
            prompt = tmpl.render(
                messages=messages,
                add_generation_prompt=True,
                enable_thinking=True,  # enable thinking so model generates reasoning before answer
            )
            return prompt
        except Exception as e:
            log.warning(f'jinja render failed: {e}, falling back to ChatML')

    # Fallback: ChatML
    has_assistant = any(m.get('role') == 'assistant' for m in messages)
    if not has_assistant:
        context = ''
        for msg in messages[:-1]:
            context += msg.get('content', '') + ' '
        prompt = context + messages[-1].get('content', '') if messages else ''
        return prompt
    prompt = ''
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        if role == 'system':
            prompt += f'<|im_start|>system\n{content}<|im_end|>\n'
        elif role == 'user':
            prompt += f'<|im_start|>user\n{content}<|im_end|>\n'
        elif role == 'assistant':
            prompt += f'<|im_start|>assistant\n{content}<|im_end|>\n'
    prompt += '<|im_start|>assistant\n'
    return prompt


def parse_gguf_header(path):
    if path in _hdr_cache:
        return _hdr_cache[path]
    info = {'arch': 'unknown', 'n_layers': 0, 'n_ctx': 0, 'n_experts': 0, 'chat_template': ''}
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
            info['chat_template'] = kvs.get('tokenizer.chat_template', '')
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
            hdr = parse_gguf_header(current_model) if current_model else {}
            self._send_json({
                'status': 'online' if current_model else 'ready',
                'model_loaded': current_model is not None,
                'model': current_model_name,
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

        elif self.path == '/v1/chat/completions':
            self._handle_chat_completions(body)

        elif self.path == '/api/generate':
            self._handle_chat_completions(body)

        else:
            self._send_json({'error': 'not found'}, 404)

    def _handle_chat_completions(self, body):
        """Handle chat completion with ChatML formatting and SSE streaming."""
        if not current_model:
            self._send_json({'error': 'no model loaded'}, 400)
            return

        messages = body.get('messages', [])
        temperature = body.get('temperature', 0.7)
        top_k = body.get('top_k', 40)
        top_p = body.get('top_p', 0.9)
        repeat_penalty = body.get('repeat_penalty', body.get('repeat_pen', 1.1))
        n_predict = body.get('max_tokens', body.get('n_predict', 128))
        seed = body.get('seed', -1)
        n_ctx = body.get('n_ctx', 2048)
        stream = body.get('stream', True)
        stop_strings = body.get('stop', ['<|im_end|>', '</s>', '<|endoftext|>'])

        # Build ChatML prompt using model's chat_template (Jinja2)
        hdr = parse_gguf_header(current_model) if current_model else {}
        chat_template = hdr.get('chat_template', '')
        full_prompt = build_chatml_prompt(messages, current_model_name or '', chat_template)
        truncated = False

        # Truncate if too long (rough: 4 chars ~= 1 token)
        max_chars = n_ctx * 4
        if len(full_prompt) > max_chars:
            # Keep system message + last N messages
            system_msgs = [m for m in messages if m.get('role') == 'system']
            non_system = [m for m in messages if m.get('role') != 'system']
            keep = []
            char_budget = max_chars - sum(len(str(m.get('content', ''))) for m in system_msgs) - 200
            for msg in reversed(non_system):
                msg_len = len(str(msg.get('content', '')))
                if char_budget - msg_len < 0:
                    break
                keep.insert(0, msg)
                char_budget -= msg_len
            truncated = True
            full_prompt = build_chatml_prompt(system_msgs + keep)

        args = [LAMIO_BIN, current_model, '--generate', '--prompt', full_prompt,
                '--n-gen', str(n_predict), '--temp', str(temperature),
                '--top-k', str(top_k), '--top-p', str(top_p),
                '--repeat-penalty', str(repeat_penalty),
                '--auto-stop']
        if seed and seed > 0:
            args.extend(['--seed', str(seed)])

        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = LD_PATH

        log.info(f'chat: messages={len(messages)} prompt_len={len(full_prompt)} '
                 f'n_predict={n_predict} stream={stream}')

        if not stream:
            # Non-streaming: wait for full output
            t0 = time.time()
            try:
                result = subprocess.run(args, capture_output=True, text=True,
                                        timeout=600, env=env)
                elapsed = time.time() - t0
                if result.returncode != 0:
                    self._send_json({'error': f'lamio exit {result.returncode}',
                                     'stderr': result.stderr[-500:]}, 500)
                    return

                text = ''
                # Collect piece: lines (only generated tokens, not prompt)
                pieces = []
                for line in result.stdout.splitlines():
                    if line.startswith('piece:'):
                        pieces.append(line[6:])
                    elif line.startswith('decoded: '):
                        text = line[9:]
                # Use pieces (generated only) if available, else decoded
                if pieces:
                    text = ''.join(pieces)
                # Strip stop strings from the end
                for ss in stop_strings:
                    if ss in text:
                        text = text[:text.index(ss)]
                # Strip <|im_end|> (keep <|im_start|> for now)
                text = text.replace('<|im_end|>', '')
                # Extract content after think block (skip reasoning)
                THINK_END = chr(60) + chr(47) + 'think' + chr(62)
                if THINK_END in text:
                    text = text.split(THINK_END)[-1]
                    # Strip leading junk (residual think remnants, newlines)
                    text = text.lstrip(' \n\r\t')
                    # Also strip any non-printable leading chars
                    while text and ord(text[0]) < 32:
                        text = text[1:]
                # Now strip <|im_start|> and any remaining special tokens
                text = text.replace('<|im_start|>', '')
                text = text.strip()

                gen_lines = [l for l in result.stdout.splitlines()
                             if l.startswith('[') and 'token=' in l]
                n_tok = len(gen_lines)
                tps = n_tok / elapsed if elapsed > 0 else 0
                telemetry['total_tokens'] += n_tok
                telemetry['last_tps'] = round(tps, 2)
                telemetry['last_elapsed'] = round(elapsed, 2)

                self._send_json({
                    'choices': [{'message': {'role': 'assistant', 'content': text}}],
                    'usage': {'total_tokens': n_tok},
                })
            except subprocess.TimeoutExpired:
                self._send_json({'error': 'generation timed out'}, 504)
            return

        # SSE streaming
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.end_headers()

        t0 = time.time()
        try:
            proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, env=env, bufsize=0)
            stdout_fd = proc.stdout.fileno()
            stderr_fd = proc.stderr.fileno()
            fl = fcntl.fcntl(stdout_fd, fcntl.F_GETFL)
            fcntl.fcntl(stdout_fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

            buf = b''
            n_tok = 0
            full_text = ''

            while True:
                # Check stderr for errors
                ready_err, _, _ = select.select([stderr_fd], [], [], 0.1)
                if ready_err:
                    try:
                        err_data = os.read(stderr_fd, 4096).decode('utf-8', errors='replace')
                        if 'assert' in err_data.lower() or 'failed' in err_data.lower():
                            log.error(f'lamio stderr: {err_data}')
                            self._sse_send({'error': err_data})
                            break
                    except BlockingIOError:
                        pass

                # Read stdout
                try:
                    chunk = os.read(stdout_fd, 4096)
                    if chunk:
                        buf += chunk
                        while b'\n' in buf:
                            line, buf = buf.split(b'\n', 1)
                            line_str = line.decode('utf-8', errors='replace').strip()
                            # Parse token lines
                            m = re.match(r'\[\d+\] token=(\d+) logit=([\d.\-]+)', line_str)
                            if m:
                                token_id = int(m.group(1))
                                n_tok += 1
                            elif line_str.startswith('piece:'):
                                # Decoded token piece
                                piece = line_str[6:]
                                # Check for stop strings
                                full_text += piece
                                # Check if any stop string is in the text
                                stop_found = False
                                for ss in stop_strings:
                                    if ss in full_text:
                                        # Truncate at stop string
                                        full_text = full_text[:full_text.index(ss)]
                                        stop_found = True
                                        break
                                if stop_found:
                                    self._sse_send({
                                        'choices': [{'delta': {'content': piece}, 'finish_reason': 'stop'}]
                                    })
                                    break
                                if not stop_found:
                                    self._sse_send({
                                        'choices': [{'delta': {'content': piece}}]
                                    })
                except BlockingIOError:
                    pass

                # Check if process finished
                if proc.poll() is not None:
                    # Read remaining stdout
                    try:
                        remaining = os.read(stdout_fd, 65536)
                        if remaining:
                            buf += remaining
                    except BlockingIOError:
                        pass
                    # Process any remaining lines
                    while b'\n' in buf:
                        line, buf = buf.split(b'\n', 1)
                        line_str = line.decode('utf-8', errors='replace').strip()
                        if line_str.startswith('decoded: '):
                            full_text = line_str[9:]
                            break
                    break

                if time.time() - t0 > 600:
                    log.error('streaming timed out')
                    self._sse_send({'error': 'timed out'})
                    proc.kill()
                    break

            fcntl.fcntl(stdout_fd, fcntl.F_SETFL, fl)

            elapsed = time.time() - t0
            tps = n_tok / elapsed if elapsed > 0 else 0
            telemetry['total_tokens'] += n_tok
            telemetry['last_tps'] = round(tps, 2)
            telemetry['last_elapsed'] = round(elapsed, 2)

            # Send final message with full text
            self._sse_send({
                'choices': [{'delta': {'content': full_text}, 'finish_reason': 'stop'}],
                'usage': {'total_tokens': n_tok},
            })
            # Send [DONE]
            try:
                self.wfile.write(b'data: [DONE]\n\n')
                self.wfile.flush()
            except BrokenPipeError:
                pass

            log.info(f'chat done: {n_tok} tokens in {elapsed:.1f}s ({tps:.1f} t/s)')

        except Exception as e:
            log.error(f'chat exception: {e}')
            try:
                self._sse_send({'error': str(e)})
            except:
                pass

    def _sse_send(self, data):
        """Send a Server-Sent Event."""
        line = f'data: {json.dumps(data)}\n\n'
        try:
            self.wfile.write(line.encode())
            self.wfile.flush()
        except BrokenPipeError:
            pass

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
        log.info('shutting down')


if __name__ == '__main__':
    main()
