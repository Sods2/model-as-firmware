"""Live story mode of the bridge: generate with a MafLM build and stream every token to the browser (ui/story.html).

Started by harness/bridge.py, which switches to this mode by itself when the board reports a story build (its INFO line
has window=), or with --task story:

  python harness/bridge.py --task story --target golden --npz train/runs/lm_tree.npz   # no hardware
  python harness/bridge.py --target pico --port COM3        # /dev/ttyACM0 on Linux; model npz from the board's INFO

Each position is one byte. The board runs it ('S' command: argmax, cycles, logits checksum, XIP counters) and the golden
model (train/lm/intref_step.py, StepLM, the token-by-token integer reference) runs the same byte. A position counts as
checked only if the board's argmax and 64-bit logits checksum equal the golden model's. The top candidates, their
probabilities and the tree leaves shown on the page come from the golden model; the checksum is what makes them the
board's too. With temperature 0 the next byte is the board's argmax (greedy, as harness/lm_run.py measures). With a
temperature, the PC samples the next byte from the golden logits with a fixed seed; the board still runs every byte.

Endpoints:
  GET  /                   the UI (ui/story.html)
  GET  /events             SSE stream of JSON events: info, status, gen_start, token, summary
  GET  /history            every token of the current generation
  GET  /results            rows of results/phase3.csv, for the comparison panel
  POST /generate?prompt=..&n=..&temp=..&seed=..   starts a new generation (n = 0: until the KV cache window is full)
  POST /control?cmd=pause|resume|step|stop  and/or  ?rate=<tokens per second, 0 = as fast as it goes>
"""
from __future__ import annotations

import csv
import json
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'train' / 'lm'))
UI = ROOT / 'ui' / 'story.html'
STAGES = ['qkv', 'attn', 'wo', 'ffn1', 'ffn2', 'unembed', 'tables']
MAX_PROMPT = 64


class TargetLost(Exception):
    pass


def checksum(logits: np.ndarray) -> int:
    """The board's logits checksum: sum_i logits[i] * (2i + 1) mod 2^64 (pico_main.c, harness/lm_run.py)."""
    w = (2 * np.arange(256, dtype=np.uint64) + 1)
    return int((logits.astype(np.int64).view(np.uint64) * w).sum(dtype=np.uint64))


class Golden:
    """StepLM plus what the page shows: the tree leaf each layer picked, and probabilities from the integer logits."""

    def __init__(self, npz: Path):
        from intref import F, load_npz
        from intref_step import StepLM
        self.im = load_npz(npz)
        self.lm = StepLM(self.im)
        self.c = self.lm.c
        self.scale = float(self.im['logit_scale']) / 2 ** F
        self.leaves = []
        ffn = self.lm.ffn

        def spy(x, i):  # record each layer's leaf (None for the dense twin)
            y, n = ffn(x, i)
            self.leaves.append(n)
            return y, n
        self.lm.ffn = spy

    def reset(self):
        self.lm.reset()

    def step(self, tok: int):
        self.leaves = []
        logits = self.lm.step(tok)
        return logits, list(self.leaves)

    def probs(self, logits: np.ndarray, temp: float = 1.0) -> np.ndarray:
        z = logits.astype(np.float64) * self.scale / max(temp, 1e-6)
        z -= z.max()
        p = np.exp(z)
        return p / p.sum()


class GoldenTarget:
    """No hardware: the golden model plays the board (no cycle counter, no XIP counters). It shares the runner's
    Golden, so each position is computed once; .last hands the result to the runner's check."""

    def __init__(self, golden: Golden, model: str, window: int = 256):
        self.g, self.model, self.window, self.last = golden, model, window, None

    def open(self):
        return {'build': 'golden', 'model': self.model, 'window': str(self.window),
                'tree': str(int(self.g.c.ffn == 'tree')), 'sys_hz': '0'}

    def reset(self):
        self.g.reset()

    def step(self, tok):
        self.last = self.g.step(tok)
        return int(np.argmax(self.last[0])), None, checksum(self.last[0]), None, None

    def stage_cycles(self):
        return None

    def close(self):
        pass


class PicoTarget:
    """The board over USB serial, same protocol as harness/lm_run.py ('?', 'R', 'S', 'Q')."""

    def __init__(self, port: str, force: bool = False):
        self.port, self.force, self.s = port, force, None
        self.want_stages = True

    def open(self):
        from bridge import ensure_board_free, TargetLost as BridgeLost
        try:
            ensure_board_free(self.force)
        except BridgeLost as e:
            raise TargetLost(str(e))
        import serial
        try:
            self.s = serial.Serial(self.port, 115200, timeout=10)
            time.sleep(0.2)
            self.s.reset_input_buffer()
            self.s.write(b'?')
            info = self.s.readline().decode(errors='replace').strip()
        except serial.SerialException as e:
            raise TargetLost(str(e))
        if not info.startswith('INFO'):
            raise TargetLost(f'no INFO reply from the board on {self.port}: {info!r}')
        return dict(kv.split('=', 1) for kv in info.split()[1:])

    def _line(self):
        import serial
        try:
            ln = self.s.readline().decode(errors='replace').strip()
        except serial.SerialException as e:
            raise TargetLost(str(e))
        if not ln:
            raise TargetLost(f'no reply from the board on {self.port} within 10 s')
        return ln

    def _write(self, b: bytes):
        import serial
        try:
            self.s.write(b)
        except serial.SerialException as e:
            raise TargetLost(str(e))

    def reset(self):
        self._write(b'R')
        self._line()
        self.last_q = None

    def step(self, tok):
        self._write(b'S' + bytes([tok]))
        ln = self._line()
        try:
            _, am, cyc, cs, acc, hit = ln.split()
            return int(am), int(cyc), int(cs, 16), int(acc), int(hit)
        except ValueError:
            raise TargetLost(f'unexpected reply from the board: {ln[:60]!r}') from None

    def stage_cycles(self):
        """Cycles per stage for the last step (Q is cumulative since 'R'), or None on builds without LM_PROFILE."""
        if not self.want_stages:
            return None
        self._write(b'Q')
        ln = self._line()
        try:
            q = [int(v) for v in ln.split()[1:]]
        except ValueError:
            raise TargetLost(f'unexpected reply from the board: {ln[:60]!r}') from None
        if not any(q):
            self.want_stages = False
            return None
        prev = self.last_q or [0] * len(q)
        self.last_q = q
        return dict(zip(STAGES, (a - b for a, b in zip(q, prev))))

    def close(self):
        if self.s:
            try:
                self.s.close()
            except Exception:
                pass
            self.s = None


class Hub:
    STICKY = ('info', 'gen_start', 'status', 'summary')   # replay order for a browser that connects mid-run

    def __init__(self):
        self.lock = threading.Lock()
        self.clients = set()
        self.sticky = {}
        self.history = []   # token events of the current generation, replayed after gen_start

    def publish(self, ev):
        data = json.dumps(ev, separators=(',', ':'))
        with self.lock:
            if ev['type'] in self.STICKY:
                self.sticky[ev['type']] = data
            if ev['type'] == 'gen_start':
                self.history = []
            elif ev['type'] == 'token':
                self.history.append(data)
            for q in self.clients:
                try:
                    q.put_nowait(data)
                except queue.Full:
                    pass

    def subscribe(self):
        q = queue.Queue(maxsize=5000)
        with self.lock:
            for k in self.STICKY:
                if k in self.sticky:
                    q.put(self.sticky[k])
                if k == 'gen_start':
                    for d in self.history:
                        q.put(d)
            self.clients.add(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            self.clients.discard(q)


class Runner(threading.Thread):
    def __init__(self, args, hub, target, golden, meta):
        super().__init__(daemon=True)
        self.args, self.hub, self.target, self.golden, self.meta = args, hub, target, golden, meta
        self.cv = threading.Condition()
        self.paused, self.steps, self.stop_req = False, 0, False
        self.rate = args.rate
        self.request = None
        self.tokens = []
        self.connected = False

    # ---- control from HTTP threads
    def generate(self, prompt: bytes, n: int, temp: float, seed: int):
        with self.cv:
            self.request = (prompt, n, temp, seed)
            self.stop_req, self.paused, self.steps = True, False, 0
            self.cv.notify_all()
        return {'ok': True}

    def control(self, cmd=None, rate=None):
        with self.cv:
            if rate is not None:
                self.rate = max(0.0, rate)
            if cmd == 'pause':
                self.paused = True
            elif cmd == 'resume':
                self.paused = False
            elif cmd == 'step':
                self.paused, self.steps = True, self.steps + 1
            elif cmd == 'stop':
                self.stop_req, self.paused = True, False
            self.cv.notify_all()
        if cmd in ('pause', 'resume', 'step') or rate is not None:
            running = self.state in ('streaming', 'paused')
            self.status(('paused' if self.paused else 'streaming') if running else self.state)
        return {'paused': self.paused, 'rate': self.rate}

    def status(self, state, reason='', pos=None):
        self.state = state
        self.hub.publish({'type': 'status', 'state': state, 'reason': reason, 'pos': pos, 'rate': self.rate,
                          'paused': self.paused})

    # ---- the loop
    def run(self):
        self.state = 'connecting'
        self.status('connecting')
        try:
            self.meta.update(self.target.open())
            self.connected = True
        except TargetLost as e:
            self.status('disconnected', str(e))
            return
        self.window = int(self.meta.get('window', 256))
        self.hub.publish({'type': 'info', 'target': self.args.target, 'model': self.meta.get('model', ''),
                          'build': self.meta.get('build', ''), 'sys_hz': int(self.meta.get('sys_hz', 0)),
                          'window': self.window, 'tree': self.golden.c.ffn == 'tree', 'layers': self.golden.c.layers,
                          'depth': self.golden.c.depth, 'leaves': 2 ** self.golden.c.depth, 'leaf': self.golden.c.leaf,
                          'hidden': self.golden.c.hidden, 'port': self.args.port or '', 'rate': self.rate,
                          'npz': str(self.args.npz), 'stage_names': STAGES, 'home': getattr(self.args, 'home', None)})
        self.status('idle', 'type a prompt and press Generate')
        if self.args.prompt:
            self.generate(self.args.prompt.encode()[:MAX_PROMPT], 0, 0.0, 0)
        while True:
            with self.cv:
                while self.request is None:
                    self.cv.wait()
                req, self.request, self.stop_req = self.request, None, False
            try:
                if not self.connected:               # after an error: reconnect on the next Generate
                    self.target.close()
                    self.meta.update(self.target.open())
                    self.connected = True
                self._generate(*req)
            except TargetLost as e:
                self.connected = False
                self.status('disconnected', f'{e}. Press Generate to reconnect and try again.')
            except Exception as e:  # keep the page usable: report it, wait for the next Generate
                self.status('disconnected', f'{type(e).__name__}: {e}. Press Generate to try again.')

    def _wait_turn(self):
        with self.cv:
            while self.paused and self.steps == 0 and not self.stop_req:
                self.cv.wait()
            if self.steps:
                self.steps -= 1
            return not self.stop_req

    def _generate(self, prompt: bytes, n: int, temp: float, seed: int):
        hz = int(self.meta.get('sys_hz', 0))
        rng = np.random.default_rng(seed)
        limit = self.window - len(prompt) if n <= 0 else min(n, self.window - len(prompt))
        self.tokens = []
        self.hub.publish({'type': 'gen_start', 'prompt': list(prompt), 'n': limit, 'temp': temp, 'seed': seed,
                          'window': self.window})
        self.status('paused' if self.paused else 'streaming', pos=0)
        self.golden.reset()
        self.target.reset()
        feed = list(prompt)
        pos, gen, mismatches, last_summary, next_t = 0, 0, 0, 0.0, time.perf_counter()
        t0 = time.perf_counter()
        while pos < self.window:
            if not self._wait_turn():
                self.status('stopped', f'stopped at position {pos}', pos)
                break
            if self.rate > 0:
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                next_t = max(next_t, time.perf_counter() - 0.5) + 1 / self.rate
            tok = feed[pos] if pos < len(feed) else None
            if tok is None:
                break
            am, cyc, cs, acc, hit = self.target.step(tok)
            stages = self.target.stage_cycles()
            if isinstance(self.target, GoldenTarget):
                logits, leaves = self.target.last
            else:
                logits, leaves = self.golden.step(tok)
            ok = am == int(np.argmax(logits)) and cs == checksum(logits)
            mismatches += not ok
            p = self.golden.probs(logits)
            top = np.argsort(-p)[:5]
            if temp > 0:
                nxt = int(rng.choice(256, p=self.golden.probs(logits, temp)))
            else:
                nxt = am
            phase = 'prompt' if pos < len(prompt) else 'gen'
            ev = {'type': 'token', 'pos': pos, 'byte': tok, 'phase': phase, 'next': nxt, 'board_argmax': am,
                  'cycles': cyc, 'ms': round(cyc / hz * 1e3, 4) if (cyc is not None and hz) else None,
                  'golden_ok': ok, 'top': [[int(b), round(float(p[b]), 4)] for b in top],
                  'leaves': leaves if self.golden.c.ffn == 'tree' else None,
                  'flash_kb': round((acc - hit) * 8 / 1024, 2) if acc is not None else None, 'stages': stages}
            self.tokens.append(ev)
            self.hub.publish(ev)
            if not ok and self.args.stop_on_mismatch:
                with self.cv:
                    self.paused = True
                self.status('paused', f'board logits differ from the golden model at position {pos}', pos)
            pos += 1
            if pos >= len(feed) and gen < limit:  # the byte this position predicts becomes the next input
                feed.append(nxt)
                gen += 1
            now = time.perf_counter()
            if now - last_summary > 0.25:
                self.summary(t0, mismatches)
                last_summary = now
        self.summary(t0, mismatches, final=True)
        if self.state not in ('stopped', 'disconnected'):
            why = 'context window full' if pos >= self.window else f'{gen} bytes generated'
            self.status('done', why, pos)

    def summary(self, t0, mismatches, final=False):
        toks = self.tokens
        gen_ms = [t['ms'] for t in toks if t['phase'] == 'gen' and t['ms'] is not None]
        f24 = [t['ms'] for t in toks if t['phase'] == 'gen' and t['ms'] is not None and t['pos'] < 24]
        fl = [t['flash_kb'] for t in toks if t['phase'] == 'gen' and t['flash_kb'] is not None]
        ev = {'type': 'summary', 'final': final, 'n': len(toks), 'generated': sum(t['phase'] == 'gen' for t in toks),
              'checked': len(toks), 'mismatches': mismatches, 'elapsed_s': round(time.perf_counter() - t0, 2)}
        if gen_ms:
            ev.update(ms_mean=round(float(np.mean(gen_ms)), 3), tok_s=round(1e3 / float(np.mean(gen_ms)), 1),
                      ms_last=gen_ms[-1])
        if f24:
            ev.update(ms_first24=round(float(np.mean(f24)), 3), n_first24=len(f24))
        if fl:
            ev['flash_kb_mean'] = round(float(np.mean(fl)), 1)
        self.hub.publish(ev)


def phase3_rows():
    p = ROOT / 'results' / 'phase3.csv'
    if not p.exists():
        return []
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def make_handler(hub, runner):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype, code=200):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(json.dumps(obj, separators=(',', ':')).encode(), 'application/json', code)

        def do_GET(self):
            u = urlparse(self.path)
            try:
                if u.path in ('/', '/index.html', '/story'):
                    self._send(UI.read_bytes(), 'text/html; charset=utf-8')
                elif u.path == '/events':
                    self._events()
                elif u.path == '/history':
                    self._json(list(runner.tokens))
                elif u.path == '/results':
                    self._json(phase3_rows())
                else:
                    self._json({'error': 'not found'}, 404)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def do_POST(self):
            u = urlparse(self.path)
            q = parse_qs(u.query, keep_blank_values=True)
            if u.path == '/generate':
                prompt = q.get('prompt', [''])[0].encode('utf-8')[:MAX_PROMPT]
                if not prompt:
                    self._json({'error': 'empty prompt'}, 400)
                    return
                try:
                    n = int(q.get('n', ['0'])[0])
                    temp = float(q.get('temp', ['0'])[0])
                    seed = int(q.get('seed', ['0'])[0])
                except ValueError:
                    self._json({'error': 'bad number'}, 400)
                    return
                self._json(runner.generate(prompt, n, min(max(temp, 0.0), 2.0), seed))
            else:
                try:
                    rate = float(q['rate'][0]) if 'rate' in q else None
                except ValueError:
                    self._json({'error': 'bad rate'}, 400)
                    return
                self._json(runner.control(q.get('cmd', [None])[0], rate))

        def _events(self):
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            q = hub.subscribe()
            try:
                while True:
                    try:
                        data = q.get(timeout=15)
                        self.wfile.write(f'data: {data}\n\n'.encode())
                        while not q.empty():
                            self.wfile.write(f'data: {q.get_nowait()}\n\n'.encode())
                    except queue.Empty:
                        self.wfile.write(b': ping\n\n')
                    self.wfile.flush()
            finally:
                hub.unsubscribe(q)

    return Handler


def serve(args, meta):
    """Called by bridge.py. args: target, port, force, npz, rate, prompt, stop_on_mismatch, http_port."""
    npz = Path(args.npz)
    if not npz.exists():
        sys.exit(f'{npz} not found: train the model (docs/REPRODUCE.md, "MafLM"; train/lm/overnight.sh) or pass --npz')
    print(f'golden model {npz} ...')
    golden = Golden(npz)
    if args.target == 'golden':
        target = GoldenTarget(golden, npz.stem)
    elif args.target == 'pico':
        target = PicoTarget(args.port, args.force)
    else:
        sys.exit('story mode supports --target golden or pico')
    hub = Hub()
    runner = Runner(args, hub, target, golden, dict(meta))
    server = ThreadingHTTPServer(('127.0.0.1', args.http_port), make_handler(hub, runner))
    server.daemon_threads = True
    runner.start()
    print(f'story UI on http://127.0.0.1:{args.http_port}  (Ctrl+C to stop)')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        target.close()
    return 0
