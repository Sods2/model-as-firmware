"""Live bridge: run MNIST test images through a target and stream what happens to a browser.

Serves the live UI (ui/live.html) and a Server-Sent Events stream on http://127.0.0.1:8765.
Nothing to install beyond what run.py already needs (numpy, torch/torchvision for MNIST, pyserial for pico).

Usage (Git Bash, from the repo root), then open http://127.0.0.1:8765:
  PY=train/.venv/Scripts/python.exe
  $PY harness/bridge.py --target golden                     # no hardware
  $PY harness/bridge.py --target host --exe firmware/baselines/ref_c/build-host/ref_host_tern80.exe
  $PY harness/bridge.py --target pico --port COM3           # flash a *_prof build to see per-stage cycles

Endpoints:
  GET  /                  the UI
  GET  /events            SSE stream of JSON events: info, status, result, stages, summary
  GET  /activations?idx=N every stage's output for image N from the golden model (bit-exact with the board),
                          plus the input, the label and what the target returned for it
  GET  /weights           FC1 ternary weights and their counts
  GET  /history           every result so far, without images
  GET  /results           rows of results/phase0.csv, for the latency ruler
  POST /control?cmd=pause|resume|step|restart  and/or  ?rate=<images per second, 0 = as fast as the target goes>
"""
import argparse
import csv
import json
import os
import re
import queue
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden import Golden, GoldenS8  # noqa: E402
from run import CONFIGS, ROOT, load_mnist, parse_r, pytorch_preds  # noqa: E402

UI = ROOT / 'ui' / 'live.html'
STAGES = ['conv1', 'dw1', 'pool1', 'dw2', 'pool2', 'fc1', 'fc2', 'fc3']
STICKY = ('info', 'stages', 'summary', 'status')  # re-sent to a browser that connects mid-run


class TargetLost(Exception):
    pass


# ---- targets: open() returns the board's INFO fields, infer(img) returns (pred, cycles, logits, stages or None)

class GoldenTarget:
    def __init__(self, golden):
        self.golden = golden

    def open(self):
        return {'build': 'golden', 'model': self.golden.name, 'sys_hz': '0'}

    def infer(self, img):
        logits = self.golden.logits(img[None])[0]
        return int(np.argmax(logits)), None, [int(v) for v in logits], None

    def close(self):
        pass


class HostTarget:
    """The C reference built for the PC. It reads all images from a file, so this streams its stdout."""

    def __init__(self, exe, img8):
        self.exe, self.img8 = Path(exe), img8

    def open(self):
        self.tmp = tempfile.TemporaryDirectory()
        p = Path(self.tmp.name) / 'images.bin'
        p.write_bytes(self.img8.tobytes())
        self.proc = subprocess.Popen([str(self.exe.resolve()), str(p)], stdout=subprocess.PIPE, text=True)
        return {'build': self.exe.stem, 'sys_hz': '0'}

    def infer(self, img):
        line = self.proc.stdout.readline()
        if not line:
            raise TargetLost('host executable stopped early')
        pred, _, logits = parse_r(line.strip())
        return pred, None, logits, None

    def close(self):
        if getattr(self, 'proc', None):
            self.proc.kill()
            self.proc.wait()
            self.tmp.cleanup()
            self.proc = None


# Anything that drives the board. While one of these runs, the bridge must not open the serial port: on
# Windows only one process can hold it, so grabbing it between two measurement runs (while measure.sh
# flashes the next build) would make the next run.py fail to open it.
BOARD_USERS = re.compile(r'harness[/\\](run\.py|measure\.sh|phase0_matrix\.sh)|picotool|openocd')


def board_users():
    """Command lines of other processes that drive the board, or None if the process list can't be read."""
    try:
        if os.name == 'nt':
            cmd = ['powershell', '-NoProfile', '-Command',
                   'Get-CimInstance Win32_Process | ForEach-Object { "$($_.ProcessId)`t$($_.CommandLine)" }']
            sep = '\t'
        else:
            cmd, sep = ['ps', '-eo', 'pid=,args='], None
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    if not out.strip():
        return None
    users = []
    for line in out.splitlines():
        parts = line.strip().split(sep, 1)
        m = BOARD_USERS.search(parts[1]) if len(parts) == 2 and parts[0] != str(os.getpid()) else None
        if m:
            what = parts[1][m.start():m.start() + 120].split('"')[0].strip()
            if what not in users:
                users.append(what)
    return users


def ensure_board_free(force):
    if force:
        return
    users = board_users()
    if users is None:
        raise TargetLost('could not read the process list to check that no test is using the board; '
                         'pass --force if you are sure it is free')
    if users:
        raise TargetLost('the board is in use, so the bridge will not open the port (pass --force to override):\n  '
                         + '\n  '.join(users))


class PicoTarget:
    """The board over USB serial, same protocol as run.py. Asks for per-stage cycles after every image
    until the build shows it was compiled without stage profiling (all zeros).
    Refuses to open the port while a harness run, measurement script or picotool is running."""

    def __init__(self, port, force=False):
        self.port, self.s, self.want_stages, self.force = port, None, True, force

    def open(self):
        ensure_board_free(self.force)  # before touching the port at all
        import serial
        try:
            self.s = serial.Serial(self.port, 115200, timeout=5)
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
            line = self.s.readline().decode(errors='replace').strip()
        except serial.SerialException as e:
            raise TargetLost(str(e))
        if not line:
            raise TargetLost(f'no reply from the board on {self.port} within 5 s')
        return line

    def infer(self, img):
        import serial
        try:
            self.s.write(b'I' + img.tobytes())
        except serial.SerialException as e:
            raise TargetLost(str(e))
        pred, cycles, logits = parse_r(self._line())
        stages = None
        if self.want_stages:
            self.s.write(b'T')
            line = self._line()
            stages = {k: int(v) for k, v in (kv.split('=') for kv in line.split()[1:])}
            if not any(stages.values()):
                self.want_stages, stages = False, None
        return pred, cycles, logits, stages

    def close(self):
        if self.s:
            try:
                self.s.close()
            except Exception:
                pass
            self.s = None


# ---- fan-out of events to every connected browser

class Hub:
    def __init__(self):
        self.lock = threading.Lock()
        self.clients = set()
        self.sticky = {}
        self.last_result = None

    def publish(self, ev):
        data = json.dumps(ev, separators=(',', ':'))
        with self.lock:
            if ev['type'] in STICKY:
                self.sticky[ev['type']] = data
            if ev['type'] == 'result':
                self.last_result = data
            for q in self.clients:
                try:
                    q.put_nowait(data)
                except queue.Full:
                    pass  # a stalled browser misses events rather than stalling the run

    def subscribe(self):
        q = queue.Queue(maxsize=20000)
        with self.lock:
            for k in STICKY:
                if k in self.sticky:
                    q.put(self.sticky[k])
            if self.last_result:
                q.put(self.last_result)
            self.clients.add(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            self.clients.discard(q)


# ---- the run itself

class Runner(threading.Thread):
    def __init__(self, args, hub, target, golden, x_img8, labels, g_logits, t_pred, meta):
        super().__init__(daemon=True)
        self.args, self.hub, self.target, self.golden = args, hub, target, golden
        self.img8, self.labels, self.g_logits, self.t_pred, self.meta = x_img8, labels, g_logits, t_pred, meta
        self.cv = threading.Condition()
        self.paused, self.steps, self.restart_req = args.paused, 0, False
        self.rate = args.rate
        self.results = []
        self.recent = deque(maxlen=60)  # completion times, for the images-per-second figure

    # control, called from HTTP threads
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
            elif cmd == 'restart':
                self.restart_req, self.paused = True, False
            self.cv.notify_all()
        if cmd in ('pause', 'resume') or rate is not None:
            self.status('paused' if self.paused else 'streaming')
        return {'paused': self.paused, 'rate': self.rate}

    def status(self, state, reason='', idx=None):
        self.hub.publish({'type': 'status', 'state': state, 'reason': reason, 'idx': idx, 'rate': self.rate})

    def _wait_turn(self):
        with self.cv:
            while self.paused and self.steps == 0 and not self.restart_req:
                self.cv.wait()
            if self.steps:
                self.steps -= 1
            return not self.restart_req

    def summary(self, final=False):
        r = self.results
        n = len(r)
        ev = {'type': 'summary', 'n': n, 'n_total': len(self.img8), 'final': final,
              'elapsed_s': round(time.perf_counter() - self.t0, 2)}
        if n:
            ok = sum(x['golden_ok'] for x in r)
            correct = sum(x['pred'] == x['label'] for x in r)
            ev.update(accuracy_pct=round(correct / n * 100, 2), wrong=n - correct,
                      golden_match_pct=round(ok / n * 100, 2), mismatches=n - ok,
                      images_per_s=self._recent_rate())
            if self.t_pred is not None:
                ev['pytorch_agree_pct'] = round(sum(x['pred'] == x['torch_pred'] for x in r) / n * 100, 2)
            cyc = [x['cycles'] for x in r if x['cycles'] is not None]
            if cyc:
                ev.update(cycles_mean=int(np.mean(cyc)), cycles_min=int(min(cyc)), cycles_max=int(max(cyc)))
                hz = int(self.meta.get('sys_hz', 0))
                if hz:
                    ev['latency_ms_mean'] = round(ev['cycles_mean'] / hz * 1e3, 3)
                    ev['latency_ms_min'] = round(ev['cycles_min'] / hz * 1e3, 3)
                    ev['latency_ms_max'] = round(ev['cycles_max'] / hz * 1e3, 3)
        self.hub.publish(ev)

    def _recent_rate(self):
        t = list(self.recent)
        if len(t) < 2 or time.perf_counter() - t[-1] > 2:
            return 0.0
        return round((len(t) - 1) / max(t[-1] - t[0], 1e-6), 1)

    def run(self):
        while True:
            self._one_pass()
            with self.cv:  # wait here after the run ends, until a restart
                while not self.restart_req:
                    self.cv.wait()
                self.restart_req = False

    def _one_pass(self):
        self.results = []
        self.recent.clear()
        self.t0 = time.perf_counter()
        self.hub.publish({'type': 'reset'})
        self.status('connecting')
        try:
            self.meta.update(self.target.open())
        except TargetLost as e:
            self.status('disconnected', str(e))
            return
        info = {'type': 'info', 'target': self.args.target, 'model': self.args.model,
                'build': self.meta.get('build', ''), 'sys_hz': int(self.meta.get('sys_hz', 0)),
                'port': self.args.port or '', 'n_images': len(self.img8),
                'golden': 'ref' if isinstance(self.golden, Golden) else 's8', 'stage_names': STAGES,
                'rate': self.rate}
        self.hub.publish(info)
        self.status('paused' if self.paused else 'streaming')
        last_summary, next_t = 0.0, time.perf_counter()
        try:
            for i, img in enumerate(self.img8):
                if not self._wait_turn():
                    break
                if self.rate > 0:
                    delay = next_t - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    next_t = max(next_t, time.perf_counter() - 0.5) + 1 / self.rate
                pred, cycles, logits, stages = self.target.infer(img)
                ok = logits == [int(v) for v in self.g_logits[i]]
                res = {'type': 'result', 'idx': i, 'label': int(self.labels[i]), 'pred': pred, 'cycles': cycles,
                       'logits': logits, 'golden_ok': ok,
                       'torch_pred': int(self.t_pred[i]) if self.t_pred is not None else None}
                self.results.append(res)
                self.recent.append(time.perf_counter())
                ev = dict(res, img=[int(v) for v in img])
                if not ok:
                    ev['golden_logits'] = [int(v) for v in self.g_logits[i]]
                self.hub.publish(ev)
                if stages:
                    self.hub.publish({'type': 'stages', 'idx': i, 'cycles': stages})
                if not ok and self.args.stop_on_mismatch:
                    with self.cv:
                        self.paused = True
                    self.status('paused', f'board logits differ from the golden model on image {i}', i)
                now = time.perf_counter()
                if now - last_summary > 0.25:
                    self.summary()
                    last_summary = now
            else:
                self.summary(final=True)
                self.status('done', f'{len(self.results)} images')
        except TargetLost as e:
            self.summary()
            self.status('disconnected', str(e), len(self.results))
        finally:
            self.target.close()

    # data for the HTTP side
    def activations(self, idx):
        idx = max(0, min(idx, len(self.img8) - 1))
        tr = self.golden.trace(self.img8[idx])
        res = self.results[idx] if idx < len(self.results) else None
        return {'idx': idx, 'label': int(self.labels[idx]), 'img': self.img8[idx].astype(int).tolist(),
                'golden_logits': [int(v) for v in self.g_logits[idx]], 'result': res,
                'layers': {k: {'shape': list(v.shape), 'data': v.astype(int).ravel().tolist()} for k, v in tr.items()}}

    def weights(self):
        w = self.golden.fc1
        return {'layer': 'fc1', 'shape': list(w.shape),
                'rows': [''.join('nzp'[int(v) + 1] for v in row) for row in w],
                'counts': {'neg': int((w == -1).sum()), 'zero': int((w == 0).sum()), 'pos': int((w == 1).sum())}}


def phase0_rows():
    p = ROOT / 'results' / 'phase0.csv'
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
            q = parse_qs(u.query)
            try:
                if u.path in ('/', '/index.html'):
                    self._send(UI.read_bytes(), 'text/html; charset=utf-8')
                elif u.path == '/events':
                    self._events()
                elif u.path == '/activations':
                    self._json(runner.activations(int(q.get('idx', ['0'])[0])))
                elif u.path == '/weights':
                    self._json(runner.weights())
                elif u.path == '/history':
                    self._json(list(runner.results))
                elif u.path == '/results':
                    self._json(phase0_rows())
                else:
                    self._json({'error': 'not found'}, 404)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def do_POST(self):
            q = parse_qs(urlparse(self.path).query)
            rate = float(q['rate'][0]) if 'rate' in q else None
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
                        # batch whatever else is queued into one write
                        while not q.empty():
                            self.wfile.write(f'data: {q.get_nowait()}\n\n'.encode())
                    except queue.Empty:
                        self.wfile.write(b': ping\n\n')
                    self.wfile.flush()
            finally:
                hub.unsubscribe(q)

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model', choices=sorted(CONFIGS),
                    help='model the target runs (default: what the board reports, else tern80)')
    ap.add_argument('--target', required=True, choices=['golden', 'host', 'pico'])
    ap.add_argument('--golden', choices=['ref', 's8'],
                    help='golden model (default: s8 when the board reports a cmsis_nn build, else ref)')
    ap.add_argument('--exe', help='host executable (target host)')
    ap.add_argument('--port', help='serial port (target pico)')
    ap.add_argument('--n', type=int, default=10000, help='number of test images')
    ap.add_argument('--rate', type=float, default=None,
                    help='images per second (default: 20 for golden/host, unthrottled for pico); 0 = unthrottled')
    ap.add_argument('--paused', action='store_true', help='start paused')
    ap.add_argument('--no-stop-on-mismatch', dest='stop_on_mismatch', action='store_false',
                    help='keep going when the board disagrees with the golden model')
    ap.add_argument('--no-pytorch', dest='pytorch', action='store_false', help='skip the PyTorch agreement check')
    ap.add_argument('--force', action='store_true',
                    help='open the board even if a harness run, measure script or picotool seems to be running')
    ap.add_argument('--http-port', type=int, default=8765)
    args = ap.parse_args()
    if args.target == 'host' and not args.exe:
        ap.error('--target host needs --exe')
    if args.target == 'pico' and not args.port:
        ap.error('--target pico needs --port')
    if args.rate is None:
        args.rate = 0.0 if args.target == 'pico' else 20.0

    meta = {}
    if args.target == 'pico':  # ask the board what it runs, to pick the matching model and golden model
        probe = PicoTarget(args.port, args.force)
        try:
            meta = probe.open()
        except TargetLost as e:
            sys.exit(f'board: {e}')
        finally:
            probe.close()
        print(f'board: build={meta.get("build")} model={meta.get("model")} sys_hz={meta.get("sys_hz")}')
        board_model = meta.get('model', '').split(':')[-1].removesuffix('.h')  # 'tern80z90.h', 'mafc:tern80'
        if args.model is None and board_model in CONFIGS:
            args.model = board_model
        elif args.model and board_model and board_model != args.model:
            sys.exit(f'the board runs {board_model} but --model is {args.model}: flash the right build or drop --model')
    if args.target == 'host' and args.model is None:  # ref_host_<model>.exe
        stem_model = Path(args.exe).stem.rsplit('_', 1)[-1]
        args.model = stem_model if stem_model in CONFIGS else None
    args.model = args.model or 'tern80'

    print(f'loading {args.n} MNIST test images ...')
    x, labels, img8 = load_mnist(args.n)

    if args.target == 'golden':
        target = None
    elif args.target == 'host':
        target = HostTarget(args.exe, img8)
    else:
        target = PicoTarget(args.port, args.force)
    if args.golden is None:
        args.golden = 's8' if meta.get('build', '').startswith('cmsis_nn') else 'ref'
    if args.golden == 'ref':
        golden = Golden(ROOT / 'train' / 'export' / f'{args.model}.h')
    else:
        golden = GoldenS8(ROOT / 'firmware' / 'baselines' / 'cmsis_nn' / f'model_s8_{args.model}.h')
    if target is None:
        target = GoldenTarget(golden)

    print('golden model logits ...')
    g_logits = golden.logits(img8)
    t_pred = None
    if args.pytorch:
        print('PyTorch predictions ...')
        t_pred = pytorch_preds(args.model, x)

    hub = Hub()
    runner = Runner(args, hub, target, golden, img8, labels, g_logits, t_pred, meta)
    server = ThreadingHTTPServer(('127.0.0.1', args.http_port), make_handler(hub, runner))
    server.daemon_threads = True
    runner.start()
    print(f'live UI on http://127.0.0.1:{args.http_port}  (Ctrl+C to stop)')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        target.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
