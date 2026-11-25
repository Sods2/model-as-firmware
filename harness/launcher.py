"""Test launcher: one home page to choose what to run, then it starts the matching live view.

    python harness/launcher.py            # then open http://127.0.0.1:8765  (python = the venv's, docs/SETUP.md)
    python harness/launcher.py --port COM3 --http-port 8765 --view-port 8766     # /dev/ttyACM0 on Linux

Choices on the page (ui/launcher.html):
  test    MNIST digits (ui/live.html) or the tiny story model (ui/story.html)
  target  golden model (no hardware), the PC build of the C reference (MNIST only), or the Pico 2 board
  board   keep what is flashed, or flash one of the existing builds first (SWD through harness/flash.sh,
          picotool as the fallback). Builds are the build-*/ folders that hold a .uf2; nothing is compiled here.

What each choice needs (docs/SETUP.md, docs/UI-GUIDE.md): MNIST needs the MNIST test images in
train/BitNetMCU/data (the PyTorch agreement check also needs the BitNetMCU clone; without it the check is skipped);
the story model needs a trained train/runs/<model>.npz; the PC build needs ref_host_<model> built; flashing needs
bash (Git Bash on Windows), picotool or OpenOCD, and build folders. The page greys out what is missing and says why.

The launcher runs harness/bridge.py as a child process on --view-port, with --home pointing back here, so each live
page has a link to this one. Starting a new test stops the running one first (which frees the serial port). Only one
test runs at a time. The launcher itself only touches the serial port to read the board's INFO line, and not while a
test, a harness run, a measurement script or picotool/OpenOCD holds the board.

Endpoints:
  GET  /                the launcher page
  GET  /api/options     what can be run: models, PC builds, board builds, the board's current build
  GET  /api/status      the running test (or the job starting one) and the last lines of its log
  POST /api/start       JSON {test, target, model?, npz?, exe?, build?}; build "" = keep what is on the board
  POST /api/stop        stops the running test
  POST /api/probe       reads the board's INFO line again
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
UI = ROOT / 'ui' / 'launcher.html'
sys.path.insert(0, str(HERE))

DEFAULT_PORT = 'COM3' if os.name == 'nt' else '/dev/ttyACM0'
MNIST_GOLDEN = ['tern80', 'tern80z70', 'tern80z90', 'int8_80']     # models with a train/export/<model>.h golden model
UNSUPPORTED_MNIST = {'fff80': 'the tree MNIST model (idea B) is not supported by the live MNIST view yet'}
BUILD_GROUPS = [  # (test, folder, firmware name, group label)
    ('mnist', 'firmware/baselines/ref_c', 'ref_c', 'Plain-C reference'),
    ('mnist', 'firmware/baselines/cmsis_nn', 'cmsis_nn', 'CMSIS-NN baseline'),
    ('mnist', 'firmware/rp2350/mafc', 'mafc', 'mafc compiler'),
    ('story', 'firmware/rp2350/maflm', 'maflm', 'MafLM runtime'),
]


def git_bash() -> str | None:
    """bash for harness/flash.sh: Git Bash on Windows (WSL's bash.exe in System32 would not work), else PATH."""
    for c in (Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Git' / 'bin' / 'bash.exe',
              Path('C:/Program Files/Git/bin/bash.exe')):
        if c.exists():
            return str(c)
    w = shutil.which('bash')
    return w if w and 'system32' not in w.lower() and 'windowsapps' not in w.lower() else None


def describe(test: str, name: str) -> dict:
    """Tags read from a build folder name, e.g. mafc2_s-tern80-ram-code-i4code-prof or lm_tree-ram-dual."""
    parts = name.split('-')
    tags = []
    model = ''
    if test == 'story':
        model = parts[0]
        tags.append('tree' if 'tree' in model else 'dense')
        if model.startswith('check_'):
            tags.append('2k-step check model')
    elif parts[0] in MNIST_GOLDEN or parts[0] in UNSUPPORTED_MNIST:   # phase0_matrix.sh names: <model>-<mem>[-prof]
        model = parts[0]
        tags.append(model)
    else:                                                            # measure.sh names: <build>-<model>-<mem>[-...]
        model = parts[1] if len(parts) > 1 else ''
        b = parts[0]
        if b.startswith('mafc'):
            tags.append('two cores' if b.startswith('mafc2') else 'one core')
            if b.endswith('_s'):
                tags.append('SMLAD')
        elif b == 'cmsis_nn_s4':
            tags.append('fc1 as s4')
        tags.append(model)
        for p in parts[3:]:
            if p not in ('prof', 'base3', 'i4packed', 'cvdata', 'nostream', 'plain', 'i8plain', 'nofuse'):   # defaults
                tags.append(p)
    mem = 'SRAM' if 'ram' in parts else ('flash' if 'flash' in parts else '')
    if mem:
        tags.insert(0, mem)
    if test == 'story' and 'dual' in parts:
        tags.append('two cores')
    if 'prof' in parts:
        tags.append('profiling')
    return {'model': model, 'tags': tags}


def list_builds() -> list[dict]:
    out = []
    for test, folder, fw, group in BUILD_GROUPS:
        d = ROOT / folder
        if not d.exists():
            continue
        for b in sorted(d.glob('build-*'), key=lambda p: (p.name.startswith('build-check_'), p.name)):  # check models last
            uf2 = b / f'{fw}.uf2'
            if not uf2.exists():
                continue
            name = b.name[len('build-'):]
            info = describe(test, name)
            item = {'test': test, 'group': group, 'name': name, 'path': str(uf2.relative_to(ROOT)).replace('\\', '/'),
                    'mtime': int(uf2.stat().st_mtime), **info}
            if test == 'mnist' and info['model'] in UNSUPPORTED_MNIST:
                item['unsupported'] = UNSUPPORTED_MNIST[info['model']]
            out.append(item)
    return out


class Launcher:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.child = None
        self.log = deque(maxlen=200)
        self.state = {'state': 'idle', 'detail': '', 'test': None, 'url': None, 'since': time.time()}
        self.board = {'state': 'unknown', 'info': None, 'error': ''}
        self.job = None
        self.cancel = threading.Event()   # set by Stop: abandons a test that is still flashing or starting
        self.pump = None

    # ---- board INFO
    def probe(self):
        if self.child and self.child.poll() is None:
            self.board = dict(self.board, state='in use', error='a test is running and holds the port')
            return
        self.board = dict(self.board, state='checking')
        try:
            from bridge import board_users
            users = board_users()
            if users:
                self.board = {'state': 'busy', 'info': None, 'error': 'in use by: ' + '; '.join(users)}
                return
            import serial
            with serial.Serial(self.args.port, 115200, timeout=2) as s:
                time.sleep(0.2)
                s.reset_input_buffer()
                s.write(b'?')
                ln = s.readline().decode(errors='replace').strip()
            if not ln.startswith('INFO'):
                self.board = {'state': 'no reply', 'info': None, 'error': f'no INFO reply on {self.args.port}'}
                return
            info = dict(kv.split('=', 1) for kv in ln.split()[1:])
            info['test'] = 'story' if 'window' in info else 'mnist'
            self.board = {'state': 'ok', 'info': info, 'error': ''}
        except Exception as e:  # serial errors, missing port
            self.board = {'state': 'not found', 'info': None, 'error': str(e)}

    def options(self):
        exes = sorted(p.name for p in (ROOT / 'firmware/baselines/ref_c/build-host').glob('ref_host_*')
                      if p.is_file() and p.suffix in ('', '.exe'))
        npz = sorted(p.name for p in (ROOT / 'train/runs').glob('*.npz'))
        bitnet = ROOT / 'train/BitNetMCU'
        return {'port': self.args.port, 'board': self.board, 'builds': list_builds(),
                'mnist_models': [m for m in MNIST_GOLDEN if (ROOT / 'train/export' / f'{m}.h').exists()],
                'host_exes': exes, 'story_npz': npz, 'flash_tool': bool(git_bash()),
                'mnist_data': (bitnet / 'data' / 'MNIST' / 'raw').exists(),
                'bitnet': (bitnet / 'exportquant.py').exists()}

    # ---- running a test
    def status(self):
        st = dict(self.state)
        st['log'] = list(self.log)[-40:]
        st['running'] = bool(self.child and self.child.poll() is None)
        return st

    def _set(self, state, detail='', **kw):
        self.state.update(state=state, detail=detail, since=time.time(), **kw)
        self.log.append(f'[{time.strftime("%H:%M:%S")}] {state}{": " + detail if detail else ""}')

    def stop(self):
        self.cancel.set()
        with self.lock:
            self._stop_child()
            self._set('idle', 'stopped', url=None)

    def _stop_child(self):
        if self.child and self.child.poll() is None:
            self.child.terminate()
            try:
                self.child.wait(10)
            except subprocess.TimeoutExpired:
                self.child.kill()
                self.child.wait()
        self.child = None

    def start(self, req: dict):
        if self.job and self.job.is_alive():
            return {'error': 'a test is already starting; wait for it or stop it'}
        self.cancel.clear()
        self.job = threading.Thread(target=self._start, args=(req,), daemon=True)
        self.job.start()
        return {'ok': True}

    def _start(self, req: dict):
        test, target = req.get('test'), req.get('target')
        py = sys.executable
        view = self.args.view_port
        home = f'http://127.0.0.1:{self.args.http_port}/'
        cmd = [py, '-u', str(HERE / 'bridge.py'), '--http-port', str(view), '--home', home]
        label = ''
        try:
            if test not in ('mnist', 'story') or target not in ('golden', 'host', 'pico'):
                raise ValueError('choose a test and a target')
            if target == 'golden' and test == 'mnist':
                m = req.get('model') or 'tern80'
                if m not in MNIST_GOLDEN:
                    raise ValueError(f'unknown MNIST model {m}')
                cmd += ['--task', 'mnist', '--target', 'golden', '--model', m]
                label = f'MNIST · golden model · {m}'
            elif target == 'golden':
                npz = ROOT / 'train/runs' / Path(req.get('npz') or 'lm_tree.npz').name
                if not npz.exists():
                    raise ValueError(f'{npz.name} not found in train/runs')
                cmd += ['--task', 'story', '--target', 'golden', '--npz', str(npz)]
                label = f'Tiny story · golden model · {npz.stem}'
            elif target == 'host':
                if test != 'mnist':
                    raise ValueError('the PC build exists for MNIST only')
                exe = ROOT / 'firmware/baselines/ref_c/build-host' / Path(req.get('exe') or '').name
                if not exe.exists():
                    raise ValueError('choose a PC build')
                cmd += ['--task', 'mnist', '--target', 'host', '--exe', str(exe)]
                label = f'MNIST · PC build · {exe.stem}'
            else:
                cmd += ['--task', test, '--target', 'pico', '--port', self.args.port]
                label = f'{"MNIST" if test == "mnist" else "Tiny story"} · board'
            b = None
            if target == 'pico' and req.get('build'):   # check the request before stopping the running test
                b = next((x for x in list_builds() if x['path'] == req['build']), None)
                if not b:
                    raise ValueError(f'unknown build {req["build"]}')
                if b['test'] != test:
                    raise ValueError(f'{b["name"]} is a {b["test"]} build')
                if b.get('unsupported'):
                    raise ValueError(b['unsupported'])
            with self.lock:
                self.log.clear()
                if self.child and self.child.poll() is None:
                    self._set('stopping', 'stopping the running test', url=None, test=label)
                else:
                    self.state['test'] = label
                self._stop_child()
            for _ in range(10):                        # a stopped view can take a moment to release its port
                if port_free(view):
                    break
                time.sleep(0.5)
            else:
                raise RuntimeError(f'port {view} is in use by another program (an old live view?). Close it, or start '
                                   f'the launcher with a different --view-port')
            if target == 'pico':
                if b:
                    self._flash(ROOT / b['path'])
                    if self.cancel.is_set():
                        return self._cancelled()
                    label += f' · {b["name"]}'
                    self.state['test'] = label
                else:
                    label += ' · build already on the board'
                    self.state['test'] = label
            self._set('starting', 'loading the MNIST images, the golden model and PyTorch (up to a minute)' if test == 'mnist'
                      else 'loading the golden model')
            with self.lock:
                if self.cancel.is_set():
                    return self._cancelled()
                child = self.child = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                      text=True, env=dict(os.environ, PYTHONIOENCODING='utf-8'), bufsize=1)
            self.pump = threading.Thread(target=self._pump, args=(child,), daemon=True)
            self.pump.start()
            url = f'http://127.0.0.1:{view}/'
            t0 = time.time()
            while time.time() - t0 < 300:
                if self.cancel.is_set():
                    return self._cancelled()     # Stop was pressed; stop() has already ended the child
                if child.poll() is not None:
                    self.pump.join(2)            # let the last lines of output reach the log
                    raise RuntimeError('the live view stopped: ' + last_error(self.log))
                try:
                    with urllib.request.urlopen(url, timeout=2) as r:
                        if r.status == 200:
                            break
                except OSError:
                    time.sleep(1)
            else:
                raise RuntimeError('the live view did not come up within 5 minutes')
            self._set('running', label, url=url)
            if target == 'pico':
                self.board = dict(self.board, state='in use', error='the running test holds the port')
        except Exception as e:  # report every failure on the page
            if not self.cancel.is_set():
                self._set('error', str(e), url=None)

    def _cancelled(self):
        if self.state.get('state') != 'idle':
            self._set('idle', 'stopped', url=None)

    def _flash(self, uf2: Path):
        bash = git_bash()
        if not bash:
            raise RuntimeError('bash not found: harness/flash.sh needs it (Git Bash on Windows)')
        self._set('flashing', f'{uf2.parent.name} (SWD, picotool as the fallback)')
        py = sys.executable.replace('\\', '/')
        p = subprocess.Popen([bash, '-c', '. harness/flash.sh && flash "$1"', '_', str(uf2).replace('\\', '/')],
                             cwd=ROOT, env=dict(os.environ, PY=py, PORT=self.args.port), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        out = []
        reader = threading.Thread(target=lambda: out.extend(p.stdout), daemon=True)
        reader.start()
        t0 = time.time()
        while p.poll() is None:
            if self.cancel.wait(0.2):
                p.kill()
                self.log.append('  flashing stopped')
                return
            if time.time() - t0 > 180:
                p.kill()
                raise RuntimeError('flashing took longer than 3 minutes')
        reader.join(2)
        for ln in out[-15:]:
            self.log.append('  ' + ln.rstrip())
        if p.returncode != 0:
            raise RuntimeError('flashing failed (see the log)')
        self.probe()

    def _pump(self, proc):
        for ln in proc.stdout:
            self.log.append('  ' + ln.rstrip())
        code = proc.wait()
        if self.state.get('state') == 'running' and proc is self.child:
            self._set('error', f'the live view exited (code {code})', url=None)


def port_free(port: int) -> bool:
    """True if nothing listens on 127.0.0.1:port."""
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(('127.0.0.1', port)) != 0


def last_error(log) -> str:
    """The most useful line of a failed child's output: its last error line, else its last line."""
    lines = [ln.strip() for ln in log if ln.strip() and not ln.startswith('[')]
    for ln in reversed(lines):
        if 'Error' in ln or 'error' in ln or 'not found' in ln:
            return ln
    return lines[-1] if lines else 'no output'


def make_handler(L: Launcher):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path).path
            if u in ('/', '/index.html'):
                body = UI.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)
            elif u == '/api/options':
                self._json(L.options())
            elif u == '/api/status':
                self._json(L.status())
            else:
                self._json({'error': 'not found'}, 404)

        def do_POST(self):
            u = urlparse(self.path).path
            n = int(self.headers.get('Content-Length') or 0)
            try:
                req = json.loads(self.rfile.read(n) or b'{}')
            except json.JSONDecodeError:
                req = {}
            if u == '/api/start':
                self._json(L.start(req))
            elif u == '/api/stop':
                L.stop()
                self._json({'ok': True})
            elif u == '/api/probe':
                threading.Thread(target=L.probe, daemon=True).start()
                self._json({'ok': True})
            else:
                self._json({'error': 'not found'}, 404)

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--port', default=DEFAULT_PORT, help=f'the board\'s serial port (default {DEFAULT_PORT})')
    ap.add_argument('--http-port', type=int, default=8765, help='this launcher page')
    ap.add_argument('--view-port', type=int, default=8766, help='where the chosen live view runs')
    args = ap.parse_args()
    L = Launcher(args)
    threading.Thread(target=L.probe, daemon=True).start()
    try:
        srv = ThreadingHTTPServer(('127.0.0.1', args.http_port), make_handler(L))
    except OSError as e:
        raise SystemExit(f'cannot listen on port {args.http_port} ({e.strerror}): is the launcher already running? '
                         f'Open http://127.0.0.1:{args.http_port}, or pick another port with --http-port') from None
    srv.daemon_threads = True
    print(f'test launcher on http://127.0.0.1:{args.http_port}  (live views on port {args.view_port}; Ctrl+C to stop)')
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        L._stop_child()


if __name__ == '__main__':
    main()
