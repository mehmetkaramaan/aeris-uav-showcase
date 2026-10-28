from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import json, urllib.parse
from controller import Controller
ROOT = Path(__file__).parent
control = None

class Handler(SimpleHTTPRequestHandler):

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT / 'panel'), **kw)

    def log_message(self, *a):
        pass

    def reply(self, data, status=200, ctype='application/json; charset=utf-8'):
        raw = json.dumps(data, ensure_ascii=False, allow_nan=False).encode() if isinstance(data, (dict, list)) else data
        self.send_response(status)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path
        if path == '/api/state':
            self.reply(control.state())
        elif path == '/events.jsonl':
            files = sorted((ROOT / 'logs').glob('events-*.jsonl'))
            self.reply(files[-1].read_bytes() if files else b'', ctype='application/x-ndjson')
        elif path == '/flight.json':
            record = control.last_record
            if record is None:
                files = sorted((ROOT / 'flight_logs').glob('*.json'))
                if files:
                    record = json.loads(files[-1].read_text())
            self.reply(record if record is not None else {'error': 'No completed flight record yet.'}, 200 if record is not None else 404)
        elif path == '/camera.jpg':
            raw = control.vision.front if urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query).get('front') else control.vision.raw if urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query).get('raw') else control.vision.jpg
            if raw:
                self.reply(raw, ctype='image/jpeg')
            else:
                self.reply(b'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480"><rect width="640" height="480" fill="#152334"/><text x="150" y="240" fill="white" font-size="22">Waiting for Gazebo camera stream</text></svg>', ctype='image/svg+xml')
        else:
            super().do_GET()

    def do_POST(self):
        origin = self.headers.get('Origin')
        if origin and urllib.parse.urlsplit(origin).hostname not in {'localhost', '127.0.0.1'}:
            return self.reply({'error': 'Local access only.'}, 403)
        try:
            if self.path != '/api/command':
                return self.reply({'error': 'Endpoint not found.'}, 404)
            size = int(self.headers.get('Content-Length', '0'))
            if size > 8192:
                raise ValueError('Request too large.')
            d = json.loads(self.rfile.read(size))
            control.command(d.get('action'), d.get('params', {}))
            self.reply({'queued': True}, 202)
        except (ValueError, TypeError, KeyError) as e:
            self.reply({'error': str(e)}, 400)
if __name__ == '__main__':
    server = ThreadingHTTPServer(('127.0.0.1', 8772), Handler)
    control = Controller()
    print('AERIS Gazebo control: http://127.0.0.1:8772', flush=True)
    server.serve_forever()
