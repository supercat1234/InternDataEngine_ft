#!/usr/bin/env python3
"""Local HTTP inference, using the checkpoint's complete LeRobot processors."""
import argparse
import json
import sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
import base64
import io
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deploy'))
from pi05_plain_server import PlainPI05
from cup_plate_adapter import ACTION_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='/home/agilex/qhx/output/pi05_simandreal/checkpoints/004000/pretrained_model')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--port', type=int, default=8097)
    args = parser.parse_args()
    root = Path(args.checkpoint).resolve()
    config = json.loads((root / 'config.json').read_text())
    if config.get('use_relative_actions') or config['output_features']['action']['shape'] != [14]:
        raise ValueError('Requires absolute 14D actions')
    if config.get('action_feature_names') != ACTION_NAMES:
        raise ValueError('Checkpoint action order does not match left/right 6 joints + gripper')
    engine = PlainPI05(str(root), str(root / 'tokenizer'), args.device)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != '/health':
                return self.reply(404, {'error': 'not found'})
            self.reply(200, {**engine.info(), 'checkpoint': str(root), 'absolute_actions': True,
                             'action_feature_names': ACTION_NAMES})

        def do_POST(self):
            if self.path != '/infer':
                return self.reply(404, {'error': 'not found'})
            try:
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size <= 8 * 1024 * 1024:
                    raise ValueError('Invalid body size')
                data = json.loads(self.rfile.read(size))
                with np.load(io.BytesIO(base64.b64decode(data['observation_npz'], validate=True)), allow_pickle=False) as obs:
                    result = engine.infer(dict(obs), data['task'])
                self.reply(200, result)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self.reply(500, {'error': str(exc)})

    print(f'PI05_READY checkpoint={root} port={args.port}', flush=True)
    HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()

if __name__ == '__main__':
    main()
