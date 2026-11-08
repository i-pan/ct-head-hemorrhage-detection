#!/usr/bin/env python3
"""Serve the static research dashboard on loopback and the supplied tailnet IP."""
import argparse,json,threading
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
class Handler(SimpleHTTPRequestHandler):
    def list_directory(self,path):
        self.send_error(403,'Directory listing disabled')
    def end_headers(self):
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Cache-Control','no-cache')
        super().end_headers()
    def do_GET(self):
        if self.path=='/health':
            body=json.dumps({'status':'ok','application':'ich-coronal-lab'}).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        else:super().do_GET()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tailnet-ip',required=True);p.add_argument('--port',type=int,default=8773);a=p.parse_args()
    handler=partial(Handler,directory=str(Path(__file__).resolve().parent/'dist'))
    servers=[ThreadingHTTPServer((host,a.port),handler) for host in ['127.0.0.1',a.tailnet_ip]]
    for server in servers[:-1]:threading.Thread(target=server.serve_forever,daemon=True).start()
    print(f'Coronal Lab: http://{a.tailnet_ip}:{a.port} / http://127.0.0.1:{a.port}',flush=True)
    try:servers[-1].serve_forever()
    finally:
        for server in servers:server.server_close()
