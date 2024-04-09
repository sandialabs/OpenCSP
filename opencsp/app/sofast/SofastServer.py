"""
A simple (unsecured & slow) web-server to provide a cURL API to the SOFAST systems.

Starter code for this server from "How to Launch an HTTP Server in One Line of Python Code" by realpython.com
(https://realpython.com/python-http-server/).
"""

from concurrent.futures import ThreadPoolExecutor
import gc
import json
from functools import cached_property
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qsl, urlparse

from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection
import opencsp.common.lib.tool.log_tools as lt
import opencsp.app.sofast.lib.ServerState as ss


class SofastServer(BaseHTTPRequestHandler):
    @cached_property
    def url(self):
        return urlparse(self.path)

    @cached_property
    def query_data(self):
        return dict(parse_qsl(self.url.query))

    @cached_property
    def post_data(self):
        content_length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(content_length)

    @cached_property
    def form_data(self):
        return dict(parse_qsl(self.post_data.decode("utf-8")))

    @cached_property
    def cookies(self):
        return SimpleCookie(self.headers.get("Cookie"))

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(self.get_response().encode("utf-8"))

    def do_POST(self):
        self.do_GET()

    def get_response(self):
        return json.dumps(
            {
                "path": self.url.path,
                "query_data": self.query_data,
                "post_data": self.post_data.decode("utf-8"),
                "form_data": self.form_data,
                "cookies": {
                    name: cookie.value
                    for name, cookie in self.cookies.items()
                },
            }
        )


if __name__ == "__main__":
    port = 8000

    # Start the server
    lt.warn("Warning in SofastServer: this server is unsecured. " +
            f"It is suggested that you restrict outside access to port {port} of the host computer.")
    lt.info(f"Starting server on port {port}...")
    server = HTTPServer(("0.0.0.0", port), SofastServer)

    # Initialize the IO devices
    state = ss.ServerState()
    state.init_io()

    # Lock in the currently allocated memory, to improve garbage collector performance
    gc.collect()
    gc.freeze()

    # Start a new thread for the server
    # The minimum time between server evaulation loops is determined by the GIL:
    # https://docs.python.org/3/library/sys.html#sys.setswitchinterval
    pool = ThreadPoolExecutor(max_workers=1)
    pool.submit(server.serve_forever)

    # Start the GUI thread
    ImageProjection.instance().root.mainloop()

    # GUI has exited, shutdown everything
    state.close_all()
    server.shutdown()
    pool.shutdown()
