"""
A simple (unsecured & slow) web-server to provide a cURL API to the SOFAST systems.

Starter code for this server from "How to Launch an HTTP Server in One Line of Python Code" by realpython.com
(https://realpython.com/python-http-server/).
"""

from concurrent.futures import ThreadPoolExecutor
import dataclasses
import gc
import json
from functools import cached_property
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
from traceback import format_exception
from urllib.parse import parse_qsl, urlparse

import numpy as np

import opencsp.app.sofast.lib.ServerState as ss
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection
from opencsp.common.lib.opencsp_path import opencsp_settings
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.time_date_tools as tdt


@dataclasses.dataclass
class _UrlParseResult:
    """Helper class to define the slot names of the ParseResult type"""

    scheme: str
    net_location: str
    path: str
    params: str
    query: str
    fragment: str


class SofastServer(BaseHTTPRequestHandler):
    @cached_property
    def url(self) -> _UrlParseResult:
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
        response_code, response_msg = self.get_response()
        self.send_response(response_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(response_msg.encode("utf-8"))

    def do_POST(self):
        self.do_GET()

    def get_response(self) -> tuple[int, str]:
        action = "N/A"
        ret = {"error": None}
        response_code = 200

        try:
            if "/" in self.url.path:
                action = self.url.path.split("/")[-1]
            else:
                action = self.url.path

            if action == "help":
                ret["actions"] = [
                    "help",
                    "start_measure_fringes",
                    "is_busy",
                    "save_measure_fringes",
                    "get_results_fringes",
                ]

            if action == "start_measure_fringes":
                with ss.ServerState.instance() as state:
                    name = self.query_data["name"]
                    ret["success"] = state.start_measure_fringes(name)

            elif action == "is_busy":
                with ss.ServerState.instance() as state:
                    ret["is_busy"] = state.busy

            elif action == "save_measure_fringes":
                if "saves_output_dir" in opencsp_settings and ft.directory_exists(opencsp_settings["saves_output_dir"]):
                    measurement = None
                    processing_error = None
                    with ss.ServerState.instance() as state:
                        if state.has_fringe_measurement:
                            measurement = state.last_measurement_fringe[0]
                            file_name_ext = state.fringe_measurement_name + ".h5"
                        else:
                            processing_error = state.processing_error
                    if measurement is not None:
                        file_path_name_ext = os.path.join(opencsp_settings["saves_output_dir"], file_name_ext)
                        measurement.save_to_hdf(file_path_name_ext)
                        ret["file_name_ext"] = file_name_ext
                    elif processing_error is not None:
                        ret["error"] = (
                            f"Unexpected {repr(processing_error)} error encountered during measurement processing"
                        )
                        ret["trace"] = "".join(format_exception(processing_error))
                        response_code = 500
                    else:
                        ret["error"] = "Fringe measurement is not ready"
                        ret["trace"] = "SofastServer.get_response::save_measure_fringes"
                        response_code = 409
                else:
                    ret["error"] = "Measurements save directory not speicified in settings"
                    ret["trace"] = "SofastServer.get_response::save_measure_fringes"
                    response_code = 500

            elif action == "get_results_fringes":
                measurement = None
                with ss.ServerState.instance() as state:
                    if state.has_fringe_measurement:
                        measurement = state.last_measurement_fringe
                        state.system_fringe
                if measurement is not None:
                    ret.update(
                        {
                            "focal_length_x": measurement.focal_length_x,
                            "focal_length_y": measurement.focal_length_y,
                            "slope_error_x": np.average(measurement.slopes_error[0]),
                            "slope_error_y": np.average(measurement.slopes_error[1]),
                            "slope_error": np.average(measurement.slopes_error),
                            "slope_stddev": np.std(measurement.slopes_error),
                        }
                    )
                else:
                    ret["error"] = "Fringe measurement is not ready"
                    ret["trace"] = "SofastServer.get_response::get_results_fringes"
                    response_code = 409

            else:
                ret["error"] = f"Unknown action \"{action}\""
                ret["trace"] = "SofastServer.get_response::N/A"
                response_code = 404

        except Exception as ex:
            lt.error("Error in SofastServer with action " + action + ": " + repr(ex))
            ret["error"] = (repr(ex),)
            ret["trace"] = "".join(format_exception(ex))
            response_code = 500

        # sanity check: did we synchronize the error and response_code?
        if response_code != 200:
            if ret["error"] is None:
                lt.error_and_raise(
                    RuntimeError,
                    f"Programmer error in SofastServer.get_response({action}): "
                    + f"did not correctly set 'error' to match {response_code=}!",
                )
        if ret["error"] is not None:
            if response_code == 200:
                lt.error_and_raise(
                    RuntimeError,
                    f"Programmer error in SofastServer.get_response({action}): "
                    + f"did not correctly set response_code to match {ret['error']=}!",
                )

        return response_code, json.dumps(ret)


if __name__ == "__main__":
    port = 8000

    # Set up the logger
    log_output_dir = opencsp_settings["sofast_server"]["log_output_dir"]
    if log_output_dir is not None and ft.directory_exists(log_output_dir):
        log_name_ext = "SofastServer_" + tdt.current_date_time_string_forfile() + ".log"
        log_path_name_ext = os.path.join(log_output_dir, log_name_ext)
        lt.logger(log_path_name_ext)

    # Start the server
    lt.warn(
        "Warning in SofastServer: this server is unsecured. "
        + f"It is suggested that you restrict outside access to port {port} of the host computer."
    )
    lt.info(f"Starting server on port {port}...")
    server = HTTPServer(("0.0.0.0", port), SofastServer)

    # Initialize the IO devices
    ss.ServerState()
    with ss.ServerState.instance() as state:
        state.init_io()
        state.load_default_settings()

    # Lock in the currently allocated memory, to improve garbage collector performance
    gc.collect()
    gc.freeze()

    # Start a new thread for the server
    # The minimum time between server evaulation loops is determined by the GIL:
    # https://docs.python.org/3/library/sys.html#sys.setswitchinterval
    server_pool = ThreadPoolExecutor(max_workers=1)
    server_pool.submit(server.serve_forever)

    # Start the GUI thread
    ImageProjection.instance().root.mainloop()

    # GUI has exited, shutdown everything
    with ss.ServerState.instance() as state:
        state.close_all()
    server.shutdown()
    server_pool.shutdown()
