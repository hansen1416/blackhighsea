#!/usr/bin/env python
import os.path
import logging
import re
import csv
import uuid
import sys

import tornado.escape
import tornado.ioloop
import tornado.web
import tornado.websocket
import tornado.locks
import tornado.gen

from tornado.options import define, options, parse_command_line

define("port", default=4601, help="run on the given port", type=int)
define("debug", default=True, help="run in debug mode")

lock = tornado.locks.Lock()

log_format = "%(levelname)s %(asctime)s - %(message)s"

logging.basicConfig(stream = sys.stdout, format = log_format, level = logging.INFO)
logger = logging.getLogger()


class MainHandler(tornado.web.RequestHandler):
    def get(self, doc_uuid=""):
        logging.info("index with (uuid: %s)" % doc_uuid)

        self.write("hi, " + doc_uuid)


class CartoonGANHandler(tornado.websocket.WebSocketHandler):
    clients = {}
    files = {}
    page_size = 100

    def __init__(self, application, request, **kwargs):
        tornado.websocket.WebSocketHandler.__init__(self, application, request, **kwargs)
        self.rows = []
        self.uuid = None

    @classmethod
    def send_message(cls, doc_uuid, client):
        clients_with_uuid = cls.clients[doc_uuid]
        logging.info("sending message to %d clients", len(clients_with_uuid))

        message = cls.make_message(doc_uuid)
        client.write_message(message)

    @classmethod
    def make_message(cls, doc_uuid):
        rows = cls.files[doc_uuid]["rows"]
        page_no = cls.files[doc_uuid]["page_no"]

        return {
            "uuid": doc_uuid,
            "page_no": page_no,
            "total_number": len(rows),
            "data": rows[cls.page_size * (page_no - 1):cls.page_size * page_no]
        }

    @classmethod
    def load_file(cls, doc_uuid, tsv_file):
        if not (bytes is str):
            tsv_file = str(tsv_file, 'utf-8')
        lines = ( x.strip() for x in tsv_file.splitlines() if x.strip() )
        rows = list( csv.reader(lines, delimiter="\t") )

        cls.files[doc_uuid] = {"rows": rows, "page_no": 1}

    @classmethod
    @tornado.gen.coroutine
    def add_clients(cls, doc_uuid, client):
        logging.info("add a client with (uuid: %s)" % doc_uuid)

        # locking clients
        with (yield lock.acquire()):
            if doc_uuid in cls.clients:
                clients_with_uuid = CartoonGANHandler.clients[doc_uuid]
                clients_with_uuid.append(client)
            else:
                CartoonGANHandler.clients[doc_uuid] = [client]

    @classmethod
    @tornado.gen.coroutine
    def remove_clients(cls, doc_uuid, client):
        logging.info("remove a client with (uuid: %s)" % doc_uuid)

        # locking clients
        with (yield lock.acquire()):
            if doc_uuid in cls.clients:
                clients_with_uuid = CartoonGANHandler.clients[doc_uuid]
                clients_with_uuid.remove(client)

                if len(clients_with_uuid) == 0:
                    del cls.clients[doc_uuid]

            if doc_uuid not in cls.clients and doc_uuid in cls.files:
                del cls.files[doc_uuid]

    def check_origin(self, origin):
        return options.debug or bool(re.match(r'^.*\catlog\.kr', origin))

    def get_compression_options(self):
        # Non-None enables compression with default options.
        return {}

    def open(self, doc_uuid=None):
        logging.info("open a websocket (uuid: %s)" % doc_uuid)

        if doc_uuid is None:
            # Generate a random UUID
            self.uuid = str(uuid.uuid4())

            logging.info("new client with (uuid: %s)" % self.uuid)
        else:
            self.uuid = doc_uuid
            CartoonGANHandler.send_message(self.uuid, self)

            logging.info("new client sharing (uuid: %s)" % self.uuid)

        CartoonGANHandler.add_clients(self.uuid, self)

    def on_close(self):
        logging.info("close a websocket")

        CartoonGANHandler.remove_clients(self.uuid, self)

    def on_message(self, message):
        logging.info("got message (uuid: %s)" % self.uuid)
        if isinstance(message, type(b'')):
            CartoonGANHandler.load_file(self.uuid, message)
        else:
            logging.info("page_no: " + message)

            page_no = int(message)
            CartoonGANHandler.files[self.uuid]["page_no"] = page_no

if __name__ == "__main__":
    parse_command_line()
    settings = dict(
            cookie_secret="SX4gEWPE6bVr0vbwGtMl",
            # template_path=os.path.join(os.path.dirname(__file__), "templates"),
            # static_path=os.path.join(os.path.dirname(__file__), "static"),
            xsrf_cookies=True,
            debug=options.debug
    )

    handlers = [
            (r"/", MainHandler),
            (r"/ws/cartoongan", CartoonGANHandler),
            # (r"/parser/ws", CartoonGANHandler),
            # (r"/parser/ws/([^/]+)", CartoonGANHandler),
            # (r"/parser/static/(.*)", tornado.web.StaticCartoonGANHandler, {"path": settings["static_path"]})
    ]

    app = tornado.web.Application(handlers, **settings)
    app.listen(options.port)

    logger.info("listening on port " + str(options.port))

    tornado.ioloop.IOLoop.current().start()