#!/usr/bin/env python
import os.path
import logging
import re
import uuid
import sys
# import glob
import socket
from time import time

import asyncio
import numpy as np
import cv2
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
        tornado.websocket.WebSocketHandler.__init__(self, \
            application, request, **kwargs)
        self.rows = []
        self.uuid = None
        self.start_time = 0

    @classmethod
    def send_message(cls, doc_uuid, client, message):

        clients_with_uuid = cls.clients[doc_uuid]
        logging.info("sending message to %d clients", len(clients_with_uuid))

        logging.info("send message" + message)

        # message = cls.make_message(doc_uuid)
        client.write_message(message)

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

        # we will pass uuid in cookie
        # print("received cookies: ", self.request.cookies)
        # print("received 'myuser': ", self.get_cookie("myuser"))

        logging.info("open a websocket (uuid: %s)" % doc_uuid)

        if doc_uuid is None:
            # Generate a random UUID
            self.uuid = str(uuid.uuid4())

            logging.info("new client with (uuid: %s)" % self.uuid)
        else:
            self.uuid = doc_uuid
            CartoonGANHandler.send_message(self.uuid, self, 'hi again')

            logging.info("new client sharing (uuid: %s)" % self.uuid)

        CartoonGANHandler.add_clients(self.uuid, self)

    def on_close(self):
        logging.info("close a websocket")

        CartoonGANHandler.remove_clients(self.uuid, self)

    @classmethod
    async def cartoongan(cls, uuid, client, input_image):

        logging.info('start cartoon gan')

        model_path = os.path.join('/opt', 'gan-generator.pt')
        # input_image = os.path.join('/sharedvol', 'test.jpg')
        output_image = os.path.join('/sharedvol', \
            uuid + str(int(time())) + '_out.jpg')

        HOST, PORT = "cpp-stylize", 8888

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        s.connect((HOST, PORT))

        send_msg = model_path + " " + input_image + " " + output_image

        s.send(send_msg.encode('ascii'))

        while True:

            recv_msg = s.recv(1024)

            if not recv_msg:
                break

            if type(recv_msg) == type(b''):
                
                # logging.info('new message {}'.format(recv_msg))

                recv_msg = recv_msg.decode('ascii')

                if recv_msg[-4:] == '.jpg':
                    cls.send_message(uuid, client, str(recv_msg))

                    logging.info('new picture {}'.format(recv_msg))

                    s.close()

                    break

        s.close()

    def on_message(self, message):
        logging.info("got message from uuid: {}".format(self.uuid))

        if isinstance(message, type(b'')):

            image_name = os.path.join('/sharedvol', self.uuid + '.jpg')

            # read image from string
            nparr = np.fromstring(message, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # resize image if either weight or height is more than 600
            # if the size is too big, it will crush the pytorch model
            max_size = 600
            scale_percent = 0

            if img_np.shape[1] > max_size:
                scale_percent = max_size / img_np.shape[1]
            elif img_np.shape[0] > max_size:
                scale_percent = max_size / img_np.shape[0]

            dim = (int(img_np.shape[1] * scale_percent), \
                int(img_np.shape[0] * scale_percent))

            if scale_percent != 0:
                logging.info('resized to {} x {}'.format(dim[0], dim[1]))

                img_np = cv2.resize(img_np, dim, interpolation = cv2.INTER_AREA)

            cv2.imwrite(image_name, img_np)

            logging.info('saved image to ' + image_name)

            # The IOLoop will catch the exception and print a stack trace in
            # the logs. Note that this doesn't look like a normal call, since
            # we pass the function object to be called by the IOLoop.
            tornado.ioloop.IOLoop.current().\
                spawn_callback(CartoonGANHandler.cartoongan, self.uuid, self, image_name)

            logging.info('on message finished')

    # @classmethod
    # def create_video_from_image(cls):
    #     img_array = []
    #     for filename in glob.glob('C:/New folder/Images/*.jpg'):
    #         img = cv2.imread(filename)
    #         height, width, layers = img.shape
    #         size = (width,height)
    #         img_array.append(img)
        
        
    #     out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        
    #     for i in range(len(img_array)):
    #         out.write(img_array[i])

    #     cv2.destroyAllWindows()        
    #     out.release()


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