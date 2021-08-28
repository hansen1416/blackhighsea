#!/usr/bin/env python
from os import removedirs
import os.path
import logging
import re
import uuid
import sys
from io import BytesIO
import inspect
import json
from collections import deque

# import glob
import socket
from time import time
import datetime

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

logging.basicConfig(stream=sys.stdout, format=log_format, level=logging.INFO)
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
        tornado.websocket.WebSocketHandler.__init__(
            self, application, request, **kwargs
        )
        self.rows = []
        self.uuid = None
        self.start_time = 0
        self.message_queue = deque([], 2)

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
        return options.debug or bool(re.match(r"^.*\catlog\.kr", origin))

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
            CartoonGANHandler.send_message(self.uuid, self, "hi again")

            logging.info("new client sharing (uuid: %s)" % self.uuid)

        CartoonGANHandler.add_clients(self.uuid, self)

    def on_close(self):
        logging.info("close a websocket")

        CartoonGANHandler.remove_clients(self.uuid, self)

    # @tornado.gen.coroutine
    @classmethod
    def cartoongan_image(cls, uuid, client, input_image):

        logging.info("start cartoon gan")

        model_path = os.path.join("/opt", "gan-generator.pt")
        # input_image = os.path.join('/sharedvol', 'test.jpg')
        output_image = os.path.join("/sharedvol", uuid + str(int(time())) + "_out.jpg")

        HOST, PORT = "cpp-stylize", 8888

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

            s.connect((HOST, PORT))

            send_msg = model_path + " " + input_image + " " + output_image

            s.send(send_msg.encode("ascii"))

            while True:
                # timeout for transfer a image is 10 seconds
                s.settimeout(10)
                try:
                    recv_msg = s.recv(1024)
                except socket.timeout:
                    cls.send_message(uuid, client, "socket timeout")
                    break

                if type(recv_msg) == type(b""):

                    # logging.info('new message {}'.format(recv_msg))

                    recv_msg = recv_msg.decode("ascii")

                    if recv_msg[-4:] == ".jpg":
                        cls.send_message(uuid, client, json.dumps({"image": recv_msg}))

                        logging.info("new picture {}".format(recv_msg))

                        s.close()

                        break
                else:
                    break

    @classmethod
    def cartoongan_video(cls, uuid, client, video_bytesio):
        video_filename = "/sharedvol/test1.mp4"

        with open(video_filename, "wb") as outfile:
            # Copy the BytesIO stream to the output file
            outfile.write(video_bytesio.getbuffer())

        logging.info("video saved to %s" % video_filename)

        ###########
        # cap = cv2.VideoCapture(video_filename)

        # # fps = cap.get(cv2.CAP_PROP_FPS)

        # # logging.info("fps %d" % fps)

        # n = 0
        # frame_images = []

        # while cap.isOpened():

        #     # Capture frame-by-frame
        #     success, frame = cap.read()

        #     if success == True:
        #         # Display the resulting frame
        #         image_name = '/sharedvol/{}_{}.jpg'.format(uuid, n)
        #         cv2.imwrite(image_name, frame)
        #         frame_images.append(image_name)

        #         logging.info("frame image saved to %s" % image_name)

        #         n += 1
        #     else:
        #         break

        # # When everything done, release the video capture object
        # cap.release()

        # # Closes all the frames
        # cv2.destroyAllWindows()

        # model_path = os.path.join("/opt", "gan-generator.pt")

        # HOST, PORT = "cpp-stylize", 8888

        # transferred_frame = []

        # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

        #     s.connect((HOST, PORT))

        #     for i, frame in enumerate(frame_images):

        #         output_image = os.path.join("/sharedvol", "{}_{}_out.jpg".format(uuid, i))

        #         send_msg = model_path + " " + frame + " " + output_image

        #         s.send(send_msg.encode("ascii"))

        #         s.settimeout(10)

        #         try:
        #             recv_msg = s.recv(1024)

        #             transferred_frame.append(recv_msg)

        #             logging.info("frame transferred saved to %s" % recv_msg)
        #         except socket.timeout:
        #             logging.info("socket timeout")
        #             break
        ###########

        transferred_frame = []
        tmp_uuid = "adf30332-e84c-4cb1-9571-098b53f7a40a"

        for i in range(2, 48):
            outimages = "/sharedvol/{}_{}_out.jpg".format(tmp_uuid, i)

            # logging.info(os.path.isfile(outimages))
            transferred_frame.append(outimages)

        frame_data_array = []

        for filename in transferred_frame:
            img = cv2.imread(filename)
            height, width, _ = img.shape
            size = (width, height)
            frame_data_array.append(img)

        out_video_path = "/sharedvol/{}_video_out.avi".format(tmp_uuid)

        out_video = cv2.VideoWriter(
            out_video_path,
            cv2.VideoWriter_fourcc(*"DIVX"),
            15,
            size,
        )

        for frame in frame_data_array:
            out_video.write(frame)

        cv2.destroyAllWindows()
        out_video.release()

        cls.send_message(uuid, client, json.dumps({"video": out_video_path}))

        logging.info("send out video path %s" % out_video_path)

    def on_message(self, message):
        logging.info("got message from uuid: {}".format(self.uuid))

        # with BytesIO(message ) as f:
        #     file_data = f.read()
        #     logging.info(file_data.name)

        # res = inspect.getmembers(f, lambda a:not(inspect.isroutine(a)))

        if type(message) == type(""):
            self.message_queue.append(message)

        if len(self.message_queue) <= 0 or not isinstance(message, type(b"")):
            return

        if self.message_queue[-1] == "image":
            nparr = np.fromstring(message, np.uint8)

            logging.info(nparr)

            image_name = os.path.join("/sharedvol", self.uuid + ".jpg")

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

            dim = (
                int(img_np.shape[1] * scale_percent),
                int(img_np.shape[0] * scale_percent),
            )

            if scale_percent != 0:
                logging.info("resized to {} x {}".format(dim[0], dim[1]))

                img_np = cv2.resize(img_np, dim, interpolation=cv2.INTER_AREA)

            cv2.imwrite(image_name, img_np)

            logging.info("saved image to " + image_name)

            # The IOLoop will catch the exception and print a stack trace in
            # the logs. Note that this doesn't look like a normal call, since
            # we pass the function object to be called by the IOLoop.
            tornado.ioloop.IOLoop.current().spawn_callback(
                CartoonGANHandler.cartoongan_image, self.uuid, self, image_name
            )

            # CartoonGANHandler.cartoongan(self.uuid, self, image_name)

            logging.info("on image message finished")
        elif self.message_queue[-1] == "video":

            logging.info("process video")

            bytesio = BytesIO(message)

            # The IOLoop will catch the exception and print a stack trace in
            # the logs. Note that this doesn't look like a normal call, since
            # we pass the function object to be called by the IOLoop.
            tornado.ioloop.IOLoop.current().spawn_callback(
                CartoonGANHandler.cartoongan_video, self.uuid, self, bytesio
            )

            logging.info("on video message finished")

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
        debug=options.debug,
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
