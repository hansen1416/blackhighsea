#!/usr/bin/env python
import os

# from os import removedirs
import os.path
import logging
import re
import uuid
import sys
from io import BytesIO
import json
import tempfile

# import glob
import socket
from time import time

# import datetime
# from collections import deque

import smtplib
import email.utils
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

import oss2

# import asyncio
import numpy as np
import cv2
import tornado.web

# import tornado.escape
# import tornado.ioloop
# import tornado.websocket
# import tornado.locks
# import tornado.gen
# from tornado.options import define, options, parse_command_line

# define("port", default=4601, help="run on the given port", type=int)
# define("debug", default=True, help="run in debug mode")

# lock = tornado.locks.Lock()

log_format = "%(levelname)s %(asctime)s - %(message)s"

logging.basicConfig(stream=sys.stdout, format=log_format, level=logging.INFO)
logger = logging.getLogger()


def send_email_with_video(to, video_file):

    # Create the message
    msg = MIMEMultipart()

    msg["To"] = email.utils.formataddr(("Recipient", to))
    msg["From"] = email.utils.formataddr(("BlackHighSea", "badapplesweetie@gmail.com"))
    msg["Subject"] = "Your video is transformed"

    # string to store the body of the mail
    body = "Hi,\n\nTanks for using my service, here is your video."

    # attach the body with the msg instance
    msg.attach(MIMEText(body, "plain"))

    # open the file to be sent
    attachment = open(video_file, "rb")

    # instance of MIMEBase and named as p
    p = MIMEBase("application", "octet-stream")

    # To change the payload into encoded form
    p.set_payload((attachment).read())

    # encode into base64
    encoders.encode_base64(p)

    filename = video_file.split("/").pop()

    p.add_header("Content-Disposition", "attachment; filename= %s" % filename)

    # attach the instance 'p' to instance 'msg'
    msg.attach(p)

    server = smtplib.SMTP()

    server.set_debuglevel(True)  # show communication with the server

    # start TLS for security
    # server.starttls()

    server.connect("in-v3.mailjet.com", 587)

    username = os.environ.get("SMTP_USERNAME")
    password = os.environ.get("SMTP_PASSWORD")

    server.login(username, password)

    try:
        server.sendmail("badapplesweetie@gmail.com", [to], msg.as_string())
    finally:
        server.quit()


def resize_image(img_np):
    # resize image if either weight or height is more than 600
    # if the size is too big, it will crush the pytorch model
    max_size = 600
    scale_percent = 0

    if img_np.shape[1] > img_np.shape[0] and img_np.shape[1] > max_size:
        scale_percent = max_size / img_np.shape[1]
    elif img_np.shape[0] > img_np.shape[1] and img_np.shape[0] > max_size:
        scale_percent = max_size / img_np.shape[0]

    dim = (
        int(img_np.shape[1] * scale_percent),
        int(img_np.shape[0] * scale_percent),
    )

    if scale_percent != 0:
        logging.info("resized to {} x {}".format(dim[0], dim[1]))

        img_np = cv2.resize(img_np, dim, interpolation=cv2.INTER_AREA)

    return img_np


def cartoongan_image(input_image, output_image):

    logging.info("start cartoon gan")

    model_path = os.path.join("/opt", "gan-generator.pt")
    # input_image = os.path.join('/sharedvol', 'test.jpg')

    HOST, PORT = "cpp-cartoongan", 4602

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

        s.connect((HOST, PORT))

        logging.info("connected to host {} port {}".format(HOST, PORT))

        send_msg = model_path + " " + input_image + " " + output_image

        s.send(send_msg.encode("ascii"))

        while True:
            # timeout for transfer a image is 10 seconds
            s.settimeout(10)
            try:
                recv_msg = s.recv(1024)
            except socket.timeout:
                logging.info("something wrong")
                break

            if type(recv_msg) == type(b""):

                # logging.info('new message {}'.format(recv_msg))

                recv_msg = recv_msg.decode("ascii")

                if recv_msg[-4:] == ".jpg":

                    logging.info("new picture {}".format(recv_msg))

                    s.close()

                    break
            else:
                break


class BaseHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        # print "setting headers!!!"
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.set_header(
            "Access-Control-Allow-Headers",
            "x-requested-with,access-control-allow-origin,authorization,content-type",
        )

    def get(self):
        self.write("get ok")

    def post(self):
        self.write("post ok")

    def options(self, *args):
        # no body
        # `*args` is for route with `path arguments` supports
        self.set_status(204)
        self.finish()


class MainHandler(BaseHandler):
    pass


class HealthHandler(BaseHandler):
    pass


class CartoonGANHandler(BaseHandler):
    def post(self):

        media = self.request.files["media"][0]
        original_fname = media["filename"]
        # media['body'] is ninary, ndarray is np.ndarray
        ndarray = cv2.imdecode(np.frombuffer(media['body'], np.uint8), cv2.IMREAD_COLOR)
        # resize if the image is too big
        ndarray = resize_image(ndarray)
        # from ndarray back to bytes
        bytes_tuple = cv2.imencode('.jpg', ndarray)

        media['body'] = bytes_tuple[1].tobytes()

        email = self.get_body_argument("email", default="")

        # 阿里云账号AccessKey拥有所有API的访问权限，风险很高。强烈建议您创建并使用RAM用户进行API访问或日常运维，请登录RAM控制台创建RAM用户。
        auth = oss2.Auth(
            os.environ.get("ALI_ACCESS_ID"), os.environ.get("ALI_ACCESS_KEY")
        )
        # yourEndpoint填写Bucket所在地域对应的Endpoint
        # 填写Bucket名称。
        bucket = oss2.Bucket(auth, "oss-cn-hongkong.aliyuncs.com", "bhs-media")

        if email:
            object_name = "videos/" + str(time()) + "_" + original_fname
            output_object = "videos/cg_" + str(time()) + "_" + original_fname
        else:
            object_name = "imgs/" + str(time()) + "_" + original_fname
            output_object = "imgs/cg_" + str(time()) + "_" + original_fname

        try:
            # 填写Object完整路径和Bytes内容。Object完整路径中不能包含Bucket名称。
            bucket.put_object(object_name, media["body"])
            # becareful, no https:// prefix here, in cpp we are searching for '/'
            hostname = "bhs-media.oss-cn-hongkong.aliyuncs.com/"

            if email:
                pass
            else:
                # The IOLoop will catch the exception and print a stack trace in
                # the logs. Note that this doesn't look like a normal call, since
                # we pass the function object to be called by the IOLoop.
                tornado.ioloop.IOLoop.current().spawn_callback(
                    cartoongan_image, hostname + object_name, output_object
                )

            self.write('https://' + hostname + output_object)

        finally:
            logging.info("upload image failed")

    @classmethod
    def cartoongan_video(cls, uuid, client, video_bytesio, email):
        video_filename = "/sharedvol/test1.mp4"

        with open(video_filename, "wb") as outfile:
            # Copy the BytesIO stream to the output file
            outfile.write(video_bytesio.getbuffer())

        logging.info("video saved to %s" % video_filename)

        cap = cv2.VideoCapture(video_filename)

        fps = cap.get(cv2.CAP_PROP_FPS)

        logging.info("fps %d" % fps)

        ###########
        n = 0
        frame_images = []

        while cap.isOpened():

            # Capture frame-by-frame
            success, frame = cap.read()

            if success == True:
                # Display the resulting frame
                image_name = "/sharedvol/{}_{}.jpg".format(uuid, n)

                frame = CartoonGANHandler.resize_image(frame)

                cv2.imwrite(image_name, frame)
                frame_images.append(image_name)

                logging.info("frame image saved to %s" % image_name)

                n += 1
            else:
                break

        # When everything done, release the video capture object
        cap.release()

        # Closes all the frames
        cv2.destroyAllWindows()

        model_path = os.path.join("/opt", "gan-generator.pt")

        HOST, PORT = "cpp-stylize", 4602

        transferred_frame = []

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

            s.connect((HOST, PORT))

            for i, frame in enumerate(frame_images):

                output_image = os.path.join(
                    "/sharedvol", "{}_{}_out.jpg".format(uuid, i)
                )

                send_msg = model_path + " " + frame + " " + output_image

                s.send(send_msg.encode("ascii"))

                s.settimeout(10)

                try:
                    recv_msg = s.recv(1024)

                    if os.path.isfile(recv_msg):
                        transferred_frame.append(recv_msg.decode("ascii"))

                    logging.info("frame transferred saved to %s" % recv_msg)
                except socket.timeout:
                    logging.info("socket timeout")
                    break
        ###########

        # transferred_frame = []

        # for i in range(0,410):
        #     o_i = '/sharedvol/27c7a16c-55bd-46b0-903a-3333a754e872_{}_out.jpg'.format(i)

        #     transferred_frame.append(o_i)

        frame_data_array = []

        for filename in transferred_frame:
            img = cv2.imread(filename)
            height, width, _ = img.shape
            size = (width, height)
            frame_data_array.append(img)

        # out_video_path = "/sharedvol/adf30332-e84c-4cb1-9571-098b53f7a40a_video_out.avi"
        out_video_path = "/sharedvol/{}_video_out.avi".format(uuid)

        # filename, encoder, fps, framesize, [isColor]
        out_video = cv2.VideoWriter(
            out_video_path,
            # cv2.VideoWriter_fourcc(*"DIVX"),
            cv2.VideoWriter_fourcc(*"XVID"),
            fps,
            size,
        )

        for frame in frame_data_array:
            out_video.write(frame)

        cv2.destroyAllWindows()
        out_video.release()

        cls.send_message(uuid, client, json.dumps({"video": out_video_path}))

        send_email_with_video(email, out_video_path)

        logging.info("send out video path %s" % out_video_path)


if __name__ == "__main__":

    port = 4601

    app = tornado.web.Application(
        [
            (r"/", MainHandler),
            (r"/health", HealthHandler),
            (r"/cartoongan", CartoonGANHandler),
        ],
        debug=True,
        autoreload=True,
    )

    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(port)

    logger.info("listening on port " + str(port))

    tornado.ioloop.IOLoop.current().start()
