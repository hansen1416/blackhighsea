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
import random
import string
import subprocess

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

# import cartoongan

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

    # server.connect("smtp.gmail.com", 465)

    # username = "badapplesweetie@gmail.com"
    # password = os.environ.get("SMTP_PASSWORD")

    try:
        server.login(username, password)

        server.sendmail("badapplesweetie@gmail.com", [to], msg.as_string())

        logging.info('email sent to {}'.format(to))
    except Exception as e:
        logging.info('Exception {}'.format(str(e)))
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

    model_path = os.path.join("/opt", "gan-generator.pt")

    proc = subprocess.Popen(["/app/cartoonGan", model_path, \
        os.path.join("/tmp", input_image), os.path.join("/tmp", output_image)])

    res = proc.wait()

    logging.info("cartoon gan result:" + str(res))

    # 阿里云账号AccessKey拥有所有API的访问权限，风险很高。强烈建议您创建并使用RAM用户进行API访问或日常运维，请登录RAM控制台创建RAM用户。
    auth = oss2.Auth(
        os.environ.get("ALI_ACCESS_ID"), os.environ.get("ALI_ACCESS_KEY")
    )
    # yourEndpoint填写Bucket所在地域对应的Endpoint
    # 填写Bucket名称。
    bucket = oss2.Bucket(auth, "oss-cn-hongkong.aliyuncs.com", "bhs-media")

    try:
        # 必须以二进制的方式打开文件。
        # 填写本地文件的完整路径。如果未指定本地路径，则默认从示例程序所属项目对应本地路径中上传文件。
        with open(os.path.join("/tmp", output_image), 'rb') as fileobj:
            # 填写Object完整路径。Object完整路径中不能包含Bucket名称。
            bucket.put_object(output_image, fileobj)

            logging.info("start upload " + str(type(fileobj)))

        # self.write("https://" + hostname + output_object)
    except Exception as e:  # work on python 3.x
        logging.info("upload image failed, " + str(e))


def cartoongan_video(input_video, email):

    cap = cv2.VideoCapture(input_video)

    fps = cap.get(cv2.CAP_PROP_FPS)

    logging.info("fps %d" % fps)

    ###########
    _, file_extension = os.path.splitext(input_video)

    random_name = (
        "".join(random.choices(string.ascii_letters + string.digits, k=8))
        + file_extension
    )

    frame_images = []

    n = 0

    while cap.isOpened():

        # Capture frame-by-frame
        success, frame = cap.read()

        if success == True:
            # Display the resulting frame

            frame_image = os.path.join('/tmp', "{}_{}.jpg".format(random_name, n))

            frame = resize_image(frame)
            # from ndarray back to bytes
            cv2.imwrite(frame_image, frame)
            frame_images.append(frame_image)

            logging.info("frame image saved to %s" % frame_image)

            n += 1
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    transferred_frames = []
    model_path = os.path.join("/opt", "gan-generator.pt")

    for n, frame_image in enumerate(frame_images):

        transferred_frame = os.path.join("/tmp", "cg_{}_{}.jpg".format(random_name, n))
        
        proc = subprocess.Popen(["/app/cartoonGan", model_path, \
        frame_image, transferred_frame])

        res = proc.wait()

        transferred_frames.append(transferred_frame)

        if res == 0:
            logging.info("frame image saved to %s" % transferred_frame)
        else:
            logging.info("transfer image failed %s" % frame_image)

    frame_data_array = []

    for frame_image in transferred_frames:
        img = cv2.imread(frame_image)
        height, width, _ = img.shape
        size = (width, height)
        frame_data_array.append(img)

    # out_video_path = "/sharedvol/adf30332-e84c-4cb1-9571-098b53f7a40a_video_out.avi"
    out_video_path = "/tmp/{}.avi".format(
        "".join(random.choices(string.ascii_letters + string.digits, k=8))
    )

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

    send_email_with_video(email, out_video_path)

    logging.info("send out video path %s" % out_video_path)

    # todo delete input_video, transferred_frame, out_video_path

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

        _, file_extension = os.path.splitext(media["filename"])

        random_name = (
            "".join(random.choices(string.ascii_letters + string.digits, k=8))
            + file_extension
        )

        hostname = "bhs-media.oss-cn-hongkong.aliyuncs.com/"

        content_type = media["content_type"]

        if content_type[0:5] == "video":
            email = self.get_body_argument("email", default="")

            if not email:
                logging.info("email is empty")
                return

            logging.info("email " + email)

            send_email_with_video(email, '/tmp/F7PduP4y.avi')
            return

            input_video = "/tmp/" + str(int(time())) + "_" + random_name

            video_bytesio = BytesIO(media["body"])

            with open(input_video, "wb") as outfile:
                # Copy the BytesIO stream to the output file
                outfile.write(video_bytesio.getbuffer())

            tornado.ioloop.IOLoop.current().spawn_callback(
                cartoongan_video, input_video, email
            )

        else:

            # from ndarray back to bytes
            input_iamge = str(time()) + random_name
            output_iamge = "cg_" + str(time()) + random_name

            # media['body'] is ninary, ndarray is np.ndarray
            ndarray = cv2.imdecode(
                np.frombuffer(media["body"], np.uint8), cv2.IMREAD_COLOR
            )
            # resize if the image is too big            
            cv2.imwrite(os.path.join("/tmp", input_iamge), resize_image(ndarray))

            # The IOLoop will catch the exception and print a stack trace in
            # the logs. Note that this doesn't look like a normal call, since
            # we pass the function object to be called by the IOLoop.
            tornado.ioloop.IOLoop.current().spawn_callback(
                cartoongan_image, input_iamge, output_iamge
            )

            self.write("https://" + hostname + output_iamge)

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
