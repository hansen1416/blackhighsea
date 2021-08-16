
#!/usr/bin/env python3
import sys
import logging
import socket

log_format = "%(levelname)s %(asctime)s - %(message)s"

logging.basicConfig(stream = sys.stdout, format = log_format, level = logging.INFO)
logger = logging.getLogger()

HOST = ""  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        logger.info("Connected by")
        while True:
            data = conn.recv(1024)
            # if not data:
            #     break
            conn.sendall(data)


# #!/usr/bin/env python3

# import logging
# import sys
# import socket
# import selectors
# import traceback

# import libserver

# log_format = "%(levelname)s %(asctime)s - %(message)s"

# logging.basicConfig(stream = sys.stdout, format = log_format, level = logging.INFO)
# logger = logging.getLogger()

# sel = selectors.DefaultSelector()

# def accept_wrapper(sock):
#     conn, addr = sock.accept()  # Should be ready to read
#     print("accepted connection from", addr)
#     conn.setblocking(False)
#     message = libserver.Message(sel, conn, addr)
#     sel.register(conn, selectors.EVENT_READ, data=message)

# # if len(sys.argv) != 3:
# #     print("usage:", sys.argv[0], "<host> <port>")
# #     sys.exit(1)

# # host, port = sys.argv[1], int(sys.argv[2])
# host, port = '', int(sys.argv[1])

# logger.info("bind on port " + str(sys.argv[1]))

# lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# # Avoid bind() exception: OSError: [Errno 48] Address already in use
# lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# lsock.bind((host, port))
# lsock.listen()

# logger.info("listening on:" + str(host) + " " + str(port))

# lsock.setblocking(False)
# sel.register(lsock, selectors.EVENT_READ, data=None)

# try:
#     while True:
#         events = sel.select(timeout=None)
#         for key, mask in events:
#             if key.data is None:
#                 accept_wrapper(key.fileobj)
#             else:
#                 message = key.data

#                 logger.info("message " + str(message))

#                 try:
#                     message.process_events(mask)
#                 except Exception:
#                     logger.info(
#                         "main: error: exception for " + 
#                         f"{message.addr}:\n{traceback.format_exc()}"
#                     )
#                     message.close()
# except KeyboardInterrupt:
#     print("caught keyboard interrupt, exiting")
# finally:
#     sel.close()