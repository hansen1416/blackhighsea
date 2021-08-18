import tornado.web
import tornado.ioloop

class basicHandler(tornado.web.RequestHandler):
    def get(self):
        self.write('hi')

if __name__ == "__main__":
    app = tornado.web.Application([
        (r"/", basicHandler)
    ])

    app.listen(4601)
    print("listening on 4061")
    tornado.ioloop.IOLoop.current().start()