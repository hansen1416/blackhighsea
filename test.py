import smtplib
import email.utils
from email.mime.text import MIMEText

# Create the message
msg = MIMEText("This is the body of the message.")
msg["To"] = email.utils.formataddr(("Recipient", "hansen1416@163.com"))
msg["From"] = email.utils.formataddr(("Author", "hansen1417@163.com"))
msg["Subject"] = "Simple test message"

server = smtplib.SMTP()

server.set_debuglevel(True)  # show communication with the server

server.connect("smtp-relay.sendinblue.com", 587)

username = 'badapplesweetie@gmail.com'
password = ''

server.login(username, password)

try:
    server.sendmail(
        "hansen1417@163.com", ["hansen1416@163.com"], msg.as_string()
    )
finally:
    server.quit()
