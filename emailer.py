# Import smtplib for the actual sending function
import smtplib
# Import the email modules we'll need
from email.message import EmailMessage
from google.cloud import secretmanager

def get_email_password():
    pass

def send_email(address, subject, msg, password=None):
    email = EmailMessage()
    email.set_content(msg)
    me='neuralnetlife@gmail.com'

    email['Subject'] = subject
    email['From'] = me
    email['To'] = address

    # Gmail SMTP settings
    smtp_server = 'smtp.gmail.com'
    port = 465  # SSL port

    # Create secure SSL connection
    with smtplib.SMTP_SSL(smtp_server, port) as s:
        # Login with app password
        if password is None:
            password = input("Enter your email APP password: ")
        s.login(me, password)

        # Send the message
        s.send_message(email)

if __name__ == '__main__':
    send_email('jbrandt4@zagmail.gonzaga.edu','Emailer Test','This is a test of the emailer.py')