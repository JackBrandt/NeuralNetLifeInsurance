# Import smtplib for the actual sending function
import smtplib
# Import the email modules we'll need
from email.message import EmailMessage

def send_email(address, subject, msg):
    '''
    Sends an email to a specified address with a given subject and message. This function uses Gmail's SMTP server to send the email, leveraging SSL for a secure connection.

    Parameters:
        address (str): The email address of the recipient.
        subject (str): The subject line of the email.
        msg (str): The body content of the email.

    Returns:
        None: This function does not return any value.

    Side Effects:
        - Asks the user to input the email app password during execution.
        - Sends an email using the provided credentials and inputs.

    Usage:
        >>> send_email('example@example.com', 'Greetings', 'Hello, this is a test email from NeuralNetLife!')

    Note:
        - It's important to ensure that 'Less secure app access' is enabled in your Gmail account settings or use an app password if 2FA is enabled.
        - The function assumes the sender's email is hardcoded as 'neuralnetlife@gmail.com'. Adjust this accordingly.
        - The password input step will pause the execution and wait for user input, which might not be suitable for automated scripts without modification.
    '''
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
        password = input("Enter your email APP password: ")
        s.login(me, password)

        # Send the message
        s.send_message(email)

if __name__ == '__main__':
    send_email('jbrandt4@zagmail.gonzaga.edu','Emailer Test','This is a test of the emailer.py')