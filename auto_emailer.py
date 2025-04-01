'''auto_emailer.py - Description:
    * Once a day, pulls list of users from server bucket.
    * Runs a what-if analysis to see what attributes would lower each users expected cost the most
    * Sends each user an email asking them to input their information and suggesting what would make the biggest difference in their premiums
'''
from emailer import send_email, get_email_password
from cloud_storage import get_all_user_data
import time

while True:
    # Get data
    users=get_all_user_data()
    print(users)
    # Run what-if analysis

    # Send email
    email_password=get_email_password()
    for user in users:
        send_email(user['address'],user['subject'],user['msg'],)
    # Sleep for 1 day
    time.sleep(60*60*24)
