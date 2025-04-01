'''auto_emailer.py - Description:
    * Once a day, pulls list of users from server bucket.
    * Runs a what-if analysis to see what attributes would lower each users expected cost the most
    * Sends each user an email asking them to input their information and suggesting what would make the biggest difference in their premiums
'''
from emailer import send_email

import time

t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
print("Current Time =", current_time)

while True:
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print("Current Time =", current_time)
    if current_time == '12:53:00':
        print("Target time:")
    sleep()
