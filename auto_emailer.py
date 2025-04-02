'''auto_emailer.py - Description:
    * Once a day, pulls list of users from server bucket.
    * Runs a what-if analysis to see what attributes would lower each users expected cost the most
    * Sends each user an email asking them to input their information and suggesting what would make the biggest difference in their premiums
'''
from emailer import send_email, get_email_password
from cloud_storage import get_all_user_data
from game_utils import get_yrs_left
import time

def user_list_of_dicts():
    list_dicts=[]
    for user in get_all_user_data():
        if len(user)<3:     #WARNING THIS DOESN'T CHECK WHAT IS MISSING AND JUST ASSUMES ITS HIGH SCORE, THIS IS A TE
            user.append(0)
        list_dicts.append({'email':user[0],
                           'data':user[1],# ISSUE, IT DOESN"T STORE AGE...
                           'high score':user[2],
                           'subject':'',
                           'msg':'',
                           'biggest change':['',0,0] # What was changed,change in life expectancy, change in cost
                            })
    return list_dicts

def whatif(cur_life_expec,data,change):
    match change[1]=='n':
        case 'n':
            data[change[2]]='n'
        case _ if change[1]<-1:
            data[change[2]]=data[change[2]]+change[1]
        case _:
            data[change[2]]=data[change[2]]-change[1]*data[change[2]]
    return get_yrs_left(data)-cur_life_expec


def whatif_analysis(users):
    for user in users:
        cur_life_expec=get_yrs_left(user['data'])
        best_change=([None,None,None],0)
        changeables = [['weight loss',-.1,0],#what it is, new value, index
                        ['weight gain', .1,0],
                         ['sys_bp',-10,3],
                          ['smoker','n',4],
                           ['nic_other','n',5],
                            ['occup_danger',-1,7],
                             ['ls_danger',-1,8],
                              ['cannabis','n',9], #YOULL NEED TO ADD PLUS ONE TO ALL OF THESE INDICES BECAUSE INPUTS DOESN"T INCLUD AGE????
                               ['opioids','n',10],
                                ['other_drugs','n',11],
                                 ['drinks_a_week',-1,12],
                                  ['addiction','n',13],
                                   ['diabetes','n',15],
                                    ['cholesterol',-20,17],
                                     ]
        for change in changeables:
            result=whatif(cur_life_expec,user['data'],change)
            print(change,result)
            if result>best_change[1]:
                best_change=(change,result)

def email_users(users):
    email_password=get_email_password()
    for user in users:
        send_email(user,email_password)
        send_email(user['email'],user['subject'],user['msg'],)

while True:
    # 0. Get data
    users=user_list_of_dicts()
    print(users) # debug statement
    # Run what-if analysis
    users=whatif_analysis(users)
    # Email
    email_users(users)
    # Sleep for 1 day
    time.sleep(60*60*24)
