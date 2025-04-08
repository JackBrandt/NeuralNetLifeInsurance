'''auto_emailer.py - Description:
    * Once a day, pulls list of users from server bucket.
    * Runs a what-if analysis to see what attributes would lower each users expected cost the most
    * Sends each user an email asking them to input their information and suggesting what would make the biggest difference in their premiums
'''
from emailer import send_email, get_email_password
from cloud_storage import get_all_user_data
from game_utils import get_yrs_left
from utils import dob_to_age
from neural_net import NeuralNet
import ast
import time

minute=60
hour=minute*60
day=hour*24

def unwrap_string(string):
    return string[string.index("'")+1:string.index("'",string.index("'")+1)]

def string_to_list(string):
    # Convert string to list
    string = string.replace('[','').replace(']','')
    string = string.split(', ')
    # Convert each element to float
    for i in range(len(string)):
        try:
            string[i] = float(string[i])
        except:
            pass
    print(string) # debug statement
    try:
        string=[2025-int(string[0][14:18])]+string[3:]
    except:
        pass
        # Unwrap each string
    for i in range(len(string)):
        try:
            string[i]=unwrap_string(string[i])
        except:
            pass
    return string

def user_list_of_dicts():
    list_dicts=[]
    for user in get_all_user_data():
        if len(user)<3:     #WARNING THIS DOESN'T CHECK WHAT IS MISSING AND JUST ASSUMES ITS HIGH SCORE, THIS IS A TE
            user.append(0)
        user[1]=string_to_list(user[1])
        #print(user[1]) # debug statement
        #print(user[1]) # debug statement
        list_dicts.append({'email':user[0],
                           'data':user[1],# ISSUE, IT DOESN"T STORE AGE...
                           'high score':user[2],
                           'subject':'',
                           'msg':'',
                           'biggest change':['',0,0] # What was changed,change in life expectancy, change in cost
                            })
    return list_dicts

def test_change(cur_life_expec,data,change):
    data_copy=data.copy()
    match change[1]:
        case 'n':
            data_copy[change[2]]='n'
        case _ if change[1]<-1:
            data_copy[change[2]]=data_copy[change[2]]+change[1]
        case _:
            data_copy[change[2]]=data_copy[change[2]]-change[1]*data_copy[change[2]]
    return get_yrs_left(data_copy,contains_name=False,model_print_statement=False)-cur_life_expec

changeables = [['loose weight',-.05,1],#what it is, new value, index
    ['gain weight', .05,1],
    ['blood pressure',-10,4],
    ['smoking','n',5],
    ['other nicotine usuage','n',6],
    ['occupational danger',-1,8],
    ['lifestyle danger',-1,9],    # Currently can go below 1, but shouldn't
    ['cannabis','n',10],
    ['opioids','n',11],
    ['other drugs','n',12],
    ['drinks a week',-1,13],
    ['addiction','n',14],
    ['diabetes','n',16],
    ['cholesterol',-20,18],
]

def find_best_change(user,user_index):
    if user['data']==['']:
        users[user_index]['subject']='Please fill out your data'
        users[user_index]['msg']='This is a test... Please fill out your data'
        return None
    # Else
    cur_life_expec=get_yrs_left(user['data'],contains_name=False,model_print_statement=False)
    best_change=([None,None,None],0)
    for change in changeables:
        result=test_change(cur_life_expec,user['data'],change)
        print(result)
        if result>best_change[1]:
            best_change=(change,result)
    return best_change

def get_subject(user,best_change):
    if best_change==None:
        return 'This is a test'
    return 'Wondering how to lower your insurance premiums?'

def change_specific_message_part(best_change):
    match best_change[0][0]:
        case 'loosing_weight':
            return 'loosing 5%% of your body weight. '
        case 'gain weight':
            return 'gaining 5%% of your body weight. '
        case 'blood pressure':
            return 'lowering your blood pressure by 10 points. '
        case 'smoking':
            return 'stopping smoking. '
        case 'other nicotine usuage':
            return 'stopping nicotine usuage. '
        case 'occupational danger':
            return 'lowering your your occupational hazard level by 1. '
        case 'lifestyle hazard':
            return 'lowering your your lifestyle hazard level by 1. '
        case 'cannabis':
            return 'eliminating cannabis usuage. '
        case 'opioids':
            return 'ceasing opioid usuage. '
        case 'other drugs':
            return 'stopping drug usuage. '
        case 'drinks a week':
            return 'lowering your drinks a week by 1. '
        case 'addiction':
            return 'beating your addiction. '
        case 'diabetes':
            return 'reversing your diabetes. '
        case 'cholesterol':
            return 'reducing your cholesterol by 20 points. '
        case _:
            return 'If you see this I fxcked up my case statements somewhere... '

def get_msg(user,best_change):
    if best_change==None:
        return 'This is a test'
    msg='Hello, ' + user['email'] + '!\n\n'
    msg+='We here at Neural Net Life are always looking for ways to help you lower your insurance premiums and help increase your lifespance. '
    msg+=f'We noticed you could increase your lifespan by {best_change[1]:.1f} years by '
    msg+=change_specific_message_part(best_change)
    msg+='We recommend you make these changes so that you can live longer. And that you update your personal info online so that we can save you money on your life insurance.'
    msg+='\n\nHere\'s the link to the website: ...\nAnd, from all of us at Neural Net Life,\n thank you for being a member.'
    return msg

def whatif_analysis(users):
    for i,user in enumerate(users):
        best_change=find_best_change(user,i)
        users[i]['biggest change']=best_change
        print(users[i])
        users[i]['subject']=get_subject(user,best_change)
        users[i]['msg']=get_msg(user,best_change)
    return users

def email_user(user,email_password=None):
    if email_password is None:
        email_password=get_email_password()
    send_email(user['email'],user['subject'],user['msg'],email_password)

def email_users(users):
    email_password=get_email_password()
    for user in users:
        email_user(user,email_password)

while True:
    users=user_list_of_dicts()
    users=whatif_analysis(users)
    for user in users:
        print(user['email'])
        print(user['subject'])
        print(user['msg'])
    email_users(users)
    time.sleep(day)
