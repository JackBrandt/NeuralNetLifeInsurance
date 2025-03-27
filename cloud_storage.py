# Imports the Google Cloud client library
from google.cloud import storage
import csv
import os
import ast
#import streamlit as st

class bucket_csv_object():
    def __init__(self,bucket_name='bucket-quickstart_snappy-rainfall-454116-t5',csv_name='users.csv'):
        # Verify
        self.bucket_name=bucket_name
        self.csv_name=csv_name
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
        self.csv_file = self.bucket.blob(csv_name)
    def refresh_file(self):
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.bucket_name)
        self.csv_file = self.bucket.blob(self.csv_name)
    def read_all(self):
        content=[]
        with self.csv_file.open('r',newline='') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                #print(row)
                if row!=[]:
                    content.append(row)
        return content
    def read_by_key(self,key):
        content=[]
        with self.csv_file.open('r',newline='') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                try:
                    if row[0]==key:
                        content.append(row)
                except:
                    pass
        if content==[]:
            content=None
        return content
    def check_key_existence(self,key):
        if self.read_by_key(key) is not None:
            return True
        else:
            return False
    def write_row(self,new_row):
        existence=self.check_key_existence(new_row[0])
        current_content=self.read_all()
        if existence == True:
            for i,row in enumerate(current_content):
                #print(f'{row[0]} vs {new_row[0]}')
                try:
                    if row[0]==new_row[0]:
                        current_content[i]=new_row # Have to do index of current_content cuz python's weird about ref. stuff
                except:
                    pass
        else:
            current_content.append(new_row)
        with self.csv_file.open('w',newline='') as f:
            writer = csv.writer(f,delimiter=',')
            #print(current_content)
            writer.writerows(current_content)
        self.refresh_file()
    def delete_row(self,key):
        existence=self.check_key_existence(key)
        current_content=self.read_all()
        new_content=[]
        if existence == True:
            for i,row in enumerate(current_content):
                #print(f'{row[0]} vs {new_row[0]}')
                if row[0]!=key:
                    new_content.append(row) # Have to do index of current_content cuz python's weird about ref. stuff
            with self.csv_file.open('w',newline='') as f:
                writer = csv.writer(f,delimiter=',')
                #print(current_content)
                writer.writerows(new_content)
                self.refresh_file()
    def wipe_data(self):
        confirmation=input("Do you really want to wipe all user data? (Y/n)")
        if confirmation=='Y':
            confirmation=input('Are you sure? (Y/n)')
            if confirmation=='Y':
                with self.csv_file.open('w',newline='') as f:
                    writer = csv.writer(f,delimiter=',')
                    writer.writerows([''])
                    self.refresh_file

def load_user_data(email):
    bucket=bucket_csv_object()
    user_data=bucket.read_by_key(email)
    if user_data==None:
        user_data=[email,[None,None,None,None,None,None,None,None,
                          None,None,None,None,None,None,None,None,
                          None,None,None,None,None,None,None]]
        bucket.write_row(user_data)
    else:
        user_data=user_data[0]
    try:
        user_data[1]=ast.literal_eval(user_data[1])
    except:
        pass
    #for i,value in enumerate(user_data[1]):
    #    if value=='None':
    #        user_data[1][i]=None
    try:
        user_data[2]=float(user_data[2])
    except:
        pass
    return user_data

def unload_user_data(user_data):
    bucket=bucket_csv_object()
    bucket.write_row(user_data)
    user_data=None

def write_user_data(updated_data):
    bucket=bucket_csv_object()
    bucket.write_row(updated_data)

def update_user_data_item(email,item_index,item):
    user_data=load_user_data(email)
    try:
        user_data[item_index]=item
    except IndexError:
        if item_index>=0:
            user_data.append(item)
    write_user_data(user_data)
    return user_data

if __name__ == '__main__':
    bucket=bucket_csv_object()
    '''print(bucket.read_all())
    print(bucket.check_key_existence('Key'))
    bucket.write_row(['Hello world','It is me'])
    bucket.write_row(['wut','wut'])
    print(bucket.read_all())
    bucket.delete_row('wut')
    print(bucket.read_by_key('wut'))
    bucket.delete_row('Hello world')'''
    bucket.wipe_data()
    #bucket.write_row(['wut','wut'])
    print(bucket.read_all())