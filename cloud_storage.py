# Imports the Google Cloud client library
from google.cloud import storage
import csv
import os
import ast
#import streamlit as st




class bucket_csv_object():
    def __init__(self,bucket_name='bucket-quickstart_snappy-rainfall-454116-t5',csv_name='users.csv'):
        '''
        Initializes an instance of the class, setting up the necessary properties and resources to interact with a CSV file stored in a Google Cloud Storage bucket. This method configures the storage client, identifies the bucket, and specifies the blob (CSV file) to be accessed.

        Parameters:
            bucket_name (str, optional): The name of the Google Cloud Storage bucket where the CSV file is stored. Defaults to 'bucket-quickstart_snappy-rainfall-454116-t5'.
            csv_name (str, optional): The name of the CSV file within the specified bucket. Defaults to 'users.csv'.

        Returns:
            None: The constructor does not return any value.

        Usage:
            >>> storage_instance = ClassName()  # Using default bucket and CSV file
            >>> storage_instance = ClassName(bucket_name='my-bucket', csv_name='data.csv')  # Specifying custom bucket and CSV file

        Note:
            - Ensure that Google Cloud Storage client libraries are installed and that the environment is authenticated to interact with Google Cloud services.
            - The credentials used must have sufficient permissions to access the specified bucket and files within it.
            - This constructor initializes the connection to Google Cloud Storage and prepares the object to perform further operations such as reading or writing data to the CSV file.
        '''
        self.bucket_name=bucket_name
        self.csv_name=csv_name
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
        self.csv_file = self.bucket.blob(csv_name)
    def refresh_file(self):
        '''
        Refreshes the connections to the Google Cloud Storage client, bucket, and the CSV file. This method is used to update the instance properties related to the cloud storage resources, ensuring that the latest configuration and file references are used. It is particularly useful if the bucket or file properties change during the runtime of the application or if there is a need to re-establish a lost connection.

        Parameters:
            None: This method does not take any arguments.

        Returns:
            None: This method does not return any value. It updates the internal state of the instance concerning the cloud storage client, bucket, and CSV file.

        Usage:
            >>> instance_of_class.refresh_file()
            # This call will refresh the cloud storage references in the instance, making sure that it points to the current blob in the specified bucket.

        Note:
            - This function should be used if there are changes in the bucket or CSV file settings, or if the initial setup might have failed or become outdated.
            - Ensure that the credentials used to create the `storage.Client` have ongoing access permissions to the specified bucket and file. Permissions issues may cause this method to fail in refreshing the references.
        '''
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.bucket_name)
        self.csv_file = self.bucket.blob(self.csv_name)



    def read_all(self):
        '''
        Reads all the data from the specified CSV file in the Google Cloud Storage bucket and returns it as a list. This method opens the CSV file for reading, iterates through all rows in the file, and collects them into a list.

        Parameters:
            None: This method does not take any arguments.

        Returns:
            list: A list of lists, where each sublist represents a row in the CSV file. Each element in the sublist represents a field in that row.

        Usage:
            >>> instance_of_class = ClassName()
            >>> content = instance_of_class.read_all()
            >>> for row in content:
            ...     print(row)
            # This will print each row from the CSV file.

        Note:
            - This function assumes that the CSV file is properly formatted with comma delimiters.
            - Ensure that the instance has been correctly initialized and that the CSV file reference (`self.csv_file`) is correctly set up and accessible.
            - The method handles the file opening in read mode, which should be considered if file access permissions or states are managed elsewhere in your application.
            - If the file is very large, consider the memory implications of loading the entire file content into a list.
        '''
        content=[]
        with self.csv_file.open('r',newline='') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                #print(row)
                if row!=[]:
                    content.append(row)

        return content
    def read_by_key(self,key):
        '''
        Searches for and retrieves rows from a CSV file in Google Cloud Storage where the first column matches a specified key. This method opens the CSV file, reads through each row, and collects rows that match the given key into a list.

        Parameters:
            key (str): The value to match against the first column of each row in the CSV file.

        Returns:
            list or None: Returns a list of rows where the first column matches the key. Each row is represented as a list of fields. If no matching rows are found, returns None.

        Usage:
            >>> instance_of_class = ClassName()
            >>> matched_rows = instance_of_class.read_by_key('specific_key')
            >>> if matched_rows:
            ...     for row in matched_rows:
            ...         print(row)
            ... else:
            ...     print("No matching rows found")
            # This will print each row that matches 'specific_key' or a message if no matches are found.

        Note:
            - The method assumes the CSV file uses comma delimiters.
            - This function reads the entire file and may not be efficient for very large files or for systems where rapid access to data is needed.
            - The key comparison is case-sensitive and must match exactly what is stored in the CSV file.
            - Ensure that the instance has been correctly initialized and that the CSV file reference (`self.csv_file`) is correctly set up and accessible.
            - Consider handling data indexing or more complex query needs if performance is critical or if the dataset is large.
        '''
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
        '''
        Checks if any row in the CSV file contains a specified key in the first column. This method uses the `read_by_key` function to search through the CSV file and determines if there is at least one matching row.

        Parameters:
            key (str): The key to check for existence in the first column of the CSV file rows.

        Returns:
            bool: Returns True if the key exists in at least one row, otherwise returns False.

        Usage:
            >>> instance_of_class = ClassName()
            >>> exists = instance_of_class.check_key_existence('search_key')
            >>> print(f'Key exists: {exists}')
            # This will print 'True' if the key 'search_key' exists in any row, and 'False' otherwise.

        Note:
            - This method depends on `read_by_key` to retrieve rows matching the key.
            - It is efficient in terms of not requiring to read the entire file if a match is found early; however, the entire file may still be scanned if the key does not exist.
            - Key matching is case-sensitive and requires an exact match.
            - Ensure that the CSV file is correctly set up and accessible in the Google Cloud Storage bucket as configured in the class instance.
            - This method is useful for quick checks of data presence and can be used for validations or conditional operations based on data existence.
        '''
        if self.read_by_key(key) is not None:
            return True
        else:
            return False
    def write_row(self,new_row):
        '''
        Inserts a new row into the CSV file or updates an existing row if the key (first element of the row) already exists. This method first checks for the existence of the key in the CSV file. If the key exists, it updates the corresponding row; if not, it appends the new row to the file.

        Parameters:
            new_row (list): A list representing the new row to be written to the CSV file. The first element in the list is considered the key for identifying duplicate entries.

        Returns:
            None: This method does not return any value. It modifies the CSV file directly by either adding a new row or updating an existing one.

        Usage:
            >>> instance_of_class = ClassName()
            >>> instance_of_class.write_row(['key123', 'data1', 'data2'])
            # This call will insert a new row or update the existing row where the first column matches 'key123'.

        Note:
            - Before writing to the file, the method checks for the existence of the key using `check_key_existence` to determine whether to update or append.
            - The entire content of the file is read into memory, which may not be efficient for very large files.
            - Changes are written back to the file by overwriting the existing content, and the file connection is refreshed after writing to ensure the latest state is accessible.
            - This method handles writing operations and refreshes the CSV file connection to ensure consistency.
        '''
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
        '''
        Removes a row from the CSV file where the first column matches the specified key. This method checks if the key exists in the file, reads all content, filters out the row with the matching key, and writes back the remaining content to the file.

        Parameters:
            key (str): The key used to identify the row to be deleted. The key is expected to match exactly with the first column of a row in the CSV file.

        Returns:
            None: This method does not return any value. It modifies the CSV file directly by removing the specified row.

        Usage:
            >>> instance_of_class = ClassName()
            >>> instance_of_class.delete_row('key123')
            # This call will delete the row where the first column matches 'key123'.

        Note:
            - If the key does not exist in the file, the function performs no deletion and the file remains unchanged.
            - The entire content of the CSV file is read into memory and then written back minus the deleted row, which may not be efficient for very large files.
            - It is important that the file connection is refreshed after writing to ensure the changes are properly reflected in subsequent operations.
            - This method ensures that only one row is deleted at a time, even if multiple rows have the same key. To handle multiple deletions, additional functionality would need to be implemented.
        '''
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
    '''
    Retrieves user data from a CSV file based on the provided email address. If the email does not exist in the file, it initializes a new row with the email as a key and saves it to the file. This method is typically used to ensure that user data is readily available for further processing.

    Parameters:
        email (str): The email address used as the key to search for the user data in the CSV file.

    Returns:
        list: Returns a list representing the user's data. If the user does not exist, returns a list containing just the email address, signifying that a new entry was created.

    Usage:
        >>> user_email = 'example@example.com'
        >>> user_data = load_user_data(user_email)
        >>> print(user_data)
        # This will print the user data associated with 'example@example.com', or a list containing only the email if it was newly created.

    Note:
        - This function interacts with a CSV file through a `bucket_csv_object` instance, which should be defined elsewhere in your codebase and properly configured to handle reading and writing to a specific CSV file in a cloud storage bucket.
        - The function uses `read_by_key` to attempt to find existing user data and `write_row` to add a new user if the email is not found.
        - Ensure that appropriate permissions and configurations are set up for accessing and modifying the CSV file in the storage bucket.
        - This method could be expanded to handle more complex user data structures or multiple fields by modifying the initialization of new user data.
    '''
    if email=='' or email==None:
        return
    bucket=bucket_csv_object()
    user_data=bucket.read_by_key(email)
    if user_data==None:
        user_data=[email,[None,None,None,None,None,None,None,None,
                          None,None,None,None,None,None,None,None,
                          None,None,None,None,None,None,None,None]]
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
    '''
    Updates user data in a CSV file by writing the provided data to the file and then clears the local variable holding the user data. This function is typically used to save changes to user data persistently and then clean up memory.

    Parameters:
        user_data (list): A list representing the user's data to be updated in the CSV file. The first element in the list should ideally be a unique identifier, such as an email address.

    Returns:
        None: This method does not return any value. It updates the CSV file and nullifies the local user data variable.

    Usage:
        >>> user_data = ['example@example.com', 'Name', 'Last Login Date']
        >>> unload_user_data(user_data)
        # This call will update the user data in the CSV and clear the user_data variable.

    Note:
        - The function assumes that `bucket_csv_object` is defined elsewhere in your codebase and is configured to interact with the CSV file in a storage bucket.
        - It is crucial to ensure that the data passed to this function is correctly formatted and complete as it will overwrite the existing data for the corresponding user in the file.
        - The `write_row` method of the `bucket_csv_object` should handle any exceptions related to file writing or data formatting.
        - Nullifying the local variable within the function may not clear it from the calling scope, depending on how the variable is managed in the broader application.
    '''
    bucket=bucket_csv_object()
    if user_data[0]=='' or user_data[0]==None:
        return
    bucket.write_row(user_data)
    return None

def write_user_data(updated_data):
    '''
    Writes or updates a user's data in a CSV file stored in a cloud storage bucket. This function uses an instance of `bucket_csv_object` to handle the CSV file operations, ensuring that the updated user data is persisted in the file.

    Parameters:
        updated_data (list): A list containing the user's data to be written to the CSV file. The data should include a unique identifier (such as an email address) as the first element, which is used to identify the user's row for updating or for appending as a new entry if it does not exist.

    Returns:
        None: This function does not return any value. It directly modifies the CSV file by writing the provided user data to it.

    Usage:
        >>> updated_data = ['user@example.com', 'User Name', 'Other Data']
        >>> write_user_data(updated_data)
        # This will update or add the user's data in the CSV file depending on whether the email 'user@example.com' exists.

    Note:
        - The function depends on `bucket_csv_object`, which should be appropriately configured to interact with the CSV file in the specified cloud storage bucket.
        - It's crucial that the CSV file and the bucket are accessible with the correct permissions set for writing data.
        - The `write_row` method of `bucket_csv_object` handles the determination of whether to update an existing row or append a new row based on the first element of the `updated_data` list.
        - Consider implementing error handling within `bucket_csv_object` to manage potential issues during file operations, such as network failures or permission errors.
    '''
    if updated_data[0]=='' or updated_data[0]==None:
        return
    bucket=bucket_csv_object()
    bucket.write_row(updated_data)

def update_user_data_item(email,item_index,item):
    if email=='' or email==None:
        return
    '''
    Updates a specific item in a user's data row in a CSV file. The function first loads the user's data based on the provided email, modifies the data at the specified index, writes the updated data back to the CSV file, and then returns the updated data.

    Parameters:
        email (str): The email address used to identify the user's data row in the CSV file.
        item_index (int): The index of the item within the user data list that needs to be updated.
        item (str): The new value to be updated at the specified index.

    Returns:
        list: The updated user data list after modifying the specified item.

    Usage:
        >>> updated_data = update_user_data_item('user@example.com', 2, 'Updated Data')
        >>> print(updated_data)
        # This will update the third item (index 2) of the user data for 'user@example.com' in the CSV file and print the updated data.

    Note:
        - This function uses `load_user_data` to retrieve the existing data and `write_user_data` to save the updated data back to the file.
        - The function assumes that the email uniquely identifies a user's row within the CSV file.
        - If the specified index is out of the range of the user data list, this will result in an IndexError. Ensure the index is valid for the length of the user data list.
        - It is crucial to handle data integrity and ensure that the updates are appropriately managed, especially in environments where multiple accesses might occur concurrently.
    '''
    user_data=load_user_data(email)
    try:
        if len(user_data)>item_index:
            user_data[item_index]=item
        else:
            for _ in range(item_index+1-len(user_data)):
                user_data.append(None)
            user_data[item_index]=item
    except IndexError:
        if item_index>=0:
            user_data.append(item)
    write_user_data(user_data)
    return user_data

def get_all_user_data():
    bucket=bucket_csv_object()
    return bucket.read_all()

def get_all_users():
    data=get_all_user_data()
    list_of_users=[]
    for user in data:
        list_of_users.append(user[0])
    return list_of_users

def get_username(email):
    data = load_user_data(email)
    print(f'Data from get_username: {data}')
    try:
        username=data[3]
    except:
        username=None
    return username

def set_username(email,new_usernmae):
    update_user_data_item(email,3,new_usernmae)

def get_friends(email):
    if email==None:
        return
    data=load_user_data(email)
    try:
        friends=convert_string_to_list(data[4])
    except IndexError:
        friends=None
    return friends

def send_friend_request(sender_email, receipient_email):
    try:
        requests=load_user_data(receipient_email)[0][5]
        if requests is None:
            requests=[sender_email]
        else:
            requests.append(sender_email)
        update_user_data_item(receipient_email,5,requests)
    except IndexError:
        update_user_data_item(receipient_email,5,[sender_email])

def get_friend_requests(email):
    data=load_user_data(email)
    try:
        requests=convert_string_to_list(data[5])
    except IndexError:
        requests=None
    return requests


def set_friend(email,new_friend):
    try:
        friends=load_user_data(email)[4]
    except IndexError:
        friends=None
    print(friends)
    try:
        if friends is None:
            friends=[new_friend]
        elif friends == '' or friends == 'None':
            friends=[new_friend]
        else:
            friends.append(new_friend)
    except AttributeError:
        friends=[friends,new_friend]
    update_user_data_item(email,4,friends)


def convert_string_to_list(string):
    """
    Convert a string representation of a list into an actual Python list.

    Parameters:
        string (str): A string that represents a Python list. Example: "['jbai@zagmail.gonzaga.edu']"

    Returns:
        list: The resulting Python list.

    Raises:
        ValueError: If the string cannot be safely converted to a list.
    """
    try:
        # Safely evaluate the string expression to a Python object
        result = ast.literal_eval(string)

        # Verify that the result is indeed a list
        if isinstance(result, list):
            return result
        else:
            raise ValueError("Input string does not evaluate to a list")
    except (SyntaxError, ValueError) as e:
        raise ValueError("Failed to convert string to list. Please check the input format.") from e

if __name__ == '__main__':
    bucket=bucket_csv_object()
    print(bucket.read_all())
    bucket.wipe_data()
    print(bucket.read_all())

    print(f'Jack\'s current friends: {get_friends('superengineerdude@gmail.com')}.')
    set_friend('superengineerdude@gmail.com','jbrandt4@zagmail.gonzaga')
    print(f'Jack\'s current friends: {get_friends('superengineerdude@gmail.com')}.')
    print(bucket.read_all())
