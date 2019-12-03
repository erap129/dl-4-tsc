from io import BytesIO

from openpyxl import load_workbook

from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth
import os


def connect_to_gdrive():
    gauth = GoogleAuth()
    # Try to load saved client credentials
    gauth.LoadCredentialsFile(f"{os.path.dirname(os.path.abspath(__file__))}/mycreds.txt")
    if gauth.credentials is None:
        # Authenticate if they're not there
        gauth.CommandLineAuth()
    elif gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()
    # Save the current credentials to a file
    gauth.SaveCredentialsFile(f"{os.path.dirname(os.path.abspath(__file__))}/mycreds.txt")
    drive = GoogleDrive(gauth)
    return drive


def get_file_from_path(path):
    path_parts = path.split('/')
    drive = connect_to_gdrive()
    folder_id = 'root'
    for folder_name in path_parts[:-1]:
        folder_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
        folder = folder_list[[f['title'] for f in folder_list].index(folder_name)]
        folder_id = folder['id']
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    file = file_list[[f['title'] for f in file_list].index(path_parts[-1])]
    file.GetContentFile(file['title'])
    return file.content


def save_file_to_path(path):
    path_parts = path.split('/')
    filename = path_parts[-1]
    drive = connect_to_gdrive()
    folder_id = 'root'
    for folder_name in path_parts[:-1]:
        folder_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
        folder = folder_list[[f['title'] for f in folder_list].index(folder_name)]
        folder_id = folder['id']
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    file = file_list[[f['title'] for f in file_list].index(path_parts[-1])]
    file.SetContentFile(filename)
    file.Upload()


def upload_exp_results_to_gdrive(results_line, path):
    file = get_file_from_path(path)
    wb = load_workbook(filename=BytesIO(file.read()))
    wb.active = 0
    results = wb.active
    results.append(results_line)
    wb.save(filename=path.split('/')[-1])
    save_file_to_path(path)
    os.remove(path.split('/')[-1])
