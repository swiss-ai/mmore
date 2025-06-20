import io
import os
from urllib.parse import parse_qs, urlparse

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_ACCOUNT_FILE = os.path.join(SCRIPT_DIR, 'client_secrets.json')


credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)

def get_folder_id_from_url(url):
    parsed = urlparse(url)
    path_parts = parsed.path.split('/')
    if 'folders' in path_parts:
        folder_index = path_parts.index('folders')
        return path_parts[folder_index + 1]
    query_params = parse_qs(parsed.query)
    if 'id' in query_params:
        return query_params['id'][0]
    raise ValueError("Could not extract folder ID from the URL")

def list_files_in_folder(folder_id):
    query = f"'{folder_id}' in parents and trashed=false"
    results = drive_service.files().list(
        q=query,
        fields="nextPageToken, files(id, name, mimeType)").execute()
    return results.get('files', [])

def download_file(file_id, file_name, mime_type):
    # Determine extension and export mime type if Google Docs editors file
    if mime_type.startswith('application/vnd.google-apps.'):
        if mime_type == 'application/vnd.google-apps.document':
            export_mime = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            ext = '.docx'
        elif mime_type == 'application/vnd.google-apps.spreadsheet':
            export_mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            ext = '.xlsx'
        elif mime_type == 'application/vnd.google-apps.presentation':
            export_mime = 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
            ext = '.pptx'
        else:
            print(f"Skipping unsupported Google Docs mime type: {mime_type} for file {file_name}")
            return None, None
        file_name += ext
        request = drive_service.files().export_media(fileId=file_id, mimeType=export_mime)
    else:
        ext = os.path.splitext(file_name)[1].lower() or '.bin'
        request = drive_service.files().get_media(fileId=file_id)

    # Create folder for this extension
    folder_name = ext.lstrip('.') if ext else 'unknown'
    download_dir = os.path.join(os.path.join(SCRIPT_DIR,'downloads'), folder_name)
    os.makedirs(download_dir, exist_ok=True)
    full_path = os.path.join(download_dir, os.path.basename(file_name))

    fh = io.FileIO(full_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Downloading {file_name} into {download_dir}: {int(status.progress() * 100)}%")
    fh.close()
    return full_path, folder_name

def download_folder_recursive(folder_id):
    items = list_files_in_folder(folder_id)
    if not items:
        print("No files found in folder.")
        return

    for item in items:
        item_name = item['name']
        item_id = item['id']
        item_mime = item['mimeType']

        if item_mime == 'application/vnd.google-apps.folder':
            # recurse into subfolder
            download_folder_recursive(item_id)
        else:
            print(f"Downloading file: {item_name}")
            path, folder = download_file(item_id, item_name, item_mime)
            if path:
                print(f"Saved to {path} (folder: {folder})")

if __name__ == "__main__":
    folder_link = input("Enter Google Drive folder URL: ").strip()
    folder_id = get_folder_id_from_url(folder_link)
    download_folder_recursive(folder_id)
    print("Done!")
