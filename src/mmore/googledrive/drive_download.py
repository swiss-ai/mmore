import io
import os
import shutil
from typing import List
from urllib.parse import parse_qs, urlparse

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


class GoogleDriveDownloader:
    download_dir = "src/mmore/googledrive/downloads"
    client_secrets = "src/mmore/googledrive/client_secrets.json"

    def __init__(self, ids: List[str]):
        self.ids = ids
        scopes = ['https://www.googleapis.com/auth/drive.readonly']


        credentials = service_account.Credentials.from_service_account_file(
            self.client_secrets, scopes=scopes)
        self.drive_service = build('drive', 'v3', credentials=credentials)

    def download_all(self):
        for ggid in self.ids:
            self.download_folder_recursive(ggid)

    def download_folder_recursive(self, ggid):
        items = self.list_files_in_folder(ggid)
        if not items:
            print("No files found in folder.")
            return

        for item in items:
            item_name = item['name']
            item_id = item['id']
            item_mime = item['mimeType']

            if item_mime == 'application/vnd.google-apps.folder':
                # recurse into subfolder
                self.download_folder_recursive(item_id)
            else:
                print(f"Downloading file: {item_name}")
                path, folder = self.download_file(item_id, item_name, item_mime)
                if path:
                    print(f"Saved to {path} (folder: {folder})")

    def download_file(self, file_id, file_name, mime_type):
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
            request = self.drive_service.files().export_media(fileId=file_id, mimeType=export_mime)
        else:
            ext = os.path.splitext(file_name)[1].lower() or '.bin'
            request = self.drive_service.files().get_media(fileId=file_id)

        # Create folder for this extension
        folder_name = ext.lstrip('.') if ext else 'unknown'
        download_dir = os.path.join(self.download_dir, folder_name)
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

    def list_files_in_folder(self, folder_id):
        query = f"'{folder_id}' in parents and trashed=false"
        results = self.drive_service.files().list(
            q=query,
            fields="nextPageToken, files(id, name, mimeType)").execute()
        return results.get('files', [])

    def remove_downloads(self):
        if os.path.exists(self.download_dir):
            shutil.rmtree(self.download_dir)



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
