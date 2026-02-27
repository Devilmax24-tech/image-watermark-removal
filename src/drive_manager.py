import os
import io
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive']

class DriveManager:
    def __init__(self, credentials_path='credentials.json', token_path='token.json'):
        self.creds = None
        if os.path.exists(token_path):
            self.creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                if not os.path.exists(credentials_path):
                    raise FileNotFoundError(f"Please provide {credentials_path} from Google Cloud Console.")
                flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
                self.creds = flow.run_local_server(port=0)
            
            with open(token_path, 'w') as token:
                token.write(self.creds.to_json())

        self.service = build('drive', 'v3', credentials=self.creds)

    def extract_id_from_link(self, link):
        """Extracts folder ID from a Google Drive URL."""
        if 'drive.google.com' not in link:
            return link  # Assume it's already an ID
        
        # Handle /folders/ID
        if '/folders/' in link:
            return link.split('/folders/')[1].split('/')[0].split('?')[0]
        
        # Handle ?id=ID
        if 'id=' in link:
            return link.split('id=')[1].split('&')[0]
            
        return link

    def get_subfolder_id(self, parent_id, folder_name):
        """Finds a subfolder ID by name within a parent folder."""
        query = f"'{parent_id}' in parents and name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        results = self.service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get('files', [])
        if files:
            return files[0]['id']
        return None

    def create_folder(self, parent_id, folder_name):
        """Creates a new folder in Drive."""
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_id]
        }
        file = self.service.files().create(body=file_metadata, fields='id').execute()
        return file.get('id')

    def list_files(self, folder_id, page_size=1000):
        query = f"'{folder_id}' in parents and mimeType contains 'image/'"
        results = self.service.files().list(
            q=query, pageSize=page_size, fields="nextPageToken, files(id, name)").execute()
        return results.get('files', [])

    def download_file(self, file_id, destination_path):
        request = self.service.files().get_media(fileId=file_id)
        fh = io.FileIO(destination_path, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        return destination_path

    def upload_file(self, file_path, folder_id):
        file_metadata = {
            'name': os.path.basename(file_path),
            'parents': [folder_id]
        }
        media = MediaFileUpload(file_path, resumable=True)
        file = self.service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        return file.get('id')
