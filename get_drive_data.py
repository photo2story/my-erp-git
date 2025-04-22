import os
import io
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from oauth2client import file, client, tools
import pandas as pd

# Google Drive API 인증 및 서비스 생성
SCOPES = 'https://www.googleapis.com/auth/drive'
store = file.Storage('storage.json')
creds = store.get()
if not creds or creds.invalid:
    flow = client.flow_from_clientsecrets('client_secret_googledrive.json', SCOPES)
    creds = tools.run_flow(flow, store)
service = build('drive', 'v3', credentials=creds)

# 파일 업로드 함수
def upload_file_to_google_drive(file_path, folder_id):
    file_name = os.path.basename(file_path)
    file_metadata = {'name': file_name, 'parents': [folder_id]}
    media = MediaFileUpload(file_path, mimetype='text/csv')
    file_info = service.files().create(body=file_metadata, media_body=media, fields='id,webViewLink').execute()
    print("File uploaded successfully. WebViewLink:", file_info.get('webViewLink'))
    return file_info.get('id'), file_info.get('webViewLink')

# 파일 다운로드 함수
def download_file_from_google_drive(file_id, destination_path):
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(destination_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print(f'Download {int(status.progress() * 100)}%')
    fh.close()
    print("File downloaded successfully to", destination_path)

# 테스트 함수
def test_google_drive_operations():
    folder_id = '1pE4BYi91hGupnFTLI5b7-gr0liUs5Bqj'  # 구글 드라이브 폴더 ID
    upload_file_path = 'path/to/your/file.csv'  # 업로드할 파일 경로
    download_file_path = 'path/to/save/downloaded/file.csv'  # 다운로드할 파일 저장 경로

    # 파일 업로드
    file_id, web_view_link = upload_file_to_google_drive(upload_file_path, folder_id)

    # 파일 다운로드
    download_file_from_google_drive(file_id, download_file_path)

    # 다운로드한 파일 확인
    df = pd.read_csv(download_file_path)
    print(df.head())

if __name__ == "__main__":
    test_google_drive_operations()