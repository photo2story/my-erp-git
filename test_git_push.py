import os
import base64
import requests
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# .env 파일에서 Personal Access Token 가져오기
GITHUB_TOKEN = os.getenv("PERSONAL_ACCESS_TOKEN")

# 테스트를 위한 함수
def test_upload_gitignore_to_github():
    # 현재 작업 디렉토리의 .gitignore 파일 경로 설정
    root_dir = os.getcwd()
    gitignore_path = os.path.join(root_dir, '.gitignore')

    # GitHub 저장소 정보
    repo_name = "photo2story/my-flask-app"  # 퍼블릭으로 만든 GitHub 저장소 경로
    destination_path = "static/images/.gitignore"  # 업로드할 경로

    if not os.path.exists(gitignore_path):
        print(f"File not found: {gitignore_path}")
        return

    # 파일을 GitHub 저장소에 업로드하는 함수 (토큰 사용)
    with open(gitignore_path, 'rb') as file:
        content = file.read()

    base64_content = base64.b64encode(content).decode('utf-8')

    url = f'https://api.github.com/repos/{repo_name}/contents/{destination_path}'

    data = {
        "message": "Add .gitignore",
        "content": base64_content
    }

    # 토큰을 사용하여 요청 보냄
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}"
    }

    response = requests.put(url, json=data, headers=headers)

    # 응답 상태 코드에 따라 처리
    if response.status_code == 201:
        print(f'Successfully uploaded {gitignore_path} to GitHub')
    else:
        print(f'Error uploading {gitignore_path} to GitHub: {response.status_code}, {response.text}')
        if response.status_code == 401:
            print("Authentication failed. Please check your GitHub token.")
        elif response.status_code == 403:
            print("Access denied or rate limit exceeded. Check your GitHub token permissions or API call limits.")
        elif response.status_code == 404:
            print("Repository or file path not found. Check your repository name or destination path.")
        else:
            print("An unknown error occurred during the upload.")

# 실행
test_upload_gitignore_to_github()

# python test_git_push.py
