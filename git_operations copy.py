# git_operations.py

import git
import os
import base64
import requests
from dotenv import load_dotenv
import hashlib
import pandas as pd
from io import StringIO
import config
import re

# .env 파일 로드
load_dotenv()

GITHUB_API_URL = "https://api.github.com"
GITHUB_ERP_REPO = "photo2story/my-erp-git"  # ERP 데이터 저장소
GITHUB_BRANCH = "main"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    GITHUB_TOKEN = os.getenv("PERSONAL_ACCESS_TOKEN")
    
print(f"Using GitHub token: {GITHUB_TOKEN[:10]}...")  # 토큰의 처음 10자만 출력

# 저장소 경로를 현재 디렉토리로 설정
repo_path = os.path.dirname(os.path.abspath(__file__))
try:
    repo = git.Repo(repo_path)
except git.exc.InvalidGitRepositoryError:
    print(f'Invalid Git repository at path: {repo_path}')
    repo = None

def calculate_file_sha(file_path):
    """파일의 SHA 해시를 계산합니다."""
    sha_hash = hashlib.sha1()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha_hash.update(chunk)
    return sha_hash.hexdigest()

async def move_files_to_github(file_path):
    """파일을 GitHub 저장소의 적절한 위치로 이동합니다."""
    print(f"move_files_to_github called for {file_path}")
    filename = os.path.basename(file_path)
    
    # 파일 종류에 따른 GitHub 경로 결정
    if filename.endswith('.csv'):
        if filename.startswith(('erp_data_', 'contract_')):
            github_path = f"static/data/{filename}"
        else:
            github_path = f"static/results/{filename}"
    elif filename.endswith(('.png', '.jpg', '.jpeg')):
        # 이미지 파일의 경우 사업코드에서 연도를 추출하여 해당 연도 폴더에 저장
        if 'comparison_' in filename:
            # 정규식을 사용하여 사업코드에서 연도 추출
            # [A-Z]: 영문자 한 글자, \d{4}: 4자리 숫자
            match = re.search(r'comparison_[A-Z](\d{4})', filename)
            if match:
                year = match.group(1)  # 첫 번째 그룹(4자리 연도) 추출
                github_path = f"static/images/{year}/{filename}"
            else:
                github_path = f"static/images/{filename}"
        else:
            github_path = f"static/images/{filename}"
    else:
        print(f"Unsupported file type: {filename}")
        return
            
    # GitHub에서 파일 정보 확인
    url = f"{GITHUB_API_URL}/repos/{GITHUB_ERP_REPO}/contents/{github_path}"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    response = requests.get(url, headers=headers, params={"ref": GITHUB_BRANCH})
    
    try:
        if response.status_code == 200:
            file_data = response.json()
            remote_sha = file_data['sha']
            local_sha = calculate_file_sha(file_path)
            
            if remote_sha == local_sha:
                print(f"{filename} is up-to-date in GitHub, skipping upload.")
                return
            
            await upload_file_to_github(file_path, github_path, remote_sha)
        elif response.status_code == 404:
            print(f"{filename} does not exist in GitHub, proceeding to upload.")
            await upload_file_to_github(file_path, github_path)
        else:
            print(f"Error checking file in GitHub: {response.status_code}, {response.text}")
            return
        
        # 로컬 Git 작업
        if repo:
            try:
                repo.git.add(file_path)
                repo.index.commit(f'Update {filename} in ERP data repository')
                origin = repo.remote(name='origin')
                push_result = origin.push()
                
                if push_result[0].flags & push_result[0].ERROR:
                    print(f"Error pushing to GitHub: {push_result[0].summary}")
                else:
                    print(f'Successfully pushed {filename} to GitHub')
            except Exception as e:
                print(f'Error during Git operation: {e}')
    except Exception as e:
        print(f'Error processing {filename}: {e}')

async def upload_file_to_github(file_path, github_path, sha=None):
    """GitHub에 파일을 업로드합니다."""
    try:
        with open(file_path, 'rb') as file:
            content = file.read()

        base64_content = base64.b64encode(content).decode('utf-8')

        data = {
            "message": f"Update {os.path.basename(file_path)} in ERP repository",
            "content": base64_content,
            "branch": GITHUB_BRANCH
        }
        
        if sha:
            data["sha"] = sha

        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }

        url = f"{GITHUB_API_URL}/repos/{GITHUB_ERP_REPO}/contents/{github_path}"
        response = requests.put(url, json=data, headers=headers)

        if response.status_code in [200, 201]:
            print(f'Successfully uploaded {file_path} to GitHub')
        else:
            print(f'Error uploading {file_path}: {response.status_code}, {response.text}')
    except Exception as e:
        print(f'Error uploading {file_path}: {e}')

def read_data_from_github(github_path):
    """GitHub에서 데이터 파일을 읽어옵니다."""
    url = f"{GITHUB_API_URL}/repos/{GITHUB_ERP_REPO}/contents/{github_path}"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    try:
        response = requests.get(url, headers=headers, params={"ref": GITHUB_BRANCH})
        if response.status_code == 200:
            file_data = response.json()
            content = base64.b64decode(file_data['content']).decode('utf-8')
            
            if github_path.endswith('.csv'):
                return pd.read_csv(StringIO(content))
            return content
        else:
            print(f"Error fetching file: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        print(f"Error reading file from GitHub: {e}")
        return None

# 테스트 코드
if __name__ == "__main__":
    # 테스트할 파일 경로
    test_file = os.path.join(config.STATIC_IMAGES_PATH, 'yearly_comparison_C20240160.png')
    
    if os.path.exists(test_file):
        import asyncio
        asyncio.run(move_files_to_github(test_file))
        print("Test completed")
    else:
        print(f"Test file not found: {test_file}")

# python git_operations.py
