import os
import concurrent.futures
import asyncio
from fastapi import FastAPI, UploadFile, Depends, HTTPException, Header, Query
from fastapi.responses import JSONResponse
from main import main as document_ocr

app = FastAPI()

user_api_keys = {
    "user1": "apikey1",
    "user2": "apikey2",
    # Add more users and their API keys as needed
}

upload_dir = "uploads"
os.makedirs(upload_dir, exist_ok=True)

def remove_null_and_empty(data):
    if isinstance(data, list):
        return [remove_null_and_empty(item) for item in data if item not in (None, "")]
    elif isinstance(data, dict):
        return {key: remove_null_and_empty(value) for key, value in data.items() if value not in (None, "")}
    else:
        return data

def process_file(file: UploadFile, ID: str = None):
    file_path = None
    response_data = {}

    try:
        file_content = file.file.read()
        file_extension = os.path.splitext(file.filename)[1].lower()
        file_name = os.path.splitext(file.filename)[0]
        file_path = os.path.join(upload_dir, f"{file_name}{file_extension}")

        with open(file_path, "wb") as temp_file:
            temp_file.write(file_content)

        if not os.path.exists(file_path):
            response_data = {"error": "File not found"}
            return JSONResponse(content=response_data, status_code=500)

        # Process the file (e.g., perform OCR)
        result = document_ocr(file_path=file_path)
        # print(result)
        result = remove_null_and_empty(result)
        result["ID"] = ID
        return result

    except FileNotFoundError:
        response_data = {"error": "File not supported"}

    except OSError as e:
        if "No space left on device" in str(e):
            response_data = {"error": "No disk space left."}
        elif "CUDA out of memory" in str(e):
            response_data = {"error": "CUDA out of memory."}
        else:
            response_data = {"error": "OS ERROR"}

    except Exception as e:
        # Customize the error messages here
        if "object does not support item assignment" in str(e):
            response_data = {"error": "Unsupported Format"}
        elif "object has no attribute 'seek'" in str(e):
            response_data = {"error": "Unsupported Format"}
        else:
            response_data = {"error": "Error processing file."}

    finally:
        if file_path:
            try:
                os.remove(file_path)
            except Exception as e:
                response_data = {"error": "File delete error}"}

    # Return JSON response for all error conditions
    return JSONResponse(content=response_data, status_code=500)

async def get_api_key(api_key: str = Header(None, convert_underscores=False)):
    if api_key not in user_api_keys.values():
        raise HTTPException(status_code=401, detail={"error": "Invalid API key"})
    return api_key

@app.post("/ocr/")
async def ocr_endpoint(
    file: UploadFile,
    api_key: str = Depends(get_api_key),
    ID: str = Query(None, title="Optional ID parameter", description="Optional ID parameter for processing"),
):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: process_file(file, ID)
        )
    return result



import requests
import socket
host_name = socket.gethostname()
URDU_API_BASE_URL=socket.gethostbyname(host_name)
# Replace with the URL where your FastAPI server is running
base_url = "http://"+URDU_API_BASE_URL+":2006"

def check_api_status(base_url):
    try:
        # Use a short timeout, for example, 5 seconds
        response = requests.get(base_url + "/docs", timeout=5)
        
        # Check if the status code is in the range 200-299, indicating a successful request
        if response.ok:
            return True
        else:
            return False

    except requests.RequestException as e:
        return (f"Could not connect to the API at {base_url}")

@app.get("/handshake/")
async def handshake_endpoint(

):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: check_api_status(base_url)
        )
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2006, reload=True)
    # uvicorn app:app --host 0.0.0.0 --port 2006 --reload