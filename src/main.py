import logging
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from starlette.requests import Request
from typing import List, Optional
import uvicorn
import os
import shutil

from loader import (
    get_chat_completion,
    load_website_urls,
    load_pdfs,
    upload_texts_to_vectorstore,
    delete_schema,
    get_chat_title,
)

from models import QueryRequest, TitleRequest
from basicauth import decode

app = FastAPI(
    docs_url="/dashboard",
    title="QMS",
    version="0.0.1",
    debug=True,
    contact={
        "name": "Novatr",
        "url": "https://github.com/novatr-tech",
        "email": "support@novatr.com",
    },
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@app.middleware("http")
async def auth_bearer(request: Request, call_next):
    path = request.scope['path']
    if path != '/':
        try:
            if request.headers.get('Authorization'):
                username, password = decode(request.headers.get('Authorization'))
                if username != os.getenv("QMS_USERNAME") or password != os.getenv("QMS_PASSWORD"):
                    return JSONResponse(status_code=401, content={'reason': "Bad auth"})
            else:
                return JSONResponse(status_code=401, content={'reason': "Bad auth"})
        except Exception as e:
            return e
    response = await call_next(request)
    return response


@app.get("/")
async def root(request: Request):
    return {"msg": "QMS"}


@app.post("/query", tags=["user"])
async def get_completion(
    body: QueryRequest,
):
    """
    Get Chat Completion
    """
    query = body.query
    sk = body.sk
    course_id, module_id, topic_id = None, None, None
    if body.filters:
        if body.filters.course_id:
            course_id = body.filters.course_id
        if body.filters.module_id:
            module_id = body.filters.module_id
        if body.filters.topic_id:
            topic_id = body.filters.topic_id
    try:
        response = await get_chat_completion(
            query, sk, course_id=course_id, module_id=module_id, topic_id=topic_id
        )
        return response
    except Exception as e:
        logger.error(e, exc_info=1)
        raise HTTPException(status_code=500, detail={"msg": str(e)})
    
@app.post("/title", tags=["user"])
async def get_title(
    body: TitleRequest,
):
    """
    Get Title
    """
    query = body.query
    try:
        response = await get_chat_title(query)
        return response
    except Exception as e:
        logger.error(e, exc_info=1)
        raise HTTPException(status_code=500, detail={"msg": str(e)})


@app.post("/load-zip")
async def load_zip(group_id: Optional[str] = None, zip_file: UploadFile = File(...)):
    try:
        # Save the uploaded zip file
        upload_dir = os.path.join("uploads")
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, zip_file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(zip_file.file, f)

        # Extract the zip file
        extracted_dir = os.path.join(upload_dir, "extracted")
        os.makedirs(extracted_dir, exist_ok=True)
        shutil.unpack_archive(file_path, extracted_dir)

        # return {"directory": extracted_dir}

        # Call the load_pdfs function with the extracted folder
        loader = await load_pdfs("./uploads/extracted/docs")
        # return {"loader": loader, "load": loader.load()}
        if not group_id:
            group_id = None
        result = await upload_texts_to_vectorstore(loader, group_id)

        # Clean up the extracted folder and the zip file
        shutil.rmtree(extracted_dir)
        os.remove(file_path)

        return result
    except Exception as e:
        print(f"An error occurred: {e}")
        # Handle the error gracefully, e.g., return an error response or log the error
        return {"error": str(e)}


@app.post("/load-urls")
async def load_urls(urls: List[str], group_id: Optional[str] = None):
    loader = await load_website_urls(urls)
    result = await upload_texts_to_vectorstore(loader, group_id)
    return result


@app.post("/delete-schema")
async def delete_all_schema():
    return await delete_schema()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
