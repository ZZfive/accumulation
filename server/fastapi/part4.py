from typing import Any

from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None
    tags: list[str] = []


'''
响应模型：对路由节点函数的输出进行校验；在任意的路径操作中使用 response_model 参数来声明用于响应的模型
FastAPI 将使用此 response_model 来：
    1、将输出数据转换为其声明的类型
    2、校验数据
    3、在OpenAPI路径操作中为响应添加一个Json Schema
    4、在自动生成文档系统中使用


响应模型在参数中被声明，而不是作为函数返回类型的注解，这是因为路径函数可能不会真正返回该响应模型，而是返回一个 dict、数据库对象或其他模型，
然后再使用 response_model 来执行字段约束和序列化
'''
@app.post("/items/", response_model=Item)  # 要求返回一个Item对象
async def create_item(item: Item) -> Any:
    return item


@app.get("/items/", response_model=list[Item])  # 要求返回Item对象的列表
async def read_items() -> Any:
    return [
        {"name": "Portal Gun", "price": 42.0},
        {"name": "Plumbus", "price": 32.0},
    ]


# 与指定响应模型的方式相同，可以在以下任意的路径操作中使用 status_code 参数来声明用于响应的 HTTP 状态码
# status_code 是「装饰器」方法（get，post 等）的一个参数。不像之前的所有参数和请求体，它不属于路径操作函数
# status_code 参数接收一个表示 HTTP 状态码的数字,也能够接收一个 IntEnum 类型，比如 Python 的 http.HTTPStatus
@app.post("/items/", status_code=201)  # 设置的状态吗会在响应中返回给调用端
async def create_item(name: str):
    return {"name": name}


'''
表单数据，接受的不是json，而是表单字段，要使用Form；要使用表单，需预先安装 python-multipart
使用 Form 可以声明与 Body （及 Query、Path、Cookie）相同的元数据和验证；要显式使用 Form ，否则，FastAPI 会把该参数当作查询参数或请求体（JSON）参数
'''
@app.post("/login/")
async def login(username: str = Form(), password: str = Form()):  # OAuth2 规范的 "密码流" 模式规定要通过表单字段发送 username 和 password；并且字段必须命名为 username 和 password，并通过表单字段发送，不能用 JSON
    return {"username": username}


'''
请求文件，用File定义客户端的上传文件，因为上传文件以「表单数据」形式发送，所以接收上传文件，要预先安装 python-multipart
File 是直接继承自 Form 的类，文件作为「表单数据」上传；声明文件体必须使用 File，否则，FastAPI 会把该参数当作查询参数或请求体（JSON）参数
'''
@app.post("/files/")
async def create_file(file: bytes = File()):  # 创建文件（File）参数的方式与 Body 和 Form 一样，以 bytes 形式读取和接收文件内容，把文件的所有内容都存储在内存里，适用于小型文件
    return {"file_size": len(file)}


'''
UploadFile的优势
    使用 spooled 文件；文件存储在内存中超出最大上限时，FastAPI会把文件存入磁盘
    更适合处理图像、视频、二进制等大型文件，好处是不会占用所有内存
    可获取上传文件的元数据
    自带file-like async接口
    暴露的 Python SpooledTemporaryFile 对象，可直接传递给其他预期「file-like」对象的库
'''
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):  # UploadFile与bytes比有更多优势
    return {"filename": file.filename}


# 可选文件上传，通过使用标准类型注解并将 None 作为默认值的方式将一个文件参数设为可选
@app.post("/files/")
async def create_file(file: bytes | None = File(default=None)):
    if not file:
        return {"message": "No file sent"}
    else:
        return {"file_size": len(file)}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile | None = None):
    if not file:
        return {"message": "No upload file sent"}
    else:
        return {"filename": file.filename}
    

# 带有额外元数据的 UploadFile
@app.post("/files/")
async def create_file(file: bytes = File(description="A file read as bytes")):
    return {"file_size": len(file)}


@app.post("/uploadfile/")
async def create_upload_file(
    file: UploadFile = File(description="A file read as UploadFile"),
):
    return {"filename": file.filename}


# 带有额外元数据的多文件上传；声明含 bytes 或 UploadFile 的列表（List），即可用同一个「表单字段」发送含多个文件的「表单数据」
@app.post("/files/")
async def create_files(
    files: list[bytes] = File(description="Multiple files as bytes"),
):
    return {"file_sizes": [len(file) for file in files]}


@app.post("/uploadfiles/")
async def create_upload_files(
    files: list[UploadFile] = File(description="Multiple files as UploadFile"),
):
    return {"filenames": [file.filename for file in files]}


@app.get("/")
async def main():
    content = """
<body>
<form action="/files/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
<form action="/uploadfiles/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)


''''
请求表单与文件，FastAPI 支持同时使用 File 和 Form 定义文件和表单字段
可在一个路径操作中声明多个 File 与 Form 参数，但不能同时声明要接收 JSON 的 Body 字段。因为此时请求体的编码为 multipart/form-data，不是 application/json
'''
@app.post("/files/")
async def create_file(
    file: bytes = File(), fileb: UploadFile = File(), token: str = Form()  # 创建文件和表单参数的方式与 Body 和 Query 一样
):
    return {
        "file_size": len(file),
        "token": token,
        "fileb_content_type": fileb.content_type,
    }