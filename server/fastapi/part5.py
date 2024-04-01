import json
from typing import Set, Union
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException  # 与fastapi的HTTPException进行区分
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.encoders import jsonable_encoder


app = FastAPI()

'''
使用HTTPException向客户端返回 HTTP 错误响应
如在调用路径操作函数里的工具函数时，触发了 HTTPException，FastAPI 就不再继续执行路径操作函数中的后续代码，而是立即终止请求，并把 HTTPException 的 HTTP 错误发送至客户端
触发 HTTPException 时，可以用参数 detail 传递任何能转换为 JSON 的值，不仅限于 str；还支持传递 dict、list 等数据结构，FastAPI 能自动处理这些数据，并将之转换为 JSON。
'''

items = {"foo": "The Foo Wrestlers"}


@app.get("/items/{item_id}")
async def read_item(item_id: str):
    if item_id not in items:
        raise HTTPException(status_code=404, detail="Item not found")  # HTTPException是python异常，只能用raise，不能用return
    return {"item": items[item_id]}


@app.get("/items-header/{item_id}")
async def read_item_header(item_id: str):
    if item_id not in items:
        raise HTTPException(
            status_code=404,
            detail="Item not found",
            headers={"X-Error": "There goes my error"},  # 一般情况下可能不会需要在代码中直接使用响应头，但对于某些高级应用场景，还是需要添加自定义响应头
        )
    return {"item": items[item_id]}



'''
自定义异常处理器
UnicornException是一个自定义的异常类型，使用@app.exception_handler()可以将unicorn_exception_handler设置为fastapi的全局异常处理器
当触发UnicornException时，exception_handler函数会进行针对性处理
'''
class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name


@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request: Request, exc: UnicornException):
    return JSONResponse(
        status_code=418,
        content={"message": f"Oops! {exc.name} did something. There goes a rainbow..."},
    )


@app.get("/unicorns/{name}")
async def read_unicorn(name: str):
    if name == "yolo":
        raise UnicornException(name=name)  # 触发UnicornException
    return {"unicorn_name": name}


'''
覆盖默认异常处理器
FastAPI 自带了一些默认异常处理器。触发 HTTPException 或请求无效数据时，这些处理器返回默认的 JSON 响应结果。不过，也可以使用自定义处理器覆盖默认异常处理器

如：请求中包含无效数据时，FastAPI 内部会触发 RequestValidationError。该异常也内置了默认异常处理器。覆盖默认异常处理器时需要导入 RequestValidationError，
并用 @app.excption_handler(RequestValidationError) 装饰异常处理器。这样，异常处理器就可以接收 Request 与异常

注意：FastAPI 的 HTTPException 继承自 Starlette 的 HTTPException 错误类。它们之间的唯一区别是，FastAPI 的 HTTPException 可以在响应中添加响应头。OAuth 2.0 等安全工具需要在内部调用这些响应头。
因此你可以继续像平常一样在代码中触发 FastAPI 的 HTTPException 。但注册异常处理器时，应该注册到来自 Starlette 的 HTTPException。这样做是为了，当 Starlette 的内部代码、扩展或插件触发 Starlette HTTPException 时，处理程序能够捕获、并处理此异常
'''
@app.exception_handler(StarletteHTTPException)  # 覆盖HTTPException异常
async def http_exception_handler(request, exc):
    return PlainTextResponse(str(exc.detail), status_code=exc.status_code)


@app.exception_handler(RequestValidationError)  # 覆盖RequestValidationError异常
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)


@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id == 3:
        raise HTTPException(status_code=418, detail="Nope! I don't like 3.")
    return {"item_id": item_id}


'''
FastAPI 支持先对异常进行某些处理，然后再使用 FastAPI 中处理该异常的默认异常处理器
'''
@app.exception_handler(StarletteHTTPException)  # 本质还是覆盖对应的默认异常处理器，只是在自定义的异常处理器中调用了对应的默认异常处理器
async def custom_http_exception_handler(request, exc):
    print(f"OMG! An HTTP error!: {repr(exc)}")  # 输出异常
    return await http_exception_handler(request, exc)  # 复用StarletteHTTPException的默认异常处理器


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    print(f"OMG! The client sent invalid data!: {exc}")
    return await request_validation_exception_handler(request, exc)  # 复用RequestValidationError的默认异常处理器


@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id == 3:
        raise HTTPException(status_code=418, detail="Nope! I don't like 3.")
    return {"item_id": item_id}


# 路径操作配置，路径操作装饰器支持多种配置参数；参数应直接传递给路径操作装饰器，不能传递给路径操作函数
# 状态码，status_code 用于定义路径操作响应中的 HTTP 状态码;可以直接传递 int 代码， 比如 404，如果记不住数字码的涵义，也可以用 status 的快捷常量
class Item(BaseModel):
    name: str
    description: Union[str, None] = None
    price: float
    tax: Union[float, None] = None
    tags: Set[str] = set()


@app.post("/items/", response_model=Item, status_code=status.HTTP_201_CREATED)  # 设施状态码
async def create_item(item: Item):
    return item


# tags 参数，值是由 str 组成的 list （一般只有一个 str ），tags 用于为路径操作添加标签；具有相同tags参数的路径函数在swaager接口中会在同一个tag下
@app.post("/items/", response_model=Item, tags=["items"])
async def create_item(item: Item):
    return item


@app.get("/items/", tags=["items"])
async def read_items():
    return [{"name": "Foo", "price": 42}]


@app.get("/users/", tags=["users"])
async def read_users():
    return [{"username": "johndoe"}]


# summary和description
@app.post(
    "/items/",
    response_model=Item,
    summary="Create an item",
    description="Create an item with all the information, name, description, price, tax and a set of unique tags",  # 描述路径操作
)
async def create_item(item: Item):
    return item


'''
文档字符串（docstring）
描述内容比较长且占用多行时，可以在函数的 docstring 中声明路径操作的描述，FastAPI 支持从文档字符串中读取描述内容。
文档字符串支持 Markdown，能正确解析和显示 Markdown 的内容，但要注意文档字符串的缩进。
'''
@app.post("/items/", response_model=Item, summary="Create an item")
async def create_item(item: Item):
    """
    Create an item with all the information:

    - **name**: each item must have a name
    - **description**: a long description
    - **price**: required
    - **tax**: if the item doesn't have tax, you can omit this
    - **tags**: a set of unique tag strings for this item
    """
    return item


# response_description 参数用于定义响应的描述说明；OpenAPI 规定每个路径操作都要有响应描述。如果没有定义响应描述
# FastAPI 则自动生成内容为 "Successful response" 的响应描述
@app.post(
    "/items/",
    response_model=Item,
    summary="Create an item",
    response_description="The created item",  # 响应的描述说明
)
async def create_item(item: Item):
    """
    Create an item with all the information:

    - **name**: each item must have a name
    - **description**: a long description
    - **price**: required
    - **tax**: if the item doesn't have tax, you can omit this
    - **tags**: a set of unique tag strings for this item
    """
    return item


# 弃用路径操作，deprecated 参数可以把路径操作标记为弃用，无需直接删除
@app.get("/items/", tags=["items"])
async def read_items():
    return [{"name": "Foo", "price": 42}]


@app.get("/users/", tags=["users"])
async def read_users():
    return [{"username": "johndoe"}]


@app.get("/elements/", tags=["items"], deprecated=True)  # 表示此路径被启用了
async def read_elements():
    return [{"item_id": "Foo"}]


'''
JSON 兼容编码器
在某些情况下，您可能需要将数据类型（如Pydantic模型）转换为与JSON兼容的数据类型（如dict、list等）。比如，如果您需要将其存储在数据库中。对于这种要求， FastAPI提供了jsonable_encoder()函数

假设有一个数据库名为fake_db，它只能接收与JSON兼容的数据,它不接收datetime这类的对象，因为这些对象与JSON不兼容。因此，datetime对象必须将转换为包含ISO格式化的str类型对象。
同样，这个数据库也不会接收Pydantic模型（带有属性的对象），而只接收dict。对此可以使用jsonable_encoder。它接收一个对象，比如Pydantic模型，并会返回一个JSON兼容的版本

经过jsonable_encoder编码后的结果，可以使用Python标准编码中的json.dumps()。这个操作不会返回一个包含JSON格式（作为字符串）数据的庞大的str。它将返回一个Python标准数据结构（例如dict），其值和子值都与JSON兼容
'''
fake_db = {}


class Item(BaseModel):
    title: str
    timestamp: datetime
    description: str | None = None


@app.put("/items/{id}")
def update_item(id: str, item: Item):
    json_compatible_item_data = jsonable_encoder(item)  # 将Pydantic模型转换为dict，并将datetime转换为str
    # json_compatible_item_data = json.dumps(json_compatible_item_data)
    fake_db[id] = json_compatible_item_data