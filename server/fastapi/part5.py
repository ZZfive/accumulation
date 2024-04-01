from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException  # 与fastapi的HTTPException进行区分
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)


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