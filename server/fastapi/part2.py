from typing import Union

from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

app = FastAPI()


# 请求体是客户端发送给 API 的数据。响应体是 API 发送给客户端的数据
# 你不能使用 GET 操作（HTTP 方法）发送请求体;要发送数据，你必须使用下列方法之一：POST（较常见）、PUT、DELETE 或 PATCH
class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None


''' FastAPI 能识别出与路径参数匹配的函数参数应从路径中获取，而声明为 Pydantic 模型的函数参数应从请求体中获取
   如果在路径中也声明了该参数，它将被用作路径参数
   如果参数属于单一类型（比如 int、float、str、bool 等）它将被解释为查询参数
   如果参数的类型被声明为一个 Pydantic 模型，它将被解释为请求体
'''    
@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item, q: str | None = None):
    result = {"item_id": item_id, **item.dict()}
    if q:
        result.update({"q": q})
    return result


if __name__ ==  "__main__":
    import uvicorn

    uvicorn.run("part1:app", host="0.0.0.0", port=8000, reload=True)