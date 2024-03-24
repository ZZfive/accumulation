from typing import Any

from fastapi import FastAPI
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