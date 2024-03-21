from typing import Union, List
from uuid import UUID
from datetime import datetime, time, timedelta

from fastapi import FastAPI, Query, Path, Body
import uvicorn
from pydantic import BaseModel, Field

app = FastAPI()

# 额外信息
# 一种是以下方式，通过在model_config给pydantic模型设置例子
class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Foo",
                    "description": "A very nice Item",
                    "price": 35.4,
                    "tax": 3.2,
                }
            ]
        }
    }


# 一种是使用Field字段中的examples参数设置附加信息
class Item(BaseModel):
    name: str = Field(examples=["Foo"])
    description: str | None = Field(default=None, examples=["A very nice Item"])
    price: float = Field(examples=[35.4])
    tax: float | None = Field(default=None, examples=[3.2])


# 还可以使用body设置额外信息，即如下在body中设置examples设置
@app.put("/items/{item_id}")
async def update_item(
    item_id: int,
    item: Item = Body(
        examples=[
            {
                "name": "Foo",
                "description": "A very nice Item",
                "price": 35.4,
                "tax": 3.2,
            }
        ],
    ),
):
    results = {"item_id": item_id, "item": item}
    return results


# 请记住，以上设置的额外参数不会添加任何验证，只会添加注释，用于文档的目的，就像sd webui的fastapi swagger页面中接口设置的默认参数示例


'''
其他数据类型
UUID:
    一种标准的 "通用唯一标识符" ，在许多数据库和系统中用作ID。
    在请求和响应中将以 str 表示。
datetime.datetime:
    一个 Python datetime.datetime.
    在请求和响应中将表示为 ISO 8601 格式的 str ，比如: 2008-09-15T15:53:00+05:00.
datetime.date:
    Python datetime.date.
    在请求和响应中将表示为 ISO 8601 格式的 str ，比如: 2008-09-15.
datetime.time:
    一个 Python datetime.time.
    在请求和响应中将表示为 ISO 8601 格式的 str ，比如: 14:23:55.003.
datetime.timedelta:
    一个 Python datetime.timedelta.
    在请求和响应中将表示为 float 代表总秒数。
    Pydantic 也允许将其表示为 "ISO 8601 时间差异编码"
frozenset:
    在请求和响应中，作为 set 对待：
    在请求中，列表将被读取，消除重复，并将其转换为一个 set。
    在响应中 set 将被转换为 list 。
    产生的模式将指定那些 set 的值是唯一的 (使用 JSON 模式的 uniqueItems)。
bytes:
    标准的 Python bytes。
    在请求和响应中被当作 str 处理。
    生成的模式将指定这个 str 是 binary "格式"。
Decimal:
    标准的 Python Decimal。
    在请求和响应中被当做 float 一样处理
'''


@app.put("/items/{item_id}")
async def read_items(
    item_id: UUID,
    start_datetime: datetime | None = Body(default=None),
    end_datetime: datetime | None = Body(default=None),
    repeat_at: time | None = Body(default=None),
    process_after: timedelta | None = Body(default=None),
):
    start_process = start_datetime + process_after
    duration = end_datetime - start_process
    return {
        "item_id": item_id,
        "start_datetime": start_datetime,
        "end_datetime": end_datetime,
        "repeat_at": repeat_at,
        "process_after": process_after,
        "start_process": start_process,
        "duration": duration,
    }