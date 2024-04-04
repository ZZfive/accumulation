from typing import List, Union

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    name: Union[str, None] = None
    description: Union[str, None] = None
    price: Union[float, None] = None
    tax: float = 10.5
    tags: List[str] = []


items = {
    "foo": {"name": "Foo", "price": 50.2},
    "bar": {"name": "Bar", "description": "The bartenders", "price": 62, "tax": 20.2},
    "baz": {"name": "Baz", "description": None, "price": 50.2, "tax": 10.5, "tags": []},
}


@app.get("/items/{item_id}", response_model=Item)
async def read_item(item_id: str):
    return items[item_id]

'''
用Put方式更新数据，把数据转换为以JSON格式存储的数据，可是哟哦那个jsonable_encoder

如果用以下数据更新items中的bar更新，因为数据未包含已存储的属性 "tax": 20.2，新的输入模型会把 "tax": 10.5 作为默认值。因此，本次操作把 tax 的值「更新」为 10.5。
{
    "name": "Barz",
    "price": 3,
    "description": None,
}
'''
@app.put("/items/{item_id}", response_model=Item)
async def update_item(item_id: str, item: Item):
    update_item_encoded = jsonable_encoder(item)
    items[item_id] = update_item_encoded
    return update_item_encoded


'''
用PATCH进行部分更新，即只发送要更新的数据，其余数据保持不变
PATCH 没有 PUT 知名，也怎么不常用。很多人甚至只用 PUT 实现部分更新。FastAPI 对此没有任何限制，可以随意互换使用这两种操作

更新部分数据应：
  使用 PATCH 而不是 PUT （可选，也可以用 PUT）；
  提取存储的数据；
  把数据放入 Pydantic 模型；
  生成不含输入模型默认值的 dict （使用 exclude_unset 参数）；
  只更新用户设置过的值，不用模型中的默认值覆盖已存储过的值。
  为已存储的模型创建副本，用接收的数据更新其属性 （使用 update 参数）。
  把模型副本转换为可存入数据库的形式（比如，使用 jsonable_encoder）。
  这种方式与 Pydantic 模型的 .dict() 方法类似，但能确保把值转换为适配 JSON 的数据类型，例如， 把 datetime 转换为 str 。
  把数据保存至数据库；
  返回更新后的模型
'''
@app.patch("/items/{item_id}", response_model=Item)
async def update_item(item_id: str, item: Item):
    stored_item_data = items[item_id]  # 先获取待更新的items中的字典对象
    stored_item_model = Item(**stored_item_data)  # 用上述字典构建一个Item对象
    update_data = item.dict(exclude_unset=True)  # 使用Pydantic模型的.dict()方法，同时设置exclude_unset=True，生成的dict对象只包含创建item模型时显示设置的数据，不包括默认值
    updated_item = stored_item_model.copy(update=update_data)  # 更新
    items[item_id] = jsonable_encoder(updated_item)  # 覆盖
    return updated_item