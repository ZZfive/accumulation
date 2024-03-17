from enum import Enum
from typing import Union

from fastapi import FastAPI
import uvicorn

app = FastAPI()


class ModelName(str, Enum):
    alexnet = 'alexnet'
    resnet = 'resnet'
    lenet = 'lenet'


fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]


# 路径参数：在路径中使用{}括起来的参数，需要与函数中的参数名相同，如下面的item_id等
@app.get("/items/{item_id}")  # item_id 的值将作为参数 item_id 传递给函数
async def read_item(item_id: int):
    # 通过类型提示，FastAPI 会检查请求参数是否与指定的类型匹配，此处如果传入数据不是int，请求直接报错
    # 如果上述函数定义中未设置参数类型，不会进行参数校验
    return {"item_id": item_id}


# 路径操作是按顺序运行的，以下的/users/me如果在/users/{user_id}后面，fastapi会将其视为正在接收一个值为 "me" 的 user_id 参数
@app.get("/users/me")  # 固定路径
async def read_user_me():
    return {"user_id": "the current user"}


@app.get("/users/{user_id}")
async def read_user(user_id: str):
    return {"user_id": user_id}


@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):  # 使用自定义的枚举类作为定义参数类型，fastapi会自动校验传入的是否是枚举类中的值之一，不是会报错
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}

    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}

    return {"model_name": model_name, "message": "Have some residuals"}


@app.get("/files/{file_path:path}")  # 参数的名称为 file_path，结尾部分的 :path 说明该参数应匹配任意的路径
async def read_file(file_path: str):  # 可用“/files//home/johndoe/myfile.txt”测试
    return {"file_path": file_path}


# 查询参数：声明不属于路径参数的其他参数时，将被自动解释为查询参数；查询字符串是键值对的集合，这些键值对位于 URL 的 ？ 之后，并以 & 符号分隔
@app.get("/items/")
async def read_item(skip: int = 0, limit: int = 10):  # 由于查询参数不是路径的固定部分，因此它们可以是可选的，并且可以有默认值
    return fake_items_db[skip : skip + limit]


@app.get("/items/{item_id}")
async def read_item(item_id: str, q: str | None = None):  # 默认值设置为 None 来声明可选查询参数，即q是可选的；FastAPI 能够分辨出参数 item_id 是路径参数， q 是查询参数
    if q:
        return {"item_id": item_id, "q": q}
    return {"item_id": item_id}


@app.get("/items/{item_id}")
async def read_item(item_id: str, q: Union[str, None] = None, short: bool = False):  # bool类型会自动转换，即short传入short=1、short=true、short=True、short=on、short=yes或任何其他的变体形式（大写，首字母大写等等）都可解析
    item = {"item_id": item_id}
    if q:
        item.update({"q": q})
    if not short:
        item.update(
            {"description": "This is an amazing item that has a long description"}
        )
    return item


@app.get("/users/{user_id}/items/{item_id}")
async def read_user_item(
    user_id: int, item_id: str, q: Union[str, None] = None, short: bool = False
):  # 可以同时声明多个路径参数和查询参数，且不需要以任何特定的顺序来声明，它们将通过名称被检测到
    item = {"item_id": item_id, "owner_id": user_id}
    if q:
        item.update({"q": q})
    if not short:
        item.update(
            {"description": "This is an amazing item that has a long description"}
        )
    return item


# 当为非路径参数声明了默认值时，则该参数不是必需的；如果不想添加一个特定的值，而只是想使该参数成为可选的，则将默认值设置为 None
# 当想让一个查询参数成为必需的，不声明任何默认值即可；还可以像在 路径参数 中那样使用 Enum
@app.get("/items/{item_id}")
async def read_user_item(item_id: str, needy: str):  # 查询参数 needy 是类型为 str 的必需查询参数
    item = {"item_id": item_id, "needy": needy}
    return item



if __name__ ==  "__main__":
    import uvicorn

    uvicorn.run("part1:app", host="0.0.0.0", port=8000, reload=True)