from typing import Union, List, Annotated

from fastapi import FastAPI, Query, Path, Body
import uvicorn
from pydantic import BaseModel, Field, HttpUrl

app = FastAPI()


# 请求体是客户端发送给 API 的数据。响应体是 API 发送给客户端的数据
# 你不能使用 GET 操作（HTTP 方法）发送请求体;要发送数据，你必须使用下列方法之一：POST（较常见）、PUT、DELETE 或 PATCH
class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None


class User(BaseModel):
    username: str
    full_name: str | None = None


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


# 使用Query添加额外的约束条件
@app.get("/items/")
async def read_items(
    q: Union[str, None] = Query(
        default=None,  # 默认值为None表示可选，如果不设置default，并改为“q: str”就表示为必需参数
        min_length=3,  # 最小长度三个字符
        max_length=50,  # 最大长度50个字符
        pattern="^fixedquery$"  # 需要满足设置的正则表达式
    ),
):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results


# 查询参数列表，即声明一个可在url中出现多次的查询参数
# 以下的q就是这样的查询参数，因为q是List或None，可如此请求：http://localhost:8000/items/?q=foo&q=bar
# 要声明类型为 list 的查询参数，如下所示，需要显式地使用 Query，否则该参数将被解释为请求体
@app.get("/items/")
async def read_items(q: Union[List[str], None] = Query(default=None)):
    query_items = {"q": q}
    return query_items


# 可以给查询参数列表设置默认值
@app.get("/items/")
async def read_items(q: List[str] = Query(default=["foo", "bar"])):
    query_items = {"q": q}
    return query_items


# 可以直接使用 list 代替 List [str]
@app.get("/items/")
async def read_items(q: list = Query(default=[])):
    query_items = {"q": q}
    return query_items


# 可以添加更多有关该参数的信息，这些信息将包含在生成的 OpenAPI 模式中，并由文档用户界面和外部工具所使用
@app.get("/items/")
async def read_items(
    q: Union[str, None] = Query(
        default=None,
        alias="item-query",  # q的别名，如果item-query出现在url中，其会被是识别为q
        title="Query string",
        description="Query string for the items to search in the database that have a good match",
        min_length=3,
        max_length=50,
        pattern="^fixedquery$",
        deprecated=True,  # 已弃用参数
    ),
):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results


# 与使用 Query 为查询参数声明更多的校验和元数据的方式相同，也可以使用 Path 为路径参数声明相同类型的校验和元数据
# Path可以声明与 Query 相同的所有参数，它们都共享相同的所有已看到并用于添加额外校验和元数据的参数
@app.get("/items/{item_id}")
async def read_items(
    *,  # python中位置参数要放在关键字参数，此处的*表示所有此函数所有参数都是关键字参数，即使没有默认值
    item_id: int = Path(title="The ID of the item to get", ge=0, le=1000),  # 要求值必须大于等于0且小于等于1000
    q: str,
    size: float = Query(gt=0, lt=10.5),  # 对浮点数进行区间约束
):
    results = {"item_id": item_id}
    if q:
        results.update({"q": q})
    return results


# 混合使用 Path、Query 和请求体参数
@app.put("/items/{item_id}")
async def update_item(
    *,
    item_id: int = Path(title="The ID of the item to get", ge=0, le=1000),  # 路径参数
    q: str | None = None,  # 查询参数
    item: Item | None = None,  # 默认值为None，表示此请求体参数是可选的；期望具有一个具有Item属性的json请求体
):
    results = {"item_id": item_id}
    if q:
        results.update({"q": q})
    if item:
        results.update({"item": item})
    return results


# 多个请求体参数同时使用
'''以下的函数update_item定义中使用了两个Pydantic 模型参数，Fastapi会自动识别，会将两个请求体参数组合起来，构建新的json请求体，如下所示
{
    "item": {
        "name": "Foo",
        "description": "The pretender",
        "price": 42.0,
        "tax": 3.2
    },
    "user": {
        "username": "dave",
        "full_name": "Dave Grohl"
    }
}
'''
@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item, user: User):
    results = {"item_id": item_id, "item": item, "user": user}
    return results


'''
为了扩展先前的模型，你可能决定除了 item 和 user 之外，还想在同一请求体中具有另一个键 importance。
如果你就按原样声明它，因为它是一个单一值，FastAPI 将假定它是一个查询参数。=
但是你可以使用 Body 指示 FastAPI 将其作为请求体的另一个键进行处理
Body 同样具有与 Query、Path 以及其他后面将看到的类完全相同的额外校验和元数据参数
{
    "item": {
        "name": "Foo",
        "description": "The pretender",
        "price": 42.0,
        "tax": 3.2
    },
    "user": {
        "username": "dave",
        "full_name": "Dave Grohl"
    },
    "importance": 5
}
'''
@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item, user: User,
                      importance: int = Body()  # 使用Body将参数importance设置为请求体中的键
                      ):
    results = {"item_id": item_id, "item": item, "user": user, "importance": importance}
    return results


'''
若只有一个来自 Pydantic 模型 Item 的请求体参数 item。默认情况下，FastAPI 将直接期望这样的请求体。
但是，如果希望它期望一个拥有 item 键并在值中包含模型内容的 JSON，就像在声明额外的请求体参数时所做的那样，则可以使用一个特殊的 Body 参数 embed
{
    "item": {  # 即这个item的key是因为设置了body的embed=True这个参数
        "name": "Foo",
        "description": "The pretender",
        "price": 42.0,
        "tax": 3.2
    }
}
'''
@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item = Body(embed=True)):
    results = {"item_id": item_id, "item": item}
    return results


# 可以使用 Pydantic 的 Field 在 Pydantic 模型内部声明校验和元数据
# Field 的工作方式和 Query、Path 和 Body 相同，包括它们的参数等等也完全相同
class Item(BaseModel):
    name: str
    description: str | None = Field(
        default=None, title="The description of the item", max_length=300
    )
    price: float = Field(gt=0, description="The price must be greater than zero")
    tax: float | None = None


# 嵌套模型，定义任意深度的嵌套模型
'''
以下例子中，Fastapi期望的请求体如下
{
    "name": "Foo",
    "description": "The pretender",
    "price": 42.0,
    "tax": 3.2,
    "tags": ["rock", "metal", "bar"],
    "image": {
        "url": "http://example.com/baz.jpg",
        "name": "The Foo live"
    }
}

{
    "name": "Foo",
    "description": "The pretender",
    "price": 42.0,
    "tax": 3.2,
    "tags": [
        "rock",
        "metal",
        "bar"
    ],
    "images": [
        {
            "url": "http://example.com/baz.jpg",
            "name": "The Foo live"
        },
        {
            "url": "http://example.com/dave.jpg",
            "name": "The Baz"
        }
    ]
}
'''
class Image(BaseModel):
    url: str  # HttpUrl, 如果类型设置为HttpUrl，字符串将被检查是否为有效的 URL，并在 JSON Schema / OpenAPI 文档中进行记录
    name: str


class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None
    tags: set[str] = set()
    image: Image | None = None
    # images: list[Image] | None = None


# 纯列表请求体，如果期望的 JSON 请求体的最外层是一个 JSON array（即 Python list），则可以在路径操作函数的参数中声明此类型，就像声明 Pydantic 模型一样
@app.post("/images/multiple/")
async def create_multiple_images(images: list[Image]):
    return images


# 任意 dict 构成的请求体，可以将请求体声明为使用某类型的键和其他类型值的 dict。无需事先知道有效的字段/属性（在使用 Pydantic 模型的场景）名称是什么
# JSON 仅支持将 str 作为键，但 Pydantic 具有自动转换数据的功能。这意味着，即使你的 API 客户端只能将字符串作为键发送，
# 只要这些字符串内容仅包含整数，Pydantic 就会对其进行转换并校验。然后你接收的名为 weights 的 dict 实际上将具有 int 类型的键和 float 类型的值
@app.post("/index-weights/")
async def create_index_weights(weights: dict[int, float]):
    return weights


if __name__ ==  "__main__":
    import uvicorn

    uvicorn.run("part1:app", host="0.0.0.0", port=8000, reload=True)