'''
“依赖注入”时声明代码（路径操作函数）运行所需的，或要使用的依赖的一种方式。由FastAPI负责执行任意需要的逻辑，为代码提供这些依赖，即注入依赖
依赖注入常用于以下场景：
 共享业务逻辑，复用相同的代码逻辑
 共享数据库连接
 实现安全、验证、角色限权等


依赖注入无非是与路径操作函数一样的函数罢了。但它依然非常强大，能够声明任意嵌套深度的「图」或树状的依赖结构
'''

from typing import Union

from fastapi import Depends, FastAPI, Cookie

app = FastAPI()


# 依赖项最常见的形式是一个函数，且可以使用与路径操作函数相同的参数；形式和结构与路径操作函数一样，可以把依赖项当作没有装饰器（即，没有 @app.get("/some-path") ）的路径操作函数。
'''
依赖项可以返回各种内容。本例中的依赖项预期接收如下参数：
  类型为 str 的可选查询参数 q
  类型为 int 的可选查询参数 skip，默认值是 0
  类型为 int 的可选查询参数 limit，默认值是 100
然后，依赖项函数返回包含这些值的 dict。
'''
async def common_parameters(q: Union[str, None] = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}


@app.get("/items/")
async def read_items(commons: dict = Depends(common_parameters)):  # 用Depends导入依赖项进行声明
    return commons


@app.get("/users/")
async def read_users(commons: dict = Depends(common_parameters)):
    return commons


'''
在路径操作函数的参数中使用Depends的方式与Body、Query相同，但Depends的工作方式略有不同；只能给Depends传一个参数，且该参数必须时一个可调用
对象，比如函数；该参数接受的参数和路径操作函数的参数一样

接受到新的请求时，FastAPI执行如下操作
  用正确的参数调用依赖项函数
  获取函数返回的结果
  把函数返回的结果赋值给路径操作函数的参数
这样只编写一次代码，FastAPI就可以为多个路径操作共享这段代码

无需创建专门的类，并将之传递给 FastAPI 以进行「注册」或执行类似的操作。只要把它传递给 Depends，FastAPI 就知道该如何执行后续操作
FastAPI 调用依赖项的方式与路径操作函数一样，因此，定义依赖项函数，也要应用与路径操作函数相同的规则。即，既可以使用异步的 async def，也可以使用普通的 def 定义依赖项。
在普通的 def 路径操作函数中，可以声明异步的 async def 依赖项；也可以在异步的 async def 路径操作函数中声明普通的 def 依赖项。上述这些操作都是可行的，FastAPI 知道该怎么处理

详情链接：https://fastapi.tiangolo.com/zh/tutorial/dependencies/
'''


''''
依赖项应该是一个可调用对象，但不仅限于函数，可以用类作为依赖项，python中类是一个可调用对象
如果在 FastAPI 中传递一个 "可调用对象" 作为依赖项，它将分析该 "可调用对象" 的参数，并以处理路径操作函数的参数的方式来处理它们。包括子依赖项。
这也适用于完全没有参数的可调用对象。这与不带参数的路径操作函数一样。所以，可以将上面的依赖项 "可依赖对象" common_parameters 更改为类 CommonQueryParams
'''
fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]


class CommonQueryParams:
    def __init__(self, q: str | None = None, skip: int = 0, limit: int = 100):  # 与common_parameters函数参数相同，这些参数就是fastapi用于处理依赖项的
        self.q = q
        self.skip = skip
        self.limit = limit


'''
以下声明依赖项时，对参数commons定义中，第一个CommonQueryParams进行类型声明，但其对fastapi来说没有任何意义，FastAPI 不会使用它进行数据转换、验证等 (因为对于这，它使用 = Depends(CommonQueryParams))
其实可以直接如此定义：commons = Depends(CommonQueryParams)；但声明类型是被鼓励的，便于编辑器指导参数具体类型，
可以按如下定义：commons: CommonQueryParams = Depends()
'''
@app.get("/items/")
# async def read_items(commons: CommonQueryParams = Depends(CommonQueryParams)):  # 声明依赖项；FastAPI 调用 CommonQueryParams 类。这将创建该类的一个 "实例"，该实例将作为参数 commons 被传递给函数
# async def read_items(commons = Depends(CommonQueryParams)):
async def read_items(commons: CommonQueryParams = Depends()):
    response = {}
    if commons.q:
        response.update({"q": commons.q})
    items = fake_items_db[commons.skip : commons.skip + commons.limit]
    response.update({"items": items})
    return response


'''
子依赖项：FastAPI 支持创建含子依赖项的依赖项。并且，可以按需声明任意深度的子依赖项嵌套层级。FastAPI 负责处理解析不同深度的子依赖项
'''
def query_extractor(q: Union[str, None] = None):  # 第一层依赖项；声明了类型为 str 的可选查询参数 q，然后返回这个查询参数
    return q


def query_or_cookie_extractor(  # 第二层依赖项
    q: str = Depends(query_extractor),  # 依赖嵌套；该函数依赖 query_extractor, 并把 query_extractor 的返回值赋给参数 q
    last_query: Union[str, None] = Cookie(default=None),  # 类型是 str 的可选 cookie（last_query）用户未提供查询参数 q 时，则使用上次使用后保存在 cookie 中的查询
):
    if not q:
        return last_query
    return q


@app.get("/items/")
async def read_query(query_or_default: str = Depends(query_or_cookie_extractor)):  # 使用依赖项；注意，这里在路径操作函数中只声明了一个依赖项，即 query_or_cookie_extractor 。但 FastAPI 必须先处理 query_extractor，以便在调用 query_or_cookie_extractor 时使用 query_extractor 返回的结果
    return {"q_or_cookie": query_or_default}


'''
如果在同一个路径操作 多次声明了同一个依赖项，例如，多个依赖项共用一个子依赖项，FastAPI 在处理同一请求时，只调用一次该子依赖项。
FastAPI 不会为同一个请求多次调用同一个依赖项，而是把依赖项的返回值进行「缓存」，并把它传递给同一请求中所有需要使用该返回值的「依赖项」。

在高级使用场景中，如果不想使用「缓存」值，而是为需要在同一请求的每一步操作（多次）中都实际调用依赖项，可以把 Depends 的参数 use_cache 的值设置为 False :
'''
async def needy_dependency(fresh_value: str = Depends(get_value, use_cache=False)):
    return {"fresh_value": fresh_value}


if __name__ ==  "__main__":
    import uvicorn

    uvicorn.run("part7:app", host="0.0.0.0", port=8000, reload=True)