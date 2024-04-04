'''
“依赖注入”时声明代码（路径操作函数）运行所需的，或要使用的依赖的一种方式。由FastAPI负责执行任意需要的逻辑，为代码提供这些依赖，即注入依赖
依赖注入常用于以下场景：
 共享业务逻辑，复用相同的代码逻辑
 共享数据库连接
 实现安全、验证、角色限权等
'''

from typing import Union

from fastapi import Depends, FastAPI

app = FastAPI()


# 依赖项就是一个函数，且可以使用与路径操作函数相同的参数；形式和结构与路径操作函数一样，可以把依赖项当作没有装饰器（即，没有 @app.get("/some-path") ）的路径操作函数。
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

if __name__ ==  "__main__":
    import uvicorn

    uvicorn.run("part7:app", host="0.0.0.0", port=8000, reload=True)