'''
路径操作装饰器依赖项：有时，并不需要在路径操作函数中使用依赖项的返回值。或者说，有些依赖项不返回值。但仍要执行或解析该依赖项。
对于这种情况，不必在声明路径操作函数的参数时使用 Depends，而是可以在路径操作装饰器中添加一个由 dependencies 组成的 list

路径操作装饰器支持可选参数“dependencies”，其值是由Depends()组成的list；路径装饰器依赖项的执行或解析方式和普通依赖项一样，但就算这些依赖项会返回值，它们的值也不会传递给路径操作函数
'''

from fastapi import Depends, FastAPI, Header, HTTPException

app = FastAPI()


async def verify_token(x_token: str = Header()):  # 路径装饰器依赖项可以声明请求的需求项（比如响应头）或其他子依赖项
    if x_token != "fake-super-secret-token":
        raise HTTPException(status_code=400, detail="X-Token header invalid")  # 路径装饰器依赖项与正常的依赖项一样，可以 raise 异常


async def verify_key(x_key: str = Header()):  # 路径装饰器依赖项可以声明请求的需求项（比如响应头）或其他子依赖项
    if x_key != "fake-super-secret-key":
        raise HTTPException(status_code=400, detail="X-Key header invalid")  # 路径装饰器依赖项与正常的依赖项一样，可以 raise 异常
    return x_key  # 无论路径装饰器依赖项是否返回值，路径操作函数都不会使用这些值。因此，可以复用在其他位置使用过的、（能返回值的）普通依赖项，即使没有使用这个值，也会执行该依赖项


@app.get("/items/", dependencies=[Depends(verify_token), Depends(verify_key)])
async def read_items():
    return [{"item": "Foo"}, {"item": "Bar"}]


'''
全局依赖项：有时，要为整个应用添加依赖项。通过与定义“路径装饰器依赖项”类似的方式，可以把依赖项添加至整个 FastAPI 应用。
这样一来，就可以为所有路径操作应用该依赖项
'''
async def verify_token(x_token: str = Header()):
    if x_token != "fake-super-secret-token":
        raise HTTPException(status_code=400, detail="X-Token header invalid")


async def verify_key(x_key: str = Header()):
    if x_key != "fake-super-secret-key":
        raise HTTPException(status_code=400, detail="X-Key header invalid")
    return x_key


app = FastAPI(dependencies=[Depends(verify_token), Depends(verify_key)])  # 直接把依赖项附加在app对象上，即实现全局依赖项


@app.get("/items/")
async def read_items():
    return [{"item": "Portal Gun"}, {"item": "Plumbus"}]


@app.get("/users/")
async def read_users():
    return [{"username": "Rick"}, {"username": "Morty"}]


'''
FastAPI支持在完成后执行一些额外步骤的依赖项.为此，请使用 yield 而不是 return，然后再编写额外的步骤（代码)；但确保只使用一次 yield

任何一个可以与以下内容一起使用的函数：
    @contextlib.contextmanager 或者
    @contextlib.asynccontextmanager
都可以作为 FastAPI 的依赖项。
实际上，FastAPI内部就使用了这两个装饰器。
'''
# 可以使用以下方式创建一个数据库会话并在完成后关闭
async def get_db():
    db = DBSession()
    try:  # 在带有yield的依赖关系中使用了try代码块，就会收到使用依赖关系时抛出的任何异常，因此可以使用except SomeException在依赖关系中查找特定异常
        yield db  # 在发送响应之前，只会执行yield之前的代码；yield生成的数据库对象db会注入到路径操作函数或其他依赖项中
    finally:  # yield之后的代码会在发送响应后执行
        db.close()  # 即在路径操作函数返回相应后关掉数据库


'''
可以拥有任意大小和形状的子依赖和子依赖的“树”，而且它们中的任何一个或所有的都可以使用yield。FastAPI 会确保每个带有yield的依赖中的“退出代码”按正确顺序运行

例如以下例子中dependency_c 依赖于 dependency_b，而 dependency_b 则依赖于 dependency_a；所有这些依赖都可以使用yield。
在这种情况下，dependency_c 在执行其退出代码时需要dependency_b（此处称为 dep_b）的值仍然可用。而dependency_b 反过来则需要dependency_a（此处称为 dep_a）的值在其退出代码中可用

同样，你可以有混合了yield和return的依赖。你也可以有一个单一的依赖需要多个其他带有yield的依赖，等等。你可以拥有任何你想要的依赖组合。FastAPI 将确保按正确的顺序运行所有内容
'''
async def dependency_a():
    dep_a = generate_dep_a()
    try:
        yield dep_a
    finally:
        dep_a.close()


async def dependency_b(dep_a=Depends(dependency_a)):
    dep_b = generate_dep_b()
    try:
        yield dep_b
    finally:
        dep_b.close(dep_a)


async def dependency_c(dep_b=Depends(dependency_b)):
    dep_c = generate_dep_c()
    try:
        yield dep_c
    finally:
        dep_c.close(dep_b)


'''
使用yield和HTTPException的依赖项和在依赖项中使用带有yield的上下文管理器详情见：https://fastapi.tiangolo.com/zh/tutorial/dependencies/dependencies-with-yield/
'''