'''
安全规范
    OAuth2：一个规范，它定义了几种处理身份认证和授权的方法。它是一个相当广泛的规范，涵盖了一些复杂的使用场景。它包括了使用「第三方」进行身份认证的方法。这就是所有带有「使用 Facebook，Google，Twitter，GitHub 登录」的系统背后所使用的机制
    OpenID Connect：一个基于 OAuth2 的规范；只是扩展了 OAuth2，并明确了一些在 OAuth2 中相对模糊的内容，以尝试使其更具互操作性。例如，Google 登录使用 OpenID Connect（底层使用OAuth2）。但是 Facebook 登录不支持 OpenID Connect。它具有自己的 OAuth2 风格
    OpenID：一个规范；试图解决与 OpenID Connect 相同的问题，但它不是基于 OAuth2。因此，它是一个完整的附加系统。如今它已经不是很流行，没有被广泛使用了
    OpenAPI（以前称为 Swagger）是用于构建 API 的开放规范（现已成为 Linux Foundation 的一部分）。
        FastAPI 基于 OpenAPI；这就是使多个自动交互式文档界面，代码生成等成为可能的原因

OpenAPI 有一种定义多个安全「方案」的方法。通过使用它们，可以利用所有这些基于标准的工具，包括交互式文档系统
OpenAPI 定义了以下安全方案：
    apiKey：一个特定于应用程序的密钥，可以来自：
        查询参数
        请求头
        cookie
    http：标准的 HTTP 身份认证系统，包括：
        bearer: 一个值为 Bearer 加令牌字符串的 Authorization 请求头。这是从 OAuth2 继承的
        HTTP Basic 认证方式
        HTTP Digest，等等
    oauth2：所有的 OAuth2 处理安全性的方式（称为「流程」）。 *以下几种流程适合构建 OAuth 2.0 身份认证的提供者（例如 Google，Facebook，Twitter，GitHub 等）： * implicit * clientCredentials * authorizationCode
        但是有一个特定的「流程」可以完美地用于直接在同一应用程序中处理身份认证：
            password
    openIdConnect：提供了一种定义如何自动发现 OAuth2 身份认证数据的方法
        此自动发现机制是 OpenID Connect 规范中定义的内容
'''

from typing import Annotated, Union

from fastapi import Depends, FastAPI
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

app = FastAPI()

# 详情链接：https://fastapi.tiangolo.com/zh/tutorial/security/first-steps/
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class User(BaseModel):
    username: str
    email: Union[str, None] = None
    full_name: Union[str, None] = None
    disabled: Union[bool, None] = None


@app.get("/items/")
async def read_items(token: Annotated[str, Depends(oauth2_scheme)]):  # 基于依赖注入系统的安全系统向路径操作函数传递了 str 类型的 token
    return {"token": token}


'''
获取当前用用户；详情链接：https://fastapi.tiangolo.com/zh/tutorial/security/get-current-user/

以下示例看起来有些冗长。毕竟这个文件同时包含了安全、数据模型的工具函数，以及路径操作等代码。但，关键是：安全和依赖注入的代码只需要写一次。
就算写得再复杂，也只是在一个位置写一次就够了。所以，要多复杂就可以写多复杂。但是，就算有数千个端点（路径操作），它们都可以使用同一个安全系统。
而且，所有端点（或它们的任何部件）都可以利用这些依赖项或任何其它依赖项。
'''
def fake_decode_token(token):
    return User(
        username=token + "fakedecoded", email="john@example.com", full_name="John Doe"
    )


# 创建依赖项，使用创建的（伪）工具函数，该函数接收 str 类型的令牌，并返回 Pydantic 的 User 模型
async def get_current_user(token: str = Depends(oauth2_scheme)):  # 使用 oauth2_scheme 作为依赖项
    user = fake_decode_token(token)
    return user


@app.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_user)):  # 依赖注入；注意，此处把 current_user 的类型声明为 Pydantic 的 User 模型，有助于在函数内部使用代码补全和类型检查
    return current_user