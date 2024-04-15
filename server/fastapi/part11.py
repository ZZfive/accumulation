'''
链接：https://fastapi.tiangolo.com/zh/tutorial/security/oauth2-jwt/
如何使用 JWT 令牌（Token）和安全密码哈希（Hash）实现真正的安全机制

JWT：
JWT是JSON Web Token的缩写，是一种将 JSON 对象编码为没有空格，且难以理解的长字符串的标准;JWT字符串没有加密，任何人都能用它回复原始信息，
但 JWT 使用了签名机制。接受令牌时，可以用签名校验令牌。使用 JWT 创建有效期为一周的令牌。第二天，用户持令牌再次访问时，仍为登录状态。
令牌于一周后过期，届时，用户身份验证就会失败。只有再次登录，才能获得新的令牌。如果用户（或第三方）篡改令牌的过期时间，因为签名不匹配会导致身份验证失败

密码哈希：
哈希是指把特定内容（本例中为密码）转换为乱码形式的字节序列（其实就是字符串）。每次传入完全相同的内容时（比如，完全相同的密码），返回的都是完全相同的乱码。但这个乱码无法转换回传入的密码。
使用密码哈希的原因是假如数据库被盗，窃贼无法获取用户的明文密码，得到的只是哈希值。这样一来，窃贼就无法在其它应用中使用窃取的密码
'''

from datetime import datetime, timedelta, timezone
from typing import Union

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext  # 用于创建密码哈希和身份校验的 PassLib 上下文
from pydantic import BaseModel

# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"  # 使用openssl rand -hex 32生成的随机密钥
ALGORITHM = "HS256"  # 指定 JWT 令牌签名算法
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # 设置令牌过期时间


fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "disabled": False,
    }
}


class Token(BaseModel):  # 令牌端点响应
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Union[str, None] = None


class User(BaseModel):
    username: str
    email: Union[str, None] = None
    full_name: Union[str, None] = None
    disabled: Union[bool, None] = None


class UserInDB(User):
    hashed_password: str


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()


def verify_password(plain_password, hashed_password):  # 用于校验接收的密码是否匹配存储的哈希值
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password): # 用于哈希用户的密码
    return pwd_context.hash(password)


def get_user(db, username: str):  # 用于身份验证，并返回用户
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):  # 用于生成新的访问令牌
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])  #  JWT 令牌
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception  # 如果令牌无效，则直接返回 HTTP 错误
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@app.post("/token")
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> Token:
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)  # 用令牌过期时间创建 timedelta 对象
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")  # 创建并返回真正的 JWT 访问令牌


@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user


@app.get("/users/me/items/")
async def read_own_items(current_user: User = Depends(get_current_active_user)):
    return [{"item_id": "Foo", "owner": current_user.username}]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("part11:app", host="0.0.0.0", port=8000, reload=True)