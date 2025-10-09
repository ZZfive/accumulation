import os

from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_sync_playwright_browser

from based_on_openai_model import ChatOpenRouter

# 初始化浏览器
sync_browser = create_sync_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
tools = toolkit.get_tools()

prompt = hub.pull("hwchase17/openai-tools-agent")

model = ChatOpenRouter(model="openrouter/sonoma-dusk-alpha")

agent = create_openai_tools_agent(model, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


if __name__ == "__main__":
    command = {"input": "访问以下链接，帮我总结具体内容：https://github.com/ZZfive/ComfyChat/blob/main/README.md"}

    # 执行
    response = agent_executor.invoke(command)
    print(response)