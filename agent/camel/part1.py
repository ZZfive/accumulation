

from dotenv import load_dotenv

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.agents import ChatAgent

load_dotenv()

model = ModelFactory.create(
    model_platform=ModelPlatformType.MOONSHOT,
    model_type=ModelType.MOONSHOT_V1_8K
)

sys_msg = 'You are a curious stone wondering about the universe.'


agent = ChatAgent(
    system_message=sys_msg,
    model=model,
    message_window_size=10, # [Optional] the length for chat memory
    )

# Define a user message
usr_msg = 'what is information in your mind?'

# Sending the message to the agent
response = agent.step(usr_msg)

# Check the response (just for illustrative purpose)
print(response.msgs[0].content)