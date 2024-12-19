from typing import Any, Dict, List, Optional
from langchain.agents.agent import AgentExecutor
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.chains.llm import LLMChain
from langchain_core.callbacks.base import BaseCallbackManager
from langchain_core.language_models import BaseLLM
from langchain.prompts import PromptTemplate
from boto3 import client
from datetime import datetime, timedelta

from langchain_experimental.agents.agent_toolkits.cloudwatch.prompt import PREFIX, SUFFIX


def _validate_cloudwatch_client(client: Any) -> bool:
    """Validates the CloudWatch client."""
    try:
        # Basic check to ensure the client is valid
        return isinstance(client, type(client))
    except Exception:
        return False

def fetch_cloudwatch_logs(
    cloudwatch_client: Any, log_group_name: str, start_time: datetime, end_time: datetime
) -> str:
    """Fetches logs from AWS CloudWatch."""
    try:
        response = cloudwatch_client.filter_log_events(
            logGroupName=log_group_name,
            startTime=int(start_time.timestamp() * 1000),
            endTime=int(end_time.timestamp() * 1000),
        )
        error_keywords = ["ERROR", "Exception", "Traceback"]
        logs = "\n".join(
            event["message"] for event in response.get("events", [])
            if any(keyword in event["message"] for keyword in error_keywords)
        )
        return logs
    except Exception as e:
        raise ValueError(f"Error fetching logs: {e}")

def create_cloudwatch_logs_agent(
    llm: BaseLLM,
    cloudwatch_client: Any,
    log_group_name: str,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = PREFIX,
    suffix: str = SUFFIX,
    input_variables: Optional[List[str]] = None,
    verbose: bool = False,
    return_intermediate_steps: bool = False,
    max_iterations: Optional[int] = 15,
    max_execution_time: Optional[float] = None,
    early_stopping_method: str = "force",
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> AgentExecutor:
    """
    Construct an AWS CloudWatch logs analyzer agent using an LLM.

    """
    if not _validate_cloudwatch_client(cloudwatch_client):
        raise ValueError("Invalid CloudWatch client provided.")

    if input_variables is None:
        input_variables = ["logs", "input", "agent_scratchpad"]

    # Fetch recent logs
    now = datetime.utcnow()
    ten_minutes_ago = now - timedelta(minutes=10)
    logs = fetch_cloudwatch_logs(cloudwatch_client, log_group_name, ten_minutes_ago, now)

    # Prompt and tools
    tools = []
    prompt = ZeroShotAgent.create_prompt(
        tools, prefix=prefix, suffix=suffix, input_variables=input_variables
    )
    partial_prompt = prompt.partial(logs=logs)
    llm_chain = LLMChain(
        llm=llm,
        prompt=partial_prompt,
        callback_manager=callback_manager,
    )
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(  # type: ignore[call-arg]
        llm_chain=llm_chain,
        allowed_tools=tool_names,
        callback_manager=callback_manager,
        **kwargs,
    )
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        verbose=verbose,
        return_intermediate_steps=return_intermediate_steps,
        max_iterations=max_iterations,
        max_execution_time=max_execution_time,
        early_stopping_method=early_stopping_method,
        **(agent_executor_kwargs or {}),
    )
