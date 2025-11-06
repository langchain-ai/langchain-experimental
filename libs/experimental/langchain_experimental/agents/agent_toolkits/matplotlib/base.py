from __future__ import annotations
import io
import math
import base64
import builtins
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, Union, Sequence, Literal, cast
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import os
from dotenv import load_dotenv
import builtins
from langchain_experimental.agents.agent_toolkits.matplotlib.prompt import (
    PREFIX_WITH_SINGLE_DF,
    PREFIX_WITH_MULTIPLE_DF,
    EXAMPLES_WITH_SINGLE_DF,
    EXAMPLES_WITH_MULTIPLE_DFS,
    SUFFIX_WITH_DF,
    SUFFIX_NO_DF,
    SUFFIX_WITH_MULTI_DF,
    FUNCTIONS_WITH_DF,
    FUNCTIONS_WITH_MULTI_DF,
    SINGLE_PREFIX_WITH_FUNCTIONS,
    MULTI_DF_PREFIX_FUNCTIONS
)
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.agents.openai_functions_agent.base import (
    OpenAIFunctionsAgent,
    create_openai_functions_agent,
)
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    PromptTemplate,
)

from langchain.agents import (
    AgentType,
    create_openai_tools_agent,
    create_react_agent,
    create_tool_calling_agent,
)
from langchain.agents.agent import (
    AgentExecutor,
    BaseMultiActionAgent,
    BaseSingleActionAgent,
    RunnableAgent,
    RunnableMultiActionAgent,
)
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.agents.openai_functions_agent.base import (
    OpenAIFunctionsAgent,
    create_openai_functions_agent,
)
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel, LanguageModelLike
from langchain_core.tools import BaseTool
from langchain_core.utils.interactive_env import is_interactive_env




def _get_prompt(df_list: Any, multi_flag: bool, **kwargs) -> BasePromptTemplate:
    if multi_flag:
        return _get_multi_prompt(df_list, **kwargs)
    else:
        return _get_single_prompt(df_list, **kwargs)




def _get_single_prompt(
    df: Any,
    *,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    include_df_in_prompt: Optional[bool] = True,
    number_of_head_rows: int = 5,
) -> BasePromptTemplate:
    if suffix is not None:
        suffix_to_use = suffix
    elif include_df_in_prompt:
        suffix_to_use = SUFFIX_WITH_DF
    else:
        suffix_to_use = SUFFIX_NO_DF
    prefix = prefix if prefix is not None else PREFIX_WITH_SINGLE_DF

    template = "\n\n".join([prefix,EXAMPLES_WITH_SINGLE_DF, "{tools}", FORMAT_INSTRUCTIONS, suffix_to_use])
    prompt = PromptTemplate.from_template(template)

    partial_prompt = prompt.partial()
    if "df_head" in partial_prompt.input_variables:
        df_head = str(df.head(number_of_head_rows).to_markdown())
        partial_prompt = partial_prompt.partial(df_head=df_head)
    return partial_prompt




def _get_multi_prompt(
    df_list: List[Any],
    *,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    include_df_in_prompt: Optional[bool] = True,
    number_of_head_rows: int = 5,
) -> BasePromptTemplate:
    if suffix is not None:
        suffix_to_use = suffix
    elif include_df_in_prompt:
        suffix_to_use = SUFFIX_WITH_MULTI_DF
    else:
        suffix_to_use = SUFFIX_NO_DF
    prefix = prefix if prefix is not None else PREFIX_WITH_MULTIPLE_DF

    template = "\n\n".join([prefix,EXAMPLES_WITH_MULTIPLE_DFS, "{tools}", FORMAT_INSTRUCTIONS, suffix_to_use])
    prompt = PromptTemplate.from_template(template)
    partial_prompt = prompt.partial()
    if "dfs_head" in partial_prompt.input_variables:
        dfs_head = "\n\n".join([d.head(number_of_head_rows).to_markdown() for d in df_list])
        partial_prompt = partial_prompt.partial(dfs_head=dfs_head)
    if "num_dfs" in partial_prompt.input_variables:
        partial_prompt = partial_prompt.partial(num_dfs=str(len(df_list)))
    return partial_prompt




def _get_functions_single_prompt(
    df: Any,
    *,
    prefix: Optional[str] = None,
    suffix: str = "",
    include_df_in_prompt: Optional[bool] = True,
    number_of_head_rows: int = 5,
) -> ChatPromptTemplate:
    if include_df_in_prompt:
        df_head = str(df.head(number_of_head_rows).to_markdown())
        suffix = (suffix or FUNCTIONS_WITH_DF).format(df_head=df_head)
    prefix = prefix if prefix is not None else SINGLE_PREFIX_WITH_FUNCTIONS
    system_message = SystemMessage(content=prefix + suffix)
    prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
    return prompt


def _get_functions_multi_prompt(
    df_list: Any,
    *,
    prefix: str = "",
    suffix: str = "",
    include_df_in_prompt: Optional[bool] = True,
    number_of_head_rows: int = 5,
) -> ChatPromptTemplate:
    if include_df_in_prompt:
        dfs_head = "\n\n".join([d.head(number_of_head_rows).to_markdown() for d in df_list])
        suffix = (suffix or FUNCTIONS_WITH_MULTI_DF).format(dfs_head=dfs_head)
    prefix = (prefix or MULTI_DF_PREFIX_FUNCTIONS).format(num_dfs=str(len(df_list)))
    system_message = SystemMessage(content=prefix + suffix)
    prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
    return prompt



def _get_functions_prompt(df_list: Any, multi_flag: bool, **kwargs) -> BasePromptTemplate:
    if multi_flag:
        return _get_functions_multi_prompt(df_list, **kwargs)
    else:
        return _get_functions_single_prompt(df_list, **kwargs)



def create_matplotlib_agent(
    llm: LanguageModelLike,
    df: Any,
    agent_type: Union[
        AgentType, Literal["openai-tools", "tool-calling"]
    ] = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    verbose: bool = False,
    return_intermediate_steps: bool = False,
    max_iterations: Optional[int] = 15,
    max_execution_time: Optional[float] = None,
    early_stopping_method: str = "force",
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    include_df_in_prompt: Optional[bool] = True,
    number_of_head_rows: int = 5,
    extra_tools: Sequence[BaseTool] = (),
    allow_dangerous_code: bool = False,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct a Matplotlib agent from an LLM and dataframe(s).

    This function creates a universal matplotlib agent that works with all supported
    LangChain agent types: ReAct, OpenAI Functions, OpenAI Tools, and Tool Calling.

    Security Notice:
        This agent relies on access to a python repl tool which can execute
        arbitrary code. This can be dangerous and requires a specially sandboxed
        environment to be safely used. Failure to run this code in a properly
        sandboxed environment can lead to arbitrary code execution vulnerabilities,
        which can lead to data breaches, data loss, or other security incidents.

        Do not use this code with untrusted inputs, with elevated permissions,
        or without consulting your security team about proper sandboxing!

        You must opt-in to use this functionality by setting allow_dangerous_code=True.

    Args:
        llm: Language model to use for the agent. If agent_type is "tool-calling" then
            llm is expected to support tool calling.
        df: Pandas dataframe or list of Pandas dataframes.
        agent_type: One of "tool-calling", "openai-tools", "openai-functions", or
            "zero-shot-react-description". Defaults to "zero-shot-react-description".
            "tool-calling" is recommended for newer LLMs.
        callback_manager: DEPRECATED. Pass "callbacks" key into 'agent_executor_kwargs'
            instead to pass constructor callbacks to AgentExecutor.
        prefix: Prompt prefix string.
        suffix: Prompt suffix string.
        input_variables: DEPRECATED. Input variables automatically inferred from
            constructed prompt.
        verbose: AgentExecutor verbosity.
        return_intermediate_steps: Passed to AgentExecutor init.
        max_iterations: Passed to AgentExecutor init.
        max_execution_time: Passed to AgentExecutor init.
        early_stopping_method: Passed to AgentExecutor init.
        agent_executor_kwargs: Arbitrary additional AgentExecutor args.
        include_df_in_prompt: Whether to include the first number_of_head_rows in the
            prompt. Must be None if suffix is not None.
        number_of_head_rows: Number of initial rows to include in prompt if
            include_df_in_prompt is True.
        extra_tools: Additional tools to give to agent on top of a PythonAstREPLTool.
        allow_dangerous_code: bool, default False
            This agent relies on access to a python repl tool which can execute
            arbitrary code. This can be dangerous and requires a specially sandboxed
            environment to be safely used.
            Failure to properly sandbox this class can lead to arbitrary code execution
            vulnerabilities, which can lead to data breaches, data loss, or
            other security incidents.
            You must opt in to use this functionality by setting
            allow_dangerous_code=True.

        **kwargs: DEPRECATED. Not used, kept for backwards compatibility.

    Returns:
        An AgentExecutor with the specified agent_type agent and access to
        a PythonAstREPLTool with the DataFrame(s) and any user-provided extra_tools.

    Raises:
        ValueError: If allow_dangerous_code is False, if invalid DataFrames are provided,
                   or if an unsupported agent type is specified.
        ImportError: If pandas is not installed.

    Example:
        .. code-block:: python

            from langchain_openai import ChatOpenAI
            from langchain_experimental.agents import create_matplotlib_agent
            import pandas as pd

            df = pd.read_csv("titanic.csv")
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            
            # Test with different agent types
            agent_types = [
                AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                "tool-calling",
                "openai-tools",
                AgentType.OPENAI_FUNCTIONS
            ]
            
            for agent_type in agent_types:
                agent_executor = create_matplotlib_agent(
                    llm,
                    df,
                    agent_type=agent_type,
                    verbose=True,
                    allow_dangerous_code=True
                )
                result = agent_executor.invoke({
                    "input": "Create a histogram of Age and save it as 'plot.png'"
                })
                print(f"Agent {agent_type}: {result['output']}")
        
    """
    # Validate security settings
    if not allow_dangerous_code:
        raise ValueError(
            "This agent relies on access to a python repl tool which can execute "
            "arbitrary code. This can be dangerous and requires a specially sandboxed "
            "environment to be safely used. Please read the security notice in the "
            "doc-string of this function. You must opt-in to use this functionality "
            "by setting allow_dangerous_code=True."
            "For general security guidelines, please see: "
            "https://python.langchain.com/docs/security/"
        )
    
    # Import validation
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "both pandas and matplotlib must be installed to use the matplotlib dataframe agent. "
            "Please install pandas with `pip install pandas matplotlib`."
        )
    
    # Set pandas display options for interactive environments
    if is_interactive_env():
        pd.set_option("display.max_columns", None)

    # Validate DataFrame inputs
    df_list = df if isinstance(df, list) else [df]
    for i, _df in enumerate(df_list):
        if not isinstance(_df, pd.DataFrame):
            raise ValueError(f"Expected pandas DataFrame at position {i}, got {type(_df)}")
        if _df.empty:
            warnings.warn(f"DataFrame at position {i} is empty")

    # Handle deprecated arguments
    if input_variables:
        kwargs = kwargs or {}
        kwargs["input_variables"] = input_variables
    if kwargs:
        warnings.warn(
            f"Received additional kwargs {kwargs} which are no longer supported."
        )

    # Prepare DataFrame locals for the Python REPL tool
    df_locals = {}
    if isinstance(df, list):
        for i, dataframe in enumerate(df):
            df_locals[f"df{i + 1}"] = dataframe
    else:
        df_locals["df"] = df
    
    # Add common imports to the namespace
    df_locals.update({
        "pd": pd,
        "plt": plt,
        "numpy": None,  # Will be imported if needed
        "math": math
    })
    
    # Create tools list
    tools = [PythonAstREPLTool(locals=df_locals)] + list(extra_tools)
    
    # Determine if we're working with multiple DataFrames
    multi_flag = isinstance(df, list) and len(df) > 1

    # Create agent based on type
    try:
        if agent_type == AgentType.ZERO_SHOT_REACT_DESCRIPTION:
            if include_df_in_prompt is not None and suffix is not None:
                raise ValueError(
                    "If suffix is specified, include_df_in_prompt should not be."
                )
           
            prompt = _get_prompt(
                df,
                multi_flag,
                prefix=prefix,
                suffix=suffix,
                include_df_in_prompt=include_df_in_prompt,
                number_of_head_rows=number_of_head_rows,
            )
            agent: Union[BaseSingleActionAgent, BaseMultiActionAgent] = RunnableAgent(
                runnable=create_react_agent(llm, tools, prompt),  # type: ignore
                input_keys_arg=["input"],
                return_keys_arg=["output"],
            )
        elif agent_type in (AgentType.OPENAI_FUNCTIONS, "openai-tools", "tool-calling"):
            prompt = _get_functions_prompt(
                df,
                multi_flag,
                prefix=prefix,
                suffix=suffix,
                include_df_in_prompt=include_df_in_prompt,
                number_of_head_rows=number_of_head_rows,
            )
            if agent_type == AgentType.OPENAI_FUNCTIONS:
                runnable = create_openai_functions_agent(
                    cast(BaseLanguageModel, llm), tools, prompt
                )
                agent = RunnableAgent(
                    runnable=runnable,
                    input_keys_arg=["input"],
                    return_keys_arg=["output"],
                )
            else:
                if agent_type == "openai-tools":
                    runnable = create_openai_tools_agent(
                        cast(BaseLanguageModel, llm), tools, prompt
                    )
                else:  # tool-calling
                    runnable = create_tool_calling_agent(
                        cast(BaseLanguageModel, llm), tools, prompt
                    )
                agent = RunnableMultiActionAgent(
                    runnable=runnable,
                    input_keys_arg=["input"],
                    return_keys_arg=["output"],
                )
        else:
            raise ValueError(
                f"Agent type {agent_type} not supported. Must be one of "
                "'tool-calling', 'openai-tools', 'openai-functions', or "
                "'zero-shot-react-description'."
            )
    except Exception as e:
        raise ValueError(f"Failed to create agent of type {agent_type}: {str(e)}")
    
    # Create and return AgentExecutor with robust error handling
    return AgentExecutor(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        verbose=verbose,
        return_intermediate_steps=return_intermediate_steps,
        max_iterations=max_iterations or 15,
        max_execution_time=max_execution_time,
        early_stopping_method=early_stopping_method,
        handle_parsing_errors=True,  # Critical for robust error handling
        **(agent_executor_kwargs or {}),
    )

