"""Agent for working with NumPy objects."""


import warnings
from typing import Any, Dict, List, Literal, Optional, Sequence, Union, cast

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
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    PromptTemplate,
)
from langchain_core.tools import BaseTool
from langchain_core.utils.interactive_env import is_interactive_env

from langchain_experimental.agents.agent_toolkits.numpy.prompt import (
    FUNCTIONS_WITH_ARR,
    FUNCTIONS_WITH_MULTI_ARR,
    MULTI_ARR_PREFIX,
    MULTI_ARR_PREFIX_FUNCTIONS,
    PREFIX,
    PREFIX_FUNCTIONS,
    SUFFIX_NO_ARR,
    SUFFIX_WITH_ARR,
    SUFFIX_WITH_MULTI_ARR,
)
from langchain_experimental.tools.python.tool import PythonAstREPLTool


def _get_multi_prompt(
    arrays: List[Any],
    *,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    include_arr_in_prompt: Optional[bool] = True,
    number_of_head_elements: int = 5,
) -> BasePromptTemplate:
    if suffix is not None:
        suffix_to_use = suffix
    elif include_arr_in_prompt:
        suffix_to_use = SUFFIX_WITH_MULTI_ARR
    else:
        suffix_to_use = SUFFIX_NO_ARR
    prefix = prefix if prefix is not None else MULTI_ARR_PREFIX

    template = "\n\n".join([prefix, "{tools}", FORMAT_INSTRUCTIONS, suffix_to_use])
    prompt = PromptTemplate.from_template(template)
    partial_prompt = prompt.partial()
    if "arrays_head" in partial_prompt.input_variables:
        arrays_head = "\n\n".join([str(arr[:number_of_head_elements]) for arr in arrays])
        partial_prompt = partial_prompt.partial(arrays_head=arrays_head)
    if "num_arrays" in partial_prompt.input_variables:
        partial_prompt = partial_prompt.partial(num_arrays=str(len(arrays)))
    return partial_prompt


def _get_single_prompt(
    arr: Any,
    *,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    include_arr_in_prompt: Optional[bool] = True,
    number_of_head_elements: int = 5,
) -> BasePromptTemplate:
    if suffix is not None:
        suffix_to_use = suffix
    elif include_arr_in_prompt:
        suffix_to_use = SUFFIX_WITH_ARR
    else:
        suffix_to_use = SUFFIX_NO_ARR
    prefix = prefix if prefix is not None else PREFIX

    template = "\n\n".join([prefix, "{tools}", FORMAT_INSTRUCTIONS, suffix_to_use])
    prompt = PromptTemplate.from_template(template)

    partial_prompt = prompt.partial()
    if "arr_head" in partial_prompt.input_variables:
        arr_head = str(arr[:number_of_head_elements])
        partial_prompt = partial_prompt.partial(arr_head=arr_head)
    return partial_prompt


def _get_prompt(arr: Any, **kwargs: Any) -> BasePromptTemplate:
    return (
        _get_multi_prompt(arr, **kwargs)
        if isinstance(arr, list)
        else _get_single_prompt(arr, **kwargs)
    )


def _get_functions_single_prompt(
    arr: Any,
    *,
    prefix: Optional[str] = None,
    suffix: str = "",
    include_arr_in_prompt: Optional[bool] = True,
    number_of_head_elements: int = 5,
) -> ChatPromptTemplate:
    if include_arr_in_prompt:
        arr_head = str(arr[:number_of_head_elements])
        suffix = (suffix or FUNCTIONS_WITH_ARR).format(arr_head=arr_head)
    prefix = prefix if prefix is not None else PREFIX_FUNCTIONS
    system_message = SystemMessage(content=prefix + suffix)
    prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
    return prompt


def _get_functions_multi_prompt(
    arrays: Any,
    *,
    prefix: str = "",
    suffix: str = "",
    include_arr_in_prompt: Optional[bool] = True,
    number_of_head_elements: int = 5,
) -> ChatPromptTemplate:
    if include_arr_in_prompt:
        arrays_head = "\n\n".join([str(arr[:number_of_head_elements]) for arr in arrays])
        suffix = (suffix or FUNCTIONS_WITH_MULTI_ARR).format(arrays_head=arrays_head)
    prefix = (prefix or MULTI_ARR_PREFIX_FUNCTIONS).format(num_arrays=str(len(arrays)))
    system_message = SystemMessage(content=prefix + suffix)
    prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
    return prompt


def _get_functions_prompt(arr: Any, **kwargs: Any) -> ChatPromptTemplate:
    return (
        _get_functions_multi_prompt(arr, **kwargs)
        if isinstance(arr, list)
        else _get_functions_single_prompt(arr, **kwargs)
    )


def create_numpy_array_agent(
    llm: LanguageModelLike,
    arr: Any,
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
    include_arr_in_prompt: Optional[bool] = True,
    number_of_head_elements: int = 5,
    extra_tools: Sequence[BaseTool] = (),
    allow_dangerous_code: bool = False,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct a NumPy agent from an LLM and array(s).

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
        arr: NumPy array or list of NumPy arrays.
        agent_type: One of "tool-calling", "openai-tools", "openai-functions", or
            "zero-shot-react-description". Defaults to "zero-shot-react-description".
            "tool-calling" is recommended over the legacy "openai-tools" and
            "openai-functions" types.
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
        include_arr_in_prompt: Whether to include the first number_of_head_elements in the
            prompt. Must be None if suffix is not None.
        number_of_head_elements: Number of initial elements to include in prompt if
            include_arr_in_prompt is True.
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
        a PythonAstREPLTool with the array(s) and any user-provided extra_tools.

    Example:
        .. code-block:: python

            from langchain_openai import ChatOpenAI
            from langchain_experimental.agents import create_numpy_array_agent
            import numpy as np

            arr = np.array([1, 2, 3, 4, 5])
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            agent_executor = create_numpy_array_agent(
                llm,
                arr,
                agent_type="tool-calling",
                verbose=True
            )

    """
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

    import numpy as np

    for _arr in arr if isinstance(arr, list) else [arr]:
        if not isinstance(_arr, np.ndarray):
            raise ValueError(f"Expected NumPy array, got {type(_arr)}")

    if input_variables:
        kwargs = kwargs or {}
        kwargs["input_variables"] = input_variables
    if kwargs:
        warnings.warn(
            f"Received additional kwargs {kwargs} which are no longer supported."
        )

    arr_locals = {}
    if isinstance(arr, list):
        for i, array in enumerate(arr):
            arr_locals[f"arr{i + 1}"] = array
    else:
        arr_locals["arr"] = arr
    tools = [PythonAstREPLTool(locals=arr_locals)] + list(extra_tools)

    if agent_type == AgentType.ZERO_SHOT_REACT_DESCRIPTION:
        if include_arr_in_prompt is not None and suffix is not None:
            raise ValueError(
                "If suffix is specified, include_arr_in_prompt should not be."
            )
        prompt = _get_prompt(
            arr,
            prefix=prefix,
            suffix=suffix,
            include_arr_in_prompt=include_arr_in_prompt,
            number_of_head_elements=number_of_head_elements,
        )
        agent: Union[BaseSingleActionAgent, BaseMultiActionAgent] = RunnableAgent(
            runnable=create_react_agent(llm, tools, prompt),  # type: ignore
            input_keys_arg=["input"],
            return_keys_arg=["output"],
        )
    elif agent_type in (AgentType.OPENAI_FUNCTIONS, "openai-tools", "tool-calling"):
        prompt = _get_functions_prompt(
            arr,
            prefix=prefix,
            suffix=suffix,
            include_arr_in_prompt=include_arr_in_prompt,
            number_of_head_elements=number_of_head_elements,
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
            else:
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
            f"Agent type {agent_type} not supported at the moment. Must be one of "
            "'tool-calling', 'openai-tools', 'openai-functions', or "
            "'zero-shot-react-description'."
        )
    return AgentExecutor(
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
