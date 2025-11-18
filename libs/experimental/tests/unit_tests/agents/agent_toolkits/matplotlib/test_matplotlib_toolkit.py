from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from langchain.agents import AgentType
from langchain_core.tools import BaseTool

from langchain_experimental.agents.agent_toolkits.matplotlib.base import (
    create_matplotlib_agent,
)


class DummyTool(BaseTool):
    name: str = "python_repl"
    description: str = "A dummy tool for testing"

    def _run(self, *args: Any, **kwargs: Any) -> str:
        return "success"

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()


@pytest.fixture
def mock_df() -> pd.DataFrame:
    """Return a small dummy DataFrame."""
    return pd.DataFrame(
        {"Age": [22, 30, 25], "Fare": [7.25, 8.05, 10.5], "Pclass": [3, 1, 2]}
    )


@pytest.fixture
def mock_llm() -> MagicMock:
    """Mock LLM to avoid API calls."""
    llm = MagicMock()
    llm.invoke = MagicMock(return_value={"output": "mocked output"})
    return llm


def test_create_agent_requires_opt_in_security(
    mock_llm: MagicMock, mock_df: pd.DataFrame
) -> None:
    """Ensure ValueError is raised if allow_dangerous_code=False."""
    with pytest.raises(ValueError, match="allow_dangerous_code=True"):
        create_matplotlib_agent(llm=mock_llm, df=mock_df)


@patch(
    "langchain_experimental.agents.agent_toolkits.matplotlib.base.PythonAstREPLTool",
    new=DummyTool,
)
def test_create_agent_react(mock_llm: MagicMock, mock_df: pd.DataFrame) -> None:
    """Test ReAct agent creation path."""
    agent = create_matplotlib_agent(
        llm=mock_llm,
        df=mock_df,
        allow_dangerous_code=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    assert agent is not None


@patch(
    "langchain_experimental.agents.agent_toolkits.matplotlib.base.PythonAstREPLTool",
    new=DummyTool,
)
def test_create_agent_tool_calling(mock_llm: MagicMock, mock_df: pd.DataFrame) -> None:
    """Test agent creation with tool-calling type."""
    agent = create_matplotlib_agent(
        llm=mock_llm,
        df=mock_df,
        allow_dangerous_code=True,
        agent_type="tool-calling",
    )
    assert agent is not None


def test_invalid_dataframe_type(mock_llm: MagicMock) -> None:
    """Ensure error is raised if invalid df passed."""
    with pytest.raises(ValueError, match="Expected pandas DataFrame"):
        create_matplotlib_agent(
            llm=mock_llm,
            df="not_a_df",  # type: ignore[arg-type]
            allow_dangerous_code=True,
        )


def test_empty_dataframe_warning(mock_llm: MagicMock) -> None:
    """Warns but doesn't fail on empty DataFrame."""
    empty_df = pd.DataFrame()
    with pytest.warns(UserWarning, match="empty"):
        create_matplotlib_agent(
            llm=mock_llm,
            df=empty_df,
            allow_dangerous_code=True,
        )


def test_multiple_dataframe_support(mock_llm: MagicMock) -> None:
    """Ensure list of DataFrames works."""
    df1 = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})
    agent = create_matplotlib_agent(
        llm=mock_llm,
        df=[df1, df2],
        allow_dangerous_code=True,
    )
    assert agent is not None


def test_invalid_agent_type(mock_llm: MagicMock, mock_df: pd.DataFrame) -> None:
    """Raise ValueError for unsupported agent types."""
    with pytest.raises(ValueError, match="not supported"):
        create_matplotlib_agent(
            llm=mock_llm,
            df=mock_df,
            agent_type="unsupported-type",  # type: ignore[arg-type]
            allow_dangerous_code=True,
        )
