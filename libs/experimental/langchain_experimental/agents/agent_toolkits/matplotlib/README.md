cat <<'EOF' > lib/langchain_experimental/agents/agent_toolkits/matplotlib/README.md
# 🧠 Matplotlib Agent Toolkit

The **Matplotlib Agent Toolkit** extends LangChain with intelligent data visualization capabilities using **Matplotlib** and **Python REPL** tools.  
It enables agents to understand your data and **generate, execute, and explain plots** — all from natural language instructions.

---

## 🚀 Overview

This toolkit lets a language model:
- Analyze one or multiple Pandas DataFrames  
- Generate valid Python + Matplotlib plotting code  
- Execute it safely in a sandboxed REPL  
- Save or display visualizations automatically  

It supports all major LangChain agent types, including:
- `AgentType.ZERO_SHOT_REACT_DESCRIPTION`
- `AgentType.OPENAI_FUNCTIONS`
- `"openai-tools"`
- `"tool-calling"`

---

## ⚙️ Installation

Clone the experimental repo and install dependencies:

```bash
git clone https://github.com/langchain-ai/langchain-experimental.git
cd langchain-experimental
pip install -e .
```


Ensure the following are installed
```bash
pip install pandas matplotlib langchain python-dotenv
```

## Example Usage:
``` bash
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits.matplotlib.base import create_matplotlib_agent
import pandas as pd
from dotenv import load_dotenv

# Load your OpenAI API key
load_dotenv()

# Load a sample dataset
df = pd.read_csv("https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv")

# Initialize an LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Create a Matplotlib-powered agent
agent = create_matplotlib_agent(
    llm=llm,
    df=df,
    verbose=True,
    allow_dangerous_code=True,  # ⚠️ Enables Python execution for plotting
)

# Natural language plotting query
query = """
Create a scatter plot of Age vs Fare, color-coded by passenger class.
Save the figure as 'titanic_scatter.png'.
"""

# Run the agent
result = agent.invoke({"input": query})

print("Agent output:", result.get("output", ""))
print("✅ Plot generated and saved as titanic_scatter.png")

```
After Execution you will have a chart named titanic_scatter.png


## Agent Creator:

``` bash

agent = create_matplotlib_agent(
    llm=llm,
    df=df,
    verbose=True,
    allow_dangerous_code=True,  
)
```


## 🧩 Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `llm` | `LanguageModelLike` | Any LLM supporting tool-calling (e.g. `ChatOpenAI`) |
| `df` | `pd.DataFrame` or `List[pd.DataFrame]` | Dataset(s) for visualization |
| `agent_type` | `AgentType` or `str` | Type of agent (`react`, `openai-functions`, etc.) |
| `allow_dangerous_code` | `bool` | Must be `True` to enable Python REPL execution |
| `include_df_in_prompt` | `bool` | Whether to embed sample DataFrame rows in the prompt |
| `extra_tools` | `Sequence[BaseTool]` | Add additional LangChain tools if needed |


## 🧠 How It Works

Under the hood, `create_matplotlib_agent()`:

- Builds a customized prompt based on your DataFrame(s)
- Uses a Python REPL tool (`PythonAstREPLTool`) for safe code execution
- Integrates Matplotlib and Pandas contexts for inline plotting
- Returns an `AgentExecutor` ready to interpret natural language queries into plots

---

## 🧪 Running Tests

To verify functionality:

```bash
pytest tests/test_matplotlib_toolkit.py -v


