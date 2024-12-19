import sys

import boto3
import pytest
from moto import mock_aws

from langchain_experimental.agents import create_pandas_dataframe_agent
from tests.unit_tests.fake_llm import FakeLLM


@pytest.mark.requires("pandas", "tabulate")
@pytest.mark.skipif(sys.version_info < (3, 9), reason="requires python3.9 or higher")
@mock_aws
def test_create_pandas_dataframe_agent() -> None:
    import pandas as pd

    # Set up AWS region
    boto3.setup_default_session(region_name="us-east-1")

    with pytest.raises(ValueError):
        create_pandas_dataframe_agent(
            FakeLLM(), pd.DataFrame(), allow_dangerous_code=False
        )

    create_pandas_dataframe_agent(FakeLLM(), pd.DataFrame(), allow_dangerous_code=True)
    create_pandas_dataframe_agent(
        FakeLLM(), [pd.DataFrame(), pd.DataFrame()], allow_dangerous_code=True
    )
