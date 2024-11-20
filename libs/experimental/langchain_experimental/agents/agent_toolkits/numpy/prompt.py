# flake8: noqa
PREFIX = """
You are working with a NumPy array in Python. The name of the array is `arr`.
You should use the tools below to answer the question posed of you:"""

MULTI_ARR_PREFIX = """
You are working with {num_arrays} NumPy arrays in Python named arr1, arr2, etc. You 
should use the tools below to answer the question posed of you:"""

SUFFIX_NO_ARR = """
Begin!
Question: {input}
{agent_scratchpad}"""

SUFFIX_WITH_ARR = """
This is the result of `print(arr)`:
{arr_head}

Begin!
Question: {input}
{agent_scratchpad}"""

SUFFIX_WITH_MULTI_ARR = """
This is the result of `print(arr)` for each array:
{arrays_head}

Begin!
Question: {input}
{agent_scratchpad}"""

PREFIX_FUNCTIONS = """
You are working with a NumPy array in Python. The name of the array is `arr`."""

MULTI_ARR_PREFIX_FUNCTIONS = """
You are working with {num_arrays} NumPy arrays in Python named arr1, arr2, etc."""

FUNCTIONS_WITH_ARR = """
This is the result of `print(arr)`:
{arr_head}"""

FUNCTIONS_WITH_MULTI_ARR = """
This is the result of `print(arr)` for each array:
{arrays_head}"""
