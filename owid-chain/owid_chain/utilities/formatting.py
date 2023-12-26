import streamlit as st
import pandas as pd
import json

def write_response(st, response_dict: dict):
    """
    Write a response from an agent to a Streamlit app.

    Args:
        response_dict: The response from the agent.

    Returns:
        None.
    """

    # Check if the response is a bar chart.
    if "bar" in response_dict:
        data = response_dict["bar"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.bar_chart(df)

    # Check if the response is a line chart.
    if "line" in response_dict:
        data = response_dict["line"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.line_chart(df)

    # Check if the response is a table.
    if "df" in response_dict:
        data = json.loads(response_dict["df"])
        df = pd.DataFrame(data)
        st.write(df)


    # Check if the response is an answer.
    if "metadata" in response_dict:
        st.write(response_dict["metadata"])

    
    # Check if the response is an answer.
    if "error" in response_dict:
        st.write(response_dict["error"])