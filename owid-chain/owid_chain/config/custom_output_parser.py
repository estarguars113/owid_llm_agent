from langchain.output_parsers import StructuredOutputParser, ResponseSchema


response_schemas = [
    ResponseSchema(name="answer", description="answer to the user's question"),
    ResponseSchema(name="source", description="source used to answer the user's question, should be a website."),
    ResponseSchema(name="data", description="data to support the responer, should be a json or a pandas dataframe json representation.")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)