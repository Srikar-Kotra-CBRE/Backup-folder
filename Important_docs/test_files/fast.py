import requests

url = "http://127.0.0.1:8000/docs#/default/ask_question_ask_post"  # Example POST endpoint URL
data = 'capital of france'

response = requests.post(url, json=data)

if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f"Error: {response.status_code}")

#Fast API

# import requests

# url = "http://127.0.0.1:8000/ask"  # Example POST endpoint URL
# data = {
#   "question": "What are the supported AWS services?",
#   "conversation_history": [
#     {
#       "role": "",
#       "content": ""
#     }
#   ]
# }

# response = requests.post(url, json=data)

# if response.status_code == 200: 
#     result = response.json()
#     print(result)
# else:
#     print(f"Error: {response.status_code} - {response.text}")

#Trimming dataset code
# # I would like to set up email services for my application. Is there a service that I can consume for this?
# # I need access to this TFE project, can you have a look?
# # What are the supported AWS services?
# # Are there any Azure services that will be supported at a later point?
# # What is the BaaS architecture for a Dedicated subscription?
# # Is there a RACI for this service?
# # What is the SLA for a restoration request?
# # What are the latest changes that happened to the tagging?
# # What is CloudHealth and tell me more about its capabilities.

# import pandas as pd

# # Load the Excel file
# df = pd.read_excel("CloudEnggAndOps2Years.xlsx", engine="openpyxl")

# # Keep only the rows from index 2995 onward
# df_trimmed = df.iloc[2999:]

# # Save the trimmed DataFrame to a new Excel file
# df_trimmed.to_excel("trimmed_dataset.xlsx", index=False)

# print("Trimmed dataset saved to 'trimmed_dataset.xlsx'.")

#dUMMY DATASET FOR RAGAS
# from datasets import Dataset

# dummy = Dataset.from_dict({
#     "user_input": ["Who wrote Pride and Prejudice?", "What is the capital of France?"],
#     "response": [
#         "Jane Austen wrote Pride and Prejudice.",
#         "The capital city of France is Paris."
#     ],
#     "ground_truth": ["Jane Austen", "Paris"],
#     "retrieved_contexts": [
#         [
#             "Jane Austen was an English novelist known for books such as Pride and Prejudice.",
#             "Pride and Prejudice is a romantic novel published in 1813."
#         ],
#         [
#             "Paris is the capital and most populous city of France.",
#             "France is located in Western Europe."
#         ]
#     ]
# })
