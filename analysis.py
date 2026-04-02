import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=key)

model = genai.GenerativeModel('gemini-2.5-flash')


def generate_summary(results_df):
    prompt = f'''You are a data scientist expert. Here are the model results:
    {results_df.to_string(index=False)}
    
    1. Identify the best model
    2. Explain why it is the best model
    3. Summarize the performance of the model
    '''
    
    response = model.generate_content(prompt)
    return response.text


# 
def generate_improvement_suggestions(results_df):
    prompt = f'''You are a data scientist expert. Here are the model results:
    {results_df.to_string(index=False)}
    
    Suggest:
    - Ways to improve the model performance
    - Better algorithms if needed
    - Data preprocessing improvements
    '''
    
    response = model.generate_content(prompt)
    return response.text