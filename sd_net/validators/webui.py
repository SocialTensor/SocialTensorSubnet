import streamlit as st
import redis
import json
import os
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_LIST = os.getenv("REDIS_LIST")

# Connect to Redis
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

st.title('User Request Queue')

# User input fields
prompt = st.text_input('Prompt')
access_key = st.text_input('Access Key')

# When the user submits the form
if st.button('Submit Request'):
    if prompt and access_key:
        # Create a request object
        request_data = {
            'prompt': prompt,
            'access_key': access_key
        }

        # Push the request to Redis
        redis_client.lpush(REDIS_LIST, json.dumps(request_data))

        st.success('Request submitted successfully!')
    else:
        st.error('Please fill all the fields')
