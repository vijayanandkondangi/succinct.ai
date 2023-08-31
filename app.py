
"""
Purpose:
This is the wrapper function to invoke the succinct.ai app

Input:
NA

Output:
NA

Version History:
Vijayanand Kondangi     31 Aug 2023     Created

"""

# Load dependent libraries & modules
import streamlit as st
from PIL import Image
import torch
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration

# Initialize T5 model & tokenizer
T5_Model = T5ForConditionalGeneration.from_pretrained('t5-small')
T5_Tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Initialize the session state key
st.session_state.key = ""

# Set the top banner of the app
topbanner = Image.open('logo.jpg')
st.image(topbanner)

# Configure the expander bar on About app
with st.expander('About app'):
    st.write('''
    This app provides a concise, succinct summary of the given text. It takes 
    in the source text and the compression ratio as inputs and outputs a 
    concise summary. Created by Vijayanand Kondangi.
    ''')

# Initialize variables
input_text = """"""
final_summary_caps = """"""
compression_ratio = 0.0

input_text = st.text_area('Enter Text to Summarize: ')

# Check input text and enable the button
if (input_text != ""):
  run = st.button('Go')
  # Check button click
  if run:
    st.session_state.key = 'Started'

# Check if session has started
if (st.session_state.key == 'Started' and input_text != None):

  # Initialize and activate wait mode
  with st.spinner('Summarizing the text, please wait...⏳'):

    # Compose the summary string
    summary_string = "summarize: " + input_text

    input_text_length = len(input_text.split())
    max_length = 100
    min_length = 80
      
    # Encoding the summary string in to Ids for summarization
    inputs = T5_Tokenizer.encode(summary_string, return_tensors='pt', truncation=True)
    summary_ids = T5_Model.generate(inputs,
                                    num_beams = 6,
                                    no_repeat_ngram_size = 0,
                                    min_new_tokens = min_length,
                                    max_new_tokens = max_length,
                                    early_stopping=True)

    # Decoding the Ids from summarization to string
    final_summary = T5_Tokenizer.decode(summary_ids[0][1:])
    final_summary = final_summary.replace("</s>", "")

    # Convert 1st letter of every sentence in the summary to caps
    for z in range(len(final_summary)):

      # Check for a full-stop followed by a space
      if ((final_summary[z-2] == '.' and final_summary[z-1] == ' ') or (z == 0)):
        final_summary_caps = final_summary_caps + final_summary[z].capitalize()
      else:
        final_summary_caps = final_summary_caps + final_summary[z]

    st.write(final_summary_caps)

input_text = ""
st.session_state.key = 'Ended'

#--------------------------------------------------------------------------------
# End of App UI build
#--------------------------------------------------------------------------------
