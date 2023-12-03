

import streamlit as st
import subprocess
import streamlit as st


st.title("Stress Detector")



from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf

# Load the saved model
model = TFAutoModelForSequenceClassification.from_pretrained('stress_model')

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('stress_model')

# Tokenize input text
text = st.text_input("How do you feel", "")
inputs = tokenizer(text, truncation=True, padding=True, return_tensors="tf")

# Make predictions
outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
predicted_stress = tf.argmax(outputs.logits, axis=1).numpy()[0]

# Print predicted stress level
print(f"Predicted stress level: {predicted_stress}")
flag=1
if st.button("Check Stress"):
    if (predicted_stress==0):
      st.write("You are not stressed....")
    else:
      subprocess.Popen(["streamlit", "run", "app.py"])
      
      


      
      


      

      
      








