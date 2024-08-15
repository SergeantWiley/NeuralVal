import streamlit as st

def show_neural_val():
    st.title("Neural Val")
    st.subheader("Object Detecton Library")
    
    st.write('''Neural Val is a object detection library meant to assist streamlining ML object detection library. No need to code datasets
             or structure neural networks. Neural Val does all that for you. Its great trying to prototype or present proof of concept without
             needing to spend hours writing code.''')
    
    st.markdown("**Install here:**")
    st.markdown("[GitHub](https://github.com/SergeantWiley/NeuralVal)")
    
    st.subheader("Example of object detection")
    st.write("Model trained on 700 images")
    st.image("NV Demo.png", width=300, caption="Test Image 683 with 78 percent confidence")
