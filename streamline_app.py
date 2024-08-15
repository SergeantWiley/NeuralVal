import streamlit as st
from home import show_home
from cpu import show_cpu
from vector_recommend import show_vector
from lone_neuron import show_neuron
from neural_val import show_neural_val
# Define the navigation menu
pages = {
    "Home": show_home,
    "Neural Val": show_neural_val,
    "4x CPU": show_cpu,
    "Vector Recommend": show_vector,
    "Lone Neuron": show_neuron
}

# Sidebar for navigation
st.sidebar.title("Portfolio Navigation")
selected_page = st.sidebar.radio("Explore Projects", list(pages.keys()))

# Display the selected page
pages[selected_page]()
