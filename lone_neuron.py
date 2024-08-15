import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def show_neuron():
    st.title("Lone Neuron")
    st.subheader("A Simple Neural Network")

    st.write('''Take a peek inside the structure of neural networks using Lone Neuron. 
    See what happens behind the scenes as a neuron transforms incoming data.''')
    st.write('''
    - **Step 1**: Create the neuron layer using the slider
    - **Step 2**: Adjust the entropy, which controls the randomness of weights and biases
    - **Step 3**: Adjust the data variation, which shows how the training data is offset from the target data
    - **Step 4**: Use the sliders to minimize the MSE between the transformed data and the target data
    ''')

    # Step 1: Number of neurons
    num_neurons = st.slider('Select the number of neurons in the layer', 1, 10, 3)
    entropy = st.slider('Entropy (randomness of weights and biases)', 1.5, 10.0, 1.5, 0.5)
    target_variation = st.slider('Data Variation (Offset from Target)', 1.0, 2.0, 1.0, 0.1)
    sample_size = st.slider('Number of samples', 1, 10, 1, 1)

    # Adjust target variation based on sample size
    target_variation +=  target_variation / np.random.uniform(1, sample_size)
    weights = np.random.uniform(0, entropy, num_neurons)
    biases = np.random.uniform(0, entropy, num_neurons)
    mse_values = []

    # Compute MSE for each sample
    for _ in range(sample_size):
        training_data = np.random.uniform(0, entropy, num_neurons)
        target_data = training_data + target_variation

        # Process training data through the neurons
        transformed_data = training_data * weights + biases
        mse = np.mean((transformed_data - target_data) ** 2)
        mse_values.append(mse)

    # Plot the MSE values as a bar graph
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(mse_values)), mse_values, color='skyblue')

    # Adding labels and title
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('MSE Value')
    ax.set_title('MSE Values by Sample Index')
    ax.set_xticks(range(len(mse_values)))

    # Display the plot in Streamlit
    st.pyplot(fig)


