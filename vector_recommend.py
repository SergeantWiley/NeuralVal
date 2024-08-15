import streamlit as st

def show_vector():
    st.title("Vector Recommend")

    st.write("""
    The **Vector Recommend Library** allows for a wide range of applications of using vectors as a quick and easy method of suggesting content
             to users.

    ### Vector Recommend Mathematics

    """)
    st.write("""1. **User Vector Normalization**:""")

    st.latex(r"""U_{i}=\sum_{i}^{n}u_{i}=1""")

    st.write("""2. **User Prefernces adjustments**:""")
    
    st.latex(r"""Nu_{i}=u_{i}+\alpha\cdot U_{i}""")

    st.write("""3. **Distance Between Vectors**:""")

    st.latex(r"""d = d(U|M_{i})=\sqrt{\sum_{j=i}^{n}(u_{i}-m_{ij})^{2}}""")    

