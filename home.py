import streamlit as st

def show_home():
    st.title("Steven Johnson")
    st.subheader("Data Analytics & Machine Learning Specialist")
    
    st.write('''Greetings! I'm Steven Johnson, a dedicated developer with a passion for Data Analytics and Machine Learning. 
    I currently teach modern technology and programming to young students at The Coder School.''')
    
    st.markdown("**Connect with me:**")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/sj-lid/) | [GitHub](https://github.com/SergeantWiley/) | [Email](mailto:steven@example.com)")
    
    st.subheader("About Me")
    st.write('''
    My journey in coding began in 2016, and since then, I’ve consistently pushed the boundaries of my skills and knowledge:
    
    - **2017**: Collaborated on a team to design a pathfinding algorithm for navigating unpredictable environments.
    - **2018**: Launched my first profitable business selling 3D prints, while also working part-time at my first job.
    - **2019**: Delved deep into linear algebra, exploring multi-dimensional mathematics.
    - **2020**: Expanded my entrepreneurial ventures by buying and reselling candy.
    - **2021**: Discovered my passion for Data Science, extensively studying internet traffic and honing my Python skills.
    - **2022**: Joined Aramark, where I developed projects like NOCAP and STARS.
    - **2023**: Transitioned to full-time skill development, culminating in my completion of Coding Temple and a newfound passion for teaching.
    
    Over the past eight years, I’ve built a foundation of knowledge and experience that I’m eager to share with others, helping them avoid the pitfalls I encountered along the way.
    ''')
    st.subheader("Companies I've Worked With")
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Aramark")
        st.write("One of the largest Food Companies in the world")
        st.image("aramark.png", width=300, caption="Aramark")

    with col2:
        st.header("The Coder School")
        st.write("Private Education for young students coding.")
        st.image("the-coder-school.png", width=100, caption="The Coder School")
