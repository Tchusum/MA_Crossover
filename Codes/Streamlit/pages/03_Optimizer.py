import streamlit

def app():
    streamlit.title('Optimize Short and Long Window')

    if st.button("Run Optimization"):
        st.write("Optimization Completed")

app()