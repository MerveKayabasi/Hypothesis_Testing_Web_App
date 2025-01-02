import streamlit as st
import pandas as pd
import io

st.title('Hypothesis Testing and Statistical Analysis Application')
st.write('This application allows you to perform hypothesis testing and various statistical analyses on the uploaded dataset or manual input.')

# Data input method selection
st.subheader('Select data input method:')
input_method = st.radio("Choose input method:", ["Upload CSV File", "Enter Data Manually"])

if input_method == "Upload CSV File":
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep=';', engine='python', header=None).fillna('')
        st.write("**Uploaded Data:**")
        st.dataframe(df.style.set_table_attributes("style='display:inline'"))

elif input_method == "Enter Data Manually":
    st.markdown("**Enter Data Manually**")
    group_count = st.number_input("How many data groups will you enter?", min_value=1, step=1, value=1)
    
    manual_data = {}
    for i in range(group_count):
        group_name = f"Group {i+1}"
        manual_data[group_name] = st.text_area(f"Enter {group_name} data separated by commas (e.g., 104.96, 98.54, 106.37)")
    
    if st.button("Enter Data"):
        if any(manual_data.values()):
            try:
                processed_data = {}
                max_length = 0
                for group, data in manual_data.items():
                    if data:
                        processed_data[group] = [float(x.strip()) for x in data.split(",")]
                        max_length = max(max_length, len(processed_data[group]))
                
                for group in processed_data:
                    if len(processed_data[group]) < max_length:
                        processed_data[group].extend([''] * (max_length - len(processed_data[group])))
                
                df = pd.DataFrame(processed_data)
                df = df.applymap(lambda x: str(x).rstrip('0').rstrip('.') if isinstance(x, float) else x)
                st.write("**Entered Manual Data:**")
                st.dataframe(df.style.set_table_attributes("style='display:inline'"))
            except ValueError:
                st.error("Please enter only numeric values!")