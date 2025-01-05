import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, friedmanchisquare
from statsmodels.stats.contingency_tables import mcnemar

# Sidebar header and logo
st.sidebar.image("tedu_logo.png", use_container_width=True) 
st.sidebar.title("ADS 511: Statistical Inference Methods")
st.sidebar.write("Developed by: Merve Kayabasi")

st.sidebar.title("Hypothesis Testing Road Map")
st.sidebar.image("Mind map (1).jpeg",use_container_width=True)

st.title('Hypothesis Testing and Statistical Analysis Application')
st.write('This application allows you to perform hypothesis testing and various statistical analyses on the uploaded dataset or manual input.')

# Data input method selection
st.markdown("<h2 style='text-align: center; color: #4a90e2;'>Select data input method</h2>", unsafe_allow_html=True)
input_method = st.radio("Choose input method:", ["Upload CSV File", "Enter Data Manually"])

if input_method == "Upload CSV File":
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep=';', engine='python', header=None).fillna('')
        st.session_state.df = df  # Keep CSV data in session_state

if input_method == "Enter Data Manually":
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
                st.session_state.df = df
            except ValueError:
                st.error("Please enter only numeric values!")

# Show the table after the data has been entered or uploaded
if 'df' in st.session_state:
    st.write("**Your Data:**")
    st.dataframe(st.session_state.df.style.set_table_attributes("style='display:inline'"))

    # Show the form after the data has been entered
    with st.form(key='data_type_form'):
        st.markdown("<h2 style='text-align: center; color: #4a90e2;'>Choose your data type for analysis!</h2>", unsafe_allow_html=True)
        data_type = st.radio(
            "Choose your data type for analysis!",
            ["Select", "Numerical Data", "Categorical Data"],
            index=0
        )
        
        # Paired/Unpaired option if 2 or more groups are entered
        if len(st.session_state.df.columns) >= 2:
            st.markdown("<h2 style='text-align: center; color: #4a90e2;'>Select the relationship between data groups:</h2>", unsafe_allow_html=True)
            paired = st.radio(
                "Are the data groups paired or unpaired?",
                ["Paired", "Unpaired"],
                horizontal=True
            )
            st.session_state.paired = paired

            # Grup sayısını Submit butonundan önce ekliyoruz
            st.session_state.groups = st.number_input(
        "Enter the number of groups",
        min_value=1,
        value=1
    )
        
        submit_button = st.form_submit_button("Submit")

        
        if submit_button and data_type != "Select":
            st.write(f"You selected {data_type} for hypothesis testing.")
            st.session_state.data_type = data_type  # Store the selected data type

# Perform assumption checks if data type is Numerical
if 'df' in st.session_state and 'data_type' in st.session_state:
    if st.session_state.data_type == "Numerical Data":
        st.markdown("<h2 style='text-align: center; color: #4a90e2;'>Assumption Check</h2>", unsafe_allow_html=True)
        st.write("Checking Assumptions for Hypothesis Testing")
        data = st.session_state.df.apply(pd.to_numeric, errors='coerce')
        data = data.dropna() 

        results = []
        iid_status = False
        for col in data.columns:
            # Normality Test (Shapiro-Wilk)
            stat, p = stats.shapiro(data[col].dropna())
            result = 'Pass' if p > 0.05 else 'Fail'
            results.append({'Group': col, 'Test': 'Normality', 'P-value': p, 'Result': result})

            # Outlier Detection (Z-Score)
            z_scores = np.abs(stats.zscore(data[col].dropna()))
            outliers = np.where(z_scores > 3)
            outlier_result = 'Pass' if len(outliers[0]) == 0 else 'Fail'
            results.append({'Group': col, 'Test': 'Outliers', 'P-value': '-', 'Result': outlier_result})

        # Homogeneity of Variance (Levene's Test)
        if len(data.columns) > 1:
            levene_stat, levene_p = stats.levene(*[data[c].dropna() for c in data.columns])
            result = 'Pass' if levene_p > 0.05 else 'Fail'
            results.append({'Group': 'All Groups', 'Test': 'Variance Homogeneity', 'P-value': levene_p, 'Result': result})

        # i.i.d. Checkbox
        iid_checkbox = st.checkbox("Check if samples are independent and identically distributed")
        iid_status = iid_checkbox
        results.append({'Group': 'All Groups', 'Test': 'i.i.d. Check', 'P-value': '-', 'Result': 'Pass' if iid_checkbox else 'Fail'})
        
        result_df = pd.DataFrame(results)
        st.table(result_df)

       # Assumption Check result
        if all(result_df['Result'] == 'Pass'):
            st.success("Based on the assumptions check, this is a parametric data set.")
            st.session_state.parametric = "Parametric"
        else:
            st.error("Based on the assumptions check, this is a non-parametric data set.")
            st.session_state.parametric = "Non-Parametric"

    if st.button("Run Analysis"):
     if st.session_state.data_type == "Numerical Data":
        if st.session_state.parametric == "Parametric":
            if st.session_state.paired == "Paired":
                if st.session_state.groups == 1:
                    st.session_state.selected_test = "One Sample T-Test"
                elif st.session_state.groups == 2:
                    st.session_state.selected_test = "Paired T-Test"
                else:
                    st.session_state.selected_test = "Repeated Measures ANOVA"
            else:
                if st.session_state.groups == 2:
                    st.session_state.selected_test = "Independent T-Test"
                else:
                    st.session_state.selected_test = "One-Way ANOVA"
        else:  # Non-Parametric
            if st.session_state.paired == "Paired":
                if st.session_state.groups == 2:
                    st.session_state.selected_test = "Wilcoxon Signed Rank Test"
                else:
                    st.session_state.selected_test = "Friedman Test"
            else:
                if st.session_state.groups == 2:
                    st.session_state.selected_test = "Mann-Whitney U Test"
                else:
                    st.session_state.selected_test = "Kruskal Wallis Test"
    
    elif st.session_state.data_type == "Categorical Data":
        if st.session_state.paired == "Paired":
            if st.session_state.groups == 2:
                st.session_state.selected_test = "McNemar Test"
            else:
                st.session_state.selected_test = "Cochran's Q Test"
        else:
            st.session_state.selected_test = "Chi-Square Test"

    # Test seçimini session_state'e ekle
    st.session_state.test_name = st.session_state.selected_test
    st.write(f"The suitable hypothesis test for your data is **{st.session_state.test_name}**")

# Run Hypothesis Test butonu
    if 'test_name' not in st.session_state:
        st.error("Please run the analysis first to determine the appropriate hypothesis test.")
    else:
        # Test adı her seferinde seçilen teste eşitlenir
        test_name = st.session_state.test_name
        data = st.session_state.df.apply(pd.to_numeric, errors='coerce')
        data = data.dropna() 
        
        if test_name == "Paired T-Test":
            stat, p_value = stats.ttest_rel(data.iloc[:, 0], data.iloc[:, 1])
        elif test_name == "Independent T-Test":
            stat, p_value = stats.ttest_ind(data.iloc[:, 0], data.iloc[:, 1])
        elif test_name == "One Sample T-Test":
            stat, p_value = stats.ttest_1samp(data.iloc[:, 0], 0)
        elif test_name == "Repeated Measures ANOVA":
            stat, p_value = stats.f_oneway(*[data.iloc[:, i] for i in range(data.shape[1])])
        elif test_name == "One-Way ANOVA":
            stat, p_value = stats.f_oneway(data.iloc[:, 0], data.iloc[:, 1])
        elif test_name == "Wilcoxon Signed Rank Test":
            stat, p_value = stats.wilcoxon(data.iloc[:, 0], data.iloc[:, 1])
        elif test_name == "Mann-Whitney U Test":
            stat, p_value = stats.mannwhitneyu(data.iloc[:, 0], data.iloc[:, 1])
        elif test_name == "Friedman Test":
            stat, p_value = stats.friedmanchisquare(*[data.iloc[:, i] for i in range(data.shape[1])])
        elif test_name == "Kruskal Wallis Test":
            stat, p_value = stats.kruskal(*[data.iloc[:, i] for i in range(data.shape[1])])
        elif test_name == "Chi-Square Test":
            contingency_table = pd.crosstab(data.iloc[:, 0], data.iloc[:, 1])
            stat, p_value, _, _ = chi2_contingency(contingency_table)
        elif test_name == "McNemar Test":
             contingency_table = pd.crosstab(data.iloc[:, 0], data.iloc[:, 1])
             result = mcnemar(contingency_table, exact=True)
             stat = result.statistic
             p_value = result.pvalue

        elif test_name == "Cochran's Q Test":
            contingency_table = pd.crosstab(data.iloc[:, 0], data.iloc[:, 1])
            stat, p_value = friedmanchisquare(*contingency_table.T.values)
        
        # Test sonuçlarının yazdırılması
        st.write(f"**Test Statistic: **{stat:.4f}")
        st.write(f"**P-Value: **{p_value:.4f}")

        if p_value < 0.05:
            st.success("The null hypothesis is rejected. Significant difference exists.")
        else:
            st.warning("Failed to reject the null hypothesis. No significant difference.")
