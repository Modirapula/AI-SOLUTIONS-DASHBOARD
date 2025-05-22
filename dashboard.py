import pandas as pd
from flask import Flask, jsonify, request
import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
import threading
import os
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Load and preprocess data (shared between Flask and Streamlit)
DATASET_PATH = "AI_Solutions_Web_Server_Logs.csv"

def load_data():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file {DATASET_PATH} not found.")
    df = pd.read_csv(DATASET_PATH)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Profit"] = df["Revenue"] - df["Transaction_Amount"] if "Revenue" in df.columns and "Transaction_Amount" in df.columns else 0
    df["Date"] = df["Timestamp"].dt.date
    if "Job_Title" in df.columns:
        df["Job_Title"] = df["Job_Title"].fillna("Unknown")
    return df.dropna(subset=["Date"])

# Load data globally
try:
    df = load_data()
except FileNotFoundError as e:
    print(e)
    df = pd.DataFrame()  # Empty DataFrame to prevent crashes

# Flask API Helper function to apply filters
def filter_dataset(dataframe, params):
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        return dataframe
    if 'start_date' in params and params['start_date']:
        dataframe = dataframe[pd.to_datetime(dataframe['Timestamp']) >= pd.to_datetime(params['start_date'])]
    if 'end_date' in params and params['end_date']:
        dataframe = dataframe[pd.to_datetime(dataframe['Timestamp']) <= pd.to_datetime(params['end_date'])]
    if 'country' in params and params['country']:
        dataframe = dataframe[dataframe['Country'] == params['country']]
    if 'region' in params and params['region']:
        dataframe = dataframe[dataframe['Region'] == params['region']]
    if 'city' in params and params['city']:
        dataframe = dataframe[dataframe['City'] == params['city']]
    if 'activity' in params and params['activity']:
        dataframe = dataframe[dataframe['Request_Category'] == params['activity']]
    if 'job_title' in params and params['job_title'] and 'Job_Title' in dataframe.columns:
        dataframe = dataframe[dataframe['Job_Title'] == params['job_title']]
    if 'sales_rep' in params and params['sales_rep']:
        dataframe = dataframe[dataframe['Sales_Rep'] == params['sales_rep']]
    return dataframe

# Flask API Endpoints
@app.route('/api/total_entries', methods=['GET'])
def total_entries():
    params = request.args.to_dict()
    filtered_df = filter_dataset(df.copy(), params)
    return jsonify({"total_entries": len(filtered_df)})

@app.route('/api/unique_visitors', methods=['GET'])
def unique_visitors():
    params = request.args.to_dict()
    filtered_df = filter_dataset(df.copy(), params)
    visitors = filtered_df['IP_Address'].nunique() if 'IP_Address' in filtered_df.columns else 0
    return jsonify({"unique_visitors": visitors})

@app.route('/api/trends', methods=['GET'])
def trends():
    params = request.args.to_dict()
    filtered_df = filter_dataset(df.copy(), params)
    filtered_df['Date'] = filtered_df['Timestamp'].dt.date
    trends_df = filtered_df.groupby('Date').size().reset_index(name='Interactions')
    return jsonify(trends_df.to_dict(orient='records'))

@app.route('/api/service_requests', methods=['GET'])
def service_requests():
    params = request.args.to_dict()
    filtered_df = filter_dataset(df.copy(), params)
    sr_df = filtered_df['Service_Type'].value_counts().reset_index() if 'Service_Type' in filtered_df.columns else pd.DataFrame()
    sr_df.columns = ['Service_Type', 'Count']
    return jsonify(sr_df.to_dict(orient='records'))

@app.route('/api/sales_metrics', methods=['GET'])
def sales_metrics():
    params = request.args.to_dict()
    filtered_df = filter_dataset(df.copy(), params)
    if filtered_df.empty:
        return jsonify({"message": "No sales metrics available for the selected filters", "filters": params, "data_count": 0})
    total_sales = filtered_df[filtered_df['Transaction_Amount'] > 0]['Transaction_Amount'].sum() if 'Transaction_Amount' in filtered_df.columns else 0
    total_profit = filtered_df[filtered_df['Profit'] > 0]['Profit'].sum() if 'Profit' in filtered_df.columns else 0
    total_loss = filtered_df[filtered_df['Profit'] < 0]['Profit'].sum() if 'Profit' in filtered_df.columns else 0
    return jsonify({"total_sales": total_sales, "total_profit": total_profit, "total_loss": total_loss})

@app.route('/api/top_products', methods=['GET'])
def top_products():
    params = request.args.to_dict()
    filtered_df = filter_dataset(df.copy(), params)
    if filtered_df.empty:
        return jsonify({"message": "No product data available for the selected filters", "filters": params, "data_count": 0})
    top_sold = filtered_df['Product_Name'].value_counts().reset_index() if 'Product_Name' in filtered_df.columns else pd.DataFrame()
    top_sold.columns = ['Product_Name', 'Sales_Count']
    top_liked = filtered_df.groupby('Product_Name')['Likes'].sum().reset_index() if 'Product_Name' in filtered_df.columns and 'Likes' in filtered_df.columns else pd.DataFrame()
    top_liked = top_liked.sort_values(by='Likes', ascending=False)
    return jsonify({"top_sold_products": top_sold.to_dict(orient='records'), "top_liked_products": top_liked.to_dict(orient='records')})

@app.route('/api/customer_locations', methods=['GET'])
def customer_locations():
    params = request.args.to_dict()
    filtered_df = filter_dataset(df.copy(), params)
    if filtered_df.empty:
        return jsonify({"message": "No customer location data available for the selected filters", "filters": params, "data_count": 0})
    location_df = filtered_df.groupby(['Country', 'Region', 'City', 'Product_Name']).size().reset_index(name='Customer_Count') if {'Country', 'Region', 'City', 'Product_Name'}.issubset(filtered_df.columns) else pd.DataFrame()
    return jsonify(location_df.to_dict(orient='records'))

@app.route('/api/page_access', methods=['GET'])
def page_access():
    params = request.args.to_dict()
    filtered_df = filter_dataset(df.copy(), params)
    if filtered_df.empty:
        return jsonify({"message": "No page access data available for the selected filters", "filters": params, "data_count": 0})
    page_df = filtered_df['Page_Accessed'].value_counts().reset_index() if 'Page_Accessed' in filtered_df.columns else pd.DataFrame()
    page_df.columns = ['Page_Accessed', 'Access_Count']
    return jsonify(page_df.to_dict(orient='records'))

@app.route('/api/job_title_counts', methods=['GET'])
def job_title_counts():
    params = request.args.to_dict()
    filtered_df = filter_dataset(df.copy(), params)
    if 'Job_Title' not in filtered_df.columns or filtered_df.empty:
        return jsonify({"message": "'Job_Title' column not found or no data after filtering", "filters": params})
    job_counts = filtered_df['Job_Title'].value_counts().reset_index()
    job_counts.columns = ['Job_Title', 'Count']
    return jsonify(job_counts.to_dict(orient='records'))

@app.route('/api/generate_insights', methods=['GET'])
def generate_insights():
    params = request.args.to_dict()
    filtered_df = filter_dataset(df.copy(), params)
    insights = {}
    if filtered_df.empty:
        return jsonify({"message": "No data available for the selected filters", "filters": params, "data_count": 0})
    filtered_df['Date'] = filtered_df['Timestamp'].dt.date
    daily = filtered_df.groupby('Date').size().reset_index(name='Interactions')
    insights['avg_daily_interactions'] = daily['Interactions'].mean() if not daily.empty else 0
    insights['top_service_request'] = filtered_df['Service_Type'].value_counts().idxmax() if 'Service_Type' in filtered_df.columns and not filtered_df.empty else 'N/A'
    total_sales = filtered_df[filtered_df['Transaction_Amount'] > 0]['Transaction_Amount'].sum() if 'Transaction_Amount' in filtered_df.columns else 0
    conversion_rate = (filtered_df[filtered_df['Conversion_Status'] == 'Converted'].shape[0] / len(filtered_df)) * 100 if 'Conversion_Status' in filtered_df.columns and len(filtered_df) > 0 else 0
    insights['total_sales'] = total_sales
    insights['conversion_rate'] = conversion_rate
    insights['message'] = "Sales performance is strong" if conversion_rate > 50 else "Sales performance needs improvement"
    return jsonify(insights)

@app.route('/api/sales_by_rep', methods=['GET'])
def sales_by_rep():
    params = request.args.to_dict()
    filtered_df = filter_dataset(df.copy(), params)
    if filtered_df.empty:
        return jsonify({"message": "No data for selected filters", "filters": params, "data_count": 0})
    sales_data = filtered_df.groupby('Sales_Rep')['Transaction_Amount'].sum().reset_index() if 'Sales_Rep' in filtered_df.columns and 'Transaction_Amount' in filtered_df.columns else pd.DataFrame()
    sales_data.columns = ['Sales_Rep', 'Total_Sales']
    return jsonify(sales_data.to_dict(orient='records'))

@app.route('/api/team_members', methods=['GET'])
def team_members():
    return jsonify({"team_members": ["Leatile Modirapula", "Treach Mpedi", "Lesly Mochabeng", 
                                    "Enough Bogadi", "Amber Lame", "Tumelo Sadie"]})

# Streamlit Dashboard
def run_streamlit():
    # Streamlit cache for data loading
    @st.cache_data
    def load_streamlit_data():
        return load_data()

    df_streamlit = load_streamlit_data()

    # Fetch team members from API
    try:
        response = requests.get('http://127.0.0.1:5000/api/team_members')
        team_members_list = response.json().get("team_members", []) if response.status_code == 200 else []
    except requests.exceptions.RequestException:
        team_members_list = []

    # Sidebar: Compact view selection and filters
    st.sidebar.header("Dashboard Controls")
    view_option = st.sidebar.selectbox("Select View", ["SALES TEAM", "Team Member"] + team_members_list, key="view_select")
    st.sidebar.subheader("Filters")
    start_date = st.sidebar.date_input("Start Date", df_streamlit["Date"].min() if not df_streamlit.empty else datetime.now().date(), key="start_date")
    end_date = st.sidebar.date_input("End Date", df_streamlit["Date"].max() if not df_streamlit.empty else datetime.now().date(), key="end_date")
    selected_country = st.sidebar.multiselect("Select Country", df_streamlit["Country"].dropna().unique() if not df_streamlit.empty else [], key="country_select")

    # Apply filters
    filtered_df = df_streamlit[(df_streamlit["Date"] >= start_date) & (df_streamlit["Date"] <= end_date)]
    if selected_country:
        filtered_df = filtered_df[filtered_df["Country"].isin(selected_country)]

    # Custom CSS for compact layout
    st.markdown("""
        <style>
        .metric-container {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            margin-bottom: 10px;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            width: 33%;
            text-align: center;
            font-family: 'Arial', sans-serif;
        }
        .metric-title {
            font-size: 14px;
            color: black;
            margin-bottom: 4px;
        }
        .metric-value {
            font-size: 18px;
            font-weight: bold;
            color: blue;
        }
        .metric-delta {
            font-size: 12px;
            color: black;
        }
        .card {
            background-color: #f5f5f5;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .card h3 {
            font-size: 16px;
            color: #333;
            margin-bottom: 4px;
        }
        .card p {
            font-size: 18px;
            font-weight: bold;
            color: #0073e6;
            margin: 0;
        }
        .stRadio > div {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        .stRadio label {
            background-color: #f0f0f5;
            border: 1px solid #ccc;
            border-radius: 6px;
            padding: 8px;
            font-size: 12px;
            font-weight: 500;
            color: #333;
            cursor: pointer;
            text-align: center;
        }
        .stRadio label:hover {
            background-color: #e6f2ff;
            border-color: #0073e6;
        }
        .stRadio input:checked + div {
            background-color: #0073e6 !important;
            color: white !important;
            border-color: #005bb5 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Team Member View
    if view_option == "Team Member" or view_option in team_members_list:
        if view_option in team_members_list:
            member_data = filtered_df[filtered_df["Sales_Rep"] == view_option]
        else:
            selected_member = st.sidebar.selectbox("Select Sales Rep", filtered_df["Sales_Rep"].unique() if not filtered_df.empty else [], key="member_select")
            member_data = filtered_df[filtered_df["Sales_Rep"] == selected_member]

        if not member_data.empty:
            st.markdown(f"<h2 style='text-align: center; font-size: 20px;'>Individual Performance Analysis for: {selected_member if view_option == 'Team Member' else view_option}</h2>", unsafe_allow_html=True)

            # Metrics
            total_sales = member_data["Transaction_Amount"].sum() if 'Transaction_Amount' in member_data.columns else 0
            total_revenue = member_data["Revenue"].sum() if 'Revenue' in member_data.columns else 0
            sales_target = member_data["Sales_Target"].iloc[0] if "Sales_Target" in member_data.columns and not member_data["Sales_Target"].isnull().all() else 0
            achievement_pct = (total_revenue / sales_target * 100) if sales_target else 0
            delta_value = total_revenue - sales_target

            st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-card">
                        <div class="metric-title">Total Sales</div>
                        <div class="metric-value">${total_sales:,.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Total Revenue</div>
                        <div class="metric-value">${total_revenue:,.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Sales Target</div>
                        <div class="metric-value">${sales_target:,.0f}</div>
                        <div class="metric-delta">Target ${delta_value:,.0f}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<br><br>", unsafe_allow_html=True)

            # Charts
            col1, col2 = st.columns(2)
            with col1:
                if "Date" in member_data.columns:
                    member_data["Date"] = pd.to_datetime(member_data["Date"], errors="coerce")
                    rev_time = member_data.groupby("Date")["Revenue"].sum().reset_index()
                    rev_time = rev_time.sort_values("Date")
                    fig1 = px.line(rev_time, x="Date", y="Revenue", title="Revenue Over Time", height=400)
                    st.plotly_chart(fig1, use_container_width=True)

            with col2:
                if {"Timestamp", "Session_Duration"}.issubset(member_data.columns):
                    member_data["Timestamp"] = pd.to_datetime(member_data["Timestamp"], errors="coerce")
                    member_data["Session_Duration"] = pd.to_datetime(member_data["Session_Duration"], errors="coerce")
                    member_data = member_data.dropna(subset=["Timestamp", "Session_Duration"])
                    member_data["Response_Time_Days"] = (member_data["Session_Duration"] - member_data["Timestamp"]).dt.days
                    fig3 = px.histogram(member_data, x="Response_Time_Days", nbins=10, title="Response Time", height=400)
                    st.plotly_chart(fig3, use_container_width=True)

    # SALES TEAM View
    else:
        section = st.sidebar.radio("Select View", ["Overview", "Key Statistics", "Jobs Analysis", "Event Analytics", "Marketing Effectiveness"], key="section_select")

        if section == "Overview":
            st.markdown("<h2 style='text-align: center; font-size: 20px;'>AI Solutions - KPI Overview</h2>", unsafe_allow_html=True)

            # Compute KPIs
            total_revenue = round(filtered_df["Revenue"].sum(), 2) if 'Revenue' in filtered_df.columns else 0
            total_sales = round(filtered_df["Transaction_Amount"].max(), 0) if 'Transaction_Amount' in filtered_df.columns else 0
            conversion_count = filtered_df[filtered_df["Conversion_Status"] == "Converted"].shape[0] if "Conversion_Status" in filtered_df.columns else 0
            total_entries = filtered_df.shape[0]
            conversion_rate = (conversion_count / total_entries * 100) if total_entries > 0 else 0
            top_product = filtered_df.groupby("Product_Name")["Revenue"].sum().idxmax() if 'Product_Name' in filtered_df.columns and not filtered_df.empty else "N/A"
            top_country = filtered_df.groupby("Country")["Revenue"].sum().idxmax() if 'Country' in filtered_df.columns and not filtered_df.empty else "N/A"
            sales_target = filtered_df["Sales_Target"].mean() if "Sales_Target" in filtered_df.columns else 0
            achievement_pct = (total_revenue / sales_target * 100) if sales_target else 0
            top_rep = filtered_df.groupby("Sales_Rep")["Transaction_Amount"].sum().idxmax() if 'Sales_Rep' in filtered_df.columns and 'Transaction_Amount' in filtered_df.columns and not filtered_df.empty else "N/A"

            # KPI Cards
            st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-card">
                        <div class="metric-title">Total Sales</div>
                        <div class="metric-value">{total_sales}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Total Revenue</div>
                        <div class="metric-value">${total_revenue}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Sales Target (Avg)</div>
                        <div class="metric-value">${sales_target:,.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Top Sales Rep</div>
                        <div class="metric-value">{top_rep}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Top Product/Service</div>
                        <div class="metric-value">{top_product}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Gauge Chart for Sales Target Achievement
            sales_target = 4000000
            actual_revenue = filtered_df["Revenue"].sum() if 'Revenue' in filtered_df.columns else 0
            achievement_pct = (actual_revenue / sales_target * 100) if sales_target > 0 else 0
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=achievement_pct,
                delta={'reference': 90, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                gauge={
                    'axis': {'range': [None, 150]},
                    'bar': {'color': "black"},
                    'steps': [
                        {'range': [0, 60], 'color': "red"},
                        {'range': [60, 90], 'color': "orange"},
                        {'range': [90, 150], 'color': "green"},
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                },
                title={'text': "Sales Target Achievement (%)"}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

        elif section == "Key Statistics":
            st.markdown("<h2 style='text-align: center; font-size: 20px;'>Key Statistics</h2>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            mean_price = filtered_df["Transaction_Amount"].mean() if 'Transaction_Amount' in filtered_df.columns else 0
            std_price = filtered_df["Transaction_Amount"].std() if 'Transaction_Amount' in filtered_df.columns else 0
            mean_revenue = filtered_df["Revenue"].mean() if 'Revenue' in filtered_df.columns else 0
            std_revenue = filtered_df["Revenue"].std() if 'Revenue' in filtered_df.columns else 0
            with col1:
                st.markdown(f"""
                    <div class="card"><h3>Mean Price</h3><p>${mean_price:,.2f}</p></div>
                    <div class="card"><h3>Mean Revenue</h3><p>${mean_revenue:,.2f}</p></div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div class="card"><h3>Std. Price</h3><p>${std_price:,.2f}</p></div>
                    <div class="card"><h3>Std. Revenue</h3><p>${std_revenue:,.2f}</p></div>
                """, unsafe_allow_html=True)

        elif section == "Jobs Analysis":
            st.markdown("<h2 style='text-align: center; font-size: 20px;'>Jobs Analysis</h2>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                if "Job_Title" in filtered_df.columns:
                    job_counts = filtered_df["Job_Title"].value_counts().nlargest(5).reset_index()
                    job_counts.columns = ["Job_Title", "Count"]
                    fig1 = px.histogram(job_counts, x="Job_Title", y="Count", title="Top 5 Job Titles", height=250)
                    fig1.update_layout(showlegend=False)
                    st.plotly_chart(fig1, use_container_width=True)
            with col2:
                if "Timestamp" in filtered_df.columns and "Job_Title" in filtered_df.columns:
                    job_timeline = filtered_df.groupby(filtered_df["Timestamp"].dt.date)["Job_Title"].count().reset_index()
                    job_timeline.columns = ["Date", "Job_Count"]
                    fig2 = px.line(job_timeline, x="Date", y="Job_Count", title="Job Requests Over Time", height=250)
                    st.plotly_chart(fig2, use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                if "Country" in filtered_df.columns and "Job_Title" in filtered_df.columns:
                    country_jobs = filtered_df.groupby("Country")["Job_Title"].count().nlargest(10).reset_index()
                    country_jobs.columns = ["Country", "Job_Count"]
                    fig3 = px.bar(country_jobs, x="Country", y="Job_Count", title="Job Requests by Country", height=300)
                    st.plotly_chart(fig3, use_container_width=True)
            with col4:
                if "Action_Type" in filtered_df.columns and "Job_Title" in filtered_df.columns:
                    action_job = filtered_df.groupby("Action_Type")["Job_Title"].count().reset_index()
                    action_job.columns = ["Action_Type", "Count"]
                    fig4 = px.pie(action_job, names="Action_Type", values="Count", title="Actions on Job Titles", height=350)
                    fig4.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig4, use_container_width=True)
                else:
                    st.info("No trends available for selected filters")

        elif section == "Event Analytics":
            st.markdown("<h2 style='text-align: center; font-size: 20px;'>Event Analytics</h2>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                if "Request_Category" in filtered_df.columns:
                    event_counts = filtered_df["Request_Category"].value_counts().nlargest(5).reset_index()
                    event_counts.columns = ["Event", "Count"]
                    fig5 = px.bar(event_counts, x="Event", y="Count", title="Top Events", height=250)
                    st.plotly_chart(fig5, use_container_width=True)
            with col2:
                if "Country" in filtered_df.columns and "Request_Category" in filtered_df.columns:
                    geo_data = filtered_df.groupby("Country")["Request_Category"].count().reset_index()
                    geo_data.columns = ["Country", "Event_Count"]
                    fig6 = px.choropleth(geo_data, locations="Country", locationmode="country names", color="Event_Count", title="Events by Country", height=250)
                    st.plotly_chart(fig6, use_container_width=True)

        elif section == "Marketing Effectiveness":
            st.markdown("<h2 style='text-align: center; font-size: 20px;'>Marketing Effectiveness</h2>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                if "Request_Category" in filtered_df.columns:
                    category_counts = filtered_df["Request_Category"].value_counts().reset_index()
                    category_counts.columns = ["Request_Category", "Count"]
                    fig8 = px.pie(category_counts, names="Request_Category", values="Count", title="Request Categories", width=400, height=400)
                    fig8.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig8, use_container_width=True)
            with col2:
                if "Date" in filtered_df.columns:
                    revenue_time = filtered_df.groupby("Date")["Revenue"].sum().reset_index()
                    fig9 = px.line(revenue_time, x="Date", y="Revenue", title="Revenue Over Time", markers=True, width=400, height=400)
                    st.plotly_chart(fig9, use_container_width=True)

    # Compact Sidebar Instructions
    with st.sidebar.expander("📘 Instructions", expanded=False):
        st.markdown("""
            - **Select Dashboard View** from the dropdown menu:
            - `SALES TEAM`: View performance metrics and analytics for the entire sales team.
            - `Team Member`: View metrics specific to an individual sales representative.
            - If `Team Member` is selected:
              - Choose a sales representative from the dropdown list to analyze their individual performance.
            - Metrics displayed include:
              - **Total Sales Made**
              - **Total Revenue**
              - **Sales Target Achievement**
            - Charts and breakdowns:
              - Revenue over time
              - Customer segment distribution
              - Response time analysis
            - Troubleshooting:
              - If data is missing, check filters
              - Error messages will appear if filters are invalid or API requests fail
        """, unsafe_allow_html=True)

# Run Flask in a separate thread
def run_flask():
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

if __name__ == '__main__':
    # Start Flask server in a background thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    # Run Streamlit
    run_streamlit()