# customer_segmentation_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

# ------------------------------
# Dark Mode Toggle
# ------------------------------
dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=False)
bg_color = "#121212" if dark_mode else "#f0f8ff"
text_color = "white" if dark_mode else "#333333"
plot_template = "plotly_dark" if dark_mode else "plotly_white"

st.markdown(f"""
    <style>
    body {{
        background-color: {bg_color};
        color: {text_color};
    }}
    .stButton>button, .stDownloadButton>button {{
        background-color: #6A5ACD;
        color: white;
        border-radius: 10px;
        height: 45px;
        width: 100%;
        font-size: 16px;
        font-weight: bold;
        transition: transform 0.2s;
    }}
    .stButton>button:hover, .stDownloadButton>button:hover {{
        transform: scale(1.05);
        background-color: #9370DB;
    }}
    .stMetric {{
        border-radius: 15px;
        padding: 10px;
        background: linear-gradient(to right, #8ec5fc, #e0c3fc);
        box-shadow: 3px 3px 10px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }}
    .stMetric:hover {{
        transform: scale(1.05);
    }}
    h1 {{
        text-align: center;
        font-family: 'Arial Black', sans-serif;
        color: #4B0082;
    }}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🛍️ Customer Segmentation Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# ------------------------------
# Sidebar Controls
# ------------------------------
st.sidebar.header("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload Dataset (xlsx/csv)", type=["xlsx", "csv"])
num_customers = st.sidebar.slider("Number of Customers for Random Dataset:", 50, 1000, 200)
num_clusters = st.sidebar.slider("Number of clusters in random dataset:", 2, 6, 3)
generate_random = st.sidebar.button("🎲 Generate Random Dataset")

# Load scaler and model
scaler = joblib.load("scaler_customer.pkl")
kmeans = joblib.load("kmeans_customer_model.pkl")

# ------------------------------
# Function to process and visualize dataset
# ------------------------------
def process_and_visualize(df, random_generated=False):
    df = df.dropna(subset=['CustomerID'])
    df = df[df['Quantity'] > 0]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    customer_df = df.groupby('CustomerID').agg({
        'InvoiceNo':'nunique',
        'Quantity':'sum',
        'TotalPrice':'sum'
    }).reset_index()
    customer_df.rename(columns={'InvoiceNo':'NumPurchases',
                                'Quantity':'TotalQuantity',
                                'TotalPrice':'TotalSpend'}, inplace=True)

    # Scale features and predict clusters
    X = customer_df[['NumPurchases','TotalQuantity','TotalSpend']]
    X_scaled = scaler.transform(X)
    customer_df['Cluster'] = kmeans.predict(X_scaled)

    # ------------------------------
    # Cluster Filter
    clusters = sorted(customer_df['Cluster'].unique())
    selected_clusters = st.sidebar.multiselect("Filter Clusters to Display", options=clusters, default=clusters)
    filtered_df = customer_df[customer_df['Cluster'].isin(selected_clusters)]

    # ------------------------------
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", filtered_df.shape[0])
    col2.metric("Total Spend", f"${filtered_df['TotalSpend'].sum():,.2f}")
    col3.metric("Avg Purchases/Customer", f"{filtered_df['NumPurchases'].mean():.1f}")

    # ------------------------------
    # Tabs for visualization
    tabs = st.tabs(["📄 Raw Data", "📉 2D Visualization", "🌀 3D Cluster", "🗂 Cluster Summary", "⚖️ Compare Clusters"])

    # Raw Data
    with tabs[0]:
        st.subheader("Customer-Level Data")
        st.dataframe(filtered_df.head())
        csv_customer = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Download Clustered Data", data=csv_customer, file_name='clustered_customer_data.csv', mime='text/csv')
        if random_generated:
            csv_random = df.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Download Random Dataset", data=csv_random, file_name='random_customer_dataset.csv', mime='text/csv')

    # 2D Scatter
    with tabs[1]:
        st.subheader("2D Cluster Visualization")
        fig, ax = plt.subplots(figsize=(8,5))
        sns.scatterplot(data=filtered_df, x='TotalSpend', y='TotalQuantity', 
                        hue='Cluster', palette='Set2', s=100, ax=ax)
        ax.set_title("Customer Segments (TotalSpend vs TotalQuantity)")
        st.pyplot(fig)

    # 3D Scatter
    with tabs[2]:
        st.subheader("3D Cluster Visualization")
        fig = px.scatter_3d(
            filtered_df,
            x='NumPurchases',
            y='TotalQuantity',
            z='TotalSpend',
            color='Cluster',
            size='TotalSpend',
            hover_data={'CustomerID': True, 'NumPurchases': True, 'TotalQuantity': True, 'TotalSpend': True, 'Cluster': True},
            color_continuous_scale=px.colors.qualitative.Set2
        )
        fig.update_layout(width=900, height=700, template=plot_template)
        st.plotly_chart(fig)

    # Cluster Summary
    with tabs[3]:
        st.subheader("Cluster Summary")
        cluster_summary = filtered_df.groupby('Cluster')[['NumPurchases','TotalQuantity','TotalSpend']].mean().round(2)
        cluster_summary['CustomerCount'] = filtered_df['Cluster'].value_counts().sort_index().values
        st.dataframe(cluster_summary)

    # Compare Two Clusters
    with tabs[4]:
        st.subheader("Compare Two Clusters Side by Side")
        if len(clusters) >= 2:
            cluster1, cluster2 = st.columns(2)
            with cluster1:
                selected_cluster_1 = st.selectbox("Select Cluster 1", clusters, index=0)
            with cluster2:
                selected_cluster_2 = st.selectbox("Select Cluster 2", clusters, index=1)

            compare_df1 = filtered_df[filtered_df['Cluster'] == selected_cluster_1]
            compare_df2 = filtered_df[filtered_df['Cluster'] == selected_cluster_2]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"### Cluster {selected_cluster_1}")
                st.metric("Customers", compare_df1.shape[0])
                st.metric("Total Spend", f"${compare_df1['TotalSpend'].sum():,.2f}")
                st.metric("Avg Purchases/Customer", f"{compare_df1['NumPurchases'].mean():.1f}")
                fig1 = px.scatter_3d(compare_df1, x='NumPurchases', y='TotalQuantity', z='TotalSpend', 
                                     color='Cluster', size='TotalSpend', color_continuous_scale=px.colors.qualitative.Set2)
                fig1.update_layout(width=450, height=450, template=plot_template)
                st.plotly_chart(fig1)

            with col2:
                st.markdown(f"### Cluster {selected_cluster_2}")
                st.metric("Customers", compare_df2.shape[0])
                st.metric("Total Spend", f"${compare_df2['TotalSpend'].sum():,.2f}")
                st.metric("Avg Purchases/Customer", f"{compare_df2['NumPurchases'].mean():.1f}")
                fig2 = px.scatter_3d(compare_df2, x='NumPurchases', y='TotalQuantity', z='TotalSpend', 
                                     color='Cluster', size='TotalSpend', color_continuous_scale=px.colors.qualitative.Set2)
                fig2.update_layout(width=450, height=450, template=plot_template)
                st.plotly_chart(fig2)
        else:
            st.info("At least 2 clusters needed for comparison.")

    st.success("✅ Clustering Complete!")

# ------------------------------
# Dataset selection logic
# ------------------------------
if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    process_and_visualize(df)

elif generate_random:
    st.info(f"Generating random dataset with {num_clusters} clusters...")
    np.random.seed()
    cluster_size = num_customers // num_clusters
    data = []

    for i in range(num_clusters):
        low = 10 + i*50
        high = 50 + i*150
        invoice = np.random.randint(low, low + 10, size=cluster_size)
        quantity = np.random.randint(low, high, size=cluster_size)
        price = np.random.uniform(low*2, high*7, size=cluster_size)
        cluster_data = np.column_stack((invoice, quantity, price))
        data.append(cluster_data)

    data = np.vstack(data)

    # Handle remaining customers
    if data.shape[0] < num_customers:
        extra = num_customers - data.shape[0]
        invoice = np.random.randint(1, 50, size=extra)
        quantity = np.random.randint(1, 100, size=extra)
        price = np.random.uniform(10, 700, size=extra)
        extra_data = np.column_stack((invoice, quantity, price))
        data = np.vstack((data, extra_data))

    df = pd.DataFrame(data, columns=['InvoiceNo','Quantity','UnitPrice'])
    df['CustomerID'] = np.arange(1, num_customers + 1)

    process_and_visualize(df, random_generated=True)

else:
    st.warning("Please upload a dataset or click 'Generate Random Dataset' in the sidebar to proceed.")


