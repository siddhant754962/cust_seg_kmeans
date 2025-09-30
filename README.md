

# 🛍️ Customer Segmentation Dashboard

## Table of Contents

1. [Project Overview](#project-overview)
2. [Objective](#objective)
3. [Dataset](#dataset)
4. [Features](#features)
5. [Technologies Used](#technologies-used)
6. [Project Workflow](#project-workflow)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Screenshots](#screenshots)
10. [Future Enhancements](#future-enhancements)
11. [Author](#author)

---

## Project Overview

The **Customer Segmentation Dashboard** is an interactive web application built using **Streamlit**, designed to **cluster retail customers** based on their purchasing behavior. The app leverages **K-Means clustering** and **feature scaling** to segment customers into distinct groups. These segments can then be used to create targeted marketing strategies, loyalty programs, and improve overall business decision-making.

---

## Objective

* Segment retail customers based on their purchase history.
* Identify **VIP customers**, **regular buyers**, **budget shoppers**, and other meaningful groups.
* Provide **interactive visualizations** for analyzing clusters in 2D and 3D.
* Enable **business insights** through cluster summaries and comparisons.

---

## Dataset

The project uses the **Online Retail Dataset** from UCI Machine Learning Repository.

**Dataset Details:**

| Attribute   | Description                                |
| ----------- | ------------------------------------------ |
| InvoiceNo   | Unique invoice number for each transaction |
| StockCode   | Product/item code                          |
| Description | Product description                        |
| Quantity    | Quantity of each product purchased         |
| InvoiceDate | Invoice date and time                      |
| UnitPrice   | Price per unit of the product              |
| CustomerID  | Unique identifier for each customer        |
| Country     | Country of the customer                    |

**Preprocessing:**

* Removed rows with missing CustomerID
* Removed negative or zero quantities
* Added `TotalPrice = Quantity * UnitPrice`
* Aggregated at **customer level**:

  * `NumPurchases` → total invoices per customer
  * `TotalQuantity` → total quantity purchased
  * `TotalSpend` → total money spent

---

## Features

1. **Automatic Data Cleaning:** Handles missing values and negative quantities.
2. **Customer Aggregation:** Computes customer-level metrics.
3. **Saved Scaler & Model:** Uses **pre-trained StandardScaler** and **K-Means model** for fast predictions.
4. **2D & 3D Visualizations:**

   * 2D scatter: TotalSpend vs TotalQuantity
   * 3D scatter: NumPurchases, TotalQuantity, TotalSpend
5. **Cluster Summary:** Displays average purchases, total spending, and number of customers per cluster.
6. **Cluster Filtering:** View only selected clusters.
7. **Compare Clusters:** Compare two clusters side-by-side.
8. **Download Data:** Download clustered dataset as CSV.
9. **Dark/Light Mode:** Switch UI themes for better visualization.

---

## Technologies Used

| Technology           | Purpose                                              |
| -------------------- | ---------------------------------------------------- |
| Python               | Core programming language                            |
| Pandas               | Data manipulation and aggregation                    |
| NumPy                | Numerical computations                               |
| Scikit-learn         | Feature scaling (StandardScaler), K-Means clustering |
| Streamlit            | Web application framework                            |
| Matplotlib / Seaborn | 2D visualizations                                    |
| Plotly               | Interactive 3D visualizations                        |
| Joblib               | Saving/loading trained models and scalers            |

---

## Project Workflow

1. **Load Dataset:** Reads local Online Retail dataset (`.xlsx`).
2. **Data Cleaning:**

   * Drop missing CustomerID
   * Remove negative/zero quantities
   * Compute `TotalPrice`
3. **Aggregate Data:** Generate customer-level metrics (`NumPurchases`, `TotalQuantity`, `TotalSpend`).
4. **Feature Scaling:** Apply **StandardScaler** to normalize data.
5. **K-Means Clustering:** Assign cluster labels to each customer using pre-trained model.
6. **Visualization:**

   * 2D scatter plots
   * Interactive 3D scatter plots
   * Cluster summaries
7. **Metrics & Insights:** Shows total customers, average purchases per customer, total spending.
8. **Cluster Comparison:** Compare metrics of two clusters side-by-side.
9. **Download Option:** Export clustered data for further analysis.

---

## Installation

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd customer-segmentation-dashboard
```

2. **Create virtual environment**

```bash
python -m venv venv
```

3. **Activate virtual environment**

* Windows:

```bash
venv\Scripts\activate
```

* Mac/Linux:

```bash
source venv/bin/activate
```

4. **Install dependencies**

```bash
pip install -r requirements.txt
```

**Sample requirements.txt**

```
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
openpyxl
joblib
```

---

## Usage

1. Place your **Online Retail dataset** in the same folder as the app.
2. Run the app:

```bash
streamlit run customer_segmentation_app.py
```

3. Explore the **dashboard**:

   * Switch between dark/light mode
   * View raw customer data
   * Analyze clusters in 2D and 3D
   * Filter and compare clusters
   * Download clustered dataset

---

## Screenshots

> *(Add screenshots of your app here for better understanding)*

* **2D Scatter Plot**
* **3D Cluster Visualization**
* **Cluster Summary Table**
* **Dark/Light Mode Example**

---

## Future Enhancements

* Add **VIP/Regular/Budget labels** for clusters automatically.
* Use **Elbow method** or **Silhouette score** to dynamically choose optimal number of clusters.
* Add **interactive filters** for country, total spend range, or product category.
* Implement **real-time dashboard** for new customer data.
* Integrate **marketing recommendations** based on cluster insights.

---

## Author

**Sidhant Patel**

* GitHub: [github.com/sidhantpatel](https://github.com/sidhantpatel)
* LinkedIn: [linkedin.com/in/sidhantpatel](https://linkedin.com/in/sidhantpatel)



