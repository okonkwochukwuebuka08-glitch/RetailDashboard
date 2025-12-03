# RetailDashboard.py
# Retail Store Management – Sales & Customer Insights Dashboard
# BUIS 305 / INSS 405 Group Project
#
# Requirements covered:
# - OOP class SalesAnalyzer (cleaning, summarizing, visualizing)
# - Streamlit interactive dashboard (file uploader, filters, charts)
# - Uses Pandas, Matplotlib, Seaborn

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Make plots look a bit nicer
sns.set(style="whitegrid")

class SalesAnalyzer:
    """
    OOP class for cleaning, summarizing, and visualizing
    Online Retail II data.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def clean_data(self) -> pd.DataFrame:
        """
        Basic cleaning:
        - Remove duplicate rows
        - Drop rows with missing CustomerID
        - Convert InvoiceDate to datetime
        - Remove negative or zero quantities
        - Remove negative or zero prices
        - Create Total column = Quantity * UnitPrice
        """
        # Remove duplicates
        self.df.drop_duplicates(inplace=True)

        # Drop rows with no customer ID (cannot analyze customer behaviour)
        if "CustomerID" in self.df.columns:
            self.df = self.df.dropna(subset=["CustomerID"])

        # Convert InvoiceDate to datetime
        if "InvoiceDate" in self.df.columns:
            self.df["InvoiceDate"] = pd.to_datetime(self.df["InvoiceDate"], errors="coerce")
            self.df = self.df.dropna(subset=["InvoiceDate"])

        # Fix dtypes
        if "Quantity" in self.df.columns:
            self.df["Quantity"] = pd.to_numeric(self.df["Quantity"], errors="coerce")
        if "UnitPrice" in self.df.columns:
            self.df["UnitPrice"] = pd.to_numeric(self.df["UnitPrice"], errors="coerce")

        # Remove invalid quantities/prices (returns or data errors)
        if "Quantity" in self.df.columns:
            self.df = self.df[self.df["Quantity"] > 0]
        if "UnitPrice" in self.df.columns:
            self.df = self.df[self.df["UnitPrice"] > 0]

        # Create Total sales column
        if {"Quantity", "UnitPrice"}.issubset(self.df.columns):
            self.df["Total"] = self.df["Quantity"] * self.df["UnitPrice"]

        # Standardize CustomerID as int (if exists)
        if "CustomerID" in self.df.columns:
            self.df["CustomerID"] = self.df["CustomerID"].astype(str)

        return self.df

    def filter_data(self, start_date=None, end_date=None, countries=None) -> pd.DataFrame:
        """
        Apply date range and country filters.
        """
        data = self.df.copy()

        if start_date is not None:
            data = data[data["InvoiceDate"] >= start_date]
        if end_date is not None:
            data = data[data["InvoiceDate"] <= end_date]

        if countries:
            data = data[data["Country"].isin(countries)]

        return data

    def compute_new_vs_returning(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Classify each transaction as 'New' or 'Returning' based on
        whether the invoice date is the customer's first purchase date.
        """
        if not {"CustomerID", "InvoiceDate"}.issubset(data.columns):
            return pd.DataFrame()

        # First purchase date per customer
        first_purchase = (
            data.groupby("CustomerID")["InvoiceDate"]
            .min()
            .rename("FirstPurchaseDate")
        )
        data = data.merge(first_purchase, on="CustomerID", how="left")

        data["CustomerType"] = np.where(
            data["InvoiceDate"].dt.date == data["FirstPurchaseDate"].dt.date,
            "New",
            "Returning",
        )

        return data

    # ---------- Summary metrics ----------

    def get_summary_metrics(self, data: pd.DataFrame) -> dict:
        """
        Returns key summary metrics:
        - Total Revenue
        - Number of Invoices
        - Number of Unique Customers
        - Number of Countries
        """
        total_revenue = data["Total"].sum() if "Total" in data.columns else None
        num_invoices = data["InvoiceNo"].nunique() if "InvoiceNo" in data.columns else None
        num_customers = data["CustomerID"].nunique() if "CustomerID" in data.columns else None
        num_countries = data["Country"].nunique() if "Country" in data.columns else None

        return {
            "Total Revenue": total_revenue,
            "Number of Invoices": num_invoices,
            "Unique Customers": num_customers,
            "Countries": num_countries,
        }

    # ---------- Aggregations for charts ----------

    def sales_by_product(self, data: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Aggregate total sales by product description.
        Returns top N products by revenue.
        """
        if not {"Description", "Total"}.issubset(data.columns):
            return pd.DataFrame()

        prod = (
            data.groupby("Description")["Total"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        return prod

    def sales_by_country(self, data: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Aggregate total sales by country.
        Returns top N countries by revenue.
        """
        if not {"Country", "Total"}.issubset(data.columns):
            return pd.DataFrame()

        country = (
            data.groupby("Country")["Total"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        return country

    def sales_over_time(self, data: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
        """
        Time series of total sales.
        freq='D' for daily, 'M' for monthly.
        """
        if not {"InvoiceDate", "Total"}.issubset(data.columns):
            return pd.DataFrame()

        temp = data.set_index("InvoiceDate").copy()
        sales_ts = temp["Total"].resample(freq).sum().reset_index()
        sales_ts.rename(columns={"Total": "TotalSales"}, inplace=True)
        return sales_ts

    def top_customers(self, data: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Top customers by total spending.
        """
        if not {"CustomerID", "Total"}.issubset(data.columns):
            return pd.DataFrame()

        cust = (
            data.groupby("CustomerID")["Total"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        return cust

    def new_vs_returning_distribution(self, data: pd.DataFrame) -> pd.Series:
        """
        Distribution of New vs Returning customers (by number of orders).
        """
        classified = self.compute_new_vs_returning(data)
        if "CustomerType" not in classified.columns:
            return pd.Series(dtype="float64")

        return classified["CustomerType"].value_counts()

    # ---------- Plotting helpers (Matplotlib + Seaborn) ----------

    def plot_bar(self, df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str):
        fig, ax = plt.subplots()
        sns.barplot(data=df, x=x_col, y=y_col, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig

    def plot_line(self, df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str):
        fig, ax = plt.subplots()
        sns.lineplot(data=df, x=x_col, y=y_col, marker="o", ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig

    def plot_pie(self, series: pd.Series, title: str):
        fig, ax = plt.subplots()
        ax.pie(series.values, labels=series.index, autopct="%1.1f%%", startangle=90)
        ax.set_title(title)
        ax.axis("equal")
        return fig


# ---------------------- STREAMLIT APP ---------------------- #

def main():
    st.title("Retail Store Management: Sales & Customer Insights Dashboard")
    st.write(
        """
        This dashboard analyzes sales performance and customer behaviour using the
        **Online Retail II** dataset. Upload the CSV file to get started.
        """
    )

    uploaded_file = st.file_uploader("Upload Online Retail II CSV file", type=["csv", "txt"])

    if uploaded_file is None:
        st.info("Please upload the Online Retail II dataset (CSV) to see the analysis.")
        return

    # Load data
    df = pd.read_csv(uploaded_file, encoding="latin1")

    analyzer = SalesAnalyzer(df)
    cleaned_df = analyzer.clean_data()

    # Sidebar filters
    st.sidebar.header("Filters")

    min_date = cleaned_df["InvoiceDate"].min().date()
    max_date = cleaned_df["InvoiceDate"].max().date()

    date_range = st.sidebar.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        # If user selects only one date
        start_date, end_date = min_date, date_range

    countries = sorted(cleaned_df["Country"].dropna().unique().tolist())
    selected_countries = st.sidebar.multiselect(
        "Select Countries (optional)", countries, default=countries
    )

    filtered_df = analyzer.filter_data(
        start_date=pd.to_datetime(start_date),
        end_date=pd.to_datetime(end_date),
        countries=selected_countries,
    )

    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        return

    # Summary metrics
    st.subheader("Summary Metrics")
    metrics = analyzer.get_summary_metrics(filtered_df)
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Revenue", f"${metrics['Total Revenue']:.2f}" if metrics["Total Revenue"] is not None else "N/A")
    col2.metric("Invoices", metrics["Number of Invoices"] if metrics["Number of Invoices"] is not None else "N/A")
    col3.metric("Unique Customers", metrics["Unique Customers"] if metrics["Unique Customers"] is not None else "N/A")
    col4.metric("Countries", metrics["Countries"] if metrics["Countries"] is not None else "N/A")

    # ---------- Charts ----------

    st.subheader("Sales by Product (Top 10)")
    prod_df = analyzer.sales_by_product(filtered_df, top_n=10)
    if not prod_df.empty:
        fig_prod = analyzer.plot_bar(
            prod_df,
            x_col="Description",
            y_col="Total",
            title="Top 10 Products by Revenue",
            xlabel="Product",
            ylabel="Total Revenue",
        )
        st.pyplot(fig_prod)
    else:
        st.info("Product data not available.")

    st.subheader("Sales by Country (Top 10)")
    country_df = analyzer.sales_by_country(filtered_df, top_n=10)
    if not country_df.empty:
        fig_country = analyzer.plot_bar(
            country_df,
            x_col="Country",
            y_col="Total",
            title="Top 10 Countries by Revenue",
            xlabel="Country",
            ylabel="Total Revenue",
        )
        st.pyplot(fig_country)
    else:
        st.info("Country data not available.")

    st.subheader("Sales Over Time")
    freq = st.radio("Time granularity", ["Monthly", "Daily"], horizontal=True)
    freq_code = "M" if freq == "Monthly" else "D"

    time_df = analyzer.sales_over_time(filtered_df, freq=freq_code)
    if not time_df.empty:
        # Rename time column for plotting
        time_col = "InvoiceDate"
        fig_time = analyzer.plot_line(
            time_df,
            x_col=time_col,
            y_col="TotalSales",
            title=f"Sales Over Time ({freq})",
            xlabel="Date",
            ylabel="Total Revenue",
        )
        st.pyplot(fig_time)
    else:
        st.info("Time series data not available.")

    st.subheader("Top Customers by Spending (Top 10)")
    cust_df = analyzer.top_customers(filtered_df, top_n=10)
    if not cust_df.empty:
        fig_cust = analyzer.plot_bar(
            cust_df,
            x_col="CustomerID",
            y_col="Total",
            title="Top 10 Customers by Total Spending",
            xlabel="Customer ID",
            ylabel="Total Revenue",
        )
        st.pyplot(fig_cust)
    else:
        st.info("Customer data not available.")

    st.subheader("New vs Returning Customers (Orders)")
    nv_series = analyzer.new_vs_returning_distribution(filtered_df)
    if not nv_series.empty:
        fig_nv = analyzer.plot_pie(nv_series, title="New vs Returning Customers (by Orders)")
        st.pyplot(fig_nv)
    else:
        st.info("Cannot compute new vs returning customers from this dataset.")

    st.markdown("---")
    st.caption("BUIS 305 / INSS 405 – Retail Store Management Dashboard (Online Retail II)")

if __name__ == "__main__":
    main()
