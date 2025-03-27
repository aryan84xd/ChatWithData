import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

class DataVisualizer:
    @staticmethod
    def visualize_data(df):
        # Smart sampling for large datasets
        if len(df) > 10000:
            sample_df = df.sample(n=10000, random_state=42)
            st.warning(f"Large dataset: Showing 10,000 random samples from {len(df)} rows")
        else:
            sample_df = df.copy()

        numerical_cols = sample_df.select_dtypes(include=['number']).columns
        categorical_cols = sample_df.select_dtypes(include=['object', 'category']).columns
        
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔢 Numerical", "🔤 Categorical"])
        
        with tab1:
            DataVisualizer._show_overview(sample_df, numerical_cols, categorical_cols)
            
        with tab2:
            DataVisualizer._show_numerical(sample_df, numerical_cols)
            
        with tab3:
            DataVisualizer._show_categorical(sample_df, categorical_cols)

    @staticmethod
    def _show_overview(df, numerical_cols, categorical_cols):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Rows", len(df))
            st.metric("Numerical Columns", len(numerical_cols))
            
        with col2:
            st.metric("Total Columns", len(df.columns))
            st.metric("Categorical Columns", len(categorical_cols))
        
        if len(numerical_cols) > 0:
            st.subheader("Numerical Summary")
            st.dataframe(df[numerical_cols].describe(), use_container_width=True)

    @staticmethod
    def _show_numerical(df, numerical_cols):
        if not numerical_cols.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                num_col = st.selectbox("Select numerical column:", numerical_cols)
                fig = px.histogram(df, x=num_col, nbins=30, 
                                  title=f'Distribution of {num_col}')
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                if len(numerical_cols) > 1:
                    num_col2 = st.selectbox("Select second column:", 
                                           [c for c in numerical_cols if c != num_col])
                    fig = px.scatter(df, x=num_col, y=num_col2,
                                    title=f'{num_col} vs {num_col2}')
                    st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _show_categorical(df, categorical_cols):
        if not categorical_cols.empty:
            cat_col = st.selectbox("Select categorical column:", categorical_cols)
            top_n = st.slider("Number of categories to show", 3, 20, 10)
            
            value_counts = df[cat_col].value_counts().nlargest(top_n)
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(value_counts, 
                            title=f'Top {top_n} {cat_col} Values',
                            labels={'index': cat_col, 'value': 'Count'})
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                if top_n <= 15:
                    fig = px.pie(value_counts, 
                               names=value_counts.index, 
                               values=value_counts.values,
                               title=f'Distribution of {cat_col}')
                    st.plotly_chart(fig, use_container_width=True)