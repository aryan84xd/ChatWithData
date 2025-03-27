import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from database.db_handler import DatabaseHandler
from analysis.report_generator import ReportGenerator
from analysis.visualizer import DataVisualizer
from llm.llm_handler import LLMHandler

def main():
    st.set_page_config(page_title="📊 Data Analysis Chatbot", layout="wide")
    st.title("📊 CSV/Excel Analysis Chatbot")

    # Initialize database handler
    db_handler = DatabaseHandler()

    # File uploader
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    # Ensure table name is consistent
    table_name = "uploaded_data"

    if uploaded_file:
        # Read file based on format
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload a CSV or Excel file.")
                st.stop()

            # Check if file is empty
            if df.empty:
                st.warning("Uploaded file is empty. Please upload a valid dataset.")
                st.stop()

            # Save to database
            db_handler.save_to_db(df, table_name)
            st.success("✅ File uploaded and stored in temp database!")

            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📋 Dataset Report", "📊 Visualizations", "💬 AI Chat", "🛠 SQL Query"])

            with tab1:
                # Generate report
                report = ReportGenerator.generate_report(df)
                st.subheader("📋 Dataset Analysis Report")
                st.json(report)

            with tab2:
                # Improved data visualization
                st.subheader("📊 Data Distribution Analysis")
                DataVisualizer.visualize_data(df)

            with tab3:
                # LLM Chat
                st.subheader("💬 Chat with AI about the dataset")
                
                # Optional: Display example queries
                st.info("""
                💡 Example Queries:
                - "Show me the top 10 most expensive items"
                - "What are the average sales by category?"
                - "Find trends in customer purchases"
                """)
                
                # Generate report once to provide context
                try:
                    report = ReportGenerator.generate_report(df)
                except Exception as e:
                    st.error(f"Could not generate dataset report: {e}")
                    report = None
                
                user_query = st.text_input("Ask a question about the dataset")
                
                if user_query:
                    # Validate inputs before processing
                    if df is not None and db_handler is not None:
                        try:
                            LLMHandler.query_database_with_followup(
                                prompt=user_query, 
                                db_handler=db_handler, 
                                df=df,
                                report=report  # Optional but recommended
                            )
                        except Exception as e:
                            st.error(f"An error occurred while processing your query: {e}")
                    else:
                        st.warning("Please upload a dataset and connect to a database first.")

            with tab4:
                # SQL Query Execution
                st.subheader("🛠 SQL Query Execution")
                sql_query = st.text_area("Write an SQL Query to analyze dataset")
                if st.button("Run SQL Query"):
                    try:
                        df_result = db_handler.query_db(sql_query)
                        st.write("### SQL Query Result")
                        st.dataframe(df_result)
                    except Exception as e:
                        st.error(f"SQL Query Execution Error: {e}")

        except Exception as e:
            st.error(f"Error processing file: {e}")
        finally:
            # Close database connection
            db_handler.close_connection()

if __name__ == "__main__":
    main()