import streamlit as st
import pandas as pd
import sqlite3
import os
import openpyxl
import numpy as np
from scipy.stats import skew, kurtosis
import plotly.express as px
import plotly.graph_objs as go
from markdown import markdown
from dotenv import load_dotenv
from google.generativeai import configure, GenerativeModel
import tabulate
import re
# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
print("Key:", gemini_api_key)

if not gemini_api_key:
    raise ValueError("⚠️ Gemini API Key is missing! Please add it to the .env file.")

# Configure Gemini API
configure(api_key=gemini_api_key)
model = GenerativeModel("gemini-1.5-flash")

# Database Handler
class DatabaseHandler:
    def __init__(self, db_name="temp_data.db"):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()

    @staticmethod
    def _format_column_name(column):
        """
        Formats column names to snake_case:
        - Converts to lowercase
        - Replaces spaces and special characters (except underscores) with underscores
        - Ensures only single underscores are used
        - Preserves existing underscores
        """
        # Remove special characters except underscores and spaces
        cleaned = re.sub(r'[^a-zA-Z0-9_\s]', '', column)
        
        # Replace spaces with underscores
        with_underscores = re.sub(r'\s+', '_', cleaned)

        # Ensure only single underscores
        snake_case = re.sub(r'_+', '_', with_underscores).strip('_').lower()
        
        return snake_case
            

    def save_to_db(self, df, table_name):
        """
        Save DataFrame to database with LLM-friendly column names
        """
        # Create a copy of the DataFrame to avoid modifying the original
        formatted_df = df.copy()
        
        # Rename columns to LLM-friendly format
        formatted_df.columns = [self._format_column_name(col) for col in formatted_df.columns]
        
        # Save to SQL with formatted column names
        formatted_df.to_sql(table_name, self.conn, if_exists='replace', index=False)
        
        # Commit the changes
        self.conn.commit()

    def query_db(self, query):
        """
        Execute a SQL query and return results
        """
        return pd.read_sql_query(query, self.conn)

    def close_connection(self):
        """
        Close the database connection
        """
        self.conn.close()

    def display_table_info(self, table_name):
        """
        Display information about the table columns
        """
        query = f"PRAGMA table_info({table_name})"
        column_info = pd.read_sql_query(query, self.conn)
        print(f"Columns in {table_name}:")
        for _, row in column_info.iterrows():
            print(f"- {row['name']}")

# Report Generator
class ReportGenerator:
    @staticmethod
    def generate_report(df, table_name="uploaded_data"):
        # Format column names to match database format
        formatted_df = df.copy()
        formatted_df.columns = [DatabaseHandler._format_column_name(col) for col in formatted_df.columns]
        df = formatted_df

        if formatted_df.empty:
            return {"Error": "Dataset is empty!"}


        report = {
            'Dataset Metadata': {
                'Table Name': table_name,
                'Number of Rows': df.shape[0],
                'Number of Columns': df.shape[1],
                'Total File Size (Estimated)': f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB"
            },
            'Column Information': {
                'Columns': {
                    col: {
                        'Data Type': str(df[col].dtype),
                        'Non-Null Count': df[col].count(),
                        'Null Percentage': round(df[col].isnull().mean() * 100, 2)
                    } for col in df.columns
                }
            },
            'Data Overview': {
                'Duplicate Entries': df.duplicated().sum(),
                'First 5 Rows': df.head(5).to_dict(orient='records'),
                'Last 5 Rows': df.tail(5).to_dict(orient='records')
            }
        }

        # Categorize and analyze columns
        numerical_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns

        # Numerical Column Analysis
        report['Numerical Columns Analysis'] = {}
        for col in numerical_cols:
            valid_values = df[col].dropna()
            if len(valid_values) > 0:
                try:
                    report['Numerical Columns Analysis'][col] = {
                        'Statistical Summary': {
                            'Mean': round(valid_values.mean(), 2),
                            'Median': round(valid_values.median(), 2),
                            'Mode': valid_values.mode().iloc[0] if not valid_values.mode().empty else None,
                            'Standard Deviation': round(valid_values.std(), 2),
                            'Variance': round(valid_values.var(), 2),
                            'Min': round(valid_values.min(), 2),
                            'Max': round(valid_values.max(), 2),
                            'Quartiles': {
                                '25%': round(valid_values.quantile(0.25), 2),
                                '50%': round(valid_values.quantile(0.50), 2),
                                '75%': round(valid_values.quantile(0.75), 2)
                            }
                        },
                        'Distribution Characteristics': {
                            'Skewness': round(skew(valid_values), 2),
                            'Kurtosis': round(kurtosis(valid_values), 2)
                        }
                    }
                except Exception as e:
                    report['Numerical Columns Analysis'][col] = f"Error computing statistics: {str(e)}"
            else:
                report['Numerical Columns Analysis'][col] = "No valid numerical data"

        # Categorical Column Analysis
        report['Categorical Columns Analysis'] = {
            col: {
                'Unique Values Count': df[col].nunique(),
                'Value Distribution': {
                    'Most Common': df[col].value_counts().head(3).to_dict(),
                    'Least Common': df[col].value_counts().tail(3).to_dict()
                }
            } for col in categorical_cols
        }

        # Datetime Column Analysis
        report['Datetime Columns Analysis'] = {
            col: {
                'Earliest Date': df[col].min(),
                'Latest Date': df[col].max(),
                'Date Range': str(df[col].max() - df[col].min())
            } for col in datetime_cols
        }

        # Correlation Matrix (if applicable)
        try:
            if len(numerical_cols) > 1:
                report['Correlation Matrix'] = df[numerical_cols].corr().round(2).to_dict()
            else:
                report['Correlation Matrix'] = "Insufficient numerical columns for correlation"
        except Exception as e:
            report['Correlation Matrix'] = f"Error computing correlation: {str(e)}"

        return report

# Visualization
class DataVisualizer:
    @staticmethod
    def visualize_data(df):
        # Numerical columns for visualization
        numerical_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(numerical_cols) == 0:
            st.warning("No numerical columns found for visualization.")
            return
        
        # Plotly color palette
        color_palette = px.colors.qualitative.Plotly

        # Boxplots for numerical columns
        boxplot_figs = []
        for col in numerical_cols:
            fig = px.box(df, y=col, title=f'Distribution of {col}')
            fig.update_layout(
                title_font_size=14,
                title_x=0.5,
                height=400,
                width=600
            )
            boxplot_figs.append(fig)
        
        # Display boxplots
        st.subheader("Boxplots of Numerical Columns")
        for fig in boxplot_figs:
            st.plotly_chart(fig)
        
        # Histogram or Kernel Density Plot
        if len(numerical_cols) > 0:
            # First numerical column
            main_col = numerical_cols[0]
            hist_fig = go.Figure()
            hist_fig.add_trace(go.Histogram(
                x=df[main_col], 
                name=main_col,
                marker_color=color_palette[0]
            ))
            hist_fig.update_layout(
                title=f'Histogram of {main_col}',
                xaxis_title=main_col,
                yaxis_title='Frequency',
                height=400,
                width=600
            )
            st.subheader("Histogram")
            st.plotly_chart(hist_fig)
        
        # Scatter plot for first two numerical columns if available
        if len(numerical_cols) > 1:
            scatter_fig = px.scatter(
                df, 
                x=numerical_cols[0], 
                y=numerical_cols[1],
                title=f'Scatter Plot: {numerical_cols[0]} vs {numerical_cols[1]}'
            )
            scatter_fig.update_layout(height=400, width=600)
            st.subheader("Scatter Plot")
            st.plotly_chart(scatter_fig)
        
        # If categorical columns exist, create a bar plot
        if len(categorical_cols) > 0:
            main_cat_col = categorical_cols[0]
            value_counts = df[main_cat_col].value_counts()
            bar_fig = px.bar(
                x=value_counts.index, 
                y=value_counts.values,
                title=f'Distribution of {main_cat_col}',
                labels={'x': main_cat_col, 'y': 'Count'}
            )
            bar_fig.update_layout(height=400, width=600)
            st.subheader("Categorical Column Distribution")
            st.plotly_chart(bar_fig)

# LLM Integration
class LLMHandler:
    @staticmethod
    def generate_contextual_query(prompt, df, report=None):
        """
        Generate a more intelligent SQL query with contextual understanding
        
        :param prompt: User's natural language query
        :param df: DataFrame containing the data
        :param report: Optional pre-generated report for additional context
        :return: Dictionary with query and full response
        """
        prompt_lower = prompt.lower()
        columns = df.columns
        column_types = df.dtypes
        
        # Prepare additional context from report if available
        report_context = ""
        if report:
            # Add relevant sections from the report
            if 'Dataset Metadata' in report:
                report_context += "\nDataset Metadata:\n"
                report_context += "\n".join([f"- {k}: {v}" for k, v in report['Dataset Metadata'].items()])
            
            if 'Numerical Columns Analysis' in report:
                report_context += "\n\nNumerical Columns Insights:\n"
                for col, analysis in report['Numerical Columns Analysis'].items():
                    if isinstance(analysis, dict):
                        stats = analysis.get('Statistical Summary', {})
                        report_context += f"- {col}: Mean={stats.get('Mean')}, Min={stats.get('Min')}, Max={stats.get('Max')}\n"
            
            if 'Categorical Columns Analysis' in report:
                report_context += "\nCategorical Columns Insights:\n"
                for col, analysis in report['Categorical Columns Analysis'].items():
                    report_context += f"- {col}: {analysis.get('Unique Values Count', 0)} unique values\n"
        
        # Advanced query context extraction
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        string_cols = df.select_dtypes(include=['object']).columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
        # Comprehensive query generation prompt for LLM
        context_prompt = f"""
You are an expert SQL query generator. Based on the following dataset information:

{report_context}

Columns Details:
{chr(10).join([f"- {col}: {str(column_types[col])} (Sample values: {', '.join(map(str, df[col].unique()[:3]))})" for col in columns])}

User Query: "{prompt}"

Your task:
1. Generate the most appropriate SQL query
2. Explain your reasoning
3. Suggest any additional queries that might provide more insights

Constraints:
- Use meaningful filters
- Limit to 20 rows
- Use aggregations if appropriate
- Consider the column types and potential relationships

Respond with:
```sql
-- Your SQL Query Here
```

Reasoning: [Your explanation]

Additional Insights Query: [Optional supplementary query]
"""
        
        try:
            # Use LLM to generate a more intelligent query
            response = model.generate_content(context_prompt)
            
            # Extract SQL query (with improved error handling)
            if not hasattr(response, 'text'):
                st.warning("LLM did not generate a valid response.")
                return None
            
            # Extract SQL query using regex
            import re
            sql_match = re.search(r'```sql\n(.*?)```', response.text, re.DOTALL)
            
            if sql_match:
                sql_query = sql_match.group(1).strip()
                return {
                    'query': sql_query,
                    'full_response': response.text
                }
            
            # Fallback to direct text parsing
            sql_match = re.search(r'SELECT.*', response.text, re.IGNORECASE)
            if sql_match:
                return {
                    'query': sql_match.group(0).strip(),
                    'full_response': response.text
                }
            
            st.warning("Could not extract a valid SQL query from the LLM response.")
            return None
        
        except Exception as e:
            st.error(f"Error generating contextual query: {e}")
            return None

    @staticmethod
    def query_database_with_followup(prompt, db_handler, df, report=None):
        """
        Advanced database querying with LLM-powered follow-up insights
        
        :param prompt: User's natural language query
        :param db_handler: Database handler object
        :param df: DataFrame containing the data
        :param report: Optional pre-generated report for additional context
        """
        try:
            # Generate contextual query
            query_result = LLMHandler.generate_contextual_query(prompt, df, report)
            
            if query_result is None:
                # Fallback to standard LLM query if no SQL query generated
                st.warning("Unable to generate a specific SQL query. Attempting general LLM query.")
                LLMHandler.query_llm(prompt)
                return
            
            sql_query = query_result['query']
            
            # Execute primary query with error handling
            try:
                primary_df = db_handler.query_db(sql_query)
            except Exception as query_e:
                st.error(f"Database Query Error: {query_e}")
                st.warning(f"Problematic Query: {sql_query}")
                
                # Create error prompt without problematic f-string
                error_prompt_lines = [
                    "An error occurred when trying to execute the following SQL query:",
                    "",
                    f"```sql",
                    f"{sql_query}",
                    f"```",
                    "",
                    f"Error Message: {str(query_e)}",
                    "",
                    "Please help diagnose the issue:",
                    "1. Analyze the query for potential SQL syntax errors",
                    "2. Check if the query matches the actual database schema",
                    "3. Suggest a corrected version of the query",
                    "4. Provide insights into why the query might have failed"
                ]
                error_prompt = "\n".join(error_prompt_lines)
                
                try:
                    error_analysis = model.generate_content(error_prompt)
                    if hasattr(error_analysis, 'text'):
                        st.subheader("🔍 Query Error Analysis")
                        st.markdown(markdown(error_analysis.text), unsafe_allow_html=True)
                except Exception as analysis_e:
                    st.error(f"Additional error analysis failed: {analysis_e}")
                
                return
            
            # Process query results
            if primary_df.empty:
                st.warning("No results found for the query.")
                return
            
            # Prepare markdown table for LLM analysis
            markdown_table = "| " + " | ".join(primary_df.columns) + " |\n"
            markdown_table += "| " + " | ".join(["---"] * len(primary_df.columns)) + " |\n"
            
            for _, row in primary_df.iterrows():
                markdown_table += "| " + " | ".join(str(val) for val in row) + " |\n"
            
            # Prepare analysis prompt with proper string formatting
            analysis_prompt_lines = [
                "Perform a comprehensive analysis of this dataset excerpt:",
                "",
                f'Original Query: "{prompt}"',
                f"SQL Query Used: ```sql",
                f"{sql_query}",
                f"```",
                "",
                f"Dataset Report Context:\n{report}" if report else "",
                "",
                "Dataset Excerpt:",
                markdown_table,
                "",
                "Provide a detailed analysis:",
                "1. Key statistical insights",
                "2. Interesting patterns or anomalies",
                "3. Potential business or research implications",
                "4. Recommended further investigations"
            ]
            analysis_prompt = "\n".join(line for line in analysis_prompt_lines if line)
            
            # Generate analysis
            try:
                analysis_response = model.generate_content(analysis_prompt)
                
                # Display results
                st.subheader("🔍 Query Results")
                st.dataframe(primary_df)
                
                st.subheader("💡 AI-Powered Analysis")
                if hasattr(analysis_response, 'text'):
                    st.markdown(markdown(analysis_response.text), unsafe_allow_html=True)
                
                # Display original LLM reasoning if available
                if 'full_response' in query_result:
                    st.subheader("🧠 Query Generation Reasoning")
                    st.markdown(markdown(query_result['full_response']), unsafe_allow_html=True)
            
            except Exception as analysis_e:
                st.error(f"Error generating analysis: {analysis_e}")
        
        except Exception as e:
            st.error(f"Error in advanced database query: {e}")
    @staticmethod
    def query_llm(prompt):
        """
        Standard LLM query method with improved error handling
        """
        try:
            response = model.generate_content(prompt)
            
            if hasattr(response, 'text'):
                st.markdown(markdown(response.text), unsafe_allow_html=True)
            else:
                st.warning("LLM did not generate a valid response.")
        
        except Exception as e:
            st.error(f"Error querying LLM: {e}")
            st.warning("Unable to process the query. Please try again or rephrase your request.")

# Streamlit App
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