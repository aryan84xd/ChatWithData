import re
import streamlit as st
from markdown import markdown
from google.generativeai import configure, GenerativeModel
from config.settings import GEMINI_API_KEY

# Configure Gemini API
configure(api_key=GEMINI_API_KEY)
model = GenerativeModel("gemini-1.5-flash")

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
