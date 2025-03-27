import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from database.db_handler import DatabaseHandler
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