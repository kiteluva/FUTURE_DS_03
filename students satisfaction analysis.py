import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from matplotlib.backends.backend_pdf import PdfPages  # Import PdfPages

# --- Custom Dark Theme for Plots ---
plt.rcParams.update({
    "figure.facecolor": "black",
    "axes.facecolor": "black",
    "axes.edgecolor": "#666666",
    "text.color": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "axes.labelcolor": "white",
    "grid.color": "#333333",
    "grid.linestyle": "--",
    "figure.titlesize": 10,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "font.size": 7,
    "lines.linewidth": 2,
    "patch.edgecolor": "white",
})


# --- Function for truncating labels ---
def truncate_label(label, max_length=40):
    if len(label) > max_length:
        return label[:max_length - 3] + '...'
    return label


# --- 1. Load the Dataset ---
try:
    df = pd.read_csv("Student_Satisfaction_Survey.csv")
    print("Dataset loaded successfully with default (UTF-8) encoding!")
except FileNotFoundError:
    print("Error: 'Student_Satisfaction_Survey.csv' not found. Please check the file path.")
    exit()
except UnicodeDecodeError:
    try:
        df = pd.read_csv("Student_Satisfaction_Survey.csv", encoding='latin1')
        print("Dataset loaded successfully with 'latin1' encoding!")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv("Student_Satisfaction_Survey.csv", encoding='cp1252')
            print("Dataset loaded successfully with 'cp1252' encoding!")
        except UnicodeDecodeError:
            print("Error: Could not decode the file with 'latin1' or 'cp1252' encoding.")
            print("Please check the file's original encoding or its content for problematic characters.")
            exit()
    except FileNotFoundError:
        print("Error: 'Student_Satisfaction_Survey.csv' not found. Please check the file path.")
        exit()

# --- Data Cleaning: Split and Convert 'Average/ Percentage' column ---
if 'Average/ Percentage' in df.columns:
    print("Splitting 'Average/ Percentage' into 'Average_Score' and 'Percentage_Value'...")
    # Split the column by '/' and expand into two new temporary columns
    split_data = df['Average/ Percentage'].str.split('/', expand=True)

    if split_data.shape[1] == 2:
        # Convert the first part to numeric for 'Average_Score'
        df['Average_Score'] = pd.to_numeric(split_data[0], errors='coerce')
        # Convert the second part to numeric for 'Percentage_Value'
        df['Percentage_Value'] = pd.to_numeric(split_data[1], errors='coerce')

        # Handle missing values introduced by coercion in new columns
        if df['Average_Score'].isnull().sum() > 0:
            median_avg_score = df['Average_Score'].median()
            if pd.isna(median_avg_score):
                print("Warning: Median for 'Average_Score' is NaN. Cannot fill missing values.")
            else:
                df['Average_Score'].fillna(median_avg_score, inplace=True)
                print(f"Missing values in 'Average_Score' filled with median: {median_avg_score:.2f}")

        if df['Percentage_Value'].isnull().sum() > 0:
            median_percentage_value = df['Percentage_Value'].median()
            if pd.isna(median_percentage_value):
                print("Warning: Median for 'Percentage_Value' is NaN. Cannot fill missing values.")
            else:
                df['Percentage_Value'].fillna(median_percentage_value, inplace=True)
                print(f"Missing values in 'Percentage_Value' filled with median: {median_percentage_value:.2f}")

        # Drop the original 'Average/ Percentage' column
        df.drop('Average/ Percentage', axis=1, inplace=True)
        print("Original 'Average/ Percentage' column removed.")
    else:
        print(
            "Warning: 'Average/ Percentage' column does not contain expected '/' delimiter or format. Skipping split.")
        # If format is unexpected, drop the original column to avoid issues later
        df.drop('Average/ Percentage', axis=1, inplace=True)
        print("Original 'Average/ Percentage' column removed due to unexpected format.")
else:
    print("'Average/ Percentage' column not found in the dataset (no action taken).")

# --- Initialize PDF for saving plots ---
print("All plots will be saved to a PDF file")
pdf_filename = "Student_Satisfaction_Analysis_Plots.pdf"
pdf_pages = PdfPages(pdf_filename)
print(f"All plots will be saved to '{pdf_filename}'")

# --- 2. Initial Inspection & Overview ---
print("--- 2. Initial Inspection & Overview ---")

# Display the first few rows of the DataFrame
print("2.1 Head of the DataFrame:")
print(df.head())

# Display the last few rows
print("2.2 Tail of the DataFrame:")
print(df.tail())

# Get a concise summary of the DataFrame, including data types and non-null values
print("2.3 DataFrame Info (Data Types & Non-Null Counts):")
df.info()

# Get descriptive statistics for numerical columns
print("2.4 Descriptive Statistics for Numerical Columns:")
print(df.describe())

# Get descriptive statistics for all columns, including categorical
print("2.5 Descriptive Statistics for All Columns (including categorical):")
print(df.describe(include='all'))

# Get the shape of the DataFrame (number of rows, number of columns)
print("2.6 Shape of the DataFrame (Rows, Columns):")
print(df.shape)

# Get the column names
print("2.7 Column Names:")
print(df.columns.tolist())

# --- 3. Missing Values Inspection ---
print("--- 3. Missing Values Inspection ---")

# Check for missing values in each column
print("3.1 Number of Missing Values per Column:")
print(df.isnull().sum())

# Check for missing values as a percentage
print("3.2 Percentage of Missing Values per Column:")
print((df.isnull().sum() / len(df)) * 100)

# Visualize missing values (requires matplotlib and seaborn)
# Plot only if there are missing values
if df.isnull().sum().sum() > 0:
    fig_missing, ax_missing = plt.subplots(figsize=(10, 6))  # Create a figure object
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax_missing)
    ax_missing.set_title('Missing Values Heatmap')
    pdf_pages.savefig(fig_missing)  # Save the figure to PDF
    plt.close(fig_missing)  # Close the figure to free memory
else:
    print("No missing values found in the dataset. Heatmap not generated.")

# --- 4. Duplicate Values Inspection ---
print("--- 4. Duplicate Values Inspection ---")

# Check for duplicate rows
print("4.1 Number of Duplicate Rows:")
print(df.duplicated().sum())

# Display the duplicate rows (if any)
if df.duplicated().sum() > 0:
    print("4.2 Displaying Duplicate Rows (first 5 if many):")
    print(df[df.duplicated()].head())
else:
    print("No duplicate rows found.")

# --- 5. Outlier Inspection (for Numerical Columns) ---
print("--- 5. Outlier Inspection (for Numerical Columns) ---")

print("5.1 Interquartile Range (IQR) and Potential Outliers (for numerical columns):")
numerical_cols_with_outliers = []  # To store columns identified with outliers for later handling
for col in df.select_dtypes(include=np.number).columns:
    # Skip 'SN' as it's a serial number and not a metric where outliers make sense for capping
    if col == 'SN':
        print(f"- Column '{col}': Skipping outlier detection as it's a serial number.")
        continue

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
    if outliers_count > 0:
        print(f"- Column '{col}': {outliers_count} potential outliers (outside [{lower_bound:.2f}, {upper_bound:.2f}])")
        print(df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].head())
        numerical_cols_with_outliers.append(col)  # Add to list for handling
    else:
        print(f"- Column '{col}': No obvious outliers detected by IQR method.")

# --- 6. Outlier Handling (Capping with Percentiles) ---
print("--- 6. Outlier Handling (Capping with Percentiles) ---")
print("Replacing outliers with 5th and 95th percentiles for identified numerical columns.")

for col in numerical_cols_with_outliers:
    lower_percentile = df[col].quantile(0.05)
    upper_percentile = df[col].quantile(0.95)

    # Cap values below the lower percentile
    df[col] = np.where(df[col] < lower_percentile, lower_percentile, df[col])
    # Cap values above the upper percentile
    df[col] = np.where(df[col] > upper_percentile, upper_percentile, df[col])
    print(f"- Column '{col}': Outliers capped between {lower_percentile:.2f} and {upper_percentile:.2f}")

# Box plots for visual outlier detection - CONSOLIDATED PLOT (after potential capping)
print(
    "6.2 Generating Consolidated Box Plots for Numerical Columns (visual outlier detection, after outlier handling):")

# Re-filter numerical columns, ensuring 'SN' is excluded if it's not a metric
numerical_cols_for_plots = [col for col in df.select_dtypes(include=np.number).columns if col != 'SN']
num_plots = len(numerical_cols_for_plots)

if num_plots > 0:
    n_cols = 3
    n_rows = math.ceil(num_plots / n_cols)

    fig_box, axes_box = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))  # Create a figure object
    axes_box = axes_box.flatten()

    for i, col in enumerate(numerical_cols_for_plots):
        # Explicitly convert Series to numpy array for robustness in boxplot
        data_for_plot = df[col].values
        sns.boxplot(x=data_for_plot, ax=axes_box[i], color='steelblue')
        axes_box[i].set_title(f'Box Plot of {col}')
        axes_box[i].set_xlabel('')

    for j in range(i + 1, len(axes_box)):
        fig_box.delaxes(axes_box[j])

    plt.tight_layout()
    plt.suptitle('Consolidated Box Plots for Numerical Columns (After Outlier Capping)', y=1.02, fontsize=16)
    pdf_pages.savefig(fig_box)
    plt.close(fig_box)
else:
    print("No numerical columns found to generate box plots after filtering 'SN'.")

# --- 7. Plotting Density Distributions ---
print("--- 7.1. Plotting Density Distributions ---")
# Use the same numerical_cols_for_plots as for box plots
if num_plots > 0:
    n_cols_density = 3
    n_rows_density = math.ceil(num_plots / n_cols_density)

    fig_density, axes_density = plt.subplots(n_rows_density, n_cols_density,
                                             figsize=(n_cols_density * 5, n_rows_density * 3))  # Create a figure object
    axes_density = axes_density.flatten()
    fig_density.suptitle('Density Distributions for Numerical Columns (After Cleaning)', y=1.02, fontsize=18)

    for i, col in enumerate(numerical_cols_for_plots):
        sns.kdeplot(x=df[col], ax=axes_density[i], fill=True, color='steelblue')
        axes_density[i].set_title(f'Density of {col}')
        axes_density[i].set_xlabel(col)
        axes_density[i].set_ylabel('Density')

    # Hide any unused subplots
    for j in range(i + 1, len(axes_density)):
        fig_density.delaxes(axes_density[j])

    plt.tight_layout()
    pdf_pages.savefig(fig_density)  # Save the figure to PDF
    plt.close(fig_density)  # Close the figure to free memory
else:
    print("No numerical columns found to generate density plots after filtering 'SN'.")

# --- 8. Summary and Descriptive Statistics (Post-Cleaning) ---
print("--- 8. Summary and Descriptive Statistics (Post-Cleaning) ---")
print("Descriptive statistics for numerical columns after outlier capping:")
desc_stats_df = df.describe()
print(desc_stats_df)

# --- 8.1. Plotting Descriptive Statistics ---
print("--- 8.1. Plotting Descriptive Statistics ---")
numerical_cols_for_desc_plots = [col for col in df.select_dtypes(include=np.number).columns if col != 'SN']

if not desc_stats_df.empty and numerical_cols_for_desc_plots:
    stats_to_plot = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
    num_stats = len(stats_to_plot)
    n_cols_desc_plot = 3
    n_rows_desc_plot = math.ceil(num_stats / n_cols_desc_plot)

    fig_desc, axes_desc = plt.subplots(n_rows_desc_plot, n_cols_desc_plot,
                                       figsize=(n_cols_desc_plot * 5, n_rows_desc_plot * 3))  # Create a figure object
    axes_desc = axes_desc.flatten()
    fig_desc.suptitle('Descriptive Statistics for Numerical Columns (After Cleaning)', y=1.02, fontsize=18)

    for i, stat in enumerate(stats_to_plot):
        if stat in desc_stats_df.index:
            plot_data = desc_stats_df.loc[stat, numerical_cols_for_desc_plots].sort_values(ascending=False)
            plot_data.index = plot_data.index.map(lambda x: x.replace("_", " "))
            plot_data.plot(kind='barh', ax=axes_desc[i], color='steelblue')
            axes_desc[i].set_title(f'{stat.capitalize()}')
            axes_desc[i].set_xlabel('Value')
            axes_desc[i].set_ylabel('Metric')
            axes_desc[i].invert_yaxis()
        else:
            print(f"Warning: Statistic '{stat}' not found in descriptive statistics DataFrame.")

    # Hide any unused subplots
    for j in range(i + 1, len(axes_desc)):
        fig_desc.delaxes(axes_desc[j])

    plt.tight_layout()
    pdf_pages.savefig(fig_desc)
    plt.close(fig_desc)
else:
    print("No numerical columns or descriptive statistics available for plotting.")

# --- 9. Grouped Analysis by Basic Course (Overall Average Weightage) ---
print("--- 9. Grouped Analysis by Basic Course (Overall Average Weightage) ---")

# column names based on df.columns.tolist() output
basic_course_col = 'Basic Course'
course_name_col = 'Course Name '

# Check if the grouping column exists
if basic_course_col in df.columns:
    print(f"Grouping data by '{basic_course_col}'.")

    # Define the metrics for analysis
    metrics_for_basic_course_analysis = [
        'Weightage 1', 'Weightage 2', 'Weightage 3', 'Weightage 4', 'Weightage 5',
        'Average_Score', 'Percentage_Value'
    ]
    # Filter to include only available numeric metrics
    available_metrics_for_basic_course_analysis = [m for m in metrics_for_basic_course_analysis if
                                                   m in df.columns and pd.api.types.is_numeric_dtype(df[m])]

    if available_metrics_for_basic_course_analysis:
        # Group by Basic Course and calculate mean of relevant columns
        grouped_by_basic_course_overall = df.groupby(basic_course_col)[
            available_metrics_for_basic_course_analysis].mean()
        print("Average Scores by Basic Course:")
        print(grouped_by_basic_course_overall)

        if not grouped_by_basic_course_overall.empty:
            if 'Average_Score' in grouped_by_basic_course_overall.columns:
                grouped_by_basic_course_overall['Overall_Metric_for_Plot'] = grouped_by_basic_course_overall[
                    'Average_Score']
                overall_metric_label = 'Average Score'
            else:
                print("No suitable metric for overall Basic Course plotting found.")
                grouped_by_basic_course_overall['Overall_Metric_for_Plot'] = 0  # Placeholder to avoid error
                overall_metric_label = 'N/A'

            if 'Overall_Metric_for_Plot' in grouped_by_basic_course_overall.columns:

                sorted_basic_courses = grouped_by_basic_course_overall.sort_values(by='Overall_Metric_for_Plot',
                                                                                   ascending=False)

                print(f"Basic Courses by {overall_metric_label} (Sorted):")
                print(sorted_basic_courses)

                # Plotting overall performance by Basic Course
                fig_overall_basic, ax_overall_basic = plt.subplots(figsize=(12, 10))  # Create a figure object
                sorted_basic_courses['Overall_Metric_for_Plot'].plot(kind='barh', color='steelblue',
                                                                     ax=ax_overall_basic)
                ax_overall_basic.set_title(
                    f'Overall Average Metric Performance by Basic Course ({overall_metric_label})')
                ax_overall_basic.set_xlabel(overall_metric_label)
                ax_overall_basic.set_ylabel('Basic Course', fontsize=8)
                ax_overall_basic.invert_yaxis()
                plt.tight_layout()
                pdf_pages.savefig(fig_overall_basic)  # Save the figure to PDF
                plt.close(fig_overall_basic)  # Close the figure to free memory

            else:
                print("No suitable overall metric for plotting.")
        else:
            print("No data to group by Basic Course after filtering.")
    else:
        print("No relevant numeric metrics found for Basic Course analysis.")
else:
    print(f"Column '{basic_course_col}' not found in the dataset. Skipping Basic Course analysis.")

# --- 10. Performance Analysis by Basic Course (All Metrics - Detailed) ---
print("--- 10. Performance Analysis by Basic Course (All Metrics - Detailed) ---")

# Define metrics to plot (now includes Average_Score and Percentage_Value)
metrics_to_plot_detailed = [
    'Total Feedback Given', 'Total Configured',
    'Weightage 1', 'Weightage 2', 'Weightage 3', 'Weightage 4', 'Weightage 5',
    'Average_Score', 'Percentage_Value'
]

# Filter metrics to only include those present in the DataFrame and are numeric
available_metrics_for_plotting_detailed = [m for m in metrics_to_plot_detailed if
                                           m in df.columns and pd.api.types.is_numeric_dtype(df[m])]

if basic_course_col in df.columns and available_metrics_for_plotting_detailed:
    grouped_by_basic_course_detailed = df.groupby(basic_course_col)[available_metrics_for_plotting_detailed].mean()
    print("Detailed Average Metric Performance by Basic Course:")
    print(grouped_by_basic_course_detailed)  # Print the grouped DataFrame to console

    num_metrics = len(available_metrics_for_plotting_detailed)
    n_cols_plot = 3
    n_rows_plot = math.ceil(num_metrics / n_cols_plot)

    fig_basic, axes_basic = plt.subplots(n_rows_plot, n_cols_plot,
                                         figsize=(n_cols_plot * 5, n_rows_plot * 3))  # Create a figure object
    axes_basic = axes_basic.flatten()
    fig_basic.suptitle('Average Metric Performance by Basic Course', y=1.02, fontsize=18)

    for i, metric in enumerate(available_metrics_for_plotting_detailed):
        if metric in grouped_by_basic_course_detailed.columns:
            plot_data = grouped_by_basic_course_detailed[metric].sort_values(ascending=False)
            # Apply truncation to the index (labels) for plot display
            plot_data.index = plot_data.index.map(truncate_label)
            plot_data.plot(kind='barh', ax=axes_basic[i], color='steelblue')
            axes_basic[i].set_title(f'Avg {metric.replace("_", " ")}')
            axes_basic[i].set_xlabel(f'Avg {metric.replace("_", " ")}')
            axes_basic[i].set_ylabel('Basic Course', fontsize=8)
            axes_basic[i].invert_yaxis()
        else:
            print(f"Warning: Metric '{metric}' not found in grouped data for Basic Course.")

    # Hide any unused subplots
    for j in range(i + 1, len(axes_basic)):
        fig_basic.delaxes(axes_basic[j])

    plt.tight_layout()
    pdf_pages.savefig(fig_basic)  # Save the figure to PDF
    plt.close(fig_basic)  # Close the figure to free memory
else:
    print(
        f"Skipping Basic Course detailed analysis: '{basic_course_col}' column or relevant numeric metrics not found.")

# --- 11. Performance Analysis by Course Name (All Metrics) ---
print("--- 11. Performance Analysis by Course Name (All Metrics) ---")

# Define metrics to plot (now includes Average_Score and Percentage_Value)
metrics_to_plot_course_name = [
    'Total Feedback Given', 'Total Configured',
    'Weightage 1', 'Weightage 2', 'Weightage 3', 'Weightage 4', 'Weightage 5',
    'Average_Score', 'Percentage_Value'
]

# Filter metrics to only include those present in the DataFrame and are numeric
available_metrics_for_plotting_course_name = [m for m in metrics_to_plot_course_name if
                                              m in df.columns and pd.api.types.is_numeric_dtype(df[m])]

if course_name_col in df.columns and available_metrics_for_plotting_course_name:
    grouped_by_course_name = df.groupby(course_name_col)[available_metrics_for_plotting_course_name].mean()
    print("Average Metric Performance by Course Name:")
    print(grouped_by_course_name)

    num_metrics = len(available_metrics_for_plotting_course_name)
    n_cols_plot = 3
    n_rows_plot = math.ceil(num_metrics / n_cols_plot)

    fig_course, axes_course = plt.subplots(n_rows_plot, n_cols_plot,
                                           figsize=(n_cols_plot * 5, n_rows_plot * 3))  # Create a figure object
    axes_course = axes_course.flatten()
    fig_course.suptitle('Average Metric Performance by Course Name', y=1.02, fontsize=18)

    for i, metric in enumerate(available_metrics_for_plotting_course_name):
        if metric in grouped_by_course_name.columns:
            plot_data = grouped_by_course_name[metric].sort_values(ascending=False)
            # Apply truncation to the index (labels) for plot display
            plot_data.index = plot_data.index.map(truncate_label)
            plot_data.plot(kind='barh', ax=axes_course[i], color='steelblue')
            axes_course[i].set_title(f'Avg {metric.replace("_", " ")}')
            axes_course[i].set_xlabel(f'Avg {metric.replace("_", " ")}')
            axes_course[i].set_ylabel('Course Name', fontsize=5)
            axes_course[i].invert_yaxis()
        else:
            print(f"Warning: Metric '{metric}' not found in grouped data for Course Name.")

    # Hide any unused subplots
    for j in range(i + 1, len(axes_course)):
        fig_course.delaxes(axes_course[j])

    plt.tight_layout()
    pdf_pages.savefig(fig_course)
    plt.close(fig_course)
else:
    print(f"Skipping Course Name analysis: '{course_name_col}' column or relevant numeric metrics not found.")

# --- 12. Question-level Analysis and Classification ---
print("--- 12. Question-level Analysis and Classification ---")

if 'Questions' in df.columns:
    # Define the primary metric for question classification
    question_metrics = ['Average_Score']
    if not all(m in df.columns for m in question_metrics):
        weightage_cols = [f'Weightage {i}' for i in range(1, 6) if f'Weightage {i}' in df.columns]
        if weightage_cols:
            df['Calculated_Avg_Weightage'] = df[weightage_cols].mean(axis=1)
            question_metrics = ['Calculated_Avg_Weightage']
            classification_metric = 'Calculated_Avg_Weightage'
        else:
            print("No 'Average_Score' or 'Weightage' columns found for question classification.")
            classification_metric = None
    else:
        classification_metric = 'Average_Score'

    if classification_metric:
        # Group by 'Questions' and calculate the mean of the classification metric
        question_performance = df.groupby('Questions')[classification_metric].mean().sort_values(ascending=False)

        # Define classification bins and labels based on a 0-5 scale
        bins = [0, 1, 2, 3, 4, 5.00001]
        labels = ['Very Poor', 'Poor', 'Average', 'Good', 'Excellent']

        # Apply classification
        question_classification = pd.cut(question_performance, bins=bins, labels=labels, right=False)
        classified_questions_df = pd.DataFrame({
            'Average Score': question_performance,
            'Classification': question_classification
        })

        print("Classification of Questions by Average Score:")
        print(classified_questions_df)

        # Plotting the classification
        fig_question_classification, ax_question_classification = plt.subplots(figsize=(14, 10))
        sns.barplot(x='Average Score', y=classified_questions_df.index.map(truncate_label),
                    hue='Classification', data=classified_questions_df.reset_index(),
                    palette={'Excellent': 'green', 'Good': 'lightgreen', 'Average': 'steelblue',
                             'Poor': 'orange', 'Very Poor': 'red'},
                    dodge=False, ax=ax_question_classification)
        ax_question_classification.set_title('Question Classification by Average Score')
        ax_question_classification.set_xlabel('Average Score')
        ax_question_classification.set_ylabel('Question')
        ax_question_classification.set_xlim(0, 5)
        ax_question_classification.legend(title='Satisfaction Level', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        pdf_pages.savefig(fig_question_classification)
        plt.close(fig_question_classification)

    else:
        print("Cannot perform question classification due to missing metric columns.")
else:
    print("Column 'Questions' not found in the dataset. Skipping question-level analysis.")

# --- 13. Correlation Analysis ---
print("--- 13. Correlation Analysis (Weightages vs. Average Score) ---")

weightage_cols = [col for col in df.columns if 'Weightage' in col and pd.api.types.is_numeric_dtype(df[col])]
target_col = 'Average_Score'

if target_col in df.columns and weightage_cols:
    cols_for_correlation = [target_col] + weightage_cols

    # Calculate the correlation matrix
    correlation_matrix = df[cols_for_correlation].corr()
    print("Correlation Matrix (Weightages and Average_Score):")
    print(correlation_matrix)

    # Extract correlations specifically with Average_Score
    if target_col in correlation_matrix.columns:
        correlations_with_avg_score = correlation_matrix[target_col].drop(target_col)
        print(f"Correlations with {target_col}:")
        print(correlations_with_avg_score.sort_values(ascending=False))

        # Plotting the correlation heatmap
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr)
        ax_corr.set_title(f'Correlation Matrix: {target_col} vs. Weightages')
        plt.tight_layout()
        pdf_pages.savefig(fig_corr)
        plt.close(fig_corr)

        # Plotting individual correlations with Average_Score (bar plot for clarity)
        fig_single_corr, ax_single_corr = plt.subplots(figsize=(10, 6))
        correlations_with_avg_score.sort_values(ascending=False).plot(kind='barh', color='lightcoral',
                                                                      ax=ax_single_corr)
        ax_single_corr.set_title(f'Correlation of Weightage Metrics with {target_col}')
        ax_single_corr.set_xlabel('Correlation Coefficient')
        ax_single_corr.set_ylabel('Weightage Metric')
        ax_single_corr.invert_yaxis()
        plt.tight_layout()
        pdf_pages.savefig(fig_single_corr)
        plt.close(fig_single_corr)


    else:
        print(f"'{target_col}' not found in the correlation matrix, perhaps due to missing data.")
else:
    print("Skipping correlation analysis: 'Average_Score' column or 'Weightage' columns not found or not numeric.")

# --- Close the PDF file ---
pdf_pages.close()
print(f"All plots have been saved to '{pdf_filename}'")

# --- Save the cleaned dataset to an Excel file ---
excel_filename = "Student_Satisfaction_Survey_Cleaned.xlsx"
df.to_excel(excel_filename, index=False)
print(f"Cleaned dataset saved successfully to '{excel_filename}'")

print("-- Data Inspection and Analysis Complete ---")