# Student Satisfaction Survey Analysis

This repository contains the analysis of a student satisfaction survey, providing insights into the overall student experience, teaching quality, and areas for improvement.

## Project Overview

The primary goal of this project was to analyze student feedback data collected via a Google Form to identify satisfaction trends and propose actionable recommendations. The analysis involved:

* **Data Preprocessing:** Cleaning and preparing the raw survey data, including handling encoding issues, splitting combined fields (`Average/ Percentage` into `Average_Score` and `Percentage_Value`), managing missing values, and addressing outliers through percentile capping.
* **Descriptive Statistics:** Summarizing key numerical metrics to understand data distribution and central tendencies.
* **Performance Analysis by Course:** Evaluating student satisfaction across different `Basic Courses` and `Course Names` to identify top-performing programs and those with opportunities for enhancement.
* **Question-Level Classification:** Categorizing individual survey questions based on average satisfaction scores (Excellent, Good, Average, Poor, Very Poor) to pinpoint specific strengths and areas needing attention.

## Key Findings

* **High Overall Satisfaction:** Students generally exhibit high satisfaction, with all survey questions falling into the "Good" or "Excellent" categories.
* **Strengths:** Key strengths identified include the fairness of internal evaluation, effective teacher communication, and teachers' ability to illustrate concepts and prepare for classes.
* **Varied Course Performance:** While overall satisfaction is positive, there are notable differences among courses. MSC Information Technology and Bachelor of Commerce (Banking and Insurance) showed the highest satisfaction, while MSC Data Science and B.SC. Computer Science had relatively lower (though still "Good") scores.
* **Areas for Enhancement:** Opportunities for improvement exist in areas such as the effective use of ICT tools by teachers, syllabus coverage, and the promotion of extracurricular and internship opportunities.

## Recommendations

Based on the analysis, recommendations include:

* **Leveraging Strengths:** Sharing best practices from high-performing areas across the institution.
* **Targeted Improvements:** Providing training for ICT tool usage, reviewing syllabus coverage, and boosting extracurricular engagement.
* **Course-Specific Deep Dives:** Conducting further qualitative analysis for courses with relatively lower satisfaction to understand specific pain points.
* **Future Survey Enhancements:** Implementing sentiment analysis for qualitative comments to gain deeper insights and refining survey questionnaires for continuous improvement.

## Data and Visualizations

The analysis utilized the `Student_Satisfaction_Survey.csv` dataset. All generated plots, including box plots, density distributions, and performance analyses by course and question classification, are available in `Student_Satisfaction_Analysis_Plots.pdf`.
