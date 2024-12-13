import pandas as pd
import numpy as np

# Загрузка данных
train = pd.read_csv("train.csv")
lectures = pd.read_csv("lectures.csv")
questions = pd.read_csv("questions.csv")

# --- Анализ и подготовка данных ---

# 1. Обработка пропусков
train['prior_question_elapsed_time'] = train['prior_question_elapsed_time'].where(
    train['prior_question_elapsed_time'].notna(), 
    train['prior_question_elapsed_time'].mean()
)
train['prior_question_had_explanation'] = train['prior_question_had_explanation'].where(
    train['prior_question_had_explanation'].notna(), 
    False
)

# 2. Добавление метаданных вопросов и лекций
train = train.merge(questions, left_on='content_id', right_on='question_id', how='left')
train = train.merge(lectures, left_on='content_id', right_on='lecture_id', how='left')

# 3. Создание новых признаков
train['is_question'] = train['content_type_id'] == 0
train['is_lecture'] = train['content_type_id'] == 1

train['question_elapsed_time_ratio'] = train['prior_question_elapsed_time'] / train['prior_question_elapsed_time'].mean()
train['correct_vs_total_ratio'] = train.groupby('user_id')['answered_correctly'].transform('mean')

# 4. Группировка и расчеты
user_stats = train.groupby('user_id').agg({
    'answered_correctly': ['mean', 'sum'],
    'prior_question_elapsed_time': 'mean',
    'prior_question_had_explanation': 'mean',
    'is_question': 'sum',
    'is_lecture': 'sum'
}).reset_index()
user_stats.columns = ['user_id', 'accuracy', 'total_correct', 'avg_time', 'explanation_ratio', 'total_questions', 'total_lectures']

# --- Анализ характеристик ---

# 1. Корреляция характеристик с успеваемостью
correlation_matrix = user_stats.corr()

# 2. Влияние времени на успеваемость
train['time_buckets'] = pd.qcut(train['prior_question_elapsed_time'], q=4, labels=['short', 'medium', 'long', 'very_long'])
time_analysis = train.groupby('time_buckets', observed=False)['answered_correctly'].mean()

# 3. Анализ лекций
lecture_analysis = train[train['is_lecture']].groupby('type_of')['user_id'].count()

# --- Выводы ---

print("Correlation matrix:\n", correlation_matrix)
print("\nImpact of time buckets:\n", time_analysis)
print("\nLecture analysis:\n", lecture_analysis)

# Сохранение пользовательской статистики для дальнейшего использования
user_stats.to_csv("user_stats.csv", index=False)
