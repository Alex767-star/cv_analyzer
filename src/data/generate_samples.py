import numpy as np
import pandas as pd
import os
import random

def generate_training_data(n_samples: int = 5000) -> pd.DataFrame:
    np.random.seed(42)
    random.seed(42)
    
    junior_skills = [
        'Python базовый', 'SQL простые запросы', 'Git основы',
        'HTML/CSS', 'JavaScript начальный', 'Docker базовый',
        'Linux командная строка', 'ООП основы', 'алгоритмы'
    ]
    
    middle_skills = [
        'Django', 'FastAPI', 'PostgreSQL', 'MongoDB', 'Redis',
        'Docker', 'Kubernetes', 'CI/CD', 'микросервисы',
        'REST API', 'gRPC', 'Kafka', 'оптимизация запросов',
        'тестирование', 'code review', 'архитектура приложений'
    ]
    
    senior_skills = [
        'высоконагруженные системы', 'архитектура микросервисов',
        'управление командой', 'техническая стратегия',
        'performance optimization', 'SRE практики', 'мониторинг Prometheus',
        'отказоустойчивость', 'собеседования', 'менторинг',
        'проектирование БД', 'capacity planning'
    ]
    
    data = []
    
    for _ in range(n_samples):
        level = np.random.choice(['junior', 'middle', 'senior'], p=[0.35, 0.40, 0.25])
        
        if level == 'junior':
            years = random.randint(0, 1)
            skill_count = random.randint(2, 4)
            skills = random.sample(junior_skills, skill_count)
            
            templates = [
                f"Ищу первую работу. Изучал {', '.join(skills)}. Опыта нет, но есть желание развиваться.",
                f"Начинающий разработчик. Знаю {', '.join(skills)}. Проходил курсы, делал пет-проекты.",
                f"Junior разработчик. {years} год опыта. Работал с {', '.join(skills)}. Хочу расти.",
                f"Выпускник курсов. Стек: {', '.join(skills)}. Готов выполнять тестовые задания.",
                f"Начальный уровень. Понимаю {', '.join(skills)}. Ищу ментора и интересные задачи."
            ]
            
        elif level == 'middle':
            years = random.randint(2, 5)
            skill_count = random.randint(4, 7)
            skills = random.sample(middle_skills, skill_count)
            
            templates = [
                f"Опыт {years} года. Стек: {', '.join(skills)}. Делал REST API, микросервисы.",
                f"Middle разработчик. {years} года опыта. Проектировал БД, писал тесты, делал код-ревью.",
                f"Разработчик с опытом {years} года. {', '.join(skills)}. Оптимизировал запросы, настраивал CI/CD.",
                f"{years} года коммерческой разработки. Владею {', '.join(skills)}. Работал в продуктовых командах.",
                f"Middle уровень. {years} года опыта. Писал микросервисы на {skills[0]}, работал с {skills[1]}."
            ]
            
        else:
            years = random.randint(5, 12)
            skill_count = random.randint(5, 8)
            skills = random.sample(senior_skills + middle_skills, skill_count)
            
            templates = [
                f"Senior разработчик. {years} лет опыта. Руководил командой из {random.randint(3,10)} человек. {', '.join(skills)}.",
                f"Lead разработчик с {years} годами опыта. Проектировал архитектуру, менторил джуниоров. Стек: {', '.join(skills)}.",
                f"Опыт {years}+ лет. Внедрял {', '.join(skills[:3])}. Оптимизировал высоконагруженные системы.",
                f"Senior level. {years} лет в индустрии. Управлял командами, принимал архитектурные решения. {', '.join(skills)}.",
                f"Технический лидер. {years} лет опыта. Строил процессы разработки, внедрял best practices. Стек: {', '.join(skills)}."
            ]
        
        text = random.choice(templates)
        
        salary = {
            'junior': np.random.randint(50000, 100000),
            'middle': np.random.randint(100000, 250000),
            'senior': np.random.randint(250000, 450000)
        }[level]
        
        data.append({
            'text': text,
            'level': level,
            'salary': salary,
            'experience_years': years
        })
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)
    
    os.makedirs(os.path.join(os.path.dirname(__file__), '../../data'), exist_ok=True)
    output_path = os.path.join(os.path.dirname(__file__), '../../data/training_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    return df

if __name__ == "__main__":
    df = generate_training_data(5000)
    print(f"Generated {len(df)} samples")
    print(df['level'].value_counts())
