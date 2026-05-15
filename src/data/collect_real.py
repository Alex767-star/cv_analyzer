import requests
import pandas as pd
import time
import os
import json
import random
import numpy as np
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HHResumeCollector:
    """Сбор реальных резюме через API hh.ru"""
    
    BASE_URL = "https://api.hh.ru"
    
    def __init__(self, access_token: str = None):
        self.headers = {"User-Agent": "CV_Analyzer/1.0"}
        if access_token:
            self.headers["Authorization"] = f"Bearer {access_token}"
    
    def search_vacancies(self, text: str, pages: int = 5) -> List[Dict]:
        all_items = []
        
        for page in range(pages):
            params = {
                "text": text,
                "page": page,
                "per_page": 100,
                "area": 1
            }
            
            try:
                response = requests.get(
                    f"{self.BASE_URL}/vacancies",
                    params=params,
                    headers=self.headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    items = data.get("items", [])
                    all_items.extend(items)
                    logger.info(f"Страница {page+1}: получено {len(items)} записей")
                else:
                    logger.warning(f"Ошибка API: {response.status_code}")
                    break
                    
            except Exception as e:
                logger.error(f"Ошибка запроса: {e}")
                break
            
            time.sleep(0.5)
        
        return all_items
    
    def parse_vacancy_details(self, vacancy_id: str) -> Dict:
        try:
            response = requests.get(
                f"{self.BASE_URL}/vacancies/{vacancy_id}",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
                
        except Exception as e:
            logger.error(f"Ошибка получения вакансии {vacancy_id}: {e}")
        
        return {}
    
    def extract_level_from_experience(self, experience: str) -> str:
        mapping = {
            "noExperience": "junior",
            "between1And3": "junior",
            "between3And6": "middle",
            "moreThan6": "senior"
        }
        return mapping.get(experience, "middle")
    
    def collect_dataset(self, queries: List[str], samples_per_query: int = 200) -> pd.DataFrame:
        all_data = []
        
        for query in queries:
            logger.info(f"Сбор данных по запросу: {query}")
            vacancies = self.search_vacancies(query, pages=samples_per_query // 100 + 1)
            
            for v in vacancies[:samples_per_query]:
                details = self.parse_vacancy_details(v.get("id"))
                
                if not details:
                    continue
                
                description = details.get("description", "")
                name = details.get("name", "")
                
                clean_text = f"{name}\n{description}"
                
                experience = details.get("experience", {}).get("id", "between1And3")
                level = self.extract_level_from_experience(experience)
                
                salary = details.get("salary")
                salary_from = salary.get("from") if salary else None
                salary_to = salary.get("to") if salary else None
                
                key_skills = [
                    skill.get("name") for skill in details.get("key_skills", [])
                ]
                
                all_data.append({
                    "text": clean_text,
                    "level": level,
                    "salary_from": salary_from,
                    "salary_to": salary_to,
                    "skills": ", ".join(key_skills),
                    "vacancy_name": name
                })
            
            time.sleep(1)
        
        df = pd.DataFrame(all_data)
        return df

def generate_semi_realistic_data(n_samples: int = 3000) -> pd.DataFrame:
    np.random.seed(42)
    random.seed(42)
    
    real_companies = [
        "Яндекс", "VK", "Сбер", "Тинькофф", "Ozon", "Wildberries",
        "Avito", "Skyeng", "Kaspersky", "Positive Technologies",
        "Лаборатория Касперского", "МТС", "Билайн", "X5 Group"
    ]
    
    real_projects = [
        "разработка high-load API", "миграция монолита на микросервисы",
        "оптимизация ML-пайплайнов", "внедрение CI/CD",
        "создание системы мониторинга", "автоматизация ETL процессов",
        "рефакторинг легаси кода", "запуск рекомендательной системы",
        "построение Data Lake", "миграция в Kubernetes"
    ]
    
    junior_phrases = [
        "прошел курсы по", "изучаю", "делаю пет-проекты",
        "хочу развиваться в", "есть базовые знания",
        "выполнял тестовые задания", "участвовал в хакатонах"
    ]
    
    middle_phrases = [
        "оптимизировал запросы", "настраивал CI/CD", "писал юнит-тесты",
        "участвовал в код-ревью", "проектировал REST API",
        "работал с высоконагруженными сервисами", "внедрял мониторинг"
    ]
    
    senior_phrases = [
        "руководил командой из", "проектировал архитектуру",
        "принимал технические решения", "менторил разработчиков",
        "оптимизировал затраты на инфраструктуру",
        "выстраивал процессы разработки", "проводил технические собеседования"
    ]
    
    data = []
    
    for _ in range(n_samples):
        level = np.random.choice(['junior', 'middle', 'senior'], p=[0.3, 0.4, 0.3])
        
        if level == 'junior':
            years = random.randint(0, 1)
            phrases = random.sample(junior_phrases, 2)
            company = random.choice(real_companies[:3])
        elif level == 'middle':
            years = random.randint(2, 4)
            phrases = random.sample(middle_phrases, 3)
            company = random.choice(real_companies[3:8])
        else:
            years = random.randint(5, 10)
            phrases = random.sample(senior_phrases, 3)
            company = random.choice(real_companies[8:])
        
        projects = random.sample(real_projects, random.randint(1, 3))
        
        text = f"{company}\n"
        text += f"Опыт: {years} {'год' if years == 1 else 'года' if years < 5 else 'лет'}\n"
        text += f"Проекты: {', '.join(projects)}\n"
        text += f"{' '.join(phrases)}\n"
        
        tech_stack = random.sample([
            "Python", "Go", "Java", "Kotlin", "TypeScript",
            "PostgreSQL", "MongoDB", "Redis", "Elasticsearch",
            "Docker", "Kubernetes", "Terraform", "Ansible",
            "Kafka", "RabbitMQ", "gRPC", "GraphQL"
        ], random.randint(3, 8))
        
        text += f"Стек: {', '.join(tech_stack)}"
        
        salary = {
            'junior': np.random.randint(60000, 120000),
            'middle': np.random.randint(120000, 280000),
            'senior': np.random.randint(280000, 500000)
        }[level]
        
        data.append({
            "text": text,
            "level": level,
            "experience_years": years,
            "salary": salary,
            "skills": ", ".join(tech_stack)
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_semi_realistic_data(5000)
    output_path = os.path.join(os.path.dirname(__file__), '../../data/training_data.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Сгенерировано {len(df)} реалистичных резюме")
    print(df['level'].value_counts())
    print(f"Сохранено в {output_path}")
