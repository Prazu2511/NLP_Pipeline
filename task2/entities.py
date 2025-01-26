import json
import re

with open("./task2/domain_knowledge.json", "r") as file:
    domain_knowledge = json.load(file)

competitors = domain_knowledge["competitors"]
features = domain_knowledge["features"]
pricing_keywords = domain_knowledge["pricing_keywords"]
security_keywords = domain_knowledge["security_keywords"]

def extract_from_knowledge_base(text):
    entities = {"competitors": [], "features": [], "pricing_keywords": [], "security_keywords": []}
    for competitor in competitors:
        if competitor.lower() in text.lower():
            entities["competitors"].append(competitor)
    
    for feature in features:
        if feature.lower() in text.lower():
            entities["features"].append(feature)
    
    for keyword in pricing_keywords:
        if keyword.lower() in text.lower():
            entities["pricing_keywords"].append(keyword)
    
    for keyword in security_keywords:
        if keyword.lower() in text.lower():
            entities["security_keywords"].append(keyword)

    return entities

def extract_with_regex(text):
    patterns = [
        r"\\b(?:CompetitorX|CompetitorY|CompetitorZ)\\b",  
        r"\\b(?:analytics|AI engine|data pipeline)\\b",   
    ]
    matches = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text, re.IGNORECASE))
    return matches

def extract_entities(text):
    entities_from_kb = extract_from_knowledge_base(text)
    entities_from_regex = extract_with_regex(text)

    final_entities = {
        "competitors": list(set(entities_from_kb["competitors"])),
        "features": list(set(entities_from_kb["features"])),
        "pricing_keywords": list(set(entities_from_kb["pricing_keywords"])),
        "security_keywords": list(set(entities_from_kb["security_keywords"])),
        "other_entities": [entity["entity"] for entity in entities_from_regex]
    }

    return final_entities

input_text = input()
with open('./task2/entities.txt', 'w') as file:
    file.write(f'{extract_entities(input_text)}')



