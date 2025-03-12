prompt = f"""Please carefully verify the following triple information. Original sentence: "{triple_dict['sentence']}"

1. Entity Check:
Head Entity: {triple_dict['head_entity']}
Tail Entity: {triple_dict['tail_entity']}
Please check:
- Entity names should be complete and standardized (use only ONE standard name, no alternative names or abbreviations)
- Entity spans should be accurate
- If any errors, provide correction suggestions

2. Entity Type Check:
Head Entity Type: {triple_dict['head_entity_type']}
Tail Entity Type: {triple_dict['tail_entity_type']}
Please confirm if the types are correct, provide corrections if needed

3. Relationship Assessment:
Current Relationship: {triple_dict['relationship']}
- Please verify if this relationship is valid based on the original sentence
- Provide a confidence score (1-10, where 1 means completely invalid and 10 means absolutely valid)
- Explain the reasoning for your score

Here's an example:
Original sentence: "Aspirin is commonly used to treat headache."
Input triple:
Head Entity: "ASA (acetylsalicylic acid)"
Tail Entity: "headaches"
Head Entity Type: "chemical"
Tail Entity Type: "symptom"
Relationship: "used_for"

Expected output format:
{{
    "corrected_head_entity": "Aspirin",
    "corrected_head_entity_type": "drug",
    "corrected_tail_entity": "headache",
    "corrected_tail_entity_type": "disease",
    "corrected_relationship": "treatment_for",
    "score": 10,
    "reason": "The sentence clearly states the treatment relationship. Corrected ASA to its standard name Aspirin, standardized relationship type to treatment_for, and corrected entity types to match standard ontology."
}}

Now, please evaluate this triple:
Head Entity: {triple_dict['head_entity']}
Tail Entity: {triple_dict['tail_entity']}
Head Entity Type: {triple_dict['head_entity_type']}
Tail Entity Type: {triple_dict['tail_entity_type']}
Relationship: {triple_dict['relationship']}
"""
