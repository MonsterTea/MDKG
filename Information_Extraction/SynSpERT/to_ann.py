import json
import os

def To_ann_annotated(merged_list, ann_fold_path):
    for i in range(len(merged_list)):
        filename = merged_list[i]['orig_id'] + '.ann'
        file_txt_name = merged_list[i]['orig_id'] + '.txt'
        
        # Write sentences to a .txt file
        with open(os.path.join(ann_fold_path, file_txt_name), 'w', encoding='utf-8') as fr:
            fr.write(merged_list[i]['sents'])

        # Write annotations to a .ann file
        with open(os.path.join(ann_fold_path, filename), 'w', encoding='utf-8') as f:
            numbers = 0
            if len(merged_list[i]['entities']) != 0:
                for j in range(len(merged_list[i]['entities'])):
                    numbers += 1
                    tag = 'T' + str(numbers) + '\t' + merged_list[i]['entities'][j]['type'] + ' ' + str(merged_list[i]['entities'][j]['entity_start']) + ' ' + str(merged_list[i]['entities'][j]['entity_end']) + '\t' + merged_list[i]['entities'][j]['span']
                    f.write(tag + '\n')
            R_numbers = 0
            if len(merged_list[i]['relations']) != 0:
                for j in range(len(merged_list[i]['relations'])):
                    R_numbers += 1
                    tag = 'R' + str(R_numbers) + '\t' + merged_list[i]['relations'][j]['type'] + ' ' + "Arg1:" + str(merged_list[i]['relations'][j]['head_id']) + ' ' + "Arg2:" + str(merged_list[i]['relations'][j]['tail_id'])
                    f.write(tag + '\n')

# Paths to the input JSON files
unann_list_path = r"your/unannotated/list/path/predictions_test_epoch_0.json"  # Set your unannotated list path here
ann_fold_path = r"your/annotated/fold/path/"  # Set your annotated fold path here

# Load the unannotated list
with open(unann_list_path, "r") as f:
    merged_list = json.load(f)

# Load the ID data
with open(r"your/val/data/path/0609_200_10001sample.json", "r") as f:  # Set your validation data path here
    id_data = json.load(f)

# Merge ID data with the unannotated list
for i in range(len(merged_list)):
    merged_list[i]['orig_id'] = id_data[i]['orig_id']
    merged_list[i]['sents'] = id_data[i]['sents']

# Update entity and relation IDs
for i, entry in enumerate(merged_list):
    new_entities = []
    entity_number = 1
    for entity in entry['entities']:
        entity['entity_id'] = "T" + str(entity_number)
        entity_number += 1

    for relation in entry['relations']:
        relation['head_id'] = entry['entities'][relation['head']]['entity_id']
        relation['tail_id'] = entry['entities'][relation['tail']]['entity_id']

    # Filter out entities that intersect with others and are shorter
    for entity in entry['entities']:
        intersects_and_shorter = any(
            (other_entity['start'] < entity['start'] < other_entity['end'] or other_entity['start'] < entity['end'] < other_entity['end']) and
            (entity['end'] - entity['start'] < other_entity['end'] - other_entity['start'])
            for other_entity in entry['entities']
        )
        if not intersects_and_shorter:
            new_entities.append(entity)
    
    entities_new_id = [i['entity_id'] for i in new_entities]
    entry['entities'] = new_entities
    relation_new = [relations for relations in entry['relations'] if relations['head_id'] in entities_new_id and relations['tail_id'] in entities_new_id]

    entry['entities'] = sorted(entry['entities'], key=lambda e: e['start'])
    entry['relations'] = relation_new

# Update entity span positions
for i, entry in enumerate(merged_list):
    current_position = 0
    for entity in entry['entities']:
        start = entity['start']
        end = entity['end']
        entity['span'] = entry['tokens'][start:end]
        entity['span'] = ' '.join(entity['span'])
        start_index = entry['sents'].find(entity['span'], current_position)
        if start_index != -1:
            end_index = start_index + len(entity['span'])
            entity['entity_start'] = start_index
            entity['entity_end'] = end_index
            current_position = end_index
        else:
            print(f"Error: '{entity['span']}' not found in sentence from position {current_position}.")

# Save the annotated data
To_ann_annotated(merged_list, ann_fold_path)
