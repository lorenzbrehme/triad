import json
import requests

with open('../static_qas/hotpot_preprocessed.json', 'r') as f:
    data_hotpot = json.load(f)
with open('../../qa_sets/final_hotpot.json', 'r') as f:
    generated_qa = json.load(f)
data_mhqa = generated_qa['data']

# with open('../static_qas/musique_preprocessed.json', 'r') as f:
#     data_hotpot = json.load(f)
# with open('../../qa_sets/final_musique.json', 'r') as f:
#     generated_qa = json.load(f)
# data_mhqa = generated_qa['data']


url_string = 'http://127.0.0.1:8000/rag'
results = []
def convert_jsnl_to_json(inputpath, outputpath):
    data = []
    with open(inputpath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    with open(outputpath, 'w') as f:
        json.dump(data, f, indent=4)
    return data

outputpath_jsonl_mhqa = '../rag_answer_data/mhqa_hotpot_gemini-2.5-flash-lite_5_chunks-embedding_01-rag.jsonl'
outputpath_json_mhqa = '../rag_answer_data/mhqa_hotpot_gemini-2.5-flash-lite_5_chunks-embedding_01-rag.json'
outputpath_jsonl_hotpot = '../rag_answer_data/hotpot_gemini-2.5-flash-lite_5_chunks-embedding_01-rag.jsonl'
outputpath_json_hotpot = '../rag_answer_data/hotpot_gemini-2.5-flash-lite_5_chunks-embedding_01-rag.json'

# outputpath_jsonl_mhqa = '../rag_answer_data/neu_mhqa_musique_gemini-2.5-flash-lite_5_chunks_embedding_02-rag.jsonl'
# outputpath_json_mhqa = '../rag_answer_data/neu_mhqa_musique_gemini-2.5-flash-lite_5_chunks_embedding_02-rag.json'
# outputpath_jsonl_hotpot = '../rag_answer_data/neu_musique_gemini-2.5-flash-lite_5_chunks_embedding_02-rag.jsonl'
# outputpath_json_hotpot = '../rag_answer_data/neu_musique_gemini-2.5-flash-lite_5_chunks_embedding_02-rag.json'

setup = [
    {
    'data': data_mhqa,
    'outputpath_jsonl': outputpath_jsonl_mhqa,
    'outputpath_json': outputpath_json_mhqa,
},
    {
    'data': data_hotpot,
    'outputpath_jsonl': outputpath_jsonl_hotpot,
    'outputpath_json': outputpath_json_hotpot,
    }
]
for item in setup:
    data = item['data']
    outputpath_jsonl = item['outputpath_jsonl']
    outputpath_json = item['outputpath_json']
    for qa in data:
        response = requests.post(url_string, json={"question": qa['multi_hop_question']})
        qa['rag_answer'] = response.json()['answer']
        qa['rag_context'] = response.json()['context']
        # results.append(qa)
        with open(outputpath_jsonl, 'a') as f:
            f.write(json.dumps(qa) + '\n')
    convert_jsnl_to_json(outputpath_jsonl, outputpath_json)