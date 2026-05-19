

import sys
from pathlib import Path
project_root = Path.cwd().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
import json
import time
from qa_generator.prompt_templates import QA_GENERATION_PROMPTS, BRIDGE_TOPIC_PROMPT



class QA_Generator:
    def __init__(self, model, vector_store):
        self.model = model
        self.vector_store = vector_store
        
    
    def send_request_to_LLM_conversation(self, prompt):
            success = False
            while not success:
                try:
                    llm_response = self.model.prompt(prompt)
                    if llm_response is not None:
                        success = True
                    else:
                        print(f"Received empty response from LLM. Retrying...")
                except Exception as e:
                    if '429' in str(e):
                        print(f"Rate limit exceeded. Waiting for 60 seconds before evaluating next batch")
                        time.sleep(60)
                        # try again
                    elif '503' in str(e):
                        print(f"Service Unavailable. Waiting for 60 seconds before evaluating next batch")
                        time.sleep(60)
                        # try again
                    else:
                        success = True
                        print(f"Error generating prompt data: {e}")
                        return None
            return llm_response

    def send_chat_to_LLM_conversation(self, prompt):
        success = False
        while not success:
            try:
                llm_response = self.model.chat_with_model(prompt)
                if llm_response is not None:
                    success = True
                else:
                    print(f"Received empty response from LLM. Retrying...")
            except Exception as e:
                if '429' in str(e):
                    print(f"Rate limit exceeded. Waiting for 60 seconds before evaluating next batch")
                    time.sleep(60)
                    # try again
                elif '503' in str(e):
                    print(f"Service Unavailable. Waiting for 60 seconds before evaluating next batch")
                    time.sleep(60)
                    # try again
                else:
                    success = True
                    print(f"Error generating prompt data: {e}")
                    return None
        return llm_response


    def bridge_topic_finder(self, chunks):
        bridign_topic_prompt = BRIDGE_TOPIC_PROMPT['bridge_topic_finder']
        bridge_topic_prompt_filled = bridign_topic_prompt.format(chunks=chunks)
        bridge_topic_response = self.send_request_to_LLM_conversation( prompt=bridge_topic_prompt_filled)
        bridge_topic_response = bridge_topic_response.replace("```json", "").replace("```", "").strip()
        bridge_topic_response = json.loads(bridge_topic_response)
        return bridge_topic_response


    def question_generator(self,bridge_topic,chunks, role ="Average Human", question_type = 'attribute_composition', binary=False):
        if binary:
            binary = "The answer to the question should be either 'Yes' or 'No'."
        else:
            binary = ""
        
        question_generation_prompt = QA_GENERATION_PROMPTS[question_type]
        prompt = question_generation_prompt.format(bridge_topic=bridge_topic, chunks=chunks, role=role, binary=binary)
        multi_hop_question_answer = self.send_chat_to_LLM_conversation(prompt=prompt)
        multi_hop_question_answer = multi_hop_question_answer.replace("```json", "").replace("```", "").strip()
        parsed_answer = json.loads(multi_hop_question_answer)
        return parsed_answer

    def chunk_finder(self,initial_chunk):
        retrieved_docs = self.vector_store.similarity_search(initial_chunk, k=2)
        doc = retrieved_docs[1]
        
        bridge_topic_response = self.bridge_topic_finder([initial_chunk, doc.page_content])
        if bridge_topic_response["shared_entity_exists"]:
            bridge_topic = bridge_topic_response["shared_entities"][0]
            return doc.page_content, bridge_topic
        else:
            chunks = [{
                'chunk': initial_chunk,
                'topics': bridge_topic_response["entities_chunk_a"],
                },{
                    'chunk': doc.page_content,
                    'topics': bridge_topic_response["entities_chunk_b"],
                }]
            for items in chunks:
                chunk = items['chunk']
                topics = items['topics']
                raw_docs = self.vector_store.similarity_search(
                    chunk, 
                    k=50, 
                )
                for bridge_topic in topics:
                    retrieved_docs = [
                        doc for doc in raw_docs 
                        if bridge_topic.lower() in doc.page_content.lower() and not doc.page_content == chunk
                    ]
                    if len(retrieved_docs) > 0:
                        return retrieved_docs[0].page_content, bridge_topic
                    
        return None, None


    def generate_question(self, chunk, qa_type, binary=False, role= "Ask short and precise questions as a government employee"):
        
        second_chunk, bridging_topic = self.chunk_finder(chunk)
        if second_chunk is not None:
            chunks = [chunk, second_chunk]
            qa_pair = self.question_generator(bridging_topic, chunks, role=role,  question_type=qa_type, binary=binary)
            # validation_response = multi_hop_validator(chunk, second_chunk, qa_pair['multi_hop_question'])
            return {
                "chunks": chunks,
                "bridging_topic": bridging_topic,
                "multi_hop_question": qa_pair['multi_hop_question'],
                "multi_hop_answer": qa_pair['multi_hop_answer'],
            }
        if second_chunk is None:
            print("No second chunk found")
            return None
    
    def feedback_loop(self, feedback, chunks, bridging_topic):
        feedback_prompt = '''
        The question was not valid with the following reason: {feedback}.
        Please modify the question to make it valid.
        Outputformat:
        # **Output Format (JSON):**  
        # {{  
        #     "multi_hop_question": "<generated question>",  
        #     "multi_hop_answer": "<short answer>"  
        # }}
                
        '''
        prompt = feedback_prompt.format(feedback=feedback)
        llm_response = self.send_chat_to_LLM_conversation(prompt)
        qa_pair = json.loads(llm_response.replace("```json", "").replace("```", "").strip())
        return {
                "chunks": chunks,
                "bridging_topic": bridging_topic,
                "multi_hop_question": qa_pair['multi_hop_question'],
                "multi_hop_answer": qa_pair['multi_hop_answer'],
            }
        

            