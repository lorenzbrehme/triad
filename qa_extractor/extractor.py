import time



class QA_Extractor:
    def __init__(self, model, vector_store):
        self.model = model
        self.vector_store = vector_store
        self.relevance_prompt = """
        Question : {question}
        Answer : {answer}
        Contextlist : {contexts}
        Evaluate this context : {sentence}
        Task : Does this sentence help support the answer , considering
        the overall contextlist ?
        Respond with :
        " Relevant " or " Not Relevant " - nothing else .

        """
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



    def extract_contexts(self, qa):
        question = qa['multi_hop_question']
        chunks = qa['chunks']
        answer = qa['multi_hop_answer']
        similar_docs = []
        for chunk in chunks:
            similar_docs += self.vector_store.similarity_search(chunk, k=10)
        unique_texts = set()
        
        

        # remove dublicates and the source documents
        filtered_docs = []
        
        for doc in similar_docs:
            if doc.page_content not in unique_texts and doc.page_content not in chunks:
                unique_texts.add(doc.page_content)
                filtered_docs.append(doc)
        extractor_list = []
        for context in filtered_docs:
            context = context.page_content
            prompt_for_llm = self.relevance_prompt.format(question=question, answer = answer, contexts = chunks, sentence = context)
            validation_chunk = self.send_request_to_LLM_conversation(prompt=prompt_for_llm)
            extractor_list.append({
                'context': context,
                'relevant': validation_chunk
            })
        qa['extracted_contexts'] = extractor_list
        return qa
    
