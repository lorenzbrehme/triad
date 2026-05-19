import time
import json

class QA_Validator:
    def __init__(self, model):
        self.model = model

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

    def quality_checker(self, question, answer, contexts):
        validation_prompt = """
        You are evaluating a question for Retrieval-Augmented Generation (RAG).

        The question will be used as a USER QUERY in a RAG pipeline.

        IMPORTANT CONTEXT:
        - The final downstream system DOES NOT initially know the context chunks.
        - The system must retrieve relevant chunks ONLY from the question itself.
        - Therefore, the question must be understandable and meaningful WITHOUT access to the chunks.
        - The question should contain enough explicit information to retrieve the correct documents.
        - Questions that rely on hidden context, vague references, or implicit entities are LOW QUALITY for RAG.

        You are given:
        - A question
        - An answer
        - Multiple context passages (chunks)

        Your task is to evaluate the question across THREE dimensions:

        1. Question Quality
        2. Multi-hop Reasoning Requirement
        3. Unambiguity

        You must evaluate strictly using ONLY the provided context.

        --------------------------------------------------
        EVALUATION DIMENSIONS
        --------------------------------------------------

        1. QUESTION QUALITY

        Determine whether the question is high quality for RAG retrieval and answering.

        A high-quality RAG question MUST satisfy ALL of the following:

        A. Correctness
        - The question must be factually meaningful.
        - It must not contain contradictions, false assumptions, or nonsensical premises.

        B. Answerability from Context
        - The question must be fully answerable using ONLY the provided context.
        - All required entities, facts, and relationships must be explicitly present in the context.
        - Do NOT rely on external knowledge or unstated inference.

        C. Standalone Retrieval Clarity
        - The question must be understandable WITHOUT seeing the context.
        - The question must independently contain enough identifying information to retrieve the correct chunks.
        - It must NOT rely on hidden context for interpretation.

        Examples of BAD RAG questions:
        - Based on the provided context, which invention is attributed to Nikola Tesla?
        - According to the player profiles in the context, which footballer plays as a goalkeeper?
        - Why was the discovery of penicillin considered a major medical breakthrough in the provided material?
        - What event occurred immediately after the signing of the peace treaty described in the context?

        These are bad because the retriever would not know:
        - who "he" is
        - which company
        - what "it" refers to
        - which event is referenced

        Examples of GOOD RAG questions:
        - "What did Nikola Tesla invent after moving to the United States?"
        - "When did Apple launch the iPhone 14?"
        - "Why was the Apollo 11 moon landing historically important?"

        D. Explicit Entity Grounding
        - All critical entities must be explicitly named in the question itself.
        - Avoid unresolved pronouns or generic references:
        "he", "she", "they", "it", "this", "that", "the company", etc.
        - The question should be semantically searchable.

        Strict Rules
        - Do NOT assume missing information.
        - Do NOT use external knowledge.
        - Do NOT infer unstated relationships.
        - If the question depends on context to make sense, mark it invalid.
        - If the retriever could not reasonably retrieve the correct chunks using only the question, mark it invalid.
        - If the context is insufficient to fully answer the question, mark it invalid.
        - When uncertain, mark invalid.

        --------------------------------------------------

        2. MULTI-HOP REASONING

        Determine whether answering the question requires combining information from MULTIPLE chunks.

        Definition:
        A question is multi-hop if the answer requires combining information from TWO OR MORE different chunks.

        Procedure
        1. Identify all information required to answer the question.
        2. Identify which chunk(s) contain each required piece.
        3. Determine:
        - If ANY SINGLE chunk fully answers the question → NOT multi-hop
        - If multiple chunks must be combined → multi-hop

        Strict Rules
        - Do NOT use external knowledge.
        - Do NOT assume missing information.
        - If any single chunk alone is sufficient, mark as NOT multi-hop.
        - When uncertain, mark as NOT multi-hop.

        --------------------------------------------------

        3. UNAMBIGUITY

        Determine whether the question is unambiguous.

        Definition:
        A question is unambiguous if it has exactly ONE clear answer based ONLY on the provided context.

        Evaluation Criteria
        - The question must refer to clearly identifiable entities.
        - The question must not allow multiple interpretations.
        - The context must not support multiple valid answers.
        - All references must be explicitly clear within the question itself.

        Strict Rules
        - Do NOT use external knowledge.
        - Do NOT assume missing information.
        - Do NOT resolve ambiguity using context if the question itself is unclear.
        - If ANY ambiguity exists, mark ambiguous.
        - When uncertain, mark ambiguous.

        --------------------------------------------------
        OUTPUT FORMAT
        --------------------------------------------------

        Return ONLY valid JSON in the following format:

        {{
            "reason": "Detailed explanation of the final decision, referencing specific evaluation failures if rejected. And a suggestion for how to improve the question if it was rejected.",
            "accepted": true | false,
        }}

        --------------------------------------------------
        INPUT
        --------------------------------------------------

        Question:
        {question}

        Answer:
        {answer}

        Context:
        {contexts}

        """
            
        prompt_for_llm = validation_prompt.format(question=question, answer = answer, contexts = contexts)
        validation = self.send_request_to_LLM_conversation(prompt=prompt_for_llm)
        validation_response = validation
        # If the response is already a dict, keep it as-is
        if isinstance(validation_response, dict):
            validation_response_json = validation_response
        else:
            # handle bytes
            if isinstance(validation_response, (bytes, bytearray)):
                try:
                    validation_response = validation_response.decode('utf-8')
                except Exception:
                    validation_response = str(validation_response)
            # if None or empty, preserve
            if validation_response is None:
                validation_response_json = None
            else:
                s = str(validation_response).strip()
                # remove common markdown code fences like ```json or ```
                if s.startswith('```'):
                    s = s.replace('```json', '', 1).replace('```', '')
                # Try a direct JSON parse, fall back to extracting first {...} block
                try:
                    validation_response_json = json.loads(s)
                except Exception:
                    import re
                    m = re.search(r'{.*}', s, re.DOTALL)
                    if m:
                        try:
                            validation_response_json = json.loads(m.group(0))
                        except Exception:
                            validation_response_json = s
                    else:
                        validation_response_json = s
        return validation_response_json, validation



    def sufficent_context_checker(self, chunk, multi_hop_question):
        prompt_template = """
    You are given a question and a context passage.

    Your task is to determine whether the context ALONE contains ALL information required to answer the question completely and unambiguously.

    Procedure:
    1. Identify every piece of information required to answer the question (entities, relationships, attributes, dates, etc.).
    2. Verify that EACH required element is explicitly stated in the context.
    3. Verify that all relationships mentioned in the question (e.g., "sequel to", "author of", "capital of", "directed by") are explicitly stated in the context.
    4. Check that the answer can be derived using only the context without any external knowledge.

    Strict rules:
    - Do NOT assume missing relationships.
    - Do NOT rely on world knowledge.
    - Do NOT infer unstated connections between entities.
    - If the question references a relationship, the context must explicitly state that relationship.
    - If even one required fact, entity link, or relationship is missing, respond "no".
    - If the answer would require guessing or prior knowledge, respond "no".
    - When uncertain, respond "no".

    Output format:
    Respond with exactly one word:
    yes
    or
    no

    Question: {question}
    Context: {context}
    """
        validation_response = self.send_request_to_LLM_conversation(prompt=prompt_template.format(question=multi_hop_question, context=chunk))
        lower_response = validation_response.lower()
        if "yes" in lower_response:
            return True, validation_response
        else:
            return False, validation_response


    def validate_qa_set(self, qa):
        question = qa['multi_hop_question']
        chunks = qa['chunks']
        answer = qa['multi_hop_answer']
                
        quality_validation, quality_response = self.quality_checker(question, answer, chunks)
        sufficent_contexts_flags = []
        sufficent_contexts_flags_original_responses = []
        for chunk in chunks:
            sufficient_context_flag, sufficent_repsonse = self.sufficent_context_checker(chunk, question)
            
            sufficent_contexts_flags.append(sufficient_context_flag)
            sufficent_contexts_flags_original_responses.append(sufficent_repsonse)
        # add qa + valaiudtion to data
        qa['quality_validation'] = quality_validation
        qa['sufficient_context_flags'] = sufficent_contexts_flags
        qa['quality_validation_original_response'] = quality_response
        qa['sufficient_context_original_responses'] = sufficent_contexts_flags_original_responses
        return qa
