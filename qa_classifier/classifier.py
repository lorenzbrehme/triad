import time


class QA_Classifier:
    def __init__(self, model):
        self.model = model

        self.temporalQA_classiification_prompt = """
        Classify the following question as temporal or non-temporal.

        Question:
        {question}

        Answer:
        {answer}
        Output format:
        Respond with exactly one word:
        Temporal
        or
        Non-Temporal
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

    def classifier(self, question, answer):
        prompt_for_llm = self.temporalQA_classiification_prompt.format(question=question, answer = answer)
        validation = self.send_request_to_LLM_conversation(prompt=prompt_for_llm)
        temporal_boolean = False if "non-temporal" in validation.lower() else True
        return temporal_boolean

