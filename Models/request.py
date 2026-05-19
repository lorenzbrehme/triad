import time

class RequestModel:
    def send_request_to_LLM_conversation(self, model, prompt):
        success = False
        while not success:
            try:
                llm_response = model.chat_with_model(prompt)
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