import time
import mlflow
import tiktoken
import openai
from openai import OpenAI

class LLMDetector:
    def __init__(self, model, api_token, host_url, rpm, tpm, dev_prompt, user_prompt, temperature, top_p, max_output_toks=None, maintain_history=True):
        '''
        Initialize detector.

        args:
            model (str) - LLM to use
            api_token (str) - OpenAI API token
            host_url (str) - host url for Databricks serving endpoint
            rpm (int) - requests per minute
            tpm (int) - tokens per minute
            temperature (float) - temperature to use for LLM
            top_p (float) - top_p to use for LLM
            dev_prompt (str) - system prompt to use for LLM
            user_prompt (str) - user prompt to use for LLM
            maintain_history (bool) - whether to maintain message history between calls, defaults to True
        '''
        self.model = model
        self.rpm = rpm
        self.tpm = tpm
        self.dev_prompt = dev_prompt
        self.user_prompt = user_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_toks = max_output_toks
        self.maintain_history = maintain_history

        # Initialize OpenAI API client
        self.client = OpenAI(api_key=api_token, base_url=host_url)

        # Initialize message history
        self.reset_messages()

        # Rate limiting metrics
        self.t_last_request = None
        self.requests_made = 0
        self.tokens_used = 0

    def get_messages(self):
        '''
        Get the message history.

        return:
            (list) message history
        '''
        return self.messages
    
    def reset_messages(self):
        '''
        Clear message history, preserving the developer prompt if it was provided.
        '''
        self.messages = []
        if self.dev_prompt is not None:
            self.messages.append({"role": "developer", "content": self.dev_prompt})

    def _count_messages_tokens():
        '''
        Count tokens in message history.
        Source: 

        return:
            (int) number of tokens
        '''
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            print("Warning: model not found. Using o200k_base encoding.")
            encoding = tiktoken.get_encoding("o200k_base")
        if model in {
            "gpt-3.5-turbo-0125",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o-2024-08-06"
            }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif "gpt-3.5-turbo" in model:
            print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0125.")
            return self._count_messages_tokens(self.messages, model="gpt-3.5-turbo-0125")
        elif "gpt-4o-mini" in model:
            print("Warning: gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-mini-2024-07-18.")
            return self._count_messages_tokens(self.messages, model="gpt-4o-mini-2024-07-18")
        elif "gpt-4o" in model:
            print("Warning: gpt-4o and gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-2024-08-06.")
            return self._count_messages_tokens(self.messages, model="gpt-4o-2024-08-06")
        elif "gpt-4" in model:
            print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
            return self._count_messages_tokens(self.messages, model="gpt-4-0613")
        else:
            raise NotImplementedError(
                f"""self._count_messages_tokens() is not implemented for model {model}."""
            )
        num_tokens = 0
        for message in self.messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def _enforce_rate_limits(self):
        '''
        Enforce rate limits with sleep.
        '''
        # Reset token count if rate limit period has passed
        if self.t_last_request and time.time() - self.t_last_request > 60:
            self.requests_made = 0
            self.tokens_used = 0
        
        # Check if rate limit will be exceeded
        input_tokens = self._count_messages_tokens()
        if self.requests_made + 1 >= self.rpm or self.tokens_used + input_tokens + self.max_output_toks >= self.tpm:
            retry_after = self.t_last_request + 60 - time.time()
            print("Rate limit exceeded. Waiting for {retry_after} seconds...")
            time.sleep(retry_after)
            self.requests_made = 0
            self.tokens_used = 0

    def _call_llm(self):
        '''
        Query LLM using OpenAI API

        return:
            (str or None) LLM response
        '''
        self._enforce_rate_limits()
        try:
            output = client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_completion_tokens=self.max_output_toks
            )

            response = output.choices[0].message.content
            self.messages.append({"role": "assistant", "content": response})

            # Update rate limiting metrics
            self.t_last_request = time.time()
            self.requests_made += 1
            self.tokens_used += output.usage.total_tokens

            return response
        except openai.error.OpenAIError as e:
            print(f"Error calling LLM: {e}")
            return None

    @mlflow.trace
    def detect(self, text):
        '''
        Run detection.

        args:
            text (str) - input for detector
        
        return:
            (str) LLM response
        '''
        content = text if self.user_prompt is None else self.user_prompt.format(text)
        self.messages.append({"role": "user", "content": content})

        output = self._call_llm()

        if not self.maintain_history:
            self.reset_messages()

        return output