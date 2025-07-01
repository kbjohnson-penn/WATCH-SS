from ..common_detectors.llm_detector import LLMDetector
# from .llm_config import dev_prompt, user_prompt

class FillerLLMDetector(LLMDetector):
    def __init__(self, model, api_token, host_url, user_prompt, rpm, tpm):
        super().__init__(model, api_token, host_url, rpm, tpm, None, user_prompt, temperature=0.0, top_p=1.0, max_output_toks=None, maintain_history=False)
