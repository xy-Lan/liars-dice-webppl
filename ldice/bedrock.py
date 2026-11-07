"""Adapted from user's scallop-semantic-mini-pilot-v2/model.py."""
import os
import time
MODEL_ALIASES = {
    "claude-haiku": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-sonnet": "us.anthropic.claude-sonnet-4-6",
}
class BedrockClient:
    def __init__(self, model_name, region=None, timeout=120, client=None):
        self.model_id = MODEL_ALIASES.get(model_name, model_name)
        self.region = region or os.getenv('AWS_REGION') or os.getenv('AWS_DEFAULT_REGION') or 'us-east-2'
        if client is None:
            import boto3
            from botocore.config import Config
            client = boto3.client('bedrock-runtime', region_name=self.region,
                config=Config(connect_timeout=10,read_timeout=timeout,
                              retries={'total_max_attempts':1}))
        self.client=client

    def generate(self,prompt,max_tokens=6400):
        start=time.perf_counter()
        response=self.client.converse(
            modelId=self.model_id,
            messages=[{'role':'user','content':[{'text':prompt}]}],
            inferenceConfig={'temperature':0.5,'maxTokens':max_tokens})
        text=''.join(b['text'] for b in response['output']['message']['content'] if 'text' in b).strip()
        if not text:
            raise RuntimeError('empty Bedrock response')
        if response.get('stopReason') == 'max_tokens':
            raise RuntimeError('Bedrock output truncated at max_tokens')
        return {'text':text,'latency_s':time.perf_counter()-start,
                'usage':response.get('usage',{}),'stop_reason':response.get('stopReason')}
