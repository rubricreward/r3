import outlines

from .evaluation_schema import PointwiseResponse, PairwiseResponse, BinaryResponse, \
    PointwiseScoreSchema, PairwiseScoreSchema, BinaryScoreSchema

@outlines.prompt
def evaluate_pointwise(task: str, input_data: str, response: str, rubric: PointwiseScoreSchema, format=PointwiseResponse):
    """
    Evaluate the response based on the given task, input, response, and evaluation rubric.  
    Provide a fair and detailed assessment following the rubric.
    
    ### TASK
    {{ task }}
    
    ### INPUT
    {{ input_data }}
    
    ### RESPONSE
    {{ response }}
    
    ### EVALUATION RUBRIC
    1: {{ rubric.score_1 }}
    2: {{ rubric.score_2 }}
    3: {{ rubric.score_3 }}
    4: {{ rubric.score_4 }}
    5: {{ rubric.score_5 }}

    ### OUTPUT FORMAT
    Return a JSON response in the following format:
    
    {{ format | schema }}
    
    ### EVALUATION
    """
    
@outlines.prompt
def evaluate_pairwise(task: str, input_data: str, response1: str, response2: str, rubric: PairwiseScoreSchema, format=PairwiseResponse):
    """
    Evaluate the response based on the given task, input, two responses, and evaluation rubric.  
    Provide a fair and detailed assessment following the rubric.
    
    ### TASK
    {{ task }}

    ### INPUT
    {{ input_data }}
    
    ### RESPONSE 1
    {{ response1 }}
    
    ### RESPONSE 2
    {{ response2 }}
    
    ### EVALUATION RUBRIC
    Response 1: {{ rubric.response_1 }}
    Response 2: {{ rubric.response_2 }}

    ### OUTPUT FORMAT
    Return a JSON response in the following format:
    
    {{ format | schema }}
    
    ### EVALUATION
    """
    
@outlines.prompt
def evaluate_binary(task: str, input_data: str, response: str, rubric: BinaryScoreSchema, format=BinaryResponse):
    """
    Evaluate the response based on the given task, input, response, and evaluation rubric.  
    Provide a fair and detailed assessment following the rubric.
    
    ### TASK
    {{ task }}
    
    ### INPUT
    {{ input_data }}
    
    ### RESPONSE
    {{ response }}
    
    ### EVALUATION RUBRIC
    true: {{ rubric.true }}
    false: {{ rubric.false }}

    ### OUTPUT FORMAT
    Return a JSON response in the following format:
 
    {{ format | schema }}
    
    ### EVALUATION
    """
