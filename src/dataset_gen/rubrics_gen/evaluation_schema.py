from typing import Optional, Union
from pydantic import BaseModel, ConfigDict, Field
from enum import Enum

# Define scoring schema
class PointwiseScoreSchema(BaseModel):
    model_config = ConfigDict(extra='forbid')
    score_1: str
    score_2: str
    score_3: str
    score_4: str
    score_5: str
    
# Define scoring schema
class PairwiseScoreSchema(BaseModel):
    model_config = ConfigDict(extra='forbid')
    response_1: str
    response_2: str
    
# Define scoring schema
class BinaryScoreSchema(BaseModel):
    model_config = ConfigDict(extra='forbid')
    true: str
    false: str

# Define Rubric schema
class EvaluationRubric(BaseModel):
    model_config = ConfigDict(extra='forbid')
    description: str
    scoring: Union[PointwiseScoreSchema, PairwiseScoreSchema, BinaryScoreSchema]
    
# Define model pointwise response schema
class PointwiseResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')
    explanation: str = Field(description="Explanation of why the response received a particular score")
    score: int = Field(description="Score assigned to the response based on the rubric between 1 to 5", ge=1, le=5)
    
# Define Enum for Pairwise Response
class PairwiseChoice(str, Enum):
    response_1 = "Response 1"
    response_2 = "Response 2"

# Define model pairwise response schema
class PairwiseResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')
    explanation: str = Field(description="Explanation of why one response is preferred over the other")
    score: PairwiseChoice = Field(description="Final selection between 'Response 1' or 'Response 2'")

# Define model binary response schema
class BinaryResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')
    explanation: str = Field(description="Explanation of why the answer is true or false")
    score: bool = Field(description="Final boolean answer between true or false")
