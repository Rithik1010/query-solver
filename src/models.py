from pydantic import BaseModel, Field
from typing import Optional

class QueryFilter(BaseModel):
    course_id: Optional[str] = Field(None)
    module_id: Optional[str] = Field(None)
    topic_id: Optional[str] = Field(None)


class QueryRequest(BaseModel):
    query: str
    sk: str
    filters: Optional[QueryFilter] = Field(None)

class TitleRequest(BaseModel):
    query: str