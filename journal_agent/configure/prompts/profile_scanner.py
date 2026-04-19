from __future__ import annotations

import json

from langchain_core.prompts import PromptTemplate

from journal_agent.configure.prompts._helpers import _schema_block
from journal_agent.configure.prompts.base_prompt_template import PromptTemplateBuilder
from journal_agent.graph.state import JournalState
from journal_agent.model.session import UserProfile, Status, ContextSpecification, PromptKey

_TEMPLATE_TEXT = """
You are a personalization scorer for a journaling AI.

PURPOSE
You will review the last message to detect whether a user is asking to change their profile.
This is a two step process.

First, if the message does NOT contain a request to update any part of their profile, then we create a default UserProfile object and return it.
The `is_modified` field of the UserProfile should be set to False, and the contents of the other fields can be anything.

Second, if the message does contain a request to update any part of their profile, then we create a UserProfile object and return it.
The is_modified field MUST be set to True in this case.

CURRENT PROFILE
This is the current profile of the user

```
{user_profile}
```

INPUT
You will receive a list of messages.  The last message is the user's current request.  And the previous messages are for your contextual understanding.

OUTPUT
You will output a UserProfile object.  Here is an example of the schema

```
{schema_block}
```
"""

class UserProfileTemplate(PromptTemplateBuilder):
    def __init__(self):
        self.template = PromptTemplate.from_template(_TEMPLATE_TEXT)

    def build(self, state: JournalState) -> str:
        user_profile = state['user_profile'].model_dump_json(indent=2)
        return self.template.format(
            user_profile=user_profile,
            schema_block=_schema_block(UserProfile)
        )



_STATIC_REGISTRY: dict[str, str] = {

}

_TEMPLATE_REGISTRY: dict[str, PromptTemplateBuilder] = {
    PromptKey.PROFILE_SCANNER.value: UserProfileTemplate(),
}







    # builder(**state)
    #
    # print(TEMPLATE.format(current_profile=current.model_dump_json(indent=2)))
