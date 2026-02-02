from typing import Optional, List, Dict


class EntityMatchPrompt:
    """
    Reusable prompt template for entity matching.
    - Optional system message
    - One user message that is UPDATED in place
    - Three setter methods to update fields independently
    """
    def __init__(self, system_message: Optional[str] = None):
        self.messages: List[Dict[str, str]] = []
        if system_message:
            self.messages.append({"role": "system", "content": system_message})

        self._template = "{entities}{rag_context}{few_shots}"
        self._state = {"entities": "", "rag_context": "", "few_shots": ""}
        self._user_index: Optional[int] = None  # index of the user message

    # --- internal helpers ---
    def _render(self) -> str:
        return self._template.format(**self._state)

    def _ensure_user_message(self) -> None:
        content = self._render()
        if self._user_index is None:
            self.messages.append({"role": "user", "content": content})
            self._user_index = len(self.messages) - 1
        else:
            self.messages[self._user_index]["content"] = content

    # --- public API: three functions, one per field ---
    def set_entities(self, entities: str) -> List[Dict[str, str]]:
        """Set or replace the entities part and update the user message."""
        self._state["entities"] = entities or ""
        self._ensure_user_message()
        return self.messages

    def set_rag_context(self, rag_context: Optional[str]) -> List[Dict[str, str]]:
        """Set or replace the RAG context part and update the user message."""
        self._state["rag_context"] = rag_context or ""
        self._ensure_user_message()
        return self.messages

    def set_few_shots(self, few_shots: Optional[str]) -> List[Dict[str, str]]:
        """Set or replace the few-shot examples part and update the user message."""
        self._state["few_shots"] = few_shots or ""
        self._ensure_user_message()
        return self.messages

    def get_messages(self) -> List[Dict[str, str]]:
        """Return all user messages."""
        return self.messages

# a = EntityMatchPrompt("hello, world")
# a.set_entities("I am here ")
# a.set_few_shots("I am python")
# a.set_few_shots("I am python+++")
# a.set_rag_context("I am java ")
# a.set_rag_context("I am java+++ ")
# print(a.messages)