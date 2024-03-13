from typing import Dict, Optional, Hashable

from .prompt_template import PromptTemplate


class ProbPromptTemplate(PromptTemplate):
    """In-context Learning Prompt Template Class
        This class represents a template that guides the generation of prompts in the retrieval or inference process.

    Attributes:
        template (:obj:`Dict` or :obj:`str`): A custom template dictionary or string. If a dictionary, the keys of the dictionary represent the values of the output_column, and the values represent the corresponding generated statement. If a string, it represents a string template.
        column_token_map (:obj:`Dict`): A dictionary mapping column names to specific tokens. The tokens will be replaced by data in the corresponding column (one piece each time) during the retrieval or inference process.
        ice_token(:obj:`str`, optional): A string that represents the specific token mapping from in-context examples. None if you want to use this template only to generate in-context examples, otherwise it can be used to generate the final prompt that is fed into the PLM. The ice_token will be invisible when generating in-context examples.
    """

    def __init__(self,
                 prefix_template: str,
                 prob_tokens: Dict,
                 concat_token: str,
                 column_token_map: Dict,
                 ice_token: Optional[str] = None,
                 sep_token: Optional[str] = None,
                 ) -> None:
        self.prefix_template = prefix_template
        self.prob_tokens = prob_tokens
        self.concat_token = concat_token
        self.column_token_map = column_token_map
        self.ice_token = ice_token
        self.sep_token = sep_token

    def _get_template(self, label) -> str:
        return f"{self.prefix_template}{self.concat_token}{self.prob_tokens[label]}"

    def __repr__(self):
        return f"ProbPromptTemplate({{\n\tprefix_template: {self.prefix_template},\n\prob_tokens: {self.prob_tokens},\n\tcolumn_token_map: {self.column_token_map},\n\tice_token: {self.ice_token}\n}})"
