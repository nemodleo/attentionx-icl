from typing import Dict, Optional, Hashable


class PromptTemplate:
    """In-context Learning Prompt Template Class
        This class represents a template that guides the generation of prompts in the retrieval or inference process.

    Attributes:
        template (:obj:`Dict` or :obj:`str`): A custom template dictionary or string. If a dictionary, the keys of the dictionary represent the values of the output_column, and the values represent the corresponding generated statement. If a string, it represents a string template.
        column_token_map (:obj:`Dict`): A dictionary mapping column names to specific tokens. The tokens will be replaced by data in the corresponding column (one piece each time) during the retrieval or inference process.
        ice_token(:obj:`str`, optional): A string that represents the specific token mapping from in-context examples. None if you want to use this template only to generate in-context examples, otherwise it can be used to generate the final prompt that is fed into the PLM. The ice_token will be invisible when generating in-context examples.
    """

    def __init__(self,
                 template: Dict,
                 column_token_map: Dict,
                 label_dict: Optional[Dict] = None,
                 ice_token: Optional[str] = None,
                 sep_token: Optional[str] = None,
                 binning: Optional[Dict] = None
                 ) -> None:
        self.template = template
        self.column_token_map = column_token_map
        self.ice_token = ice_token
        self.sep_token = sep_token
        self.label_dict = label_dict
        self.binning = binning

    def generate_ice_item(self, entry: Dict, label: Hashable, use_ordering=False) -> str:
        """Generate in-context example based on the provided :obj:`entry` data.

        Args:
            entry (:obj:`Dict`): A piece of data to be used for generating the in-context example.
            label (:obj:`Hashable`): The value of the output field.

        Returns:
            :obj:`str`: The generated in-context example.
        """

        # Select the corresponding template
        tp = self.template[label] if isinstance(self.template, Dict) else self.template
        # Remove sep token
        if self.sep_token is not None:
            tp.replace(self.sep_token, '')
        # Remove ice_token
        if self.ice_token is not None:
            tp = tp.replace(self.ice_token, '')

        if use_ordering:
            label_dict = {v : entry[k] for k , v in self.label_dict.items()}
            sorted_dict = {k: v for k, v in sorted(label_dict.items(), key=lambda item: item[1], reverse=True)}
            labels = {'Label'+str(i+1): list(sorted_dict.keys())[i] for i in range(len(sorted_dict.keys()))}
            probs = list(sorted_dict.values())

        # Replace context token
        for key, token in self.column_token_map.items():
            if 'Label' in token and use_ordering:
                tp = tp.replace(token, labels[key])
            elif key != 'text' and use_ordering:
                if self.binning is None:
                    text = str(round(float(probs[int(key)])*100, 2))
                else:
                    text = self.binning[key]
                tp = tp.replace(token, text)
            else:
                text = str(entry[key])
                if key != 'text':
                    text = str(round(float(text)*100, 2))
                tp = tp.replace(token, text)

        return tp

    def generate_label_prompt_item(self, entry: Dict, ice: str, label: Hashable, remain_sep: Optional[bool] = False) -> str:
        """Generate prompt based on :obj:`entry` data, :obj:`ice` in-context example, and the corresponding :obj:`label`.

        Args:

            entry (:obj:`Dict`): A piece of data containing the input field content.
            ice (:obj:`str`): The generated in-context example.
            label (:obj:`Hashable`): The value of the output field.
            remain_sep (:obj:`bool`): If remain sep_token

        Raises:
            ValueError: If the :obj:`ice_token` attribute of the :obj:`PromptTemplate` instance is :obj:`None`.

        Returns:
            :obj:`str`: The generated prompt.
        """
        if self.ice_token is None:
            raise ValueError("PromptTemplate.ice_token should be not None when generates prompt")
        # Select the corresponding template
        tp = self.template[label] if isinstance(self.template, Dict) else self.template
        # Remove sep token
        if not remain_sep and self.sep_token is not None:
            tp.replace(self.sep_token, '')
        # Insert in-context examples
        tp = tp.replace(self.ice_token, ice)
        # Replace context token
        for key, token in self.column_token_map.items():
            tp = tp.replace(token, str(entry[key]))
        return tp

    def __repr__(self):
        return f"PromptTemplate({{\n\ttemplate: {self.template},\n\tcolumn_token_map: {self.column_token_map},\n\tice_token: {self.ice_token}\n}})"