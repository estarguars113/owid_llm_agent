from typing import Dict, Optional, Tuple
from pydantic import BaseModel


from owid import catalog
import pandas as pd
import json


class OWIDAPIWrapper(BaseModel):
    doc_content_chars_max: int = 4000

    def run(self, query: str) -> Dict:
        try:
            response = catalog.find_latest(query)
            metadata = response.metadata
            text_response = "".join(
                self._format_metadata_output(metadata.to_dict())
            )[: self.doc_content_chars_max]

            full_response = {
                'metadata': text_response
            }
            if not response.empty:
                response.to_csv(f'data.csv')
                full_response['extra_data'] = 'True'
            
            return json.dumps(full_response)
        except ValueError as e:
            return json.dumps({'error': f'We sorry to say that OWID does not contain any data related to {query}'})
        except Exception as e:
            return json.dumps({'error': f'{e} error found while querying {query}'})
            
        
    @staticmethod
    def _format_metadata_output(metadata: Dict) -> Optional[str]:
        return (
            f"Title: {metadata.get('title', '')}" +
            f"Description: {metadata.get('description', '')}"
        )
