import os
import sys
import hydra
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service.service import create_attributes, model_schema, create_service
from config import CONFIG_DIR, CONFIG_NAME
from bentoml.io import JSON
from bentoml.io import NumpyNdarray

@hydra.main(version_base="1.1", config_path=CONFIG_DIR, config_name=CONFIG_NAME)
def main(config):
    create_attributes(model_schema, config)
    service, model = create_service(config)
    
    @service.api(input=JSON(pydantic_model=model_schema), output=NumpyNdarray())
    def predict(data):
        df = pd.DataFrame(data.dict(), index = [0])
        result = model.run(df)[0]
        return result
    
if __name__ == "__main__":
    main()