from dotenv import load_dotenv
import os 
load_dotenv()


class Config:
    def __init__(self) -> None:
        self.URL = os.getenv('URL_API')        

settings = Config()