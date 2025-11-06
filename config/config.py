import os
from dotenv import load_dotenv

load_dotenv()

mongoLiveURI='mongodb+srv://narratixdev:sltgyS9EjJPqfXQw@narratixcluster.00fijbr.mongodb.net/narratixdb?retryWrites=true&w=majority&appName=NarratixCluster'
mongoLocalURI='mongodb://localhost:27017/smcps'
class Config:
    ENV = os.getenv('ENV', 'development')  # fallback to 'development' if ENV is not set
    MONGO_URI = mongoLiveURI if ENV == 'production' else mongoLocalURI