# from langchain.document_loaders import CSVLoader
# from langchain_ollama import OllamaEmbeddings
# from langchain_postgres import PGVector

# def create_embeddings():
#     ''' Function to create vector embeddings '''
#     ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
#     return ollama_embeddings

# connection_string = "postgresql://postgres:123456@localhost:5433/intellab-db"
# loader = CSVLoader(file_path="/Users/mac/HCMUS/datn/agent-service-toolkit/src/researchs/courses.csv", source_column="course_id")
# loader = CSVLoader(file_path="/Users/mac/HCMUS/datn/agent-service-toolkit/src/researchs/courses.csv", source_column="course_id")

# data = loader.load()
# embeddings = create_embeddings()
# docsearch = PGVector.from_documents(documents=data, embedding=embeddings, connection_string=connection_string, collection_name="courses_csv")
# vectorstore = PGVector(embedding_function=embeddings, collection_name="courses_csv", connection_string=connection_string)
from sqlalchemy import create_engine
import pandas as pd

# PostgreSQL connection string
connection_string = "postgresql://postgres:123456@localhost:5433/intellab-db"
engine = create_engine(connection_string)

# SQL query to fetch data
query = "select p.problem_id, p.problem_name, p.description, p.score, p.problem_level from problems p;"

try:
    # Use pandas with SQLAlchemy to execute query
    df = pd.read_sql(query, engine)

    # Save DataFrame to CSV
    df.to_csv("./src/researchs/problems.csv", index=False)

    print(f"Data successfully saved to students_courses.csv {df}")
except Exception as e:
    print(f"Error: {e}")
