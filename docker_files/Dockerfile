#can use python3.10 and develop too 
FROM langchain/langchain 

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --no-deps -r requirements.txt
RUN pip install ipython 

ENTRYPOINT ["ipython"]
