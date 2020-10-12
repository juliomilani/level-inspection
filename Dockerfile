FROM tensorflow/tensorflow

WORKDIR /code


RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev

COPY requirements.txt .

RUN pip install -r requirements.txt

EXPOSE 5000

COPY . .

CMD ["streamlit", "run","st_app.py"]
