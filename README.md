# ml-docker-fastapi
Deploying vaccine sentiment detection model using FastApi and Docker 

# How to run
- clone the repo: git clone https://github.com/sheikhhanif/ml-docker-fastapi.git
- build the docker image: docker build -t vaccine-app .
- run: docker run -p 80:80 vaccine-app
- swagger UI: localhost/docs

# Prerequisite:
- Python
- Scikit learn
- FastAPI
- Docker

# Warning!
- Test only
- Not optimized for production used

# References
- https://youtu.be/h5wLuVDr0oc?si=Q5ZZP3vrUgOJvUPO
- https://fastapi.tiangolo.com
- https://docs.docker.com/?_gl=1*at4txk*_ga*MzY5MDUxMzUuMTY5Mjg4MTU1Mw..*_ga_XJWPQMJYHQ*MTY5Mjg4MTU1Mi4xLjEuMTY5Mjg4MTU1My41OS4wLjA.
- https://www.neoito.com/blog/ml-model-with-fastapi-and-docker/
- https://medium.com/analytics-vidhya/serve-a-machine-learning-model-using-sklearn-fastapi-and-docker-85aabf96729b
