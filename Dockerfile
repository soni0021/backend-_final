FROM public.ecr.aws/lambda/python:3.9

COPY requirements.txt .

# Force reinstall pydantic with its binary dependencies
RUN pip install --upgrade pip && \
    pip install --force-reinstall pydantic==2.0.3 pydantic-core==2.3.0 && \
    pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

COPY main.py ${LAMBDA_TASK_ROOT}/
COPY Corrected_Marks_vs_Rank.xlsx ${LAMBDA_TASK_ROOT}/
COPY cleaned_data/ ${LAMBDA_TASK_ROOT}/cleaned_data/

CMD ["main.handler"] 