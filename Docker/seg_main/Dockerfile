FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt ${LAMBDA_TASK_ROOT}

RUN dnf update -y && \
    dnf install -y \
    mesa-libGL \
    mesa-libGLU \
    libXext \
    libXrender \
    libXtst \
    libXi \
    libXft \
    gcc \
    gcc-c++ \
    glib2 \
    libSM \
    libXext \
    libXrender \
    fontconfig \
    && dnf clean all

RUN pip install -r requirements.txt \
    && pip install matplotlib && pip install psutil
COPY . ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]