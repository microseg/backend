# Use AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.10

# Install system dependencies
RUN yum update -y && \
    yum install -y \
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
    && yum clean all

# Copy all files and directories
COPY . ${LAMBDA_TASK_ROOT}/

# Install Python dependencies
# First install numpy (base dependency)
RUN pip install --no-cache-dir numpy==1.24.4

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Install current package in editable mode (equivalent to pip install -e .)
RUN pip install --no-cache-dir -e ${LAMBDA_TASK_ROOT}/

# Set environment variables
ENV PYTHONPATH=${LAMBDA_TASK_ROOT}

# Set Lambda handler function
CMD ["demo/lambda_function.lambda_handler"] 