FROM maven:3-eclipse-temurin-8 AS datax-builder

RUN apt-get update && \
    apt-get install -y git && \
    git clone https://github.com/alibaba/DataX.git

COPY runtime/datax/ DataX/

RUN cd DataX && \
    sed -i "s/com.mysql.jdbc.Driver/com.mysql.cj.jdbc.Driver/g" \
    plugin-rdbms-util/src/main/java/com/alibaba/datax/plugin/rdbms/util/DataBaseType.java && \
    mvn -U clean package assembly:assembly -Dmaven.test.skip=true

FROM python:3.12-slim

# Single-stage image with build cache optimization using BuildKit cache mounts.
# Note: to use the cache mount syntax you must build with BuildKit enabled:
# DOCKER_BUILDKIT=1 docker build . -f scripts/images/datamate-python/Dockerfile -t datamate-backend-python

RUN apt-get update && \
    apt-get install -y --no-install-recommends vim openjdk-21-jre nfs-common glusterfs-client rsync dos2unix && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.2.1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken-cache

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk

ENV PATH="/root/.local/bin:$JAVA_HOME/bin:$PATH"

WORKDIR /app

# Install Poetry via pipx as Poetry officially recommends
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade --root-user-action=ignore pip \
    && pip install --root-user-action=ignore pipx \
    && pipx install "poetry==$POETRY_VERSION"

# Copy only dependency files first (leverages layer caching when dependencies don't change)
COPY runtime/datamate-python/pyproject.toml runtime/datamate-python/poetry.lock* /app/

# Install dependencies using Poetry with cache mount
# --no-root: don't install the project itself yet (only dependencies)
# --only main: only install main dependencies, not dev dependencies
RUN --mount=type=cache,target=$POETRY_CACHE_DIR \
    poetry install --no-root --only main

# Preload the tokenizer vocabulary so knowledge-base operations work offline.
RUN mkdir -p "$TIKTOKEN_CACHE_DIR" \
    && python -c "import tiktoken; [tiktoken.get_encoding(name) for name in ('cl100k_base', 'o200k_base')]"

# Download NLTK data
RUN python -c "import nltk; nltk.download(['punkt_tab','averaged_perceptron_tagger_eng'], download_dir='/usr/local/nltk_data')"
ENV NLTK_DATA=/usr/local/nltk_data

# Copy the rest of the application
COPY runtime/datamate-python /app
COPY runtime/ops/examples/test_operator/test_operator.tar /app/test_operator.tar

COPY --from=datax-builder /DataX/target/datax/datax /opt/datax
RUN cp /opt/datax/plugin/reader/mysqlreader/libs/mysql* /opt/datax/plugin/reader/starrocksreader/libs/

COPY runtime/datamate-python/deploy/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh \
    && dos2unix /docker-entrypoint.sh || true \
    && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# Expose the application port
EXPOSE 18000

ENTRYPOINT ["/docker-entrypoint.sh"]
