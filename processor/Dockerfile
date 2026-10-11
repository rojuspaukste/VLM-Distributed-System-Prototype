# Processor image: Python 3.14 + Java 17 + PySpark 4.1.3 (requirements.txt). Built by docker-compose.yml.
FROM python:3.14.4-slim-bookworm

# Spark runs on Java; Spark's launch scripts also call `ps` (procps).
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The job loads the Kafka connector through spark.jars.packages. Resolving it once at build time puts the
# jars into Spark's Ivy cache (/root/.ivy2.5.2), so the container starts without downloading anything.
RUN python -c "import pyspark; from pyspark.sql import SparkSession; \
SparkSession.builder.master('local[1]').config('spark.jars.packages', \
f'org.apache.spark:spark-sql-kafka-0-10_2.13:{pyspark.__version__}').getOrCreate().stop()"

COPY keyframe_job.py .
CMD ["python", "keyframe_job.py"]
